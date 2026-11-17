//! The Coulomb integrals of the pair densities of `n` supercell functions, streamed: the functions
//! are held (`n` grid values each), not the `n^2` transforms of their products that
//! [`super::pairs::Pairs`] stores, and each Cholesky column, contracted row or shape motion forms
//! the transforms it needs and takes one matrix product over the grid.
//!
//! The integral `(pq|rs)` of [`super::pairs::coulomb_rows`] is, with the kernel `K`, the sphere
//! `outside` of the grid (`K` beyond it) and the Nyquist planes left out,
//!
//! ```text
//! (pq|rs) = 1/V sum_G [ conj(L(G)) K(G) R(G) - conj(Lc(G)) outside(G) Rc(G) ] + tail,
//! ```
//!
//! `L = rho_qp` (the swapped pair), `R = rho_rs`, `Lc`, `Rc` their compensation parts, and
//! `rho_ab(G) = s(G) F[phi_a^* phi_b](G) + sum_c g_c(G) S_c[u(G)]` with `s = dv e^{-iG.r0}`, `g_c`
//! the compensation shapes and `S_c[u]` the lattice sums of the pair's charges ([`Pairs::sums`]).
//! With `U(G) = conj(L) K - conj(Lc) outside` and `W(G) = conj(L) K s` the column of the left pair is
//!
//! ```text
//! (pq|rs) = 1/V [ sum_x phi_r^*(x) phi_s(x) What(x) + sum_{c,u} Z_{cu} S_{cu}^{rs} ] + tail,
//! What = F[W],  Z_{cu} = sum_{G in u} U(G) g_c(G):
//! ```
//!
//! one FFT for the left density, one for `What`, a matrix product over the grid for every right
//! pair at once (`n^2` grid points, the cost of a column of the stored pairs) and `channels x
//! cells` charges per pair. A right density contracted with weights `w` (the forces' Coulomb
//! forms) is the same with the roles of the two sides exchanged ([`Streamed::contracted`]). The
//! functions are the supercell's own (every mode at the home cell). Atomic units.
use super::{
    coulomb::{Cell, Kernel},
    functions::Functions,
    pairs::{onsite_contracted, Mode, Pairs},
};
use crate::{
    math::{
        fft::Fft3,
        gemm::{multiply, MatrixMut, MatrixRef},
    },
    periodic::{invalid, response::setup::Setup, Crystal},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex, MutexGuard};

/// Grid points one block of the products over the grid takes.
const BLOCK: usize = 512;

/// What the supercell's grid and its atoms hold for every pair of functions, whatever the
/// functions: the kernel, the sphere, the phases, the compensation shapes and the self-energy
/// tails. Built once for a supercell and kernel, shared by every [`Streamed`] set of functions.
pub(in crate::periodic) struct Grid {
    context: Context,
    cell: Cell,
    channels: usize,
    /// Per grid index: `K(G)` (zero on the Nyquist planes), the `outside` weight, `s(G)`, the cell
    /// index `u` of the frequencies of `G`, and `G` itself.
    kernel: Buffer<f64>,
    outside: Buffer<f64>,
    scale: Buffer<C64>,
    cell_of: Buffer<u32>,
    wave: Buffer<f64>,
    /// `shape[c * points + g]`: the compensation shape of channel `c` at `G`.
    shape: Buffer<C64>,
    /// The self-energy of a unit charge of each channel beyond the grid's sphere.
    tails: Vec<f64>,
    fft: Mutex<Fft3>,
}

impl Grid {
    /// The tables of `cell` and `kernel_of` for the atoms of `setup`.
    pub(in crate::periodic) fn new(
        crystal: &Crystal,
        setup: &Setup,
        cell: &Cell,
        kernel_of: Kernel,
    ) -> Result<Arc<Self>> {
        let ctx = &crystal.context;
        let points = cell.points();
        let channels = crystal.channels;
        let mut kernel = ctx.allocate::<f64>(points)?;
        let mut outside = ctx.allocate::<f64>(points)?;
        let mut scale = ctx.allocate::<C64>(points)?;
        let mut cell_of = ctx.allocate::<u32>(points)?;
        let mut wave = ctx.allocate::<f64>(product(&[3, points])?)?;
        let mut shape = ctx.allocate::<C64>(product(&[channels, points])?)?;
        let sphere2 = cell.grid_sphere().powi(2);
        let sites = setup.sites();
        let mut block = vec![C64::default(); channels * BLOCK.min(points)];
        for start in (0..points).step_by(BLOCK) {
            let count = BLOCK.min(points - start);
            let mut vectors = Vec::with_capacity(count);
            for g in start..start + count {
                let m = cell.frequencies(g);
                let v = cell.vector(m);
                let g2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                // A Nyquist plane carries no interaction (see `pairs::coulomb_rows`); the
                // sphere's own surface belongs to the tail.
                let k = if cell.at_nyquist(m) {
                    0.
                } else {
                    kernel_of.value(g2)
                };
                kernel.as_mut_slice()[g] = k;
                outside.as_mut_slice()[g] = if g2 >= sphere2 * (1. - 1e-12) { k } else { 0. };
                // dv e^{-i G.r0}.
                let angle =
                    -(v[0] * cell.origin[0] + v[1] * cell.origin[1] + v[2] * cell.origin[2]);
                scale.as_mut_slice()[g] = C64::from_polar(cell.dv, angle);
                cell_of.as_mut_slice()[g] = u32::try_from(cell.cell_of(m))
                    .map_err(|_| invalid("the supercell has too many cells"))?;
                wave.as_mut_slice()[3 * g..3 * g + 3].copy_from_slice(&v);
                vectors.push(v);
            }
            if channels > 0 {
                setup.shapes().transforms(
                    &sites,
                    &vectors,
                    &mut block[..channels * count],
                    count,
                )?;
                for c in 0..channels {
                    shape.as_mut_slice()[c * points + start..][..count]
                        .copy_from_slice(&block[c * count..][..count]);
                }
            }
        }
        let cutoff = cell.grid_sphere();
        let tails = sites
            .channels
            .iter()
            .map(|&(atom, l, _)| {
                setup
                    .shapes()
                    .tail(sites.atoms[atom].compensation_radius, l, cutoff)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(Self {
            context: ctx.clone(),
            cell: cell.clone(),
            channels,
            kernel,
            outside,
            scale,
            cell_of,
            wave,
            shape,
            tails,
            fft: Mutex::new(Fft3::new(ctx, cell.total)?),
        }))
    }

    fn fft(&self) -> MutexGuard<'_, Fft3> {
        self.fft.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `G` carries no interaction at all (a Nyquist plane, or `G = 0` of the periodic
    /// kernel): both its terms vanish.
    fn silent(&self, g: usize) -> bool {
        self.kernel.as_slice()[g] == 0. && self.outside.as_slice()[g] == 0.
    }

    /// `sum_c g_c(G) sums[c * cells + u(G)]` at every grid index.
    fn compensation(&self, sums: &[C64], out: &mut [C64]) {
        let (points, cells) = (self.cell.points(), self.cell.cells);
        out.fill(C64::default());
        for c in 0..self.channels {
            let row = &sums[c * cells..][..cells];
            let shape = &self.shape.as_slice()[c * points..][..points];
            for ((o, s), u) in out.iter_mut().zip(shape).zip(self.cell_of.as_slice()) {
                *o += s * row[*u as usize];
            }
        }
    }

    /// `Z[c * cells + u] = sum_{G in u} g_c(G) weight(G)`.
    fn collect(&self, weight: &[C64]) -> Vec<C64> {
        let (points, cells) = (self.cell.points(), self.cell.cells);
        let mut z = vec![C64::default(); self.channels * cells];
        for c in 0..self.channels {
            let shape = &self.shape.as_slice()[c * points..][..points];
            for ((u, w), s) in self.cell_of.as_slice().iter().zip(weight).zip(shape) {
                z[c * cells + *u as usize] += w * s;
            }
        }
        z
    }
}

/// The functions of a supercell, their compensation charges and the grid they interact on.
pub(in crate::periodic) struct Streamed {
    grid: Arc<Grid>,
    n: usize,
    values: Vec<Buffer<C64>>,
    /// The compensation charges, their lattice sums and the projector coefficients of the pairs.
    charges: Pairs,
    /// The charges and their lattice sums (`channels x cells x n^2` each) as held.
    _tables: Reservation,
}

impl Streamed {
    /// The functions `functions` (all of them, each at the home cell) on `grid`.
    pub(in crate::periodic) fn new(
        crystal: &Crystal,
        setup: &Setup,
        grid: Arc<Grid>,
        functions: Functions,
    ) -> Result<Self> {
        let n = functions.ids.len();
        let modes: Vec<Mode> = (0..n).map(|at| Mode { at, cell: [0; 3] }).collect();
        let charges = Pairs::charges_only(crystal, setup, &grid.cell, &functions, &modes)?;
        let tables = grid.context.reserve(product(&[
            grid.channels.max(1),
            grid.cell.cells,
            n * n,
            32,
        ])?)?;
        Ok(Self {
            grid,
            n,
            values: functions.values,
            charges,
            _tables: tables,
        })
    }

    /// The charges and projector coefficients of the pairs ([`Pairs::charges_only`]).
    pub(in crate::periodic) fn charges(&self) -> &Pairs {
        &self.charges
    }

    pub(in crate::periodic) fn context(&self) -> &Context {
        &self.grid.context
    }

    pub(in crate::periodic) fn cell(&self) -> &Cell {
        &self.grid.cell
    }

    /// `(<phi_a|S|phi_b>, rho_ab(G = 0))` as [`Pairs::norms`]: the first from the grid overlap
    /// plus the overlap correction of the projectors, the second from the zeroth Fourier component
    /// of the pair density (its compensation charges carry the same overlap in their monopole).
    pub(in crate::periodic) fn norms(&self, setup: &Setup) -> (DMatrix<C64>, DMatrix<C64>) {
        let (n, cells) = (self.n, self.grid.cell.cells);
        let dv = self.grid.scale.as_slice()[0];
        let smooth = DMatrix::from_fn(n, n, |a, b| {
            let (left, right) = (self.values[a].as_slice(), self.values[b].as_slice());
            let sum: C64 = left.iter().zip(right).map(|(l, r)| l.conj() * r).sum();
            sum * dv
        });
        let metric = self.charges.corrected_overlap(setup, |a, b| smooth[(a, b)]);
        // The compensation part at G = 0, the first grid index (the cell index 0).
        let charge = DMatrix::from_fn(n, n, |a, b| {
            let compact: C64 = (0..self.grid.channels)
                .map(|c| {
                    self.grid.shape.as_slice()[c * self.grid.cell.points()]
                        * self.charges.sums[(c * cells) * n * n + a * n + b]
                })
                .sum();
            smooth[(a, b)] + compact
        });
        (metric, charge)
    }

    /// The Fourier transform of `phi_l^* phi_r` on the grid in `out` (not yet scaled).
    fn transform_pair(&self, fft: &mut Fft3, l: usize, r: usize, out: &mut [C64]) -> Result<()> {
        let (left, right) = (self.values[l].as_slice(), self.values[r].as_slice());
        for ((o, a), b) in out.iter_mut().zip(left).zip(right) {
            *o = a.conj() * b;
        }
        fft.transform(out, false)
    }

    /// The lattice sums of the pair `ab` of every channel, `[c * cells + u]`.
    fn pair_sums(&self, ab: usize) -> Vec<C64> {
        let (n2, cells) = (self.n * self.n, self.grid.cell.cells);
        let mut out = Vec::with_capacity(self.grid.channels * cells);
        for c in 0..self.grid.channels {
            for u in 0..cells {
                out.push(self.charges.sums[(c * cells + u) * n2 + ab]);
            }
        }
        out
    }

    /// The self-energy beyond the sphere of the charges `left` (the swapped pair) and `right`:
    /// `sum_{c,t} tail_c conj(Q^{left}_{cT}) Q^{right}_{cT}`.
    fn tail(&self, left: usize, right: usize) -> C64 {
        let (n2, cells) = (self.n * self.n, self.grid.cell.cells);
        let mut sum = C64::default();
        for (c, tail) in self.grid.tails.iter().enumerate() {
            for t in 0..cells {
                let q = &self.charges.charges[(c * cells + t) * n2..][..n2];
                sum += q[left].conj() * q[right] * *tail;
            }
        }
        sum
    }

    /// `(pr|pr)` of each pair `(p, r)` of `pairs`, without the one-center remainder.
    pub(in crate::periodic) fn diagonal(&self, pairs: &[(usize, usize)]) -> Result<Vec<C64>> {
        let (n, grid) = (self.n, &*self.grid);
        let points = grid.cell.points();
        let _work = grid.context.reserve(product(&[points, 16, 4])?)?;
        let mut fft = grid.fft();
        let mut left = vec![C64::default(); points];
        let mut right = vec![C64::default(); points];
        let (mut left_compact, mut right_compact) =
            (vec![C64::default(); points], vec![C64::default(); points]);
        let mut out = Vec::with_capacity(pairs.len());
        for &(p, r) in pairs {
            // The left density of (pr|pr) is the swapped pair (r, p).
            self.transform_pair(&mut fft, r, p, &mut left)?;
            self.transform_pair(&mut fft, p, r, &mut right)?;
            grid.compensation(&self.pair_sums(r * n + p), &mut left_compact);
            grid.compensation(&self.pair_sums(p * n + r), &mut right_compact);
            let mut sum = C64::default();
            for g in 0..points {
                if grid.silent(g) {
                    continue;
                }
                let (k, outside) = (grid.kernel.as_slice()[g], grid.outside.as_slice()[g]);
                let s = grid.scale.as_slice()[g];
                let (l, lc) = (left[g] * s + left_compact[g], left_compact[g]);
                let (rr, rc) = (right[g] * s + right_compact[g], right_compact[g]);
                sum += l.conj() * rr * k - lc.conj() * rc * outside;
            }
            out.push(sum / grid.cell.volume + self.tail(r * n + p, p * n + r));
        }
        Ok(out)
    }

    /// `(pr|qs)` for every `(q, s)`, `[q * n + s]`, of the pivot pair `(p, r)`, without the
    /// one-center remainder.
    pub(in crate::periodic) fn column(&self, p: usize, r: usize) -> Result<Vec<C64>> {
        let (n, grid) = (self.n, &*self.grid);
        let (points, cells) = (grid.cell.points(), grid.cell.cells);
        if p >= n || r >= n {
            return Err(invalid("a Coulomb pivot is out of range"));
        }
        let swapped = r * n + p;
        let _work = grid.context.reserve(product(&[points, 16, 2])?)?;
        let mut work = vec![C64::default(); points];
        let mut compact = vec![C64::default(); points];
        // The left density rho_{rp} and the weights W and U.
        let mut fft = grid.fft();
        self.transform_pair(&mut fft, r, p, &mut work)?;
        grid.compensation(&self.pair_sums(swapped), &mut compact);
        for g in 0..points {
            if grid.silent(g) {
                work[g] = C64::default();
                compact[g] = C64::default();
                continue;
            }
            let (k, outside) = (grid.kernel.as_slice()[g], grid.outside.as_slice()[g]);
            let s = grid.scale.as_slice()[g];
            let lc = compact[g];
            let l = work[g] * s + lc;
            work[g] = l.conj() * k * s;
            compact[g] = l.conj() * k - lc.conj() * outside;
        }
        fft.transform(&mut work, false)?;
        drop(fft);
        let z = grid.collect(&compact);
        let mut out = self.overlaps(&work)?;
        // The compensation charges, and the 1/V of the Fourier sum.
        let n2 = n * n;
        for (cu, weight) in z.iter().enumerate() {
            let row = &self.charges.sums[cu * n2..][..n2];
            for (o, s) in out.iter_mut().zip(row) {
                *o += weight * s;
            }
        }
        let scale = 1. / grid.cell.volume;
        for o in out.iter_mut() {
            *o *= scale;
        }
        for (c, tail) in grid.tails.iter().enumerate() {
            for t in 0..cells {
                let q = &self.charges.charges[(c * cells + t) * n2..][..n2];
                let left = q[swapped].conj() * *tail;
                for (o, right) in out.iter_mut().zip(q) {
                    *o += left * right;
                }
            }
        }
        Ok(out)
    }

    /// The functions on the grid points `start..start + count` as planes `[re, im]` of an
    /// `n x count` column-major matrix (function fastest), with the imaginary part negated for
    /// the conjugate.
    fn block(&self, start: usize, count: usize, conjugate: bool, planes: &mut [Vec<f64>; 2]) {
        let n = self.n;
        let sign = if conjugate { -1. } else { 1. };
        for q in 0..n {
            let f = &self.values[q].as_slice()[start..][..count];
            for (x, v) in f.iter().enumerate() {
                planes[0][q + n * x] = v.re;
                planes[1][q + n * x] = sign * v.im;
            }
        }
    }

    /// `M[q * n + s] = sum_x phi_q^*(x) weight(x) phi_s(x)` over the grid, by blocks.
    fn overlaps(&self, weight: &[C64]) -> Result<Vec<C64>> {
        let n = self.n;
        let points = self.grid.cell.points();
        let block = BLOCK.min(points);
        let _work = self
            .grid
            .context
            .reserve(product(&[n, block, 8, 4])? + product(&[n, n, 8, 2])?)?;
        let mut planes = [vec![0.; n * n], vec![0.; n * n]];
        let mut left = [vec![0.; n * block], vec![0.; n * block]];
        let mut right = [vec![0.; block * n], vec![0.; block * n]];
        for start in (0..points).step_by(block) {
            let count = block.min(points - start);
            self.block(start, count, true, &mut left);
            for q in 0..n {
                let f = &self.values[q].as_slice()[start..][..count];
                for (x, (v, w)) in f.iter().zip(&weight[start..start + count]).enumerate() {
                    let t = w * v;
                    right[0][x + count * q] = t.re;
                    right[1][x + count * q] = t.im;
                }
            }
            // Re M += Ar Tr - Ai Ti, Im M += Ar Ti + Ai Tr.
            for (plane, l, t, alpha) in
                [(0, 0, 0, 1.), (0, 1, 1, -1.), (1, 0, 1, 1.), (1, 1, 0, 1.)]
            {
                multiply(
                    MatrixRef {
                        values: &left[l],
                        rows: n,
                        columns: count,
                        leading: n,
                    },
                    MatrixRef {
                        values: &right[t],
                        rows: count,
                        columns: n,
                        leading: count,
                    },
                    MatrixMut {
                        values: &mut planes[plane],
                        rows: n,
                        columns: n,
                        leading: n,
                    },
                    alpha,
                    1.,
                )?;
            }
        }
        let mut out = vec![C64::default(); n * n];
        for q in 0..n {
            for s in 0..n {
                out[q * n + s] = C64::new(planes[0][q + n * s], planes[1][q + n * s]);
            }
        }
        Ok(out)
    }

    /// `d(x) = sum_rs w[r * n + s] phi_r^*(x) phi_s(x)` on the grid, by blocks.
    fn density(&self, w: &[C64], out: &mut [C64]) -> Result<()> {
        let n = self.n;
        let points = self.grid.cell.points();
        let block = BLOCK.min(points);
        let _work = self
            .grid
            .context
            .reserve(product(&[n, block, 8, 4])? + product(&[n, n, 8, 2])?)?;
        // W[r, s] column-major, and whether it has an imaginary part at all.
        let mut weights = [vec![0.; n * n], vec![0.; n * n]];
        for r in 0..n {
            for s in 0..n {
                weights[0][r + n * s] = w[r * n + s].re;
                weights[1][r + n * s] = w[r * n + s].im;
            }
        }
        let complex = weights[1].iter().any(|v| *v != 0.);
        let mut functions = [vec![0.; n * block], vec![0.; n * block]];
        let mut product_planes = [vec![0.; n * block], vec![0.; n * block]];
        for start in (0..points).step_by(block) {
            let count = block.min(points - start);
            self.block(start, count, false, &mut functions);
            // E = W Phi: Re E = Wr Pr - Wi Pi, Im E = Wr Pi + Wi Pr.
            let mut steps = vec![(0, 0, 0, 1., 0.), (1, 0, 1, 1., 0.)];
            if complex {
                steps.extend([(0, 1, 1, -1., 1.), (1, 1, 0, 1., 1.)]);
            }
            for (plane, w_plane, f_plane, alpha, beta) in steps {
                multiply(
                    MatrixRef {
                        values: &weights[w_plane],
                        rows: n,
                        columns: n,
                        leading: n,
                    },
                    MatrixRef {
                        values: &functions[f_plane],
                        rows: n,
                        columns: count,
                        leading: n,
                    },
                    MatrixMut {
                        values: &mut product_planes[plane],
                        rows: n,
                        columns: count,
                        leading: n,
                    },
                    alpha,
                    beta,
                )?;
            }
            for x in 0..count {
                let mut sum = C64::default();
                for r in 0..n {
                    let phi = C64::new(functions[0][r + n * x], functions[1][r + n * x]);
                    sum += phi.conj()
                        * C64::new(product_planes[0][r + n * x], product_planes[1][r + n * x]);
                }
                out[start + x] = sum;
            }
        }
        Ok(())
    }

    /// The smooth part of the right density `D(G) = sum_rs w_rs rho_rs(G)` in `work` and its
    /// compensation part in `compact`.
    fn contracted_density(
        &self,
        fft: &mut Fft3,
        w: &[C64],
        work: &mut [C64],
        compact: &mut [C64],
    ) -> Result<()> {
        let (n2, cells) = (self.n * self.n, self.grid.cell.cells);
        self.density(w, work)?;
        fft.transform(work, false)?;
        let mut sums = vec![C64::default(); self.grid.channels * cells];
        for (cu, slot) in sums.iter_mut().enumerate() {
            let row = &self.charges.sums[cu * n2..][..n2];
            *slot = w.iter().zip(row).map(|(x, y)| x * y).sum();
        }
        self.grid.compensation(&sums, compact);
        for (d, s) in work.iter_mut().zip(self.grid.scale.as_slice()) {
            *d *= s;
        }
        Ok(())
    }

    /// `sum_rs w_f[r * n + s] (pq|rs)` for every left pair `rows` (`p * n + q`) and every weight
    /// set, `values[row * weights.len() + f]`: the smooth products, the compensation charges
    /// inside the grid's sphere, their self-energy beyond it and the one-center remainder
    /// ([`super::pairs::onsite_contracted`]). With `Y = K D_f`, `Yc = outside Dc_f` and
    /// `T = s conj(Y)`, a row's value is `conj(M_T[q, p] + sum_{cu} S_{cu}^{qp} Z_{cu}) / V` with
    /// `M_T` the product of [`Streamed::overlaps`] with `F[T]` and `Z_{cu} = sum_{G in u} g_c(G)
    /// conj(Y - Yc)(G)`, plus the tail.
    pub(in crate::periodic) fn contracted(
        &self,
        setup: &Setup,
        rows: &[usize],
        weights: &[Vec<C64>],
    ) -> Result<Vec<C64>> {
        let (n, grid) = (self.n, &*self.grid);
        let (n2, points, cells) = (n * n, grid.cell.points(), grid.cell.cells);
        if weights.is_empty() || weights.iter().any(|w| w.len() != n2) {
            return Err(invalid(
                "every weight set of a contracted Coulomb row needs one weight per right pair",
            ));
        }
        if rows.iter().any(|&pq| pq >= n2) {
            return Err(invalid("a left pair of the Coulomb rows is out of range"));
        }
        let forms = weights.len();
        let _work = grid.context.reserve(product(&[points, 16, 3])?)?;
        let _values = grid.context.reserve(product(&[rows.len(), forms, 16])?)?;
        let mut values = vec![C64::default(); rows.len() * forms];
        let mut work = vec![C64::default(); points];
        let mut compact = vec![C64::default(); points];
        let mut potential = vec![C64::default(); points];
        let scale = 1. / grid.cell.volume;
        for (f, w) in weights.iter().enumerate() {
            let mut fft = grid.fft();
            self.contracted_density(&mut fft, w, &mut work, &mut compact)?;
            // Y - Yc and T.
            for g in 0..points {
                if grid.silent(g) {
                    work[g] = C64::default();
                    potential[g] = C64::default();
                    continue;
                }
                let (k, outside) = (grid.kernel.as_slice()[g], grid.outside.as_slice()[g]);
                let y = (work[g] + compact[g]) * k;
                potential[g] = (y - compact[g] * outside).conj();
                work[g] = grid.scale.as_slice()[g] * y.conj();
            }
            fft.transform(&mut work, false)?;
            drop(fft);
            let z = grid.collect(&potential);
            let mut matrix = self.overlaps(&work)?;
            for (cu, weight) in z.iter().enumerate() {
                for (pair, slot) in matrix.iter_mut().enumerate() {
                    *slot += weight * self.charges.sums[cu * n2 + pair];
                }
            }
            let charge: Vec<C64> = {
                let mut by_channel = Vec::with_capacity(grid.channels * cells);
                for c in 0..grid.channels {
                    for t in 0..cells {
                        let q = &self.charges.charges[(c * cells + t) * n2..][..n2];
                        by_channel.push(w.iter().zip(q).map(|(x, y)| x * y).sum());
                    }
                }
                by_channel
            };
            for (row, &pq) in rows.iter().enumerate() {
                let (p, q) = (pq / n, pq % n);
                let swapped = q * n + p;
                let mut value = matrix[q * n + p].conj() * scale;
                for (c, tail) in grid.tails.iter().enumerate() {
                    for t in 0..cells {
                        let charges = self.pair_charges_at(c, t, swapped);
                        value += *tail * charges.conj() * charge[c * cells + t];
                    }
                }
                values[row * forms + f] = value;
            }
        }
        onsite_contracted(
            &grid.context,
            setup,
            &grid.cell,
            &self.charges,
            rows,
            weights,
            &mut values,
        )?;
        Ok(values)
    }

    fn pair_charges_at(&self, channel: usize, cell: usize, pair: usize) -> C64 {
        let (n2, cells) = (self.n * self.n, self.grid.cell.cells);
        self.charges.charges[(channel * cells + cell) * n2 + pair]
    }

    /// The motion of the compensation shapes of every atom (all its copies moving together) in the
    /// Coulomb energy of each contracted density `rho_f = sum_pq w_f[p n + q] rho_pq`, at fixed
    /// charges: `out[atom][f][axis] = (1/V) sum_G conj(d c'_f(G)) [K(G) rho_f(G) - outside(G)
    /// c_f(G)]` with `c_f` the compensation part of `rho_f`, `c'_f` that of the left density
    /// `sum_pq w_f[p n + q] rho_qp` (the swapped pairs, as the left of `(pq|rs)` enters), and `d` the
    /// `-i G_axis` of the shapes of the atom (each carries `e^{-iG.R_atom}`). The self-energy beyond
    /// the sphere and the one-center remainder do not move with the shapes. For real symmetric
    /// weights the real part is `d/dR_axis 1/2 (rho_f|rho_f)` from the shapes alone.
    pub(in crate::periodic) fn shape_motions(
        &self,
        setup: &Setup,
        weights: &[Vec<C64>],
    ) -> Result<Vec<Vec<[C64; 3]>>> {
        let (n, grid) = (self.n, &*self.grid);
        let (n2, points, cells, channels) =
            (n * n, grid.cell.points(), grid.cell.cells, grid.channels);
        if weights.iter().any(|w| w.len() != n2) {
            return Err(invalid("a contracted density needs one weight per pair"));
        }
        let sites = setup.sites();
        let atoms = sites.atoms.len();
        let owner: Vec<usize> = sites.channels.iter().map(|&(a, _, _)| a).collect();
        let _work = grid.context.reserve(product(&[points, 16, 2])?)?;
        let mut out = vec![vec![[C64::default(); 3]; weights.len()]; atoms];
        let mut work = vec![C64::default(); points];
        let mut compact = vec![C64::default(); points];
        for (f, w) in weights.iter().enumerate() {
            let mut fft = grid.fft();
            self.contracted_density(&mut fft, w, &mut work, &mut compact)?;
            drop(fft);
            // The left density sum_pq w[pq] rho_qp has the swapped lattice sums.
            let mut left_sums = vec![C64::default(); channels * cells];
            for (cu, slot) in left_sums.iter_mut().enumerate() {
                let row = &self.charges.sums[cu * n2..][..n2];
                for p in 0..n {
                    for q in 0..n {
                        *slot += w[p * n + q] * row[q * n + p];
                    }
                }
            }
            for g in 0..points {
                if grid.silent(g) {
                    continue;
                }
                let (k, outside) = (grid.kernel.as_slice()[g], grid.outside.as_slice()[g]);
                let u = grid.cell_of.as_slice()[g] as usize;
                let mut moving = vec![C64::default(); atoms];
                for c in 0..channels {
                    moving[owner[c]] +=
                        grid.shape.as_slice()[c * points + g] * left_sums[c * cells + u];
                }
                let partner = work[g] + compact[g];
                let partner = partner * k - compact[g] * outside;
                let v = &grid.wave.as_slice()[3 * g..3 * g + 3];
                for (atom, m) in moving.iter().enumerate() {
                    for axis in 0..3 {
                        // d/dR e^{-iG.R} = -i G e^{-iG.R}.
                        let slope = C64::new(0., -v[axis]) * m;
                        out[atom][f][axis] += slope.conj() * partner;
                    }
                }
            }
        }
        let scale = 1. / grid.cell.volume;
        for atom in &mut out {
            for form in atom.iter_mut() {
                for v in form.iter_mut() {
                    *v *= scale;
                }
            }
        }
        Ok(out)
    }
}
