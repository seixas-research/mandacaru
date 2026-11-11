//! Pair densities `rho_pq = w_p^* w_q` of a fragment's functions on the supercell's reciprocal
//! grid, and the Coulomb integrals between them.
//!
//! In PAW the density of a pair is the smooth product on the grid plus the compensation charges,
//! one shape `g_c` per multipole channel `c = (atom, L, M)` of every atom copy `T` of the
//! supercell,
//!
//! ```text
//! rho_pq(G) = dv e^{-i G.r0} DFT[ w~_p^* w~_q ](G)  +  sum_{c,T} Q^{cT}_{pq} g_c(G) e^{-i G.T},
//! Q^{cT}_{pq} = sum_{ij} <w_p|p_iT> B_c[i,j] <p_jT|w_q>.
//! ```
//!
//! The lattice sum over `T` depends on `G` only through `G mod` the cell's reciprocal lattice, so
//! it is kept as a small table `S_c(u)` per pair (`u` the cell index of the frequencies of `G`).
//! Atomic units; `rho(G) = int rho(r) e^{-iG.r}`.
use super::{
    coulomb::{truncated, Cell},
    functions::{axis_sources, shifted_coefficients, Functions},
};
use crate::{
    math::fft::Fft3,
    paw::onecenter::{Basis, Projector as OneCenterProjector},
    periodic::{
        invalid, response::setup::Setup, screening::transitions::Remainders,
        setup::remainder::onsite_remainder, Crystal,
    },
    product,
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::{DMatrix, Vector3};
use std::f64::consts::PI;

/// One mode of a fragment: the position of its function in [`Functions::ids`] and its cell.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    pub at: usize,
    pub cell: [i64; 3],
}

/// Reciprocal-grid blocks the lattice sums and the contractions are evaluated in.
const BLOCK: usize = 4096;

pub struct Pairs {
    pub n: usize,
    pub points: usize,
    /// `smooth[(p * n + q) * points + g]`: the transform of the smooth product on the supercell.
    pub smooth: Buffer<C64>,
    /// `charges[(c * cells + t) * n * n + p * n + q]`: `Q^{cT}_{pq}`.
    pub charges: Vec<C64>,
    /// `sums[(c * cells + u) * n * n + p * n + q]`: `sum_T Q^{cT}_{pq} e^{-2 pi i u.T / size}`.
    pub sums: Vec<C64>,
    /// `coefficients[(p * cells + t) * P + j]`: `<p_{jT}|w_p>` of the fragment's mode `p`.
    pub coefficients: Vec<C64>,
    pub channels: usize,
    pub projectors: usize,
    pub cells: usize,
}

impl Pairs {
    /// The pair densities of `modes` (of `functions`), with the charges of every channel.
    pub(in crate::periodic) fn new(
        crystal: &Crystal,
        setup: &Setup,
        cell: &Cell,
        functions: &Functions,
        modes: &[Mode],
    ) -> Result<Self> {
        let ctx = &crystal.context;
        let n = modes.len();
        let points = cell.points();
        let p = crystal.p;
        let channels = crystal.channels;
        if n == 0 || modes.iter().any(|m| m.at >= functions.ids.len()) {
            return Err(invalid("a fragment needs functions of the channel"));
        }
        let smooth = ctx.allocate::<C64>(product(&[n, n, points])?)?;
        let mut pairs = Self {
            n,
            points,
            smooth,
            charges: Vec::new(),
            sums: Vec::new(),
            coefficients: Vec::new(),
            channels,
            projectors: p,
            cells: cell.cells,
        };
        let _scratch = ctx.reserve(product(&[channels.max(1), cell.cells, n * n, 16, 2])?)?;
        // The projector coefficients of every mode, moved to its cell.
        for mode in modes {
            pairs
                .coefficients
                .extend(shifted_coefficients(functions, cell, mode.at, mode.cell));
        }
        pairs.fill_smooth(ctx, cell, functions, modes)?;
        pairs.fill_charges(crystal, setup, cell)?;
        Ok(pairs)
    }

    fn fill_smooth(
        &mut self,
        ctx: &Context,
        cell: &Cell,
        functions: &Functions,
        modes: &[Mode],
    ) -> Result<()> {
        let n = self.n;
        let total = cell.total;
        let points = self.points;
        let mut fft = Fft3::new(ctx, cell.total)?;
        let mut work = ctx.allocate::<C64>(points)?;
        // dv e^{-i G.r0} per grid index.
        let mut scale = ctx.allocate::<C64>(points)?;
        for (g, slot) in scale.as_mut_slice().iter_mut().enumerate() {
            let v = cell.wavevector(g);
            let angle = -(v[0] * cell.origin[0] + v[1] * cell.origin[1] + v[2] * cell.origin[2]);
            *slot = C64::from_polar(cell.dv, angle);
        }
        let sources: Vec<[Vec<usize>; 3]> =
            modes.iter().map(|m| axis_sources(cell, m.cell)).collect();
        for p in 0..n {
            let left = functions.values[modes[p].at].as_slice();
            for q in 0..n {
                let right = functions.values[modes[q].at].as_slice();
                let (sp, sq) = (&sources[p], &sources[q]);
                let out = work.as_mut_slice();
                for x0 in 0..total[0] {
                    for x1 in 0..total[1] {
                        let row = (x0 * total[1] + x1) * total[2];
                        let bp = (sp[0][x0] * total[1] + sp[1][x1]) * total[2];
                        let bq = (sq[0][x0] * total[1] + sq[1][x1]) * total[2];
                        for x2 in 0..total[2] {
                            out[row + x2] = left[bp + sp[2][x2]].conj() * right[bq + sq[2][x2]];
                        }
                    }
                }
                fft.transform(out, false)?;
                let target = &mut self.smooth.as_mut_slice()[(p * n + q) * points..][..points];
                for ((t, f), s) in target.iter_mut().zip(out.iter()).zip(scale.as_slice()) {
                    *t = f * s;
                }
            }
        }
        Ok(())
    }

    fn fill_charges(&mut self, crystal: &Crystal, setup: &Setup, cell: &Cell) -> Result<()> {
        let (n, cells, p, channels) = (self.n, self.cells, self.projectors, self.channels);
        let owners = setup.projector_atoms();
        let sites = setup.sites();
        self.charges = vec![C64::default(); channels * cells * n * n];
        self.sums = vec![C64::default(); channels * cells * n * n];
        for (c, &(atom, l, m)) in sites.channels.iter().enumerate() {
            let own: Vec<usize> = (0..p).filter(|j| owners[*j] == atom).collect();
            // The moment blocks of the channels `+-M` of one real density are related by
            // `B_{L,-M} = (-1)^M B_{LM}^+`; the radial quadrature that built them keeps that to
            // 1e-4 only, which would give a real pair density a small imaginary Coulomb energy.
            // The block is projected onto the relation (a channel without its mirror is kept).
            let mirror = sites
                .channels
                .iter()
                .position(|&(a, ll, mm)| a == atom && ll == l && mm == -m);
            let sign = if m % 2 == 0 { 1. } else { -1. };
            let raw = |k: usize, i: usize, j: usize| crystal.blocks.as_slice()[(k * p + i) * p + j];
            let block = |i: usize, j: usize| match mirror {
                Some(k) => 0.5 * (raw(c, i, j) + sign * raw(k, j, i).conj()),
                None => raw(c, i, j),
            };
            for t in 0..cells {
                for a in 0..n {
                    for b in 0..n {
                        let mut sum = C64::default();
                        for &i in &own {
                            let left = self.coefficients[(a * cells + t) * p + i].conj();
                            for &j in &own {
                                sum +=
                                    left * block(i, j) * self.coefficients[(b * cells + t) * p + j];
                            }
                        }
                        self.charges[((c * cells) + t) * n * n + a * n + b] = sum;
                    }
                }
            }
        }
        let vectors: Vec<[i64; 3]> = (0..cells).map(|t| cell.cell_vector(t)).collect();
        for c in 0..channels {
            for u in 0..cells {
                let uv = vectors[u];
                for (t, tv) in vectors.iter().enumerate() {
                    let angle = -2.
                        * PI
                        * (0..3)
                            .map(|a| (uv[a] * tv[a]) as f64 / cell.size[a] as f64)
                            .sum::<f64>();
                    let phase = C64::from_polar(1., angle);
                    for ab in 0..n * n {
                        self.sums[(c * cells + u) * n * n + ab] +=
                            phase * self.charges[(c * cells + t) * n * n + ab];
                    }
                }
            }
        }
        Ok(())
    }

    /// `(<w_p|S|w_q>, rho_pq(G = 0))` of the fragment's functions: the first from the smooth
    /// overlap on the grid plus the overlap correction `Delta S` of the projectors, the second
    /// from the zeroth Fourier component of the pair density, whose compensation charges carry
    /// the same overlap in their monopole. Both are the identity for orthonormal functions.
    pub(in crate::periodic) fn norms(
        &self,
        setup: &Setup,
        cell: &Cell,
    ) -> Result<(DMatrix<C64>, DMatrix<C64>)> {
        let (n, cells, p) = (self.n, self.cells, self.projectors);
        let correction = setup.overlap_correction();
        let mut metric = DMatrix::<C64>::zeros(n, n);
        for a in 0..n {
            for b in 0..n {
                let mut sum = self.smooth.as_slice()[(a * n + b) * self.points];
                for t in 0..cells {
                    for i in 0..p {
                        let left = self.coefficients[(a * cells + t) * p + i].conj();
                        for j in 0..p {
                            sum += left
                                * correction[(i, j)]
                                * self.coefficients[(b * cells + t) * p + j];
                        }
                    }
                }
                metric[(a, b)] = sum;
            }
        }
        let mut origin = vec![C64::default(); n * n];
        self.compact(setup, cell, &[0], &mut origin)?;
        let charge = DMatrix::from_fn(n, n, |a, b| {
            self.smooth.as_slice()[(a * n + b) * self.points] + origin[a * n + b]
        });
        Ok((metric, charge))
    }

    /// The compensation part of every pair at the supercell grid indices `indices`:
    /// `out[(p * n + q) * indices.len() + g]`.
    pub(in crate::periodic) fn compact(
        &self,
        setup: &Setup,
        cell: &Cell,
        indices: &[usize],
        out: &mut [C64],
    ) -> Result<()> {
        let mut vectors = Vec::with_capacity(indices.len());
        let mut cells = Vec::with_capacity(indices.len());
        let mut nyquist = Vec::with_capacity(indices.len());
        for &g in indices {
            let m = cell.frequencies(g);
            vectors.push(cell.vector(m));
            cells.push(cell.cell_of(m));
            nyquist.push(cell.at_nyquist(m));
        }
        self.compact_at(setup, &vectors, &cells, out)?;
        // A compact charge's Nyquist component is an alias of its transform, and the frequency
        // has no `-G` partner on the grid, so the charge of a real density would not stay real
        // there: it is left out (the Fourier sum of the compensation never reaches that far).
        let count = indices.len();
        for (g, flagged) in nyquist.iter().enumerate() {
            if *flagged {
                for ab in 0..self.n * self.n {
                    out[ab * count + g] = C64::default();
                }
            }
        }
        Ok(())
    }

    /// The compensation part at the wave vectors `vectors`, whose frequencies' lattice sums are
    /// read at the cell indices `cells`.
    fn compact_at(
        &self,
        setup: &Setup,
        vectors: &[[f64; 3]],
        cells: &[usize],
        out: &mut [C64],
    ) -> Result<()> {
        let n2 = self.n * self.n;
        let count = vectors.len();
        if out.len() != n2 * count || cells.len() != count {
            return Err(invalid("the compensation block has the wrong size"));
        }
        out.fill(C64::default());
        let sites = setup.sites();
        let mut shape = vec![C64::default(); self.channels * count];
        setup
            .shapes()
            .transforms(&sites, vectors, &mut shape, count)?;
        for c in 0..self.channels {
            for (g, &u) in cells.iter().enumerate() {
                let t = shape[c * count + g];
                let row = &self.sums[(c * self.cells + u) * n2..][..n2];
                for ab in 0..n2 {
                    out[ab * count + g] += t * row[ab];
                }
            }
        }
        Ok(())
    }

    /// The complete pair densities at the supercell frequencies `m` (`Q = B_s m`, each below the
    /// grid's Nyquist): the smooth part read from the grid transform, the compensation charges
    /// and the augmentation remainder of every atom copy, `out[(p * n + q) * m.len() + g]`.
    pub(in crate::periodic) fn at(
        &self,
        setup: &Setup,
        cell: &Cell,
        remainders: &Remainders,
        m: &[[i64; 3]],
    ) -> Result<Vec<C64>> {
        let n2 = self.n * self.n;
        let count = m.len();
        let vectors: Vec<[f64; 3]> = m.iter().map(|v| cell.vector(*v)).collect();
        let cells: Vec<usize> = m.iter().map(|v| cell.cell_of(*v)).collect();
        let mut out = vec![C64::default(); n2 * count];
        self.compact_at(setup, &vectors, &cells, &mut out)?;
        for (g, v) in m.iter().enumerate() {
            if (0..3).any(|a| 2 * v[a].unsigned_abs() as usize >= cell.total[a]) {
                return Err(invalid(
                    "a screened vector reaches the supercell grid's Nyquist wave number",
                ));
            }
            let index = (0..3).fold(0usize, |acc, a| {
                acc * cell.total[a] + v[a].rem_euclid(cell.total[a] as i64) as usize
            });
            for ab in 0..n2 {
                out[ab * count + g] += self.smooth.as_slice()[ab * self.points + index];
            }
        }
        // The remainder of each atom copy: sum_T e^{-iK.T} sum_AB conj(c_p) c_q E_AB(K).
        let translations: Vec<Vector3<f64>> = (0..cell.cells)
            .map(|t| {
                let v = cell.cell_vector(t);
                cell.lattice * Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64)
            })
            .collect();
        let p = self.projectors;
        for (columns, table) in remainders.tables(&vectors)? {
            let w = columns.len();
            for pp in 0..self.n {
                for qq in 0..self.n {
                    for (t, shift) in translations.iter().enumerate() {
                        let mut per_cell = vec![C64::default(); count];
                        for (a, &i) in columns.iter().enumerate() {
                            let left = self.coefficients[(pp * self.cells + t) * p + i].conj();
                            for (b, &j) in columns.iter().enumerate() {
                                let pair = left * self.coefficients[(qq * self.cells + t) * p + j];
                                if pair.norm_sqr() == 0. {
                                    continue;
                                }
                                let row = &table[(a * w + b) * count..(a * w + b + 1) * count];
                                for (slot, e) in per_cell.iter_mut().zip(row) {
                                    *slot += pair * e;
                                }
                            }
                        }
                        for (g, slot) in per_cell.iter().enumerate() {
                            let k = vectors[g];
                            let angle = -(k[0] * shift[0] + k[1] * shift[1] + k[2] * shift[2]);
                            out[(pp * self.n + qq) * count + g] +=
                                slot * C64::from_polar(1., angle);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Integrals `(n1, n1, n2, n2)` in chemists' order, `[((p * n1 + q) * n2 + r) * n2 + s]`.
pub struct Integrals {
    pub first: usize,
    pub second: usize,
    pub values: Vec<C64>,
}

/// `(pq|rs) = int int rho_pq(r) v(r - r') rho_rs(r')` between two sets of pair densities on the
/// same supercell: the truncated kernel of the box on the grid, the compensation charges
/// interacting only inside the sphere the grid carries, and each charge's own self-energy beyond
/// it. `radius` is the truncation radius of the kernel. With `grid_only` only the smooth products
/// on the grid interact: no compensation charge, no self-energy tail, no one-center remainder.
pub(in crate::periodic) fn coulomb(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    radius: f64,
    first: &Pairs,
    second: &Pairs,
    grid_only: bool,
) -> Result<Integrals> {
    let (n1, n2) = (first.n, second.n);
    let points = cell.points();
    if first.points != points || second.points != points {
        return Err(invalid("the pair densities belong to another supercell"));
    }
    let sphere2 = cell.grid_sphere().powi(2);
    let (a, b) = (n1 * n1, n2 * n2);
    let block = BLOCK.min(points);
    let _work = ctx.reserve(product(&[a + b, block, 16, 3])?)?;
    let mut values = vec![C64::default(); a * b];
    let (mut left, mut left_compact) = (
        vec![C64::default(); a * block],
        vec![C64::default(); a * block],
    );
    let (mut right, mut right_compact) = (
        vec![C64::default(); b * block],
        vec![C64::default(); b * block],
    );
    let mut kernel = vec![0.; block];
    let mut outside = vec![0.; block];
    for start in (0..points).step_by(block) {
        let count = block.min(points - start);
        let indices: Vec<usize> = (start..start + count).collect();
        for (g, &index) in indices.iter().enumerate() {
            let v = cell.wavevector(index);
            let g2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            // A Nyquist plane carries no interaction: an even grid's `-N/2` has no `+N/2`
            // partner, so a real pair density's transform there is not conjugate symmetric
            // (the origin phase `e^{-iG.r0}` of the grid's first node is not even), and the
            // component is an alias of whatever lies beyond the grid. The sphere's own surface
            // belongs to the tail (the planes touch it).
            let plane = cell.at_nyquist(cell.frequencies(index));
            kernel[g] = if plane { 0. } else { truncated(g2, radius) };
            outside[g] = if g2 >= sphere2 * (1. - 1e-12) {
                kernel[g]
            } else {
                0.
            };
        }
        let assemble = |pairs: &Pairs, full: &mut [C64], compact: &mut [C64]| -> Result<()> {
            let n2 = pairs.n * pairs.n;
            if grid_only {
                compact[..n2 * count].fill(C64::default());
            } else {
                pairs.compact(setup, cell, &indices, &mut compact[..n2 * count])?;
            }
            for ab in 0..n2 {
                let smooth = &pairs.smooth.as_slice()[ab * points + start..][..count];
                for g in 0..count {
                    full[ab * count + g] = smooth[g] + compact[ab * count + g];
                }
            }
            Ok(())
        };
        assemble(first, &mut left, &mut left_compact)?;
        assemble(second, &mut right, &mut right_compact)?;
        for p in 0..n1 {
            for q in 0..n1 {
                // The pair (pq) enters as the conjugate of (qp): the left density of (pq|rs).
                let swapped = q * n1 + p;
                let l = &left[swapped * count..][..count];
                let lc = &left_compact[swapped * count..][..count];
                for r in 0..b {
                    let rr = &right[r * count..][..count];
                    let rc = &right_compact[r * count..][..count];
                    let mut sum = C64::default();
                    for g in 0..count {
                        sum += l[g].conj() * rr[g] * kernel[g] - lc[g].conj() * rc[g] * outside[g];
                    }
                    values[(p * n1 + q) * b + r] += sum;
                }
            }
        }
    }
    let scale = 1. / cell.volume;
    for v in &mut values {
        *v *= scale;
    }
    if grid_only {
        return Ok(Integrals {
            first: n1,
            second: n2,
            values,
        });
    }
    // The self-energy of every compensation charge beyond the sphere, on its own site.
    let cutoff = cell.grid_sphere();
    let sites = setup.sites();
    for (c, &(atom, l, _)) in sites.channels.iter().enumerate() {
        let tail = setup
            .shapes()
            .tail(sites.atoms[atom].compensation_radius, l, cutoff)?;
        for t in 0..cell.cells {
            let qa = &first.charges[(c * cell.cells + t) * a..][..a];
            let qb = &second.charges[(c * cell.cells + t) * b..][..b];
            // The left density enters as the conjugate of the swapped pair, as in the Fourier sum
            // (`conj(rho_qp)`): the complex harmonics `Y_LM` pair with their own conjugate, so a
            // channel's self-energy is `conj(Q^{qp}_c) Q^{rs}_c`, not `Q^{pq}_c Q^{rs}_c`.
            for p in 0..n1 {
                for q in 0..n1 {
                    let left = qa[q * n1 + p].conj();
                    for j in 0..b {
                        values[(p * n1 + q) * b + j] += tail * left * qb[j];
                    }
                }
            }
        }
    }
    onsite(ctx, setup, cell, first, second, &mut values)?;
    Ok(Integrals {
        first: n1,
        second: n2,
        values,
    })
}

/// The Coulomb energy inside the augmentation spheres that the smooth pair densities and their
/// compensation charges leave out, `sum_{A,T} sum_{abcd} <w_p|a><b|w_q> DeltaW^A_{abcd} <w_r|c><d|w_s>`
/// with `DeltaW` the one-center remainder of the dataset.
fn onsite(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    first: &Pairs,
    second: &Pairs,
    values: &mut [C64],
) -> Result<()> {
    let (n1, n2) = (first.n, second.n);
    let (a, b) = (n1 * n1, n2 * n2);
    let owners = setup.projector_atoms();
    let labels = setup.projector_labels();
    let p = first.projectors;
    let mut cache: std::collections::BTreeMap<usize, crate::paw::onecenter::Tensor> =
        Default::default();
    let atoms = setup.sites().atoms.len();
    for atom in 0..atoms {
        let own: Vec<usize> = (0..p).filter(|j| owners[*j] == atom).collect();
        if own.is_empty() {
            continue;
        }
        let projectors: Vec<OneCenterProjector> = own
            .iter()
            .map(|&j| OneCenterProjector {
                l: labels[j].0,
                m: labels[j].1,
                index: labels[j].2,
                real: false,
            })
            .collect();
        // Atoms of one species share a dataset and a projector list.
        let key = setup.species()[atom];
        if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(key) {
            // The radial quadrature leaves DeltaW short of the symmetries of a Coulomb pair
            // integral by about 1e-5 of itself: (ab|cd) = (cd|ab) and (ab|cd)^* = (ba|dc). The
            // tensor is projected onto them, which keeps the interaction's pair symmetries (and
            // so a Hermitian double counting) exact.
            let raw = onsite_remainder(ctx, setup.dataset(atom), Basis::Raw, &projectors)?;
            entry.insert(coulomb_symmetrized(&raw));
        }
        let tensor = &cache[&key];
        let m = own.len();
        for t in 0..cell.cells {
            // left[pq][c][d] = sum_ab conj(a_p,a) a_q,b DeltaW[abcd]
            let mut left = vec![C64::default(); a * m * m];
            for pp in 0..n1 {
                for qq in 0..n1 {
                    for (ia, &ja) in own.iter().enumerate() {
                        let ca = first.coefficients[(pp * cell.cells + t) * p + ja].conj();
                        if ca.norm_sqr() == 0. {
                            continue;
                        }
                        for (ib, &jb) in own.iter().enumerate() {
                            let cb = first.coefficients[(qq * cell.cells + t) * p + jb];
                            let w = ca * cb;
                            if w.norm_sqr() == 0. {
                                continue;
                            }
                            let base = ((ia * m + ib) * m) * m;
                            let dst = &mut left[(pp * n1 + qq) * m * m..][..m * m];
                            for (slot, d) in dst.iter_mut().zip(&tensor.data[base..base + m * m]) {
                                *slot += w * d;
                            }
                        }
                    }
                }
            }
            for r in 0..n2 {
                for s in 0..n2 {
                    let mut right = vec![C64::default(); m * m];
                    for (ic, &jc) in own.iter().enumerate() {
                        let cc = second.coefficients[(r * cell.cells + t) * p + jc].conj();
                        for (id, &jd) in own.iter().enumerate() {
                            right[ic * m + id] =
                                cc * second.coefficients[(s * cell.cells + t) * p + jd];
                        }
                    }
                    for pq in 0..a {
                        let row = &left[pq * m * m..][..m * m];
                        let mut sum = C64::default();
                        for (x, y) in row.iter().zip(&right) {
                            sum += x * y;
                        }
                        values[pq * b + r * n2 + s] += sum;
                    }
                }
            }
        }
    }
    Ok(())
}

/// `1/4 (W[abcd] + W[cdab] + conj W[badc] + conj W[dcba])`: the part of a one-center tensor that
/// has the symmetries of a Coulomb pair integral of complex orbitals.
fn coulomb_symmetrized(w: &crate::paw::onecenter::Tensor) -> crate::paw::onecenter::Tensor {
    let n = w.n;
    let at = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    let mut data = vec![C64::default(); w.data.len()];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    data[at(a, b, c, d)] = (w.data[at(a, b, c, d)]
                        + w.data[at(c, d, a, b)]
                        + w.data[at(b, a, d, c)].conj()
                        + w.data[at(d, c, b, a)].conj())
                        * 0.25;
                }
            }
        }
    }
    crate::paw::onecenter::Tensor {
        n,
        real: w.real,
        data,
    }
}
