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
    coulomb::{Cell, Kernel},
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

    /// The projector coefficients and compensation charges of `modes` without their smooth
    /// products: `smooth` is empty. The streamed columns ([`super::streamed`]) hold the functions
    /// themselves and form each pair density's transform when a column asks for it; of this
    /// value they read only `charges`, `sums` and `coefficients`.
    pub(in crate::periodic) fn charges_only(
        crystal: &Crystal,
        setup: &Setup,
        cell: &Cell,
        functions: &Functions,
        modes: &[Mode],
    ) -> Result<Self> {
        let ctx = &crystal.context;
        let n = modes.len();
        if n == 0 || modes.iter().any(|m| m.at >= functions.ids.len()) {
            return Err(invalid("a fragment needs functions of the channel"));
        }
        let mut pairs = Self {
            n,
            points: cell.points(),
            smooth: ctx.allocate::<C64>(0)?,
            charges: Vec::new(),
            sums: Vec::new(),
            coefficients: Vec::new(),
            channels: crystal.channels,
            projectors: crystal.p,
            cells: cell.cells,
        };
        let _scratch = ctx.reserve(product(&[
            crystal.channels.max(1),
            cell.cells,
            n * n,
            16,
            2,
        ])?)?;
        for mode in modes {
            pairs
                .coefficients
                .extend(shifted_coefficients(functions, cell, mode.at, mode.cell));
        }
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
        let (n, points) = (self.n, self.points);
        let metric =
            self.corrected_overlap(setup, |a, b| self.smooth.as_slice()[(a * n + b) * points]);
        let mut origin = vec![C64::default(); n * n];
        self.compact(setup, cell, &[0], &mut origin)?;
        let charge = DMatrix::from_fn(n, n, |a, b| {
            self.smooth.as_slice()[(a * n + b) * self.points] + origin[a * n + b]
        });
        Ok((metric, charge))
    }

    /// `<w_a|S|w_b>` from the smooth overlap `smooth_overlap(a, b)` on the grid (the pair density's
    /// zeroth component) plus the overlap correction `Delta S` of the projectors.
    pub(in crate::periodic) fn corrected_overlap(
        &self,
        setup: &Setup,
        smooth_overlap: impl Fn(usize, usize) -> C64,
    ) -> DMatrix<C64> {
        let (n, cells, p) = (self.n, self.cells, self.projectors);
        let correction = setup.overlap_correction();
        let mut metric = DMatrix::<C64>::zeros(n, n);
        for a in 0..n {
            for b in 0..n {
                let mut sum = smooth_overlap(a, b);
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
        metric
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
/// same supercell: the `kernel` of the box on the grid, the compensation charges interacting
/// only inside the sphere the grid carries, and each charge's own self-energy beyond it. With
/// `grid_only` only the smooth products on the grid interact: no compensation charge, no
/// self-energy tail, no one-center remainder.
pub(in crate::periodic) fn coulomb(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    kernel_of: Kernel,
    first: &Pairs,
    second: &Pairs,
    grid_only: bool,
) -> Result<Integrals> {
    coulomb_rows(
        ctx, setup, cell, kernel_of, first, second, grid_only, true, None, false,
    )
}

/// [`coulomb`] for the left pairs `rows` only (`p * n1 + q`; `None`: every pair), in that order:
/// `values[(row * n2 + r) * n2 + s]`; with `diagonal`, only each row's own pair on the right
/// (`(pq|pq)`, `values[row]`; `first` and `second` the same pairs). Each integral is summed in the
/// same order as in [`coulomb`]: a row of the pair matrix (a Cholesky column) costs one pass over
/// the grid. Without `onsite` the one-center remainder is left out ([`onsite_factors`] holds it).
#[allow(clippy::too_many_arguments)]
pub(in crate::periodic) fn coulomb_rows(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    kernel_of: Kernel,
    first: &Pairs,
    second: &Pairs,
    grid_only: bool,
    onsite: bool,
    rows: Option<&[usize]>,
    diagonal: bool,
) -> Result<Integrals> {
    let values = coulomb_engine(
        ctx,
        setup,
        cell,
        kernel_of,
        first,
        second,
        grid_only,
        onsite,
        rows,
        Right::Pairs { diagonal },
    )?;
    Ok(Integrals {
        first: first.n,
        second: second.n,
        values,
    })
}

/// What the left pairs of the Coulomb engine meet on the right.
#[derive(Clone, Copy)]
enum Right<'a> {
    /// Every right pair, or (`diagonal`) each row's own pair only.
    Pairs { diagonal: bool },
    /// The right pairs contracted with each weight set.
    Contracted(&'a [Vec<C64>]),
}

#[allow(clippy::too_many_arguments)]
fn coulomb_engine(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    kernel_of: Kernel,
    first: &Pairs,
    second: &Pairs,
    grid_only: bool,
    with_onsite: bool,
    rows: Option<&[usize]>,
    right_side: Right<'_>,
) -> Result<Vec<C64>> {
    let (n1, n2) = (first.n, second.n);
    let diagonal = matches!(right_side, Right::Pairs { diagonal: true });
    let contracted = match right_side {
        Right::Contracted(weights) => Some(weights),
        Right::Pairs { .. } => None,
    };
    if diagonal && n1 != n2 {
        return Err(invalid(
            "the Coulomb diagonal pairs one set of pair densities with itself",
        ));
    }
    let all: Vec<usize>;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            all = (0..n1 * n1).collect();
            &all
        }
    };
    if rows.iter().any(|&pq| pq >= n1 * n1) {
        return Err(invalid("a left pair of the Coulomb rows is out of range"));
    }
    let points = cell.points();
    if first.points != points || second.points != points {
        return Err(invalid("the pair densities belong to another supercell"));
    }
    let sphere2 = cell.grid_sphere().powi(2);
    let (a, b) = (n1 * n1, n2 * n2);
    let block = BLOCK.min(points);
    let forms = contracted.map_or(0, |w| w.len());
    let _work = ctx.reserve(product(&[a + b + 2 * forms, block, 16, 3])?)?;
    let width = match contracted {
        Some(_) => forms,
        None if diagonal => 1,
        None => b,
    };
    let targets = |pq: usize| if diagonal { pq..pq + 1 } else { 0..b };
    let _values = ctx.reserve(product(&[rows.len(), width, 16])?)?;
    let mut values = vec![C64::default(); rows.len() * width];
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
            kernel[g] = if plane { 0. } else { kernel_of.value(g2) };
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
        if let Some(weights) = contracted {
            // The right density of each set, and its compensation part, at this block.
            let mut density = vec![C64::default(); forms * count];
            let mut density_compact = vec![C64::default(); forms * count];
            for (f, w) in weights.iter().enumerate() {
                let (d, dc) = (
                    &mut density[f * count..][..count],
                    &mut density_compact[f * count..][..count],
                );
                for (rs, weight) in w.iter().enumerate() {
                    if weight.norm_sqr() == 0. {
                        continue;
                    }
                    let (rr, rc) = (
                        &right[rs * count..][..count],
                        &right_compact[rs * count..][..count],
                    );
                    for g in 0..count {
                        d[g] += weight * rr[g];
                        dc[g] += weight * rc[g];
                    }
                }
            }
            for (row, &pq) in rows.iter().enumerate() {
                let (p, q) = (pq / n1, pq % n1);
                let swapped = q * n1 + p;
                let l = &left[swapped * count..][..count];
                let lc = &left_compact[swapped * count..][..count];
                for f in 0..forms {
                    let (d, dc) = (
                        &density[f * count..][..count],
                        &density_compact[f * count..][..count],
                    );
                    let mut sum = C64::default();
                    for g in 0..count {
                        sum += l[g].conj() * d[g] * kernel[g] - lc[g].conj() * dc[g] * outside[g];
                    }
                    values[row * width + f] += sum;
                }
            }
            continue;
        }
        for (row, &pq) in rows.iter().enumerate() {
            {
                let (p, q) = (pq / n1, pq % n1);
                // The pair (pq) enters as the conjugate of (qp): the left density of (pq|rs).
                let swapped = q * n1 + p;
                let l = &left[swapped * count..][..count];
                let lc = &left_compact[swapped * count..][..count];
                let span = targets(pq);
                let offset = span.start;
                for r in span {
                    let rr = &right[r * count..][..count];
                    let rc = &right_compact[r * count..][..count];
                    let mut sum = C64::default();
                    for g in 0..count {
                        sum += l[g].conj() * rr[g] * kernel[g] - lc[g].conj() * rc[g] * outside[g];
                    }
                    values[row * width + r - offset] += sum;
                }
            }
        }
    }
    let scale = 1. / cell.volume;
    for v in &mut values {
        *v *= scale;
    }
    if grid_only {
        return Ok(values);
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
            if let Some(weights) = contracted {
                let charge: Vec<C64> = weights
                    .iter()
                    .map(|w| w.iter().zip(qb).map(|(x, y)| x * y).sum())
                    .collect();
                for (row, &pq) in rows.iter().enumerate() {
                    let (p, q) = (pq / n1, pq % n1);
                    let left = qa[q * n1 + p].conj();
                    for (f, qw) in charge.iter().enumerate() {
                        values[row * width + f] += tail * left * qw;
                    }
                }
                continue;
            }
            for (row, &pq) in rows.iter().enumerate() {
                let (p, q) = (pq / n1, pq % n1);
                let left = qa[q * n1 + p].conj();
                let span = targets(pq);
                let offset = span.start;
                for j in span {
                    values[row * width + j - offset] += tail * left * qb[j];
                }
            }
        }
    }
    if with_onsite {
        onsite(
            ctx,
            setup,
            cell,
            first,
            second,
            rows,
            right_side,
            &mut values,
        )?;
    }
    Ok(values)
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
    rows: &[usize],
    right_side: Right<'_>,
    values: &mut [C64],
) -> Result<()> {
    let diagonal = matches!(right_side, Right::Pairs { diagonal: true });
    let (n1, n2) = (first.n, second.n);
    let (a, b) = (rows.len(), n2 * n2);
    let p = first.projectors;
    let (atoms, tensors) = onsite_tensors(ctx, setup, p)?;
    for (own, key) in atoms {
        let tensor = &tensors[&key];
        let m = own.len();
        for t in 0..cell.cells {
            // left[pq][c][d] = sum_ab conj(a_p,a) a_q,b DeltaW[abcd]
            let mut left = vec![C64::default(); a * m * m];
            for (row, &pq) in rows.iter().enumerate() {
                {
                    let (pp, qq) = (pq / n1, pq % n1);
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
                            let dst = &mut left[row * m * m..][..m * m];
                            for (slot, d) in dst.iter_mut().zip(&tensor.data[base..base + m * m]) {
                                *slot += w * d;
                            }
                        }
                    }
                }
            }
            let right_of = |r: usize, s: usize| {
                let mut right = vec![C64::default(); m * m];
                for (ic, &jc) in own.iter().enumerate() {
                    let cc = second.coefficients[(r * cell.cells + t) * p + jc].conj();
                    for (id, &jd) in own.iter().enumerate() {
                        right[ic * m + id] =
                            cc * second.coefficients[(s * cell.cells + t) * p + jd];
                    }
                }
                right
            };
            let dot = |row: usize, right: &[C64]| {
                let mut sum = C64::default();
                for (x, y) in left[row * m * m..][..m * m].iter().zip(right) {
                    sum += x * y;
                }
                sum
            };
            if diagonal {
                for (row, &pq) in rows.iter().enumerate() {
                    values[row] += dot(row, &right_of(pq / n2, pq % n2));
                }
                continue;
            }
            if let Right::Contracted(weights) = right_side {
                let forms = weights.len();
                for (f, w) in weights.iter().enumerate() {
                    let mut right = vec![C64::default(); m * m];
                    for (rs, weight) in w.iter().enumerate() {
                        if weight.norm_sqr() == 0. {
                            continue;
                        }
                        for (slot, x) in right.iter_mut().zip(right_of(rs / n2, rs % n2)) {
                            *slot += weight * x;
                        }
                    }
                    for row in 0..a {
                        values[row * forms + f] += dot(row, &right);
                    }
                }
                continue;
            }
            for r in 0..n2 {
                for s in 0..n2 {
                    let right = right_of(r, s);
                    for pq in 0..a {
                        values[pq * b + r * n2 + s] += dot(pq, &right);
                    }
                }
            }
        }
    }
    Ok(())
}

/// The one-center remainder of the contracted Coulomb rows ([`super::streamed::Streamed::contracted`]) for the left pairs `rows` of
/// `pairs` against the weight sets `weights`, added to `values[row * weights.len() + f]`: all the
/// remainder needs of the pairs is their projector coefficients.
pub(in crate::periodic) fn onsite_contracted(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    pairs: &Pairs,
    rows: &[usize],
    weights: &[Vec<C64>],
    values: &mut [C64],
) -> Result<()> {
    onsite(
        ctx,
        setup,
        cell,
        pairs,
        pairs,
        rows,
        Right::Contracted(weights),
        values,
    )
}

/// The projectors (columns of the `p` of the layout) of every atom that has any, with its species,
/// and the one-center remainder `DeltaW` of each species in the complex harmonics of the layout.
#[allow(clippy::type_complexity)]
fn onsite_tensors(
    ctx: &Context,
    setup: &Setup,
    p: usize,
) -> Result<(
    Vec<(Vec<usize>, usize)>,
    std::collections::BTreeMap<usize, crate::paw::onecenter::Tensor>,
)> {
    let owners = setup.projector_atoms();
    let labels = setup.projector_labels();
    let mut cache: std::collections::BTreeMap<usize, crate::paw::onecenter::Tensor> =
        Default::default();
    let mut atoms = Vec::new();
    for atom in 0..setup.sites().atoms.len() {
        let own: Vec<usize> = (0..p).filter(|j| owners[*j] == atom).collect();
        if own.is_empty() {
            continue;
        }
        // Atoms of one species share a dataset and a projector list.
        let key = setup.species()[atom];
        if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(key) {
            let projectors: Vec<OneCenterProjector> = own
                .iter()
                .map(|&j| OneCenterProjector {
                    l: labels[j].0,
                    m: labels[j].1,
                    index: labels[j].2,
                    real: false,
                })
                .collect();
            // The radial quadrature leaves DeltaW short of the symmetries of a Coulomb pair
            // integral by about 1e-5 of itself: (ab|cd) = (cd|ab) and (ab|cd)^* = (ba|dc). The
            // tensor is projected onto them, which keeps the interaction's pair symmetries (and
            // so a Hermitian double counting) exact.
            let raw = onsite_remainder(ctx, setup.dataset(atom), Basis::Raw, &projectors)?;
            entry.insert(coulomb_symmetrized(&raw));
        }
        atoms.push((own, key));
    }
    Ok((atoms, cache))
}

/// The one-center remainder of the Coulomb integrals of the real functions of `pairs` (the part
/// [`coulomb_rows`] leaves out without `onsite`) as signed rank-one terms, `(pq|rs) = sum_k
/// weights[k] factors[k][(p, q)] factors[k][(r, s)]`. On each atom copy the remainder is the
/// bilinear form `x_pq^T M x_rs` of the projector pairs `x_pq[ab] = conj(<p_a|w_p>) <p_b|w_q>`;
/// its real part, the integral of real functions, is `y_pq^T K y_rs` with `y = (Re x, Im x)` and
/// `K = [[Re M, -Im M], [-Im M, -Re M]]` real symmetric, whose eigenvectors give the factors. The
/// remainder is not positive semidefinite (a difference of two Coulomb energies inside the
/// sphere), so it stays out of the pivoted Cholesky of the rest.
pub(in crate::periodic) fn onsite_factors(
    ctx: &Context,
    setup: &Setup,
    cell: &Cell,
    pairs: &Pairs,
) -> Result<(Vec<DMatrix<f64>>, Vec<f64>)> {
    let (n, p) = (pairs.n, pairs.projectors);
    let (atoms, tensors) = onsite_tensors(ctx, setup, p)?;
    let (mut factors, mut weights) = (Vec::new(), Vec::new());
    for (own, key) in atoms {
        let tensor = &tensors[&key];
        let m = own.len();
        let d = m * m;
        let k = DMatrix::from_fn(2 * d, 2 * d, |row, column| {
            let value = tensor.data[(row % d) * d + column % d];
            match (row < d, column < d) {
                (true, true) => value.re,
                (false, false) => -value.re,
                _ => -value.im,
            }
        });
        let eigen = ((&k + k.transpose()) * 0.5).symmetric_eigen();
        let largest = eigen.eigenvalues.iter().fold(0f64, |a, v| a.max(v.abs()));
        let kept: Vec<usize> = (0..2 * d)
            .filter(|&j| eigen.eigenvalues[j].abs() > 1e-14 * largest)
            .collect();
        for t in 0..cell.cells {
            let c = |q: usize, j: usize| pairs.coefficients[(q * cell.cells + t) * p + own[j]];
            // y[(row, p * n + q)]: the real and imaginary parts of the projector pairs.
            let y = DMatrix::from_fn(2 * d, n * n, |row, pq| {
                let (a, b) = ((row % d) / m, (row % d) % m);
                let x = c(pq / n, a).conj() * c(pq % n, b);
                if row < d {
                    x.re
                } else {
                    x.im
                }
            });
            for &j in &kept {
                let g = eigen.eigenvectors.column(j).transpose() * &y;
                let g = DMatrix::from_fn(n, n, |r, s| 0.5 * (g[r * n + s] + g[s * n + r]));
                factors.push(g);
                weights.push(eigen.eigenvalues[j]);
            }
        }
    }
    Ok((factors, weights))
}

/// `1/4 (W[abcd] + W[cdab] + conj W[badc] + conj W[dcba])`: the part of a one-center tensor that
/// has the symmetries of a Coulomb pair integral of complex orbitals.
pub(super) fn coulomb_symmetrized(
    w: &crate::paw::onecenter::Tensor,
) -> crate::paw::onecenter::Tensor {
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
