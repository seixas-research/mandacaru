use super::{growing_rows::GrowingRows, invalid, SpatialGrid};
use crate::basis::Basis;
use crate::math::fft::{spectral_laplacian, Poisson};
use crate::math::linalg::real_eigen;
use crate::math::storage::{with_rows, Precision, RealSamples, Sample, SampleBlocks};
use crate::quantum::C64;
use crate::{product, Buffer, Context, Geometry, Reservation, Result};
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex};

pub fn pair(a: usize, b: usize) -> usize {
    let (a, b) = (a.max(b), a.min(b));
    a * (a + 1) / 2 + b
}

pub(crate) fn packed_len(n: usize) -> Result<usize> {
    n.checked_mul(n.checked_add(1).ok_or(crate::Error::Overflow)?)
        .map(|v| v / 2)
        .ok_or(crate::Error::Overflow)
}
/// Signed separable corrections to a Coulomb pair matrix. Factors are in
/// the same orthonormal spatial basis as the grid samples.
pub struct PairCorrection {
    pub factors: Vec<DMatrix<f64>>,
    pub weights: Vec<f64>,
    pub memory: Reservation,
}
impl PairCorrection {
    pub fn value(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.factors
            .iter()
            .zip(&self.weights)
            .map(|(g, w)| w * g[(a, b)] * g[(c, d)])
            .sum()
    }
}
/// Density-independent additions from a separable pseudopotential setup.
pub struct MolecularCorrections {
    pub external: Buffer<f64>,
    pub overlap: DMatrix<f64>,
    pub potential: DMatrix<f64>,
    pub pairs: Arc<PairCorrection>,
    pub charges: Vec<f64>,
    pub electrons: usize,
    pub dft: Option<PawKohnSham>,
    /// Why Hartree-Fock has no frozen-core exchange for one of the molecule's species (it names
    /// the element); `None` when every core is empty or has its core-valence exchange in `one`.
    pub core_refusal: Option<String>,
    /// Whether every species' core is one a screened hybrid supports (empty or a valid closed
    /// `FrozenCore`).
    pub hybrid_core_supported: bool,
    pub hubbard: Option<crate::molecular::hubbard::MolecularHubbard>,
    pub projections: Option<Vec<DMatrix<f64>>>,
    pub electrostatic: Option<DMatrix<f64>>,
    pub ionic: Option<Vec<f64>>,
}
/// Frozen atomic one-center KS model, separate from the interacting HF model.
pub struct PawKohnSham {
    pub one: DMatrix<f64>,
    pub pairs: PairCorrection,
    pub core: Buffer<f64>,
    pub core_tau: Buffer<f64>,
    pub constant: f64,
    pub relativistic_exchange: bool,
    pub memory: Reservation,
}
pub enum Eri {
    Augmented {
        m: usize,
        base: Arc<Eri>,
        correction: Arc<PairCorrection>,
    },
    Packed {
        m: usize,
        values: Buffer<f64>,
    },
    Cholesky {
        m: usize,
        rank: usize,
        factors: Buffer<f64>,
        /// Maximum unused diagonal when measured by adaptive pivot selection.
        /// Fixed-pivot response does not scan unused Coulomb columns.
        residual: Option<f64>,
        pivots: Buffer<usize>,
    },
    /// The Cholesky factors of a pair matrix the lattice translations of a supercell leave
    /// invariant, stored once per translation orbit of pivots
    /// ([`cholesky_by_translation_orbits`]): factor `(o, t)` is the stored vector `o` carried by
    /// translation `t`, `L_pq = w_o[pair(t^-1 p, t^-1 q)]`, over the distinct translates of the
    /// orbit's pivot (`columns`, in factor order).
    Translated {
        m: usize,
        /// The stored vectors, pair-major `n x stored`.
        stored: usize,
        factors: Buffer<f64>,
        translations: Arc<Translations>,
        /// (stored vector, translation) of every factor.
        columns: Vec<(usize, usize)>,
        /// As for [`Eri::Cholesky`].
        residual: Option<f64>,
        /// The representative pivot of every orbit and how many eigenvalues of its block were
        /// kept: the plan a fixed-pivot factorization replays.
        pivots: Buffer<usize>,
        kept: Vec<usize>,
    },
}
/// The lattice translations of a Born-von Karman supercell as permutations of its orbitals:
/// `orbital[t][p]` is the orbital translation `t` carries `p` to, `inverse[t]` its inverse;
/// translation 0 is the identity.
pub struct Translations {
    pub orbital: Vec<Vec<usize>>,
    pub inverse: Vec<Vec<usize>>,
}
impl Translations {
    pub fn new(orbital: Vec<Vec<usize>>) -> Result<Self> {
        let m = orbital.first().map_or(0, Vec::len);
        let mut inverse = vec![vec![usize::MAX; m]; orbital.len()];
        for (map, back) in orbital.iter().zip(&mut inverse) {
            if map.len() != m {
                return Err(invalid("every translation must permute the same orbitals"));
            }
            for (p, &q) in map.iter().enumerate() {
                if q >= m || back[q] != usize::MAX {
                    return Err(invalid(
                        "a translation must be a permutation of the orbitals",
                    ));
                }
                back[q] = p;
            }
        }
        if orbital
            .first()
            .is_some_and(|t| t.iter().enumerate().any(|(p, &q)| p != q))
        {
            return Err(invalid("translation 0 must be the identity"));
        }
        Ok(Self { orbital, inverse })
    }
    /// The identity alone: the plain pivoted Cholesky.
    fn identity(m: usize) -> Self {
        Self {
            orbital: vec![(0..m).collect()],
            inverse: vec![(0..m).collect()],
        }
    }
}
/// How a two-body factorization is stored, for a log: the representation, the
/// rank of the augmentation correction when there is one, and for a Cholesky
/// base its rank and the largest unused diagonal (`None` when the pivots were
/// fixed and the diagonal was not scanned).
#[derive(Debug, Clone, PartialEq)]
pub struct EriReport {
    pub representation: &'static str,
    pub augmentation_rank: Option<usize>,
    pub cholesky: Option<(usize, Option<f64>)>,
}
impl Eri {
    pub fn report(&self) -> EriReport {
        match self {
            Self::Augmented {
                base, correction, ..
            } => EriReport {
                representation: "cholesky+augmentation",
                augmentation_rank: Some(correction.factors.len()),
                cholesky: base.report().cholesky,
            },
            Self::Packed { .. } => EriReport {
                representation: "packed",
                augmentation_rank: None,
                cholesky: None,
            },
            Self::Cholesky { rank, residual, .. } => EriReport {
                representation: "cholesky",
                augmentation_rank: None,
                cholesky: Some((*rank, *residual)),
            },
            // The rank of the approximation: the directions kept, not the translates of an
            // orbit whose block dropped eigenvalues.
            Self::Translated { kept, residual, .. } => EriReport {
                representation: "cholesky",
                augmentation_rank: None,
                cholesky: Some((kept.iter().sum(), *residual)),
            },
        }
    }
    pub fn pivots(&self) -> Option<&[usize]> {
        match self {
            Self::Cholesky { rank, pivots, .. } => Some(&pivots.as_slice()[..*rank]),
            Self::Translated { stored, pivots, .. } => Some(&pivots.as_slice()[..*stored]),
            Self::Augmented { base, .. } => base.pivots(),
            Self::Packed { .. } => None,
        }
    }
    /// The eigenvalues kept per orbit of a factorization stored per translation orbit.
    pub fn kept(&self) -> Option<&[usize]> {
        match self {
            Self::Translated { kept, .. } => Some(kept),
            Self::Augmented { base, .. } => base.kept(),
            Self::Cholesky { .. } | Self::Packed { .. } => None,
        }
    }
    pub fn m(&self) -> usize {
        match self {
            Self::Packed { m, .. }
            | Self::Cholesky { m, .. }
            | Self::Translated { m, .. }
            | Self::Augmented { m, .. } => *m,
        }
    }
    /// How many Cholesky factors a factored base has (`None` for a packed table or an
    /// augmentation, whose base and correction are read apart).
    pub fn factor_count(&self) -> Option<usize> {
        match self {
            Self::Cholesky { rank, .. } => Some(*rank),
            Self::Translated { columns, .. } => Some(columns.len()),
            Self::Packed { .. } | Self::Augmented { .. } => None,
        }
    }
    /// Cholesky factor `column` of a factored base as the symmetric `m x m` matrix `L_pq`
    /// (`(pq|rs) = sum_P L^P_pq L^P_rs`); empty for any other representation.
    pub fn factor(&self, column: usize) -> DMatrix<f64> {
        match self {
            Self::Cholesky {
                m, rank, factors, ..
            } => DMatrix::from_fn(*m, *m, |p, q| {
                factors.as_slice()[pair(p, q) * rank + column]
            }),
            Self::Translated {
                m,
                stored,
                factors,
                translations,
                columns,
                ..
            } => {
                let (o, t) = columns[column];
                let back = &translations.inverse[t];
                DMatrix::from_fn(*m, *m, |p, q| {
                    factors.as_slice()[pair(back[p], back[q]) * stored + o]
                })
            }
            Self::Packed { .. } | Self::Augmented { .. } => DMatrix::zeros(0, 0),
        }
    }
    pub fn value(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let a = pair(p, r);
        let b = pair(q, s);
        match self {
            Self::Augmented {
                base, correction, ..
            } => base.value(p, q, r, s) + correction.value(p, r, q, s),
            Self::Packed { values, .. } => values.as_slice()[pair(a, b)],
            Self::Cholesky { rank, factors, .. } => (0..*rank)
                .map(|k| factors.as_slice()[a * rank + k] * factors.as_slice()[b * rank + k])
                .sum(),
            Self::Translated {
                stored,
                factors,
                translations,
                columns,
                ..
            } => columns
                .iter()
                .map(|&(o, t)| {
                    let back = &translations.inverse[t];
                    factors.as_slice()[pair(back[p], back[r]) * stored + o]
                        * factors.as_slice()[pair(back[q], back[s]) * stored + o]
                })
                .sum(),
        }
    }
    pub fn dense(&self, ctx: &Context) -> Result<Buffer<f64>> {
        let m = self.m();
        let mut out = ctx.allocate(product(&[m, m, m, m])?)?;
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        out.as_mut_slice()[((p * m + q) * m + r) * m + s] = self.value(p, q, r, s);
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn cholesky(self, ctx: &Context, tolerance: f64) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0. {
            return Err(invalid("Cholesky tolerance must be finite and positive"));
        }
        let m = self.m();
        let n = packed_len(m)?;
        let mut residual = ctx.allocate::<f64>(n)?;
        let mut columns = ctx.allocate::<f64>(product(&[n, n])?)?;
        let indices: Vec<_> = (0..m).flat_map(|p| (0..=p).map(move |q| (p, q))).collect();
        for (a, &(p, q)) in indices.iter().enumerate() {
            residual.as_mut_slice()[a] = self.value(p, p, q, q);
        }
        let mut rank = 0;
        let mut pivots = ctx.allocate::<usize>(n)?;
        loop {
            let pivot = (0..n)
                .max_by(|&a, &b| residual.as_slice()[a].total_cmp(&residual.as_slice()[b]))
                .unwrap();
            let diagonal = residual.as_slice()[pivot];
            if diagonal <= tolerance || rank == n {
                break;
            }
            let (p, r) = indices[pivot];
            pivots.as_mut_slice()[rank] = pivot;
            for (a, &(q, s)) in indices.iter().enumerate() {
                let previous: f64 = (0..rank)
                    .map(|k| columns.as_slice()[a * n + k] * columns.as_slice()[pivot * n + k])
                    .sum();
                let value = (self.value(p, q, r, s) - previous) / diagonal.sqrt();
                columns.as_mut_slice()[a * n + rank] = value;
                residual.as_mut_slice()[a] -= value * value;
            }
            rank += 1;
        }
        let error = residual.as_slice().iter().copied().fold(0f64, f64::max);
        if residual.as_slice().iter().any(|d| *d < -1e-9) {
            return Err(invalid(
                "Coulomb pair matrix is not positive semidefinite at the requested accuracy",
            ));
        }
        let mut factors = ctx.allocate(product(&[n, rank])?)?;
        for a in 0..n {
            for k in 0..rank {
                factors.as_mut_slice()[a * rank + k] = columns.as_slice()[a * n + k];
            }
        }
        Ok(Self::Cholesky {
            m,
            rank,
            factors,
            residual: Some(error),
            pivots,
        })
    }
}
pub struct MolecularIntegrals {
    pub context: Context,
    pub geometry: Geometry,
    pub grid: SpatialGrid,
    pub basis: Basis,
    pub samples: RealSamples,
    pub orthonormal_samples: RealSamples,
    pub overlap: Buffer<f64>,
    pub kinetic: Buffer<f64>,
    pub potential: Buffer<f64>,
    pub lowdin: DMatrix<f64>,
    pub one: Buffer<f64>,
    eri: Mutex<Option<Arc<Eri>>>,
    cholesky_tolerance: f64,
    fixed_pivots: Option<Buffer<usize>>,
    /// The kept eigenvalue counts that go with `fixed_pivots` of a factorization stored per
    /// translation orbit.
    fixed_kept: Option<Vec<usize>>,
    pub nuclear: f64,
    pub kinetic_method: String,
    pub overlap_min: f64,
    pub resolution_ratios: Vec<f64>,
    pub corrections: Option<MolecularCorrections>,
    pub paw_setup: Option<Arc<crate::paw::molecular::Setup>>,
    /// The Born-von Karman supercell this Hamiltonian describes, when the grid is periodic
    /// (`molecular::periodic`); `None` for a molecule.
    pub supercell: Option<Arc<super::periodic::Supercell>>,
    /// The Madelung term `1/2 N_e v_M` of a periodic cell, already included in `nuclear`;
    /// zero for a molecule.
    pub madelung: f64,
    /// The PAW-LCAO supercell this Hamiltonian was built from (`molecular::periodic::paw`);
    /// `None` for every other kind.
    pub paw_supercell: Option<Arc<PawSupercellSource>>,
    _matrices: Reservation,
}
/// The Coulomb pair matrix `(pr|qs)` of `m` real orbitals, read one Cholesky column at a time
/// (pairs packed as [`pair`], `p >= r`): the two-body source a Hamiltonian that has no orbital
/// samples to integrate on its grid factors through [`cholesky_from_columns`].
pub trait PairColumns: Send + Sync {
    fn m(&self) -> usize;
    /// `(pr|pr)` of every packed pair.
    fn diagonal(&self) -> Result<Vec<f64>>;
    /// `(pr|qs)` of the packed pair `pivot = pair(p, r)` against every packed pair `(q, s)`.
    fn column(&self, pivot: usize) -> Result<Vec<f64>>;
    /// The largest imaginary part and the largest departure from the pair symmetry the columns
    /// read so far carried before they were made real and symmetric (Hartree).
    fn departures(&self) -> (f64, f64);
}

/// The pivoted Cholesky of the pair matrix of `columns`: the largest residual diagonal is taken
/// until it falls to `tolerance` (or exactly `fixed_pivots`, in that order, with no scan of the
/// unused diagonal); each column is read once, when it is chosen.
pub fn cholesky_from_columns(
    ctx: &Context,
    columns: &dyn PairColumns,
    tolerance: f64,
    fixed_pivots: Option<&[usize]>,
) -> Result<Eri> {
    let ones = fixed_pivots.map(|p| vec![1; p.len()]);
    let fixed = fixed_pivots.zip(ones.as_deref());
    let orbits = factor_orbits(
        ctx,
        columns,
        &Translations::identity(columns.m()),
        tolerance,
        fixed,
    )?;
    let mut pivots = ctx.allocate::<usize>(orbits.stored.max(1))?;
    pivots.as_mut_slice()[..orbits.stored].copy_from_slice(&orbits.pivots);
    Ok(Eri::Cholesky {
        m: columns.m(),
        rank: orbits.stored,
        factors: orbits.factors,
        residual: orbits.residual,
        pivots,
    })
}

/// The pivoted Cholesky of the pair matrix of `columns` when `translations` leave it invariant
/// (a supercell's frame), stored once per translation orbit ([`Eri::Translated`]): at the
/// largest residual diagonal the whole orbit of its pair is taken as one block, from the one
/// column of the pair (a translate's column is the column carried by the translation). The
/// block `A` of the orbit's residual is symmetric under the translations, so the vectors
/// `C A^-1/2` of its columns `C` are translates of one another and the residual stays invariant;
/// the eigenvalues of `A` at or below `tolerance` are dropped (they leave at most that on the
/// orbit's diagonal). The memory and the columns read fall by the number of cells. `fixed`
/// replays the representative pivots and kept eigenvalue counts of an earlier factorization
/// ([`Eri::pivots`], [`Eri::kept`]) without a scan of the unused diagonal.
pub fn cholesky_by_translation_orbits(
    ctx: &Context,
    columns: &dyn PairColumns,
    translations: &Arc<Translations>,
    tolerance: f64,
    fixed: Option<(&[usize], &[usize])>,
) -> Result<Eri> {
    let orbits = factor_orbits(ctx, columns, translations, tolerance, fixed)?;
    let mut pivots = ctx.allocate::<usize>(orbits.stored.max(1))?;
    pivots.as_mut_slice()[..orbits.stored].copy_from_slice(&orbits.pivots);
    Ok(Eri::Translated {
        m: columns.m(),
        stored: orbits.stored,
        factors: orbits.factors,
        translations: translations.clone(),
        columns: orbits.columns,
        residual: orbits.residual,
        pivots,
        kept: orbits.kept,
    })
}

/// What [`factor_orbits`] gives: the stored vectors (pair-major `n x stored`), the
/// representative pivot and kept eigenvalue count of each orbit, every factor as (stored vector,
/// translation) and the largest unused diagonal (`None` for fixed pivots).
struct Orbits {
    stored: usize,
    factors: Buffer<f64>,
    pivots: Vec<usize>,
    kept: Vec<usize>,
    columns: Vec<(usize, usize)>,
    residual: Option<f64>,
}

fn factor_orbits(
    ctx: &Context,
    columns: &dyn PairColumns,
    translations: &Translations,
    tolerance: f64,
    fixed: Option<(&[usize], &[usize])>,
) -> Result<Orbits> {
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(invalid("Cholesky tolerance must be finite and positive"));
    }
    let m = columns.m();
    let n = packed_len(m)?;
    if translations.orbital.first().map_or(true, |t| t.len() != m) {
        return Err(invalid("the translations do not permute these orbitals"));
    }
    let diagonal = columns.diagonal()?;
    if diagonal.len() != n {
        return Err(invalid("the pair diagonal does not match the orbitals"));
    }
    let mut residual = ctx.allocate::<f64>(n)?;
    residual.as_mut_slice().copy_from_slice(&diagonal);
    if let Some((pivots, kept)) = fixed {
        let mut seen = vec![false; n];
        for &pivot in pivots {
            if pivot >= n || std::mem::replace(&mut seen[pivot], true) {
                return Err(invalid(
                    "fixed Coulomb pivots must be distinct orbital pairs",
                ));
            }
        }
        if kept.len() != pivots.len() {
            return Err(invalid("fixed Coulomb pivots need one kept count each"));
        }
    }
    // The translations on the packed pairs: `forward[t][a]` is pair `a` carried by `t`,
    // `back[t][a]` by its inverse.
    let cells = translations.orbital.len();
    let on_pairs = |maps: &[Vec<usize>]| -> Vec<Vec<usize>> {
        maps.iter()
            .map(|map| {
                (0..m)
                    .flat_map(|p| (0..=p).map(move |r| (p, r)))
                    .map(|(p, r)| pair(map[p], map[r]))
                    .collect()
            })
            .collect()
    };
    let (forward, back) = if cells > 1 {
        (
            on_pairs(&translations.orbital),
            on_pairs(&translations.inverse),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut factors = GrowingRows::new(ctx, n)?;
    let mut used = vec![false; n];
    // Each stored vector's share of the `cells` translates the update sums over (orbit size
    // over cells: a pair its stabilizer fixes is counted once).
    let mut shares = Vec::new();
    let (mut pivots, mut kept, mut factor_columns) = (Vec::new(), Vec::new(), Vec::new());
    let mut previous = ctx.allocate::<f64>(n)?;
    let mut value = ctx.allocate::<f64>(n)?;
    let mut work = ctx.allocate::<f64>(if cells > 1 { n } else { 0 })?;
    loop {
        let index = factors.count();
        let pivot = match fixed {
            Some((fixed, _)) if index == fixed.len() => break,
            Some((fixed, _)) => fixed[index],
            None if used.iter().all(|&u| u) => break,
            None => (0..n)
                .max_by(|&a, &b| residual.as_slice()[a].total_cmp(&residual.as_slice()[b]))
                .unwrap(),
        };
        let top = residual.as_slice()[pivot];
        if fixed.is_none() && (top <= tolerance || used[pivot]) {
            break;
        }
        if top <= 0. {
            return Err(invalid(
                "a Coulomb pivot has no positive residual diagonal: the pair matrix is not \
                 positive definite there",
            ));
        }
        // The orbit: the distinct translates of the pivot, the identity first.
        let mut members = vec![(0, pivot)];
        for t in 1..cells {
            let q = forward[t][pivot];
            if members.iter().all(|&(_, x)| x != q) {
                members.push((t, q));
            }
        }
        if members.iter().any(|&(_, q)| used[q]) {
            return Err(invalid(
                "fixed Coulomb pivots must lie in distinct translation orbits",
            ));
        }
        let column = columns.column(pivot)?;
        if column.len() != n {
            return Err(invalid("a pair column does not match the orbitals"));
        }
        // `previous[a] = sum_P L^P_a L^P_pivot` over every factor so far.
        if cells == 1 {
            let pivot_row = factors.row(pivot).to_vec();
            factors.apply(&pivot_row, previous.as_mut_slice());
        } else {
            previous.as_mut_slice().fill(0.);
            for map in &back {
                let weighted: Vec<f64> = factors
                    .row(map[pivot])
                    .iter()
                    .zip(&shares)
                    .map(|(v, c)| v * c)
                    .collect();
                factors.apply(&weighted, work.as_mut_slice());
                for (slot, &a) in previous.as_mut_slice().iter_mut().zip(map) {
                    *slot += work.as_slice()[a];
                }
            }
        }
        let s = members.len();
        let count = if s == 1 {
            let scale = 1. / top.sqrt();
            for a in 0..n {
                value.as_mut_slice()[a] = (column[a] - previous.as_slice()[a]) * scale;
            }
            1
        } else {
            // The residual column of the pivot, then the orbit's block from it:
            // `R[p_i, t_j p] = R[t_j^-1 p_i, p]`.
            for (slot, c) in previous.as_mut_slice().iter_mut().zip(&column) {
                *slot = c - *slot;
            }
            let r = previous.as_slice();
            let block = DMatrix::from_fn(s, s, |i, j| {
                0.5 * (r[back[members[j].0][members[i].1]] + r[back[members[i].0][members[j].1]])
            });
            let (values, vectors) = real_eigen(block)?;
            let count = match fixed {
                Some((_, kept)) => kept[index],
                None => values.iter().filter(|&&v| v > tolerance).count(),
            };
            if count == 0 || count > s || values[s - count] <= 0. {
                return Err(invalid(
                    "a translation orbit of Coulomb pivots has no positive block there",
                ));
            }
            // Column 0 of `U_k L_k^-1/2 U_k^T`, the representative's weights on the members.
            let weights: Vec<f64> = (0..s)
                .map(|j| {
                    (s - count..s)
                        .map(|l| vectors[(j, l)] * vectors[(0, l)] / values[l].sqrt())
                        .sum()
                })
                .collect();
            for a in 0..n {
                value.as_mut_slice()[a] = members
                    .iter()
                    .zip(&weights)
                    .map(|(&(t, _), w)| w * r[back[t][a]])
                    .sum();
            }
            count
        };
        let v = value.as_slice();
        for (a, slot) in residual.as_mut_slice().iter_mut().enumerate() {
            *slot -= if cells == 1 {
                v[a] * v[a]
            } else {
                members
                    .iter()
                    .map(|&(t, _)| v[back[t][a]] * v[back[t][a]])
                    .sum::<f64>()
            };
        }
        factors.push(v)?;
        shares.push(s as f64 / cells as f64);
        pivots.push(pivot);
        kept.push(count);
        for &(t, q) in &members {
            factor_columns.push((index, t));
            used[q] = true;
        }
    }
    if residual.as_slice().iter().any(|d| *d < -1e-9) {
        return Err(invalid(
            "Coulomb pair matrix is not positive semidefinite at the requested accuracy",
        ));
    }
    let stored = factors.count();
    drop((previous, value, work));
    let packed = if stored == 0 {
        ctx.allocate::<f64>(n)?
    } else {
        factors.into_row_major()?
    };
    let error = residual.as_slice().iter().copied().fold(0f64, f64::max);
    Ok(Orbits {
        stored,
        factors: packed,
        pivots,
        kept,
        columns: factor_columns,
        residual: fixed.is_none().then_some(error),
    })
}

/// What a PAW-LCAO supercell Hamiltonian keeps of its build (`molecular::periodic::paw`): the
/// recipe its geometry continuation rebuilds from, its valence electron count, and the column
/// source of the two-body integrals of its Lowdin frame, which [`MolecularIntegrals::eri`] factors.
pub struct PawSupercellSource {
    pub setup: Arc<crate::periodic::setup::PeriodicSetup>,
    pub mesh: [usize; 3],
    pub electrons: usize,
    pub columns: Box<dyn PairColumns>,
    /// The part of the pair integrals the columns leave out (the one-center remainder, not
    /// positive semidefinite), added to their Cholesky factors as an augmentation.
    pub correction: Arc<PairCorrection>,
    /// The largest departure of `<phi_p|S|phi_q>` (grid and projectors) from the identity.
    pub metric_error: f64,
    /// The supercell's lattice translations on its functions, which leave the pair matrix
    /// invariant: its factors are stored per translation orbit.
    pub translations: Arc<Translations>,
}
/// The lattice-summed electrostatics of a periodic cell, built by `molecular::periodic`.
pub(super) struct PeriodicElectrostatics {
    /// Zero-mean Ewald potential of the ion lattice on the grid (what an electron feels).
    pub external: Buffer<f64>,
    /// Ion-ion Ewald energy of the cell.
    pub ions: f64,
    /// `1/2 N_e v_M` for the neutral cell.
    pub madelung: f64,
    pub supercell: Arc<super::periodic::Supercell>,
}
pub fn pair_integrals<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
) -> Result<Eri> {
    pair_integrals_with_solver(ctx, grid, samples, m, &mut Poisson::isolated(ctx, grid)?)
}
/// Pivoted Cholesky using individual Coulomb columns. Neither the pair matrix
/// nor a four-index tensor is allocated, including during factorization.
pub(super) fn cholesky_from_grid<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
    tolerance: f64,
    solver: &mut Poisson,
    fixed_pivots: Option<&[usize]>,
) -> Result<Eri> {
    if m == 0
        || samples.len() != product(&[m, grid.index.points()])?
        || !tolerance.is_finite()
        || tolerance <= 0.
    {
        return Err(invalid("invalid grid Cholesky dimensions or tolerance"));
    }
    let n = packed_len(m)?;
    if let Some(pivots) = fixed_pivots {
        let _validation_memory = ctx.reserve(n)?;
        let mut seen = vec![false; n];
        for &pivot in pivots {
            if pivot >= n || seen[pivot] {
                return Err(invalid(
                    "fixed Coulomb pivots must be distinct orbital pairs",
                ));
            }
            seen[pivot] = true;
        }
    }
    let g = grid.index.points();
    let mut diagonal = ctx.allocate::<f64>(n)?;
    let mut rho = ctx.allocate::<C64>(g)?;
    let mut potential = ctx.allocate::<C64>(g)?;
    let _index_memory = ctx.reserve(product(&[n, 32])?)?;
    let mut projector = if m >= 16 {
        Some(crate::math::projection::RealProjector::new(
            ctx, samples, m, g,
        )?)
    } else {
        None
    };
    let mut field = if projector.is_some() {
        Some(ctx.allocate::<f64>(g)?)
    } else {
        None
    };
    let indices: Vec<_> = (0..m).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    if fixed_pivots.is_none() {
        for (i, &(a, b)) in indices.iter().enumerate() {
            for k in 0..g {
                rho.as_mut_slice()[k] =
                    C64::new(samples[a * g + k].widen() * samples[b * g + k].widen(), 0.);
            }
            diagonal.as_mut_slice()[i] = solver.quadratic_form(rho.as_slice())?;
        }
    }
    let mut columns: Vec<Buffer<f64>> = Vec::new();
    let mut pivots = ctx.allocate::<usize>(fixed_pivots.map_or(n, |p| p.len()))?;
    loop {
        if columns.len() == fixed_pivots.map_or(n, |p| p.len()) {
            break;
        }
        let pivot = fixed_pivots.map_or_else(
            || {
                (0..n)
                    .max_by(|&a, &b| diagonal.as_slice()[a].total_cmp(&diagonal.as_slice()[b]))
                    .unwrap()
            },
            |p| p[columns.len()],
        );
        let mut maximum = diagonal.as_slice()[pivot];
        if fixed_pivots.is_none() && maximum <= tolerance {
            break;
        }
        pivots.as_mut_slice()[columns.len()] = pivot;
        let (a, b) = indices[pivot];
        for k in 0..g {
            rho.as_mut_slice()[k] =
                C64::new(samples[a * g + k].widen() * samples[b * g + k].widen(), 0.);
        }
        solver.solve_into(rho.as_slice(), potential.as_mut_slice())?;
        if fixed_pivots.is_some() {
            maximum = rho
                .as_slice()
                .iter()
                .zip(potential.as_slice())
                .map(|(r, v)| r.re * v.re)
                .sum::<f64>()
                * grid.index.dv()
                - columns
                    .iter()
                    .map(|c| c.as_slice()[pivot].powi(2))
                    .sum::<f64>();
        }
        if !maximum.is_finite() || maximum <= 0. {
            return Err(invalid(
                "a retained Coulomb pivot lost positive rank at the displaced geometry",
            ));
        }
        let mut column = match ctx.allocate::<f64>(n) {
            Err(crate::Error::Budget { .. })
                if projector
                    .as_mut()
                    .is_some_and(|p| p.release_extra_workspaces()) =>
            {
                ctx.allocate::<f64>(n)?
            }
            allocation => allocation?,
        };
        let projected = if let (Some(projector), Some(field)) = (projector.as_mut(), field.as_mut())
        {
            for (value, potential) in field.as_mut_slice().iter_mut().zip(potential.as_slice()) {
                *value = potential.re;
            }
            Some(projector.project(field.as_slice())?)
        } else {
            None
        };
        for (i, &(p, q)) in indices.iter().enumerate() {
            let integral = projected.map_or_else(
                || {
                    (0..g)
                        .map(|k| {
                            samples[p * g + k].widen()
                                * samples[q * g + k].widen()
                                * potential.as_slice()[k].re
                        })
                        .sum::<f64>()
                },
                |matrix| matrix[p + q * m],
            ) * grid.index.dv();
            let previous = columns
                .iter()
                .map(|c| c.as_slice()[i] * c.as_slice()[pivot])
                .sum::<f64>();
            let value = (integral - previous) / maximum.sqrt();
            column.as_mut_slice()[i] = value;
            if fixed_pivots.is_none() {
                diagonal.as_mut_slice()[i] -= value * value;
            }
        }
        columns.push(column);
    }
    if diagonal
        .as_slice()
        .iter()
        .any(|v| !v.is_finite() || *v < -1e-9)
    {
        return Err(invalid(
            "Coulomb operator is not positive semidefinite at this grid accuracy",
        ));
    }
    let residual = fixed_pivots
        .is_none()
        .then(|| diagonal.as_slice().iter().copied().fold(0f64, f64::max));
    let rank = columns.len();
    // Projection scratch is no longer needed while columns are copied into
    // their compact retained representation.
    drop(projector);
    drop(field);
    drop(rho);
    drop(potential);
    let mut factors = ctx.allocate::<f64>(product(&[n, rank])?)?;
    for i in 0..n {
        for (k, column) in columns.iter().enumerate() {
            factors.as_mut_slice()[i * rank + k] = column.as_slice()[i];
        }
    }
    Ok(Eri::Cholesky {
        m,
        rank,
        factors,
        residual,
        pivots,
    })
}
pub(super) fn pair_integrals_with_solver<T: Sample>(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &[T],
    m: usize,
    poisson: &mut Poisson,
) -> Result<Eri> {
    let g = grid.index.points();
    if samples.len() != product(&[m, g])? || m == 0 {
        return Err(invalid("orbital samples must match the molecular grid"));
    }
    let n = packed_len(m)?;
    let mut values = ctx.allocate(packed_len(n)?)?;
    let mut density = ctx.allocate::<C64>(g)?;
    let mut potential = ctx.allocate::<C64>(g)?;
    for b in 0..m {
        for d in 0..=b {
            let bd = pair(b, d);
            for k in 0..g {
                density.as_mut_slice()[k] =
                    C64::new(samples[b * g + k].widen() * samples[d * g + k].widen(), 0.);
            }
            poisson.solve_into(density.as_slice(), potential.as_mut_slice())?;
            for a in 0..m {
                for c in 0..=a {
                    let ac = pair(a, c);
                    if ac > bd {
                        continue;
                    }
                    let integral: f64 = (0..g)
                        .map(|k| {
                            samples[a * g + k].widen()
                                * samples[c * g + k].widen()
                                * potential.as_slice()[k].re
                        })
                        .sum();
                    values.as_mut_slice()[pair(ac, bd)] = integral * grid.index.dv();
                }
            }
        }
    }
    Ok(Eri::Packed { m, values })
}
pub(super) fn laplacian_of(
    ctx: &Context,
    grid: &SpatialGrid,
    kinetic: &str,
    field: &[f64],
) -> Result<Buffer<f64>> {
    if kinetic == "spectral" {
        spectral_laplacian(ctx, grid, field)
    } else {
        crate::kernels::laplacian(ctx, &grid.index, 1, field)
    }
}
/// Kinetic matrix `t[a * m + b] = -1/2 <psi_a, laplacian psi_b>` of exact
/// float64 orbitals for a float32 stack, whose stored rows would put a
/// rounding error of 1e-7 relative on each side. Laplacian rows are resident
/// one third of the orbitals at a time (about 2.7 stack-lengths of float64 per
/// third), and the other side is re-evaluated in point blocks. Pairs with the
/// row group at or below the Laplacian group are computed once and mirrored.
fn exact_kinetic(
    ctx: &Context,
    basis: &Basis,
    grid: &SpatialGrid,
    kinetic: &str,
    t: &mut [f64],
) -> Result<()> {
    let m = basis.orbitals.len();
    let g = grid.index.points();
    let group = m.div_ceil(3).max(1);
    let stride = g.min(SampleBlocks::BLOCK);
    let mut laplacians = ctx.allocate::<f64>(product(&[group, g])?)?;
    let mut row = ctx.allocate::<f64>(g)?;
    let mut block = ctx.allocate::<f64>(product(&[group, stride])?)?;
    let mut raw = vec![0.; group];
    for b0 in (0..m).step_by(group) {
        let nb = group.min(m - b0);
        for j in 0..nb {
            for (k, value) in row.as_mut_slice().iter_mut().enumerate() {
                *value = basis.orbitals[b0 + j].evaluate(grid.point(k));
            }
            let lap = laplacian_of(ctx, grid, kinetic, row.as_slice())?;
            laplacians.as_mut_slice()[j * g..(j + 1) * g].copy_from_slice(lap.as_slice());
        }
        for a0 in (0..=b0).step_by(group) {
            let na = group.min(m - a0);
            for start in (0..g).step_by(stride) {
                let count = stride.min(g - start);
                for i in 0..count {
                    basis.evaluate_at(a0, grid.point(start + i), &mut raw[..na]);
                    for (a, value) in raw[..na].iter().enumerate() {
                        block.as_mut_slice()[a * stride + i] = *value;
                    }
                }
                for a in 0..na {
                    let left = &block.as_slice()[a * stride..a * stride + count];
                    for j in 0..nb {
                        let right = &laplacians.as_slice()[j * g + start..j * g + start + count];
                        let mut sum = t[(a0 + a) * m + b0 + j];
                        for i in 0..count {
                            sum += left[i] * right[i];
                        }
                        t[(a0 + a) * m + b0 + j] = sum;
                    }
                }
            }
        }
    }
    let scale = -0.5 * grid.index.dv();
    for (a, b) in (0..m).flat_map(|a| (0..m).map(move |b| (a, b))) {
        if a / group <= b / group {
            t[a * m + b] *= scale;
        }
    }
    // Mirror the pairs that were evaluated with the left orbital in an
    // earlier group than the Laplacian row.
    for a in 0..m {
        for b in 0..m {
            if a / group < b / group {
                t[b * m + a] = t[a * m + b];
            }
        }
    }
    Ok(())
}
/// The exact float64 values of all orbitals on a block of grid points. A
/// float64 stack IS the exact value, so its block is widened from storage; a
/// float32 stack has already been rounded, so the block is re-evaluated from
/// the orbitals. Everything that must not inherit rounding noise (overlap,
/// potential, the orthonormal rows) reads this, never the stored float32 rows.
struct ExactBlocks<'a> {
    basis: &'a Basis,
    grid: &'a SpatialGrid,
    stride: usize,
    points: usize,
    raw: Vec<f64>,
    source: ExactSource<'a>,
}
enum ExactSource<'a> {
    Stored(SampleBlocks<'a>),
    Evaluated(Buffer<f64>),
}
impl<'a> ExactBlocks<'a> {
    fn new(
        ctx: &Context,
        basis: &'a Basis,
        grid: &'a SpatialGrid,
        samples: &'a RealSamples,
    ) -> Result<Self> {
        let (m, points) = (basis.orbitals.len(), grid.index.points());
        let stride = points.min(SampleBlocks::BLOCK);
        let source = match samples.precision() {
            Precision::Float64 => ExactSource::Stored(SampleBlocks::new(ctx, samples, m, points)?),
            Precision::Float32 => ExactSource::Evaluated(ctx.allocate(product(&[m, stride])?)?),
        };
        Ok(Self {
            basis,
            grid,
            stride,
            points,
            raw: vec![0.; m],
            source,
        })
    }
    /// Orbital `a` of the loaded block is at `a * stride`; returns the point
    /// count and all values.
    fn load(&mut self, start: usize) -> Result<(usize, &[f64])> {
        let count = self.stride.min(self.points - start);
        match &mut self.source {
            ExactSource::Stored(blocks) => {
                blocks.load(start)?;
                Ok((count, blocks.values()))
            }
            ExactSource::Evaluated(block) => {
                for i in 0..count {
                    self.basis
                        .evaluate_at(0, self.grid.point(start + i), &mut self.raw);
                    for (a, value) in self.raw.iter().enumerate() {
                        block.as_mut_slice()[a * self.stride + i] = *value;
                    }
                }
                Ok((count, block.as_slice()))
            }
        }
    }
}
impl MolecularIntegrals {
    /// Rebuild on the same fixed integration grid with the same basis recipe
    /// and PAW species. Numerical setup objects stay owned in Rust.
    pub fn at_geometry(&self, geometry: Geometry) -> Result<Self> {
        if geometry.numbers() != self.geometry.numbers() || geometry.pbc() != self.geometry.pbc() {
            return Err(invalid(
                "geometry continuation needs the same atomic composition and boundary conditions",
            ));
        }
        self.context.run(|| {
            if let Some(source) = &self.paw_supercell {
                return super::periodic::paw::continued(self, source, geometry);
            }
            if let Some(setup) = &self.paw_setup {
                return setup.at_geometry(
                    &self.context,
                    geometry,
                    self.grid.clone(),
                    &self.kinetic_method,
                    self.cholesky_tolerance,
                    self.precision(),
                );
            }
            if self.corrections.is_some() {
                return Err(invalid(
                    "custom molecular corrections have no geometry-dependent setup",
                ));
            }
            let mut orbitals = self.basis.orbitals.clone();
            for orbital in &mut orbitals {
                orbital.center = geometry.positions()[orbital.atom];
            }
            let basis = Basis::from_orbitals(&self.context, orbitals)?;
            if let Some(supercell) = &self.supercell {
                return super::periodic::build(
                    &self.context,
                    supercell.clone(),
                    geometry,
                    self.grid.clone(),
                    basis,
                    Some(self.cholesky_tolerance),
                );
            }
            let samples = basis.sample(&self.context, &self.grid, self.precision())?;
            Self::from_basis(
                &self.context,
                geometry,
                self.grid.clone(),
                basis,
                samples,
                &self.kinetic_method,
                Some(self.cholesky_tolerance),
                None,
            )
        })
    }
    pub fn has_eri(&self) -> bool {
        self.eri.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    /// Keep the same low-rank Coulomb approximation while differentiating a
    /// geometry. Fresh calculations continue to select their own pivots.
    pub fn at_geometry_with_fixed_pivots(&self, geometry: Geometry) -> Result<Self> {
        let source = self.eri()?;
        let plan = source
            .pivots()
            .ok_or_else(|| invalid("geometry response needs a Coulomb factorization"))?;
        let mut rebuilt = self.at_geometry(geometry)?;
        let mut pivots = self.context.allocate::<usize>(plan.len())?;
        pivots.as_mut_slice().copy_from_slice(plan);
        rebuilt.fixed_pivots = Some(pivots);
        rebuilt.fixed_kept = source.kept().map(<[usize]>::to_vec);
        Ok(rebuilt)
    }
    /// Refuse a Hartree-Fock Hamiltonian when a PAW species' frozen core has no exchange
    /// operator with the valence (owner Q20, 2026-11-12: the guard sits at the Hartree-Fock
    /// entry points, not on the Coulomb integrals, which Kohn-Sham runs on every element use
    /// without that term). The entry points are every route that builds the second-quantized
    /// Hamiltonian over the frozen core, whatever the reference's method: an HF mean field and
    /// its continuation, TDHF and CIS, real-time TDHF, the model and active-space Hamiltonians
    /// (and the MP2 natural orbitals and correlating partners an active space selects with),
    /// local MP2 and the analytic variational force.
    pub fn require_hartree_fock_core(&self) -> Result<()> {
        if let Some(reason) = self
            .corrections
            .as_ref()
            .and_then(|c| c.core_refusal.as_ref())
        {
            return Err(invalid(format!(
                "native PAW frozen-core exchange is not available: {reason}. Hartree-Fock of this \
                 molecule, its excitations and every Hamiltonian over its frozen core (the model \
                 or an active space, MP2, a quantum method) are refused; Kohn-Sham DFT runs on \
                 every element"
            )));
        }
        Ok(())
    }
    /// Build Coulomb factors only when a solver or Hamiltonian asks for them.
    /// Failed allocations do not poison the cache and can be retried.
    pub fn eri(&self) -> Result<Arc<Eri>> {
        let mut cache = self.eri.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = cache.as_ref() {
            return Ok(value.clone());
        }
        if let Some(source) = &self.paw_supercell {
            let fixed =
                match (&self.fixed_pivots, &self.fixed_kept) {
                    (Some(pivots), Some(kept)) => Some((pivots.as_slice(), kept.as_slice())),
                    (None, _) => None,
                    (Some(_), None) => return Err(invalid(
                        "fixed Coulomb pivots of a supercell need the kept counts of their orbits",
                    )),
                };
            let base = Arc::new(cholesky_by_translation_orbits(
                &self.context,
                source.columns.as_ref(),
                &source.translations,
                self.cholesky_tolerance,
                fixed,
            )?);
            let factors = Arc::new(Eri::Augmented {
                m: self.m(),
                base,
                correction: source.correction.clone(),
            });
            *cache = Some(factors.clone());
            return Ok(factors);
        }
        let mut solver = if self.grid.periodic {
            Poisson::periodic(&self.context, &self.grid)?
        } else {
            Poisson::isolated(&self.context, &self.grid)?
        };
        let base = Arc::new(with_rows!(self.orthonormal_samples.rows(), |samples| {
            cholesky_from_grid(
                &self.context,
                &self.grid,
                samples,
                self.m(),
                self.cholesky_tolerance,
                &mut solver,
                self.fixed_pivots.as_ref().map(|p| p.as_slice()),
            )?
        }));
        let factors = if let Some(correction) = self.corrections.as_ref() {
            Arc::new(Eri::Augmented {
                m: self.m(),
                base,
                correction: correction.pairs.clone(),
            })
        } else {
            base
        };
        *cache = Some(factors.clone());
        Ok(factors)
    }
    /// [`Self::particles`] from the initial magnetic moment of each atom (their sum is the
    /// total moment).
    pub fn particles_from_moments(
        &self,
        charge: i32,
        moments: &[f64],
        electrons: Option<usize>,
    ) -> Result<[usize; 2]> {
        if moments.iter().any(|m| !m.is_finite()) {
            return Err(invalid("initial magnetic moments must be finite"));
        }
        self.particles(charge, moments.iter().sum(), electrons)
    }
    pub fn particles(
        &self,
        charge: i32,
        moment: f64,
        electrons: Option<usize>,
    ) -> Result<[usize; 2]> {
        if !moment.is_finite() {
            return Err(invalid("initial magnetic moment must be finite"));
        }
        let neutral = self.neutral_electrons() as i128;
        let total = electrons
            .map(|n| n as i128)
            .unwrap_or_else(|| neutral - charge as i128);
        if self.supercell.is_some() {
            if total != neutral {
                return Err(invalid(
                    "a periodic cell must be neutral: the periodic Coulomb kernel drops G = 0 against a uniform background, and a charged supercell needs a background correction that is not implemented",
                ));
            }
        }
        if total < 0 || total > 2 * self.m() as i128 {
            return Err(invalid("electron count does not fit the spatial basis"));
        }
        let n = total as usize;
        let unpaired = if moment.abs() > 1e-8 {
            moment.abs().round_ties_even() as usize
        } else {
            n % 2
        };
        if unpaired > n || (n - unpaired) % 2 != 0 {
            return Err(invalid(
                "electron count and initial magnetic moment have inconsistent parity",
            ));
        }
        let particles = [(n + unpaired) / 2, (n - unpaired) / 2];
        if particles.iter().any(|p| *p > self.m()) {
            return Err(invalid("spin occupation does not fit the spatial basis"));
        }
        Ok(particles)
    }
    /// The electrons of the neutral system the Hamiltonian describes: the valence electrons of
    /// a PAW one (molecule or supercell), every electron of an all-electron one.
    pub fn neutral_electrons(&self) -> usize {
        if let Some(source) = &self.paw_supercell {
            return source.electrons;
        }
        self.corrections.as_ref().map_or_else(
            || self.geometry.numbers().iter().map(|&z| z as usize).sum(),
            |c| c.electrons,
        )
    }
    /// `v_M` (Hartree): the exchange's dropped `G = 0` term of a periodic Hamiltonian (an HAO or
    /// PAW-LCAO supercell), the Madelung constant of its own lattice; zero for a molecule. The
    /// supercell's occupied levels lie `-v_M` above the isolated system's (the virtual ones do
    /// not move), so MP-type denominators add `v_M` to the occupied levels
    /// ([`super::correlation::natural_orbitals`]); the stored levels, the Fock and every orbital
    /// Hessian (A + B, CPHF, z-vector) keep it, where it cancels against the `+v_M` of `(ab|ij)`.
    pub fn exchange_madelung(&self) -> Result<f64> {
        use super::periodic::paw::HamiltonianKind;
        match self.kind() {
            HamiltonianKind::Hao | HamiltonianKind::PawSupercell => {
                super::periodic::madelung_constant(&self.context, self.geometry.cell())
            }
            HamiltonianKind::PawMolecule | HamiltonianKind::AllElectronMolecule => Ok(0.),
        }
    }
    /// Where this Hamiltonian comes from: the one seam the routes that differ by kind dispatch on.
    pub fn kind(&self) -> super::periodic::paw::HamiltonianKind {
        use super::periodic::paw::HamiltonianKind;
        match (&self.paw_supercell, &self.supercell, &self.paw_setup) {
            (Some(_), _, _) => HamiltonianKind::PawSupercell,
            (None, Some(_), _) => HamiltonianKind::Hao,
            (None, None, Some(_)) => HamiltonianKind::PawMolecule,
            (None, None, None) => HamiltonianKind::AllElectronMolecule,
        }
    }
    /// The PAW-LCAO supercell Hamiltonian from its assembled parts (`molecular::periodic::paw`):
    /// the raw overlap `S`, the raw bare one-body operator split as `fixed` (kinetic, short-range
    /// local and one-center: stored as `kinetic`) and the field of the Gaussian ions (stored as
    /// `potential`), the Lowdin frame, the smooth samples of the raw and orthonormal functions on
    /// the supercell grid, and the constants.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_paw_supercell(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis: Basis,
        (samples, orthonormal_samples): (RealSamples, RealSamples),
        (overlap, fixed, field): (&DMatrix<f64>, &DMatrix<f64>, &DMatrix<f64>),
        (lowdin, overlap_min): (DMatrix<f64>, f64),
        (ions, madelung): (f64, f64),
        cholesky: Option<f64>,
        supercell: Arc<super::periodic::Supercell>,
        source: PawSupercellSource,
    ) -> Result<Self> {
        let m = lowdin.nrows();
        if basis.orbitals.len() != m || overlap.shape() != (m, m) || fixed.shape() != (m, m) {
            return Err(invalid("the PAW supercell parts disagree with its basis"));
        }
        let matrices = ctx.reserve(product(&[m, m, 8, 12])?)?;
        let buffer = |matrix: &DMatrix<f64>| -> Result<Buffer<f64>> {
            let mut out = ctx.allocate::<f64>(m * m)?;
            for a in 0..m {
                for b in 0..m {
                    out.as_mut_slice()[a * m + b] = matrix[(a, b)];
                }
            }
            Ok(out)
        };
        let h = lowdin.transpose() * (fixed + field) * &lowdin;
        let h = (&h + h.transpose()) * 0.5;
        let cholesky_tolerance = cholesky.unwrap_or(1e-10);
        if !cholesky_tolerance.is_finite() || cholesky_tolerance <= 0. {
            return Err(invalid("Cholesky tolerance must be finite and positive"));
        }
        Ok(Self {
            context: ctx.clone(),
            geometry,
            grid,
            basis,
            samples,
            orthonormal_samples,
            overlap: buffer(overlap)?,
            kinetic: buffer(fixed)?,
            potential: buffer(field)?,
            lowdin,
            one: buffer(&h)?,
            eri: Mutex::new(None),
            cholesky_tolerance,
            fixed_pivots: None,
            fixed_kept: None,
            nuclear: ions + madelung,
            kinetic_method: "spectral".into(),
            overlap_min,
            resolution_ratios: Vec::new(),
            corrections: None,
            paw_setup: None,
            supercell: Some(supercell),
            madelung,
            paw_supercell: Some(Arc::new(source)),
            _matrices: matrices,
        })
    }
    /// The Cholesky tolerance this Hamiltonian factors its integrals with.
    pub fn cholesky_tolerance(&self) -> f64 {
        self.cholesky_tolerance
    }
    /// The precision is fixed at construction: the orbital stacks are built
    /// directly in it and never exist in float64 beside it.
    pub fn new(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis_spec: &str,
        kinetic: &str,
        cholesky: Option<f64>,
        precision: Precision,
    ) -> Result<Self> {
        let basis = Basis::build(ctx, &geometry, basis_spec)?;
        let samples = basis.sample(ctx, &grid, precision)?;
        Self::from_basis(ctx, geometry, grid, basis, samples, kinetic, cholesky, None)
    }
    /// Integrals from a basis and its raw orbital samples on `grid`; the
    /// samples' precision is the calculation's precision.
    #[allow(clippy::too_many_arguments)]
    pub fn from_basis(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis: Basis,
        samples: RealSamples,
        kinetic: &str,
        cholesky: Option<f64>,
        corrections: Option<MolecularCorrections>,
    ) -> Result<Self> {
        if grid.periodic {
            return Err(invalid("molecular integrals require an isolated grid; a periodic cell is built by molecular::periodic (the Born-von Karman supercell)"));
        }
        Self::assemble(
            ctx,
            geometry,
            grid,
            basis,
            samples,
            kinetic,
            cholesky,
            corrections,
            None,
        )
    }
    /// The shared assembly: isolated electrostatics when `periodic` is `None`, the
    /// lattice sums of `molecular::periodic` otherwise (then `samples` are image-summed).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble(
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        basis: Basis,
        samples: RealSamples,
        kinetic: &str,
        cholesky: Option<f64>,
        mut corrections: Option<MolecularCorrections>,
        periodic: Option<PeriodicElectrostatics>,
    ) -> Result<Self> {
        if grid.periodic != periodic.is_some() {
            return Err(invalid("a periodic grid needs lattice-summed electrostatics, and only a periodic grid has them"));
        }
        if periodic.is_some() && (corrections.is_some() || kinetic != "spectral") {
            return Err(invalid("the periodic supercell Hamiltonian takes an all-electron basis and the spectral kinetic operator"));
        }
        if kinetic != "spectral" && kinetic != "fd" {
            return Err(invalid("kinetic must be spectral or fd"));
        }
        let m = basis.orbitals.len();
        let g = grid.index.points();
        if let Some(c) = &corrections {
            if c.external.as_slice().len() != g
                || c.overlap.shape() != (m, m)
                || c.potential.shape() != (m, m)
                || c.pairs.weights.len() != c.pairs.factors.len()
                || c.pairs.factors.iter().any(|f| f.shape() != (m, m))
                || c.charges.len() != geometry.numbers().len()
            {
                return Err(invalid(
                    "molecular correction dimensions disagree with the basis/grid",
                ));
            }
            crate::finite(c.external.as_slice(), "external potential")?;
            crate::finite(c.overlap.as_slice(), "overlap correction")?;
            crate::finite(c.potential.as_slice(), "potential correction")?;
            crate::finite(&c.charges, "ionic charges")?;
            crate::finite(&c.pairs.weights, "Coulomb correction weights")?;
            for factor in &c.pairs.factors {
                crate::finite(factor.as_slice(), "Coulomb correction factors")?;
            }
        }
        let matrices = ctx.reserve(product(&[m, m, 8, 12])?)?;
        if samples.len() != product(&[m, g])? {
            return Err(invalid("orbital samples must match the basis and grid"));
        }
        let precision = samples.precision();
        let mut overlap = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut t = ctx.allocate::<f64>(product(&[m, m])?)?;
        let mut v = ctx.allocate::<f64>(product(&[m, m])?)?;
        if periodic
            .as_ref()
            .is_some_and(|p| p.external.as_slice().len() != g)
        {
            return Err(invalid(
                "the periodic external potential does not match the grid",
            ));
        }
        let mut external = ctx.allocate::<f64>(g)?;
        let softening = 0.5 * grid.spacings().into_iter().fold(f64::INFINITY, f64::min);
        for k in 0..g {
            if let Some(c) = &corrections {
                external.as_mut_slice()[k] = c.external.as_slice()[k];
                continue;
            }
            if let Some(p) = &periodic {
                external.as_mut_slice()[k] = p.external.as_slice()[k];
                continue;
            }
            let point = grid.point(k);
            external.as_mut_slice()[k] = geometry
                .numbers()
                .iter()
                .zip(geometry.positions())
                .map(|(&z, r)| {
                    let distance = (0..3)
                        .map(|j| (point[j] - r[j]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        .max(softening);
                    -(z as f64) / distance
                })
                .sum();
        }
        // Matrix elements are contracted from the EXACT float64 orbital values,
        // not from rounded stored rows: the nuclear potential weights a few
        // points so heavily that rounding noise in stored values would move the
        // spectrum by 1e-6 Ha. A float64 stack IS the exact value; a float32
        // stack re-evaluates the orbitals, so the stack is only ever read to
        // build the orthonormal rows. Overlap and potential run over point
        // blocks (all orbitals of a block at once), the kinetic matrix over
        // rows (the Laplacian is a whole-row operation).
        let mut exact = ExactBlocks::new(ctx, &basis, &grid, &samples)?;
        let stride = exact.stride;
        for start in (0..g).step_by(stride) {
            let (count, phi) = exact.load(start)?;
            let ext = &external.as_slice()[start..start + count];
            for a in 0..m {
                let left = &phi[a * stride..a * stride + count];
                for b in a..m {
                    let right = &phi[b * stride..b * stride + count];
                    let mut s = overlap.as_slice()[a * m + b];
                    let mut potential = v.as_slice()[a * m + b];
                    for i in 0..count {
                        s += left[i] * right[i];
                        potential += left[i] * right[i] * ext[i];
                    }
                    overlap.as_mut_slice()[a * m + b] = s;
                    v.as_mut_slice()[a * m + b] = potential;
                }
            }
        }
        match precision {
            // A float64 stack is exact: one resident Laplacian row at a time.
            Precision::Float64 => {
                let mut row = ctx.allocate::<f64>(g)?;
                for b in 0..m {
                    samples.widen_range(b * g, row.as_mut_slice())?;
                    let lap = laplacian_of(ctx, &grid, kinetic, row.as_slice())?;
                    with_rows!(samples.rows(), |stored| {
                        for a in 0..m {
                            let mut sum = 0.;
                            for (k, &psi) in stored[a * g..(a + 1) * g].iter().enumerate() {
                                sum += psi.widen() * lap.as_slice()[k];
                            }
                            t.as_mut_slice()[a * m + b] = -0.5 * sum * grid.index.dv();
                        }
                    });
                }
            }
            Precision::Float32 => exact_kinetic(ctx, &basis, &grid, kinetic, t.as_mut_slice())?,
        }
        for a in 0..m {
            for b in a..m {
                let s = overlap.as_slice()[a * m + b] * grid.index.dv();
                let potential = v.as_slice()[a * m + b] * grid.index.dv();
                overlap.as_mut_slice()[a * m + b] = s;
                overlap.as_mut_slice()[b * m + a] = s;
                v.as_mut_slice()[a * m + b] = potential;
                v.as_mut_slice()[b * m + a] = potential;
            }
        }
        if let Some(c) = &corrections {
            for a in 0..m {
                for b in 0..m {
                    overlap.as_mut_slice()[a * m + b] += c.overlap[(a, b)];
                    v.as_mut_slice()[a * m + b] += c.potential[(a, b)];
                }
            }
        }
        let (values, vectors) = real_eigen(DMatrix::from_row_slice(m, m, overlap.as_slice()))?;
        let overlap_min = values[0];
        if overlap_min <= 1e-10 {
            return Err(invalid(format!(
                "basis is linearly dependent on this grid (overlap eigenvalue {overlap_min:e})"
            )));
        }
        let lowdin = &vectors
            * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                m,
                values.iter().map(|s| 1. / s.sqrt()),
            ))
            * vectors.transpose();
        if let Some(c) = corrections.as_mut() {
            if let Some(dft) = c.dft.as_mut() {
                dft.one = lowdin.transpose() * &dft.one * &lowdin;
                for factor in &mut dft.pairs.factors {
                    *factor = lowdin.transpose() * &*factor * &lowdin;
                }
            }
            for factor in &mut Arc::get_mut(&mut c.pairs)
                .ok_or_else(|| invalid("setup Coulomb corrections require unique ownership"))?
                .factors
            {
                *factor = lowdin.transpose() * &*factor * &lowdin;
            }
            if let Some(hubbard) = c.hubbard.as_mut() {
                for shell in &mut hubbard.shells {
                    shell.q = lowdin.transpose() * &shell.q_ao;
                }
            }
        }
        let h = DMatrix::from_fn(m, m, |a, b| {
            0.5 * (t.as_slice()[a * m + b]
                + v.as_slice()[a * m + b]
                + t.as_slice()[b * m + a]
                + v.as_slice()[b * m + a])
        });
        let h = lowdin.transpose() * h * &lowdin;
        let mut one = ctx.allocate(m * m)?;
        for a in 0..m {
            for b in 0..m {
                one.as_mut_slice()[a * m + b] = h[(a, b)];
            }
        }
        // Orthonormal rows by point blocks: widen a raw block to float64,
        // multiply by the Lowdin matrix there and round the block once into
        // the second stack. Only O(m B) float64 scratch exists.
        let mut orthonormal_samples = RealSamples::zeros(ctx, precision, product(&[m, g])?)?;
        {
            let stride = exact.stride;
            let mut block = ctx.allocate::<f64>(product(&[m, stride])?)?;
            let _packing = if m >= 16 {
                Some(ctx.reserve(product(&[m, 256, 16])?)?)
            } else {
                None
            };
            for start in (0..g).step_by(stride) {
                let (count, raw) = exact.load(start)?;
                if m >= 16 {
                    let input = nalgebra::DMatrixView::from_slice(raw, stride, m);
                    let mut output =
                        nalgebra::DMatrixViewMut::from_slice(block.as_mut_slice(), stride, m);
                    output.gemm(1., &input, &lowdin, 0.);
                } else {
                    block.as_mut_slice().fill(0.);
                    for p in 0..m {
                        for a in 0..m {
                            for k in 0..count {
                                block.as_mut_slice()[p * stride + k] +=
                                    lowdin[(a, p)] * raw[a * stride + k];
                            }
                        }
                    }
                }
                for p in 0..m {
                    orthonormal_samples.fill_range(
                        p * g + start,
                        &block.as_slice()[p * stride..p * stride + count],
                    )?;
                }
            }
        }
        drop(exact);
        let cholesky_tolerance = cholesky.unwrap_or(1e-10);
        if !cholesky_tolerance.is_finite() || cholesky_tolerance <= 0. {
            return Err(invalid("Cholesky tolerance must be finite and positive"));
        }
        let mut nuclear = 0.;
        let isolated = periodic.is_none();
        for a in 0..geometry.numbers().len() {
            for b in 0..a {
                if !isolated {
                    break;
                }
                let distance = (0..3)
                    .map(|j| (geometry.positions()[a][j] - geometry.positions()[b][j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if distance < 1e-12 {
                    return Err(invalid("nuclei must not coincide"));
                }
                let charges = |i| {
                    corrections
                        .as_ref()
                        .map_or(geometry.numbers()[i] as f64, |c| c.charges[i])
                };
                nuclear += charges(a) * charges(b) / distance;
            }
        }
        let (madelung, supercell) = match periodic {
            Some(p) => {
                nuclear = p.ions + p.madelung;
                (p.madelung, Some(p.supercell))
            }
            None => (0., None),
        };
        let resolution_ratios = basis
            .orbitals
            .iter()
            .enumerate()
            .map(|(a, o)| {
                t.as_slice()[a * m + a] / overlap.as_slice()[a * m + a] / o.exact_kinetic()
            })
            .collect();
        Ok(Self {
            context: ctx.clone(),
            geometry,
            grid,
            basis,
            samples,
            orthonormal_samples,
            overlap,
            kinetic: t,
            potential: v,
            lowdin,
            one,
            eri: Mutex::new(None),
            cholesky_tolerance,
            fixed_pivots: None,
            fixed_kept: None,
            nuclear,
            kinetic_method: kinetic.into(),
            overlap_min,
            resolution_ratios,
            corrections,
            paw_setup: None,
            supercell,
            madelung,
            paw_supercell: None,
            _matrices: matrices,
        })
    }
    /// Precision of the stored orbital stacks.
    pub fn precision(&self) -> Precision {
        self.samples.precision()
    }
    pub fn m(&self) -> usize {
        self.basis.orbitals.len()
    }
}
