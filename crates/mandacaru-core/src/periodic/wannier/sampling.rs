//! Full-mesh Bloch frames, PAW neighbor links and localized trial projections.
//! A frame owns its band samples; large samples are archived by the context.
use super::{invalid, Matrix, Mesh};
use crate::periodic::bloch_mesh::MeshIndex;
use crate::{
    math::{
        angular,
        storage::{ComplexSamples, Precision},
        SpatialGrid,
    },
    paw::multipole,
    periodic::{
        setup::PeriodicSetup, solver::diagonalize_with_exchange, Crystal, KPoint, PeriodicResult,
    },
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

pub struct Neighbors {
    pub vectors: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub steps: Vec<[i32; 3]>,
}

pub fn lattice(grid: &SpatialGrid) -> Matrix3<f64> {
    grid.step * Matrix3::from_diagonal(&Vector3::from(grid.index.shape().map(|n| n as f64)))
}

/// The finite-difference neighbors of a mesh (Marzari and Vanderbilt): shells of equal-length
/// mesh steps, taken in order of length, each accepted only when it is not parallel to an
/// accepted shell and raises the rank of the second-moment equations, until one positive weight
/// per shell makes `sum_b w_b b_a b_b = delta_ab` exactly. Parallel or redundant shells would put
/// weight on longer steps and coarsen the finite differences.
pub fn neighbor_shells(cell: Matrix3<f64>, size: [usize; 3]) -> Result<Neighbors> {
    if size.contains(&0) || cell.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid Wannier cell or mesh size"));
    }
    let reciprocal = 2.
        * PI
        * cell
            .try_inverse()
            .ok_or_else(|| invalid("singular Wannier cell"))?
            .transpose();
    let mesh = reciprocal * Matrix3::from_diagonal(&Vector3::from(size.map(|n| 1. / n as f64)));
    const REACH: i32 = 3;
    let mut candidates = vec![];
    for i in -REACH..=REACH {
        for j in -REACH..=REACH {
            for k in -REACH..=REACH {
                if [i, j, k] == [0, 0, 0] {
                    continue;
                }
                let v = mesh * Vector3::new(i as f64, j as f64, k as f64);
                candidates.push(((v.norm() * 1e8).round() as i64, [i, j, k], v));
            }
        }
    }
    candidates.sort_by_key(|v| (v.0, v.1));
    type Candidate = (i64, [i32; 3], Vector3<f64>);
    let mut shells: Vec<Vec<Candidate>> = vec![];
    for candidate in candidates {
        if shells.last().is_none_or(|s| s[0].0 != candidate.0) {
            shells.push(vec![]);
        }
        shells.last_mut().unwrap().push(candidate);
    }
    let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let column = |shell: &[Candidate]| {
        DVector::from_iterator(
            6,
            pairs
                .iter()
                .map(|&(a, b)| shell.iter().map(|(_, _, v)| v[a] * v[b]).sum::<f64>()),
        )
    };
    let rank = |columns: &[DVector<f64>]| {
        if columns.is_empty() {
            return 0;
        }
        let matrix = DMatrix::from_columns(columns);
        let svd = matrix.svd(false, false);
        let largest = svd.singular_values.max();
        svd.singular_values
            .iter()
            .filter(|s| **s > 1e-10 * largest.max(1e-300))
            .count()
    };
    let parallel =
        |a: &Vector3<f64>, b: &Vector3<f64>| a.cross(b).norm() <= 1e-8 * a.norm() * b.norm();
    let target = DVector::from_vec(vec![1., 1., 1., 0., 0., 0.]);
    let mut accepted: Vec<usize> = vec![];
    let mut columns: Vec<DVector<f64>> = vec![];
    // Once the independent shells solve the equations only with a negative weight, further
    // non-parallel shells are added and the weights found by nonnegative least squares.
    let mut full = false;
    let mut extra = 0;
    for (s, shell) in shells.iter().enumerate() {
        if accepted.iter().any(|&a| {
            shells[a]
                .iter()
                .any(|(_, _, u)| shell.iter().any(|(_, _, v)| parallel(u, v)))
        }) {
            continue;
        }
        let mut trial = columns.clone();
        trial.push(column(shell));
        if !full && rank(&trial) <= columns.len() {
            continue;
        }
        accepted.push(s);
        columns = trial;
        let moments = DMatrix::from_columns(&columns);
        let weights = if full {
            extra += 1;
            nonnegative_least_squares(&moments, &target)?
        } else {
            moments
                .clone()
                .svd(true, true)
                .solve(&target, 1e-12)
                .map_err(invalid)?
        };
        if (&moments * &weights - &target).amax() >= 1e-8 {
            if extra >= SHELLS_BEYOND_RANK {
                break;
            }
            continue;
        }
        // A shell the solution does not need has a zero weight and is left out.
        let negligible = 1e-12 * weights.amax();
        if weights.iter().any(|w| *w < -negligible) {
            full = true;
            continue;
        }
        let mut out = Neighbors {
            vectors: vec![],
            weights: vec![],
            steps: vec![],
        };
        for (&a, weight) in accepted.iter().zip(weights.iter()) {
            if *weight <= negligible {
                continue;
            }
            for (_, step, v) in &shells[a] {
                out.vectors.push((*v).into());
                out.steps.push(*step);
                out.weights.push(*weight);
            }
        }
        return Ok(out);
    }
    Err(invalid(format!(
        "the neighbor shells of the {size:?} mesh reproduce the Wannier Cartesian metric only \
         with a negative weight, or not at all: use a mesh whose spacing is closer to isotropic \
         along the cell"
    )))
}

/// Non-parallel shells tried beyond the six independent ones before the shell search gives up.
const SHELLS_BEYOND_RANK: usize = 12;

/// `min |A x - b|` over `x >= 0` (Lawson and Hanson's active-set method), for the few columns of
/// a shell search.
fn nonnegative_least_squares(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>> {
    let n = a.ncols();
    let mut x = DVector::zeros(n);
    let mut passive = vec![false; n];
    let tolerance = 1e-12 * a.amax().max(1.) * b.amax().max(1.);
    let solve = |passive: &[bool]| -> Result<DVector<f64>> {
        let index: Vec<usize> = (0..n).filter(|j| passive[*j]).collect();
        let sub = DMatrix::from_fn(a.nrows(), index.len(), |i, j| a[(i, index[j])]);
        let z = sub.svd(true, true).solve(b, 1e-14).map_err(invalid)?;
        let mut out = DVector::zeros(n);
        for (k, &j) in index.iter().enumerate() {
            out[j] = z[k];
        }
        Ok(out)
    };
    for _ in 0..3 * n + 10 {
        let w = a.transpose() * (b - a * &x);
        let Some(next) = (0..n)
            .filter(|j| !passive[*j] && w[*j] > tolerance)
            .max_by(|i, j| w[*i].total_cmp(&w[*j]))
        else {
            break;
        };
        passive[next] = true;
        for _ in 0..3 * n + 10 {
            let z = solve(&passive)?;
            if (0..n).all(|j| !passive[j] || z[j] > 0.) {
                x = z;
                break;
            }
            let alpha = (0..n)
                .filter(|j| passive[*j] && z[*j] <= 0.)
                .map(|j| x[j] / (x[j] - z[j]))
                .fold(f64::INFINITY, f64::min);
            x += (&z - &x) * alpha;
            for j in 0..n {
                if passive[j] && x[j] <= tolerance {
                    passive[j] = false;
                    x[j] = 0.;
                }
            }
        }
    }
    Ok(x)
}

/// The fractional point `index` of the Gamma-anchored mesh `size` (C order).
pub fn mesh_point(size: [usize; 3], index: usize) -> Result<[f64; 3]> {
    let mesh = MeshIndex::new(size, [0.; 3])?;
    if index >= mesh.points() {
        return Err(invalid("invalid Wannier mesh index"));
    }
    Ok(mesh.point(index))
}
/// `out[k * steps + b]`: the mesh point reached from point `k` by the integer step `steps[b]`.
pub fn neighbor_indices(
    ctx: &Context,
    size: [usize; 3],
    steps: &[[i32; 3]],
) -> Result<Buffer<usize>> {
    let mesh = MeshIndex::new(size, [0.; 3])?;
    if steps.is_empty() {
        return Err(invalid("invalid Wannier neighbor indices"));
    }
    let points = mesh.points();
    let mut out = ctx.allocate(product(&[points, steps.len()])?)?;
    for k in 0..points {
        for (b, step) in steps.iter().enumerate() {
            out.as_mut_slice()[k * steps.len() + b] = mesh.shift(k, *step);
        }
    }
    Ok(out)
}

pub struct Frame {
    pub fractional: [f64; 3],
    pub samples: ComplexSamples,
    /// Rows are (component,band); columns are atom-centered projectors.
    pub projections: Matrix,
    pub coefficients: Matrix,
    pub energies: Buffer<f64>,
    /// Every eigenvalue of the diagonalization at this point (Hartree, on the scale of
    /// `PeriodicResult::fermi`), ascending: the gaps around the selected bands.
    pub levels: Buffer<f64>,
    pub components: usize,
    _memory: Reservation,
}

impl Frame {
    pub fn from_point(
        c: &Crystal,
        result: &PeriodicResult,
        point: &KPoint,
        spin: usize,
        bands: &[usize],
    ) -> Result<Self> {
        if spin >= c.spins || bands.is_empty() || bands.iter().any(|i| *i >= c.bands()) {
            return Err(invalid("invalid Wannier spin or band selection"));
        }
        let unique: std::collections::BTreeSet<_> = bands.iter().collect();
        if unique.len() != bands.len() {
            return Err(invalid("Wannier band indices must be unique"));
        }
        let ctx = &c.context;
        let n = c.grid.index.points();
        let components = if c.spinor { 2 } else { 1 };
        let _work = ctx.reserve(product(&[c.bands(), c.bands(), 16, 16])?)?;
        // A hybrid frame needs the exact exchange K(k) of the converged density matrices, as the
        // bands path does: without it the frame is a different Hamiltonian (no error, wrong bands).
        let exchange = if c.screened_exchange()?.is_some() {
            Some(c.hybrid_exchange_at_k(
                &result.matrices,
                point,
                spin,
                result.options.exxdiv_treatment,
            )?)
        } else {
            None
        };
        let (e, coefficients, _) = diagonalize_with_exchange(
            c,
            point,
            if c.spinor {
                result.potential.as_slice()
            } else {
                &result.potential.as_slice()[spin * n..(spin + 1) * n]
            },
            result.moment_potential.as_slice(),
            result
                .kinetic_potential
                .as_ref()
                .map(|v| &v.as_slice()[spin * n..(spin + 1) * n]),
            // The converged DFT+U operator of this spin channel.
            result.hubbard.as_ref().map(|h| &h.operators[spin]),
            exchange.as_ref(),
        )?;
        let memory = ctx.reserve(product(&[c.bands() + components * c.p, bands.len(), 16])?)?;
        let coefficients =
            Matrix::from_fn(c.bands(), bands.len(), |i, j| coefficients[(i, bands[j])]);
        let projections = Matrix::from_fn(components * bands.len(), c.p, |row, p| {
            let component = row / bands.len();
            let band = row % bands.len();
            (0..c.m)
                .map(|a| {
                    coefficients[(component * c.m + a, band)].conj() * point.projections[(a, p)]
                })
                .sum()
        });
        let mut values = ctx.allocate::<C64>(product(&[components, bands.len(), n])?)?;
        // Stream an AO row: archived inputs need only one grid row resident.
        for a in 0..c.m {
            let ao = point.psi.read_range(ctx, a * n, n)?;
            for component in 0..components {
                for band in 0..bands.len() {
                    let factor = coefficients[(component * c.m + a, band)];
                    for (to, from) in values.as_mut_slice()[(component * bands.len() + band) * n
                        ..(component * bands.len() + band + 1) * n]
                        .iter_mut()
                        .zip(ao.iter())
                    {
                        *to += factor * from;
                    }
                }
            }
        }
        let samples = ComplexSamples::archived_copy(ctx, values.as_slice(), Precision::Float64)?;
        let energies = ctx.copy_slice(
            &bands
                .iter()
                .map(|i| e[*i] + c.reference)
                .collect::<Vec<_>>(),
        )?;
        let levels = ctx.copy_slice(&e.iter().map(|v| v + c.reference).collect::<Vec<_>>())?;
        Ok(Self {
            fractional: point.fractional,
            samples,
            projections,
            coefficients,
            energies,
            levels,
            components,
            _memory: memory,
        })
    }
}

/// The PAW one-center geometry of the overlaps `<psi_k| e^{-ib.r} |psi_k+b>`: the atom centers
/// and compensation channels `(atom, L, M)` of the setup, and the one-center remainders of the
/// plane wave for every `|b|` used, computed once.
pub struct Augmentation<'a> {
    pub centers: Vec<[f64; 3]>,
    pub labels: Vec<[i32; 3]>,
    setup: &'a PeriodicSetup,
    remainders: Mutex<Vec<(f64, Arc<Vec<Matrix>>, Reservation)>>,
}
impl<'a> Augmentation<'a> {
    /// The geometry of `setup`, the one owner of the atom centers and channels.
    pub fn from_setup(setup: &'a PeriodicSetup, c: &Crystal) -> Result<Self> {
        let labels = setup
            .channels
            .iter()
            .map(|&(atom, l, m)| {
                Ok([
                    i32::try_from(atom).map_err(|_| crate::Error::Overflow)?,
                    i32::try_from(l).map_err(|_| crate::Error::Overflow)?,
                    m,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        if setup.projectors.len() != c.p {
            return Err(invalid(
                "the setup's projectors do not match the crystal's projector count",
            ));
        }
        let out = Self {
            centers: setup.centers.clone(),
            labels,
            setup,
            remainders: Mutex::new(vec![]),
        };
        out.validate(c)?;
        Ok(out)
    }
    pub fn validate(&self, c: &Crystal) -> Result<()> {
        if self.labels.len() != c.channels
            || self.centers.iter().flatten().any(|v| !v.is_finite())
            || self.labels.iter().any(|v| {
                v[0] < 0
                    || v[0] as usize >= self.centers.len()
                    || v[1] < 0
                    || v[2].unsigned_abs() > v[1] as u32
            })
        {
            return Err(invalid("invalid Wannier PAW augmentation geometry"));
        }
        Ok(())
    }
    /// The plane-wave remainders of `|b| = length`, cached.
    fn remainders(&self, c: &Crystal, length: f64) -> Result<Arc<Vec<Matrix>>> {
        let mut cache = self
            .remainders
            .lock()
            .map_err(|_| invalid("the plane-wave remainder cache is poisoned"))?;
        if let Some((_, found, _)) = cache
            .iter()
            .find(|(l, _, _)| (l - length).abs() <= 1e-12 * length.max(1.))
        {
            return Ok(found.clone());
        }
        let memory = c
            .context
            .reserve(product(&[self.labels.len(), c.p, c.p, 16])?)?;
        let matrices = Arc::new(self.setup.plane_wave_remainders(&c.context, length)?);
        if matrices.len() != self.labels.len() || matrices.iter().any(|m| m.shape() != (c.p, c.p)) {
            return Err(invalid(
                "the plane-wave remainders do not match the compensation channels",
            ));
        }
        cache.push((length, matrices.clone(), memory));
        Ok(matrices)
    }
    /// The exact one-center matrix of `e^{-ib.r}` between projectors (`P x P`):
    ///
    /// ```text
    /// O(b) = sum_a e^{-ib.R_a} sum_LM 4 pi (-i)^L Y_LM(b^) [|b|^L/(2L+1)!! B^{LM}_a + R^{LM}_a(|b|)]
    /// ```
    ///
    /// from the plane-wave expansion `e^{-ib.r} = 4 pi sum_LM (-i)^L j_L(|b| r) Y*_LM(r^) Y_LM(b^)`
    /// (complex harmonics with the Condon-Shortley phase): `B^{LM}` are the crystal's multipole
    /// blocks (the leading term of `j_L`), `R^{LM}` the rest of the Bessel function
    /// ([`PeriodicSetup::plane_wave_remainders`]). To first order in `b` it is the monopole
    /// `sqrt(4 pi) B^{00}` and the dipole `-i sqrt(4 pi / 3) b.B^1`.
    pub fn operator(&self, c: &Crystal, b: [f64; 3]) -> Result<Matrix> {
        self.validate(c)?;
        if b.iter().any(|v| !v.is_finite()) {
            return Err(invalid("a plane-wave vector must be finite"));
        }
        let length = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let remainders = self.remainders(c, length)?;
        let direction = if length > 0. {
            b.map(|v| v / length)
        } else {
            [0., 0., 1.]
        };
        let mut out = Matrix::zeros(c.p, c.p);
        for (channel, [atom, l, m]) in self.labels.iter().enumerate() {
            let l = *l as usize;
            if l > 0 && length == 0. {
                continue;
            }
            let leading =
                length.powi(l as i32) / (1..=l).map(|k| (2 * k + 1) as f64).product::<f64>();
            let phase = C64::from_polar(
                1.,
                -(0..3)
                    .map(|a| b[a] * self.centers[*atom as usize][a])
                    .sum::<f64>(),
            );
            let weight = phase
                * C64::new(4. * PI, 0.)
                * C64::new(0., -1.).powi(l as i32)
                * multipole::harmonic(l, *m, direction, false);
            let remainder = &remainders[channel];
            for i in 0..c.p {
                for j in 0..c.p {
                    let block = c.blocks.as_slice()[(channel * c.p + i) * c.p + j];
                    out[(i, j)] += weight * (block * leading + remainder[(i, j)]);
                }
            }
        }
        Ok(out)
    }
}

/// Periodic-part overlap with the true displacement b, including across a
/// wrapped mesh boundary. Both spin components contribute coherently.
pub fn link(
    c: &Crystal,
    left: &Frame,
    right: &Frame,
    b: [f64; 3],
    augmentation: &Augmentation,
) -> Result<Matrix> {
    let bands = left.energies.as_slice().len();
    if right.energies.as_slice().len() != bands
        || right.components != left.components
        || b.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "Wannier neighbor frames must share their band and spin spaces",
        ));
    }
    let ctx = &c.context;
    let _work = ctx.reserve(product(&[bands + c.p, bands + c.p, 16, 4])?)?;
    let n = c.grid.index.points();
    let mut out = Matrix::zeros(bands, bands);
    let tile = n.min(2048);
    let mut lr = ctx.allocate::<f64>(product(&[bands, tile])?)?;
    let mut li = ctx.allocate::<f64>(product(&[bands, tile])?)?;
    let mut rr = ctx.allocate::<f64>(product(&[bands, tile])?)?;
    let mut ri = ctx.allocate::<f64>(product(&[bands, tile])?)?;
    let mut real = ctx.allocate::<f64>(product(&[bands, bands])?)?;
    let mut imaginary = ctx.allocate::<f64>(product(&[bands, bands])?)?;
    let _packing = ctx.reserve(product(&[bands, tile, 16])?)?;
    for s in 0..left.components {
        for start in (0..n).step_by(tile) {
            let count = tile.min(n - start);
            for i in 0..bands {
                let l = left
                    .samples
                    .read_range(ctx, (s * bands + i) * n + start, count)?;
                let r = right
                    .samples
                    .read_range(ctx, (s * bands + i) * n + start, count)?;
                for p in 0..count {
                    let coordinate = c.grid.point(start + p);
                    let phase = C64::from_polar(
                        c.grid.index.dv(),
                        -(0..3).map(|a| b[a] * coordinate[a]).sum::<f64>(),
                    );
                    let right = phase * r[p];
                    lr.as_mut_slice()[p * bands + i] = l[p].re;
                    li.as_mut_slice()[p * bands + i] = l[p].im;
                    rr.as_mut_slice()[i * tile + p] = right.re;
                    ri.as_mut_slice()[i * tile + p] = right.im;
                }
            }
            for (l, r, imag, sign) in [
                (&lr, &rr, false, 1.),
                (&li, &ri, false, 1.),
                (&lr, &ri, true, 1.),
                (&li, &rr, true, -1.),
            ] {
                use crate::math::gemm::{self, MatrixMut, MatrixRef};
                let target = if imag { &mut imaginary } else { &mut real };
                gemm::multiply(
                    MatrixRef {
                        values: l.as_slice(),
                        rows: bands,
                        columns: count,
                        leading: bands,
                    },
                    MatrixRef {
                        values: r.as_slice(),
                        rows: count,
                        columns: bands,
                        leading: tile,
                    },
                    MatrixMut {
                        values: target.as_mut_slice(),
                        rows: bands,
                        columns: bands,
                        leading: bands,
                    },
                    sign,
                    1.,
                )?;
            }
        }
        out += left.projections.rows(s * bands, bands)
            * augmentation.operator(c, b)?
            * right.projections.rows(s * bands, bands).adjoint();
    }
    for j in 0..bands {
        for i in 0..bands {
            out[(i, j)] += C64::new(
                real.as_slice()[i + j * bands],
                imaginary.as_slice()[i + j * bands],
            );
        }
    }
    Ok(out)
}

/// Mesh points whose links one parallel region forms before they are stored.
const LINK_CHUNK: usize = 16;

pub fn overlap_mesh(
    c: &Crystal,
    frames: &[Frame],
    size: [usize; 3],
    augmentation: &Augmentation,
) -> Result<Mesh> {
    let ctx = &c.context;
    if frames.len() != product(&size)? || frames.is_empty() {
        return Err(invalid("Wannier frames must fill the full mesh"));
    }
    for (k, frame) in frames.iter().enumerate() {
        let expected = mesh_point(size, k)?;
        if (0..3).any(|a| {
            (frame.fractional[a] - expected[a] - (frame.fractional[a] - expected[a]).round()).abs()
                > 1e-8
        }) {
            return Err(invalid(
                "Wannier frames must follow the full Gamma mesh in C order",
            ));
        }
    }
    let shells = neighbor_shells(lattice(&c.grid), size)?;
    let neighbors = neighbor_indices(ctx, size, &shells.steps)?;
    // The localization reads every link several times per step (the spread and the gradient at
    // each point of a line search): keep them resident when they take at most a quarter of the
    // free budget, archive them otherwise. The values are the same either way.
    let bands = frames[0].energies.as_slice().len();
    let total = product(&[frames.len(), shells.vectors.len(), bands, bands, 16])?;
    let free = ctx.memory_limit().saturating_sub(ctx.memory_used());
    let archive = total > free / 4;
    // The links of a point are one task (P6: on the layout's lanes); the points go in chunks so
    // that an archived mesh never holds more than a chunk of links in memory.
    let mut links = Vec::with_capacity(frames.len() * shells.vectors.len());
    let targets = neighbors.as_slice();
    for first in (0..frames.len()).step_by(LINK_CHUNK) {
        let count = LINK_CHUNK.min(frames.len() - first);
        let chunk = ctx.map_independent(count, |i| {
            let k = first + i;
            shells
                .vectors
                .iter()
                .enumerate()
                .map(|(b, vector)| {
                    link(
                        c,
                        &frames[k],
                        &frames[targets[k * shells.vectors.len() + b]],
                        *vector,
                        augmentation,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for matrix in chunk.into_iter().flatten() {
            links.push(crate::math::storage::StoredMatrix::new(ctx, matrix, archive)?);
        }
    }
    let out = Mesh {
        points: frames.len(),
        bands: frames[0].energies.as_slice().len(),
        vectors: ctx.copy_slice(&shells.vectors)?,
        weights: ctx.copy_slice(&shells.weights)?,
        neighbors,
        links,
    };
    out.validate(ctx)?;
    Ok(out)
}

pub struct Trial {
    pub center: [f64; 3],
    pub angular: Vec<(usize, i32, f64)>,
    pub width: f64,
    pub spin: Option<[C64; 2]>,
    pub label: String,
}
impl Trial {
    pub fn validate(&self) -> Result<()> {
        if !self.width.is_finite()
            || self.width <= 0.
            || self.center.iter().any(|v| !v.is_finite())
            || self.angular.is_empty()
            || self
                .angular
                .iter()
                .any(|(l, m, c)| *l > 16 || m.unsigned_abs() as usize > *l || !c.is_finite())
            || self.angular.iter().all(|(_, _, c)| *c == 0.)
            || self.spin.is_some_and(|s| {
                s.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
                    || (s.iter().map(|v| v.norm_sqr()).sum::<f64>() - 1.).abs() > 1e-8
            })
        {
            return Err(invalid("invalid localized trial orbital"));
        }
        Ok(())
    }
    pub fn value(&self, offset: [f64; 3]) -> f64 {
        let r2 = offset.iter().map(|v| v * v).sum::<f64>();
        if r2 >= 36. * self.width * self.width {
            return 0.;
        }
        self.angular
            .iter()
            .map(|(l, m, c)| {
                let mut gamma = PI.sqrt() / 2.;
                for degree in 0..*l {
                    gamma *= degree as f64 + 1.5;
                }
                let normalization = (2. / (gamma * self.width.powi(2 * (*l as i32) + 3))).sqrt();
                c * normalization
                    * r2.sqrt().powi(*l as i32)
                    * (-0.5 * r2 / (self.width * self.width)).exp()
                    * angular::cartesian(*l, *m, offset)
            })
            .sum()
    }
}

pub fn trial_projections(c: &Crystal, frame: &Frame, trials: &[Trial]) -> Result<Matrix> {
    let ctx = &c.context;
    let bands = frame.energies.as_slice().len();
    let n = c.grid.index.points();
    let _work = ctx.reserve(product(&[bands, trials.len(), 16])?)?;
    let mut out = Matrix::zeros(bands, trials.len());
    let cell = lattice(&c.grid);
    let inverse = cell
        .try_inverse()
        .ok_or_else(|| invalid("singular Wannier trial cell"))?;
    for (j, trial) in trials.iter().enumerate() {
        trial.validate()?;
        let spin = if frame.components == 2 {
            trial
                .spin
                .ok_or_else(|| invalid("spinor Bloch states need two-component trial orbitals"))?
        } else {
            [C64::new(1., 0.), C64::default()]
        };
        let reach = 6. * trial.width;
        let mut values = ctx.allocate::<C64>(n)?;
        for (p, value) in values.as_mut_slice().iter_mut().enumerate() {
            let offset = Vector3::from(c.grid.point(p)) - Vector3::from(trial.center);
            let fractional = inverse * offset;
            let bounds: [(i32, i32); 3] = std::array::from_fn(|a| {
                let extent = inverse.row(a).norm() * reach;
                (
                    (fractional[a] - extent).ceil() as i32,
                    (fractional[a] + extent).floor() as i32,
                )
            });
            for x in bounds[0].0..=bounds[0].1 {
                for y in bounds[1].0..=bounds[1].1 {
                    for z in bounds[2].0..=bounds[2].1 {
                        let image = [x, y, z];
                        let local = offset - cell * Vector3::new(x as f64, y as f64, z as f64);
                        let phase = 2.
                            * PI
                            * (0..3)
                                .map(|a| frame.fractional[a] * image[a] as f64)
                                .sum::<f64>();
                        *value += C64::from_polar(trial.value(local.into()), phase);
                    }
                }
            }
        }
        for (s, spin_weight) in spin.iter().enumerate().take(frame.components) {
            for band in 0..bands {
                let sample = frame.samples.read_range(ctx, (s * bands + band) * n, n)?;
                out[(band, j)] += spin_weight
                    * (0..n)
                        .map(|p| sample[p].conj() * values.as_slice()[p])
                        .sum::<C64>()
                    * c.grid.index.dv();
            }
        }
    }
    Ok(out)
}
