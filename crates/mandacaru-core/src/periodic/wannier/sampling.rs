//! Full-mesh Bloch frames, PAW neighbor links and localized trial projections.
//! A frame owns its band samples; large samples are archived by the context.
use super::{invalid, Matrix, Mesh};
use crate::{
    math::{
        angular,
        storage::{ComplexSamples, Precision},
        SpatialGrid,
    },
    periodic::{solver::diagonalize, Crystal, KPoint, PeriodicResult},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::f64::consts::PI;

pub struct Neighbors {
    pub vectors: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub steps: Vec<[i32; 3]>,
}

pub fn lattice(grid: &SpatialGrid) -> Matrix3<f64> {
    grid.step * Matrix3::from_diagonal(&Vector3::from(grid.index.shape().map(|n| n as f64)))
}

/// Add complete equal-length shells until their second moments reproduce the
/// Cartesian metric. Each shell has one weight, including both vector signs.
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
    let mut candidates = vec![];
    for i in -2..=2 {
        for j in -2..=2 {
            for k in -2..=2 {
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
    let target = DVector::from_vec(vec![1., 1., 1., 0., 0., 0.]);
    for count in 1..=shells.len() {
        let moments = DMatrix::from_fn(6, count, |a, s| {
            shells[s]
                .iter()
                .map(|(_, _, v)| {
                    let (i, j) = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)][a];
                    v[i] * v[j]
                })
                .sum::<f64>()
        });
        let weights = moments
            .clone()
            .svd(true, true)
            .solve(&target, 1e-12)
            .map_err(invalid)?;
        if (&moments * &weights - &target).amax() < 1e-8 {
            let mut out = Neighbors {
                vectors: vec![],
                weights: vec![],
                steps: vec![],
            };
            for s in 0..count {
                if weights[s].abs() < 1e-12 {
                    continue;
                }
                for (_, step, v) in &shells[s] {
                    out.vectors.push((*v).into());
                    out.steps.push(*step);
                    out.weights.push(weights[s]);
                }
            }
            return Ok(out);
        }
    }
    Err(invalid(
        "the candidate neighbor shells cannot reproduce the Wannier Cartesian metric",
    ))
}

pub fn mesh_point(size: [usize; 3], index: usize) -> Result<[f64; 3]> {
    if size.contains(&0) || index >= product(&size)? {
        return Err(invalid("invalid Wannier mesh index"));
    }
    Ok([
        (index / (size[1] * size[2])) as f64 / size[0] as f64,
        ((index / size[2]) % size[1]) as f64 / size[1] as f64,
        (index % size[2]) as f64 / size[2] as f64,
    ])
}
pub fn neighbor_indices(
    ctx: &Context,
    size: [usize; 3],
    steps: &[[i32; 3]],
) -> Result<Buffer<usize>> {
    let points = product(&size)?;
    if size.contains(&0) || size.iter().any(|n| *n > i32::MAX as usize) || steps.is_empty() {
        return Err(invalid("invalid Wannier neighbor indices"));
    }
    let mut out = ctx.allocate(product(&[points, steps.len()])?)?;
    for k in 0..points {
        let origin = [
            k / (size[1] * size[2]),
            (k / size[2]) % size[1],
            k % size[2],
        ];
        for (b, step) in steps.iter().enumerate() {
            let at: [usize; 3] = std::array::from_fn(|a| {
                ((origin[a] as i64 + step[a] as i64).rem_euclid(size[a] as i64)) as usize
            });
            out.as_mut_slice()[k * steps.len() + b] = (at[0] * size[1] + at[1]) * size[2] + at[2];
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
        let (e, coefficients, _) = diagonalize(
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
        let samples = ComplexSamples::archived_copy(ctx, values.as_slice(), Precision::Double)?;
        let energies = ctx.copy_slice(
            &bands
                .iter()
                .map(|i| e[*i] + c.reference)
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            fractional: point.fractional,
            samples,
            projections,
            coefficients,
            energies,
            components,
            _memory: memory,
        })
    }
}

pub struct Augmentation {
    pub centers: Vec<[f64; 3]>,
    pub labels: Vec<[i32; 3]>,
}
impl Augmentation {
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
    pub fn operator(&self, c: &Crystal, b: [f64; 3]) -> Result<Matrix> {
        self.validate(c)?;
        let mut out = Matrix::zeros(c.p, c.p);
        for (channel, [atom, l, m]) in self.labels.iter().enumerate() {
            let radial = match (*l, *m) {
                (0, 0) => C64::new((4. * PI).sqrt(), 0.),
                (1, 0) => C64::new(0., -b[2] * (4. * PI / 3.).sqrt()),
                (1, -1) => C64::new(-b[1], -b[0]) * (8. * PI / 3.).sqrt() / 2.,
                (1, 1) => C64::new(-b[1], b[0]) * (8. * PI / 3.).sqrt() / 2.,
                _ => C64::default(),
            };
            let factor = C64::from_polar(
                1.,
                -(0..3)
                    .map(|a| b[a] * self.centers[*atom as usize][a])
                    .sum::<f64>(),
            ) * radial;
            for i in 0..c.p {
                for j in 0..c.p {
                    out[(i, j)] += factor * c.blocks.as_slice()[(channel * c.p + i) * c.p + j];
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
    let mut links = vec![];
    for k in 0..frames.len() {
        for (b, vector) in shells.vectors.iter().enumerate() {
            let matrix = link(
                c,
                &frames[k],
                &frames[neighbors.as_slice()[k * shells.vectors.len() + b]],
                *vector,
                augmentation,
            )?;
            links.push(crate::math::storage::StoredMatrix::new(ctx, matrix, true)?);
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
