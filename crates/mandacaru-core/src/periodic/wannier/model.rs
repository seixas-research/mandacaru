//! Localized Bloch subspaces and their Fourier-interpolated Hamiltonians.
use super::{
    disentangle, invalid, localize,
    sampling::{self, Augmentation, Frame, Trial},
    Localized, Matrix, Options,
};
use crate::{
    math::linalg::eigen,
    periodic::{Crystal, PeriodicResult},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DVector, Matrix3, Vector3};
use std::{collections::BTreeMap, f64::consts::PI};

pub struct Windows {
    pub outer: [f64; 2],
    pub frozen: Option<[f64; 2]>,
}
pub struct Channel {
    pub frames: Vec<Frame>,
    pub localized: Localized,
    pub disentanglement: Vec<f64>,
    pub disentanglement_converged: bool,
    pub hamiltonian_bvk: Buffer<C64>,
    pub density_bvk: Buffer<C64>,
    pub hoppings: Hoppings,
    pub size: [usize; 3],
}
pub struct Hoppings {
    pub vectors: Buffer<[i32; 3]>,
    pub values: Buffer<C64>,
    pub functions: usize,
}
pub struct Localization<'a> {
    pub size: [usize; 3],
    pub augmentation: &'a Augmentation,
    pub trials: &'a [Trial],
    pub windows: Option<&'a Windows>,
}

/// Rectangular polar factor, with explicit rank validation. Singular trials
/// must be repaired by the caller rather than inventing missing columns.
pub fn polar(a: &Matrix) -> Result<Matrix> {
    if a.nrows() < a.ncols() || a.ncols() == 0 {
        return Err(invalid("invalid Wannier trial projection shape"));
    }
    let (s, u) = eigen(a.adjoint() * a)?;
    if s[0] <= 1e-12 * s[s.len() - 1].max(1.) {
        return Err(invalid("Wannier trial projections are rank deficient"));
    }
    Ok(a * &u
        * Matrix::from_diagonal(&DVector::from_iterator(
            s.len(),
            s.iter().map(|x| C64::new(1. / x.sqrt(), 0.)),
        ))
        * u.adjoint())
}

pub fn localize_channel(
    c: &Crystal,
    state: &PeriodicResult,
    frames: Vec<Frame>,
    settings: Localization<'_>,
    options: &Options,
) -> Result<Channel> {
    let Localization {
        size,
        augmentation,
        trials,
        windows,
    } = settings;
    let ctx = &c.context;
    let mesh = sampling::overlap_mesh(c, &frames, size, augmentation)?;
    let functions = trials.len();
    let _work = ctx.reserve(product(&[frames.len(), mesh.bands, functions, 16, 3])?)?;
    let projections = frames
        .iter()
        .map(|f| sampling::trial_projections(c, f, trials))
        .collect::<Result<Vec<_>>>()?;
    let selected;
    let (initial, history, selection_converged) = if let Some(windows) = windows {
        for window in [Some(windows.outer), windows.frozen].into_iter().flatten() {
            if window.iter().any(|x| !x.is_finite()) || window[0] > window[1] {
                return Err(invalid(
                    "Wannier energy windows must be finite ordered pairs",
                ));
            }
        }
        if windows
            .frozen
            .is_some_and(|w| w[0] < windows.outer[0] || w[1] > windows.outer[1])
        {
            return Err(invalid(
                "frozen Wannier window must be inside the outer window",
            ));
        }
        let mask = |window: [f64; 2]| {
            frames
                .iter()
                .flat_map(|f| {
                    f.energies
                        .as_slice()
                        .iter()
                        .map(move |e| u8::from(*e >= window[0] && *e <= window[1]))
                })
                .collect::<Vec<_>>()
        };
        let masks = disentangle::Windows {
            inside: ctx.copy_slice(&mask(windows.outer))?,
            frozen: ctx.copy_slice(
                &windows
                    .frozen
                    .map_or_else(|| vec![0; mesh.points * mesh.bands], mask),
            )?,
            functions,
        };
        selected = disentangle::select(
            ctx,
            &mesh,
            &projections,
            &masks,
            None,
            &disentangle::Options::default(),
        )?;
        let initial = selected
            .gauge
            .iter()
            .zip(&projections)
            .map(|(s, a)| polar(&(s.adjoint() * a)).map(|u| s * u))
            .collect::<Result<Vec<_>>>()?;
        (initial, selected.history.clone(), selected.converged)
    } else {
        if functions != mesh.bands {
            return Err(invalid("isolated Wannier groups need one trial per selected band; use energy windows to disentangle"));
        }
        (
            projections.iter().map(polar).collect::<Result<Vec<_>>>()?,
            vec![],
            true,
        )
    };
    let centers = trials.iter().map(|t| t.center).collect::<Vec<_>>();
    let localized = localize(ctx, &mesh, &initial, Some(&centers), options)?;
    drop(mesh);
    let n = product(&size)?;
    let count = product(&[n, functions, functions])?;
    let mut hk = ctx.allocate::<C64>(count)?;
    let mut rho = ctx.allocate::<C64>(count)?;
    for (k, (frame, gauge)) in frames.iter().zip(&localized.gauge).enumerate() {
        let h = gauge.adjoint()
            * Matrix::from_diagonal(&DVector::from_iterator(
                frame.energies.as_slice().len(),
                frame.energies.as_slice().iter().map(|e| C64::new(*e, 0.)),
            ))
            * gauge;
        let occupations = frame
            .energies
            .as_slice()
            .iter()
            .map(|e| {
                C64::new(
                    c.degeneracy()
                        * state
                            .options
                            .smearing
                            .occupation((*e - state.fermi) / state.options.width),
                    0.,
                )
            })
            .collect::<Vec<_>>();
        let d = gauge.adjoint() * Matrix::from_diagonal(&DVector::from_vec(occupations)) * gauge;
        for i in 0..functions {
            for j in 0..functions {
                hk.as_mut_slice()[(k * functions + i) * functions + j] = h[(i, j)];
                rho.as_mut_slice()[(k * functions + i) * functions + j] = d[(i, j)];
            }
        }
    }
    let hamiltonian_bvk = fourier_blocks(ctx, size, functions, hk.as_slice())?;
    let density_bvk = fourier_blocks(ctx, size, functions, rho.as_slice())?;
    let hoppings = nearest_hoppings(
        ctx,
        sampling::lattice(&c.grid),
        size,
        localized.spread.centers.as_slice(),
        hamiltonian_bvk.as_slice(),
    )?;
    Ok(Channel {
        frames,
        localized,
        disentanglement: history,
        disentanglement_converged: selection_converged,
        hamiltonian_bvk,
        density_bvk,
        hoppings,
        size,
    })
}

/// H(R) = sum_k exp(-ik.R) H(k)/N, all arrays in C order.
pub fn fourier_blocks(
    ctx: &Context,
    size: [usize; 3],
    functions: usize,
    values: &[C64],
) -> Result<Buffer<C64>> {
    let n = product(&size)?;
    if size.contains(&0)
        || functions == 0
        || values.len() != product(&[n, functions, functions])?
        || values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("invalid Wannier Fourier table"));
    }
    let mut out = ctx.allocate::<C64>(values.len())?;
    for r in 0..n {
        let index = [
            r / (size[1] * size[2]),
            (r / size[2]) % size[1],
            r % size[2],
        ];
        for k in 0..n {
            let fractional = sampling::mesh_point(size, k)?;
            let phase = C64::from_polar(
                1. / n as f64,
                -2. * PI * (0..3).map(|a| fractional[a] * index[a] as f64).sum::<f64>(),
            );
            for i in 0..functions * functions {
                out.as_mut_slice()[r * functions * functions + i] +=
                    phase * values[k * functions * functions + i];
            }
        }
    }
    Ok(out)
}

/// Every pair chooses all equally near BvK replicas; the tied contributions
/// share a weight. Bounds from the inverse supercell prove the search finite
/// for a general skew lattice rather than assuming the nearest image lies in
/// the adjacent 27 cells.
pub fn nearest_hoppings(
    ctx: &Context,
    cell: Matrix3<f64>,
    size: [usize; 3],
    centers: &[[f64; 3]],
    blocks: &[C64],
) -> Result<Hoppings> {
    let functions = centers.len();
    let n = product(&size)?;
    if size.contains(&0)
        || functions == 0
        || centers
            .iter()
            .flatten()
            .chain(cell.iter())
            .any(|v| !v.is_finite())
        || blocks.len() != product(&[n, functions, functions])?
    {
        return Err(invalid("invalid Wannier hopping geometry or blocks"));
    }
    let supercell = cell * Matrix3::from_diagonal(&Vector3::from(size.map(|v| v as f64)));
    let inverse = supercell
        .try_inverse()
        .ok_or_else(|| invalid("singular Wannier hopping cell"))?;
    let mut table: BTreeMap<[i32; 3], (Matrix, Reservation)> = BTreeMap::new();
    for r in 0..n {
        let vector = [
            r / (size[1] * size[2]),
            (r / size[2]) % size[1],
            r % size[2],
        ];
        for i in 0..functions {
            for j in 0..functions {
                let displacement = cell * Vector3::from(vector.map(|v| v as f64))
                    + Vector3::from(centers[j])
                    - Vector3::from(centers[i]);
                let fractional = inverse * displacement;
                let first = fractional.map(|v| -v.round());
                let mut best = (displacement + supercell * first).norm_squared();
                let radius = best.sqrt() + 1e-8;
                let bounds: [(i32, i32); 3] = std::array::from_fn(|a| {
                    let extent = inverse.row(a).norm() * radius;
                    (
                        (-fractional[a] - extent).ceil() as i32,
                        (-fractional[a] + extent).floor() as i32,
                    )
                });
                let count = bounds
                    .iter()
                    .map(|(lo, hi)| (i64::from(*hi) - i64::from(*lo) + 1).max(0) as usize)
                    .collect::<Vec<_>>();
                if product(&count)? > 1_000_000 {
                    return Err(invalid(
                        "Wannier nearest-replica search exceeds its geometric work bound",
                    ));
                }
                let mut chosen = vec![];
                for x in bounds[0].0..=bounds[0].1 {
                    for y in bounds[1].0..=bounds[1].1 {
                        for z in bounds[2].0..=bounds[2].1 {
                            let distance = (displacement
                                + supercell * Vector3::new(x as f64, y as f64, z as f64))
                            .norm_squared();
                            if distance < best - 1e-8 {
                                best = distance;
                                chosen.clear();
                            }
                            if (distance - best).abs() < 1e-8 {
                                chosen.push([x, y, z]);
                            }
                        }
                    }
                }
                if chosen.is_empty() {
                    return Err(invalid("Wannier replica search found no image"));
                }
                let weight = 1. / chosen.len() as f64;
                for image in chosen {
                    let mut vector_out = [0; 3];
                    for a in 0..3 {
                        vector_out[a] =
                            i32::try_from(vector[a] as i64 + image[a] as i64 * size[a] as i64)
                                .map_err(|_| crate::Error::Overflow)?;
                    }
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        table.entry(vector_out)
                    {
                        entry.insert((
                            Matrix::zeros(functions, functions),
                            ctx.reserve(product(&[functions, functions, 16])?)?,
                        ));
                    }
                    table.get_mut(&vector_out).unwrap().0[(i, j)] +=
                        blocks[(r * functions + i) * functions + j] * weight;
                }
            }
        }
    }
    let mut vectors = ctx.allocate::<[i32; 3]>(table.len())?;
    let mut values = ctx.allocate::<C64>(product(&[table.len(), functions, functions])?)?;
    for (r, (vector, (matrix, _))) in table.into_iter().enumerate() {
        vectors.as_mut_slice()[r] = vector;
        for i in 0..functions {
            for j in 0..functions {
                values.as_mut_slice()[(r * functions + i) * functions + j] = matrix[(i, j)];
            }
        }
    }
    Ok(Hoppings {
        vectors,
        values,
        functions,
    })
}
impl Hoppings {
    pub fn interpolate(&self, ctx: &Context, kpoints: &[[f64; 3]]) -> Result<Buffer<f64>> {
        ctx.run(|| self.interpolate_inner(ctx, kpoints))
    }
    fn interpolate_inner(&self, ctx: &Context, kpoints: &[[f64; 3]]) -> Result<Buffer<f64>> {
        if kpoints.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid(
                "Wannier interpolation needs finite fractional k-points",
            ));
        }
        let _work = ctx.reserve(product(&[self.functions, self.functions, 16, 8])?)?;
        let mut out = ctx.allocate::<f64>(product(&[kpoints.len(), self.functions])?)?;
        for (k, point) in kpoints.iter().enumerate() {
            let mut h = Matrix::zeros(self.functions, self.functions);
            for (r, vector) in self.vectors.as_slice().iter().enumerate() {
                let phase = C64::from_polar(
                    1.,
                    2. * PI * (0..3).map(|a| point[a] * vector[a] as f64).sum::<f64>(),
                );
                for i in 0..self.functions {
                    for j in 0..self.functions {
                        h[(i, j)] += phase
                            * self.values.as_slice()[(r * self.functions + i) * self.functions + j];
                    }
                }
            }
            if (&h - h.adjoint()).norm() > 1e-7 * h.norm().max(1.) {
                return Err(invalid(
                    "Wannier interpolation produced a non-Hermitian Hamiltonian",
                ));
            }
            let (e, _) = eigen((h.clone() + h.adjoint()) * C64::new(0.5, 0.))?;
            out.as_mut_slice()[k * self.functions..(k + 1) * self.functions]
                .copy_from_slice(e.as_slice());
        }
        Ok(out)
    }
}
