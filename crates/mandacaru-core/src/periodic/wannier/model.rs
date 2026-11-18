//! Localized Bloch subspaces and their Fourier-interpolated Hamiltonians.
use super::{
    disentangle, interpolation, invalid, localize_with_populations,
    populations::{AtomPartition, PopulationMesh},
    sampling::{self, Augmentation, Frame, Trial},
    Localization as ObjectiveMode, Localized, Matrix, Options,
};
use crate::{
    math::linalg::eigen,
    periodic::{bloch_mesh::MeshIndex, Crystal, PeriodicResult},
    product,
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::{DVector, Matrix3};
use std::f64::consts::PI;

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
    pub positions_r: [Buffer<C64>; 3],
    pub interpolation: interpolation::Model,
    pub subspace_electrons: f64,
    pub size: [usize; 3],
    /// `[n * trials + m]`: `|<w_n0|g_m>|^2`, the weight of the trial orbital `m` (normalized,
    /// smooth part of the states, no one-center term) in the home-cell function `n`. The trial
    /// order is the one that seeded the gauge, but a localized function need not keep its
    /// trial's character: select functions by these weights, not by index.
    pub trial_weights: Vec<f64>,
}
pub struct Hoppings {
    pub vectors: Buffer<[i32; 3]>,
    pub values: Buffer<C64>,
    pub functions: usize,
}
pub struct Localization<'a> {
    pub size: [usize; 3],
    pub augmentation: &'a Augmentation<'a>,
    pub trials: &'a [Trial],
    pub windows: Option<&'a Windows>,
    /// Reciprocal lattice vectors `G_j` as columns (1/Bohr).
    pub reciprocal: Matrix3<f64>,
    /// Refuse a subspace with a Chern number ([`super::chern_guard`]) before localizing it.
    pub chern_guard: bool,
    /// Atom partition + `(S_k, S_k^{-1/2})` metrics per mesh point when using `Populations`.
    pub populations: Option<(&'a AtomPartition, &'a [(Matrix, Matrix)])>,
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
        reciprocal,
        chern_guard,
        populations,
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
    if chern_guard {
        // The Chern number is a property of the subspace, so the initial gauge serves.
        let index = MeshIndex::new(size, [0.; 3])?;
        let mut along = Vec::with_capacity(frames.len());
        for k in 0..frames.len() {
            let mut row: [Option<Matrix>; 3] = [None, None, None];
            for (j, slot) in row.iter_mut().enumerate() {
                if size[j] < 2 {
                    continue;
                }
                let next = index.advance(k, j, 1);
                let b: [f64; 3] = (reciprocal.column(j) / size[j] as f64).into();
                let link = sampling::link(c, &frames[k], &frames[next], b, augmentation)?;
                *slot = Some(initial[k].adjoint() * link * &initial[next]);
            }
            along.push(row);
        }
        super::chern_guard(&index, &along)?;
    }
    let centers = trials.iter().map(|t| t.center).collect::<Vec<_>>();
    let pop_mesh = match options.localization {
        ObjectiveMode::Spread => None,
        ObjectiveMode::Populations { scheme, exponent } => {
            let (partition, metrics) = populations.ok_or_else(|| {
                invalid("Pipek-Mezey localization requires AO overlaps and atom partition")
            })?;
            let coeffs: Vec<&Matrix> = frames.iter().map(|f| &f.coefficients).collect();
            let metric_refs: Vec<(&Matrix, &Matrix)> =
                metrics.iter().map(|(s, lowdin)| (s, lowdin)).collect();
            Some(PopulationMesh::new(
                size,
                frames[0].components,
                &coeffs,
                &metric_refs,
                partition.clone(),
                scheme,
                exponent,
            )?)
        }
    };
    let localized = localize_with_populations(
        ctx,
        &mesh,
        pop_mesh.as_ref(),
        &initial,
        Some(&centers),
        options,
    )?;
    let nb = mesh.vectors.as_slice().len();
    let mesh_vectors = mesh.vectors.as_slice().to_vec();
    let mesh_weights = mesh.weights.as_slice().to_vec();
    let _rot_work = ctx.reserve(product(&[frames.len(), nb, functions, functions, 16])?)?;
    let mut rotated_links = Vec::with_capacity(frames.len() * nb);
    for k in 0..frames.len() {
        for b in 0..nb {
            rotated_links.push(mesh.rotated(ctx, &localized.gauge, k, b)?);
        }
    }
    drop(mesh);
    // <w_n0|g_m> = (1/N) sum_k (U_k^+ A_k)_nm, A_k the projections of the Bloch states on the
    // Bloch sums of the trials (both normalized).
    let mut overlap = Matrix::zeros(functions, trials.len());
    for (gauge, projection) in localized.gauge.iter().zip(&projections) {
        overlap += gauge.adjoint() * projection;
    }
    let trial_weights = overlap
        .transpose()
        .iter()
        .map(|v| v.norm_sqr() / (frames.len() * frames.len()) as f64)
        .collect();
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
    let subspace_electrons: f64 = (0..functions)
        .map(|i| density_bvk.as_slice()[i * functions + i].re)
        .sum();
    let cell = sampling::lattice(&c.grid);
    let replica =
        interpolation::ReplicaTable::build(cell, size, localized.spread.centers.as_slice())?;
    let h_r = replica.scatter(ctx, hamiltonian_bvk.as_slice())?;
    let hoppings = Hoppings {
        vectors: ctx.copy_slice(&replica.vectors)?,
        values: replica.pack(ctx, &h_r)?,
        functions,
    };
    let interp = interpolation::Model::from_rotated_links(
        ctx,
        &replica,
        &mesh_vectors,
        &mesh_weights,
        &rotated_links,
        localized.spread.centers.as_slice(),
        h_r,
    )?;
    let positions_r = [
        replica.pack(ctx, &interp.a_r[0])?,
        replica.pack(ctx, &interp.a_r[1])?,
        replica.pack(ctx, &interp.a_r[2])?,
    ];
    Ok(Channel {
        frames,
        localized,
        disentanglement: history,
        disentanglement_converged: selection_converged,
        hamiltonian_bvk,
        density_bvk,
        hoppings,
        positions_r,
        interpolation: interp,
        subspace_electrons,
        size,
        trial_weights,
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
/// share a weight ([`interpolation::ReplicaTable`]).
pub fn nearest_hoppings(
    ctx: &Context,
    cell: Matrix3<f64>,
    size: [usize; 3],
    centers: &[[f64; 3]],
    blocks: &[C64],
) -> Result<Hoppings> {
    let replica = interpolation::ReplicaTable::build(cell, size, centers)?;
    let h_r = replica.scatter(ctx, blocks)?;
    Ok(Hoppings {
        vectors: ctx.copy_slice(&replica.vectors)?,
        values: replica.pack(ctx, &h_r)?,
        functions: replica.functions,
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
