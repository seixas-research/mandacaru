//! Basis-function motion on the grid: contraction of an action on the occupied
//! orbitals with the derivative of the basis functions.
//!
//! For an energy `E[psi_i]` of the occupied orbitals, `dE / d psi_i = 2 f_i A_i`
//! with `A_i` the action of the part of the Fock operator that acts on the
//! grid. Moving atom `X` moves its basis functions rigidly, `d chi_a / d R_X =
//! -grad chi_a`, so the force from that part is `sum_i 2 f_i sum_{a in X}
//! c_ia <-grad chi_a | A_i>`. One function at a time is differentiated, so the
//! derivative stack of the whole basis never exists.
use crate::basis::Radial;
use crate::molecular::MolecularIntegrals;
use crate::paw::molecular::gradient::Occupied;
use crate::{product, Buffer, Result};
use nalgebra::{Matrix3, Vector3};

/// The grid points as `[x, y, z]` triples.
pub(super) fn points(integrals: &MolecularIntegrals) -> Result<Buffer<f64>> {
    let n = integrals.grid.index.points();
    let mut out = integrals.context.allocate::<f64>(product(&[3, n])?)?;
    let workers = integrals.context.threads().max(1).min(n.max(1));
    if workers == 1 {
        for k in 0..n {
            out.as_mut_slice()[3 * k..3 * k + 3].copy_from_slice(&integrals.grid.point(k));
        }
    } else {
        let chunk = n.div_ceil(workers);
        let out_slice = out.as_mut_slice();
        integrals.context.run(|| {
            std::thread::scope(|scope| {
                for (w, part) in out_slice.chunks_mut(3 * chunk).enumerate() {
                    let start = w * chunk;
                    let grid = &integrals.grid;
                    mandacaru_kernels::runtime::spawn(scope, move || {
                        let count = part.len() / 3;
                        for i in 0..count {
                            part[3 * i..3 * i + 3].copy_from_slice(&grid.point(start + i));
                        }
                    });
                }
            });
        });
    }
    Ok(out)
}

fn orbital_bounding_box(
    grid: &crate::math::grid::SpatialGrid,
    inv_step: &Matrix3<f64>,
    row_norms: &[f64; 3],
    center: [f64; 3],
    r_cut: f64,
) -> Option<[std::ops::RangeInclusive<usize>; 3]> {
    let shape = grid.index.shape();
    if !r_cut.is_finite() {
        return Some([0..=shape[0] - 1, 0..=shape[1] - 1, 0..=shape[2] - 1]);
    }
    let c_center = inv_step * (Vector3::from(center) - Vector3::from(grid.origin));
    let mut ranges = [0..=0, 0..=0, 0..=0];
    for d in 0..3 {
        let delta = r_cut * row_norms[d];
        let min_idx = ((c_center[d] - delta).floor() as isize).max(0) as usize;
        let max_idx = (c_center[d] + delta).ceil() as isize;
        if max_idx < 0 {
            return None;
        }
        let max_idx = (max_idx as usize).min(shape[d] - 1);
        if min_idx > max_idx {
            return None;
        }
        ranges[d] = min_idx..=max_idx;
    }
    Some(ranges)
}

#[derive(Clone, Copy)]
pub(super) struct CombinedInputs<'a> {
    pub psi: &'a [f64],
    pub external: &'a [f64],
    pub hartree: &'a [f64],
    pub xc: Option<&'a [f64]>,
    pub coulomb: &'a [f64],
    pub kinetic: Option<&'a [f64]>,
    pub xc_gradient: Option<&'a [f64]>,
    pub xc_tau: Option<&'a [f64]>,
}

#[derive(Default)]
pub(super) struct CombinedOutputs {
    pub external: Vec<f64>,
    pub hartree: Vec<f64>,
    pub xc: Vec<f64>,
    pub coulomb: Vec<f64>,
    pub pulay: Vec<f64>,
    pub kinetic: Option<Vec<f64>>,
    pub xc_gradient: Option<Vec<f64>>,
    pub xc_tau: Option<Vec<f64>>,
}

impl CombinedOutputs {
    fn new(atoms: usize, inputs: &CombinedInputs<'_>) -> Self {
        Self {
            external: vec![0.; 3 * atoms],
            hartree: vec![0.; 3 * atoms],
            xc: vec![0.; 3 * atoms],
            coulomb: vec![0.; 3 * atoms],
            pulay: vec![0.; 3 * atoms],
            kinetic: inputs.kinetic.map(|_| vec![0.; 3 * atoms]),
            xc_gradient: inputs.xc_gradient.map(|_| vec![0.; 3 * atoms]),
            xc_tau: inputs.xc_tau.map(|_| vec![0.; 3 * atoms]),
        }
    }

    fn accumulate(&mut self, other: &Self) {
        for (a, b) in self.external.iter_mut().zip(&other.external) {
            *a += b;
        }
        for (a, b) in self.hartree.iter_mut().zip(&other.hartree) {
            *a += b;
        }
        for (a, b) in self.xc.iter_mut().zip(&other.xc) {
            *a += b;
        }
        for (a, b) in self.coulomb.iter_mut().zip(&other.coulomb) {
            *a += b;
        }
        for (a, b) in self.pulay.iter_mut().zip(&other.pulay) {
            *a += b;
        }
        if let (Some(a), Some(b)) = (&mut self.kinetic, &other.kinetic) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        if let (Some(a), Some(b)) = (&mut self.xc_gradient, &other.xc_gradient) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        if let (Some(a), Some(b)) = (&mut self.xc_tau, &other.xc_tau) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

pub(super) fn contract_combined(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    points: &[f64],
    inputs: CombinedInputs<'_>,
) -> CombinedOutputs {
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let atoms = integrals.geometry.numbers().len();
    let num_orbitals = integrals.basis.orbitals.len();
    if num_orbitals == 0 {
        return CombinedOutputs::new(atoms, &inputs);
    }

    let inv_step = integrals.grid.step.try_inverse().unwrap_or(Matrix3::identity());
    let row_norms = [
        (inv_step[(0, 0)].powi(2) + inv_step[(0, 1)].powi(2) + inv_step[(0, 2)].powi(2)).sqrt(),
        (inv_step[(1, 0)].powi(2) + inv_step[(1, 1)].powi(2) + inv_step[(1, 2)].powi(2)).sqrt(),
        (inv_step[(2, 0)].powi(2) + inv_step[(2, 1)].powi(2) + inv_step[(2, 2)].powi(2)).sqrt(),
    ];
    let shape = integrals.grid.index.shape();
    let [_, ny, nz] = shape;

    let worker_fn = |start: usize, end: usize| -> CombinedOutputs {
        let mut local_out = CombinedOutputs::new(atoms, &inputs);
        let mut weight = vec![0.; occupied.len()];
        // The same orbital's weight in W = sum w_weight a a^T, for the grid Pulay term.
        let mut pulay_weight = vec![0.; occupied.len()];
        for a in start..end {
            let orbital = &integrals.basis.orbitals[a];
            for ((w, pw), o) in weight.iter_mut().zip(pulay_weight.iter_mut()).zip(occupied) {
                *w = 2. * o.weight * o.ao[a];
                *pw = 2. * o.w_weight * o.ao[a];
            }
            if weight.iter().chain(&pulay_weight).all(|&w| w == 0.) {
                continue;
            }
            let r_cut = match &orbital.radial {
                Radial::Tabulated(table) => table
                    .knots
                    .as_slice()
                    .last()
                    .copied()
                    .unwrap_or(f64::INFINITY),
                _ => f64::INFINITY,
            };
            let support2 = if r_cut.is_finite() {
                r_cut * r_cut
            } else {
                f64::INFINITY
            };
            let ranges = match orbital_bounding_box(
                &integrals.grid,
                &inv_step,
                &row_norms,
                orbital.center,
                r_cut,
            ) {
                Some(r) => r,
                None => continue,
            };

            let mut sum_ext = [0.; 3];
            let mut sum_har = [0.; 3];
            let mut sum_xc = [0.; 3];
            let mut sum_coul = [0.; 3];
            let mut sum_pulay = [0.; 3];
            let mut sum_kin = [0.; 3];
            let mut sum_grad = [0.; 3];
            let mut sum_tau = [0.; 3];

            for ix in ranges[0].clone() {
                for iy in ranges[1].clone() {
                    let base = (ix * ny + iy) * nz;
                    for iz in ranges[2].clone() {
                        let k = base + iz;
                        let point = [points[3 * k], points[3 * k + 1], points[3 * k + 2]];
                        let r2 = (point[0] - orbital.center[0]).powi(2)
                            + (point[1] - orbital.center[1]).powi(2)
                            + (point[2] - orbital.center[2]).powi(2);
                        if r2 > support2 {
                            continue;
                        }

                        let mut s_wv = 0.;
                        let mut t_pulay = 0.;
                        let mut t_xc = 0.;
                        let mut t_kin = 0.;
                        let mut t_grad = 0.;
                        let mut t_tau = 0.;

                        for (i, o) in occupied.iter().enumerate() {
                            let (w, pw) = (weight[i], pulay_weight[i]);
                            if w == 0. && pw == 0. {
                                continue;
                            }
                            let v = inputs.psi[i * n + k];
                            let wv = w * v;
                            s_wv += wv;
                            t_pulay -= pw * v;
                            if let Some(xc) = inputs.xc {
                                t_xc += wv * xc[o.spin * n + k];
                            }
                            if let Some(kin) = inputs.kinetic {
                                t_kin += w * kin[i * n + k];
                            }
                            if let Some(grad) = inputs.xc_gradient {
                                t_grad += w * grad[i * n + k];
                            }
                            if let Some(tau) = inputs.xc_tau {
                                t_tau += w * tau[i * n + k];
                            }
                        }

                        let t_ext = inputs.external[k] * s_wv;
                        let t_har = inputs.hartree[k] * s_wv;
                        let t_coul = inputs.coulomb[k] * s_wv;

                        if s_wv == 0.
                            && t_pulay == 0.
                            && t_xc == 0.
                            && t_kin == 0.
                            && t_grad == 0.
                            && t_tau == 0.
                        {
                            continue;
                        }

                        let gradient = orbital.gradient(point);
                        for axis in 0..3 {
                            let g = gradient[axis];
                            sum_ext[axis] += g * t_ext;
                            sum_har[axis] += g * t_har;
                            sum_xc[axis] += g * t_xc;
                            sum_coul[axis] += g * t_coul;
                            sum_pulay[axis] += g * t_pulay;
                            if inputs.kinetic.is_some() {
                                sum_kin[axis] += g * t_kin;
                            }
                            if inputs.xc_gradient.is_some() {
                                sum_grad[axis] += g * t_grad;
                            }
                            if inputs.xc_tau.is_some() {
                                sum_tau[axis] += g * t_tau;
                            }
                        }
                    }
                }
            }

            let atom_base = 3 * orbital.atom;
            for axis in 0..3 {
                local_out.external[atom_base + axis] -= dv * sum_ext[axis];
                local_out.hartree[atom_base + axis] -= dv * sum_har[axis];
                local_out.xc[atom_base + axis] -= dv * sum_xc[axis];
                local_out.coulomb[atom_base + axis] -= dv * sum_coul[axis];
                local_out.pulay[atom_base + axis] -= dv * sum_pulay[axis];
                if let Some(ref mut k_out) = local_out.kinetic {
                    k_out[atom_base + axis] -= dv * sum_kin[axis];
                }
                if let Some(ref mut g_out) = local_out.xc_gradient {
                    g_out[atom_base + axis] -= dv * sum_grad[axis];
                }
                if let Some(ref mut t_out) = local_out.xc_tau {
                    t_out[atom_base + axis] -= dv * sum_tau[axis];
                }
            }
        }
        local_out
    };

    let workers = integrals.context.threads().max(1).min(num_orbitals.max(1));
    if workers <= 1 {
        return worker_fn(0, num_orbitals);
    }

    let chunk = num_orbitals.div_ceil(workers);
    let mut total_out = CombinedOutputs::new(atoms, &inputs);
    integrals.context.run(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .filter_map(|w| {
                    let start = w * chunk;
                    let end = (start + chunk).min(num_orbitals);
                    if start < end {
                        Some(mandacaru_kernels::runtime::spawn(scope, move || {
                            worker_fn(start, end)
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            for handle in handles {
                let worker_out = handle.join().unwrap();
                total_out.accumulate(&worker_out);
            }
        });
    });
    total_out
}


/// The force on every atom from one action (`nocc * n` values).
pub(super) fn contract(
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
    points: &[f64],
    action: &[f64],
) -> Vec<f64> {
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let atoms = integrals.geometry.numbers().len();
    let num_orbitals = integrals.basis.orbitals.len();
    if num_orbitals == 0 {
        return vec![0.; 3 * atoms];
    }

    let inv_step = integrals.grid.step.try_inverse().unwrap_or(Matrix3::identity());
    let row_norms = [
        (inv_step[(0, 0)].powi(2) + inv_step[(0, 1)].powi(2) + inv_step[(0, 2)].powi(2)).sqrt(),
        (inv_step[(1, 0)].powi(2) + inv_step[(1, 1)].powi(2) + inv_step[(1, 2)].powi(2)).sqrt(),
        (inv_step[(2, 0)].powi(2) + inv_step[(2, 1)].powi(2) + inv_step[(2, 2)].powi(2)).sqrt(),
    ];
    let shape = integrals.grid.index.shape();
    let [_, ny, nz] = shape;

    let worker_fn = |start: usize, end: usize| -> Vec<f64> {
        let mut local_out = vec![0.; 3 * atoms];
        let mut weight = vec![0.; occupied.len()];
        for a in start..end {
            let orbital = &integrals.basis.orbitals[a];
            for (w, o) in weight.iter_mut().zip(occupied) {
                *w = 2. * o.weight * o.ao[a];
            }
            if weight.iter().all(|&w| w == 0.) {
                continue;
            }
            let r_cut = match &orbital.radial {
                Radial::Tabulated(table) => table
                    .knots
                    .as_slice()
                    .last()
                    .copied()
                    .unwrap_or(f64::INFINITY),
                _ => f64::INFINITY,
            };
            let support2 = if r_cut.is_finite() {
                r_cut * r_cut
            } else {
                f64::INFINITY
            };
            let ranges = match orbital_bounding_box(
                &integrals.grid,
                &inv_step,
                &row_norms,
                orbital.center,
                r_cut,
            ) {
                Some(r) => r,
                None => continue,
            };
            let mut sum = [0.; 3];
            for ix in ranges[0].clone() {
                for iy in ranges[1].clone() {
                    let base = (ix * ny + iy) * nz;
                    for iz in ranges[2].clone() {
                        let k = base + iz;
                        let point = [points[3 * k], points[3 * k + 1], points[3 * k + 2]];
                        let r2 = (point[0] - orbital.center[0]).powi(2)
                            + (point[1] - orbital.center[1]).powi(2)
                            + (point[2] - orbital.center[2]).powi(2);
                        if r2 > support2 {
                            continue;
                        }
                        let mut t = 0.;
                        for (i, &w) in weight.iter().enumerate() {
                            if w != 0. {
                                t += w * action[i * n + k];
                            }
                        }
                        if t == 0. {
                            continue;
                        }
                        let gradient = orbital.gradient(point);
                        for axis in 0..3 {
                            sum[axis] += gradient[axis] * t;
                        }
                    }
                }
            }
            let atom_base = 3 * orbital.atom;
            for axis in 0..3 {
                local_out[atom_base + axis] -= dv * sum[axis];
            }
        }
        local_out
    };

    let workers = integrals.context.threads().max(1).min(num_orbitals.max(1));
    if workers <= 1 {
        return worker_fn(0, num_orbitals);
    }

    let chunk = num_orbitals.div_ceil(workers);
    let mut total_out = vec![0.; 3 * atoms];
    integrals.context.run(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .filter_map(|w| {
                    let start = w * chunk;
                    let end = (start + chunk).min(num_orbitals);
                    if start < end {
                        Some(mandacaru_kernels::runtime::spawn(scope, move || {
                            worker_fn(start, end)
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            for handle in handles {
                let worker_out = handle.join().unwrap();
                for (tot, part) in total_out.iter_mut().zip(worker_out) {
                    *tot += part;
                }
            }
        });
    });
    total_out
}
