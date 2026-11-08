//! The terms of the gradient that sum over the atoms' short-range spheres or
//! over the grid with a field centered on an atom: the short-range local
//! potential, the long-range ionic potential and the partial core density.
use super::{
    super::{distance, local_sphere_radius, long_range, sphere, Setup, LOCAL},
    radial::{core_slope, long_range_slope},
    Occupied,
};
use crate::{molecular::MolecularIntegrals, Result};

/// The width of the Gaussian that splits the ionic potential in the setup.
pub(super) fn sigma(integrals: &MolecularIntegrals) -> f64 {
    1.4 * integrals.grid.spacings().into_iter().fold(0., f64::max)
}

/// The force from `sum_o f_o <o| V^sr |o>`, with `V^sr` the short-range part of
/// the local ionic potential integrated on a sphere around each atom. Moving
/// the atom moves the sphere and its functions together; the functions of
/// another atom stay behind.
pub(super) fn local_force(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    occupied: &[Occupied],
) -> Result<Vec<f64>> {
    let m = integrals.m();
    let positions = integrals.geometry.positions();
    let n_atoms = setup.atoms.len();
    let sigma = sigma(integrals);

    let compute_atom = |a: usize| -> Result<Vec<f64>> {
        let s = setup.atoms[a];
        let species = &setup.species[s];
        let radius = local_sphere_radius(species, sigma);
        let mut local_force = vec![0.; 3 * positions.len()];
        let mut phi = vec![0.; m];
        let mut half = vec![0.; m];

        let active_orbitals: Vec<(usize, f64)> = integrals
            .basis
            .orbitals
            .iter()
            .enumerate()
            .filter_map(|(i, orbital)| {
                let r_cut = match &orbital.radial {
                    crate::basis::Radial::Tabulated(table) => {
                        *table.knots.as_slice().last().unwrap_or(&f64::INFINITY)
                    }
                    _ => f64::INFINITY,
                };
                let d = distance(positions[a], orbital.center);
                if d <= radius + r_cut {
                    Some((i, if r_cut.is_finite() { r_cut * r_cut } else { f64::INFINITY }))
                } else {
                    None
                }
            })
            .collect();

        sphere(
            radius,
            species.local_radius,
            LOCAL[0],
            LOCAL[1],
            LOCAL[2],
            |offset, r, weight| {
                let g_r = species.local(r) - long_range(species.charge, sigma, r);
                if g_r.abs() < 1e-15 {
                    return Ok(());
                }
                let g = weight * g_r;
                let point = std::array::from_fn(|i| positions[a][i] + offset[i]);
                phi.fill(0.);
                for &(i, support2) in &active_orbitals {
                    let orbital = &integrals.basis.orbitals[i];
                    let r2 = (point[0] - orbital.center[0]).powi(2)
                        + (point[1] - orbital.center[1]).powi(2)
                        + (point[2] - orbital.center[2]).powi(2);
                    if r2 <= support2 {
                        phi[i] = orbital.evaluate(point);
                    }
                }
                half.fill(0.);
                for o in occupied {
                    let amplitude: f64 = active_orbitals
                        .iter()
                        .map(|&(i, _)| o.ao[i] * phi[i])
                        .sum();
                    if amplitude == 0. {
                        continue;
                    }
                    for &(i, _) in &active_orbitals {
                        half[i] += o.weight * o.ao[i] * amplitude;
                    }
                }
                for &(i, support2) in &active_orbitals {
                    let h = half[i];
                    if h == 0. {
                        continue;
                    }
                    let orbital = &integrals.basis.orbitals[i];
                    let r2 = (point[0] - orbital.center[0]).powi(2)
                        + (point[1] - orbital.center[1]).powi(2)
                        + (point[2] - orbital.center[2]).powi(2);
                    if r2 > support2 {
                        continue;
                    }
                    let gradient = orbital.gradient(point);
                    for axis in 0..3 {
                        let value = 2. * g * h * gradient[axis];
                        local_force[3 * a + axis] += value;
                        local_force[3 * orbital.atom + axis] -= value;
                    }
                }
                Ok(())
            },
        )?;
        Ok(local_force)
    };

    let workers = integrals.context.threads().max(1).min(n_atoms.max(1));
    if workers <= 1 {
        let mut total = vec![0.; 3 * positions.len()];
        for a in 0..n_atoms {
            let part = compute_atom(a)?;
            for (tot, val) in total.iter_mut().zip(part) {
                *tot += val;
            }
        }
        return Ok(total);
    }

    let mut total = vec![0.; 3 * positions.len()];
    integrals.context.run(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n_atoms)
                .map(|a| {
                    mandacaru_kernels::runtime::spawn(scope, move || compute_atom(a))
                })
                .collect();
            for handle in handles {
                let part = handle.join().unwrap().unwrap();
                for (tot, val) in total.iter_mut().zip(part) {
                    *tot += val;
                }
            }
        });
    });
    Ok(total)
}

/// The force from moving the long-range ionic potential of each atom against
/// the valence density `rho`.
pub(super) fn external_force(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    rho: &[f64],
) -> Vec<f64> {
    let positions = integrals.geometry.positions();
    let dv = integrals.grid.index.dv();
    let sigma = sigma(integrals);
    let mut force = vec![0.; 3 * positions.len()];
    for (k, &density) in rho.iter().enumerate() {
        if density == 0. {
            continue;
        }
        let point = integrals.grid.point(k);
        for (a, &s) in setup.atoms.iter().enumerate() {
            let delta: [f64; 3] = std::array::from_fn(|i| point[i] - positions[a][i]);
            let d = distance(delta, [0.; 3]);
            if d == 0. {
                continue;
            }
            let slope = long_range_slope(setup.species[s].charge, sigma, d);
            for axis in 0..3 {
                force[3 * a + axis] -= dv * density * slope * delta[axis] / d;
            }
        }
    }
    force
}

/// The force from moving the partial core density of each atom against what
/// it feels: `potential` (`n` values, half the core in each spin already
/// folded in) and, for a meta-GGA, the kinetic-energy-density potential `tau`
/// against the core's kinetic energy density.
pub(super) fn core_force(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    potential: &[f64],
    tau: Option<&[f64]>,
) -> Vec<f64> {
    let positions = integrals.geometry.positions();
    let dv = integrals.grid.index.dv();
    let shape = integrals.grid.index.shape();
    let [_, ny, nz] = shape;
    let inv_step = integrals.grid.step.try_inverse().unwrap_or(nalgebra::Matrix3::identity());
    let row_norms = [
        (inv_step[(0, 0)].powi(2) + inv_step[(0, 1)].powi(2) + inv_step[(0, 2)].powi(2)).sqrt(),
        (inv_step[(1, 0)].powi(2) + inv_step[(1, 1)].powi(2) + inv_step[(1, 2)].powi(2)).sqrt(),
        (inv_step[(2, 0)].powi(2) + inv_step[(2, 1)].powi(2) + inv_step[(2, 2)].powi(2)).sqrt(),
    ];
    let mut force = vec![0.; 3 * positions.len()];
    for (a, &s) in setup.atoms.iter().enumerate() {
        let center = positions[a];
        let r_core = *setup.species[s].xc_core.knots.as_slice().last().unwrap_or(&0.0);
        let r_tau = if tau.is_some() {
            *setup.species[s].xc_core_tau.knots.as_slice().last().unwrap_or(&0.0)
        } else {
            0.0
        };
        let r_cut = r_core.max(r_tau);
        if r_cut <= 0. {
            continue;
        }
        let c_center = inv_step * (nalgebra::Vector3::from(center) - nalgebra::Vector3::from(integrals.grid.origin));
        let mut min_idx = [0; 3];
        let mut max_idx = [0; 3];
        let mut inside = true;
        for d in 0..3 {
            let delta = r_cut * row_norms[d];
            let mn = ((c_center[d] - delta).floor() as isize).max(0) as usize;
            let mx = (c_center[d] + delta).ceil() as isize;
            if mx < 0 {
                inside = false;
                break;
            }
            let mx = (mx as usize).min(shape[d] - 1);
            if mn > mx {
                inside = false;
                break;
            }
            min_idx[d] = mn;
            max_idx[d] = mx;
        }
        if !inside {
            continue;
        }
        for ix in min_idx[0]..=max_idx[0] {
            for iy in min_idx[1]..=max_idx[1] {
                let base = (ix * ny + iy) * nz;
                for iz in min_idx[2]..=max_idx[2] {
                    let k = base + iz;
                    let point = integrals.grid.point(k);
                    let delta: [f64; 3] = std::array::from_fn(|i| point[i] - center[i]);
                    let d = distance(delta, [0.; 3]);
                    if d == 0. || d > r_cut {
                        continue;
                    }
                    let (_, slope) = core_slope(&setup.species[s].xc_core, d);
                    let tau_slope = tau.map_or(0., |_| core_slope(&setup.species[s].xc_core_tau, d).1);
                    let weight = potential[k] * slope + tau.map_or(0., |t| t[k]) * tau_slope;
                    if weight == 0. {
                        continue;
                    }
                    for axis in 0..3 {
                        force[3 * a + axis] -= dv * weight * delta[axis] / d;
                    }
                }
            }
        }
    }
    force
}
