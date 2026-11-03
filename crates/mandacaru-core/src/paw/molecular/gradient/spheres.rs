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
    let sigma = sigma(integrals);
    let mut force = vec![0.; 3 * positions.len()];
    let mut phi = vec![0.; m];
    let mut half = vec![0.; m];
    for (a, &s) in setup.atoms.iter().enumerate() {
        let species = &setup.species[s];
        let radius = local_sphere_radius(species, sigma);
        sphere(
            radius,
            species.local_radius,
            LOCAL[0],
            LOCAL[1],
            LOCAL[2],
            |offset, r, weight| {
                let g = weight * (species.local(r) - long_range(species.charge, sigma, r));
                let point = std::array::from_fn(|i| positions[a][i] + offset[i]);
                for (value, orbital) in phi.iter_mut().zip(&integrals.basis.orbitals) {
                    *value = orbital.evaluate(point);
                }
                half.fill(0.);
                for o in occupied {
                    let amplitude: f64 = o.ao.iter().zip(&phi).map(|(c, p)| c * p).sum();
                    for (h, c) in half.iter_mut().zip(&o.ao) {
                        *h += o.weight * c * amplitude;
                    }
                }
                for (i, orbital) in integrals.basis.orbitals.iter().enumerate() {
                    let gradient = orbital.gradient(point);
                    for axis in 0..3 {
                        let value = 2. * g * half[i] * gradient[axis];
                        force[3 * a + axis] += value;
                        force[3 * orbital.atom + axis] -= value;
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(force)
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
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let mut force = vec![0.; 3 * positions.len()];
    for k in 0..n {
        let point = integrals.grid.point(k);
        for (a, &s) in setup.atoms.iter().enumerate() {
            let delta: [f64; 3] = std::array::from_fn(|i| point[i] - positions[a][i]);
            let d = distance(delta, [0.; 3]);
            if d == 0. {
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
    force
}
