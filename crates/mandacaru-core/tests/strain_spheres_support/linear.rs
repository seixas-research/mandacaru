//! The oracle side of `strain_spheres_linear.rs`: `short_range_ion_compensation` on a carried
//! cell and its Richardson central difference, the pinned image list (`members`), and the
//! library's `linear_strain` written out with switches for the own images and for the long-range
//! slope (the wrong variants the controls compare with the difference).
#![allow(dead_code)]

use super::local::Local;
use crate::setup_gradients_support::Cell;
use mandacaru_core::{
    math::angular,
    paw::{
        local_split::{angular_rule, radial_rule, short_range_potential_derivative},
        multipole::shape_function,
    },
    periodic::setup::{
        ions::{
            short_range_ion_compensation, LocalPotential, SPHERE_AZIMUTHAL, SPHERE_POLAR,
            SPHERE_RADIAL,
        },
        shapes::{visit_lattice, Sites},
    },
    Context,
};
use nalgebra::Vector3;
use num_complex::Complex64;

/// `short_range_ion_compensation` of the cell, per channel.
pub fn linear_at(ctx: &Context, local: &Local, cell: &Cell) -> Vec<Complex64> {
    short_range_ion_compensation(ctx, &cell.sites(), &local.potentials).unwrap()
}

/// The central difference of [`linear_at`] along `eps_ab` with a Richardson pair (`step`, `2 step`).
pub fn linear_derivative(
    ctx: &Context,
    local: &Local,
    (a, b): (usize, usize),
    step: f64,
) -> Vec<Complex64> {
    let at = |e: f64| {
        let mut eps = [[0.; 3]; 3];
        eps[a][b] = e;
        linear_at(ctx, local, &local.cell.strained(eps))
    };
    let (p1, m1, p2, m2) = (at(step), at(-step), at(2. * step), at(-2. * step));
    (0..p1.len())
        .map(|i| {
            let one = (p1[i] - m1[i]) / (2. * step);
            let two = (p2[i] - m2[i]) / (4. * step);
            (one * 4. - two) / 3.
        })
        .collect()
}

/// The `(owner, other, integer triple)` of every image `short_range_ion_compensation` sums.
pub fn members(sites: &Sites, potentials: &[LocalPotential]) -> Vec<(usize, usize, [i64; 3])> {
    let inverse = sites.lattice.try_inverse().unwrap();
    let mut out = Vec::new();
    for (owner, a) in sites.atoms.iter().enumerate() {
        let radius = a.compensation_radius;
        let center = Vector3::from(a.center);
        for (other, species) in potentials.iter().enumerate() {
            let reach = species.reach(sites.sigma).unwrap() + radius;
            let d0 = center - Vector3::from(sites.atoms[other].center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let shift = Vector3::from(*image);
                        if (other == owner && shift.norm() < 1e-10) || (d0 - shift).norm() > reach {
                            continue;
                        }
                        let n = inverse * shift;
                        out.push((owner, other, [0, 1, 2].map(|i| n[i].round() as i64)));
                    }
                    Ok(())
                },
            )
            .unwrap();
        }
    }
    out
}

/// What [`linear_variant`] leaves out or gets wrong.
#[derive(Clone, Copy, Debug)]
pub struct Switch {
    /// The images `T != 0` of the sphere's own atom (the force routine skips all of them).
    pub own_images: bool,
    /// The slope of the long-range part subtracted from the potential slope.
    pub long_range: bool,
}

/// The library's `linear_strain` written out, with the switches of [`Switch`].
pub fn linear_variant(
    ctx: &Context,
    sites: &Sites,
    potentials: &[LocalPotential],
    switch: Switch,
) -> Vec<[[Complex64; 3]; 3]> {
    let sigma = sites.sigma;
    let reaches: Vec<f64> = potentials.iter().map(|p| p.reach(sigma).unwrap()).collect();
    let mut out = vec![[[Complex64::new(0., 0.); 3]; 3]; sites.channels.len()];
    let (axes, angular_weights) = angular_rule(SPHERE_POLAR, SPHERE_AZIMUTHAL).unwrap();
    let directions: Vec<[f64; 3]> = (0..angular_weights.len())
        .map(|j| [axes[0][j], axes[1][j], axes[2][j]])
        .collect();
    let _ = ctx;
    for owner in 0..sites.atoms.len() {
        let radius = sites.atoms[owner].compensation_radius;
        let (r, radial_weights) = radial_rule(radius, 0., SPHERE_RADIAL).unwrap();
        let points = r.len() * directions.len();
        let center = Vector3::from(sites.atoms[owner].center);
        let at = |p: usize| {
            center + Vector3::from(directions[p % directions.len()]) * r[p / directions.len()]
        };
        let mut tensor = vec![[0f64; 9]; points];
        for (other, species) in potentials.iter().enumerate() {
            let reach = reaches[other] + radius;
            let d0 = center - Vector3::from(sites.atoms[other].center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let shift = Vector3::from(*image);
                        if other == owner && (shift.norm() < 1e-10 || !switch.own_images) {
                            continue;
                        }
                        if (d0 - shift).norm() > reach {
                            continue;
                        }
                        let origin = Vector3::from(sites.atoms[other].center) + shift;
                        let delta = origin - center;
                        let distance: Vec<f64> =
                            (0..points).map(|p| (at(p) - origin).norm()).collect();
                        let sampled: Vec<f64> =
                            distance.iter().map(|&d| species.slope(d)).collect();
                        let slope = if switch.long_range {
                            short_range_potential_derivative(
                                &sampled,
                                species.charge,
                                sigma,
                                &distance,
                            )?
                        } else {
                            sampled
                        };
                        for p in 0..points {
                            let y = at(p) - origin;
                            for a in 0..3 {
                                for b in 0..3 {
                                    tensor[p][a * 3 + b] +=
                                        slope[p] / distance[p].max(1e-300) * y[a] * delta[b];
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
        }
        for row in (0..out.len()).filter(|&c| sites.channels[c].0 == owner) {
            let (_, l, m) = sites.channels[row];
            let shape = shape_function(&r, radius, l).unwrap();
            for (p, t) in tensor.iter().enumerate() {
                let (i, j) = (p / directions.len(), p % directions.len());
                let y = angular::complex(l, m, directions[j])
                    * (radial_weights[i] * angular_weights[j] * shape[i]);
                for a in 0..3 {
                    for b in 0..3 {
                        out[row][a][b] -= y * t[a * 3 + b];
                    }
                }
            }
        }
    }
    out
}
