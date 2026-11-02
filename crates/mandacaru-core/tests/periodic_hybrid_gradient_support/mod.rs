//! Shared by the `periodic_hybrid_gradient*` tests: central differences of the REAL energy
//! entries of the hybrid exchange under motions that need no displaced `Crystal` (the
//! compensation centers moved through `Sites`, the stored projections perturbed), on the
//! phase-5 fixtures of `periodic_screened_augmentation_support`.
#![allow(dead_code)]

use crate::periodic_screened_augmentation_support::{Geometry, Matrix, Rig};
use mandacaru_core::periodic::{
    screened_augmentation::Augmentation,
    screened_exchange::{exchange_operators_with, ExxDivTreatment, Screening},
    setup::shapes::Atom,
};
use mandacaru_core::{
    math::storage::ComplexSamples,
    periodic::{
        hybrid_gradient::{explicit, Gradient, Motion},
        Crystal,
    },
    quantum::C64,
    Buffer, Result,
};
use num_complex::Complex64;

/// The Richardson central difference `f'(0)` from `f(+-step)` and `f(+-2 step)` (error
/// `O(step^4)`), and its measured step floor: the distance to the same difference at `step/2`,
/// which holds the truncation error of the step and the round-off of the energy together.
pub fn central(step: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let (a, b) = (f(step) - f(-step), f(2. * step) - f(-2. * step));
    let rich = (8. * a - b) / (12. * step);
    let half = f(step / 2.) - f(-step / 2.);
    let finer = (8. * half - a) / (12. * (step / 2.));
    (rich, (rich - finer).abs())
}

/// The gates of the positive tests, each about 100 x the worst measured step floor of its
/// comparison (phase 6.4e-13, response 1.4e-12, function 4.1e-13; the log, HX-1 results); the negative controls must miss them by more than 100 x.
pub const PHASE_GATE: f64 = 6e-11;
pub const RESPONSE_GATE: f64 = 1.5e-10;
/// The synthetic motion of the Bloch samples.
pub const FUNCTION_GATE: f64 = 4e-11;
/// Atom 1 of the Si fixture is moved off the diamond site so the forces are not zero by
/// symmetry.
pub const OFF_SITE: [f64; 3] = [0.21, -0.13, 0.17];

/// `geometry` with `atom` moved by `delta` along `axis`.
pub fn moved(g: &Geometry, atom: usize, axis: usize, delta: f64) -> Geometry {
    let mut atoms: Vec<Atom> = g.atoms.clone();
    atoms[atom].center[axis] += delta;
    Geometry {
        atoms,
        channels: g.channels.clone(),
        lattice: g.lattice,
    }
}

pub fn screening(rig: &Rig) -> Screening {
    Screening::of(&rig.crystal).unwrap()
}

/// The grid-plus-compensation exchange energy with the compensation centers of `geometry`,
/// through the production entry `exchange_operators_with` (no one-center model attached).
pub fn compensation_energy(rig: &Rig, geometry: &Geometry, exxdiv: ExxDivTreatment) -> f64 {
    let aug = Augmentation::new(&rig.crystal, geometry.sites(), &rig.shapes).unwrap();
    exchange_operators_with(&rig.crystal, &rig.d, screening(rig), exxdiv, Some(&aug))
        .unwrap()
        .energy
}

/// The explicit terms of the rig with the compensation centers of `geometry`, no one-center
/// model.
pub fn gradient(rig: &Rig, geometry: &Geometry, exxdiv: ExxDivTreatment) -> Gradient {
    let aug = Augmentation::new(&rig.crystal, geometry.sites(), &rig.shapes).unwrap();
    explicit(&rig.crystal, &rig.d, screening(rig), exxdiv, &aug, None).unwrap()
}

/// The explicit terms of the crystal's own spheres (`Crystal::hybrid`: compensation charges AND
/// the one-center model), and the production energy `Crystal::hybrid_exchange` to difference.
pub fn full_gradient(rig: &Rig, exxdiv: ExxDivTreatment) -> Gradient {
    let aug = rig
        .crystal
        .hybrid
        .as_ref()
        .unwrap()
        .augmentation(&rig.crystal)
        .unwrap();
    explicit(&rig.crystal, &rig.d, screening(rig), exxdiv, &aug, None).unwrap()
}
pub fn full_energy(rig: &Rig, exxdiv: ExxDivTreatment) -> f64 {
    rig.crystal.hybrid_exchange(&rig.d, exxdiv).unwrap().energy
}

/// `delta` perturbations of the stored projections (`m x p` per point), applied as
/// `base + eps delta`.
pub fn perturb(rig: &mut Rig, base: &[Matrix], delta: &[Matrix], eps: f64) {
    for (k, point) in rig.crystal.points.iter_mut().enumerate() {
        point.projections = &base[k] + &delta[k] * Complex64::new(eps, 0.);
    }
}

/// `2 Re sum_k tr[delta_k Z_k]` over every (spin, k) of the response `z`.
pub fn chain(delta: &[Matrix], z: &[Matrix]) -> f64 {
    z.iter()
        .enumerate()
        .map(|(at, z)| 2. * (&delta[at % delta.len()] * z).trace().re)
        .sum()
}

/// A reproducible dense complex matrix with entries of order one.
pub fn pseudo_random(rows: usize, columns: usize, seed: f64) -> Matrix {
    Matrix::from_fn(rows, columns, |i, j| {
        let x = (i * 13 + j * 5) as f64 * 0.731 + seed;
        Complex64::new((x * 1.3).sin(), (x * 0.7 + 0.4).cos())
    })
}

/// A synthetic motion of the Bloch samples: `chi_{mu k}(r) += eps s_{mu k}(r)` for the functions
/// owned by one atom, with `s` pseudo-random, so the energy slope of the samples is the analytic
/// `functions` term along axis 0 (axes 1 and 2 are zero). Defined at every k, complex included.
pub struct Perturbation {
    pub owner: Vec<usize>,
}

impl Perturbation {
    pub fn value(k: usize, mu: usize, r: usize) -> C64 {
        let x = (k * 31 + mu * 17 + r) as f64 * 0.37 + 0.2;
        C64::new((x * 1.3).sin(), (x * 0.7 + 0.4).cos()) * 0.3
    }
}

impl Motion for Perturbation {
    fn owner(&self) -> &[usize] {
        &self.owner
    }
    fn center_derivative(&self, c: &Crystal, k: usize) -> Result<[Buffer<C64>; 3]> {
        let n = c.grid.index.points();
        let mut out = [
            c.context.allocate::<C64>(c.m * n)?,
            c.context.allocate::<C64>(c.m * n)?,
            c.context.allocate::<C64>(c.m * n)?,
        ];
        for mu in 0..c.m {
            for r in 0..n {
                out[0].as_mut_slice()[mu * n + r] = Perturbation::value(k, mu, r);
            }
        }
        Ok(out)
    }
}

/// `base + eps s` on the samples of the functions of `atom`, at every point of the rig's crystal.
pub fn perturb_samples(rig: &mut Rig, base: &[Vec<C64>], owner: &[usize], atom: usize, eps: f64) {
    let (m, n) = (rig.crystal.m, rig.crystal.grid.index.points());
    for (k, values) in base.iter().enumerate() {
        let mut v = values.clone();
        for mu in (0..m).filter(|&mu| owner[mu] == atom) {
            for r in 0..n {
                v[mu * n + r] += Perturbation::value(k, mu, r) * eps;
            }
        }
        rig.crystal.points[k].psi =
            ComplexSamples::Float64(rig.crystal.context.copy_slice(&v).unwrap());
    }
}
