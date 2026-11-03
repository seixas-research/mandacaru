//! The oracle of the spin-orbit term (SO-1): `E_SO(eps) = sum_k w Re tr[D_k h^SO_k(eps)]` rebuilt
//! at the strained cell from the setup's own spin-orbit builder, with the frozen spinor density
//! `D_k` held fixed. `h^SO` is [`matrices`] on the strained shells, union projectors, lattice and
//! Cartesian k (fractional k fixed), the rigid quadrature spheres about the carried atoms, and the
//! radial couplings of the dataset times a scale. Written from the setup's public builders and
//! nothing of the strain code. Included by `#[path]` where it is used (it needs
//! `strain_samples_support` and `setup_pulay_support` in the crate root).
#![allow(dead_code)]

use crate::{periodic_gradient_spinor_support::Spinor, strain_samples_support::Carried};

type Grid = [[f64; 3]; 3];
use mandacaru_core::periodic::{gradient::Matrix, setup::spin_orbit::matrices};
use nalgebra::Matrix3;

/// `sum_k w_k Re tr[D_k h^SO_k(eps)]` for the k-points `ks` of the crystal with weights `weights`,
/// the couplings scaled by `scale`.
pub fn energy(
    sp: &Spinor,
    ks: &[usize],
    weights: &[f64],
    densities: &[Matrix],
    scale: f64,
    eps: &Grid,
) -> f64 {
    let data = &sp.data;
    let ctx = &data.crystal.context;
    let mut couplings = sp.couplings.clone();
    for c in couplings.values_mut() {
        c.values.iter_mut().for_each(|v| *v *= scale);
    }
    let m2 = 2 * data.crystal.m;
    let mut total = 0.;
    for ((&k, &w), d) in ks.iter().zip(weights).zip(densities) {
        let carried = Carried::new(
            &data.lattice,
            &data.shells,
            &sp.projectors,
            &Matrix3::from_fn(|a, b| eps[a][b]),
            data.kpoints[k],
        );
        let h = matrices(
            ctx,
            &carried.shells,
            &carried.lattice,
            &[carried.k],
            &carried.projectors,
            &couplings,
        )
        .unwrap();
        let h = h.as_slice();
        let mut trace = 0.;
        for i in 0..m2 {
            for j in 0..m2 {
                trace += (d[(i, j)] * h[j * m2 + i]).re;
            }
        }
        total += w * trace;
    }
    total
}

fn entry(a: usize, b: usize, delta: f64) -> Grid {
    let mut eps = [[0.; 3]; 3];
    eps[a][b] = delta;
    eps
}

/// `dE/d eps_ab` by the central difference of `f` (one entry at a time), Richardson over `step`
/// and `2 step`.
pub fn derivative(a: usize, b: usize, step: f64, f: impl Fn(&Grid) -> f64) -> f64 {
    let c = |h: f64| (f(&entry(a, b, h)) - f(&entry(a, b, -h))) / (2. * h);
    let (c1, c2) = (c(step), c(2. * step));
    c1 + (c1 - c2) / 3.
}
