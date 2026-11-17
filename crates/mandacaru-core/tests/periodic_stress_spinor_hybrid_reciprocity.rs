//! AS-10: is the discretized screened exchange a SYMMETRIC bilinear form of the density
//! matrices, `sum_k w_k tr(D1_k K_k[D2]) = sum_k w_k tr(D2_k K_k[D1])`? The SCF diagonalizes with
//! `-a K[D]`, which is the functional derivative of `E_x` only if it is. The smooth pair densities
//! give a symmetric form to rounding. The compensation charges once broke it, by a relative 1e-6
//! (LiH at Gamma), 2e-9 (Si, 2x2x2 time-reversal-invariant points) and 4e-8 (Si, 3x1x1, complex
//! pair wave vectors), because the shapes on a face of the FFT box (the frequency `N/2`, its own
//! negative) were read at one of its two aliased wave vectors; they are now the mean of the
//! aliases the grid's lattice operations pick, both faces among them (`Augmentation::pair_shapes`),
//! and the form is symmetric to rounding with them as well. The
//! one-sided control, which breaks it, is in `periodic_screened_augmentation_pairs.rs`.
//!
//! Hand-built densities (no SCF), pair part only (no one-center model: the model is not
//! bilinear). Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_hybrid_support;
mod periodic_stress_spinor_hybrid_mesh_support;
mod periodic_stress_spinor_hybrid_support;

use mandacaru_core::{
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{exchange_operators_with, ExxDivTreatment},
        setup::shapes::Sites,
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use periodic_stress_hybrid_support::Rig;
use periodic_stress_spinor_hybrid_mesh_support::mesh_rig;
use periodic_stress_spinor_hybrid_support::rig;

fn density_of(m2: usize, k: usize, salt: usize, real: bool) -> DMatrix<C64> {
    let d = periodic_stress_hybrid_support::density(m2, k + 17 * salt);
    if real {
        // The real part of a rank-4 positive semidefinite matrix is real symmetric and positive
        // semidefinite too (`Re D = (D + conj D) / 2`).
        d.map(|x| C64::new(x.re, 0.))
    } else {
        d
    }
}

/// `(sum_k w_k tr(D1_k K_k[D2]), sum_k w_k tr(D2_k K_k[D1]))` of the pair part.
fn cross(rig: &Rig, augmented: bool, real: bool) -> (f64, f64, f64) {
    let data = &rig.data;
    let sites = Sites {
        atoms: &data.atoms,
        channels: &data.channels,
        lattice: data.lattice,
        sigma: data.sigma,
    };
    let aug = Augmentation::new(&data.crystal, sites, &data.shapes).unwrap();
    let c = &data.crystal;
    let m2 = if c.spinor { 2 * c.m } else { c.m };
    let d1: Vec<_> = (0..c.points.len())
        .map(|k| density_of(m2, k, 1, real))
        .collect();
    let d2: Vec<_> = (0..c.points.len())
        .map(|k| density_of(m2, k, 2, real))
        .collect();
    let k = |d: &Vec<DMatrix<C64>>| {
        exchange_operators_with(c, d, rig.screening, rig.exxdiv, augmented.then_some(&aug))
            .unwrap()
            .operators
    };
    let (k1, k2) = (k(&d1), k(&d2));
    let trace = |d: &[DMatrix<C64>], op: &[DMatrix<C64>]| -> f64 {
        c.points
            .iter()
            .zip(d.iter().zip(op))
            .map(|(p, (d, op))| p.weight * (d * op).trace().re)
            .sum()
    };
    let (a, b) = (trace(&d1, &k2), trace(&d2, &k1));
    (a, b, (a - b).abs())
}

#[test]
fn the_exchange_is_a_symmetric_bilinear_form_with_the_compensation_charges_too() {
    for (name, rig) in [
        (
            "Gamma only (spinor)",
            rig("lih_hse06", ExxDivTreatment::None, false),
        ),
        (
            "Si 2x2x2 (scalar, time-reversal-invariant points)",
            Rig::new("si_hse06", ExxDivTreatment::None),
        ),
        (
            "Si 3x1x1 (spinor, complex pair wave vectors)",
            mesh_rig("si_hsefr_tr3", ExxDivTreatment::None, false),
        ),
    ] {
        for (augmented, real) in [(false, false), (true, false), (true, true)] {
            let (a, b, gap) = cross(&rig, augmented, real);
            println!(
                "{name}, {} {} densities: {a:.12e} / {b:.12e}, gap {gap:.2e} (relative {:.2e})",
                if augmented {
                    "with compensation,"
                } else {
                    "grid pair alone,"
                },
                if real { "real" } else { "complex" },
                gap / a.abs()
            );
            assert!(a.abs() > 1e-2 && gap < 1e-12 * a.abs(), "{name}: {gap:e}");
        }
    }
}
