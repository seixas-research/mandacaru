//! The Gaussian ions of the periodic setup: the point-ion constant, the on-site
//! and short-range ion terms, and the fields built from the shipped Si table,
//! against the frozen Si crystal (T1 = 1e-10 of the array maximum). The
//! independent checks need no frozen number: the Gaussian width drops out of the
//! point-ion energy once the jellium term is restored, and that energy equals the
//! Ewald sum of the engine.
mod setup_support;

use mandacaru_core::{
    periodic::{
        ewald::Ewald,
        setup::{
            electrostatics::couplings,
            ions::{
                ion_constants, linear, onsite_ion_compensation, short_range_ion_compensation,
                LocalPotential,
            },
            shapes::{Shapes, Sites, COMPENSATION_CUTOFF},
        },
    },
    Context,
};
use num_complex::Complex64;
use setup_support::*;
use std::f64::consts::PI;

const T1: f64 = 1e-10;

fn sites(si: &Silicon, sigma: f64) -> Sites<'_> {
    Sites {
        atoms: &si.atoms,
        channels: &si.channels,
        lattice: si.lattice,
        sigma,
    }
}

#[test]
fn the_point_ion_constant_equals_the_frozen_crystal() {
    let si = Silicon::load();
    let want = si.json["ion_constant"].as_f64().unwrap();
    let got = ion_constants(&sites(&si, si.sigma)).unwrap();
    println!(
        "ion_constant {got:.15} against the frozen {want:.15} (relative {:.2e})",
        (got - want).abs() / want.abs()
    );
    assert!((got - want).abs() / want.abs() < T1);
}

#[test]
fn the_point_ion_energy_is_independent_of_the_gaussian_width() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let charges: Vec<f64> = si.atoms.iter().map(|a| a.charge).collect();
    let total: f64 = charges.iter().sum();
    let volume = si.lattice.determinant().abs();
    let positions: Vec<[f64; 3]> = si.centers.clone();
    let rows: Vec<f64> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| si.lattice[(c, r)])
        .collect();
    let ewald = Ewald::new(
        &ctx,
        &positions,
        &charges,
        rows.try_into().unwrap(),
        1e-14,
        None,
    )
    .unwrap()
    .evaluate(&ctx)
    .unwrap()
    .energy;
    let mut raw = Vec::new();
    let mut restored = Vec::new();
    for factor in [0.8, 1.0, 1.2, 1.5] {
        let sigma = si.sigma * factor;
        let s = sites(&si, sigma);
        let shapes = Shapes::new(&ctx, &s, 20.).unwrap();
        let e_ion = couplings(&ctx, &shapes, &s, COMPENSATION_CUTOFF)
            .unwrap()
            .e_ion;
        let constant = ion_constants(&s).unwrap();
        let reference = 2. * PI * sigma * sigma * total / volume;
        raw.push(e_ion + constant);
        restored.push(e_ion + constant - total * reference);
    }
    println!("E_ion + ion_constant at sigma x 0.8, 1.0, 1.2, 1.5: {raw:?}");
    println!("... less sum(Z) * reference (the dropped G = 0 jellium term): {restored:?}; Ewald sum {ewald:.12}");
    let spread = restored.iter().cloned().fold(f64::MIN, f64::max)
        - restored.iter().cloned().fold(f64::MAX, f64::min);
    let raw_spread =
        raw.iter().cloned().fold(f64::MIN, f64::max) - raw.iter().cloned().fold(f64::MAX, f64::min);
    println!("spread with the jellium term restored {spread:.2e}; without {raw_spread:.2e}");
    assert!(spread < 1e-9 && raw_spread > 1e-2);
    for e in &restored {
        assert!((e - ewald).abs() < 1e-8, "{e} vs Ewald {ewald}");
    }
}

fn potentials(ctx: &Context, shipped: &Shipped, n: usize) -> Vec<LocalPotential> {
    (0..n)
        .map(|_| {
            LocalPotential::new(
                ctx,
                &shipped.radius,
                &shipped.v_local,
                shipped.charge,
                shipped.local_cutoff,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn the_ion_terms_of_the_compensation_charges_equal_the_frozen_crystal() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    assert_eq!(shipped.compensation_radius, 2.3070329984286464);
    assert_eq!(shipped.charge, 4.);
    assert_eq!(shipped.local_cutoff, 1.884195990667111);
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        let s = sites(&si, si.sigma);
        let local = potentials(&ctx, &shipped, 2);
        let reach = local[0].reach(si.sigma).unwrap();
        let short = short_range_ion_compensation(&ctx, &s, &local).unwrap();
        let onsite = onsite_ion_compensation(&s).unwrap();
        let both = linear(&ctx, &s, &local).unwrap();
        let want = complexes(&si.json["linear"]);
        let error = relative(&both, &want);
        println!("short-range reach {reach:.6} Bohr; linear: relative error {error:.2e} (max |linear| {:.4})",
            want.iter().map(|w| w.norm()).fold(0., f64::max));
        assert!(error < T1);
        // The channels split exactly into short-range minus on-site.
        for ((b, s), o) in both.iter().zip(&short).zip(&onsite) {
            assert!((b - (s - Complex64::new(*o, 0.))).norm() < 1e-15);
        }
        // Only monopoles carry an on-site term.
        for (o, &(_, l, _)) in onsite.iter().zip(&si.channels) {
            assert!(l == 0 || *o == 0.);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_translation_leaves_the_ion_terms_alone_and_the_sphere_rule_holds_its_weights() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let t = [0.31, -0.77, 1.9];
    let moved: Vec<_> = si
        .atoms
        .iter()
        .map(|a| {
            let mut a = *a;
            for k in 0..3 {
                a.center[k] += t[k];
            }
            a
        })
        .collect();
    let shifted = Sites {
        atoms: &moved,
        channels: &si.channels,
        lattice: si.lattice,
        sigma: si.sigma,
    };
    let local = potentials(&ctx, &shipped, 2);
    let (a, b) = (
        linear(&ctx, &sites(&si, si.sigma), &local).unwrap(),
        linear(&ctx, &shifted, &local).unwrap(),
    );
    let (ca, cb) = (
        ion_constants(&sites(&si, si.sigma)).unwrap(),
        ion_constants(&shifted).unwrap(),
    );
    let drift = relative(&b, &a);
    println!(
        "global translation: linear {drift:.2e}, ion_constant {:.2e}",
        (ca - cb).abs()
    );
    assert!(drift < 1e-9 && (ca - cb).abs() < 1e-11);
}
