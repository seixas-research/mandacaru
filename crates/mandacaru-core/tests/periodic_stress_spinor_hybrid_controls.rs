//! AS-10: SU(2) covariance and the negative controls of the spinor exchange strain, on the
//! LiH box at Gamma (spinor twin flagged magnetic, hand-built `2m x 2m` densities with
//! populated cross-spin blocks), with `none` divergence treatment.
//!
//! * Covariance: the pair density is spin-summed and the strain is spatial, so every one of the
//!   eight exchange terms is invariant under `D -> (u (x) 1) D (u (x) 1)^+` (the rotation moves
//!   `D` and its cross blocks by more than a percent: asserted first). A rotation whose angle
//!   depends on the spatial function is not a symmetry and must change the terms.
//! * Wrong spin structures against the SAME central differences (all nine entries, 100 x the
//!   floor): the cross-spin blocks of `D` dropped (the cross blocks of `K` zeroed), and the
//!   sphere response from the spin-diagonal blocks of the one-center operator only.
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_hybrid_support;
mod periodic_stress_spinor_hybrid_support;

use mandacaru_core::{
    periodic::{
        screened_exchange::{ExxDivTreatment, Screening},
        stress_terms::hybrid::{Production, Tangents},
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use periodic_gradient_crystals_support::inputs;
use periodic_stress_hybrid_support::*;
use periodic_stress_spinor_hybrid_support::*;

const STEP: f64 = 2.5e-6;
/// The terms must agree to rounding under a spin rotation (relative to the peak of the term).
const COVARIANCE: f64 = 1e-9;

#[test]
fn the_exchange_strain_is_su2_covariant_and_a_function_dependent_rotation_is_not() {
    let mut r = rig("lih_hse06", ExxDivTreatment::None, false);
    let base = r.analytic(&Production);
    let original = r.d.clone();
    let u = su2(0.9, [0.48, -0.6, 0.64]);
    r.d = original.iter().map(|d| rotated(d, u)).collect();
    let moved: f64 = original
        .iter()
        .zip(&r.d)
        .map(|(a, b)| (a - b).norm() / a.norm())
        .fold(0., f64::max);
    println!("  the rotation moves D by {moved:.2e} of its norm");
    assert!(moved > 1e-2, "the rotation must move D");
    let turned = r.analytic(&Production);
    let mut worst = 0f64;
    for ((name, a), (_, b)) in base.parts().iter().zip(turned.parts().iter()) {
        let scale = peak(a);
        let gap = miss(a, b) / scale.max(1e-12);
        println!("  {name}: peak {scale:.3e}, rotated - original {gap:.2e} (relative)");
        assert!(scale > 1e-4, "{name} is live");
        assert!(gap < COVARIANCE, "{name}: {gap:e}");
        worst = worst.max(gap);
    }
    assert!((base.energy - turned.energy).abs() < 1e-9 * base.energy.abs());
    // The wrong transformation: the same family of matrices, the angle depending on the function.
    r.d = original
        .iter()
        .map(|d| rotated_by_function(d, 0.7, [0.48, -0.6, 0.64]))
        .collect();
    let wrong = r.analytic(&Production);
    let bite = miss(&wrong.total(), &base.total()) / (COVARIANCE * peak(&base.total()));
    println!("  control function-dependent rotation: bites {bite:.2e} gates (worst covariance {worst:.2e})");
    assert!(bite > 100., "{bite:e}");
}

#[test]
fn the_hand_built_densities_are_magnetic_so_the_remainder_of_the_spheres_is_live() {
    // A crystal flagged time-reversal symmetric refuses a density whose spheres carry a
    // magnetization above 1e-6 of their charge: the guard firing on these densities proves the
    // non-collinear remainder the magnetic flag evaluates is not zero here.
    let r = rig("lih_hse06", ExxDivTreatment::None, true);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        r.analytic(&Production);
    }));
    let message = refused
        .expect_err("a time-reversal-flagged crystal must refuse a magnetized density")
        .downcast::<String>()
        .map(|s| *s)
        .unwrap_or_default();
    println!("  guard: {message}");
    assert!(message.contains("magnetization"), "{message}");
}

#[test]
fn wrong_spin_structures_miss_the_difference_by_more_than_100_gates() {
    let mut r = rig("lih_hse06", ExxDivTreatment::None, false);
    let right = r.analytic(&Production);
    let (all, floor_all) = r.tensor(&ALL, STEP);
    let g = gate(floor_all);
    assert!(
        miss(&right.total(), &all) < g,
        "the right stress meets the gate"
    );

    // Cross-spin blocks of D dropped: the cross blocks of K are zero.
    let original = r.d.clone();
    r.d = original.iter().map(without_cross_blocks).collect();
    let blockdiag = r.analytic(&Production);
    r.d = original.clone();
    let bite = miss(&blockdiag.total(), &all) / g;
    println!("  control cross-spin blocks dropped: bites {bite:.2e} gates (gate {g:.1e})");
    assert!(bite > 100., "{bite:e}");

    // The sphere response from the spin-diagonal blocks of the one-center operator only: the
    // projection group is the oracle (compensation + sphere), the wrong sphere replaces H-6.
    let (projections, floor_p) = r.tensor(&PROJECTIONS, STEP);
    let gp = gate(floor_p);
    let c = &r.data.crystal;
    let (m, p) = (c.m, c.p);
    let a_fraction = Screening::of(c).unwrap().fraction;
    let model = c.hybrid.as_ref().unwrap().one_center();
    let mut projected = DMatrix::<C64>::zeros(2 * p, 2 * p);
    for (point, d) in c.points.iter().zip(&r.d) {
        for s in 0..2 {
            for t in 0..2 {
                let block = point.projections.adjoint()
                    * d.view((s * m, t * m), (m, m))
                    * &point.projections
                    * C64::new(point.weight, 0.);
                let mut view = projected.view_mut((s * p, t * p), (p, p));
                view += block;
            }
        }
    }
    let operator = model.spinor(&c.context, &projected).unwrap().operators[0].clone();
    let cross = operator.view((0, p), (p, p)).norm();
    println!(
        "  cross block of the one-center operator: {cross:.3e} (norm {:.3e})",
        operator.norm()
    );
    assert!(
        cross > 1e-3 * operator.norm(),
        "the cross blocks of O are live"
    );
    let cores = r.data.cores();
    let inp = inputs(&r.data, &cores);
    let tangents = Tangents::new(&c.context, &inp).unwrap();
    // The response of the blocks that the wrong variant drops: `Z_off = -a w sum_{s != t} ...`.
    let mut dropped = [[0f64; 3]; 3];
    let mut kept = [[0f64; 3]; 3];
    for (k, (point, d)) in c.points.iter().zip(&r.d).enumerate() {
        let dc = tangents.projection_strain(k).unwrap();
        for (target, wanted_equal) in [(&mut dropped, false), (&mut kept, true)] {
            let mut z = DMatrix::<C64>::zeros(p, m);
            for s in 0..2 {
                for t in 0..2 {
                    if (s == t) == wanted_equal {
                        z += operator.view((s * p, t * p), (p, p))
                            * point.projections.adjoint()
                            * d.view((t * m, s * m), (m, m));
                    }
                }
            }
            let z = z * C64::new(-a_fraction * point.weight, 0.);
            for a in 0..3 {
                for b in 0..3 {
                    let tr: C64 = (0..m)
                        .map(|i| (0..p).map(|j| dc[a][b][(i, j)] * z[(j, i)]).sum::<C64>())
                        .sum();
                    target[a][b] += 2. * tr.re;
                }
            }
        }
    }
    let full = sum(&[&dropped, &kept]);
    println!(
        "  test-side sphere response against production: {:.2e} (peak {:.3e})",
        miss(&full, &right.sphere),
        peak(&right.sphere)
    );
    assert!(miss(&full, &right.sphere) < 1e-12 + 1e-9 * peak(&right.sphere));
    let wrong_projections = sum(&[&right.compensation, &kept]);
    assert!(
        miss(&sum(&[&right.compensation, &right.sphere]), &projections) < gp,
        "the right projection group meets its gate"
    );
    let bite = miss(&wrong_projections, &projections) / gp;
    println!("  control spin-diagonal O only: bites {bite:.2e} gates (gate {gp:.1e})");
    assert!(bite > 100., "{bite:e}");
}
