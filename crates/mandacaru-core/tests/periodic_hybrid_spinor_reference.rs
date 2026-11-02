//! The spinor exchange at complex k against an independent direct Fourier sum (SH-1a, gate g3 and
//! unknown U-1): `reference::exchange` of `periodic_screened_augmentation_support` (own Q
//! assignment, no FFT, the compensation charges from the raw definition), on the 3x1x1 mesh whose
//! points `0, +-1/3` include complex k, for a time-reversal-symmetric and for a polarized and
//! rotated ensemble. Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_spinor_support;
mod setup_support;

use periodic_hybrid_spinor_support::{rig::Rig, *};

const NAME: &str = "si_so_tr3.json";
const GATE: f64 = 1e-12;

#[test]
fn the_spinor_operator_equals_the_direct_fourier_sum_at_complex_k() {
    let c = hybrid(&context(), NAME, &|_| {});
    let rig = Rig::new(NAME, &c);
    let u = time_reversal(&pi_ao(&read(NAME)));
    let ensembles = [
        ("time-reversal symmetric", kramers_densities(&c, &u, 8)),
        (
            "polarized and rotated",
            rotated_polarized(&c, 5, 3, &su2(0.4)),
        ),
    ];
    for (what, d) in &ensembles {
        for exxdiv in [NONE, GB] {
            let got = rig.grid_and_compensation(&c, d, exxdiv);
            // k = 0 is a complex target relative to its sources (k_diff = +-1/3), k = 1 is Gamma.
            for target in [0, 1] {
                let want = rig.direct(&c, d, target, exxdiv);
                let miss = relative(&got[target], &want);
                let other = relative(&got[(target + 2) % 3], &want);
                println!("{what} {exxdiv:?} k{target} ({:?}): operator vs direct sum {miss:.2e}; the other target {other:.2e}",
                    c.points[target].fractional.map(|x| (x * 3.).round() / 3.));
                assert!(want.norm() > 1e-3, "the reference is vacuous");
                assert!(miss < GATE, "{what} {exxdiv:?} k{target}: {miss:e}");
                // The sum for another target is another operator: the comparison discriminates.
                assert!(
                    other > 100. * GATE,
                    "the wrong-target control does not bite ({other:e})"
                );
            }
        }
    }
}
