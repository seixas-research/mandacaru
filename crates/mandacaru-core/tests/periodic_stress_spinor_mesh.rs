//! AS-8, Kramers / time reversal: the stress of a time-reversal-symmetric spinor on the mesh
//! HALVED by time reversal equals the stress on the whole mesh, with no symmetrizer.
//!
//! The cell is the displaced `lda-fr` Si of `si_ldafr` on a 3x1x1 Gamma-centered mesh where `+k` and
//! `-k` differ (0 and +-1/3): the natural build keeps two points (weights 2/3 and 1/3), the
//! `si_ldafr_tr3_full` snapshot keeps all three. Both are solved and differentiated; the sixteen
//! terms and the sum must agree (every contribution is `Re tr[..]` and the Bloch sums obey
//! `chi(-k) = conj chi(k)`, so the half mesh is exact).
//!
//! Needs `MANDACARU_PAW_PATH`; two stresses of a 2- and a 3-point cell: run with
//! `cargo test --release -p mandacaru-core --test periodic_stress_spinor_mesh -- --ignored --test-threads=1`
//! if the machine is loaded (each takes minutes).
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_spinor_support;

use periodic_gradient_spinor_support::Spinor;
use periodic_stress_spinor_support::{miss, peak, stress_at};

/// The half-mesh and whole-mesh stresses are the same functional; the SCF tolerance is 1e-11.
const GATE: f64 = 1e-9;

fn run(case: &'static str) -> (usize, mandacaru_core::periodic::stress_terms::Stress, f64) {
    let spinor = Spinor::load(case);
    let result = spinor.data.solve();
    let stress = stress_at(&spinor, &spinor.data.state(&result), &spinor.data.options);
    (spinor.data.crystal.points.len(), stress, result.free_energy)
}

#[test]
#[ignore = "a 2- and a 3-point spinor stress: minutes; see the file header for the run command"]
fn the_half_mesh_stress_is_the_whole_mesh_stress() {
    let (nh, half, eh) = run("si_ldafr_tr3");
    let (nf, full, ef) = run("si_ldafr_tr3_full");
    assert_eq!(
        (nh, nf),
        (2, 3),
        "time reversal halves 0, +1/3, -1/3 to 0, 1/3"
    );
    println!(
        "free energies: half {eh:.12}, whole {ef:.12} ({:.1e})",
        eh - ef
    );
    let mut worst = 0f64;
    for ((name, a), (_, b)) in half
        .breakdown
        .terms()
        .iter()
        .zip(full.breakdown.terms().iter())
    {
        let relative = miss(a, b) / peak(b).max(1e-6);
        println!(
            "{name:12} peak {:.3e}  half - whole {:.2e}",
            peak(b),
            miss(a, b)
        );
        worst = worst.max(relative);
    }
    let total = miss(&half.gradient, &full.gradient) / peak(&full.gradient);
    println!("sum: relative gap {total:.2e}; worst term {worst:.2e}");
    assert!(
        worst < GATE && total < GATE,
        "half {total:.2e} worst {worst:.2e}"
    );
    // The guard: +k and -k are different points of the whole mesh and the spin-orbit term is live,
    // else the comparison would prove nothing.
    assert!(peak(&full.breakdown.spin_orbit) > 1e-8);
}
