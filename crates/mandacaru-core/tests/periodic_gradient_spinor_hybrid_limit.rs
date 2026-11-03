//! SH-F1 / SH-F2: the force of a HYBRID (HSE06) crystal with spin-orbit coupling, in the limit
//! where the answer is known exactly. A spinor eigenproblem without spin-orbit coupling is two
//! copies of the scalar one, so a scalar hybrid crystal rewritten as a time-reversal spinor with
//! a zero spin-orbit matrix must give
//!
//! * the same free energy and the same energy tie (`hybrid_state_energy` of the spinor density),
//! * the same force, term by term, as the scalar hybrid engine of A2b (the strongest gate: it
//!   exercises every row, owner and spin sum of the doubled pass, with the exchange response `Z`
//!   in the scalar frame).
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::{gradient, scf, Breakdown};
use periodic_gradient_crystals_support::{provider, Data};
use periodic_gradient_spinor_support::Spinor;

fn worst(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |i| (x[i] - y[i]).abs()))
        .fold(0f64, f64::max)
}

pub struct Pair {
    pub scalar: Breakdown,
    pub spinor: Breakdown,
    pub energy_gap: f64,
}

/// The scalar hybrid force of `case` and the force of its zero-coupling spinor twin.
pub fn pair(case: &'static str) -> Pair {
    let scalar = Data::load(case);
    let result = scalar.solve();
    let cores = scalar.cores();
    let reference = gradient(&provider(&scalar, &cores), &scalar.state(&result), None).unwrap();
    let twin = Spinor::zero_soc_twin(case);
    let twin_result = twin.data.solve();
    let twin_cores = twin.data.cores();
    let tie = scf::hybrid_state_energy(
        &twin.data.crystal,
        &twin_result.matrices,
        twin.data.options.exxdiv_treatment,
    )
    .unwrap()
        - twin_result.energy;
    eprintln!("{case} twin: hybrid_state_energy - SCF energy = {tie:e}");
    assert!(tie.abs() < 1e-12, "{tie:e}");
    let spinor = gradient(
        &twin.provider(&twin_cores),
        &twin.data.state(&twin_result),
        None,
    )
    .unwrap();
    Pair {
        scalar: reference,
        spinor,
        energy_gap: (twin_result.free_energy - result.free_energy).abs(),
    }
}

#[test]
fn the_zero_coupling_limit_of_the_hybrid_force_is_the_scalar_hybrid_force() {
    let p = pair("lih_hse06");
    let (s, t) = (&p.scalar, &p.spinor);
    let terms = [
        ("exchange_phase", &s.exchange_phase, &t.exchange_phase),
        (
            "exchange_functions",
            &s.exchange_functions,
            &t.exchange_functions,
        ),
        ("exchange_rows", &s.exchange_rows, &t.exchange_rows),
        ("exchange_columns", &s.exchange_columns, &t.exchange_columns),
        ("overlap", &s.overlap, &t.overlap),
        ("hats", &s.hats, &t.hats),
    ];
    for (name, a, b) in terms {
        let peak = a.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
        let gap = worst(a, b);
        eprintln!("LiH HSE06 twin {name}: peak {peak:.3e}, gap {gap:.2e} Ha/Bohr");
        assert!(gap < 1e-9, "{name}: {gap:e}");
    }
    let (fs, ft) = (s.forces(), t.forces());
    let gap = worst(&fs, &ft);
    let exchange: f64 = s
        .exchange_functions
        .iter()
        .flatten()
        .fold(0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "LiH HSE06 zero-coupling twin: energy gap {:e}, force gap {gap:e} Ha/Bohr (exchange functions term peak {exchange:.2e})",
        p.energy_gap
    );
    assert!(exchange > 1e-3, "the exchange term is not vacuous");
    assert!(p.energy_gap < 1e-9, "{:e}", p.energy_gap);
    assert!(gap < 1e-9, "{gap:e}");
}
