//! AS-6 refusals: what the crystal stress does not differentiate raises with the cause, taken from
//! the crystal itself (a hybrid, a spinor without its spin-orbit projectors, float32, a meta-GGA, rVV10, spin polarization, DFT+U, a
//! k mesh reduced by space-group symmetry), and a setup that is not the crystal's is refused (the
//! cell, the atom centers, the weights, the reference). No refusal reaches an SCF or a stress:
//! nothing is silently differentiated as another calculation. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use mandacaru_core::{
    math::storage::Precision,
    models::hubbard::Form,
    periodic::{
        gradient::State,
        hubbard::{Seed, Shell, Spec},
        stress_terms::crystal::crystal_stress,
    },
    Error,
};
use periodic_gradient_crystals_support::{inputs, Data};
use periodic_stress_crystals_support::load;

/// The error of a stress asked of `data` with no bands (every refusal precedes the freeze).
fn refused(data: &Data) -> Error {
    let cores = data.cores();
    crystal_stress(
        &data.crystal.context,
        inputs(data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        &data.options,
        &State { bands: Vec::new() },
        None,
    )
    .map(|s| s.volume)
    .expect_err("this crystal must be refused")
}

fn unsupported(data: &Data) -> String {
    match refused(data) {
        Error::Unsupported(message) => message,
        other => panic!("expected an Unsupported refusal, got {other:?}"),
    }
}

#[test]
fn what_is_not_differentiated_raises_with_the_cause() {
    let mut data = load("lih");
    let original = data.crystal.functional.clone();
    for (functional, word) in [
        ("hse06", "hybrid"),
        ("r2scan", "meta-GGA"),
        ("pbe+rvv10", "rVV10"),
    ] {
        data.crystal.functional = functional.to_string();
        let message = unsupported(&data);
        assert!(message.contains(word), "{functional}: {message}");
    }
    data.crystal.functional = original;
    // A spinor crystal is differentiated only with its spin-orbit projectors (AS-8); a spinor hybrid
    // (AS-10) is a hybrid like any other: a crystal without the PAW spheres of its sites block is
    // refused as one.
    data.crystal.spinor = true;
    assert!(unsupported(&data).contains("spin-orbit projectors"));
    data.crystal.functional = "hse06".to_string();
    assert!(unsupported(&data).contains("PAW spheres"));
    data.crystal.functional = "lda".to_string();
    data.crystal.spinor = false;
    data.crystal.precision = Precision::Float32;
    assert!(unsupported(&data).contains("float32"));
    data.crystal.precision = Precision::Float64;
    data.crystal.spins = 2;
    assert!(unsupported(&data).contains("spin-polarized"));
    data.crystal.spins = 1;
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![Shell {
            atom: 0,
            symbol: "Li".into(),
            l: 0,
            u: 0.1,
            j: 0.,
            projectors: vec![0],
            weights: vec![1.],
            basis: None,
            seed: Seed::Moment {
                n_electrons: 1.,
                moment: 0.,
            },
        }],
    });
    assert!(unsupported(&data).contains("DFT+U"));
    data.crystal.hubbard = None;
    data.crystal.permutations = vec![vec![0]];
    assert!(unsupported(&data).contains("space-group"));
    data.crystal.permutations.clear();
    data.crystal.qsym.as_mut_slice()[1] = num_complex::Complex64::new(0.5, 0.);
    assert!(unsupported(&data).contains("space-group"));
}

#[test]
fn a_setup_that_is_not_the_crystals_is_refused() {
    let data = load("lih");
    let cores = data.cores();
    let ctx = &data.crystal.context;
    let ask = |data: &Data, layout| {
        crystal_stress(
            ctx,
            inputs(data, &cores),
            layout,
            data.operators(),
            &data.crystal,
            &data.options,
            &State { bands: Vec::new() },
            None,
        )
        .map(|s| s.volume)
    };
    // The transposed lattice of a skew cell is another lattice, not the crystal's.
    let mut transposed = load("lih");
    transposed.lattice = data.lattice.transpose();
    assert!(
        matches!(ask(&transposed, data.layout()), Err(Error::Invalid(m)) if m.contains("lattice")),
        "a transposed lattice must be refused"
    );
    // A moved atom is not the crystal whose ion charges the SCF used.
    let mut moved = load("lih");
    moved.atoms[0].center[0] += 0.1;
    assert!(
        matches!(ask(&moved, data.layout()), Err(Error::Invalid(m)) if m.contains("ion")),
        "a moved atom must be refused"
    );
    let mut layout = data.layout();
    layout.weights[0] += 0.125;
    assert!(matches!(ask(&data, layout), Err(Error::Invalid(_))));
    let mut layout = data.layout();
    layout.reference += 0.01;
    assert!(matches!(ask(&data, layout), Err(Error::Invalid(_))));
    // A polarized layout over a restricted crystal.
    let mut layout = data.layout();
    layout.spins = 2;
    layout.degeneracy = 1.;
    assert!(matches!(ask(&data, layout), Err(Error::Unsupported(_))));
}
