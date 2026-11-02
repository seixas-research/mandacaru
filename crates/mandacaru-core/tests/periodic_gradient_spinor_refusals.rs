//! CS-3 refusals and ties: what the spinor force does not differentiate raises with its cause, and
//! the spin-orbit data of the provider must be the crystal's own.
//!
//! * a spinor hybrid (stage 3), a meta-GGA, rVV10 and float32 data raise with the cause named;
//! * a spinor request whose provider carries no spin-orbit projectors raises, never a zero term;
//!   a scalar provider refuses spin-orbit data, and a scalar layout tied to a spinor crystal (or
//!   the reverse) is refused;
//! * the spin-orbit matrix of the force's projections and couplings must equal the matrix the
//!   crystal diagonalizes with at every k-point: a coupling scaled by 1.5 or by 0 is refused.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::{
    math::storage::Precision,
    periodic::{
        gradient::{gradient, scf},
        setup::provider::SetupDerivatives,
        Crystal,
    },
    Error, Result,
};
use periodic_gradient_crystals_support::inputs;
use periodic_gradient_spinor_support::Spinor;

fn cause<T>(r: Result<T>) -> String {
    match r {
        Err(Error::Unsupported(m)) | Err(Error::Invalid(m)) => m,
        Err(e) => panic!("{e:?}"),
        Ok(_) => panic!("expected a refusal"),
    }
}

fn refused_by_the_crystal(crystal: &mut Crystal) {
    scf::refuse(crystal).expect("a spinor LDA crystal is supported");
    let original = crystal.functional.clone();
    for (functional, word) in [
        ("hse06", "spinor hybrid"),
        ("r2scan", "meta-GGA"),
        ("pbe+rvv10", "rVV10"),
    ] {
        crystal.functional = functional.to_string();
        let message = cause(scf::refuse(crystal));
        assert!(message.contains(word), "{functional}: {message}");
    }
    crystal.functional = original;
    crystal.precision = Precision::Float32;
    assert!(cause(scf::refuse(crystal)).contains("float32"));
    crystal.precision = Precision::Float64;
    scf::refuse(crystal).expect("restored");
}

#[test]
fn what_a_spinor_does_not_differentiate_raises_with_the_cause() {
    let mut spinor = Spinor::load("si_ldafr");
    refused_by_the_crystal(&mut spinor.data.crystal);
}

#[test]
fn the_spin_orbit_data_of_the_provider_is_the_crystals_own() {
    let spinor = Spinor::load("si_ldafr");
    let cores = spinor.data.cores();
    let tied = || {
        SetupDerivatives::new(
            &spinor.data.crystal.context,
            inputs(&spinor.data, &cores),
            spinor.data.layout(),
            spinor.data.operators(),
        )
        .unwrap()
        .with_scf(&spinor.data.crystal, spinor.data.options.clone())
        .unwrap()
    };
    tied()
        .with_spin_orbit(&spinor.projectors, &spinor.couplings)
        .expect("the dataset's own couplings reproduce the crystal's h^SO");
    for (scale, name) in [(1.5, "scaled by 1.5"), (0.0, "zeroed")] {
        let mut couplings = spinor.couplings.clone();
        for c in couplings.values_mut() {
            c.values.iter_mut().for_each(|v| *v *= scale);
        }
        let message = cause(
            tied()
                .with_spin_orbit(&spinor.projectors, &couplings)
                .map(|_| ()),
        );
        eprintln!("couplings {name}: {message}");
        assert!(message.contains("spin-orbit matrix"), "{name}: {message}");
    }
    // The tie holds in the other order too: the spin-orbit data first, the crystal second.
    let first = || {
        SetupDerivatives::new(
            &spinor.data.crystal.context,
            inputs(&spinor.data, &cores),
            spinor.data.layout(),
            spinor.data.operators(),
        )
        .unwrap()
    };
    first()
        .with_spin_orbit(&spinor.projectors, &spinor.couplings)
        .unwrap()
        .with_scf(&spinor.data.crystal, spinor.data.options.clone())
        .expect("the right couplings tie in either order");
    let mut scaled = spinor.couplings.clone();
    scaled
        .values_mut()
        .for_each(|c| c.values.iter_mut().for_each(|v| *v *= 1.5));
    let message = cause(
        first()
            .with_spin_orbit(&spinor.projectors, &scaled)
            .unwrap()
            .with_scf(&spinor.data.crystal, spinor.data.options.clone())
            .map(|_| ()),
    );
    assert!(message.contains("spin-orbit matrix"), "{message}");
    // A scalar provider takes no spin-orbit data; a scalar layout does not tie to a spinor
    // crystal.
    let mut layout = spinor.data.layout();
    layout.request.spinor = false;
    let scalar = SetupDerivatives::new(
        &spinor.data.crystal.context,
        inputs(&spinor.data, &cores),
        layout,
        spinor.data.operators(),
    )
    .unwrap();
    let message = cause(
        scalar
            .with_scf(&spinor.data.crystal, spinor.data.options.clone())
            .map(|_| ()),
    );
    assert!(message.contains("disagree"), "{message}");
    let scalar = SetupDerivatives::new(
        &spinor.data.crystal.context,
        inputs(&spinor.data, &cores),
        {
            let mut l = spinor.data.layout();
            l.request.spinor = false;
            l
        },
        spinor.data.operators(),
    )
    .unwrap();
    let message = cause(
        scalar
            .with_spin_orbit(&spinor.projectors, &spinor.couplings)
            .map(|_| ()),
    );
    assert!(message.contains("spinor request"), "{message}");
}

#[test]
fn a_spinor_provider_without_spin_orbit_projectors_raises_instead_of_a_zero_term() {
    let twin = Spinor::zero_soc_twin("lih");
    let result = twin.data.solve();
    let cores = twin.data.cores();
    let bare = periodic_gradient_crystals_support::provider(&twin.data, &cores);
    let err = gradient(&bare, &twin.data.state(&result), None)
        .err()
        .unwrap();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("spin-orbit projectors")),
        "{err:?}"
    );
    // With the (zero-coupling) projectors the same state is differentiated.
    gradient(&twin.provider(&cores), &twin.data.state(&result), None)
        .expect("the zero-coupling twin is differentiated");
}
