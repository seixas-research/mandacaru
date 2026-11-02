//! The hook of the spin-orbit term in the force engine: `pulay::contract` calls
//! `SpinOrbit::add_k` once per k-point for a spinor request, with the spinor density of the frozen
//! state and the provider's spin-orbit projections, and the `Breakdown` carries the result in
//! `spin_orbit_rows` / `spin_orbit_columns`. The provider is the spinor toy in the doubled frame
//! (`periodic_gradient_spin_orbit_support/spinor.rs`), whose scalar part is empty, so the whole
//! engine total is the spin-orbit term and its oracle is the exact `dC_X` of the toy through
//! `models::spin_orbit::pair`. `gradient()` refuses a spinor HYBRID request (stage 3), through the
//! engine and through a hand-built state alike.
mod hubbard_gradient_support;
mod periodic_gradient_spin_orbit_support;

use hubbard_gradient_support::{Matrix, Toy};
use mandacaru_core::{
    periodic::gradient::{contract, gradient, spin_orbit::SpinOrbit, Breakdown},
    Error,
};
use periodic_gradient_spin_orbit_support::{spinor::*, *};

const GATE: f64 = 1e-10;
const CONTROL: f64 = 100. * GATE;
const LAMBDA: [f64; 2] = [0.31, 0.57];

fn total(b: &Breakdown) -> Vec<[f64; 3]> {
    b.spin_orbit_rows
        .iter()
        .zip(&b.spin_orbit_columns)
        .map(|(r, c)| std::array::from_fn(|a| r[a] + c[a]))
        .collect()
}

fn others_vanish(b: &Breakdown) {
    for (name, term) in b.terms() {
        if !name.starts_with("spin_orbit") {
            assert_eq!(peak(term), 0., "{name}");
        }
    }
}

#[test]
fn the_engine_total_of_a_spinor_state_is_the_spin_orbit_oracle() {
    let ctx = context();
    for (index, toy) in [Toy::two_site(), Toy::three_site()].iter().enumerate() {
        let (chs, so) = (channels(&ctx, toy, &LAMBDA), spin_orbit(&ctx, toy, &LAMBDA));
        let provider = SpinorToy::new(toy, so, true, true);
        let state = frozen(toy, 21 + index as u64);
        let d = state.state.bands[0][0].density();
        let want = oracle(&ctx, toy, &chs, &d);
        let breakdown = contract(&provider, &state).unwrap();
        let got = total(&breakdown);
        let error = largest(&got, &want);
        eprintln!(
            "toy {index}: |engine - oracle| = {error:e}, peak {:e}",
            peak(&want)
        );
        // The branch was taken: the oracle is large, and the engine carries both shares.
        assert!(peak(&want) > 1e-3 && peak(&breakdown.spin_orbit_rows) > 1e-3);
        assert!(peak(&breakdown.spin_orbit_columns) > 1e-3);
        assert!(error < GATE, "{error:e}");
        // The scalar part of this provider is empty: nothing else is in the total, and the
        // gradient() sum is the spin-orbit term.
        others_vanish(&breakdown);
        assert!(largest(&breakdown.gradient(), &want) < GATE);

        // The same provider and state under a scalar request: the branch is not taken.
        let scalar = SpinorToy::new(toy, spin_orbit(&ctx, toy, &LAMBDA), false, true);
        let off = contract(&scalar, &state).unwrap();
        assert_eq!(peak(&off.spin_orbit_rows), 0.);
        assert_eq!(peak(&off.spin_orbit_columns), 0.);

        // Negative controls through the engine, each against the oracle.
        assert!(largest(&breakdown.spin_orbit_rows, &want) > CONTROL);
        assert!(largest(&breakdown.spin_orbit_columns, &want) > CONTROL);
        let flipped: Vec<[f64; 3]> = got.iter().map(|g| g.map(|x| -x)).collect();
        assert!(largest(&flipped, &want) > CONTROL);
        let mut owner = toy.basis_owner();
        owner.rotate_left(2);
        let confused = SpinOrbit::new(
            spin_orbit(&ctx, toy, &LAMBDA).operator().clone(),
            owner,
            toy.projector_owner(),
            toy.atoms(),
        )
        .unwrap();
        let wrong = contract(&SpinorToy::new(toy, confused, true, true), &state).unwrap();
        assert!(largest(&total(&wrong), &want) > CONTROL);
        // A different coupling strength is a different force.
        let other = spin_orbit(&ctx, toy, &LAMBDA.map(|x| 1.5 * x));
        let scaled = contract(&SpinorToy::new(toy, other, true, true), &state).unwrap();
        assert!(largest(&total(&scaled), &want) > CONTROL);
    }
}

#[test]
fn the_engine_refuses_a_spinor_hybrid_and_a_provider_without_spin_orbit_data() {
    let ctx = context();
    let toy = Toy::two_site();
    let so = spin_orbit(&ctx, &toy, &LAMBDA);
    let state = frozen(&toy, 3);
    // A spinor hybrid request is refused, by gradient() and by contract() on a hand-built state:
    // the exact exchange of a spin-orbit crystal is not differentiated, and must never be dropped.
    let mut provider = SpinorToy::new(&toy, so.clone(), true, true);
    provider.layout.request.functional = "hse06".into();
    for err in [
        gradient(&provider, &state.state, None).err().unwrap(),
        contract(&provider, &state).err().unwrap(),
    ] {
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("spinor hybrid")),
            "{err:?}"
        );
    }
    provider.layout.request.functional = "lda".into();
    // A spinor request whose provider has no spin-orbit data raises, never a zero term.
    let bare = SpinorToy::new(&toy, so, true, false);
    let err = contract(&bare, &state).err().unwrap();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    // A spinor layout has one spin channel of degeneracy 1: anything else would double-count.
    let mut doubled = SpinorToy::new(&toy, spin_orbit(&ctx, &toy, &LAMBDA), true, true);
    doubled.layout.degeneracy = 2.;
    assert!(contract(&doubled, &state).is_err());
    // A density of the wrong frame is refused by the term.
    let mut narrow = frozen(&toy, 3);
    let n = 2 * toy.basis() - 2;
    narrow.state.bands[0][0].coefficients = Matrix::identity(n, n);
    narrow.state.bands[0][0].eigenvalues = vec![0.; n];
    narrow.state.bands[0][0].occupations = vec![1.; n];
    assert!(contract(&provider, &narrow).is_err());
}
