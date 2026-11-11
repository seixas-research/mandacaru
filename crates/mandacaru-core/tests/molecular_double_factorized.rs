//! The double-factorized two-body representation (`molecular::double_factorized`): its qubit
//! operator against the dense route's (the `n^4` tensor and its fermion monomials), the exact
//! sector energies at stated truncations, and the readers that refuse it by name.
use mandacaru_core::{
    math::SpatialGrid,
    molecular::{
        double_factorized::Representation, transform::Model, MeanField, MolecularIntegrals,
        ScfOptions,
    },
    quantum::{Basis, Encoding, Operator},
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::BTreeMap;

fn chain(ctx: &Context, atoms: usize) -> MolecularIntegrals {
    let positions = (0..atoms).map(|i| [3., 3., 2.2 + 0.9 * i as f64]).collect();
    let geometry = Geometry::new(
        vec![1; atoms],
        positions,
        [[6., 0., 0.], [0., 6., 0.], [0., 0., 7.5]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.4 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO","virtual_orbitals":1}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}

fn terms(operator: &Operator) -> BTreeMap<(u64, u64), (f64, f64)> {
    operator
        .terms
        .iter()
        .map(|t| ((t.x, t.z), (t.coefficient.re, t.coefficient.im)))
        .collect()
}

/// Largest coefficient difference over the union of the two operators' Pauli strings.
fn difference(a: &Operator, b: &Operator) -> f64 {
    let (a, b) = (terms(a), terms(b));
    a.keys()
        .chain(b.keys())
        .map(|k| {
            let x = a.get(k).copied().unwrap_or_default();
            let y = b.get(k).copied().unwrap_or_default();
            (x.0 - y.0).hypot(x.1 - y.1)
        })
        .fold(0., f64::max)
}

/// The exact ground energy of `model` in its particle sector, by dense diagonalization.
fn ground(ctx: &Context, model: &Model) -> f64 {
    let encoding = Encoding::new("jordan_wigner", 2 * model.m, model.particles).unwrap();
    let basis = Basis::new(ctx, &encoding, true).unwrap();
    let operator = model.operator(ctx, "jordan_wigner").unwrap();
    let d = basis.len();
    let dense = operator.dense(ctx, &basis).unwrap();
    let matrix = DMatrix::from_fn(d, d, |i, j| dense.as_slice()[i * d + j].re);
    SymmetricEigen::new(matrix).eigenvalues.min()
}

fn models(
    integrals: &MolecularIntegrals,
    state: &MeanField,
    spec: &str,
    truncation: f64,
) -> (Model, Model) {
    let dense = state.active_model(integrals, spec).unwrap();
    let factorized = state
        .active_model_with(
            integrals,
            spec,
            Representation::DoubleFactorized { truncation },
        )
        .unwrap();
    (dense, factorized)
}

#[test]
fn untruncated_it_is_the_dense_operator() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = chain(&ctx, 4);
    let state = integrals
        .mean_field("rhf", [2, 2], &ScfOptions::default())
        .unwrap();
    for spec in [r#"{}"#, r#"{"orbitals": 5, "frozen": [0]}"#] {
        let (dense, factorized) = models(&integrals, &state, spec, 0.);
        assert_eq!(factorized.representation(), "double-factorized");
        assert_eq!(
            (factorized.m, factorized.particles, factorized.constant),
            (dense.m, dense.particles, dense.constant)
        );
        assert_eq!(factorized.one.as_slice(), dense.one.as_slice());
        for mapping in ["jordan_wigner", "parity", "bravyi_kitaev"] {
            let a = dense.operator(&ctx, mapping).unwrap();
            let b = factorized.operator(&ctx, mapping).unwrap();
            let worst = difference(&a, &b);
            eprintln!(
                "{spec} {mapping}: {} terms, largest difference {worst:.3e} Ha",
                a.terms.len()
            );
            assert!(worst < 1e-10, "{spec} {mapping}: {worst:.3e}");
            assert_eq!(
                a.terms.len(),
                b.terms.len(),
                "{spec} {mapping}: no roundoff strings"
            );
            assert_eq!(a.decomposition.len(), b.decomposition.len());
            for (x, y) in a.decomposition.iter().zip(&b.decomposition) {
                assert!(difference(x, y) < 1e-10);
            }
        }
        let f = factorized.factorized.as_ref().unwrap();
        assert!(f.stored_numbers() <= f.candidates * factorized.m * (factorized.m + 1));
        assert!(f.error_bound < 1e-10, "{}", f.error_bound);
    }
}

#[test]
fn the_truncation_changes_the_energy_by_what_it_drops() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = chain(&ctx, 4);
    let state = integrals
        .mean_field("rhf", [2, 2], &ScfOptions::default())
        .unwrap();
    let spec = r#"{}"#;
    let exact = ground(&ctx, &state.active_model(&integrals, spec).unwrap());
    let mut previous = usize::MAX;
    for truncation in [0., 1e-4, 1e-3, 1e-2, 1e-1] {
        let (_, factorized) = models(&integrals, &state, spec, truncation);
        let f = factorized.factorized.as_ref().unwrap();
        let error = ground(&ctx, &factorized) - exact;
        eprintln!(
            "truncation {truncation:.0e} Ha: {} of {} eigenpairs in {} leaves, error bound \
             {:.3e} Ha, ground-energy error {error:+.3e} Ha",
            f.kept,
            f.eigenpairs,
            f.leaves.len(),
            f.error_bound
        );
        assert!(f.error_bound <= truncation + 1e-10, "{}", f.error_bound);
        assert!(f.kept <= previous);
        previous = f.kept;
        // The bound holds for every state, the ground state included.
        assert!(error.abs() <= f.error_bound + 1e-10, "{error:.3e}");
    }
}

#[test]
fn readers_of_the_dense_tensor_refuse_it_by_name() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = chain(&ctx, 2);
    let state = integrals
        .mean_field("rhf", [1, 1], &ScfOptions::default())
        .unwrap();
    let (_, factorized) = models(&integrals, &state, r#"{}"#, 1e-6);
    assert!(factorized.two.as_slice().is_empty());
    let message = |e: mandacaru_core::Error| {
        let text = e.to_string();
        assert!(text.contains("double-factorized"), "{text}");
        assert!(matches!(e, mandacaru_core::Error::Unsupported(_)), "{text}");
    };
    message(factorized.fermion_terms().unwrap_err());
    message(factorized.reduce(&ctx, &[], &[0]).err().unwrap());
    message(factorized.factorized_measurement(&ctx).err().unwrap());
    message(factorized.dense_two("a reader").unwrap_err());
}

#[test]
fn the_representation_is_named_and_its_truncation_stated() {
    assert_eq!(
        Representation::parse("dense", None).unwrap(),
        Representation::Dense
    );
    assert_eq!(
        Representation::parse("double-factorized", Some(1e-4)).unwrap(),
        Representation::DoubleFactorized { truncation: 1e-4 }
    );
    for (name, truncation) in [
        ("dense", Some(1e-4)),
        ("double-factorized", None),
        ("double-factorized", Some(-1.)),
        ("double-factorized", Some(f64::NAN)),
        ("tensor-hypercontraction", None),
    ] {
        assert!(Representation::parse(name, truncation).is_err(), "{name}");
    }
}

/// The leaves of a signed augmentation (`Eri::Augmented`, as a PAW basis carries) rebuild the
/// tensor `transform::four` forms, element by element.
#[test]
fn signed_augmentation_leaves_rebuild_the_tensor() {
    use mandacaru_core::molecular::{
        double_factorized::DoubleFactorized,
        integrals::{Eri, PairCorrection},
        transform,
    };
    use mandacaru_core::quantum::C64;
    use std::sync::Arc;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = chain(&ctx, 4);
    let state = integrals
        .mean_field("rhf", [2, 2], &ScfOptions::default())
        .unwrap();
    let base = integrals.eri().unwrap();
    let m = base.m();
    let mut seed = 7u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let factors: Vec<DMatrix<f64>> = (0..4)
        .map(|_| {
            let a = DMatrix::from_fn(m, m, |_, _| random());
            0.5 * (&a + a.transpose())
        })
        .collect();
    let eri = Eri::Augmented {
        m,
        base,
        correction: Arc::new(PairCorrection {
            factors,
            weights: vec![0.3, -0.2, 0.05, -0.01],
            memory: ctx.reserve(0).unwrap(),
        }),
    };
    let c: DMatrix<C64> = state.model_orbitals.clone();
    let n = c.ncols();
    let dense = transform::four(&ctx, &eri, [&c; 4]).unwrap();
    let f = DoubleFactorized::from_eri(&ctx, &eri, &c, 0.).unwrap();
    let mut rebuilt = vec![0f64; n * n * n * n];
    for leaf in &f.leaves {
        let l = leaf.matrix();
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        rebuilt[((p * n + q) * n + r) * n + s] += leaf.sign * l[(p, r)] * l[(q, s)];
                    }
                }
            }
        }
    }
    let worst = dense
        .as_slice()
        .iter()
        .zip(&rebuilt)
        .map(|(a, b)| (a.re - b).abs())
        .fold(0., f64::max);
    eprintln!(
        "{n} orbitals, {} leaves: largest element difference {worst:.3e}, bound {:.3e}",
        f.leaves.len(),
        f.error_bound
    );
    assert!(worst < 1e-10, "{worst:.3e}");
}

/// A saved factorized model records its representation, truncation and bound (eV); a dense
/// one records nothing, and a malformed record is refused.
#[test]
fn a_saved_model_records_its_two_body_representation() {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    use mandacaru_core::units::HARTREE_TO_EV;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = chain(&ctx, 2);
    let state = integrals
        .mean_field("rhf", [1, 1], &ScfOptions::default())
        .unwrap();
    let (dense, factorized) = models(&integrals, &state, r#"{}"#, 1e-3);
    let f = factorized.factorized.as_ref().unwrap();
    let document = factorized.document(&ctx, "jordan_wigner").unwrap();
    let record = document.two_body().unwrap().unwrap();
    assert_eq!(record.representation, "double-factorized");
    assert_eq!(record.truncation_ev, 1e-3 * HARTREE_TO_EV);
    assert_eq!(record.error_bound_ev, f.error_bound * HARTREE_TO_EV);
    assert!(dense
        .document(&ctx, "jordan_wigner")
        .unwrap()
        .two_body()
        .unwrap()
        .is_none());
    for (key, value) in [
        ("truncation_ev", serde_json::json!(-1.0)),
        ("error_bound_ev", serde_json::json!("small")),
        (
            "representation",
            serde_json::json!("tensor-hypercontraction"),
        ),
    ] {
        let mut payload = document.payload().clone();
        payload["metadata"]["two_body"][key] = value;
        let tampered = Document::from_value(&ctx, payload, HAMILTONIAN).unwrap();
        assert!(tampered.two_body().is_err(), "{key}");
    }
}
