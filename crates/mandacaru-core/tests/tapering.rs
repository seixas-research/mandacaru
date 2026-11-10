//! MG-Q7: Z2 tapering (`quantum::tapering`) against the frozen outputs of the Python module it
//! replaced (`test/migration/reference/tapering.json`, the numbers its tests pinned): for H2, LiH and H2O
//! under every mapping, the generators found, their reduced form, the anchors, the sector signs
//! and the tapered Hamiltonian's terms, and the tapered register's lowest eigenvalue equal to the
//! untapered one (the reference's sector holds the ground state).
use mandacaru_core::{
    quantum::{
        algebra::{Basis, Operator},
        tapering::Taper,
        C64,
    },
    Context,
};
use serde_json::Value;

fn frozen() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/tapering.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn terms(v: &Value) -> Vec<(String, C64)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| (t[0].as_str().unwrap().to_string(), C64::new(t[1].as_f64().unwrap(), t[2].as_f64().unwrap())))
        .collect()
}

fn label(n: usize, mask: u64) -> String {
    (0..n).map(|j| if mask & (1 << (n - 1 - j)) != 0 { 'Z' } else { 'I' }).collect()
}

fn lowest(ctx: &Context, operator: &Operator) -> f64 {
    let n = operator.n;
    let basis = Basis::register(ctx, n).unwrap();
    let dense = operator.dense(ctx, &basis).unwrap();
    let size = 1 << n;
    let matrix = nalgebra::DMatrix::from_fn(size, size, |i, j| dense.as_slice()[i * size + j]);
    matrix.symmetric_eigen().eigenvalues.iter().fold(f64::INFINITY, |m, &e| m.min(e))
}

#[test]
fn the_tapered_hamiltonians_are_the_frozen_python_ones() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    for case in frozen()["cases"].as_array().unwrap() {
        let name = format!("{} {}", case["molecule"].as_str().unwrap(), case["mapping"].as_str().unwrap());
        let n = case["n_qubits"].as_u64().unwrap() as usize;
        let h = Operator::new(&ctx, n, &terms(&case["terms"])).unwrap();
        let reference = case["reference_bits"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .fold(0u64, |out, (j, b)| if b.as_u64().unwrap() == 1 { out | (1 << (n - 1 - j)) } else { out });
        if case["generators"].as_array().unwrap().is_empty() {
            assert!(Taper::find(&h, reference).is_none(), "{name}: the Python module found no symmetry");
            println!("{name}: no Z2 symmetry beyond the identity");
            continue;
        }
        let taper = Taper::find(&h, reference).unwrap();
        let strings = |v: &Value| -> Vec<String> { v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect() };
        assert_eq!(taper.found.iter().map(|&m| label(n, m)).collect::<Vec<_>>(), strings(&case["generators"]), "{name}: generators");
        assert_eq!(taper.generators.iter().map(|&m| label(n, m)).collect::<Vec<_>>(), strings(&case["reduced"]), "{name}: reduced");
        let anchors: Vec<usize> = case["anchors"].as_array().unwrap().iter().map(|a| a.as_u64().unwrap() as usize).collect();
        assert_eq!(taper.anchors, anchors, "{name}: anchors");
        let signs: Vec<i8> = case["signs"].as_array().unwrap().iter().map(|s| s.as_i64().unwrap() as i8).collect();
        assert_eq!(taper.signs, signs, "{name}: signs");
        let tapered = taper.operator(&ctx, &h).unwrap();
        assert_eq!(tapered.n, case["tapered_qubits"].as_u64().unwrap() as usize, "{name}: width");
        let mut got = tapered.labels();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let want = terms(&case["tapered_terms"]);
        assert_eq!(got.len(), want.len(), "{name}: term count");
        let miss = got
            .iter()
            .zip(&want)
            .map(|((la, ca), (lb, cb))| {
                assert_eq!(la, lb, "{name}: labels");
                (ca - cb).norm()
            })
            .fold(0f64, f64::max);
        let e = lowest(&ctx, &tapered);
        let full = case["lowest"].as_f64().unwrap();
        println!("{name}: {n} -> {} qubits, worst coefficient miss {miss:.1e}, lowest {e:+.12} (untapered {full:+.12})", tapered.n);
        assert!(miss <= 1e-12, "{name}: {miss:e}");
        assert!((e - full).abs() <= 1e-10, "{name}: {e} against {full}");
    }
}

#[test]
fn a_leaking_operator_is_refused_and_an_observable_is_projected() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let one = C64::new(1., 0.);
    // ZZ is a symmetry of XX + ZZ; XI anticommutes with it.
    let h = Operator::new(&ctx, 2, &[("XX".into(), one), ("ZZ".into(), one)]).unwrap();
    let taper = Taper::find(&h, 0).unwrap();
    assert_eq!(taper.width(), 1);
    let leaking = Operator::new(&ctx, 2, &[("XI".into(), one), ("ZZ".into(), one)]).unwrap();
    assert!(taper.operator(&ctx, &leaking).unwrap_err().to_string().contains("does not commute"));
    let projected = taper.observable(&ctx, &leaking).unwrap();
    assert_eq!(projected.labels().len(), 1);
    // No symmetry: nothing to taper.
    let x = Operator::new(&ctx, 1, &[("X".into(), one), ("Z".into(), one)]).unwrap();
    assert!(Taper::find(&x, 0).is_none());
}
