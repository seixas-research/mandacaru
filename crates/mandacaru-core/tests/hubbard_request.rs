//! The `hubbard_u` option as Rust reads it: the typed request, its units, its form field and
//! every refusal (each names `hubbard_u`, and the element when there is one).
use mandacaru_core::models::hubbard::{shell_l, shell_letter, Form};
use mandacaru_core::models::hubbard_request::Request;
use mandacaru_core::units::HARTREE_TO_EV;
use mandacaru_core::Error;

fn message(text: &str) -> String {
    match Request::parse(text) {
        Err(Error::Invalid(m)) => m,
        other => panic!("{text}: expected an Invalid refusal, got {other:?}"),
    }
}

#[test]
fn a_mapping_resolves_into_typed_entries_in_hartree() {
    let request = Request::parse(r#"{"Ni": {"l": "d", "U": 6.0, "J": 0.5}}"#).unwrap();
    let ni = request.entry("Ni").unwrap();
    assert_eq!((ni.l, ni.letter(), ni.form), (2, 'd', Form::DudarevFll));
    // Exact: the conversion is one division, the same one the Python constant is used in.
    assert_eq!(ni.u, 6.0 / HARTREE_TO_EV);
    assert_eq!(ni.j, 0.5 / HARTREE_TO_EV);
    assert_eq!(ni.u_eff(), 6.0 / HARTREE_TO_EV - 0.5 / HARTREE_TO_EV);
    assert_eq!(request.form(), Form::DudarevFll);
}

#[test]
fn j_defaults_to_zero_integers_are_numbers_and_entries_follow_the_symbols() {
    let request =
        Request::parse(r#"{"Gd": {"l": "f", "U": 6}, "Cu": {"l": "d", "U": 4}}"#).unwrap();
    let symbols: Vec<_> = request.entries.iter().map(|e| e.symbol.as_str()).collect();
    assert_eq!(symbols, ["Cu", "Gd"]);
    let gd = request.entry("Gd").unwrap();
    assert_eq!((gd.l, gd.letter(), gd.j), (3, 'f', 0.));
    assert_eq!(gd.u, 6.0 / HARTREE_TO_EV);
    assert!(request.entry("Ni").is_none());
}

#[test]
fn the_form_field_is_optional_and_names_a_form_the_engine_implements() {
    for form in Form::ALL {
        let text = format!(
            r#"{{"Ni": {{"l": "d", "U": 3.0, "form": "{}"}}}}"#,
            form.name()
        );
        assert_eq!(Request::parse(&text).unwrap().form(), form);
        assert_eq!(Form::parse(form.name()).unwrap(), form);
    }
    let omitted = Request::parse(r#"{"Ni": {"l": "d", "U": 3.0}}"#).unwrap();
    let explicit =
        Request::parse(r#"{"Ni": {"l": "d", "U": 3.0, "form": "dudarev-fll"}}"#).unwrap();
    assert_eq!(omitted, explicit);
    let unknown = message(r#"{"Ni": {"l": "d", "U": 3.0, "form": "liechtenstein"}}"#);
    for part in ["hubbard_u", "Ni", "liechtenstein", "dudarev-fll"] {
        assert!(unknown.contains(part), "{unknown}");
    }
    assert!(message(r#"{"Ni": {"l": "d", "U": 3.0, "form": 1}}"#).contains("form must be a string"));
}

#[test]
fn the_shell_letters_are_d_and_f() {
    assert_eq!((shell_l("d"), shell_l("f")), (Some(2), Some(3)));
    for letter in ["p", "s", "", "dd", "D"] {
        assert_eq!(shell_l(letter), None, "{letter:?}");
    }
    assert_eq!(
        (shell_letter(2), shell_letter(3), shell_letter(1)),
        (Some('d'), Some('f'), None)
    );
}

#[test]
fn every_malformed_option_is_refused_with_its_reason() {
    let cases: [(&str, &[&str]); 29] = [
        ("{}", &["must map an element symbol"]),
        ("[]", &["must map an element symbol"]),
        (r#""Ni""#, &["must map an element symbol", "'Ni'"]),
        ("3", &["must map an element symbol"]),
        ("null", &["must map an element symbol"]),
        ("{", &["not valid JSON"]),
        (
            r#"{"Xx": {"l": "d", "U": 1.0}}"#,
            &["'Xx' is not an element symbol"],
        ),
        (
            r#"{"X": {"l": "d", "U": 1.0}}"#,
            &["'X' is not an element symbol"],
        ),
        (
            r#"{"ni": {"l": "d", "U": 1.0}}"#,
            &["'ni' is not an element symbol"],
        ),
        (
            r#"{"NaN": {"l": "d", "U": 1.0}}"#,
            &["'NaN' is not an element symbol"],
        ),
        (r#"{"Ni": 6.0}"#, &["'Ni'", "needs the keys"]),
        (r#"{"Ni": {"U": 6.0}}"#, &["'Ni'", "needs the keys"]),
        (r#"{"Ni": {"l": "d"}}"#, &["'Ni'", "needs the keys"]),
        (
            r#"{"Ni": {"l": "d", "U": 6.0, "K": 1.0}}"#,
            &["'Ni'", "needs the keys"],
        ),
        (r#"{"Ni": {"l": "p", "U": 6.0}}"#, &["'Ni'", "'d' or 'f'"]),
        (r#"{"Ni": {"l": 2, "U": 6.0}}"#, &["'Ni'", "'d' or 'f'"]),
        (
            r#"{"Ni": {"l": "d", "U": -1.0}}"#,
            &["'Ni'", "U must be a finite number >= 0 in eV"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": true}}"#,
            &["'Ni'", "U must be a finite number >= 0", "true"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": "6"}}"#,
            &["U must be a finite number >= 0", "\"6\""],
        ),
        (
            r#"{"Ni": {"l": "d", "U": null}}"#,
            &["U must be a finite number >= 0"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": NaN}}"#,
            &["'Ni'", "U must be a finite number >= 0", "NaN"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": Infinity}}"#,
            &["U must be a finite number >= 0", "Infinity"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": -Infinity}}"#,
            &["U must be a finite number >= 0"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": 1.0, "J": -0.1}}"#,
            &["'Ni'", "J must be a finite number >= 0"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": 1.0, "J": NaN}}"#,
            &["J must be a finite number >= 0"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": 1.0, "J": false}}"#,
            &["J must be a finite number >= 0"],
        ),
        (
            r#"{"Ni": {"l": "d", "U": 1.0, "J": 2.0}}"#,
            &["'Ni'", "J = 2 eV exceeds U = 1 eV"],
        ),
        (r#"{"Ni": {"l": "d", "U": 1e999}}"#, &["not valid JSON"]),
        (
            r#"{"Ni": {"l": "d", "U": 1.0, "form": "x"}}"#,
            &["'Ni'", "unknown Hubbard form"],
        ),
    ];
    for (text, parts) in cases {
        let m = message(text);
        assert!(m.contains("hubbard_u"), "{text}: {m}");
        for part in parts {
            assert!(m.contains(part), "{text}: {m:?} lacks {part:?}");
        }
    }
}

#[test]
fn the_non_finite_tokens_inside_strings_are_left_alone() {
    // A string that spells a token is a string, not a number: the value is refused as one, and
    // the quoted text comes back unchanged in the message.
    let m = message(r#"{"Ni": {"l": "d", "U": "NaN -Infinity"}}"#);
    assert!(m.contains(r#""NaN -Infinity""#), "{m}");
    // An escaped quote does not end the string early.
    let m = message(r#"{"Ni": {"l": "d", "U": "a\"NaN"}}"#);
    assert!(m.contains(r#"a\"NaN"#), "{m}");
}

#[test]
fn the_scope_refuses_a_hybrid_and_a_precision_other_than_float64() {
    let request = Request::parse(r#"{"Ni": {"l": "d", "U": 3.0}}"#).unwrap();
    for xc in ["lda", "pbe", "r2scan", "pbe+rvv10", "r2scan+rvv10"] {
        request.require_scope(xc, "float64").unwrap();
    }
    match request.require_scope("hse06", "float64") {
        Err(Error::Unsupported(m)) => {
            assert!(
                m.contains("hubbard_u") && m.contains("hybrid functional 'hse06'"),
                "{m}"
            )
        }
        other => panic!("{other:?}"),
    }
    match request.require_scope("lda", "float32") {
        Err(Error::Invalid(m)) => assert!(m.contains("hubbard_u needs precision='float64'"), "{m}"),
        other => panic!("{other:?}"),
    }
    // The hybrid is refused first, as the options are read.
    assert!(matches!(
        request.require_scope("hse06", "float32"),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn an_unknown_form_lists_the_implemented_ones() {
    let Err(Error::Invalid(m)) = Form::parse("around-mean-field") else {
        panic!("must raise")
    };
    for form in Form::ALL {
        assert!(m.contains(form.name()), "{m}");
    }
}
