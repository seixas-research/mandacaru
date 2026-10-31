//! The DFT+U shell table built from the option and the labels of a crystal, and the rule that
//! compares an option with the table of a loaded crystal file.
use mandacaru_core::math::storage::Precision;
use mandacaru_core::models::hubbard::Form;
use mandacaru_core::models::hubbard_request::Request;
use mandacaru_core::paw::onecenter::{Basis, Channel};
use mandacaru_core::paw::shell;
use mandacaru_core::periodic::hubbard::{Seed, Spec};
use mandacaru_core::periodic::hubbard_table::{
    build, check_loaded, ChannelData, Layout, Projector, Species,
};
use mandacaru_core::{Context, Error};
use std::collections::BTreeMap;

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

/// A reference atom with a `d` channel of two partial waves and a `p` channel of one.
fn species(shift: f64) -> Species {
    let r: Vec<f64> = (1..=400).map(|i| 0.02 * i as f64).collect();
    let wave = |scale: f64, width: f64| -> Vec<f64> {
        r.iter()
            .map(|x| scale * x * x * (-x / width).exp())
            .collect()
    };
    let mut ae = wave(1.0, 1.0 + shift);
    ae.extend(wave(0.7, 1.6 + shift));
    let channels = BTreeMap::from([
        (
            2,
            ChannelData {
                waves: 2,
                ae,
                vanderbilt: vec![1., 0.3, 0.3, 1.],
                r_cut: 2.5,
                occupations: vec![8., 0.],
            },
        ),
        (
            1,
            ChannelData {
                waves: 1,
                ae: wave(1.0, 0.8),
                vanderbilt: vec![1.],
                r_cut: 2.0,
                occupations: vec![4.],
            },
        ),
    ]);
    Species { r, channels }
}

/// Projector labels of one atom's channel `l` with `waves` radial indices, ordered as the
/// crystal orders them: `m`, then radial index.
fn channel(symbol: &str, atom: usize, l: usize, waves: usize, basis: &str) -> Vec<Projector> {
    let l_signed = l as i64;
    (-l_signed..=l_signed)
        .flat_map(|m| (0..waves).map(move |index| (m, index)))
        .map(|(m, index)| Projector {
            symbol: symbol.into(),
            basis: basis.into(),
            atom,
            l,
            m,
            index,
        })
        .collect()
}

fn request(text: &str) -> Request {
    Request::parse(text).unwrap()
}

fn invalid(error: Error) -> String {
    match error {
        Error::Invalid(m) => m,
        other => panic!("expected an Invalid refusal, got {other:?}"),
    }
}

/// Two Ni atoms (listed second atom first) and one O atom.
fn nio() -> (Vec<Projector>, Vec<usize>, Vec<Species>, Vec<f64>) {
    let mut projectors = channel("Ni", 1, 2, 2, "raw");
    projectors.extend(channel("O", 2, 1, 1, "raw"));
    projectors.extend(channel("Ni", 0, 2, 2, "raw"));
    (
        projectors,
        vec![0, 0, 1],
        vec![species(0.), species(0.2)],
        vec![2., -2., 0.],
    )
}

#[test]
fn the_table_has_one_shell_per_atom_of_the_element_in_projector_order() {
    let ctx = context();
    let (projectors, of_atom, species, moments) = nio();
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit: false,
    };
    let spec = build(
        &ctx,
        &request(r#"{"Ni": {"l": "d", "U": 3.0, "J": 0.5}}"#),
        &layout,
    )
    .unwrap();
    assert_eq!(spec.form, Form::DudarevFll);
    // Atom 1 is listed first, so it is the first shell.
    let atoms: Vec<_> = spec.shells.iter().map(|s| s.atom).collect();
    assert_eq!(atoms, [1, 0]);
    let first = &spec.shells[0];
    assert_eq!((first.symbol.as_str(), first.l), ("Ni", 2));
    assert_eq!(first.projectors, (0..10).collect::<Vec<_>>());
    assert_eq!(spec.shells[1].projectors, (13..23).collect::<Vec<_>>());
    assert_eq!(first.u, 3.0 / mandacaru_core::units::HARTREE_TO_EV);
    assert_eq!(first.j, 0.5 / mandacaru_core::units::HARTREE_TO_EV);
    assert_eq!(first.basis.as_deref(), Some("raw"));
    // The seed carries the shell's reference electrons and the atom's own signed moment.
    let seeds: Vec<_> = spec
        .shells
        .iter()
        .map(|s| match s.seed {
            Seed::Moment {
                n_electrons,
                moment,
            } => (n_electrons, moment),
            _ => panic!("not a moment seed"),
        })
        .collect();
    assert_eq!(seeds, [(8., -2.), (8., 2.)]);
}

#[test]
fn the_weights_are_the_partial_wave_weights_of_the_reference_channel() {
    let ctx = context();
    let (projectors, of_atom, species, moments) = nio();
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit: false,
    };
    let spec = build(&ctx, &request(r#"{"Ni": {"l": "d", "U": 3.0}}"#), &layout).unwrap();
    let data = &species[0].channels[&2];
    let view = Channel {
        l: 2,
        waves: 2,
        ae: &data.ae,
        ps: &[],
        vanderbilt: &data.vanderbilt,
        r_cut: data.r_cut,
    };
    let expected =
        shell::weights(&ctx, &view, &species[0].r, &data.occupations, Basis::Raw).unwrap();
    assert_eq!(expected.len(), 2);
    for s in &spec.shells {
        assert_eq!(s.weights, expected);
    }
}

#[test]
fn the_table_round_trips_through_the_snapshot_text() {
    let ctx = context();
    let (projectors, of_atom, species, moments) = nio();
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit: false,
    };
    let spec = build(&ctx, &request(r#"{"Ni": {"l": "d", "U": 6.0}}"#), &layout).unwrap();
    let text = serde_json::to_string(&spec.to_json()).unwrap();
    let back = Spec::parse(
        &serde_json::from_str(&text).unwrap(),
        projectors.len(),
        2,
        false,
        Precision::Float64,
        0,
    )
    .unwrap()
    .expect("the table parses");
    assert_eq!(back, spec);
    assert_eq!(back.form, Form::DudarevFll);
}

#[test]
fn a_dual_basis_crystal_takes_its_weights_in_that_basis() {
    let ctx = context();
    let projectors = channel("Ni", 0, 2, 2, "dual");
    let species = [species(0.)];
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &[0],
        species: &species,
        moments: &[0.],
        spin_orbit: false,
    };
    let spec = build(&ctx, &request(r#"{"Ni": {"l": "d", "U": 3.0}}"#), &layout).unwrap();
    let data = &species[0].channels[&2];
    let view = Channel {
        l: 2,
        waves: 2,
        ae: &data.ae,
        ps: &[],
        vanderbilt: &data.vanderbilt,
        r_cut: data.r_cut,
    };
    let dual = shell::weights(&ctx, &view, &species[0].r, &data.occupations, Basis::Dual).unwrap();
    assert_eq!(spec.shells[0].weights, dual);
    assert_eq!(spec.shells[0].basis.as_deref(), Some("dual"));
}

#[test]
fn what_the_crystal_cannot_give_is_refused_naming_the_option_and_the_element() {
    let ctx = context();
    let (projectors, of_atom, species, moments) = nio();
    let layout = |spin_orbit| Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit,
    };
    let ni = request(r#"{"Ni": {"l": "d", "U": 3.0}}"#);
    match build(&ctx, &ni, &layout(true)) {
        Err(Error::Unsupported(m)) => {
            assert!(m.contains("hubbard_u") && m.contains("spin-orbit"), "{m}")
        }
        other => panic!("{other:?}"),
    }
    let absent = invalid(
        build(
            &ctx,
            &request(r#"{"Cu": {"l": "d", "U": 3.0}, "Fe": {"l": "d", "U": 1.0}}"#),
            &layout(false),
        )
        .unwrap_err(),
    );
    assert!(
        absent.contains("hubbard_u names Cu, Fe, which is not in the system"),
        "{absent}"
    );
    let wrong = invalid(
        build(
            &ctx,
            &request(r#"{"Ni": {"l": "f", "U": 3.0}}"#),
            &layout(false),
        )
        .unwrap_err(),
    );
    assert!(
        wrong.contains("hubbard_u: the Ni dataset has no 'f' channel"),
        "{wrong}"
    );
    // O has projectors but none of the requested shell.
    let none = invalid(
        build(
            &ctx,
            &request(r#"{"O": {"l": "d", "U": 3.0}}"#),
            &layout(false),
        )
        .unwrap_err(),
    );
    assert!(none.contains("the O dataset has no 'd' channel"), "{none}");
}

#[test]
fn an_incomplete_or_inconsistent_crystal_raises_and_never_panics() {
    let ctx = context();
    let ni = request(r#"{"Ni": {"l": "d", "U": 3.0}}"#);
    let (projectors, of_atom, species, moments) = nio();
    // A projector of the shell is missing.
    let mut cut = projectors.clone();
    cut.remove(3);
    let layout = Layout {
        projectors: &cut,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit: false,
    };
    let m = invalid(build(&ctx, &ni, &layout).unwrap_err());
    assert!(m.contains("hubbard_u") && m.contains("lack m ="), "{m}");
    // An atom with no moment, a non-finite moment, no dataset, an unknown basis.
    let mut layout = Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments[..1],
        spin_orbit: false,
    };
    assert!(
        invalid(build(&ctx, &ni, &layout).unwrap_err()).contains("finite initial magnetic moment")
    );
    let nan = [f64::NAN, 1., 0.];
    layout.moments = &nan;
    assert!(
        invalid(build(&ctx, &ni, &layout).unwrap_err()).contains("finite initial magnetic moment")
    );
    layout.moments = &moments;
    layout.species_of_atom = &of_atom[..1];
    assert!(invalid(build(&ctx, &ni, &layout).unwrap_err()).contains("has no dataset"));
    layout.species_of_atom = &[0, 9, 1];
    assert!(invalid(build(&ctx, &ni, &layout).unwrap_err()).contains("has no dataset"));
    layout.species_of_atom = &of_atom;
    let odd = channel("Ni", 0, 2, 2, "orthonormal");
    layout.projectors = &odd;
    assert!(invalid(build(&ctx, &ni, &layout).unwrap_err()).contains("unknown projector basis"));
    // A dataset that lacks the channel the projectors have.
    let mut bare = species.clone();
    bare[0].channels.remove(&2);
    layout.projectors = &projectors;
    layout.species = &bare;
    assert!(invalid(build(&ctx, &ni, &layout).unwrap_err())
        .contains("the Ni dataset has no 'd' channel"));
}

#[test]
fn an_option_agrees_with_a_file_only_when_it_names_the_same_shells_and_energies() {
    let ctx = context();
    let (projectors, of_atom, species, moments) = nio();
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &of_atom,
        species: &species,
        moments: &moments,
        spin_orbit: false,
    };
    let option = request(r#"{"Ni": {"l": "d", "U": 3.0, "J": 0.25}}"#);
    let file = build(&ctx, &option, &layout).unwrap();
    // Through the snapshot text, as a saved crystal stores it.
    let text = serde_json::to_string(&file.to_json()).unwrap();
    let loaded = Spec::parse(
        &serde_json::from_str(&text).unwrap(),
        projectors.len(),
        2,
        false,
        Precision::Float64,
        0,
    )
    .unwrap()
    .unwrap();
    check_loaded(&option, Some(&loaded)).unwrap();
    for other in [
        r#"{"Ni": {"l": "d", "U": 3.1, "J": 0.25}}"#,
        r#"{"Ni": {"l": "d", "U": 3.0}}"#,
        r#"{"Ni": {"l": "f", "U": 3.0, "J": 0.25}}"#,
        r#"{"Ni": {"l": "d", "U": 3.0, "J": 0.25}, "O": {"l": "d", "U": 1.0}}"#,
        r#"{"O": {"l": "d", "U": 3.0, "J": 0.25}}"#,
    ] {
        let m = invalid(check_loaded(&request(other), Some(&loaded)).unwrap_err());
        assert!(
            m.contains("hubbard_u disagrees with the crystal file"),
            "{other}: {m}"
        );
    }
    // A file with no correction disagrees with any option.
    let m = invalid(check_loaded(&option, None).unwrap_err());
    assert!(m.contains("file: no correction") && m.contains("Ni"), "{m}");
}
