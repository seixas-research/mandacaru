//! The DFT+U shell table built from the option and the labels of a crystal, and the rule that
//! compares an option with the table of a loaded crystal file.
use mandacaru_core::math::storage::Precision;
use mandacaru_core::models::hubbard::Form;
use mandacaru_core::models::hubbard_request::Request;
use mandacaru_core::paw::onecenter::{Basis, Channel};
use mandacaru_core::paw::shell;
use mandacaru_core::periodic::hubbard::{Seed, Spec};
use mandacaru_core::periodic::hubbard_table::{
    build, build_paw, check_datasets, check_loaded, ChannelData, Dataset, Layout, PawLayout,
    Projector, Species,
};
use mandacaru_core::{Context, Error};
use std::collections::BTreeMap;

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

#[test]
fn paw_records_produce_the_same_table_without_flattening_waves() {
    let ctx = context();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let path = std::path::PathBuf::from(root).join("lda-sr/Fe.parquet");
    let ds = std::sync::Arc::new(mandacaru_core::paw::record::PawDataset::read(&ctx, &path).unwrap());
    let d = &ds.channels[&2];
    let old = Species { r: ds.r.as_slice().to_vec(), source: path.to_string_lossy().into(),
        channels: BTreeMap::from([(2, ChannelData { waves: d.ae_waves.len(),
            ae: d.ae_waves.iter().flat_map(|w| w.as_slice().iter().copied()).collect(),
            vanderbilt: d.vanderbilt.data().to_vec(), r_cut: d.r_cut,
            occupations: d.occupations.data().to_vec() })]) };
    let projectors = channel("Fe", 0, 2, d.ae_waves.len(), "raw");
    let request = request(r#"{"Fe": {"l": "d", "U": 4.0}}"#);
    let old_species = [old];
    let expected = build(&ctx, &request, &Layout { projectors: &projectors,
        species_of_atom: &[0], species: &old_species, moments: &[2.], spin_orbit: false }).unwrap();
    let records = [ds];
    let sources = [path.to_string_lossy().into_owned()];
    let actual = build_paw(&ctx, &request, &PawLayout { projectors: &projectors,
        species_of_atom: &[0], species: &records, sources: &sources,
        moments: &[2.], spin_orbit: false }).unwrap();
    assert_eq!(actual.to_json(), expected.to_json());
}

#[test]
fn paw_table_matches_flattened_mixed_species_and_repeated_atoms() {
    let ctx = context();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let folder = std::path::PathBuf::from(root).join("lda-sr");
    let paths: Vec<_> = ["Fe", "Ce"].iter().map(|s| folder.join(format!("{s}.parquet"))).collect();
    let records: Vec<_> = paths.iter().map(|p| std::sync::Arc::new(
        mandacaru_core::paw::record::PawDataset::read(&ctx, p).unwrap())).collect();
    let old_species: Vec<_> = records.iter().zip(&paths).map(|(ds, path)| Species {
        r: ds.r.as_slice().to_vec(), source: path.to_string_lossy().into(),
        channels: ds.channels.iter().map(|(&l, d)| (l as usize, ChannelData {
            waves: d.ae_waves.len(),
            ae: d.ae_waves.iter().flat_map(|w| w.as_slice().iter().copied()).collect(),
            vanderbilt: d.vanderbilt.data().to_vec(), r_cut: d.r_cut,
            occupations: d.occupations.data().to_vec(),
        })).collect(),
    }).collect();
    let mut projectors = channel("Ce", 0, 3, records[1].channels[&3].ae_waves.len(), "raw");
    projectors.extend(channel("Fe", 1, 2, records[0].channels[&2].ae_waves.len(), "raw"));
    projectors.extend(channel("Fe", 2, 2, records[0].channels[&2].ae_waves.len(), "raw"));
    let of_atom = [1, 0, 0];
    let moments = [0.5, 2., -2.];
    let request = request(r#"{"Ce": {"l": "f", "U": 5.0}, "Fe": {"l": "d", "U": 4.0}}"#);
    let expected = build(&ctx, &request, &Layout { projectors: &projectors,
        species_of_atom: &of_atom, species: &old_species, moments: &moments, spin_orbit: false }).unwrap();
    let sources: Vec<_> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    let actual = build_paw(&ctx, &request, &PawLayout { projectors: &projectors,
        species_of_atom: &of_atom, species: &records, sources: &sources,
        moments: &moments, spin_orbit: false }).unwrap();
    assert_eq!(actual.to_json(), expected.to_json());
    assert_eq!(actual.shells.iter().map(|s| s.atom).collect::<Vec<_>>(), [0, 1, 2]);
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
    Species {
        r,
        channels,
        source: "/library/lda-sr/Ni.json".into(),
    }
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

/// An element whose reference atom holds no electron in the requested channel (Ba 5d or Au 5f in
/// `lda-sr`) cannot define the shell: the table and the geometry check alone both refuse, naming
/// the element, the shell and the dataset set, and neither substitutes a partial wave.
#[test]
fn a_channel_with_no_occupied_reference_wave_is_refused_naming_element_shell_and_set() {
    let ctx = context();
    let mut empty = species(0.);
    empty.channels.get_mut(&2).unwrap().occupations = vec![0., 0.];
    empty.source = "/library/lda-sr/Ba.json".into();
    let projectors = channel("Ba", 0, 2, 2, "raw");
    let all = [empty];
    let layout = Layout {
        projectors: &projectors,
        species_of_atom: &[0],
        species: &all,
        moments: &[0.],
        spin_orbit: false,
    };
    let ba = request(r#"{"Ba": {"l": "d", "U": 3.0}}"#);
    let refused = invalid(build(&ctx, &ba, &layout).unwrap_err());
    for part in [
        "hubbard_u",
        "'d' shell of Ba",
        "lda-sr",
        "no occupied reference wave",
    ] {
        assert!(refused.contains(part), "{part} missing from: {refused}");
    }
    let geometry_only =
        invalid(check_datasets(&ba, false, &[Dataset::of("Ba", &all[0])]).unwrap_err());
    assert_eq!(geometry_only, refused);
    // An element whose d channel is occupied passes the same check.
    check_datasets(
        &request(r#"{"Ba": {"l": "d", "U": 3.0}}"#),
        false,
        &[Dataset::of("Ba", &species(0.))],
    )
    .unwrap();
}

/// The geometry check is the prelude of the build: it needs no projector, so the calculator can
/// run it at construction. Each refusal names `hubbard_u`.
#[test]
fn the_geometry_check_refuses_from_the_elements_and_their_datasets_alone() {
    let ni = species(0.);
    let sets = [Dataset::of("Ni", &ni), Dataset::of("O", &ni)];
    let d = request(r#"{"Ni": {"l": "d", "U": 3.0}}"#);
    check_datasets(&d, false, &sets).unwrap();
    assert!(matches!(
        check_datasets(&d, true, &sets),
        Err(Error::Unsupported(m)) if m.contains("spin-orbit")
    ));
    let absent = invalid(
        check_datasets(&request(r#"{"Cu": {"l": "d", "U": 1.0}}"#), false, &sets).unwrap_err(),
    );
    assert!(
        absent.contains("hubbard_u names Cu, which is not in the system"),
        "{absent}"
    );
    let wrong = invalid(
        check_datasets(&request(r#"{"Ni": {"l": "f", "U": 1.0}}"#), false, &sets).unwrap_err(),
    );
    assert!(
        wrong.contains("the Ni dataset has no 'f' channel") && wrong.contains("dataset set lda-sr"),
        "{wrong}"
    );
    // A dataset with no file names no folder.
    let mut loose = species(0.);
    loose.source.clear();
    let m = invalid(
        check_datasets(
            &request(r#"{"Ni": {"l": "f", "U": 1.0}}"#),
            false,
            &[Dataset::of("Ni", &loose)],
        )
        .unwrap_err(),
    );
    assert!(m.contains("dataset set unknown"), "{m}");
}
