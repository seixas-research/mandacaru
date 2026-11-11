//! Invalid native periodic requests must be refused before dataset indexing or setup.
use mandacaru_core::{
    paw::record::PawDataset,
    periodic::setup::{create_setup, parse_input_json},
    Context,
};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};

fn request() -> Value {
    json!({"lattice": [6.,0.,0., 0.,6.,0., 0.,0.,6.],
           "grid_shape": [8,8,8],
           "atoms": [{"species":0,"symbol":"H","fractional":[0.,0.,0.]}],
           "kpoints": [{"fractional":[0.,0.,0.],"weight":1.}]})
}

#[test]
fn native_hubbard_setup_keeps_its_table_in_saved_crystal() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join("lda-sr/Fe.parquet");
    let dataset = Arc::new(PawDataset::read(&ctx, &path).unwrap());
    let mut raw = request();
    raw["atoms"][0]["symbol"] = json!("Fe");
    raw["atoms"][0]["moment"] = json!(2.);
    raw["symmetry"] = json!(false);
    raw["hubbard_u"] = json!({"Fe": {"l": "d", "U": 4.0}});
    let setup = create_setup(&ctx, parse_input_json(&raw).unwrap(), vec![dataset]).unwrap();
    let expected = setup.hubbard.as_ref().unwrap().to_json();
    assert_eq!(setup.with_cell(&ctx, &setup.lattice).unwrap().hubbard.unwrap().to_json(), expected);
    let crystal = setup.build(&ctx).unwrap();
    assert_eq!(crystal.hubbard.as_ref().unwrap().to_json(), expected);
    let file = std::env::temp_dir().join(format!("mandacaru-native-hubbard-{}.json", std::process::id()));
    crystal.save(file.to_str().unwrap()).unwrap();
    let loaded = mandacaru_core::periodic::Crystal::load(&ctx, file.to_str().unwrap()).unwrap();
    std::fs::remove_file(file).unwrap();
    assert_eq!(loaded.hubbard.unwrap().to_json(), expected);
}

#[test]
fn native_hubbard_rejects_invalid_option_and_single_precision() {
    let mut raw = request();
    raw["hubbard_u"] = json!({"Fe": 4.});
    assert!(parse_input_json(&raw).is_err());
    raw["hubbard_u"] = json!({"Fe": {"l": "d", "U": 4.}});
    raw["precision"] = json!("float32");
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join("lda-sr/Fe.parquet");
    let dataset = Arc::new(PawDataset::read(&ctx, &path).unwrap());
    raw["atoms"][0]["symbol"] = json!("Fe");
    let error = match create_setup(&ctx, parse_input_json(&raw).unwrap(), vec![dataset]) {
        Ok(_) => panic!("single-precision DFT+U was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("float64"));
}

#[test]
fn native_hubbard_refuses_spin_orbit_before_building_the_setup() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join("lda-fr/Fe.parquet");
    let dataset = Arc::new(PawDataset::read(&ctx, &path).unwrap());
    let mut raw = request();
    raw["atoms"][0]["symbol"] = json!("Fe");
    raw["spinor"] = json!(true);
    raw["hubbard_u"] = json!({"Fe": {"l": "d", "U": 4.}});
    let error = match create_setup(&ctx, parse_input_json(&raw).unwrap(), vec![dataset]) {
        Ok(_) => panic!("spin-orbit DFT+U was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("spin-orbit"));
}

#[test]
fn native_hubbard_on_a_symmetry_reduced_mesh_carries_the_operations_that_complete_it() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join("lda-sr/Fe.parquet");
    let dataset = Arc::new(PawDataset::read(&ctx, &path).unwrap());
    let mut raw = request();
    raw["atoms"][0]["symbol"] = json!("Fe");
    raw["symmetry"] = json!(true);
    raw["hubbard_u"] = json!({"Fe": {"l": "d", "U": 4.}});
    let setup = create_setup(&ctx, parse_input_json(&raw).unwrap(), vec![dataset]).unwrap();
    let crystal = setup.build(&ctx).expect("DFT+U on a reduced mesh is built (HU-4)");
    assert!(!crystal.permutations.is_empty(), "the cubic group is kept");
    let group = crystal.description.as_ref().and_then(|d| d.point_group.as_ref());
    assert_eq!(group.map(|g| g.len()), Some(crystal.permutations.len()));
}

#[test]
fn malformed_vectors_and_values_are_errors_not_panics_or_defaults() {
    let mut cases = Vec::new();
    for shape in [
        json!([]),
        json!([8, 8]),
        json!([8, 8, 0]),
        json!([8, 8, "8"]),
    ] {
        let mut input = request();
        input["grid_shape"] = shape;
        cases.push(input);
    }
    for value in [json!([]), json!([0., 0.]), json!([0., 0., "x"])] {
        let mut input = request();
        input["atoms"][0]["fractional"] = value.clone();
        cases.push(input);
        let mut input = request();
        input["kpoints"][0]["fractional"] = value;
        cases.push(input);
    }
    let mut input = request();
    input["lattice"][0] = json!("invalid");
    cases.push(input);
    let mut input = request();
    input["atoms"][0]["species"] = json!("invalid");
    cases.push(input);
    let mut input = request();
    input["kpoints"][0]["weight"] = json!("invalid");
    cases.push(input);
    for (field, value) in [
        ("kpoints", json!([])),
        ("kpoints", json!("invalid")),
        ("spinor", json!("true")),
        ("spin", json!(1)),
        ("symmetry", json!(1)),
        ("functional", json!(42)),
        ("precision", json!(false)),
        ("sigma", json!("invalid")),
        ("filter_cutoff", json!("invalid")),
    ] {
        let mut input = request();
        input[field] = value;
        cases.push(input);
    }
    let mut input = request();
    input["atoms"][0]["symbol"] = json!(42);
    cases.push(input);
    for input in cases {
        assert!(parse_input_json(&input).is_err(), "accepted {input}");
    }
}

#[test]
fn unsupported_calculation_options_are_not_silently_ignored() {
    for name in [
        "hubbard",
        "ghosts",
        "diffuse",
        "energy_shift",
        "confinement_potential",
        "polarization_method",
        "zeta_split",
        "size",
        "basis",
    ] {
        let mut input = request();
        input[name] = json!("requested");
        let error = parse_input_json(&input).err().unwrap();
        assert!(error.to_string().contains(name), "{error}");
    }
    let mut input = request();
    // A charged cell is a supported request (neutralizing background): the charge is carried.
    input["charge"] = json!(1.);
    assert_eq!(parse_input_json(&input).unwrap().charge, Some(1.));
    input["charge"] = json!(0.);
    assert!(parse_input_json(&input).is_ok());
    // A screened hybrid is a supported functional of the periodic setup (it keeps the whole mesh,
    // see `setup/parse.rs`), so the request parses; the setup still refuses an atom whose dataset
    // is not supplied, before any dataset is read.
    input["functional"] = json!("hse06");
    let parsed = parse_input_json(&input).expect("an hse06 request parses");
    assert_eq!(parsed.functional, "hse06");
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let error = create_setup(&ctx, parsed, vec![]).err().unwrap();
    assert!(error.to_string().contains("dataset index"), "{error}");
}

#[test]
fn unsupported_spinor_and_invalid_dataset_indices_are_refused() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let mut spinor = request();
    spinor["spinor"] = json!(true);
    let error = create_setup(
        &ctx,
        parse_input_json(&spinor).unwrap(),
        vec![dataset(&ctx, "lda-sr", "H")],
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("spin-orbit"));
    let error = create_setup(&ctx, parse_input_json(&request()).unwrap(), vec![])
        .err()
        .unwrap();
    assert!(error.to_string().contains("dataset index"));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn collinear_moments_select_two_spins_and_keep_the_moment_preserving_symmetry() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let mut input = request();
    input["atoms"][0]["moment"] = json!(0.7);
    let setup = create_setup(
        &ctx,
        parse_input_json(&input).unwrap(),
        vec![dataset(&ctx, "lda-sr", "H")],
    )
    .unwrap();
    assert_eq!(setup.spins, 2);
    assert_eq!(setup.moments, [0.7]);
    assert!(setup.symmetry.is_some(), "a collinear moment keeps the operations that preserve it");
    let strained = setup.with_cell(&ctx, &(setup.lattice * 1.01)).unwrap();
    assert_eq!(strained.spins, 2);
    assert_eq!(strained.moments, [0.7]);
}

#[test]
fn mixed_relativistic_exchange_flags_are_refused() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let hydrogen = dataset(&ctx, "lda-sr", "H");
    let mut lithium = dataset(&ctx, "lda-sr", "Li");
    Arc::get_mut(&mut lithium).unwrap().relativistic_exchange = !hydrogen.relativistic_exchange;
    let mut input = request();
    input["atoms"] = json!([
        {"species": 0, "symbol": "H", "fractional": [0., 0., 0.]},
        {"species": 1, "symbol": "Li", "fractional": [0.5, 0.5, 0.5]}
    ]);
    let error = create_setup(
        &ctx,
        parse_input_json(&input).unwrap(),
        vec![hydrogen, lithium],
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("mix relativistic"), "{error}");
}

#[test]
fn malformed_kpoint_weights_are_refused_before_setup() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let valid = parse_input_json(&request()).unwrap();
    for weights in [vec![], vec![0.], vec![0.5], vec![f64::NAN], vec![1., 0.]] {
        let mut input = valid.clone();
        input.weights = weights;
        let error = create_setup(&ctx, input, vec![]).err().unwrap();
        assert!(error.to_string().contains("k-point"), "{error}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

fn dataset(ctx: &Context, set: &str, symbol: &str) -> Arc<PawDataset> {
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is required");
    let path = PathBuf::from(root)
        .join(set)
        .join(format!("{symbol}.parquet"));
    Arc::new(PawDataset::read(ctx, &path).unwrap())
}

#[test]
fn semicore_shells_match_the_existing_basis_and_share_species_data() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let sodium = dataset(&ctx, "lda-sr", "Na");
    let request = json!({
        "lattice": [12.,0.,0., 0.,12.,0., 0.,0.,12.],
        "grid_shape": [8,8,8],
        "atoms": [
            {"species":0,"symbol":"Na","fractional":[0.,0.,0.]},
            {"species":0,"symbol":"Na","fractional":[0.5,0.5,0.5]}
        ],
        "symmetry": false
    });
    let setup = create_setup(&ctx, parse_input_json(&request).unwrap(), vec![sodium]).unwrap();
    let shells = setup.shells.shells();
    assert_eq!(shells.len(), 6);
    assert_eq!(
        shells.iter().map(|s| s.l).collect::<Vec<_>>(),
        [0, 0, 1, 0, 0, 1]
    );
    for i in 0..3 {
        assert!(Arc::ptr_eq(&shells[i].radial, &shells[i + 3].radial));
    }
    for first in setup.projectors.iter().filter(|p| p.atom == 0) {
        let second = setup
            .projectors
            .iter()
            .find(|p| p.atom == 1 && p.l == first.l && p.m == first.m && p.index == first.index)
            .unwrap();
        assert!(Arc::ptr_eq(&first.radial, &second.radial));
    }
    let operators = setup.operators().unwrap();
    let a = setup.projectors.iter().position(|p| p.atom == 0).unwrap();
    let b = setup.projectors.iter().position(|p| p.atom == 1).unwrap();
    assert!(operators.ion_operator[(a, a)].norm() > 1e-10);
    assert!((operators.ion_operator[(a, a)] - operators.ion_operator[(b, b)]).norm() < 1e-12);
    let atom_datasets: Vec<_> = setup
        .atoms_species
        .iter()
        .map(|&i| setup.datasets[i].clone())
        .collect();
    let stored = mandacaru_core::periodic::setup::blocks::store_multipole_blocks(
        &ctx,
        &operators.blocks,
        setup.projectors.len(),
    )
    .unwrap();
    let rebuilt_ops = mandacaru_core::periodic::setup::blocks::raw_operators(
        &setup.projectors,
        &atom_datasets,
        &setup.channels,
    )
    .unwrap();
    let rebuilt = mandacaru_core::periodic::setup::blocks::store_multipole_blocks(
        &ctx,
        &rebuilt_ops.blocks,
        setup.projectors.len(),
    )
    .unwrap();
    assert_eq!(stored.as_slice(), rebuilt.as_slice());
    assert!(mandacaru_core::periodic::setup::blocks::operators(
        &setup.projectors,
        &setup.datasets,
        &setup.channels,
    )
    .is_err());
    let description = setup.description.as_ref().unwrap();
    assert_eq!(description.projector_basis, "raw");
    assert_eq!(description.tables.len(), 3);
    assert_eq!(description.function_table, [0, 1, 2, 2, 2, 0, 1, 2, 2, 2]);
    // Samples from the active SZ pseudo_basis path with the same lattice grid,
    // default confinement and cutoff pi/1.5 Bohr^-1, using lda-sr/Na.parquet
    // (SHA-256 b7f3e0493bcfaedb4d9f83f800200d8c52521163d85b88303f38b1da0a696b31).
    let reference = [
        [
            -1.0731954315219443,
            -1.071173397515892,
            -0.9114567134754544,
            0.03977921418862721,
        ],
        [
            -0.05312238329546367,
            -0.05257590525468026,
            -0.009766843468821684,
            0.1589884447923307,
        ],
        [
            0.003951766880833864,
            0.04340632425266938,
            0.35224272768557185,
            -0.0334385681080411,
        ],
    ];
    for (table, expected) in description.tables.iter().zip(reference) {
        for (index, value) in [0, 10, 100, 500].into_iter().zip(expected) {
            assert!(
                (table.values[index] - value).abs() < 1e-9,
                "radial table l={}, index={index}: {} != {value}",
                table.l,
                table.values[index]
            );
        }
    }
}

#[test]
fn operators_use_each_atoms_species_even_when_dataset_order_differs() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let sodium = dataset(&ctx, "lda-sr", "Na");
    let hydrogen = dataset(&ctx, "lda-sr", "H");
    let request = json!({
        "lattice": [14.,0.,0., 0.,14.,0., 0.,0.,14.],
        "grid_shape": [8,8,8],
        "atoms": [
            {"species":1,"symbol":"H","fractional":[0.,0.,0.]},
            {"species":0,"symbol":"Na","fractional":[0.3,0.3,0.3]},
            {"species":1,"symbol":"H","fractional":[0.6,0.6,0.6]}
        ],
        "symmetry": false
    });
    let setup = create_setup(
        &ctx,
        parse_input_json(&request).unwrap(),
        vec![sodium, hydrogen],
    )
    .unwrap();
    let atom_datasets: Vec<_> = setup
        .atoms_species
        .iter()
        .map(|&s| setup.datasets[s].clone())
        .collect();
    let operators = mandacaru_core::periodic::setup::blocks::operators(
        &setup.projectors,
        &atom_datasets,
        &setup.channels,
    )
    .unwrap();
    let raw = mandacaru_core::periodic::setup::blocks::raw_operators(
        &setup.projectors,
        &atom_datasets,
        &setup.channels,
    )
    .unwrap();
    for atom in 0..3 {
        let (index, _) = setup
            .projectors
            .iter()
            .enumerate()
            .find(|(_, p)| p.atom == atom && p.l == 0 && p.m == 0 && p.index == 0)
            .unwrap();
        let channel = &setup.datasets[setup.atoms_species[atom]].channels[&0];
        let expected_d = channel.coupling.data()[0];
        let expected_q = channel.overlap_correction.data()[0];
        assert!(expected_d.abs() > 1e-10 && expected_q.abs() > 1e-10);
        assert!((operators.ion_operator[(index, index)].re - expected_d).abs() < 1e-12);
        assert!((operators.augmentation[(index, index)].re - expected_q).abs() < 1e-12);
        let (d, q) = mandacaru_core::paw::basis_blocks::projector_basis_transform(
            channel.vanderbilt.data(),
            channel.coupling.data(),
            channel.overlap_correction.data(),
            channel.projectors.len(),
            "raw",
        )
        .unwrap();
        assert!((raw.ion_operator[(index, index)].re - d[0]).abs() < 1e-10);
        assert!((raw.augmentation[(index, index)].re - q[0]).abs() < 1e-10);
        let moment = setup
            .channels
            .iter()
            .position(|&c| c == (atom, 0, 0))
            .unwrap();
        assert!(operators.blocks[moment][(index, index)].norm() > 1e-10);
    }
}

#[test]
fn spin_orbit_dataset_is_not_silently_used_as_scalar() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let silicon = dataset(&ctx, "lda-fr", "Si");
    assert!(!silicon.spin_orbit.is_empty());
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Si");
    let error = create_setup(&ctx, parse_input_json(&input).unwrap(), vec![silicon])
        .err()
        .unwrap();
    assert!(error.to_string().contains("spin-orbit"), "{error}");
}

#[test]
fn spinor_setup_keeps_distinct_spin_orbit_projectors() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let silicon = dataset(&ctx, "lda-fr", "Si");
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Si");
    input["spinor"] = json!(true);
    let setup = create_setup(&ctx, parse_input_json(&input).unwrap(), vec![silicon]).unwrap();
    assert_eq!(setup.spins, 1);
    assert!(setup.spinor && setup.time_reversal && setup.symmetry.is_none());
    assert!(!setup.spin_orbit_projectors.is_empty());
    assert!(!setup.spin_orbit_couplings.is_empty());
    let strained = setup.with_cell(&ctx, &(setup.lattice * 1.01)).unwrap();
    assert_eq!(
        strained.spin_orbit_projectors.len(),
        setup.spin_orbit_projectors.len()
    );
    assert!(strained.time_reversal);
}

#[test]
fn empty_bound_channels_follow_the_active_basis_selection() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for (symbol, expected_l) in [("Ca", vec![0, 0, 1, 2]), ("Li", vec![0]), ("Be", vec![0])] {
        let ds = dataset(&ctx, "lda-sr", symbol);
        let input = json!({
            "lattice": [14.,0.,0., 0.,14.,0., 0.,0.,14.],
            "grid_shape": [8,8,8],
            "atoms": [{"species":0,"symbol":symbol,"fractional":[0.,0.,0.]}],
            "symmetry": false
        });
        let setup = create_setup(&ctx, parse_input_json(&input).unwrap(), vec![ds]).unwrap();
        assert_eq!(
            setup
                .shells
                .shells()
                .iter()
                .map(|s| s.l)
                .collect::<Vec<_>>(),
            expected_l,
            "{symbol}"
        );
        if symbol == "Ca" {
            let table = &setup.description.as_ref().unwrap().tables[3];
            // Production SZ calcium d orbital at the same confinement and filter cutoff.
            for (i, expected) in [
                (0, 1.2886763166880412e-06),
                (100, 0.012890955375185009),
                (500, 0.1970501719773982),
            ] {
                assert!(
                    (table.values[i] - expected).abs() < 1e-9,
                    "Ca d table[{i}] = {}, expected {expected}",
                    table.values[i]
                );
            }
        }
    }
}

#[test]
fn multiple_zeta_tables_match_the_active_basis_and_keep_semicore_single() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Si");
    input["symmetry"] = json!(false);
    input["basis_size"] = json!("DZ");
    input["lattice"] = json!([12., 0., 0., 0., 12., 0., 0., 0., 12.]);
    let setup = create_setup(
        &ctx,
        parse_input_json(&input).unwrap(),
        vec![dataset(&ctx, "lda-sr", "Si")],
    )
    .unwrap();
    assert_eq!(
        setup
            .shells
            .shells()
            .iter()
            .map(|s| s.l)
            .collect::<Vec<_>>(),
        [0, 0, 1, 1]
    );
    let reference = [
        [0.5633283209235455, 0.5406476708412389, 0.18698782018076537],
        [0.7492883208985807, 0.7068901882456299, 0.1350323266945252],
        [
            -0.0017108891919458307,
            -0.16375993534103153,
            -0.21999510697405233,
        ],
        [
            -0.002190702954435485,
            -0.20863490752496344,
            -0.22070978410030861,
        ],
    ];
    for (table, expected) in setup
        .description
        .as_ref()
        .unwrap()
        .tables
        .iter()
        .zip(reference)
    {
        for (index, value) in [0, 100, 500].into_iter().zip(expected) {
            assert!(
                (table.values[index] - value).abs() < 1e-9,
                "l={} index={index}: {} != {value}",
                table.l,
                table.values[index]
            );
        }
    }
    let mut ca = input.clone();
    ca["atoms"][0]["symbol"] = json!("Ca");
    let setup = create_setup(
        &ctx,
        parse_input_json(&ca).unwrap(),
        vec![dataset(&ctx, "lda-sr", "Ca")],
    )
    .unwrap();
    assert_eq!(
        setup
            .shells
            .shells()
            .iter()
            .map(|s| s.l)
            .collect::<Vec<_>>(),
        [0, 0, 0, 1, 2, 2]
    );
    for (size, expected) in [("TZ", 6), ("QZ", 8)] {
        let mut input = input.clone();
        input["basis_size"] = json!(size);
        let setup = create_setup(
            &ctx,
            parse_input_json(&input).unwrap(),
            vec![dataset(&ctx, "lda-sr", "Si")],
        )
        .unwrap();
        assert_eq!(setup.shells.shells().len(), expected, "{size}");
    }
    input["basis_size"] = json!("invalid");
    let error = create_setup(&ctx, parse_input_json(&input).unwrap(), vec![])
        .err()
        .unwrap();
    assert!(error.to_string().contains("basis size"));
}

#[test]
fn polarized_tables_match_the_active_silicon_basis() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Si");
    input["symmetry"] = json!(false);
    input["lattice"] = json!([12., 0., 0., 0., 12., 0., 0., 0., 12.]);
    input["basis_size"] = json!("DZ2P");
    let setup = create_setup(
        &ctx,
        parse_input_json(&input).unwrap(),
        vec![dataset(&ctx, "lda-sr", "Si")],
    )
    .unwrap();
    assert_eq!(
        setup
            .shells
            .shells()
            .iter()
            .map(|s| s.l)
            .collect::<Vec<_>>(),
        [0, 0, 1, 1, 2, 2]
    );
    let reference = [
        [
            4.670214882556094e-06,
            0.04511979845325068,
            0.26676883894382064,
        ],
        [
            6.269404676874154e-06,
            0.060151379617870065,
            0.272320013776184,
        ],
    ];
    for (table, expected) in setup.description.as_ref().unwrap().tables[4..]
        .iter()
        .zip(reference)
    {
        for (index, value) in [0, 100, 500].into_iter().zip(expected) {
            assert!(
                (table.values[index] - value).abs() < 1e-9,
                "polarization index={index}: {} != {value}",
                table.values[index]
            );
        }
    }
}

#[test]
fn f_channel_setup_builds_symmetry_through_multipole_degree_six() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Ce");
    let setup = create_setup(
        &ctx,
        parse_input_json(&input).unwrap(),
        vec![dataset(&ctx, "lda-sr", "Ce")],
    )
    .unwrap();
    assert!(setup.channels.iter().any(|&(_, l, _)| l == 6));
    let symmetry = setup.symmetry.as_ref().unwrap();
    assert_eq!(symmetry.harmonic(0, 6).shape(), (13, 13));
}

#[test]
fn mixed_channel_moments_match_the_existing_silicon_builder() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let silicon = dataset(&ctx, "lda-sr", "Si");
    let mut input = request();
    input["atoms"][0]["symbol"] = json!("Si");
    input["symmetry"] = json!(false);
    let setup = create_setup(&ctx, parse_input_json(&input).unwrap(), vec![silicon]).unwrap();
    let projector = setup
        .projectors
        .iter()
        .find(|p| p.l == 0 && p.m == 0 && p.index == 0)
        .unwrap();
    let ds = &setup.datasets[0];
    let radius = ds.r.as_slice()[100];
    let x = [radius];
    let zero = [0.];
    let images = [[0., 0., 0.]];
    let phases = [num_complex::Complex64::new(1., 0.)];
    let spline = projector.radial.spline(0).unwrap();
    let call = mandacaru_core::kernels::ShellImages {
        xyz: [&x, &zero, &zero],
        center: projector.center,
        images: &images,
        phases: &phases,
        support: projector.radial.extent(),
        spline: &spline,
        ms: &[0],
        derivative: -1,
    };
    let mut sample = [num_complex::Complex64::new(0., 0.)];
    mandacaru_core::kernels::bloch_shell(&ctx, &call, &[0], 1, &mut sample).unwrap();
    let expected =
        ds.channels[&0].raw_projectors[0].as_slice()[100] / (4. * std::f64::consts::PI).sqrt();
    assert!((sample[0].re - expected).abs() < 1e-10);
    let operators = mandacaru_core::periodic::setup::blocks::operators(
        &setup.projectors,
        &setup.datasets,
        &setup.channels,
    )
    .unwrap();
    let raw = mandacaru_core::periodic::setup::blocks::raw_operators(
        &setup.projectors,
        &setup.datasets,
        &setup.channels,
    )
    .unwrap();
    for l in [0u32, 1] {
        let channel = &setup.datasets[0].channels[&l];
        let n = channel.projectors.len();
        let (d, q) = mandacaru_core::paw::basis_blocks::projector_basis_transform(
            channel.vanderbilt.data(),
            channel.coupling.data(),
            channel.overlap_correction.data(),
            n,
            "raw",
        )
        .unwrap();
        for i in 0..n {
            for j in 0..n {
                let a = setup
                    .projectors
                    .iter()
                    .position(|p| p.l == l as usize && p.m == 0 && p.index == i)
                    .unwrap();
                let b = setup
                    .projectors
                    .iter()
                    .position(|p| p.l == l as usize && p.m == 0 && p.index == j)
                    .unwrap();
                assert!((raw.ion_operator[(a, b)].re - d[i * n + j]).abs() < 1e-10);
                assert!((raw.augmentation[(a, b)].re - q[i * n + j]).abs() < 1e-10);
            }
        }
    }
    let moment = setup.channels.iter().position(|&c| c == (0, 1, 0)).unwrap();
    // Dual s-p L=1 moments from the production builder, lda-sr/Si.parquet
    // SHA-256 18d1efd8cf71adfcfd6c4394db8014bac2db78393fa3e455b43d8a0ad008f9db.
    let reference = [
        [0.012003452533096714, 0.019985499800347216],
        [0.020861574600437794, 0.035795712600606516],
    ];
    for (i, row) in reference.iter().enumerate() {
        for (j, expected) in row.iter().enumerate() {
            let a = setup
                .projectors
                .iter()
                .position(|p| p.l == 0 && p.m == 0 && p.index == i)
                .unwrap();
            let b = setup
                .projectors
                .iter()
                .position(|p| p.l == 1 && p.m == 0 && p.index == j)
                .unwrap();
            let value = operators.blocks[moment][(a, b)];
            let target = expected / (4. * std::f64::consts::PI).sqrt();
            assert!(
                (value.re - target).abs() < 1e-10 && value.im.abs() < 1e-12,
                "s-p moment ({i},{j}) is {value}, expected {target}"
            );
        }
    }
    let raw_reference = [
        [14.997021901772492, -5.365223024124088],
        [-6.245847053407676, 2.2594748153863],
    ];
    for (i, row) in raw_reference.iter().enumerate() {
        for (j, expected) in row.iter().enumerate() {
            let a = setup
                .projectors
                .iter()
                .position(|p| p.l == 0 && p.m == 0 && p.index == i)
                .unwrap();
            let b = setup
                .projectors
                .iter()
                .position(|p| p.l == 1 && p.m == 0 && p.index == j)
                .unwrap();
            let target = expected / (4. * std::f64::consts::PI).sqrt();
            assert!(
                (raw.blocks[moment][(a, b)].re - target).abs() < 1e-8,
                "raw moment {i},{j}: {} expected {target}",
                raw.blocks[moment][(a, b)].re
            );
        }
    }
}
