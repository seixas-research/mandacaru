//! Integration test verifying DFT+U crystal forces and stresses wiring.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;
mod setup_shells_support;

use mandacaru_core::{
    models::{
        hubbard::Form,
        hubbard_request::Request,
        hubbard_slater::Ratios,
        hubbard_spinor::{
            require_time_reversal_invariant, Layout as SpinorLayout, Occupation as SpinorOcc,
        },
    },
    paw::digest::dataset_content_sha256,
    periodic::{
        gradient::{scf, Derivatives},
        hubbard::{Seed, Shell, Spec},
        response::{forces as crystal_forces, stress_all_entries},
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options as SolverOptions,
        stress_terms::crystal::{crystal_strain, crystal_stress, refuse as stress_refuse},
    },
    Context, Error,
};
use nalgebra::{Matrix3, Vector3};
use periodic_gradient_crystals_support::{inputs, provider, Data};
use periodic_stress_crystals_support::load;
use setup_shells_support::dataset;
use std::sync::Arc;

fn dummy_shell(u: f64) -> Shell {
    Shell {
        atom: 0,
        symbol: "Li".into(),
        l: 0,
        u,
        j: 0.,
        ratios: Ratios {
            f4_f2: 0.625,
            f6_f2: None,
        },
        projectors: vec![0],
        weights: vec![1.],
        basis: None,
        source: None,
        seed: Seed::Moment {
            n_electrons: 1.,
            moment: 0.,
        },
    }
}

#[test]
fn hubbard_crystal_forces_and_stress_refusals_and_acceptance() {
    let mut data = Data::load("lih");

    // 1. Inactive/None hubbard hook returns None
    assert!(data.crystal.hubbard_hook().is_none());
    scf::refuse(&data.crystal).expect("plain crystal is differentiated");
    stress_refuse(&data.crystal).expect("plain crystal stress is differentiated");

    // Inactive hubbard (U - J = 0)
    data.crystal.hubbard = Some(Spec {
        form: Form::Dudarev,
        shells: vec![dummy_shell(0.)],
    });
    assert!(!data.crystal.hubbard.as_ref().unwrap().is_active());
    assert!(data.crystal.hubbard_hook().is_none());
    scf::refuse(&data.crystal).expect("inactive hubbard crystal is differentiated");
    stress_refuse(&data.crystal).expect("inactive hubbard crystal stress is differentiated");

    // 2. Active hubbard (U = 0.1) on semilocal scalar crystal is ACCEPTED
    data.crystal.hubbard = Some(Spec {
        form: Form::Dudarev,
        shells: vec![dummy_shell(0.1)],
    });
    assert!(data.crystal.hubbard.as_ref().unwrap().is_active());
    assert!(data.crystal.hubbard_hook().is_some());
    scf::refuse(&data.crystal).expect("active DFT+U semilocal crystal must be accepted");
    stress_refuse(&data.crystal).expect("active DFT+U semilocal crystal stress must be accepted");

    // 3. Active Dudarev and Liechtenstein DFT+U on spinor crystal are both ACCEPTED
    data.crystal.spinor = true;
    scf::refuse(&data.crystal).expect("Dudarev spinor DFT+U forces must be accepted");
    stress_refuse(&data.crystal).expect("Dudarev spinor DFT+U stress must be accepted");
    data.crystal.hubbard.as_mut().unwrap().form = Form::Liechtenstein;
    scf::refuse(&data.crystal).expect("Liechtenstein spinor DFT+U forces must be accepted");
    stress_refuse(&data.crystal).expect("Liechtenstein spinor DFT+U stress must be accepted");
    data.crystal.hubbard.as_mut().unwrap().form = Form::Dudarev;
    data.crystal.spinor = false;

    // 4. Active hubbard on hybrid crystal must be REFUSED
    let original_functional = data.crystal.functional.clone();
    data.crystal.functional = "hse06".to_string();
    let err_hyb = scf::refuse(&data.crystal).expect_err("hybrid DFT+U forces must be refused");
    assert!(
        matches!(&err_hyb, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err_hyb:?}"
    );
    let err_hyb_stress =
        stress_refuse(&data.crystal).expect_err("hybrid DFT+U stress must be refused");
    assert!(
        matches!(&err_hyb_stress, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err_hyb_stress:?}"
    );
    data.crystal.functional = original_functional;
}

#[test]
fn hubbard_provider_and_stress_wiring() {
    let mut data = load("lih");
    data.crystal.hubbard = Some(Spec {
        form: Form::Dudarev,
        shells: vec![dummy_shell(0.2)],
    });

    let cores = data.cores();
    let prov = provider(&data, &cores);

    // Provider automatically carries the hubbard hook when crystal has active hubbard
    assert!(prov.hubbard().is_some());

    // Solve and obtain state
    let result = data.solve();
    let state = data.state(&result);

    // crystal_strain should successfully construct and carry the hubbard operator
    let strain_result = crystal_strain(
        &data.crystal.context,
        inputs(&data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        &data.options,
        &state,
        None,
    );
    assert!(
        strain_result.is_ok(),
        "crystal_strain failed: {:?}",
        strain_result.err()
    );

    let c_strain = strain_result.unwrap();
    // Verify frozen.hubbard is populated!
    assert!(
        c_strain.frozen.hubbard.is_some(),
        "frozen.hubbard must be populated in CrystalStrain"
    );
    let hub_ops = c_strain.frozen.hubbard.as_ref().unwrap();
    assert_eq!(hub_ops.len(), data.crystal.spins);
    assert_eq!(hub_ops[0].shape(), (data.crystal.p, data.crystal.p));

    // Calculate crystal stress
    let stress_res = crystal_stress(
        &data.crystal.context,
        inputs(&data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        &data.options,
        &state,
        None,
    );
    assert!(
        stress_res.is_ok(),
        "crystal_stress failed: {:?}",
        stress_res.err()
    );
    let stress = stress_res.unwrap();
    assert!(stress.volume > 0.0);
    assert!(stress.pressure().is_finite());
}

fn dim_setup_input(
    ctx: &Context,
    library: &str,
    species: [&str; 2],
    lattice: Matrix3<f64>,
    fractional: [[f64; 3]; 2],
    moments: [f64; 2],
    spinor: bool,
    spin: bool,
    mesh: Option<([usize; 3], bool)>,
    symmetry: bool,
    hubbard_json: Option<&str>,
) -> (
    SetupInput,
    Vec<Arc<mandacaru_core::paw::record::PawDataset>>,
) {
    let root =
        std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(library);
    let files: Vec<_> = species
        .iter()
        .map(|s| {
            let path = root.join(format!("{s}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                dataset_content_sha256(ctx, &path).unwrap(),
            )
        })
        .collect();
    let datasets: Vec<_> = species
        .iter()
        .map(|s| Arc::new(dataset(ctx, library, s)))
        .collect();
    let (kpoints_frac, weights) = if mesh.is_some() {
        (vec![], vec![])
    } else {
        (vec![[-0.25, 0., 0.], [0.25, 0., 0.]], vec![0.5, 0.5])
    };
    let input = SetupInput {
        mesh,
        recipes: None,
        projector_basis: "raw".into(),
        lattice,
        grid_shape: [20, 20, 20],
        atoms: [0, 1]
            .into_iter()
            .map(|i| AtomSpec {
                species: i,
                symbol: species[i].into(),
                fractional: fractional[i],
                moment: moments[i],
            })
            .collect(),
        kpoints_frac,
        weights,
        functional: "lda".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor,
        spin,
        symmetry,
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(files),
        hubbard_u: hubbard_json.map(|s| Request::parse(s).unwrap()),
        charge: None,
    };
    (input, datasets)
}

fn skew_cuh_lattice() -> Matrix3<f64> {
    // Columns are primitive lattice vectors in Bohr (~4.2 A anisotropic skew cell).
    Matrix3::new(7.8, 0.3, 0.1, 0.2, 8.1, 0.25, 0.15, 0.2, 8.4)
}

fn check_stress_and_forces_fd(
    ctx: &Context,
    library: &str,
    spinor: bool,
    spin: bool,
    moments: [f64; 2],
    entries: &[(usize, usize)],
    hub_spec: &str,
) {
    let lattice = skew_cuh_lattice();
    let fractional = [[0.15, 0.18, 0.20], [0.51, 0.22, 0.23]];
    let (input, datasets) = dim_setup_input(
        ctx,
        library,
        ["Cu", "H"],
        lattice,
        fractional,
        moments,
        spinor,
        spin,
        None,
        false,
        Some(hub_spec),
    );
    let setup = create_setup(ctx, input, datasets).unwrap();
    let crystal = setup.build(ctx).unwrap();
    let opts = SolverOptions {
        tolerance: 1e-11,
        density_tolerance: 1e-9,
        max_iterations: 150,
        ..SolverOptions::default()
    };
    let result = crystal.solve(&opts).unwrap();
    assert!(result.converged, "reference DFT+U SCF did not converge");
    assert!(result.hubbard_energy.is_some());

    // 1. Finite-difference stress oracle (reconverged free energy at fixed discretization pins)
    let stress = stress_all_entries(&crystal, &result).unwrap();
    let h_strain = 5e-4;
    for &(a, b) in entries {
        let eval_strain = |eps_val: f64| -> f64 {
            let mut eps = Matrix3::zeros();
            eps[(a, b)] = eps_val;
            let strained = setup
                .with_geometry(ctx, &((Matrix3::identity() + eps) * lattice), &fractional)
                .unwrap();
            let state = strained.build(ctx).unwrap().solve(&opts).unwrap();
            assert!(state.converged, "strained DFT+U SCF did not converge");
            state.free_energy
        };
        let fd = (eval_strain(h_strain) - eval_strain(-h_strain)) / (2. * h_strain);
        let err = (stress.gradient[a][b] - fd).abs();
        assert!(
            err < 2e-5,
            "spinor={spinor} stress dF/d eps_{a}{b}: analytic {:.8e}, fd {:.8e}, diff {err:.2e}",
            stress.gradient[a][b],
            fd
        );
    }

    // 2. Finite-difference force oracle (-dF/dtau_X) and translational sum rule
    let forces = crystal_forces(&crystal, &result).unwrap();
    let total_force: [f64; 3] = std::array::from_fn(|ax| forces.iter().map(|f| f[ax]).sum::<f64>());
    assert!(
        total_force.iter().all(|f| f.abs() < 5e-4),
        "DFT+U translational sum rule broken: {total_force:?}"
    );
    let inv_lat = lattice.try_inverse().unwrap();
    let h_pos = 1e-3;
    for axis in 0..3 {
        let eval_disp = |disp: f64| -> f64 {
            let mut moved = fractional;
            let mut cart = Vector3::zeros();
            cart[axis] = disp;
            let dfrac = inv_lat * cart;
            for j in 0..3 {
                moved[0][j] += dfrac[j];
            }
            let displaced = setup.with_geometry(ctx, &lattice, &moved).unwrap();
            let state = displaced.build(ctx).unwrap().solve(&opts).unwrap();
            assert!(state.converged, "displaced DFT+U SCF did not converge");
            state.free_energy
        };
        let fd_force = -(eval_disp(h_pos) - eval_disp(-h_pos)) / (2. * h_pos);
        let err = (forces[0][axis] - fd_force).abs();
        assert!(
            err < 5e-5,
            "spinor={spinor} force[0][{axis}]: analytic {:.8e}, fd {:.8e}, diff {err:.2e}",
            forces[0][axis],
            fd_force
        );
    }
}

/// CF-5 gate (Wave E4-2): analytic DFT+U crystal stress and forces match reconverged finite
/// differences on an anisotropic skew PAW-LCAO cell (checking diagonal and shear strains).
#[test]
fn cf5_anisotropic_dft_plus_u_crystal_stress_and_forces_match_finite_differences() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    check_stress_and_forces_fd(
        &ctx,
        "lda-sr",
        false,
        false,
        [0., 0.],
        &[(0, 0), (1, 1), (0, 1)],
        r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.5}}"#,
    );
}

/// Wave E4-1 (`SO-3a`..`SO-6`) & spinor Liechtenstein: spinor periodic DFT+U SCF, time-reversal and
/// magnetic/double-group symmetry completion on reduced k meshes, and spinor +U analytic forces &
/// stress vs finite differences for both Dudarev and Liechtenstein forms.
#[test]
fn spinor_dft_plus_u_scf_symmetry_forces_and_stress() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let opts = SolverOptions {
        tolerance: 1e-10,
        density_tolerance: 1e-8,
        max_iterations: 150,
        ..SolverOptions::default()
    };

    // 1. Symmetric cells: reduced mesh vs full mesh parity for nonmagnetic TR spinor +U (CuH)
    //    and magnetic spinor +U (NiH with open 3d shell, U = 4 eV, J = 0.5 eV), in both Dudarev
    //    and Liechtenstein forms.
    let sym_lat = Matrix3::from_diagonal(&Vector3::new(8.0, 8.0, 8.0));
    let sym_frac = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.35]];
    let cases = [
        (
            false,
            ["Cu", "H"],
            [0., 0.],
            r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.5}}"#,
        ),
        (
            true,
            ["Ni", "H"],
            [2.0, 0.],
            r#"{"Ni": {"l": "d", "U": 4.0, "J": 0.5}}"#,
        ),
        (
            false,
            ["Cu", "H"],
            [0., 0.],
            r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.5, "form": "liechtenstein"}}"#,
        ),
        (
            true,
            ["Ni", "H"],
            [2.0, 0.],
            r#"{"Ni": {"l": "d", "U": 4.0, "J": 0.5, "form": "liechtenstein"}}"#,
        ),
    ];
    for (spin, species, moments, hub_spec) in cases {
        let mut results = Vec::new();
        for symmetry in [true, false] {
            let (input, datasets) = dim_setup_input(
                &ctx,
                "lda-fr",
                species,
                sym_lat,
                sym_frac,
                moments,
                true,
                spin,
                Some(([2, 2, 1], true)),
                symmetry,
                Some(hub_spec),
            );
            let setup = create_setup(&ctx, input, datasets).unwrap();
            let crystal = setup.build(&ctx).unwrap();
            let res = crystal.solve(&opts).unwrap();
            assert!(
                res.converged,
                "spinor +U (spin={spin}, symmetry={symmetry}) did not converge"
            );
            results.push((crystal, res));
        }
        let ((c_sym, r_sym), (c_full, r_full)) = (&results[0], &results[1]);
        assert!(c_sym.points.len() < c_full.points.len());
        assert!(
            (r_sym.free_energy - r_full.free_energy).abs() < 1e-7,
            "spin={spin}: free_energy mismatch sym={} full={}",
            r_sym.free_energy,
            r_full.free_energy
        );
        let eu_sym = r_sym.hubbard_energy.unwrap();
        let eu_full = r_full.hubbard_energy.unwrap();
        assert!(
            (eu_sym - eu_full).abs() < 1e-7,
            "spin={spin}: hubbard_energy mismatch sym={eu_sym} full={eu_full}"
        );
        let occ_sym = &r_sym.hubbard.as_ref().unwrap().occupations[0];
        let occ_full = &r_full.hubbard.as_ref().unwrap().occupations[0];
        let occ_diff = occ_sym
            .iter()
            .zip(occ_full)
            .map(|(a, b)| (a - b).norm())
            .fold(0., f64::max);
        assert!(
            occ_diff < 5e-6,
            "spin={spin}: 10x10 occupation diff reduced vs full = {occ_diff:e}"
        );
        let spinor_occ =
            SpinorOcc::new(&ctx, SpinorLayout::new(2, false).unwrap(), occ_sym).unwrap();
        if !spin {
            require_time_reversal_invariant(&spinor_occ, 1e-10).unwrap();
        } else {
            let svec = spinor_occ.spin_vector();
            assert!(
                svec.iter().map(|x| x * x).sum::<f64>().sqrt() > 1e-2,
                "magnetic spinor +U shell lost spin polarization: {svec:?}"
            );
        }
    }

    // 2. Analytic spinor DFT+U forces and stress vs reconverged finite differences on skew CuH
    //    for both Dudarev and Liechtenstein forms.
    check_stress_and_forces_fd(
        &ctx,
        "lda-fr",
        true,
        false,
        [0., 0.],
        &[(0, 0), (0, 1)],
        r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.5}}"#,
    );
    check_stress_and_forces_fd(
        &ctx,
        "lda-fr",
        true,
        false,
        [0., 0.],
        &[(0, 0), (0, 1)],
        r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.5, "form": "liechtenstein"}}"#,
    );
}

fn check_liechtenstein_functionals_fd(
    spinor: bool,
    spin: bool,
    library: &str,
    species: [&str; 2],
    moments: [f64; 2],
    hub_spec: &str,
) {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let opts = SolverOptions {
        tolerance: 1e-10,
        density_tolerance: 1e-8,
        width: 0.01,
        max_iterations: 300,
        ..SolverOptions::default()
    };
    let lattice = skew_cuh_lattice();
    let fractional = [[0.15, 0.18, 0.20], [0.51, 0.22, 0.23]];
    let inv_lat = lattice.try_inverse().unwrap();

    for functional in ["pbe", "pbe+rvv10", "r2scan", "r2scan+rvv10"] {
        let (mut input, datasets) = dim_setup_input(
            &ctx,
            library,
            species,
            lattice,
            fractional,
            moments,
            spinor,
            spin,
            None,
            false,
            Some(hub_spec),
        );
        input.functional = functional.into();
        let setup = create_setup(&ctx, input, datasets).unwrap();
        let crystal = setup.build(&ctx).unwrap();
        let result = crystal.solve(&opts).unwrap();
        assert!(
            result.converged,
            "E8-6 {functional} (spinor={spinor}, spin={spin}) SCF did not converge"
        );
        assert!(result.hubbard_energy.is_some());

        // Stress vs reconverged central FD on axial eps_(0, 0) and shear eps_(0, 1).
        let stress = stress_all_entries(&crystal, &result).unwrap();
        let h_strain = 5e-4;
        for (a, b) in [(0, 0), (0, 1)] {
            let eval_strain = |eps_val: f64| -> f64 {
                let mut eps = Matrix3::zeros();
                eps[(a, b)] = eps_val;
                let strained = setup
                    .with_geometry(&ctx, &((Matrix3::identity() + eps) * lattice), &fractional)
                    .unwrap();
                let state = strained
                    .build(&ctx)
                    .unwrap()
                    .solve_with(&opts, Some(&result), None)
                    .unwrap();
                assert!(
                    state.converged,
                    "strained {functional} (spinor={spinor}, spin={spin}) SCF failed"
                );
                state.free_energy
            };
            let fd = (eval_strain(h_strain) - eval_strain(-h_strain)) / (2. * h_strain);
            let err = (stress.gradient[a][b] - fd).abs();
            assert!(
                err < 5e-5,
                "E8-6 {functional} (spinor={spinor}, spin={spin}) stress dF/d eps_{a}{b}: analytic {:.8e}, fd {:.8e}, diff {err:.2e}",
                stress.gradient[a][b],
                fd
            );
        }

        // Force vs reconverged central FD along axis 0 + translational sum rule.
        let forces = crystal_forces(&crystal, &result).unwrap();
        let total_force: [f64; 3] =
            std::array::from_fn(|ax| forces.iter().map(|f| f[ax]).sum::<f64>());
        assert!(
            total_force.iter().all(|f| f.abs() < 1e-2),
            "E8-6 {functional} (spinor={spinor}, spin={spin}) translational sum rule: {total_force:?}"
        );
        let h_pos = 1e-3;
        let eval_disp = |disp: f64| -> f64 {
            let mut moved = fractional;
            let dfrac = inv_lat * Vector3::new(disp, 0., 0.);
            for j in 0..3 {
                moved[0][j] += dfrac[j];
            }
            let displaced = setup.with_geometry(&ctx, &lattice, &moved).unwrap();
            let state = displaced
                .build(&ctx)
                .unwrap()
                .solve_with(&opts, Some(&result), None)
                .unwrap();
            assert!(
                state.converged,
                "displaced {functional} (spinor={spinor}, spin={spin}) SCF failed"
            );
            state.free_energy
        };
        let fd_force = -(eval_disp(h_pos) - eval_disp(-h_pos)) / (2. * h_pos);
        let err = (forces[0][0] - fd_force).abs();
        assert!(
            err < 5e-5,
            "E8-6 {functional} (spinor={spinor}, spin={spin}) force[0][0]: analytic {:.8e}, fd {:.8e}, diff {err:.2e}",
            forces[0][0],
            fd_force
        );
    }
}

#[test]
fn e8_6_liechtenstein_unpolarized_forces_and_stress() {
    check_liechtenstein_functionals_fd(
        false,
        false,
        "pbe-sr",
        ["Cu", "H"],
        [0., 0.],
        r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.6, "form": "liechtenstein", "f4_f2": 0.7}}"#,
    );
}

#[test]
#[ignore = "slow: ~1217 s for NiH r2scan+rvv10 integer-start sweep (verified 2026-11-18); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_crystal_forces_and_stress -- --ignored e8_6_liechtenstein_spin_polarized"]
fn e8_6_liechtenstein_spin_polarized_forces_and_stress() {
    check_liechtenstein_functionals_fd(
        false,
        true,
        "pbe-sr",
        ["Ni", "H"],
        [2.0, 0.],
        r#"{"Ni": {"l": "d", "U": 4.0, "J": 0.6, "form": "liechtenstein", "f4_f2": 0.7}}"#,
    );
}

#[test]
fn e8_6_liechtenstein_spinor_forces_and_stress() {
    check_liechtenstein_functionals_fd(
        true,
        false,
        "pbe-fr",
        ["Cu", "H"],
        [0., 0.],
        r#"{"Cu": {"l": "d", "U": 4.0, "J": 0.6, "form": "liechtenstein", "f4_f2": 0.7}}"#,
    );
}
