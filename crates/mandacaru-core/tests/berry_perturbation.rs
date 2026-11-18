//! B3 (`periodic::berry::perturbation`): the folding of polarization steps (an electronic branch
//! jump and an ion crossing the cell boundary carry whole quanta), the quarter-quantum refusal,
//! the assembly of Born charges and of the proper piezoelectric tensor from folded steps against
//! sums written out by hand, the Richardson combination, the acoustic sum rule (reported, imposed
//! on request), the Voigt strains, and the setup perturber's refusals. Model-free and fast, except
//! the last test, which builds a crystal (`MANDACARU_PAW_PATH`).
mod berry_support;
mod setup_shells_support;

use mandacaru_core::{
    math::linalg::real_eigen,
    periodic::{
        berry::{
            perturbation::{
                acoustic_sum_rule, born_charges, born_effective_charges_with, channel_steps, fold,
                folded_step, impose, piezoelectric, piezoelectric_tensor,
                relaxed_piezoelectric_from_responses, richardson, voigt_strain, Perturbation,
                Perturber, SetupPerturber, ASR_WARNING,
            },
            polarization::{assemble, polarization},
        },
        bloch_mesh::Sampler,
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options,
    },
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;

fn cell() -> Matrix3<f64> {
    Matrix3::new(6., 0.5, 0., 0., 7., 0.3, 0., 0., 8.)
}

#[test]
fn a_whole_quantum_from_a_branch_jump_or_an_ion_crossing_the_boundary_folds_away() {
    let lattice = cell();
    let charges = [1., 1.];
    // A branch jump: the same physical state with one string phase shifted by 2 pi.
    let minus = assemble(
        lattice,
        &[[0.5, 1., 1.], [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    let plus = assemble(
        lattice,
        &[[0.5, 1., 1.], [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3 + 1e-3 - 2. * PI, -0.2, 0.1]],
    )
    .unwrap();
    let step = folded_step(&plus, &minus, "jump").unwrap();
    let expected = -2. * 1e-3 / (2. * PI); // f = 2, g = gcd(2, 1, 1) = 1
    assert!((step[0] - expected).abs() < 1e-14, "{step:?}");
    assert!(step[1].abs() < 1e-15 && step[2].abs() < 1e-15);
    // An ion crossing the boundary: atom 0 at s = -1e-4 against s = 1 - 1e-4 changes `raw` by
    // exactly Z_0 / g = 1 quantum along a_0, nothing after folding.
    let a0 = lattice.column(0);
    let near = |s: f64| [s * a0[0] + 1., s * a0[1] + 1., s * a0[2] + 1.];
    let inside = assemble(
        lattice,
        &[near(1. - 1e-4), [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    let outside = assemble(
        lattice,
        &[near(-1e-4), [3., 3., 4.]],
        &charges,
        2.,
        vec![[0.3, -0.2, 0.1]],
    )
    .unwrap();
    assert!((inside.raw[0] - outside.raw[0] - 1.).abs() < 1e-9);
    let crossed = folded_step(&inside, &outside, "crossing").unwrap();
    assert!(crossed.iter().all(|d| d.abs() < 1e-9), "{crossed:?}");
    assert_eq!(fold(0.7), 0.7 - 1.);
}

#[test]
fn a_step_beyond_a_quarter_quantum_is_refused() {
    let lattice = cell();
    let a = assemble(lattice, &[[1., 1., 1.]], &[1.], 2., vec![[0., 0., 0.]]).unwrap();
    let b = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        2.,
        vec![[2. * PI * 0.15, 0., 0.]],
    )
    .unwrap();
    // f = 2: the electronic coordinate moves by -0.3 of a quantum.
    let message = folded_step(&b, &a, "atom 0").unwrap_err().to_string();
    assert!(
        message.contains("atom 0") && message.contains("more than a quarter"),
        "{message}"
    );
}

#[test]
fn channel_steps_fold_modulo_the_channel_period() {
    let lattice = cell();
    // Two spin channels (f = 1), g = gcd(1, 1) = 1: a channel's coordinate is continuous modulo 1.
    let minus = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        1.,
        vec![[0.1, 0., 0.], [0.2, 0., 0.]],
    )
    .unwrap();
    let plus = assemble(
        lattice,
        &[[1., 1., 1.]],
        &[1.],
        1.,
        vec![[0.1 + 2e-3, 0., 0.], [0.2 - 2. * PI + 1e-3, 0., 0.]],
    )
    .unwrap();
    let steps = channel_steps(&plus, &minus);
    assert!((steps[0][0] + 2e-3 / (2. * PI)).abs() < 1e-14);
    assert!((steps[1][0] + 1e-3 / (2. * PI)).abs() < 1e-14);
}

#[test]
fn born_charges_and_the_piezoelectric_tensor_assemble_as_written_by_hand() {
    let lattice = cell();
    let p = assemble(lattice, &[[1., 1., 1.]], &[1.], 2., vec![[0., 0., 0.]]).unwrap();
    let (quanta, volume) = (p.quanta, p.volume);
    let mut steps = vec![[[0.; 3]; 3]; 2];
    steps[0][1] = [1e-3, -2e-3, 5e-4];
    steps[1][2] = [-3e-4, 0., 7e-4];
    let delta = 0.02;
    let z = born_charges(&quanta, volume, &steps, delta);
    for (a, j) in [(0, 1), (1, 2)] {
        for i in 0..3 {
            let dp: f64 = (0..3)
                .map(|alpha| steps[a][j][alpha] * quanta[alpha][i])
                .sum();
            assert!((z[a][i][j] - volume * dp / (2. * delta)).abs() < 1e-15);
        }
    }
    assert_eq!(z[0][0][0], 0.);
    // The quanta are g a_j / Omega: one quantum along a_0 per unit displacement gives Z* = a_0.
    let mut unit = vec![[[0.; 3]; 3]; 1];
    unit[0][0] = [2. * delta, 0., 0.];
    let z = born_charges(&quanta, volume, &unit, delta);
    for i in 0..3 {
        assert!((z[0][i][0] - lattice[(i, 0)]).abs() < 1e-13);
    }
    let mut strain_steps = [[0.; 3]; 6];
    strain_steps[3] = [0., 4e-4, -1e-4];
    let e = piezoelectric(&quanta, &strain_steps, 0.005);
    for i in 0..3 {
        let dp: f64 = (0..3)
            .map(|alpha| strain_steps[3][alpha] * quanta[alpha][i])
            .sum();
        assert!((e[i][3] - dp / 0.01).abs() < 1e-15);
        assert_eq!(e[i][0], 0.);
    }
}

#[test]
fn richardson_removes_the_second_order_error_of_a_central_difference() {
    // A central difference of f(x) = x^3 + x^5 at 0: f'(0) + h^2 f'''/6 + O(h^4) = h^2 + h^4.
    let central = |h: f64| h * h + h.powi(4);
    let r = richardson(central(0.1), central(0.05));
    assert!(r.abs() < 1e-4 && r.abs() < central(0.05) / 10., "{r}");
}

#[test]
fn the_acoustic_sum_rule_is_reported_and_imposed_only_on_request() {
    let z = vec![
        [[1.1, 0.01, 0.], [0., 1.1, 0.], [0., 0., 1.1]],
        [[-1.0, 0., 0.], [0., -1.0, 0.], [0., 0.02, -1.0]],
    ];
    let rule = acoustic_sum_rule(&z);
    assert!((rule.residual - 0.1).abs() < 1e-14);
    assert!(rule.residual > ASR_WARNING && rule.warning.as_ref().unwrap().contains("0.100 e"));
    let fixed = impose(&z, &rule);
    assert!(acoustic_sum_rule(&fixed).residual < 1e-15);
    assert!(acoustic_sum_rule(&fixed).warning.is_none());
    // Imposing removes the mean from every atom and keeps their differences.
    assert!(((fixed[0][0][0] - fixed[1][0][0]) - (z[0][0][0] - z[1][0][0])).abs() < 1e-15);
}

#[test]
fn voigt_strains_are_symmetric_with_engineering_shear() {
    let xx = voigt_strain(0, 0.01).unwrap();
    assert_eq!(xx[(0, 0)], 0.01);
    let yz = voigt_strain(3, 0.01).unwrap();
    assert_eq!((yz[(1, 2)], yz[(2, 1)]), (0.005, 0.005));
    let xy = voigt_strain(5, 0.01).unwrap();
    assert_eq!((xy[(0, 1)], xy[(1, 0)]), (0.005, 0.005));
    assert!(voigt_strain(6, 0.01).is_err());
    assert!(Perturbation::Strain {
        voigt: 4,
        delta: -0.005
    }
    .name()
    .contains("xz"));
}

#[test]
fn the_setup_perturber_rebuilds_strained_and_displaced_geometries() {
    let solved = berry_support::solve(&berry_support::h2_chain("lda-sr", "lda"));
    let perturber = SetupPerturber {
        setup: &solved.setup,
    };
    let strained = perturber
        .setup(
            &solved.ctx,
            Perturbation::Strain {
                voigt: 0,
                delta: 0.005,
            },
        )
        .unwrap();
    assert!((strained.lattice[(0, 0)] - 1.005 * solved.setup.lattice[(0, 0)]).abs() < 1e-12);
    assert_eq!(strained.grid.index.shape(), solved.setup.grid.index.shape());

    let displaced = perturber
        .setup(
            &solved.ctx,
            Perturbation::Displacement {
                atom: 0,
                axis: 0,
                delta: 0.02,
            },
        )
        .unwrap();
    assert_eq!(
        displaced.grid.index.shape(),
        solved.setup.grid.index.shape()
    );
    let base_frac = solved.setup.fractional().unwrap();
    let moved_frac = displaced.fractional().unwrap();
    let expected_df = 0.02 / solved.setup.lattice[(0, 0)];
    assert!((moved_frac[0][0] - base_frac[0][0] - expected_df).abs() < 1e-12);
    assert!((moved_frac[1][0] - base_frac[1][0]).abs() < 1e-12);
}

#[test]
fn relaxed_piezoelectric_projects_translations_and_refuses_soft_modes() {
    // 2-atom diatomic chain along x with spring constant k = 0.5 Ha/Bohr^2 along x and
    // transverse k_perp = 0.25 Ha/Bohr^2: K is symmetric with 3 rigid-translation zero modes.
    let mut k = vec![vec![0.; 6]; 6];
    let ks = [0.5, 0.25, 0.25];
    for l in 0..3 {
        k[l][l] = ks[l];
        k[l][3 + l] = -ks[l];
        k[3 + l][l] = -ks[l];
        k[3 + l][3 + l] = ks[l];
    }
    let charges = vec![
        [[1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]],
        [[-1.2, 0., 0.], [0., -1.2, 0.], [0., 0., -1.2]],
    ];
    let clamped = [[0.; 6]; 3];
    let mut lambda = vec![[0.; 6]; 6];
    lambda[0][0] = 0.1;
    lambda[3][0] = -0.1;
    let out = relaxed_piezoelectric_from_responses(&clamped, 100., &charges, &k, &lambda).unwrap();
    assert!(out.acoustic_eigenvalues.iter().all(|&v| v.abs() < 1e-12));
    // Relative displacement u_0 - u_1 = 0.2 / (2 * 0.5)? Wait: K_eff * u = lambda => 0.5(u0-u1)=0.1,
    // with u0 = -u1 => u0 = 0.1, u1 = -0.1, contribution = (1.2*0.1 + (-1.2)*(-0.1))/100 = 0.0024.
    assert!((out.internal_strain[0][0] - 0.1).abs() < 1e-12);
    assert!((out.internal_strain[3][0] + 0.1).abs() < 1e-12);
    assert!((out.relaxed[0][0] - 0.0024).abs() < 1e-12);

    // Soft optical mode below SOFT_MODE_THRESHOLD = 1e-4 must be refused.
    let mut soft_k = vec![vec![0.; 6]; 6];
    let soft_ks = [5e-5, 0.25, 0.25];
    for l in 0..3 {
        soft_k[l][l] = soft_ks[l];
        soft_k[l][3 + l] = -soft_ks[l];
        soft_k[3 + l][l] = -soft_ks[l];
        soft_k[3 + l][3 + l] = soft_ks[l];
    }
    let err = relaxed_piezoelectric_from_responses(&clamped, 100., &charges, &soft_k, &lambda)
        .unwrap_err()
        .to_string();
    assert!(err.contains("soft or unstable mode"), "{err}");
}

#[test]
fn rigid_grid_shift_keeps_folded_polarization_unchanged() {
    let base = berry_support::lih("lda-sr");
    let shifted_cell = berry_support::shifted(&base, [1, 0, 0]);
    let s0 = berry_support::solve(&base);
    let s1 = berry_support::solve(&shifted_cell);
    let p0 = polarization(&s0.sampler(), [3, 2, 2]).unwrap();
    let p1 = polarization(&s1.sampler(), [3, 2, 2]).unwrap();
    let step = folded_step(&p1, &p0, "rigid grid shift").unwrap();
    assert!(step.iter().all(|v| v.abs() < 1e-8), "{step:?}");
}

#[test]
fn born_charges_symmetry_reduction_and_relaxed_piezoelectric_work_on_crystals() {
    // Symmetric H2 chain: symmetry-reduced Z* matches the full 3N finite-difference Z*.
    let mut cell = berry_support::h2_chain("lda-sr", "lda");
    cell.mesh = [2, 2, 2];
    let base = berry_support::solve(&cell);
    let mut recipe = base.setup.recipe.clone().unwrap();
    recipe.symmetry = true;
    recipe.mesh = Some(([2, 2, 2], true));
    recipe.kpoints_frac.clear();
    recipe.weights.clear();
    let sym_setup = create_setup(&base.ctx, recipe, base.setup.datasets.clone()).unwrap();
    assert!(sym_setup.symmetry.as_ref().is_some_and(|s| s.len() > 1));
    let sym_crystal = sym_setup.build(&base.ctx).unwrap();
    let sym_state = sym_crystal
        .solve(&Options {
            tolerance: 1e-10,
            density_tolerance: 1e-8,
            width: cell.width,
            ..Options::default()
        })
        .unwrap();
    assert!(sym_state.converged);
    let perturber = SetupPerturber { setup: &sym_setup };
    let sampler = Sampler::new(&sym_crystal, &sym_state, &sym_setup).unwrap();
    let full = born_effective_charges_with(
        &base.ctx,
        &sampler,
        &perturber,
        [3, 2, 2],
        0.02,
        false,
        false,
        false,
    )
    .unwrap();
    let reduced = born_effective_charges_with(
        &base.ctx,
        &sampler,
        &perturber,
        [3, 2, 2],
        0.02,
        false,
        false,
        true,
    )
    .unwrap();
    assert!(reduced.iterations.len() < full.iterations.len());
    for a in 0..2 {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reduced.charges[a][i][j] - full.charges[a][i][j]).abs() < 5e-3,
                    "a={a} i={i} j={j}: reduced={} full={}",
                    reduced.charges[a][i][j],
                    full.charges[a][i][j]
                );
            }
        }
    }
}

#[test]
fn polar_lih_soft_mode_gate() {
    // Polar skew LiH (1D chain in skew box): Born charges succeed cleanly,
    // while relaxed-ion refuses its soft rotational mode below 1e-4 Ha/Bohr^2.
    let lih = berry_support::solve(&berry_support::lih("lda-sr"));
    let lih_perturber = SetupPerturber { setup: &lih.setup };
    let lih_sampler = lih.sampler();
    let lih_z = born_effective_charges_with(
        &lih.ctx,
        &lih_sampler,
        &lih_perturber,
        [2, 2, 2],
        0.02,
        false,
        false,
        true,
    )
    .unwrap();
    let max_z = lih_z
        .charges
        .iter()
        .flatten()
        .flatten()
        .fold(0f64, |m, v| m.max(v.abs()));
    assert!(
        max_z > 0.5 && lih_z.sum_rule.residual < 0.25 * max_z,
        "{:?}",
        lih_z.sum_rule
    );
    let soft_err = piezoelectric_tensor(
        &lih.ctx,
        &lih_sampler,
        &lih_perturber,
        [2, 2, 2],
        0.005,
        0.02,
        false,
        false,
        true,
        true,
        None,
    )
    .err()
    .expect("1D LiH chain in skew box has a soft rotational mode")
    .to_string();
    assert!(soft_err.contains("soft or unstable mode"), "{soft_err}");
}

#[test]
fn float32_crystal_is_refused() {
    let cell = berry_support::h2_chain("lda-sr", "lda");
    let base = berry_support::solve(&cell);
    let mut f32_recipe = base.setup.recipe.clone().unwrap();
    f32_recipe.precision = "float32".into();
    let f32_setup = create_setup(&base.ctx, f32_recipe, base.setup.datasets.clone()).unwrap();
    let f32_crystal = f32_setup.build(&base.ctx).unwrap();
    let f32_state = f32_crystal
        .solve(&Options {
            width: cell.width,
            ..Options::default()
        })
        .unwrap();
    let f32_perturber = SetupPerturber { setup: &f32_setup };
    let f32_sampler = Sampler::new(&f32_crystal, &f32_state, &f32_setup).unwrap();
    let f32_err = born_effective_charges_with(
        &base.ctx,
        &f32_sampler,
        &f32_perturber,
        [3, 2, 2],
        0.02,
        false,
        false,
        true,
    )
    .err()
    .expect("float32 must be refused for finite-difference Z*")
    .to_string();
    assert!(f32_err.contains("float64"), "{f32_err}");
}

#[test]
#[ignore = "slow: zincblende BN symmetry-reduced Z* and relaxed-ion piezoelectric (~15 min); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_perturbation -- --ignored zincblende_bn_born_charges_symmetry_and_relaxed_piezoelectric"]
fn zincblende_bn_born_charges_symmetry_and_relaxed_piezoelectric() {
    // Zincblende BN (equilibrium cubic T_d crystal): symmetry-reduced Z* & clamped/relaxed piezo.
    let a = 3.615 / 0.529_177_210_903;
    let h = 0.5 * a;
    let lattice = Matrix3::new(0., h, h, h, 0., h, h, h, 0.);
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let root =
        std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join("lda-sr");
    let dataset_files: Vec<(String, String)> = ["B", "N"]
        .iter()
        .map(|s| {
            let path = root.join(format!("{s}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                mandacaru_core::paw::digest::dataset_content_sha256(&ctx, &path).unwrap(),
            )
        })
        .collect();
    let ds_b = std::sync::Arc::new(setup_shells_support::dataset(&ctx, "lda-sr", "B"));
    let ds_n = std::sync::Arc::new(setup_shells_support::dataset(&ctx, "lda-sr", "N"));
    let input = SetupInput {
        lattice,
        grid_shape: [12, 12, 12],
        atoms: vec![
            AtomSpec {
                species: 0,
                symbol: "B".into(),
                fractional: [0.; 3],
                moment: 0.,
            },
            AtomSpec {
                species: 1,
                symbol: "N".into(),
                fractional: [0.25; 3],
                moment: 0.,
            },
        ],
        kpoints_frac: vec![],
        weights: vec![],
        mesh: Some(([2, 2, 2], true)),
        functional: "lda".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: false,
        symmetry: true,
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(dataset_files),
        hubbard_u: None,
        charge: None,
        recipes: None,
        projector_basis: "raw".into(),
    };
    let bn_setup = create_setup(&ctx, input, vec![ds_b, ds_n]).unwrap();
    assert!(bn_setup.symmetry.as_ref().is_some_and(|s| s.len() > 1));
    let bn_crystal = bn_setup.build(&ctx).unwrap();
    let bn_state = bn_crystal
        .solve(&Options {
            tolerance: 1e-8,
            density_tolerance: 1e-6,
            ..Options::default()
        })
        .unwrap();
    assert!(bn_state.converged);
    let bn_perturber = SetupPerturber { setup: &bn_setup };
    let bn_sampler = Sampler::new(&bn_crystal, &bn_state, &bn_setup).unwrap();
    let piezo = piezoelectric_tensor(
        &ctx,
        &bn_sampler,
        &bn_perturber,
        [2, 2, 2],
        0.005,
        0.02,
        false,
        false,
        true,
        true,
        None,
    )
    .unwrap();
    assert!(piezo.relaxed.is_some());
    // T_d cubic: only 2 displacements needed (1 per atom), instead of 6!
    assert_eq!(piezo.born_charges.as_ref().unwrap().iterations.len(), 4);
    let k = piezo.force_constants.as_ref().unwrap();
    let dim = k.len();
    for r in 0..dim {
        for c in 0..dim {
            assert!((k[r][c] - k[c][r]).abs() < 1e-12);
        }
    }
    let k_mat = DMatrix::from_fn(dim, dim, |r, c| k[r][c]);
    let (eigs, _) = real_eigen(k_mat).unwrap();
    assert!(
        eigs[..3].iter().all(|v| v.abs() < 5e-3),
        "acoustic={:?}",
        &eigs[..3]
    );
    assert!(
        eigs[3..].iter().all(|v| *v > 0.1),
        "optical={:?}",
        &eigs[3..]
    );
    let z = piezo.born_charges.as_ref().unwrap();
    // B nominal/Born charge positive (~ +1.8..+2.0 e), N negative, isotropic T_d diagonal.
    assert!(
        z.charges[0][0][0] > 0.5 && z.charges[1][0][0] < -0.5,
        "{:?}",
        z.charges
    );
    assert!((z.charges[0][0][0] - z.charges[0][1][1]).abs() < 1e-4);
    assert!(z.sum_rule.residual < 0.05, "{:?}", z.sum_rule);
    // Zincblende e_14 = e_25 = e_36 nonzero and equal, others ~0.
    let e_clamped = piezo.clamped;
    assert!(e_clamped[0][3].abs() > 1e-4, "{e_clamped:?}");
    assert!((e_clamped[0][3] - e_clamped[1][4]).abs() < 2e-3);
    assert!((e_clamped[0][3] - e_clamped[2][5]).abs() < 2e-3);
    assert!(e_clamped[0][0].abs() < 2e-3);
}
