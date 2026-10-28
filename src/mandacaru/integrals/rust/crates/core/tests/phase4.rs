use mandacaru_core::{
    molecular::{atom, basis::Basis, radial, xc, MolecularIntegrals, ScfOptions, SpatialGrid},
    Context, Geometry, LengthUnit,
};
fn geometry(displacement: f64) -> Geometry {
    Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87 + displacement]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap()
}
#[test]
fn standalone_kohn_sham_and_stationary_hf_gradients_follow_the_energy() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let grid = SpatialGrid::from_geometry(&geometry(0.), [0.35 / 0.529177210903; 3], false, false)
        .unwrap();
    let make = |delta| {
        MolecularIntegrals::new(
            &ctx,
            geometry(delta),
            grid.clone(),
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
        )
        .unwrap()
    };
    let integrals = make(0.);
    for name in ["rhf", "lda", "pbe", "r2scan", "hse06"] {
        let options = ScfOptions {
            tolerance: 1e-12,
            ..Default::default()
        };
        let solve = |mol: &MolecularIntegrals| {
            if name == "rhf" {
                mol.mean_field("rhf", [1, 1], &options)
            } else {
                mol.kohn_sham(name, [1, 1], false, &options)
            }
            .unwrap()
        };
        let state = solve(&integrals);
        assert!(state.converged);
        let gradient = integrals.gradient(&state).unwrap();
        let eps = 1e-5;
        let energy = |delta| {
            let mol = make(delta);
            solve(&mol).electronic_energy + mol.nuclear
        };
        let fd = (energy(eps) - energy(-eps)) / (2. * eps) * 0.529177210903;
        assert!(
            (fd - gradient.total.as_slice()[5]).abs() < 1e-7,
            "{name}: finite difference {fd}, analytic {}",
            gradient.total.as_slice()[5]
        );
        if name == "lda" {
            assert!(
                (state.electronic_energy
                    + integrals.nuclear
                    + 25.804963910710086 / 27.211386245988)
                    .abs()
                    < 1e-12
            );
        }
    }
}
#[test]
fn xc_partials_are_energy_derivatives_and_respect_spin_swap() {
    for name in ["lda", "pbe"] {
        let point = [0.12, 0.03, 0.015, -0.002, 0.008];
        let (f, partials) = xc::partials(name, point).unwrap();
        let swapped =
            xc::partials(name, [point[1], point[0], point[4], point[3], point[2]]).unwrap();
        assert!((f - swapped.0).abs() < 1e-15);
        for axis in 0..5 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += 1e-7;
            minus[axis] -= 1e-7;
            let fd =
                (xc::partials(name, plus).unwrap().0 - xc::partials(name, minus).unwrap().0) / 2e-7;
            assert!((fd - partials[axis]).abs() < 1e-8);
        }
    }
}
#[test]
fn analytic_basis_gradients_are_regular_at_poles_and_match_displacements() {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024).unwrap();
    let atom = Geometry::new(
        vec![8],
        vec![[0.; 3]],
        [[0.; 3]; 3],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    for spec in [
        r#"{"name":"HAO","virtual_orbitals":2}"#,
        r#"{"name":"NAO","n_grid":400,"r_c":7.0,"n_zeta":2,"n_polarization":1,"split_norm":0.15,"tail_norms":[0.2]}"#,
    ] {
        let basis = Basis::build(&ctx, &atom, spec).unwrap();
        for orbital in &basis.orbitals {
            for point in [[0.5, 0.7, -0.3], [0., 0., 1.], [0., 0., -1.]] {
                let derivative = orbital.gradient(point);
                for axis in 0..3 {
                    let mut plus = point;
                    let mut minus = point;
                    plus[axis] += 1e-5;
                    minus[axis] -= 1e-5;
                    let fd = (orbital.evaluate(plus) - orbital.evaluate(minus)) / 2e-5;
                    assert!(
                        (fd - derivative[axis]).abs() < 3e-6,
                        "l={} m={} axis={axis} actual={} expected={fd}",
                        orbital.l,
                        orbital.m,
                        derivative[axis]
                    );
                }
            }
        }
    }
}
#[test]
fn radial_bound_state_hydrogen_limit_and_wall_are_correct() {
    let ctx = Context::new(Some(1), 32 * 1024 * 1024).unwrap();
    let table = radial::confined(&ctx, 1, 0, 1., 20., 1000).unwrap();
    assert!((table.energy + 0.5).abs() < 6e-5);
    assert_eq!(table.value_slope(20.), (0., 0.));
    assert_eq!(table.value_slope(21.), (0., 0.));
    let (value, derivative) = table.value_slope(0.7);
    assert!((value - 2. * (-0.7f64).exp()).abs() < 2e-4);
    assert!((derivative + value).abs() < 2e-4);
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
    assert!(radial::confined(&ctx, usize::MAX, 0, 1., 20., 10).is_err());
}
#[test]
fn spherical_helium_has_normalized_density_and_finite_lda_fixed_point() {
    let ctx = Context::new(Some(1), 32 * 1024 * 1024).unwrap();
    let options = atom::AtomOptions {
        points: 500,
        r_max: 15.,
        max_iterations: 200,
        tolerance: 1e-9,
        mixing: 0.3,
    };
    let atom = atom::solve(&ctx, 2, &options, None, None).unwrap();
    assert!(atom.converged);
    assert!((atom.energy + 2.8314443984402025).abs() < 1e-7);
    let n = atom.r.as_slice().len();
    let h = 15. / 501.;
    let charge = (0..n)
        .map(|i| {
            4. * std::f64::consts::PI
                * atom.r.as_slice()[i].powi(2)
                * atom.density.as_slice()[i]
                * if i == 0 || i == n - 1 { 0.5 } else { 1. }
        })
        .sum::<f64>()
        * h;
    assert!((charge - 2.).abs() < 1e-8);
    drop(atom);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn all_electron_species_is_constructed_without_python_and_respects_budget_lifetime() {
    let ctx = Context::new(Some(1), 16 * 1024 * 1024).unwrap();
    let geometry = Geometry::new(
        vec![2],
        vec![[0.; 3]],
        [[0.; 3]; 3],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let spec = r#"{"name":"NAO-AE","tier":2,"onset":2.5,"width":1.5,"wall_radius":4.0,"scale":1.0,"points":400,"r_max":12.0,"linear_dependence_tol":0.0001,"tail_norm":0.0001}"#;
    let basis = Basis::build(&ctx, &geometry, spec).unwrap();
    assert!(basis.orbitals.len() > 5);
    for orbital in &basis.orbitals {
        assert!(orbital.evaluate([0.3, -0.2, 0.7]).is_finite());
        assert_eq!(orbital.evaluate([0., 0., 4.1]), 0.);
    }
    drop(basis);
    assert_eq!(ctx.memory_used(), 0);
}
