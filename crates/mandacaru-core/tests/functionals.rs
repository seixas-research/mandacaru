use mandacaru_core::{
    basis::{radial, Basis},
    math::SpatialGrid,
    models::{atom, xc},
    molecular::{MolecularIntegrals, ScfOptions},
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
/// A Gaussian orbital psi = exp(-r^2 / 2 w^2) sampled on the grid of H2 at
/// spacing `h` Bohr; returns the grid, the sampled psi and the analytic
/// gradient of psi^2 (`[axis][point]`).
fn gaussian_orbital(h: f64, w: f64) -> (SpatialGrid, Vec<f64>, Vec<f64>) {
    let grid = SpatialGrid::from_geometry(&geometry(0.), [h; 3], false, false).unwrap();
    let n = grid.index.points();
    let shape = grid.index.shape();
    let center = grid.point((shape[0] / 2 * shape[1] + shape[1] / 2) * shape[2] + shape[2] / 2);
    let (mut psi, mut density_gradient) = (vec![0.; n], vec![0.; 3 * n]);
    for k in 0..n {
        let point = grid.point(k);
        let delta: Vec<f64> = (0..3).map(|a| point[a] - center[a]).collect();
        let r2: f64 = delta.iter().map(|x| x * x).sum();
        psi[k] = (-r2 / (2. * w * w)).exp();
        for a in 0..3 {
            density_gradient[a * n + k] = -2. * psi[k] * delta[a] / (w * w) * psi[k];
        }
    }
    (grid, psi, density_gradient)
}
fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
#[test]
fn the_orbital_product_rule_gradient_of_a_gaussian_density_is_the_analytic_one() {
    // Resolved: the grid derivative of the sampled density agrees with the
    // analytic gradient as well, to the (high) order of the stencil.
    let (grid, psi, exact) = gaussian_orbital(0.3, 1.2);
    let n = grid.index.points();
    let density: Vec<f64> = psi.iter().map(|p| p * p).collect();
    let (mut product_rule, mut stencil) = (vec![0.; 3 * n], vec![0.; 3 * n]);
    for axis in 0..3 {
        let slice = axis * n..(axis + 1) * n;
        xc::derivative(&grid, &psi, axis, &mut product_rule[slice.clone()]).unwrap();
        for k in 0..n {
            product_rule[axis * n + k] *= 2. * psi[k];
        }
        xc::derivative(&grid, &density, axis, &mut stencil[slice]).unwrap();
    }
    let scale = exact.iter().fold(0f64, |m, v| m.max(v.abs()));
    let product_error = max_difference(&product_rule, &exact) / scale;
    let stencil_error = max_difference(&stencil, &exact) / scale;
    assert!(product_error < 5e-5, "product rule error {product_error}");
    assert!(stencil_error < 5e-4, "stencil error {stencil_error}");
}
#[test]
fn only_the_orbital_product_rule_keeps_an_unresolved_orbital_iso_orbital() {
    // psi as narrow as the grid spacing: |grad rho|^2 / 8 rho = |grad psi|^2 / 2
    // is an algebraic identity of the product rule, and fails for the grid
    // derivative of rho = psi^2 where rho is not resolved.
    let (grid, psi, _) = gaussian_orbital(0.3, 0.3);
    let n = grid.index.points();
    let density: Vec<f64> = psi.iter().map(|p| p * p).collect();
    let (mut orbital, mut stencil) = (vec![0.; 3 * n], vec![0.; 3 * n]);
    for axis in 0..3 {
        let slice = axis * n..(axis + 1) * n;
        xc::derivative(&grid, &psi, axis, &mut orbital[slice.clone()]).unwrap();
        xc::derivative(&grid, &density, axis, &mut stencil[slice]).unwrap();
    }
    let (mut product_worst, mut stencil_worst) = (0f64, 0f64);
    for k in (0..n).filter(|&k| density[k] > 1e-12) {
        let tau: f64 = (0..3).map(|a| 0.5 * orbital[a * n + k].powi(2)).sum();
        let product_rule: f64 = (0..3)
            .map(|a| (2. * psi[k] * orbital[a * n + k]).powi(2) / (8. * density[k]))
            .sum();
        let sampled: f64 = (0..3)
            .map(|a| stencil[a * n + k].powi(2) / (8. * density[k]))
            .sum();
        product_worst = product_worst.max((product_rule - tau).abs() / tau);
        stencil_worst = stencil_worst.max((sampled - tau).abs() / tau);
    }
    assert!(product_worst < 1e-12, "{product_worst}");
    assert!(stencil_worst > 1e-2, "{stencil_worst}");
}
