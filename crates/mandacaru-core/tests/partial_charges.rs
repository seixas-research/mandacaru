use mandacaru_core::{
    math::partial_charges::{compute_partial_charges, ProatomicRadial},
    math::SpatialGrid,
    molecular::{
        observables::{partial_charges, ActiveDensity},
        symmetry::molecular_symmetry_fields,
        MolecularIntegrals, ScfOptions,
    },
    periodic::space_group::report::crystal_symmetry_fields,
    quantum::SolveOptions,
    Context, Geometry, LengthUnit,
};
use std::collections::BTreeMap;

fn gaussian_grid_density(
    grid: &SpatialGrid,
    centers: &[[f64; 3]],
    weights: &[f64],
    alpha: f64,
) -> Vec<f64> {
    let norm = (alpha / std::f64::consts::PI).powf(1.5);
    let n = grid.index.points();
    let shape = grid.index.shape();
    let mut rho = vec![0.0; n];
    for k in 0..n {
        let p = grid.point(k);
        let mut sum = 0.0;
        for (c, &w) in centers.iter().zip(weights) {
            let mut min_r2 = f64::INFINITY;
            if grid.periodic {
                for ix in -1isize..=1 {
                    for iy in -1isize..=1 {
                        for iz in -1isize..=1 {
                            let shift = [
                                grid.step[(0, 0)] * (shape[0] as f64 * ix as f64)
                                    + grid.step[(0, 1)] * (shape[1] as f64 * iy as f64)
                                    + grid.step[(0, 2)] * (shape[2] as f64 * iz as f64),
                                grid.step[(1, 0)] * (shape[0] as f64 * ix as f64)
                                    + grid.step[(1, 1)] * (shape[1] as f64 * iy as f64)
                                    + grid.step[(1, 2)] * (shape[2] as f64 * iz as f64),
                                grid.step[(2, 0)] * (shape[0] as f64 * ix as f64)
                                    + grid.step[(2, 1)] * (shape[1] as f64 * iy as f64)
                                    + grid.step[(2, 2)] * (shape[2] as f64 * iz as f64),
                            ];
                            let dx = p[0] - (c[0] + shift[0]);
                            let dy = p[1] - (c[1] + shift[1]);
                            let dz = p[2] - (c[2] + shift[2]);
                            min_r2 = min_r2.min(dx * dx + dy * dy + dz * dz);
                        }
                    }
                }
            } else {
                let dx = p[0] - c[0];
                let dy = p[1] - c[1];
                let dz = p[2] - c[2];
                min_r2 = dx * dx + dy * dy + dz * dz;
            }
            sum += w * norm * (-alpha * min_r2).exp();
        }
        rho[k] = sum;
    }
    rho
}

fn gaussian_radial(weight: f64, alpha: f64) -> ProatomicRadial {
    let norm = (alpha / std::f64::consts::PI).powf(1.5);
    let r: Vec<f64> = (0..=400).map(|i| 0.03 * i as f64).collect();
    let density: Vec<f64> = r
        .iter()
        .map(|&rv| weight * norm * (-alpha * rv * rv).exp())
        .collect();
    ProatomicRadial {
        r,
        density,
        valence_charge: weight,
        fallback_scale: 0.75,
    }
}

#[test]
fn bader_hirshfeld_and_voronoi_conserve_charge_and_reproduce_symmetry_and_polarity() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let grid = SpatialGrid::new(
        [28, 28, 36],
        [-3.375, -3.375, -4.375],
        [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25],
        false,
    )
    .unwrap();
    let centers = [[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]];
    let valences = [1.0, 1.0];
    let aug = [0.0, 0.0];

    // 1. Symmetric homonuclear dimer: equal populations and zero partial charges (both interpolated and uninterpolated).
    let rho_sym = gaussian_grid_density(&grid, &centers, &[1.0, 1.0], 1.2);
    let rad_sym = vec![gaussian_radial(1.0, 1.2), gaussian_radial(1.0, 1.2)];
    for interpolate in [true, false] {
        for method in ["voronoi", "hirshfeld", "bader"] {
            let out = compute_partial_charges(
                &ctx,
                method,
                &grid,
                &rho_sym,
                &centers,
                &valences,
                &aug,
                Some(&rad_sym),
                interpolate,
            )
            .unwrap();
            assert_eq!(out.interpolated, interpolate);
            let pop_sum: f64 = out.populations.iter().sum();
            let chg_sum: f64 = out.charges.iter().sum();
            assert!(
                (pop_sum - out.total_electrons).abs() < 1e-10,
                "{method}/{interpolate}"
            );
            assert!(
                (chg_sum - out.net_charge).abs() < 1e-10,
                "{method}/{interpolate}"
            );
            assert!(
                (out.charges[0] - out.charges[1]).abs() < 1e-8,
                "{method}/{interpolate}: {:?}",
                out.charges
            );
        }
    }

    // 2. Polar heteronuclear dimer: atom 0 loses density (q > 0), atom 1 gains density (q < 0).
    let rho_polar = gaussian_grid_density(&grid, &centers, &[0.55, 1.45], 1.2);
    for method in ["voronoi", "hirshfeld", "bader"] {
        let out = compute_partial_charges(
            &ctx,
            method,
            &grid,
            &rho_polar,
            &centers,
            &valences,
            &aug,
            Some(&rad_sym),
            true,
        )
        .unwrap();
        assert!(out.interpolated);
        assert!(out.charges[0] > 0.25, "{method}: {:?}", out.charges);
        assert!(out.charges[1] < -0.25, "{method}: {:?}", out.charges);
        assert!((out.charges[0] + out.charges[1] - out.net_charge).abs() < 1e-10);
    }
}

#[test]
fn periodic_skewed_cell_bader_hirshfeld_voronoi_conserve_electrons() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    // Non-orthogonal triclinic cell step vectors
    let grid = SpatialGrid::new(
        [20, 20, 20],
        [0.0, 0.0, 0.0],
        [0.28, 0.08, 0.0, 0.0, 0.28, 0.06, 0.0, 0.0, 0.28],
        true,
    )
    .unwrap();
    let step = grid.step;
    let a1 = step.column(0) * 20.0;
    let a2 = step.column(1) * 20.0;
    let a3 = step.column(2) * 20.0;
    let c0 = (a1 * 0.25 + a2 * 0.25 + a3 * 0.25).into();
    let c1 = (a1 * 0.75 + a2 * 0.75 + a3 * 0.75).into();
    let centers = [c0, c1];
    let rho = gaussian_grid_density(&grid, &centers, &[0.7, 1.3], 1.1);
    let rad = vec![gaussian_radial(1.0, 1.1), gaussian_radial(1.0, 1.1)];
    for method in ["voronoi", "hirshfeld", "bader"] {
        let out = compute_partial_charges(
            &ctx,
            method,
            &grid,
            &rho,
            &centers,
            &[1.0, 1.0],
            &[0.05, 0.05],
            Some(&rad),
            true,
        )
        .unwrap();
        assert!((out.populations.iter().sum::<f64>() - out.total_electrons).abs() < 1e-10);
        assert!(
            out.charges[0] > 0.1 && out.charges[1] < -0.2,
            "{method}: {:?}",
            out.charges
        );
    }
}

#[test]
fn molecular_hf_and_vqe_partial_charges_match_symmetry() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let geom = Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geom, [0.4 / 0.529177210903; 3], false, false).unwrap();
    let ints = MolecularIntegrals::new(
        &ctx,
        geom,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap();
    let mf = ints
        .mean_field("rhf", [1, 1], &ScfOptions::default())
        .unwrap();
    let model = mf.model(&ints).unwrap();
    let problem = model
        .problem(&ctx, "jordan_wigner", false, false, false)
        .unwrap();
    let ans = problem.solve(&SolveOptions::default(), None).unwrap();
    let rdms = problem
        .rdms(ans.state.complex128().unwrap(), false)
        .unwrap();
    let gamma = rdms.one.as_slice();

    for method in ["voronoi", "hirshfeld", "bader"] {
        let q_mf = partial_charges(&ints, &mf, None, method, None, true).unwrap();
        assert!(q_mf.interpolated);
        assert!(
            (q_mf.charges[0] - q_mf.charges[1]).abs() < 1e-4,
            "RHF {method}: {:?}",
            q_mf.charges
        );
        assert!((q_mf.charges.iter().sum::<f64>()).abs() < 1e-6);

        let q_vqe = partial_charges(
            &ints,
            &mf,
            Some(ActiveDensity {
                gamma,
                active: &model.orbital_indices,
                frozen: &model.frozen_indices,
                rotation: model.orbital_rotation.as_ref().map(|r| &r.matrix),
            }),
            method,
            None,
            true,
        )
        .unwrap();
        assert!(
            (q_vqe.charges[0] - q_vqe.charges[1]).abs() < 1e-4,
            "VQE {method}: {:?}",
            q_vqe.charges
        );
        assert!((q_vqe.charges.iter().sum::<f64>()).abs() < 1e-6);

        let q_raw = partial_charges(&ints, &mf, None, method, None, false).unwrap();
        assert!(!q_raw.interpolated);
    }
}

#[test]
fn crystal_and_molecular_symmetry_reports_cover_magnetic_and_double_groups() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let a = 5.43 / 0.529177210903;
    let lattice = [
        [0.0, 0.5 * a, 0.5 * a],
        [0.5 * a, 0.0, 0.5 * a],
        [0.5 * a, 0.5 * a, 0.0],
    ];
    let positions = [[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]];
    let afm_moments = [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]];
    let pairs = crystal_symmetry_fields(
        &ctx,
        lattice,
        &positions,
        &[14, 14],
        Some(&afm_moments),
        true,
        true,
        false,
        false,
        None,
    )
    .unwrap();
    let map: BTreeMap<_, _> = pairs.into_iter().collect();
    assert!(map["space_group"].contains("Fd-3m") && map["space_group"].contains("227"));
    assert!(map["magnetic_group_type"].contains("Type III"), "{map:?}");
    assert!(map["spin_orbit_double_group"].contains("SU(2)"), "{map:?}");
    assert!(map["double_group_discussion"].contains("bar_E"), "{map:?}");

    let h2o_pos = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
    let mol_pairs = molecular_symmetry_fields(
        &h2o_pos,
        &[8, 1, 1],
        Some(&[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        true,
        true,
    )
    .unwrap();
    let mol_map: BTreeMap<_, _> = mol_pairs.into_iter().collect();
    assert_eq!(mol_map["point_group"], "C2v");
    assert!(
        mol_map["magnetic_group_type"].contains("magnetic"),
        "{mol_map:?}"
    );
    assert!(mol_map["double_group_order"].contains("4"), "{mol_map:?}");
}
