//! Smoothness of the molecular PAW-LCAO energy under a step of one atom: linear HCN, the
//! production DZP basis (`paw::basis_recipe`) on `lda-sr`, grid step 0.25 A, the hydrogen moved
//! alone along the axis through the position where a derivative jump was found (H at 2.9978 A,
//! between +4.0 and +4.2 mA from the scan's base frame). Every one-body matrix contracted with
//! the base density is printed per step, so the term that jumps is named. Needs
//! `MANDACARU_PAW_PATH`.
mod paw_molecular_support;

use mandacaru_core::{
    math::{storage::Precision, SpatialGrid},
    molecular::{MolecularIntegrals, ScfOptions},
    paw::{
        basis_recipe::{self, Recipe},
        molecular::{build, Arrays},
    },
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use paw_molecular_support::{dataset, table};
use serde_json::json;

const BOHR: f64 = 0.529177210903;
/// The scan's base frame (Angstrom): H, C, N on the axis of a 6 x 6 x 8.3 A cell.
const BASE: [[f64; 3]; 3] = [
    [3.0, 3.0, 2.9936708910884553],
    [3.0, 3.0, 4.106433274173262],
    [3.0, 3.0, 5.299895834738284],
];

fn hcn(ctx: &Context, dz: f64) -> MolecularIntegrals {
    let symbols = ["H", "C", "N"];
    let numbers = [1u8, 6, 7];
    let mut arrays = Arrays::new();
    let mut species = Vec::new();
    let mut basis = Vec::new();
    for (atom, symbol) in symbols.iter().enumerate() {
        let data = dataset::read(ctx, "lda-sr", symbol);
        species.push(json!({"record": dataset::species(ctx, &mut arrays, &data)}));
        // The recipe the calculator builds for `basis={"name": "PAW-LCAO", "size": "DZP"}`.
        let recipe = Recipe::from_json(
            &json!({"zetas": 2, "polarizations": 1, "cutoff": null, "energy_shift": 0.1,
                    "wall": [12.0, 0.6], "split_norm": 0.15, "tail_norms": [0.15, 0.31, 0.54],
                    "polarization": "gaussian", "diffuse": null}),
            None,
        )
        .unwrap();
        let (shells, _) = basis_recipe::species(ctx, &data, &recipe).unwrap();
        for shell in shells {
            let r = table(ctx, &mut arrays, &shell.r);
            let values = table(ctx, &mut arrays, &shell.values);
            for m in -(shell.l as i32)..=shell.l as i32 {
                basis.push(
                    json!({"atom": atom, "l": shell.l, "m": m, "r": r, "values": values,
                                  "diffuse": shell.diffuse}),
                );
            }
        }
    }
    let positions: Vec<[f64; 3]> = BASE
        .iter()
        .enumerate()
        .map(|(a, p)| {
            let z = if a == 0 { p[2] + dz } else { p[2] };
            [p[0] / BOHR, p[1] / BOHR, z / BOHR]
        })
        .collect();
    let cell = [
        [6. / BOHR, 0., 0.],
        [0., 6. / BOHR, 0.],
        [0., 0., 8.3 / BOHR],
    ];
    let geometry = Geometry::new(
        numbers.to_vec(),
        positions,
        cell,
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [0.25 / BOHR; 3], false, false).unwrap();
    let species_list: Vec<_> = species.iter().map(|s| s["record"].clone()).collect();
    let metadata = json!({
        "format": "molecular-paw-input", "version": 1, "length_unit": "bohr", "energy_unit": "Ha",
        "projector_basis": "raw", "atoms": [0, 1, 2], "species": species_list, "basis": basis,
    });
    build(
        ctx,
        geometry,
        grid,
        &metadata.to_string(),
        arrays,
        "spectral",
        3.6749322175654994e-10,
        Precision::Float64,
    )
    .unwrap()
}

fn contract(d: &DMatrix<f64>, m: &[f64]) -> f64 {
    let n = d.nrows();
    (0..n * n).map(|k| d[(k / n, k % n)] * m[k]).sum()
}

#[test]
#[ignore = "slow: 100 s (22 SCFs, measured 2026-11-14); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test molecular_smoothness -- --ignored --nocapture"]
fn the_hcn_energy_and_its_terms_are_smooth_across_the_reported_hydrogen_kink() {
    let ctx = Context::new(Some(4), 2 << 30).unwrap();
    let base = hcn(&ctx, 0.004);
    let state = base
        .mean_field(
            "rhf",
            [5, 5],
            &ScfOptions {
                tolerance: 1e-11,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(state.converged);
    // A minimum of the closed-shell surface, not a saddle (its orbital Hessian is positive).
    assert!(base.stability(&state).unwrap().unwrap() > 0.);
    let d = state.densities[0].map(|v| v.re) * 2.;
    let steps: Vec<f64> = (0..=20).map(|k| 0.0038 + 2e-5 * k as f64).collect();
    let mut rows = Vec::new();
    for &dz in &steps {
        let moved = hcn(&ctx, dz);
        let scf = moved
            .mean_field(
                "rhf",
                [5, 5],
                &ScfOptions {
                    tolerance: 1e-11,
                    ..Default::default()
                },
            )
            .unwrap();
        let ao = |x: &DMatrix<f64>| x.clone();
        let s = contract(&ao(&d), moved.overlap.as_slice());
        let t = contract(&d, moved.kinetic.as_slice());
        let v = contract(&d, moved.potential.as_slice());
        rows.push((dz, scf.reference_energy, s, t, v, moved.nuclear));
        let eps = &scf.energies[0];
        println!(
            "dz {dz:.5} A: E {:.12} Ha, D.S {s:.12}, D.T {t:.12}, D.V {v:.12}, E_nn {:.12}, HOMO {:.8} LUMO {:.8}, lowest Hessian root {:?}, iterations {}",
            scf.reference_energy, moved.nuclear, eps[4], eps[5], moved.stability(&scf).unwrap(), scf.iterations.len()
        );
    }
    // The slopes between neighbors, and the jump of the slope (second difference) at each step:
    // a smooth surface changes its slope by the curvature times the step at every step; a kink
    // (the 73 meV/A jump of the force once reported here, 2.7e-3 Ha/A) stands out of that.
    let names = ["E", "D.S", "D.T", "D.V", "E_nn"];
    for (k, name) in names.iter().enumerate() {
        let value = |r: &(f64, f64, f64, f64, f64, f64)| [r.1, r.2, r.3, r.4, r.5][k];
        let slopes: Vec<f64> = rows
            .windows(2)
            .map(|w| (value(&w[1]) - value(&w[0])) / (w[1].0 - w[0].0))
            .collect();
        let jumps: Vec<f64> = slopes.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = jumps.iter().sum::<f64>() / jumps.len() as f64;
        let worst = jumps.iter().fold(0f64, |a, j| a.max((j - mean).abs()));
        println!("{name}: slope change per step {mean:.3e} Ha/A, largest departure {worst:.3e}");
        assert!(
            worst < 1e-4,
            "{name}: the slope jumps by {worst:.3e} Ha/A between two steps"
        );
    }
}
