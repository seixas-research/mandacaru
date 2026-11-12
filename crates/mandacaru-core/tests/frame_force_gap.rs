//! Does the analytic force of a truncated unrestricted ADAPT state follow the production
//! energy surface, which now has a deterministic orbital frame? A frozen-ansatz finite
//! difference of the production energy over an axial (breathing) step of linear H3 against the
//! analytic force, with the active space (a) only a symmetry-protected pair of pi orbitals and
//! (b) two empty sigma orbitals cut out of the exact zero-occupation cluster, which no symmetry
//! protects.
mod paw_forces_support;
mod variational_forces_support;

use mandacaru_core::{
    math::storage::SampleBlocks,
    molecular::{response, MeanField, MolecularIntegrals},
    quantum::{lagrangian::Ansatz, SolveOptions},
    Context, Geometry, LengthUnit,
};
use paw_forces_support::synthetic::{integrals, solve, Method};

const H3: [[f64; 3]; 3] = [[0., 0., -1.3], [0., 0., 0.], [0., 0., 1.3]];
const MEV_PER_ANGSTROM: f64 = 27211.386245988 / 0.529177210903;

/// Linear functionals of every model orbital (columns over the orthonormal frame) under a
/// Gaussian envelope about the origin: `[1, x, y, xz, yz]`. An orbital odd in `z` and in `x`
/// only has an `xz` moment, so these tell the sigma and the pi orbitals and their `z` parity.
fn functionals(base: &MolecularIntegrals, state: &MeanField) -> Vec<[f64; 5]> {
    let ctx = &base.context;
    let m = base.m();
    let points = base.grid.index.points();
    let c = state.model_orbitals.map(|v| v.re);
    let mut blocks = SampleBlocks::new(ctx, &base.orthonormal_samples, m, points).unwrap();
    let mut out = vec![[0f64; 5]; m];
    for start in (0..points).step_by(blocks.stride()) {
        let count = blocks.load(start).unwrap();
        for g in 0..count {
            let [x, y, z] = base.grid.point(start + g);
            let e = (-(x * x + y * y + z * z) / 18.).exp();
            let weights = [e, x * e, y * e, x * z * e, y * z * e];
            for (col, row) in out.iter_mut().enumerate() {
                let psi: f64 = (0..m).map(|i| c[(i, col)] * blocks.row(i)[g]).sum();
                for k in 0..5 {
                    row[k] += psi * weights[k];
                }
            }
        }
    }
    out
}

/// The geometry of `base` with the end atoms moved apart by `2 delta` along the axis.
fn breathing(base: &MolecularIntegrals, delta: f64) -> MolecularIntegrals {
    let mut positions = base.geometry.positions().to_vec();
    positions[0][2] -= delta;
    positions[2][2] += delta;
    let geometry = Geometry::new(
        base.geometry.numbers().to_vec(),
        positions,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap();
    base.at_geometry(geometry).unwrap()
}

/// `dE/d(delta)` of the fixed ansatz on the active space `spec`, by the analytic force and by a
/// Richardson difference of the production energy, in meV per Angstrom.
fn compare(base: &MolecularIntegrals, state: &MeanField, spec: &str) -> (f64, f64, f64) {
    let ctx = &base.context;
    let model = state.active_model(base, spec).unwrap();
    let problem = model
        .problem(ctx, "jordan_wigner", true, false, true)
        .unwrap();
    let solved = problem
        .solve(
            &SolveOptions {
                adapt: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let prepared = problem
        .prepare_state(&solved.indices, &solved.parameters, true)
        .unwrap();
    let analytic = response::gradient(
        base,
        state,
        &model,
        &problem,
        prepared.complex128().unwrap(),
        &response::Options {
            ansatz: Some(Ansatz {
                indices: &solved.indices,
                parameters: &solved.parameters,
                product_form: true,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let reference = variational_forces_support::oracle::gradient(
        base,
        state,
        &model,
        &problem,
        prepared.complex128().unwrap(),
        &variational_forces_support::oracle::Options {
            active_spec: Some(spec),
            ansatz: Some(Ansatz {
                indices: &solved.indices,
                parameters: &solved.parameters,
                product_form: true,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let r = reference.gradient.as_slice();
    let oracle = -r[2] + r[8];
    println!("{spec}: polar-aligned oracle {oracle:.8e} Ha/Bohr");
    let g = analytic.gradient.as_slice();
    // Atom 0 moves by -delta, atom 2 by +delta.
    let analytic = -g[2] + g[8];
    let energy = |delta: f64| {
        let moved = breathing(base, delta);
        let after = solve(&moved, Method::Hf("uhf"), [2, 1]);
        let model = after.active_model(&moved, spec).unwrap();
        let problem = model
            .problem(&moved.context, "jordan_wigner", true, false, true)
            .unwrap();
        problem
            .evaluate(&solved.indices, &solved.parameters, true)
            .unwrap()
            .energy
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    let numeric = (4. * fine - coarse) / 3.;
    println!(
        "{spec}: {} operators, analytic {:.8e} Ha/Bohr, finite difference {:.8e} (coarse {:.8e})",
        solved.indices.len(),
        analytic,
        numeric,
        coarse
    );
    (
        analytic * MEV_PER_ANGSTROM,
        numeric * MEV_PER_ANGSTROM,
        oracle * MEV_PER_ANGSTROM,
    )
}

#[test]
#[ignore = "slow: 50 s (measured 2026-11-11); run: cargo test --profile release-test -p mandacaru-core --test frame_force_gap -- --ignored"]
fn the_unrestricted_force_gap_of_a_protected_pair_and_of_the_empty_block() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H3, "lda");
        let state = solve(&base, Method::Hf("uhf"), [2, 1]);
        let m = state.model_orbitals.ncols();
        let moments = functionals(&base, &state);
        // The exact zero-occupation cluster (the doubly and singly occupied orbitals and the
        // one with a small spin-polarized occupation are outside it).
        let empty: Vec<usize> = (2..m)
            .filter(|&c| state.natural_occupations[c] < 1e-8)
            .collect();
        assert!(empty.len() > 4, "{:?}", state.natural_occupations);
        let best = |k: usize| {
            *empty
                .iter()
                .max_by(|&&a, &&b| moments[a][k].abs().total_cmp(&moments[b][k].abs()))
                .unwrap()
        };
        // The pair odd in z: its x and y members; the sigma-type orbitals even in z.
        let (odd_x, odd_y) = (best(3), best(4));
        let mut sigma: Vec<usize> = empty.clone();
        sigma.sort_by(|&a, &b| moments[b][0].abs().total_cmp(&moments[a][0].abs()));
        let sigma = [sigma[0], sigma[1]];
        println!("odd pi pair {odd_x}, {odd_y}; sigma {sigma:?}; moments {moments:?}");
        let pair = format!(r#"{{"orbitals": [0, 1, {odd_x}, {odd_y}]}}"#);
        let cut = format!(r#"{{"orbitals": [0, 1, {}, {}]}}"#, sigma[0], sigma[1]);
        let (a_analytic, a_numeric, a_oracle) = compare(&base, &state, &pair);
        let (b_analytic, b_numeric, b_oracle) = compare(&base, &state, &cut);
        println!(
            "(a) protected pair: analytic {a_analytic:.3} meV/A, finite difference {a_numeric:.3}, gap {:.3}",
            (a_analytic - a_numeric).abs()
        );
        println!(
            "(b) empty-block cut: analytic {b_analytic:.3} meV/A, finite difference {b_numeric:.3}, gap {:.3}",
            (b_analytic - b_numeric).abs()
        );
        println!(
            "(a) pair: analytic {a_analytic:.3}, polar oracle {a_oracle:.3}, finite difference {a_numeric:.3} meV/A"
        );
        println!(
            "(b) cut: analytic {b_analytic:.3}, polar oracle {b_oracle:.3}, finite difference {b_numeric:.3} meV/A"
        );
        // (a) A symmetry-protected pair: the analytic force is the derivative of the production
        // energy.
        assert!((a_analytic - a_numeric).abs() < 5., "protected pair");
        assert!((a_analytic - a_oracle).abs() < 5., "protected pair, oracle");
        // (b) Two orbitals cut out of the exact zero-occupation cluster. Measured 2026-11-11
        // (linear H3, 11 operators): the production energy surface equals the polar-aligned
        // (parallel-transported) oracle to 1e-3 meV/A, so the deterministic frame costs the
        // force nothing here, but the analytic force differs from both by 20.1 meV/A (0.13 % of
        // 15.35 eV/A), a gap the force already had against its own oracle for a cut through
        // this cluster. It is pinned at twice the measurement: a change that grows it fails.
        assert!(
            (b_numeric - b_oracle).abs() < 0.1,
            "cut, production against oracle"
        );
        assert!(
            (b_analytic - b_numeric).abs() < 40.,
            "cut: gap {:.3} meV/A, measured 20.126",
            (b_analytic - b_numeric).abs()
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
