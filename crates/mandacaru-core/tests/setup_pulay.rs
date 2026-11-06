//! The derivative samples of the periodic setup (`periodic::setup::pulay`): the Cartesian
//! derivative of the Bloch sums on the grid, and the row-center blocks `G` of the overlap, the
//! kinetic operator and the local potential built on them. Every gate is a central difference
//! of the Bloch sums with the CENTER of the atom's functions moved (grid fixed), which pins the
//! sign, the ownership identity `dM_X = P_X G + G^+ P_X` and the images together. Needs
//! `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;

use mandacaru_core::{
    math::reciprocal,
    periodic::setup::pulay::{
        grid_cross, kinetic_cross, kinetic_form, kinetic_grid, kinetic_grid_f32, kinetic_self,
        samples_bytes, KSamples, Matrix, Mesh,
    },
    Context,
};
use nalgebra::DMatrix;
use num_complex::Complex64;
use setup_gradients_support::context;
use setup_pulay_support::*;

/// Gate of every central difference: the Richardson-extrapolated difference at `H` has an error
/// of order `H^4` of a fifth derivative and a roundoff of `1e-16 / H`, both far below it.
const GATE: f64 = 1e-8;
const H: f64 = 2e-4;
const CASES: [&str; 2] = ["si", "al2"];

#[test]
fn single_transform_kinetic_matches_general_form_and_uses_less_workspace() {
    for name in CASES {
        let setup_ctx = context();
        let fx = Fixture::new(&setup_ctx, name);
        let setup_mesh = fx.mesh(&setup_ctx);
        for k in [[0., 0., 0.], KPOINT] {
            let psi = bloch_values(&setup_ctx, &fx, &fx.shells, &setup_mesh, k);
            let m = fx.functions();
            let general_ctx = context();
            let general_mesh = fx.mesh(&general_ctx);
            let expected = kinetic_form(&general_ctx, &fx.grid, &general_mesh, k,
                (&psi, m), (&psi, m)).unwrap();
            let single_ctx = context();
            let single_mesh = fx.mesh(&single_ctx);
            let actual = kinetic_self(&single_ctx, &fx.grid, &single_mesh, k, &psi, m).unwrap();
            println!("{name} k={k:?}: relative error {:.3e}, context peak {} -> {} bytes",
                gap(&actual, &expected), general_ctx.memory_peak(), single_ctx.memory_peak());
            assert!(gap(&actual, &expected) < 1e-12, "{name}: kinetic mismatch");
            assert!(single_ctx.memory_peak() < general_ctx.memory_peak(),
                "{name}: single-transform peak {} vs general {}",
                single_ctx.memory_peak(), general_ctx.memory_peak());
            let direct_ctx = context();
            let direct = kinetic_grid(&direct_ctx, &fx.grid, k, &psi, m).unwrap();
            assert!(gap(&direct, &expected) < 1e-12, "{name}: grid kinetic mismatch");
            let narrow: Vec<_> = psi.iter().map(|z| num_complex::Complex32::new(z.re as f32, z.im as f32)).collect();
            let promoted: Vec<_> = narrow.iter().map(|z| Complex64::new(z.re as f64, z.im as f64)).collect();
            let reference = kinetic_form(&general_ctx, &fx.grid, &general_mesh, k,
                (&promoted, m), (&promoted, m)).unwrap();
            let narrow_result = kinetic_grid_f32(&direct_ctx, &fx.grid, k, &narrow, m).unwrap();
            assert!(gap(&narrow_result, &reference) < 1e-12, "{name}: float32 kinetic mismatch");
        }
    }
}

#[test]
fn the_samples_are_the_negative_center_derivative_of_the_bloch_sums() {
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let mesh = fx.mesh(&ctx);
        let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
        let n = mesh.len();
        let mut worst = 0f64;
        for atom in 0..fx.atoms() {
            for axis in 0..3 {
                let at = |delta: f64| -> Matrix {
                    let psi =
                        bloch_values(&ctx, &fx, &fx.moved_basis(atom, axis, delta), &mesh, KPOINT);
                    Matrix::from_row_slice(fx.functions(), n, &psi)
                };
                let fd = central(H, &mut |d| at(d));
                // The value samples are the rows of the unmoved basis; the derivative is minus
                // the kernel's Cartesian one on the moved atom's rows and zero elsewhere.
                let want = Matrix::from_fn(fx.functions(), n, |mu, x| {
                    if fx.owner[mu] == atom {
                        -samples.slope(axis)[mu * n + x]
                    } else {
                        Complex64::new(0., 0.)
                    }
                });
                let error = gap(&fd, &want);
                worst = worst.max(error);
                assert!(error < GATE, "{name} atom {atom} axis {axis}: {error:.2e}");
                // Controls: the value in place of the derivative, the opposite sign.
                let values = Matrix::from_row_slice(fx.functions(), n, samples.values());
                assert!(gap(&fd, &values) > 100. * GATE);
                assert!(gap(&fd, &(-&want)) > 100. * GATE);
            }
        }
        println!("{name}: samples against the central difference of the Bloch sums {worst:.2e}");
    }
}

/// The three matrices at the frozen basis, from the samples alone (no `G`).
fn references(
    ctx: &Context,
    fx: &Fixture,
    shells: &mandacaru_core::periodic::setup::shells::Shells,
    mesh: &mandacaru_core::periodic::setup::pulay::Mesh,
    v: &[f64],
) -> [Matrix; 3] {
    let n = mesh.len();
    let psi = bloch_values(ctx, fx, shells, mesh, KPOINT);
    [
        form(&psi, &psi, n, mesh.dv, None),
        form(&psi, &psi, n, mesh.dv, Some(v)),
        kinetic_form(
            ctx,
            &fx.grid,
            mesh,
            KPOINT,
            (&psi, shells.functions()),
            (&psi, shells.functions()),
        )
        .unwrap(),
    ]
}

#[test]
fn the_grid_blocks_obey_the_ownership_identity_against_the_moved_matrices() {
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let mesh = fx.mesh(&ctx);
        let v = rough_potential(mesh.len());
        let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
        let overlap = grid_cross(&ctx, &mesh, &samples, None).unwrap();
        let local = grid_cross(&ctx, &mesh, &samples, Some(&v)).unwrap();
        let kinetic = kinetic_cross(&ctx, &fx.grid, &mesh, &samples).unwrap();
        let mut all = [0f64; 3];
        for atom in 0..fx.atoms() {
            let own = fx.rows_of(atom);
            for axis in 0..3 {
                let mut worst = [0f64; 3];
                for (term, blocks) in [&overlap, &local, &kinetic].into_iter().enumerate() {
                    let fd = central(H, &mut |d| {
                        let shells = fx.moved_basis(atom, axis, d);
                        references(&ctx, &fx, &shells, &mesh, &v)[term].clone()
                    });
                    let mine = ownership(&blocks[axis], &own);
                    worst[term] = gap(&fd, &mine);
                    all[term] = all[term].max(worst[term]);
                    // Controls: the sign, and another axis' block.
                    assert!(gap(&fd, &(-&mine)) > 100. * GATE, "{name} term {term}");
                    let other = ownership(&blocks[(axis + 1) % 3], &own);
                    assert!(gap(&fd, &other) > 100. * GATE, "{name} term {term} axis");
                }
                assert!(
                    worst.iter().all(|w| *w < GATE),
                    "{name} atom {atom} axis {axis}: overlap/local/kinetic {worst:?}"
                );
            }
        }
        println!("{name}: overlap / local / kinetic against the moved matrices {all:?}");
        // The terms are different matrices: the overlap block does not stand in for the local
        // one, nor the local for the kinetic.
        let fd_local = ownership(&local[0], &fx.rows_of(0));
        assert!(gap(&fd_local, &ownership(&overlap[0], &fx.rows_of(0))) > 100. * GATE);
        assert!(gap(&ownership(&kinetic[0], &fx.rows_of(0)), &fd_local) > 100. * GATE);
    }
}

#[test]
fn the_kinetic_form_is_the_spectral_sum_with_the_alias_averaged_symbol() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    let mesh = fx.mesh(&ctx);
    let (m, n) = (fx.functions(), mesh.len());
    let psi = bloch_values(&ctx, &fx, &fx.shells, &mesh, KPOINT);
    let got = kinetic_form(&ctx, &fx.grid, &mesh, KPOINT, (&psi, m), (&psi, m)).unwrap();
    // The same sum written out: a direct transform of the periodic part (no FFT), the symbol of
    // `reciprocal::bloch` and the volume factors of the definition.
    let shape = fx.grid.index.shape();
    let symbols = reciprocal::bloch(&ctx, &fx.grid, KPOINT).unwrap();
    // Anchor: the symbol at G = 0 is |k|^2 (no alias averaging reaches it).
    let k2: f64 = KPOINT.iter().map(|x| x * x).sum();
    assert!((symbols.squares.as_slice()[0] - k2).abs() < 1e-12);
    let [x, y, z] = mesh.points();
    let u: Vec<Vec<Complex64>> = (0..m)
        .map(|mu| {
            (0..n)
                .map(|j| {
                    let angle = -(KPOINT[0] * x[j] + KPOINT[1] * y[j] + KPOINT[2] * z[j]);
                    psi[mu * n + j] * Complex64::new(angle.cos(), angle.sin())
                })
                .collect()
        })
        .collect();
    let transform = |row: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0., 0.); n];
        for q in 0..n {
            let (qi, qj, qk) = (
                q / (shape[1] * shape[2]),
                q / shape[2] % shape[1],
                q % shape[2],
            );
            let mut sum = Complex64::new(0., 0.);
            for (j, value) in row.iter().enumerate() {
                let (i0, i1, i2) = (
                    j / (shape[1] * shape[2]),
                    j / shape[2] % shape[1],
                    j % shape[2],
                );
                let turns = (qi * i0) as f64 / shape[0] as f64
                    + (qj * i1) as f64 / shape[1] as f64
                    + (qk * i2) as f64 / shape[2] as f64;
                let angle = -2. * std::f64::consts::PI * turns;
                sum += value * Complex64::new(angle.cos(), angle.sin());
            }
            out[q] = sum * mesh.dv;
        }
        out
    };
    let hats: Vec<Vec<Complex64>> = u.iter().map(|row| transform(row)).collect();
    let want = DMatrix::from_fn(m, m, |a, b| {
        (0..n)
            .map(|q| {
                hats[a][q].conj() * hats[b][q] * (0.5 * symbols.squares.as_slice()[q] / mesh.volume)
            })
            .sum::<Complex64>()
    });
    println!(
        "kinetic form against the written-out spectral sum {:.2e}",
        gap(&got, &want)
    );
    assert!(gap(&got, &want) < 1e-12, "{:.2e}", gap(&got, &want));
    // Hermitian and positive semidefinite (a kinetic operator).
    assert!(peak(&(&got - got.adjoint())) < 1e-12 * peak(&got));
    let eigenvalues =
        nalgebra::SymmetricEigen::new((&got + got.adjoint()) / Complex64::new(2., 0.));
    assert!(eigenvalues.eigenvalues.iter().all(|e| *e > -1e-10));
    // Control: the symbol of another k-point is more than 100 x the gate away.
    let wrong = reciprocal::bloch(&ctx, &fx.grid, [0.; 3]).unwrap();
    let other = DMatrix::from_fn(m, m, |a, b| {
        (0..n)
            .map(|q| {
                hats[a][q].conj() * hats[b][q] * (0.5 * wrong.squares.as_slice()[q] / mesh.volume)
            })
            .sum::<Complex64>()
    });
    assert!(gap(&got, &other) > 1e-4);
}

#[test]
fn one_k_point_of_samples_is_reserved_from_the_context_and_released() {
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let mesh = fx.mesh(&ctx);
        let before = ctx.memory_used();
        let need = samples_bytes(fx.functions(), mesh.len()).unwrap();
        {
            let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
            let held = ctx.memory_used() - before;
            assert!(
                held >= need && held < 2 * need,
                "{name}: held {held} of {need}"
            );
            // Three derivatives beside the values: four arrays of the size of psi.
            assert_eq!(samples.values().len() * 16 * 4, need);
        }
        assert_eq!(
            ctx.memory_used(),
            before,
            "{name}: the samples are released"
        );
        println!(
            "{name}: M = {}, N = {}, samples {} KiB, context peak {} KiB",
            fx.functions(),
            mesh.len(),
            need / 1024,
            ctx.memory_peak() / 1024
        );
        // A context that cannot hold the four arrays refuses instead of allocating.
        let small = Context::new(Some(1), need / 2).unwrap();
        let tight = Fixture::new(&context(), name);
        let little = Mesh::new(&small, &tight.grid);
        assert!(
            little.is_err()
                || KSamples::new(
                    &small,
                    &tight.shells,
                    &tight.lattice,
                    &little.unwrap(),
                    KPOINT
                )
                .is_err()
        );
    }
}

/// Refined grids (the recorded ones are coarse): all four mesh blocks of one k-point at a time,
/// under a one-gigabyte context, for the Si and Al cells. The peak is the context's own account
/// of what was reserved (the process RSS is measured by running this binary under `time -l`).
#[test]
fn the_four_mesh_blocks_of_a_refined_cell_fit_one_gigabyte_per_k_point() {
    for (name, shape) in [
        ("si", [36, 36, 36]),
        ("al", [44, 44, 44]),
        ("al2", [36, 36, 50]),
    ] {
        let ctx = context();
        let fx = Fixture::with_shape(&ctx, name, Some(shape));
        let mesh = fx.mesh(&ctx);
        let v = rough_potential(mesh.len());
        let need = samples_bytes(fx.functions(), mesh.len()).unwrap();
        let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
        let overlap = grid_cross(&ctx, &mesh, &samples, None).unwrap();
        let local = grid_cross(&ctx, &mesh, &samples, Some(&v)).unwrap();
        let kinetic = kinetic_cross(&ctx, &fx.grid, &mesh, &samples).unwrap();
        assert!(peak(&overlap[0]) > 1e-3 && peak(&local[0]) > 1e-3 && peak(&kinetic[0]) > 1e-3);
        println!(
            "{name}: M = {}, N = {}, samples {:.1} MiB, context peak {:.1} MiB",
            fx.functions(),
            mesh.len(),
            need as f64 / 1048576.,
            ctx.memory_peak() as f64 / 1048576.
        );
        assert!(
            ctx.memory_peak() < 2 * need,
            "{name}: the reserved peak is the samples and the mesh"
        );
    }
}
