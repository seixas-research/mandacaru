//! The strain tangent of the Bloch sums on the grid (`periodic::setup::strain_samples`,
//! AS-1): `A_ab(r_i) = sum_T e^{ik.T} (d_a phi)(y) y_b` with `y = r_i - c - T`, one pass of the
//! Bloch kernel with a weight axis. Every gate compares it with a central difference of the Bloch
//! sums on a STRAINED cell (lattice, atoms, basis centers and grid nodes carried by
//! `x -> (1 + eps) x`; radial tables fixed; the Cartesian k-point carried so that the fractional
//! one is held) and, separately, with the one-image-per-call pattern (an exact identity, so the
//! kernel is pinned independently of the finite-difference step). Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;

use mandacaru_core::{
    kernels::{bloch_shell, bloch_shell_weighted, ShellImages},
    periodic::setup::{
        images::Images,
        pulay::{KSamples, Matrix, Mesh},
        shells::{DenseRows, Sampling},
        strain_samples::{tangent_bytes, weighted_derivatives, StrainSamples},
    },
    Context,
};
use num_complex::Complex64;
use setup_gradients_support::context;
use setup_pulay_support::*;
use strain_samples_support::*;

/// The gate is the plan's number. The measured floor of the difference is 8e-10 (Si) and 1.1e-9
/// (Al2) at `H = 1e-5` and 1.7e-9 / 2.0e-9 at `5e-5`: it does not fall with the step, so it is not
/// the Richardson truncation. The cause is probably a node crossing the support of a table (the
/// kernel cuts at the last point above 1e-12 of the peak, a jump the finite difference sees and the
/// analytic tangent does not); that reading is not verified.
const GATE: f64 = 1e-8;
const H: f64 = 1e-5;
const CASES: [&str; 2] = ["si", "al2"];

fn peak_of(values: &[Complex64]) -> f64 {
    values.iter().map(|x| x.norm()).fold(0., f64::max)
}

fn all_pairs() -> impl Iterator<Item = (usize, usize)> {
    (0..3).flat_map(|a| (0..3).map(move |b| (a, b)))
}

#[test]
fn the_tangent_is_the_central_difference_of_the_strained_bloch_sums() {
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let mesh = fx.mesh(&ctx);
        let (m, n) = (fx.functions(), mesh.len());
        let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
        let strain = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT);
        let mut worst = 0f64;
        for (a, b) in all_pairs() {
            let tangent = strain.tangent(&ctx, a, b).unwrap();
            let got = Matrix::from_row_slice(m, n, tangent.as_slice());
            let fd = central(H, &mut |d| {
                Strained::new(&fx, &entry(a, b, d), KPOINT).bloch_values(&ctx)
            });
            let error = gap(&got, &fd);
            worst = worst.max(error);
            assert!(error < GATE, "{name} ({a},{b}): {error:.2e}");
            // Controls, each compared with the central difference: the unweighted derivative
            // (the weight dropped), the opposite sign, the summed derivative weighted by the
            // distance to the home center (the image translation dropped), and for an
            // off-diagonal pair the transposed component.
            let slope = Matrix::from_row_slice(m, n, samples.slope(a));
            assert!(
                gap(&slope, &fd) > 100. * GATE,
                "{name} ({a},{b}) unweighted"
            );
            assert!(gap(&(-&got), &fd) > 100. * GATE, "{name} ({a},{b}) sign");
            let home = Matrix::from_fn(m, n, |mu, x| {
                let c = fx.centers[fx.owner[mu]][b];
                slope[(mu, x)] * (mesh.points()[b][x] - c)
            });
            assert!(
                gap(&home, &fd) > 100. * GATE,
                "{name} ({a},{b}) home center"
            );
            if a != b {
                let other = strain.tangent(&ctx, b, a).unwrap();
                let other = Matrix::from_row_slice(m, n, other.as_slice());
                assert!(
                    gap(&other, &fd) > 100. * GATE,
                    "{name} ({a},{b}) transposed"
                );
            }
        }
        println!("{name}: tangent against the central difference, worst {worst:.2e}");
    }
}

/// The step enters the oracle only through the spline's third-derivative jumps: the spread
/// between the extrapolated differences at `H` and `H/2` is reported and bounded.
#[test]
fn the_oracle_step_is_settled() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    let mut worst = 0f64;
    for (a, b) in [(0, 0), (0, 2), (2, 1)] {
        let at = |h: f64| {
            central(h, &mut |d| {
                Strained::new(&fx, &entry(a, b, d), KPOINT).bloch_values(&ctx)
            })
        };
        let (coarse, fine) = (at(2. * H), at(H));
        let spread = gap(&coarse, &fine);
        worst = worst.max(spread);
        println!("si ({a},{b}): spread 2H vs H {spread:.2e}");
    }
    assert!(worst < GATE, "the oracle is not converged in the step");
}

/// `sum_T e^{ik.T} g_a(y) y_b`, one image per kernel call: the unweighted derivative of the shell
/// moved to `c + T` (the plain kernel, one image, the shell's own support), weighted outside the
/// kernel by `r_b - (c + T)_b`.
fn one_image_per_call(ctx: &Context, fx: &Fixture, mesh: &Mesh, a: usize, b: usize) -> Matrix {
    let (m, n) = (fx.functions(), mesh.len());
    let mut out = Matrix::zeros(m, n);
    for shell in fx.shells.shells() {
        let images = Images::around(
            ctx,
            &fx.lattice,
            &mesh.region(),
            shell.center,
            shell.radial.support(),
        )
        .unwrap();
        let spline = shell.radial.spline(shell.l).unwrap();
        let nm = shell.ms.len();
        let rows: Vec<i64> = (0..nm as i64).collect();
        for t in images.vectors() {
            let mut center = shell.center;
            for c in 0..3 {
                center[c] += t[c];
            }
            let angle = KPOINT[0] * t[0] + KPOINT[1] * t[1] + KPOINT[2] * t[2];
            let phase = Complex64::new(angle.cos(), angle.sin());
            let mut g = vec![Complex64::new(0., 0.); nm * n];
            bloch_shell(
                ctx,
                &ShellImages {
                    xyz: mesh.points(),
                    center,
                    images: &[[0.; 3]],
                    phases: &[Complex64::new(1., 0.)],
                    support: shell.radial.support(),
                    spline: &spline,
                    ms: &shell.ms,
                    derivative: a as i32,
                },
                &rows,
                nm,
                &mut g,
            )
            .unwrap();
            for (i, &row) in shell.rows.iter().enumerate() {
                for x in 0..n {
                    out[(row, x)] += phase * g[i * n + x] * (mesh.points()[b][x] - center[b]);
                }
            }
        }
    }
    out
}

#[test]
fn the_weighted_kernel_equals_the_one_image_per_call_pattern() {
    // Same support, same images, same quadrature of the radial table: the two routes differ by
    // roundoff of the sums only.
    const IDENTITY: f64 = 1e-12;
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let mesh = fx.mesh(&ctx);
        let (m, n) = (fx.functions(), mesh.len());
        let strain = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT);
        let mut worst = 0f64;
        for (a, b) in all_pairs() {
            let tangent = strain.tangent(&ctx, a, b).unwrap();
            let got = Matrix::from_row_slice(m, n, tangent.as_slice());
            let want = one_image_per_call(&ctx, &fx, &mesh, a, b);
            let error = gap(&got, &want);
            worst = worst.max(error);
            assert!(error < IDENTITY, "{name} ({a},{b}): {error:.2e}");
        }
        println!("{name}: weighted kernel against one image per call, worst {worst:.2e}");
    }
}

#[test]
fn the_weight_is_a_kernel_argument_not_a_post_processing_of_summed_derivatives() {
    // On a cell whose basis reaches the neighbor cells the tangent is NOT (r - c) times the summed
    // derivative; this is the measurement that justifies the kernel mode.
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    let mesh = fx.mesh(&ctx);
    let (m, n) = (fx.functions(), mesh.len());
    let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, KPOINT).unwrap();
    let strain = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT);
    let (a, b) = (0, 1);
    let tangent = strain.tangent(&ctx, a, b).unwrap();
    let got = Matrix::from_row_slice(m, n, tangent.as_slice());
    let slope = Matrix::from_row_slice(m, n, samples.slope(a));
    let home = Matrix::from_fn(m, n, |mu, x| {
        slope[(mu, x)] * (mesh.points()[b][x] - fx.centers[fx.owner[mu]][b])
    });
    let miss = gap(&home, &got);
    println!("si: summed-derivative stand-in misses by {miss:.2e}");
    assert!(miss > 1e-3);
}

#[test]
fn the_tangent_is_reserved_from_the_context_and_released() {
    for name in CASES {
        let fx = Fixture::new(&context(), name);
        // A context of its own, so the peak is that of the mesh and the tangent alone.
        let ctx = context();
        let mesh = Mesh::new(&ctx, &fx.grid).unwrap();
        let n = mesh.len();
        let strain = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT);
        let before = ctx.memory_used();
        let need = tangent_bytes(fx.functions(), n).unwrap();
        let tangent = strain.tangent(&ctx, 1, 2).unwrap();
        assert_eq!(ctx.memory_used() - before, need, "{name}: one component");
        drop(tangent);
        assert_eq!(ctx.memory_used(), before, "{name}: released");
        // Beyond the component the call holds the bin order (32 bytes a node), the scratch of the
        // widest shell and one row (16 bytes a node and function each), and the image lists and
        // phases of one shell (a few KiB, bounded here by 64 KiB).
        let widest = fx.shells.shells().iter().map(|s| s.ms.len()).max().unwrap();
        let transient = (32 + 16 * (widest + 1)) * n + (64 << 10);
        assert!(
            ctx.memory_peak() <= before + need + transient,
            "{name}: peak {} of {}",
            ctx.memory_peak() - before,
            need + transient
        );
        // A context that cannot hold a component refuses.
        let small = Context::new(Some(1), 24 * n + need / 2).unwrap();
        let mesh_small = Mesh::new(&small, &fx.grid).unwrap();
        let tight = StrainSamples::new(&fx.shells, &fx.lattice, &mesh_small, KPOINT);
        assert!(tight.tangent(&small, 0, 0).is_err());
    }
}

/// Refined grids (the recorded ones are coarse): the values, one tangent and a potential-weighted
/// copy of the values, which is what the grid terms of the stress hold at once, are three times
/// the values; the context's own account of the peak is compared with that.
#[test]
fn a_refined_cell_forms_a_tangent_beside_the_values_within_three_times_psi() {
    for (name, shape) in [("si", [36, 36, 36]), ("al2", [36, 36, 50])] {
        let fx = Fixture::with_shape(&context(), name, Some(shape));
        let ctx = context();
        let mesh = Mesh::new(&ctx, &fx.grid).unwrap();
        let (m, n) = (fx.functions(), mesh.len());
        let need = tangent_bytes(m, n).unwrap();
        let mut psi = ctx.allocate::<Complex64>(m * n).unwrap();
        fx.shells
            .values(
                &ctx,
                &Sampling {
                    lattice: &fx.lattice,
                    region: mesh.region(),
                    kpoints: &[KPOINT],
                },
                mesh.points(),
                &mut DenseRows::new(psi.as_mut_slice(), m, n),
            )
            .unwrap();
        let tangent = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT)
            .tangent(&ctx, 0, 1)
            .unwrap();
        let mut weighted = ctx.allocate::<Complex64>(m * n).unwrap();
        let v = rough_potential(n);
        for (row, source) in weighted
            .as_mut_slice()
            .chunks_mut(n)
            .zip(psi.as_slice().chunks(n))
        {
            for ((w, p), v) in row.iter_mut().zip(source).zip(&v) {
                *w = p * *v;
            }
        }
        assert!(peak_of(tangent.as_slice()) > 1e-3 && peak_of(weighted.as_slice()) > 1e-3);
        let widest = fx.shells.shells().iter().map(|s| s.ms.len()).max().unwrap();
        let held = 24 * n + 3 * need;
        println!(
            "{name}: M = {m}, N = {n}, psi {:.1} MiB, context peak {:.1} MiB",
            need as f64 / 1048576.,
            ctx.memory_peak() as f64 / 1048576.
        );
        assert!(
            ctx.memory_peak() <= held + (32 + 16 * (widest + 1)) * n + (64 << 10),
            "{name}: the peak is the mesh, three arrays of psi and the call's scratch"
        );
    }
}

#[test]
fn a_tangent_needs_an_axis_and_a_weight_inside_the_cell_of_indices() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    let mesh = fx.mesh(&ctx);
    let (m, n) = (fx.functions(), mesh.len());
    let strain = StrainSamples::new(&fx.shells, &fx.lattice, &mesh, KPOINT);
    assert!(strain.tangent(&ctx, 3, 0).is_err());
    assert!(strain.tangent(&ctx, 0, 3).is_err());
    // The raw wrapper: a value sum (no derivative) cannot be weighted.
    let shell = &fx.shells.shells()[0];
    let spline = shell.radial.spline(shell.l).unwrap();
    let images: Vec<[f64; 3]> = vec![[0.; 3]];
    let phases = vec![Complex64::new(1., 0.)];
    let mut out = vec![Complex64::new(0., 0.); shell.ms.len() * n];
    let rows: Vec<i64> = (0..shell.ms.len() as i64).collect();
    let call = |derivative: i32| ShellImages {
        xyz: mesh.points(),
        center: shell.center,
        images: &images,
        phases: &phases,
        support: shell.radial.support(),
        spline: &spline,
        ms: &shell.ms,
        derivative,
    };
    assert!(bloch_shell_weighted(&ctx, &call(-1), 0, &rows, shell.ms.len(), &mut out).is_err());
    assert!(bloch_shell_weighted(&ctx, &call(0), 3, &rows, shell.ms.len(), &mut out).is_err());
    assert!(bloch_shell_weighted(&ctx, &call(0), 2, &rows, shell.ms.len(), &mut out).is_ok());
    // A sampling without a k-point is refused.
    let mut sink = vec![Complex64::new(0., 0.); m * n];
    assert!(weighted_derivatives(
        &ctx,
        &fx.shells,
        &Sampling {
            lattice: &fx.lattice,
            region: mesh.region(),
            kpoints: &[],
        },
        mesh.points(),
        0,
        0,
        &mut DenseRows::new(&mut sink, m, n),
    )
    .is_err());
}
