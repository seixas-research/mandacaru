//! Wigner-Seitz minimum-image kernel, voxel self-potential and real-kernel transform
//! (`math/wigner_seitz.rs`): parity with the frozen pre-port Python numbers in
//! `test/migration/reference/finite_size/{ws_kernel,voxel,madelung,mpc_n1}.json` and the
//! independent checks of `dev_plan/FINITE_SIZE_PLAN.md` T1-T4.
//!
//! Not checked here (stated, not hidden): the full `ws_kernel.npz` arrays and their sha256 are
//! checked by `test/migration/freeze_finite_size.py verify --full` (the npz is deflated, and a
//! reader would be a new dependency); this file compares the digest (sum, min, max), 16 flat
//! samples and 14 node values of every frozen kernel. Non-cubic Wigner-Seitz cells have no
//! independent `f~(Q)` for `Q != 0` in Rust; their `Q = 0` reference is the frozen Voronoi polyhedron.
use mandacaru_core::math::radial::gauss_legendre;
use mandacaru_core::math::wigner_seitz::{
    asymmetry, minimum_image_kernel, minimum_image_radius, voxel_self_potential, wigner_seitz,
    RealKernel, CUBE_CONSTANT,
};
use mandacaru_core::math::SpatialGrid;
use mandacaru_core::{periodic::ewald::Ewald, Context, Error};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64 as C64;
use serde_json::Value;
use std::f64::consts::PI;

fn load(name: &str) -> Value {
    let path = format!(
        "{}/../../test/migration/reference/finite_size/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn arr(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}
fn matrix(v: &Value) -> [f64; 9] {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(arr).collect();
    std::array::from_fn(|k| rows[k / 3][k % 3])
}
fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}
fn ctx() -> Context {
    Context::new(Some(4), 256 << 20).unwrap()
}
fn grid(n: usize, step: [f64; 9]) -> SpatialGrid {
    SpatialGrid::new([n; 3], [0.; 3], step, true).unwrap()
}
/// Richardson in `h` for an `h^2` law: the value at `h/2` and `h`.
fn richardson(coarse: f64, fine: f64) -> f64 {
    (4. * fine - coarse) / 3.
}

#[test]
fn voxel_self_potential_matches_the_freeze_the_cube_and_an_independent_quadrature() {
    let frozen = load("voxel");
    let nodes = gauss_legendre(64).unwrap();
    let (mut vs_frozen, mut vs_faces, mut vs_cube) = (0_f64, 0_f64, 0_f64);
    for (name, case) in frozen["cases"].as_object().unwrap() {
        let step = Matrix3::from_row_slice(&matrix(&case["step"]));
        let value = voxel_self_potential(&step).unwrap();
        vs_frozen = vs_frozen.max(rel(value, num(&case["value"])));
        // Independent: the same divergence-theorem faces, smooth Gauss-Legendre in place of
        // the closed-form triangles.
        let corner = |s: [usize; 3]| step * Vector3::from_fn(|i, _| s[i] as f64 - 0.5);
        let mut faces = 0.;
        for axis in 0..3 {
            let (o0, o1) = [(1, 2), (0, 2), (0, 1)][axis];
            for side in 0..2 {
                let at = |p, q| {
                    let mut s = [0; 3];
                    s[axis] = side;
                    s[o0] = p;
                    s[o1] = q;
                    corner(s)
                };
                let (v0, e1, e2) = (at(0, 0), at(1, 0) - at(0, 0), at(0, 1) - at(0, 0));
                let normal = e1.cross(&e2);
                let mut integral = 0.;
                for &(x, wx) in &nodes {
                    for &(y, wy) in &nodes {
                        let p = v0 + e1 * (0.5 * (x + 1.)) + e2 * (0.5 * (y + 1.));
                        integral += 0.25 * wx * wy / p.norm();
                    }
                }
                faces += 0.5 * normal.dot(&v0).abs() * integral;
            }
        }
        vs_faces = vs_faces.max(rel(value, faces));
        if let Some(cube) = case.get("analytic_cube") {
            vs_cube = vs_cube.max(rel(value, num(cube)));
        }
        assert!(
            rel(value, faces) < 1e-14,
            "{name}: {value:e} vs faces {faces:e}"
        );
    }
    println!(
        "voxel: vs frozen {vs_frozen:.2e}, vs 64x64 GL faces {vs_faces:.2e}, vs analytic cube {vs_cube:.2e}"
    );
    assert!(vs_frozen < 2e-15 && vs_cube < 1e-15);
    let cube = voxel_self_potential(&Matrix3::identity()).unwrap();
    assert!(rel(cube, CUBE_CONSTANT) < 1e-15);
    let closed = 3. * (2. + 3f64.sqrt()).ln() - PI / 2.;
    assert!((CUBE_CONSTANT - closed).abs() < 1e-15);
    assert!(voxel_self_potential(&Matrix3::zeros()).is_err());
    assert!(voxel_self_potential(&Matrix3::from_element(f64::NAN)).is_err());
}

#[test]
fn kernels_reproduce_the_frozen_samples_digest_and_symmetry() {
    let frozen = load("ws_kernel");
    let ctx = ctx();
    let (mut worst_node, mut worst_digest, mut worst_sum, mut worst_asym) =
        (0_f64, 0_f64, 0_f64, 0_f64);
    for (name, rec) in frozen["kernels"].as_object().unwrap() {
        let n = rec["n"].as_u64().unwrap() as usize;
        let g = grid(n, matrix(&rec["step"]));
        let kernel = minimum_image_kernel(&ctx, &g, 2).unwrap();
        let k = kernel.as_slice();
        for (index, value) in rec["node_index"]
            .as_array()
            .unwrap()
            .iter()
            .zip(arr(&rec["node_value"]))
        {
            let [i, j, l] = [0, 1, 2].map(|a| index[a].as_u64().unwrap() as usize);
            worst_node = worst_node.max(rel(k[(i * n + j) * n + l], value));
        }
        let samples = &rec["digest"]["samples"];
        for (index, value) in samples["index"]
            .as_array()
            .unwrap()
            .iter()
            .zip(arr(&samples["value"]))
        {
            worst_node = worst_node.max(rel(k[index.as_u64().unwrap() as usize], value));
        }
        let digest = &rec["digest"];
        let sum: f64 = k.iter().sum();
        worst_digest = worst_digest
            .max(rel(sum, num(&digest["sum"])))
            .max(rel(
                k.iter().copied().fold(0., f64::max),
                num(&digest["max"]),
            ))
            .max(rel(
                k.iter().copied().fold(f64::INFINITY, f64::min),
                num(&digest["min"]),
            ));
        let dv = g.index.dv();
        worst_sum = worst_sum
            .max(rel(sum * dv, num(&rec["sum_times_dV"])))
            .max(rel(k[0], num(&rec["K000"])))
            .max(rel(minimum_image_radius(k), num(&rec["min_image_radius"])));
        // Every node is covered: the "enclosed volume" is the cell volume (trivially).
        assert!(k.iter().all(|&v| v > 0.), "{name}");
        assert!((k.len() as f64 * dv - num(&rec["volume"])).abs() < 1e-9 * num(&rec["volume"]));
        // Even to rounding, not exactly: the triclinic cell has 2.8e-17 asymmetric nodes.
        let asym = asymmetry(k, [n; 3]).unwrap();
        worst_asym = worst_asym.max(asym);
        assert!(
            asym < 1e-15 && (asym - num(&rec["asymmetry_max"])).abs() < 1e-15,
            "{name}: {asym:e}"
        );
    }
    println!(
        "kernels: node/sample {worst_node:.2e}, digest sum/max/min {worst_digest:.2e}, sum*dV/K000/radius {worst_sum:.2e}, asymmetry {worst_asym:.2e}"
    );
    assert!(worst_node < 1e-13 && worst_digest < 1e-12 && worst_sum < 1e-12);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn kernel_sum_converges_to_the_cell_integral_with_order_two() {
    let frozen = load("ws_kernel");
    let ctx = ctx();
    let cube = num(&frozen["cube_constant"]);
    assert!((cube - CUBE_CONSTANT).abs() < 1e-15);
    // (lattice, L) -> sums times dV at n = 12, 24, 48, (96) and the cell integral.
    let mut groups: std::collections::BTreeMap<String, Vec<(usize, f64, f64)>> = Default::default();
    for (name, rec) in frozen["kernels"].as_object().unwrap() {
        let n = rec["n"].as_u64().unwrap() as usize;
        let g = grid(n, matrix(&rec["step"]));
        let kernel = minimum_image_kernel(&ctx, &g, 2).unwrap();
        let sum = kernel.as_slice().iter().sum::<f64>() * g.index.dv();
        let key = name.rsplit_once("_n").unwrap().0.to_string();
        groups
            .entry(key)
            .or_default()
            .push((n, sum, num(&rec["reference"])));
    }
    let mut worst_order = 0_f64;
    for (key, rows) in &mut groups {
        rows.sort_by_key(|r| r.0);
        let errors: Vec<f64> = rows.iter().map(|r| r.1 - r.2).collect();
        if key.starts_with("sc_") {
            let l: f64 = key.rsplit_once('L').unwrap().1.parse().unwrap();
            assert!(
                rel(rows[0].2, cube * l * l) < 1e-15,
                "{key}: reference is the cube integral"
            );
        }
        let key_r = if rows.len() == 4 {
            "richardson_with_96"
        } else {
            "richardson"
        };
        let frozen_orders =
            arr(&frozen[key_r][key.as_str()]["observed_order_between_consecutive_n"]);
        for (w, frozen_order) in errors.windows(2).zip(frozen_orders) {
            let order = (w[0] / w[1]).log2();
            worst_order = worst_order.max((order - 2.).abs());
            assert!(
                (order - frozen_order).abs() < 1e-6,
                "{key}: order {order} vs frozen {frozen_order}"
            );
            assert!((order - 2.).abs() < 5e-2, "{key}: order {order}");
        }
        // T4(a): f~(0) = int_cube 1/r. The first Richardson stage leaves an h^4 term (its
        // error shrinks by 16 per halving); a second stage removes it.
        if rows.len() == 4 {
            let r1 = [
                richardson(rows[0].1, rows[1].1) - rows[0].2,
                richardson(rows[1].1, rows[2].1) - rows[0].2,
                richardson(rows[2].1, rows[3].1) - rows[0].2,
            ];
            let ratio = r1[1] / r1[2];
            let second = ((16. * r1[2] - r1[1]) / 15.).abs() / rows[0].2;
            println!("{key}: R1 error ratio {ratio:.3}, two-stage Richardson relative error {second:.2e}");
            assert!((ratio - 16.).abs() < 0.8, "{key}: R1 ratio {ratio}");
            assert!(second < 1e-9, "{key}: T4(a) two-stage error {second:e}");
        }
        // The n = 96 value lands on the extrapolation of the coefficient measured at 24 and 48.
        if rows.len() == 4 {
            let predicted = errors[2] / 4.;
            assert!(
                rel(errors[3], predicted) < 1e-3,
                "{key}: {} vs {predicted}",
                errors[3]
            );
        }
        // Richardson: the frozen extrapolated error (absolute) is reproduced.
        let want = arr(&frozen[key_r][key.as_str()]["extrapolated_error"]);
        for (i, w) in want.iter().enumerate() {
            let got = richardson(rows[i].1, rows[i + 1].1) - rows[0].2;
            assert!(
                (got - w).abs() < 1e-6 * w.abs() + 1e-9,
                "{key}: {got:e} vs {w:e}"
            );
        }
    }
    println!(
        "order two: worst |order - 2| over {} groups = {worst_order:.2e}",
        groups.len()
    );
}

#[test]
fn a_wider_search_changes_nothing_and_bad_input_raises() {
    let ctx = ctx();
    let frozen = load("ws_kernel");
    let rec = &frozen["kernels"]["triclinic_L12.0_n12"];
    let g = grid(
        24,
        matrix(&frozen["kernels"]["triclinic_L12.0_n24"]["step"]),
    );
    let two = minimum_image_kernel(&ctx, &g, 2).unwrap();
    let three = minimum_image_kernel(&ctx, &g, 3).unwrap();
    let one = minimum_image_kernel(&ctx, &g, 1).unwrap();
    assert_eq!(two.as_slice(), three.as_slice());
    let changed = one
        .as_slice()
        .iter()
        .zip(two.as_slice())
        .filter(|(a, b)| a != b)
        .count();
    println!("triclinic n=24: shells 1 vs 2 differ at {changed} nodes; shells 2 vs 3 differ at 0");
    drop((one, two, three));
    let open = SpatialGrid::new([4; 3], [0.; 3], matrix(&rec["step"]), false).unwrap();
    assert!(matches!(
        minimum_image_kernel(&ctx, &open, 2),
        Err(Error::Invalid(_))
    ));
    assert!(
        minimum_image_kernel(&ctx, &g, 0).is_err() && minimum_image_kernel(&ctx, &g, 5).is_err()
    );
    let tiny = Context::new(Some(1), 1 << 16).unwrap();
    let big = grid(48, matrix(&frozen["kernels"]["sc_L12.0_n48"]["step"]));
    assert!(matches!(
        minimum_image_kernel(&tiny, &big, 2),
        Err(Error::Budget { .. })
    ));
    assert!(matches!(
        wigner_seitz(&tiny, &big, 2),
        Err(Error::Budget { .. })
    ));
    assert_eq!((tiny.memory_used(), ctx.memory_used()), (0, 0));
}

#[test]
fn memory_is_reserved_from_the_context_and_bounded() {
    let frozen = load("ws_kernel");
    let ctx = ctx();
    let n = 48;
    let g = grid(n, matrix(&frozen["kernels"]["sc_L12.0_n48"]["step"]));
    let points = n * n * n;
    let solver = wigner_seitz(&ctx, &g, 2).unwrap();
    let resident = ctx.memory_used();
    // Kernel (dropped) 8 N + spectrum 8 N + workspace 16 N, plus the FFT plans.
    assert!(
        ctx.memory_peak() <= 32 * points + (2 << 20),
        "peak {}",
        ctx.memory_peak()
    );
    assert!(
        resident <= 24 * points + (2 << 20) && resident >= 24 * points,
        "resident {resident}"
    );
    println!(
        "n=48: peak {} B, resident {} B, 32 N = {} B",
        ctx.memory_peak(),
        resident,
        32 * points
    );
    drop(solver);
    assert_eq!(ctx.memory_used(), 0);
}

/// The transform and the convolution against a direct O(N^2) evaluation, on a skewed grid.
#[test]
fn transform_and_convolution_match_the_direct_sums() {
    let ctx = ctx();
    let shape = [5, 4, 6];
    let step = [0.5, 0.1, 0., 0., 0.4, 0.05, 0.1, 0., 0.45];
    let g = SpatialGrid::new(shape, [0.; 3], step, true).unwrap();
    let n = g.index.points();
    let index = |i: usize, j: usize, k: usize| (i * shape[1] + j) * shape[2] + k;
    let mut kernel = vec![0.; n];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let (a, b) = (index(i, j, k), index((5 - i) % 5, (4 - j) % 4, (6 - k) % 6));
                let v = ((a * 37 + 11) % 17) as f64 / 7. + 0.1;
                kernel[a] = v;
                kernel[b] = v;
            }
        }
    }
    let mut solver = RealKernel::new(&ctx, &g, &kernel).unwrap();
    let dv = g.index.dv();
    let mut worst = 0_f64;
    for m in 0..n {
        let (mi, mj, mk) = (m / 24, (m / 6) % 4, m % 6);
        let mut direct = C64::default();
        for d in 0..n {
            let (di, dj, dk) = (d / 24, (d / 6) % 4, d % 6);
            let phase = -2. * PI * (mi * di) as f64 / 5.
                - 2. * PI * (mj * dj) as f64 / 4.
                - 2. * PI * (mk * dk) as f64 / 6.;
            direct += C64::from_polar(kernel[d] * dv, phase);
        }
        assert!(direct.im.abs() < 1e-12);
        worst = worst.max((direct.re - solver.spectrum()[m]).abs() / direct.re.abs().max(1.));
    }
    assert!(worst < 1e-13, "spectrum {worst:e}");
    let rho: Vec<C64> = (0..n)
        .map(|i| C64::new((0.3 * i as f64).sin(), (0.11 * i as f64).cos()))
        .collect();
    let mut out = vec![C64::default(); n];
    solver.convolve_into(&rho, &mut out).unwrap();
    let mut conv_worst = 0_f64;
    for a in 0..n {
        let (ai, aj, ak) = (a / 24, (a / 6) % 4, a % 6);
        let mut direct = C64::default();
        for b in 0..n {
            let (bi, bj, bk) = (b / 24, (b / 6) % 4, b % 6);
            direct += rho[b]
                * kernel[index((ai + 5 - bi) % 5, (aj + 4 - bj) % 4, (ak + 6 - bk) % 6)]
                * dv;
        }
        conv_worst = conv_worst.max((direct - out[a]).norm() / direct.norm());
    }
    assert!(conv_worst < 1e-13, "convolution {conv_worst:e}");
    // Parseval: the quadratic form needs no inverse transform.
    let direct: f64 = rho
        .iter()
        .zip(&out)
        .map(|(r, v)| (r.conj() * v).re)
        .sum::<f64>()
        * dv;
    let form = solver.quadratic_form(&rho).unwrap();
    assert!(rel(form, direct) < 1e-13, "{form} vs {direct}");
    // An odd kernel, a non-finite density and a wrong length are refused.
    let mut odd = kernel.clone();
    odd[1] += 1.;
    assert!(RealKernel::new(&ctx, &g, &odd).is_err());
    assert!(solver.convolve_into(&rho[1..], &mut out).is_err());
    let mut bad = rho.clone();
    bad[3] = C64::new(f64::NAN, 0.);
    assert!(solver.quadratic_form(&bad).is_err());
    println!("transform {worst:.1e}, convolution {conv_worst:.1e}");
    drop(solver);
    assert_eq!(ctx.memory_used(), 0);
}

/// Analytic check of transform and convolution: a Gaussian kernel sampled well inside the cell.
#[test]
fn gaussian_kernel_transform_and_convolution_are_analytic() {
    let ctx = ctx();
    // The Gaussians are sampled on one cell of L = 6 and must be negligible at its edge and
    // free of aliasing: exp(-a L^2/4) = 2e-16 at a = 4, and exp(-pi^2/(c h^2)) < 1e-17 for the
    // widest product (c = a + b = 7) at n = 32. Smaller a or n leaves cut-off or aliasing error.
    let (n, l, a) = (32, 6., 4.);
    let h = l / n as f64;
    let g = grid(n, [h, 0., 0., 0., h, 0., 0., 0., h]);
    let signed = |i: usize| if i <= n / 2 { i as f64 } else { i as f64 - n as f64 } * h;
    let mut kernel = vec![0.; n * n * n];
    let mut density = vec![C64::default(); n * n * n];
    let b = 3.0; // density exp(-b r^2)
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = signed(i).powi(2) + signed(j).powi(2) + signed(k).powi(2);
                kernel[(i * n + j) * n + k] = (-a * r2).exp();
                density[(i * n + j) * n + k] = C64::new((-b * r2).exp(), 0.);
            }
        }
    }
    let mut solver = RealKernel::new(&ctx, &g, &kernel).unwrap();
    let q = |m: usize| {
        2. * PI
            * (if m <= n / 2 {
                m as f64
            } else {
                m as f64 - n as f64
            })
            / l
    };
    let mut worst = 0_f64;
    for (mi, mj, mk) in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 3), (31, 1, 0)] {
        let q2 = q(mi).powi(2) + q(mj).powi(2) + q(mk).powi(2);
        let exact = (PI / a).powf(1.5) * (-q2 / (4. * a)).exp();
        worst = worst.max(rel(solver.spectrum()[(mi * n + mj) * n + mk], exact));
    }
    // Convolution of exp(-a r^2) with exp(-b r^2) at the origin: (pi/(a+b))^{3/2}.
    let mut out = vec![C64::default(); n * n * n];
    solver.convolve_into(&density, &mut out).unwrap();
    let want = (PI / (a + b)).powf(1.5);
    println!(
        "gaussian: spectrum {worst:.2e}, conv(0) {} vs {want}",
        out[0].re
    );
    assert!(worst < 1e-10 && rel(out[0].re, want) < 1e-10);
    drop(solver);
    assert_eq!(ctx.memory_used(), 0);
}

/// Cube `f~(Q) = int_cube e^{-iQ.r}/r` by an independent quadrature: six faces, analytic
/// radial integral, Gauss-Legendre in the face coordinates.
fn cube_transform(l: f64, q: [f64; 3]) -> f64 {
    let nodes = gauss_legendre(48).unwrap();
    let mut total = 0.;
    for axis in 0..3 {
        for sign in [1., -1.] {
            for &(u, wu) in &nodes {
                for &(v, wv) in &nodes {
                    let norm = (1. + u * u + v * v).sqrt();
                    let mut omega = [0.; 3];
                    let others: Vec<usize> = (0..3).filter(|&i| i != axis).collect();
                    omega[axis] = sign / norm;
                    omega[others[0]] = u / norm;
                    omega[others[1]] = v / norm;
                    let radius = 0.5 * l * norm;
                    let z = radius * (q[0] * omega[0] + q[1] * omega[1] + q[2] * omega[2]);
                    let radial = if z.abs() < 1e-3 {
                        radius * radius * (0.5 - z * z / 8. + z.powi(4) / 144.)
                    } else {
                        radius * radius * (z.sin() / z + (z.cos() - 1.) / (z * z))
                    };
                    total += wu * wv / norm.powi(3) * radial;
                }
            }
        }
    }
    total
}

#[test]
fn cube_transform_at_the_reciprocal_nodes_converges_to_the_independent_quadrature() {
    let ctx = ctx();
    let frozen = load("ws_kernel");
    let l = 12.;
    assert!(rel(cube_transform(l, [0.; 3]), CUBE_CONSTANT * l * l) < 1e-14);
    let modes = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0], [3, 0, 0]];
    let mut errors = vec![];
    for n in [24, 48, 96] {
        let g = grid(
            n,
            matrix(&frozen["kernels"][format!("sc_L12.0_n{n}")]["step"]),
        );
        let solver = wigner_seitz(&ctx, &g, 2).unwrap();
        errors.push(modes.map(|m| {
            let q = m.map(|c| 2. * PI * c as f64 / l);
            solver.spectrum()[(m[0] * n + m[1]) * n + m[2]] - cube_transform(l, q)
        }));
    }
    for (i, m) in modes.iter().enumerate() {
        let (o1, o2) = (
            (errors[0][i] / errors[1][i]).log2(),
            (errors[1][i] / errors[2][i]).log2(),
        );
        let rich = richardson(errors[1][i], errors[2][i]);
        println!("f~{m:?}: errors {:.3e} {:.3e} {:.3e}, orders {o1:.3} {o2:.3}, Richardson(48,96) {rich:.2e}", errors[0][i], errors[1][i], errors[2][i]);
        // (3,0,0) is not asymptotic yet at n = 48 (order 1.904): printed, not asserted.
        if *m != [3, 0, 0] {
            assert!(o2 > 1.95 && o2 < 2.05, "{m:?}: order {o2}");
        }
        assert!(
            rich.abs() < 0.1 * errors[2][i].abs(),
            "{m:?}: Richardson {rich:e} vs raw {:e}",
            errors[2][i]
        );
    }
}

#[test]
fn a_sphere_truncated_kernel_goes_through_the_same_transform() {
    let ctx = ctx();
    let frozen = load("ws_kernel");
    let (l, radius) = (12., 4.2);
    let mut report = vec![];
    for n in [24, 48, 96] {
        let g = grid(
            n,
            matrix(&frozen["kernels"][format!("sc_L12.0_n{n}")]["step"]),
        );
        let mut kernel = minimum_image_kernel(&ctx, &g, 2).unwrap();
        for k in kernel.as_mut_slice().iter_mut().skip(1) {
            if 1. / *k > radius {
                *k = 0.;
            }
        }
        let solver = RealKernel::new(&ctx, &g, kernel.as_slice()).unwrap();
        let mut worst = (0_f64, 0_f64);
        for m in [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 1, 0],
        ] {
            let q = m.map(|c| 2. * PI * c as f64 / l);
            let q2: f64 = q.iter().map(|x| x * x).sum();
            let exact = if q2 == 0. {
                2. * PI * radius * radius
            } else {
                4. * PI * (1. - (q2.sqrt() * radius).cos()) / q2
            };
            let err = (solver.spectrum()[(m[0] * n + m[1]) * n + m[2]] - exact).abs()
                / (2. * PI * radius * radius);
            worst = (worst.0.max(err), if q2 == 0. { err } else { worst.1 });
        }
        report.push(worst.0);
        println!(
            "sphere R={radius} n={n}: worst |f~ - exact| / (2 pi R^2) = {:.3e} (Q = 0: {:.3e})",
            worst.0, worst.1
        );
    }
    // A sharp cutoff misclassifies boundary voxels: the order is NOT two; it must still fall.
    assert!(report[2] < report[0] && report[2] < 5e-3, "{report:?}");
}

/// Independent point-charge Ewald (real + reciprocal + self + background); `v_M` per unit charge.
fn madelung(rows: [f64; 9], eta: f64) -> f64 {
    let cell = Matrix3::from_row_slice(&rows).transpose(); // columns are lattice vectors
    let omega = cell.determinant().abs();
    let dual = cell.try_inverse().unwrap().transpose() * (2. * PI);
    let range = -9i32..=9;
    let (mut real, mut recip) = (0., 0.);
    for i in range.clone() {
        for j in range.clone() {
            for k in range.clone() {
                if (i, j, k) == (0, 0, 0) {
                    continue;
                }
                let m = Vector3::new(i as f64, j as f64, k as f64);
                let r = (cell * m).norm();
                real += libm::erfc(eta * r) / r;
                let g2 = (dual * m).norm_squared();
                recip += (-g2 / (4. * eta * eta)).exp() / g2;
            }
        }
    }
    real + 4. * PI / omega * recip - 2. * eta / PI.sqrt() - PI / (eta * eta * omega)
}

#[test]
fn madelung_constants_agree_with_the_freeze_an_independent_ewald_and_the_native_ewald() {
    let frozen = load("madelung");
    let ctx = ctx();
    for name in ["sc", "fcc", "bcc", "hex"] {
        let entry = &frozen[name];
        let cell = matrix(&entry["cell"]);
        let (v2, v3) = (madelung(cell, 2.), madelung(cell, 3.));
        let native = 2.
            * Ewald::new(&ctx, &[[0.; 3]], &[1.], cell, 1e-14, None)
                .unwrap()
                .evaluate(&ctx)
                .unwrap()
                .energy;
        let rws = (3. * entry["volume"].as_f64().unwrap() / (4. * PI)).cbrt();
        println!(
            "{name}: v_M {v2:.15} (eta spread {:.1e}, vs freeze {:.1e}, native {:.1e}), v_M r_WS {:.10}",
            (v2 - v3).abs(), (v2 - num(&entry["vM_eta2"])).abs(), (native - v2).abs(), v2 * rws
        );
        assert!((v2 - v3).abs() < 1e-13, "{name}: eta spread");
        assert!(
            (v2 - num(&entry["vM_eta2"])).abs() < 1e-13,
            "{name}: freeze"
        );
        assert!((native - v2).abs() < 1e-9, "{name}: native Ewald");
        assert!((v2 * rws - num(&entry["vM_times_rws_eta2"])).abs() < 1e-13);
    }
    let sc = madelung(matrix(&frozen["sc"]["cell"]), 2.);
    assert!((sc + 2.837297479).abs() < 1e-9, "sc L v_M = {sc}");
    for (name, want) in [
        ("sc", -1.7601188842),
        ("fcc", -1.7917472304),
        ("bcc", -1.7918585114),
    ] {
        let entry = &frozen[name];
        let rws = (3. * entry["volume"].as_f64().unwrap() / (4. * PI)).cbrt();
        assert!(
            (madelung(matrix(&entry["cell"]), 2.) * rws - want).abs() < 1e-9,
            "{name}"
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The N = 1 Gaussian of `_jgauss`: Gaussian at signed offsets, cut at the cell edge and
/// normalized by the grid sum; `J_f` through the Rust transform.
fn pair_integral(ctx: &Context, l: f64, n: usize, s: f64) -> f64 {
    let h = l / n as f64;
    let g = grid(n, [h, 0., 0., 0., h, 0., 0., 0., h]);
    let off = |i: usize| {
        (if i <= n / 2 {
            i as f64
        } else {
            i as f64 - n as f64
        }) * h
    };
    let mut rho = vec![C64::default(); n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = off(i).powi(2) + off(j).powi(2) + off(k).powi(2);
                rho[(i * n + j) * n + k] = C64::new((-r2 / (2. * s * s)).exp(), 0.);
            }
        }
    }
    let total: f64 = rho.iter().map(|v| v.re).sum::<f64>() * h * h * h;
    rho.iter_mut().for_each(|v| *v /= total);
    wigner_seitz(ctx, &g, 2)
        .unwrap()
        .quadratic_form(&rho)
        .unwrap()
}

/// T3, kernel half: the N = 1 correction `1/2 (J_E - J_f) - 1/2 v_M` must equal `2 pi s^2/Omega`
/// to 1 % of itself at L = 10, 14, 18. The raw n = 48 kernel FAILS that gate (documented
/// negative control, so nobody loosens it); the Richardson value from n = 48 and 96 passes.
/// A voxel-integrated near-node kernel was measured too and is worse (see the log), so the
/// gate is on Richardson only.
#[test]
fn the_n1_gaussian_gate_is_met_by_richardson_and_failed_by_the_raw_kernel() {
    let frozen = load("mpc_n1");
    let ctx = ctx();
    for l in [10., 14., 18.] {
        let mut corrected = vec![];
        for n in [32, 48, 96] {
            let rec = &frozen["cases"][format!("L{l:?}_s1.0_n{n}")];
            let jf = pair_integral(&ctx, l, n, 1.);
            assert!(
                rel(jf, num(&rec["Jf"])) < 1e-12,
                "L={l} n={n}: Jf {jf:e} vs {:e}",
                num(&rec["Jf"])
            );
            corrected.push(0.5 * (num(&rec["JE"]) - jf) - 0.5 * num(&rec["vM"]));
            if n == 48 {
                assert!(rel(corrected[1], num(&rec["corrected_by_formula"])) < 1e-9);
            }
        }
        let signal = 2. * PI / (l * l * l);
        assert!(
            rel(
                signal,
                num(&frozen["cases"][format!("L{l:?}_s1.0_n48")]["corrected_expected"])
            ) < 1e-14
        );
        let raw = (corrected[1] - signal).abs() / signal;
        let rich = (richardson(corrected[1], corrected[2]) - signal).abs() / signal;
        let rich_coarse = (richardson(corrected[0], corrected[1]) - signal).abs() / signal;
        println!(
            "L={l}: raw n=48 off by {:.2} % of 2 pi s^2/Omega (n=96 {:.2} %), Richardson(32,48) {:.3} %, Richardson(48,96) {:.4} %",
            100. * raw, 100. * (corrected[2] - signal).abs() / signal, 100. * rich_coarse, 100. * rich
        );
        assert!(
            raw > 0.01,
            "negative control: the raw n = 48 kernel must fail the 1 % gate"
        );
        assert!(rich < 0.01, "L={l}: Richardson {rich}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
