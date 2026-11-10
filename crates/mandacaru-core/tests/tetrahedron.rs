//! The linear tetrahedron density of states (`periodic::tetrahedron`) against analytic densities.
use mandacaru_core::periodic::tetrahedron::TetrahedronMesh;
use nalgebra::Matrix3;
use std::f64::consts::PI;

fn mesh_points(shape: [usize; 3]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                out.push([
                    i as f64 / shape[0] as f64,
                    j as f64 / shape[1] as f64,
                    k as f64 / shape[2] as f64,
                ]);
            }
        }
    }
    out
}

fn cubic_band(k: &[f64; 3]) -> f64 {
    -2. * k.iter().map(|x| (2. * PI * x).cos()).sum::<f64>()
}

#[test]
fn a_simple_cubic_tight_binding_band_holds_one_state_and_is_half_filled_at_zero() {
    let shape = [16, 16, 16];
    let mesh = TetrahedronMesh::new(shape, &(Matrix3::identity() * (2. * PI))).unwrap();
    assert_eq!(mesh.tetrahedra.len(), 6 * 16 * 16 * 16);
    let energies: Vec<f64> = mesh_points(shape).iter().map(cubic_band).collect();
    let axis = [-7., 0., 7.];
    let (_, n) = mesh.dos(&energies, 1, &axis).unwrap();
    assert!(n[0].abs() < 1e-15 && (n[2] - 1.).abs() < 1e-12);
    // k -> k + (1/2, 1/2, 1/2) maps eps to -eps on an even mesh: N(0) = 1/2 exactly.
    assert!((n[1] - 0.5).abs() < 1e-12, "N(0) = {}", n[1]);
    // The density integrates to the band count.
    let fine: Vec<f64> = (0..=6000).map(|i| -6.0 + i as f64 * 0.002).collect();
    let (g, _) = mesh.dos(&energies, 1, &fine).unwrap();
    let total: f64 = g.windows(2).map(|w| 0.5 * (w[0] + w[1]) * 0.002).sum();
    assert!((total - 1.).abs() < 1e-5, "{total}");
}

#[test]
fn the_free_electron_count_converges_at_second_order() {
    // The lowest empty-lattice band of a unit simple-cubic cell, below the inscribed sphere:
    // N(E) = k^3 / (6 pi^2) with k = sqrt(2 E) (one spin channel, unit cell volume).
    let error = |n: usize| -> f64 {
        let shape = [n, n, n];
        let mesh = TetrahedronMesh::new(shape, &(Matrix3::identity() * (2. * PI))).unwrap();
        let energies: Vec<f64> = mesh_points(shape)
            .iter()
            .map(|k| {
                // The nearest image of k: the lowest band of the empty lattice.
                let mut best = f64::INFINITY;
                for a in [0., -1.] {
                    for b in [0., -1.] {
                        for c in [0., -1.] {
                            let q = [k[0] + a, k[1] + b, k[2] + c];
                            let e =
                                0.5 * (2. * PI).powi(2) * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                            best = best.min(e);
                        }
                    }
                }
                best
            })
            .collect();
        let axis = [1.0, 2.0, 4.0];
        let (_, count) = mesh.dos(&energies, 1, &axis).unwrap();
        axis.iter()
            .zip(&count)
            .map(|(e, n)| {
                let exact = (2. * e).powf(1.5) / (6. * PI * PI);
                ((n - exact) / exact).abs()
            })
            .fold(0f64, f64::max)
    };
    // Measured 2026-11-09: 1.03e-1, 2.53e-2, 6.47e-3 at 12^3, 24^3, 48^3 (the linear interpolation of a
    // convex band overestimates it, so the count is low by O(h^2)).
    let errors = [error(12), error(24), error(48)];
    assert!(errors[2] < 1e-2, "relative error {:e} at 48^3", errors[2]);
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!((3.5..4.5).contains(&ratio), "not second order: {errors:?}");
    }
}

#[test]
fn the_projected_density_sums_to_the_total() {
    let shape = [8, 8, 8];
    let mesh = TetrahedronMesh::new(shape, &(Matrix3::identity() * (2. * PI))).unwrap();
    let points = mesh_points(shape);
    let bands = 2;
    let channels = 3;
    let mut energies = Vec::new();
    let mut projections = Vec::new();
    for (i, k) in points.iter().enumerate() {
        for b in 0..bands {
            energies.push(cubic_band(k) + 3. * b as f64);
            let raw: Vec<f64> = (0..channels)
                .map(|c| 1. + ((i * 7 + b * 3 + c * 5) % 11) as f64)
                .collect();
            let sum: f64 = raw.iter().sum();
            projections.extend(raw.iter().map(|v| v / sum));
        }
    }
    let axis: Vec<f64> = (0..400).map(|i| -7. + i as f64 * 0.04).collect();
    let (total, _) = mesh.dos(&energies, bands, &axis).unwrap();
    let projected = mesh
        .projected_dos(&energies, bands, &projections, channels, &axis)
        .unwrap();
    for (x, t) in total.iter().enumerate() {
        let sum: f64 = projected[x * channels..(x + 1) * channels].iter().sum();
        assert!((sum - t).abs() < 1e-12 * t.max(1.), "{sum} vs {t}");
    }
    // A constant projection reproduces the total times that constant.
    let half = vec![0.5; points.len() * bands];
    let p = mesh
        .projected_dos(&energies, bands, &half, 1, &axis)
        .unwrap();
    for (a, t) in p.iter().zip(&total) {
        assert!((a - 0.5 * t).abs() < 1e-12 * t.max(1.));
    }
}

#[test]
fn a_one_dimensional_mesh_reduces_to_the_chain_count() {
    // (n, 1, 1): the tetrahedra degenerate to the linear interpolation along the chain.
    let shape = [200, 1, 1];
    let reciprocal = Matrix3::from_diagonal(&nalgebra::Vector3::new(2. * PI, 0.6, 0.6));
    let mesh = TetrahedronMesh::new(shape, &reciprocal).unwrap();
    let energies: Vec<f64> = mesh_points(shape)
        .iter()
        .map(|k| -2. * (2. * PI * k[0]).cos())
        .collect();
    let axis = [-1.5, -0.3, 0.0, 1.2];
    let (_, count) = mesh.dos(&energies, 1, &axis).unwrap();
    for (e, n) in axis.iter().zip(&count) {
        let exact = (-e / 2.).acos() / PI;
        assert!((n - exact).abs() < 1e-4, "E = {e}: {n} vs {exact}");
    }
}

#[test]
fn mesh_indices_accept_any_order_and_refuse_a_shifted_mesh() {
    let shape = [4, 3, 1];
    let mesh = TetrahedronMesh::new(shape, &(Matrix3::identity() * (2. * PI))).unwrap();
    let mut points = mesh_points(shape);
    points.reverse();
    for p in &mut points {
        if p[0] > 0.5 {
            p[0] -= 1.;
        }
    }
    let indices = mesh.indices(&points).unwrap();
    let mut sorted = indices.clone();
    sorted.sort();
    assert_eq!(sorted, (0..12).collect::<Vec<_>>());
    let shifted: Vec<[f64; 3]> = points.iter().map(|p| [p[0] + 0.125, p[1], p[2]]).collect();
    assert!(mesh.indices(&shifted).is_err());
    assert!(mesh.indices(&points[..11]).is_err());
}

#[test]
fn the_spin_sum_weights_each_channel_and_takes_the_k_points_in_any_order() {
    // Two spin channels (the cubic band, and it shifted up by 1), given at reversed k-points and
    // with a two-channel projection: the sum is the per-channel densities times the degeneracy.
    let shape = [6, 6, 6];
    let mesh = TetrahedronMesh::new(shape, &(Matrix3::identity() * (2. * PI))).unwrap();
    let ordered = mesh_points(shape);
    let up: Vec<f64> = ordered.iter().map(cubic_band).collect();
    let down: Vec<f64> = up.iter().map(|e| e + 1.).collect();
    let reversed: Vec<[f64; 3]> = ordered.iter().rev().copied().collect();
    let levels: Vec<f64> = up.iter().rev().chain(down.iter().rev()).copied().collect();
    // Projection channel 0 carries 0.25 of each state, channel 1 the rest.
    let projections: Vec<f64> = (0..2 * ordered.len()).flat_map(|_| [0.25, 0.75]).collect();
    let axis = [-4., -1., 0.5, 2., 6.];
    let (dos, integrated, projected) = mesh
        .spin_summed(&reversed, &levels, 2, 1, Some((&projections, 2)), 1.5, &axis)
        .unwrap();
    let (d_up, n_up) = mesh.dos(&up, 1, &axis).unwrap();
    let (d_down, n_down) = mesh.dos(&down, 1, &axis).unwrap();
    let projected = projected.unwrap();
    for i in 0..axis.len() {
        let want = 1.5 * (d_up[i] + d_down[i]);
        assert!((dos[i] - want).abs() < 1e-12, "{i}: {} vs {want}", dos[i]);
        assert!((integrated[i] - 1.5 * (n_up[i] + n_down[i])).abs() < 1e-12);
        assert!((projected[2 * i] - 0.25 * want).abs() < 1e-12);
        assert!((projected[2 * i] + projected[2 * i + 1] - dos[i]).abs() < 1e-12);
    }
    assert!(mesh.spin_summed(&reversed, &levels, 2, 1, None, 0., &axis).is_err());
    assert!(mesh.spin_summed(&reversed, &levels[1..], 2, 1, None, 1., &axis).is_err());
}
