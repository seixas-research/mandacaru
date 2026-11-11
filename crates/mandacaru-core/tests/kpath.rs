//! Band paths and the irreducible zone (`periodic::kpath`), against analytic Brillouin-zone
//! geometry: the distances of the square lattice, the high-symmetry points of every named
//! lattice lying on the zone boundary (and at their known Cartesian positions), the mesh points
//! on a path, and the orbits of the point group, weights summing to one.
use mandacaru_core::periodic::kpath::{convention, irreducible_zone, select, BandPath, Metric};
use mandacaru_core::periodic::spectrum;
use mandacaru_core::Context;
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}
const ALL: [bool; 3] = [true; 3];
const PLANE: [bool; 3] = [true, true, false];

fn diagonal(a: f64, b: f64, c: f64) -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(a, b, c))
}
fn rows(v: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| v[i][j])
}

/// The Cartesian wavevector of fractional `f`: `2 pi f^T L^-1` for the rows `L` of the cell.
fn cartesian(cell: &Matrix3<f64>, f: [f64; 3]) -> Vector3<f64> {
    2. * PI * cell.try_inverse().unwrap() * Vector3::from(f)
}

/// Every point of a named lattice that is not the zone center lies on the boundary of the first
/// Brillouin zone: it is as close to the origin as to some other reciprocal-lattice point, and
/// no reciprocal-lattice point is closer.
fn on_the_zone_boundary(cell: &Matrix3<f64>, periodic: [bool; 3]) {
    let found = convention(cell, periodic).unwrap();
    // The columns of 2 pi L^-1 are the reciprocal vectors.
    let reciprocal = 2. * PI * cell.try_inverse().unwrap();
    let axes: Vec<usize> = (0..3).filter(|&a| periodic[a]).collect();
    let metric = Metric::new(cell, periodic).unwrap();
    for (label, local) in &found.points {
        let mut f = [0.; 3];
        for (c, &a) in axes.iter().enumerate() {
            f[a] = local[c];
        }
        let k = cartesian(cell, f);
        assert!((k.norm() - metric.norm(f)).abs() < 1e-12 * (1. + k.norm()));
        if *label == "G" {
            assert!(k.norm() < 1e-14);
            continue;
        }
        let mut nearest = f64::INFINITY;
        for n0 in -3..=3 {
            for n1 in -3..=3 {
                for n2 in -3..=3 {
                    let n = [n0, n1, n2];
                    if n.iter().enumerate().any(|(a, &v)| v != 0 && !periodic[a]) || n == [0, 0, 0]
                    {
                        continue;
                    }
                    let g = reciprocal.column(0) * f64::from(n0)
                        + reciprocal.column(1) * f64::from(n1)
                        + reciprocal.column(2) * f64::from(n2);
                    nearest = nearest.min((k - g).norm());
                }
            }
        }
        // |k - G| >= |k| for every G (inside the zone) with equality for some (on its boundary).
        assert!(
            (nearest - k.norm()).abs() < 1e-10,
            "{} {label}: nearest image {nearest}, |k| {}",
            found.name,
            k.norm()
        );
    }
}

#[test]
fn the_high_symmetry_points_of_every_named_lattice_lie_on_the_zone_boundary() {
    let a = 3.1;
    on_the_zone_boundary(&diagonal(a, 20., 20.), [true, false, false]);
    on_the_zone_boundary(&diagonal(a, a, 20.), PLANE);
    on_the_zone_boundary(&diagonal(a, 1.4 * a, 20.), PLANE);
    let hex = |gamma: f64| {
        rows([
            [a, 0., 0.],
            [a * gamma.cos(), a * gamma.sin(), 0.],
            [0., 0., 20.],
        ])
    };
    on_the_zone_boundary(&hex(120_f64.to_radians()), PLANE);
    on_the_zone_boundary(&hex(60_f64.to_radians()), PLANE);
    on_the_zone_boundary(&diagonal(a, a, a), ALL);
    on_the_zone_boundary(&diagonal(a, a, 1.7 * a), ALL);
    on_the_zone_boundary(&diagonal(a, 1.3 * a, 1.7 * a), ALL);
    let h = a / 2.;
    on_the_zone_boundary(&rows([[0., h, h], [h, 0., h], [h, h, 0.]]), ALL);
    on_the_zone_boundary(&rows([[-h, h, h], [h, -h, h], [h, h, -h]]), ALL);
    let hex3 = rows([
        [a, 0., 0.],
        [
            a * 120_f64.to_radians().cos(),
            a * 120_f64.to_radians().sin(),
            0.,
        ],
        [0., 0., 1.6 * a],
    ]);
    on_the_zone_boundary(&hex3, ALL);
}

#[test]
fn the_fcc_and_bcc_points_are_at_their_known_cartesian_positions() {
    let a = 4.05;
    let h = a / 2.;
    let fcc = rows([[0., h, h], [h, 0., h], [h, h, 0.]]);
    let unit = 2. * PI / a;
    let at = |cell: &Matrix3<f64>, label: &str| {
        let f = convention(cell, ALL).unwrap().point(label).unwrap();
        cartesian(cell, f)
    };
    let expect = |v: Vector3<f64>, w: [f64; 3]| {
        assert!(
            (v - Vector3::from(w) * unit).norm() < 1e-12,
            "{v:?} vs {w:?}"
        );
    };
    expect(at(&fcc, "X"), [0., 1., 0.]);
    expect(at(&fcc, "L"), [0.5, 0.5, 0.5]);
    expect(at(&fcc, "K"), [0.75, 0.75, 0.]);
    expect(at(&fcc, "W"), [0.5, 1., 0.]);
    expect(at(&fcc, "U"), [0.25, 1., 0.25]);
    let bcc = rows([[-h, h, h], [h, -h, h], [h, h, -h]]);
    expect(at(&bcc, "H"), [0., 1., 0.]);
    expect(at(&bcc, "N"), [0.5, 0.5, 0.]);
    expect(at(&bcc, "P"), [0.5, 0.5, 0.5]);
}

#[test]
fn the_square_lattice_path_has_the_distances_of_its_zone() {
    let a = 2.6;
    let cell = diagonal(a, a, 9.);
    let ctx = context();
    let mesh = spectrum::mesh(&ctx, [4, 4, 1], true, &[], false).unwrap();
    let path = BandPath::named(&cell, PLANE, Some("GXMG")).unwrap();
    assert_eq!(path.convention, "square");
    let selected = select(&path, &Metric::new(&cell, PLANE).unwrap(), &mesh.full).unwrap();
    assert!(selected.complete() && selected.missing.is_empty());
    let unit = PI / a;
    let expect = [
        0.,
        0.5,
        1.,
        1.5,
        2.,
        2. + 0.5 * 2f64.sqrt(),
        2. + 2f64.sqrt(),
    ];
    assert_eq!(selected.distances.len(), expect.len());
    for (d, e) in selected.distances.iter().zip(expect) {
        assert!((d - e * unit).abs() < 1e-12, "{d} vs {}", e * unit);
    }
    assert_eq!(selected.labels, ["G", "X", "M", "G"]);
    let ticks = [0., 1., 2., 2. + 2f64.sqrt()];
    for (d, e) in selected.label_distances.iter().zip(ticks) {
        assert!((d - e * unit).abs() < 1e-12);
    }
    // The k-points are the vertices and the mesh points between them.
    let k: Vec<_> = selected.kpoints.iter().map(|k| [k[0], k[1]]).collect();
    assert_eq!(k.len(), 7);
    assert_eq!(k[1], [0.25, 0.]);
    assert_eq!(k[2], [0.5, 0.]);
    assert_eq!(k[3], [0.5, 0.25]);
    assert_eq!(k[5], [0.25, 0.25]);
}

#[test]
fn a_mesh_that_misses_a_vertex_reports_it_and_keeps_the_points_it_has() {
    let cell = diagonal(2.6, 2.6, 9.);
    let ctx = context();
    let mesh = spectrum::mesh(&ctx, [3, 3, 1], false, &[], false).unwrap();
    let path = BandPath::named(&cell, PLANE, Some("GXMG")).unwrap();
    let selected = select(&path, &Metric::new(&cell, PLANE).unwrap(), &mesh.full).unwrap();
    assert!(!selected.complete());
    assert_eq!(selected.missing, ["X", "M"]);
    // G, (1/3, 0) on the first segment, (1/3, 1/3) on the diagonal, and G again.
    let k: Vec<_> = selected.kpoints.iter().map(|k| [k[0], k[1]]).collect();
    assert_eq!(k.len(), 4);
    assert!(k[0] == [0., 0.] && k[3] == [0., 0.]);
    assert!((k[1][0] - 1. / 3.).abs() < 1e-12 && k[1][1] == 0.);
    assert!((k[2][0] - 1. / 3.).abs() < 1e-12 && (k[2][1] - 1. / 3.).abs() < 1e-12);
    let unit = PI / 2.6;
    assert!((selected.distances[1] - unit * 2. / 3.).abs() < 1e-12);
    // (1/3, 1/3) is a third of the way from M to G.
    assert!((selected.distances[2] - unit * (2. + 2f64.sqrt() / 3.)).abs() < 1e-12);
}

#[test]
fn a_reciprocal_lattice_image_of_a_mesh_point_is_on_the_path_and_a_missing_path_raises() {
    let cell = diagonal(2.6, 2.6, 9.);
    let ctx = context();
    let mesh = spectrum::mesh(&ctx, [4, 4, 1], true, &[], false).unwrap();
    let metric = Metric::new(&cell, PLANE).unwrap();
    // From G to (-1/2, 0): the mesh holds (1/2, 0), the same point shifted by a lattice vector.
    let path =
        BandPath::explicit(&[("A".into(), [0., 0., 0.]), ("B".into(), [-0.5, 0., 0.])]).unwrap();
    let selected = select(&path, &metric, &mesh.full).unwrap();
    assert!(selected.complete());
    let k: Vec<f64> = selected.kpoints.iter().map(|k| k[0]).collect();
    assert_eq!(k, [0., -0.25, -0.5]);
    assert_eq!(mesh.full[selected.mesh_index[2]][0], 0.5);
    // A path through no mesh point raises.
    let off = BandPath::explicit(&[("A".into(), [0.1, 0.05, 0.]), ("B".into(), [0.2, 0.05, 0.])])
        .unwrap();
    assert!(select(&off, &metric, &mesh.full).is_err());
}

#[test]
fn path_text_is_parsed_with_runs_and_refuses_what_it_cannot_name() {
    let cubic = diagonal(3., 3., 3.);
    let path = BandPath::named(&cubic, ALL, None).unwrap();
    let labels: String = path.vertices.iter().map(|v| v.label.as_str()).collect();
    assert_eq!(labels, "GXMGRXMR");
    assert_eq!(path.jump, [false, false, false, false, false, true, false]);
    let slab = diagonal(2.6, 2.6, 9.);
    for bad in ["GXQ", "G", "GX,", ",GX", "GX,,MG", "gx", "G-X"] {
        assert!(BandPath::named(&slab, PLANE, Some(bad)).is_err(), "{bad}");
    }
    // A label of another lattice is named with the lattice's own points.
    let error = BandPath::named(&slab, PLANE, Some("GXR"))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("square") && error.contains("\"R\""),
        "{error}"
    );
    // An oblique plane has no names; an explicit path still works.
    let oblique = rows([[2., 0., 0.], [0.7, 2.9, 0.], [0., 0., 9.]]);
    let error = BandPath::named(&oblique, PLANE, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("explicit points"), "{error}");
}

#[test]
fn a_jump_adds_no_distance_and_merges_the_labels() {
    let cell = diagonal(3., 3., 3.);
    let ctx = context();
    let mesh = spectrum::mesh(&ctx, [2, 2, 2], true, &[], false).unwrap();
    let path = BandPath::named(&cell, ALL, Some("GX,MR")).unwrap();
    let selected = select(&path, &Metric::new(&cell, ALL).unwrap(), &mesh.full).unwrap();
    assert_eq!(selected.labels, ["G", "X|M", "R"]);
    let unit = PI / 3.;
    assert!((selected.label_distances[1] - unit).abs() < 1e-12);
    // M to R is half a cube edge: the jump added nothing before it.
    assert!((selected.label_distances[2] - 2. * unit).abs() < 1e-12);
    assert!(selected.complete());
}

/// The star of every point of the full mesh by brute force: the orbit under the rotations (k W),
/// time reversal and reciprocal-lattice shifts.
fn orbit(full: &[[f64; 3]], start: usize, rotations: &[[i32; 9]]) -> Vec<usize> {
    let key = |k: [f64; 3]| k.map(|x| ((x - x.floor()) * 1e8).round() as i64 % 100_000_000);
    let lookup: std::collections::HashMap<_, _> =
        full.iter().enumerate().map(|(i, k)| (key(*k), i)).collect();
    let mut seen = vec![start];
    let mut queue = vec![start];
    while let Some(i) = queue.pop() {
        let k = full[i];
        let mut images = vec![k.map(|x| -x)];
        for w in rotations {
            images.push(std::array::from_fn(|a| {
                (0..3).map(|b| k[b] * f64::from(w[3 * b + a])).sum()
            }));
        }
        for image in images {
            let j = *lookup
                .get(&key(image))
                .expect("the group preserves the mesh");
            if !seen.contains(&j) {
                seen.push(j);
                queue.push(j);
            }
        }
    }
    seen.sort_unstable();
    seen
}

fn reduce(
    cell: &Matrix3<f64>,
    fractional: &[[f64; 3]],
    numbers: &[u32],
    size: [usize; 3],
    gamma: bool,
) -> mandacaru_core::periodic::kpath::Zone {
    irreducible_zone(&context(), cell, fractional, numbers, size, gamma, 1e-5).unwrap()
}

fn check_zone(zone: &mandacaru_core::periodic::kpath::Zone) {
    let mesh = &zone.mesh;
    assert!((mesh.weights.iter().sum::<f64>() - 1.).abs() < 1e-14);
    assert_eq!(mesh.mapping.len(), mesh.full.len());
    for (star, weight) in mesh.weights.iter().enumerate() {
        let members: Vec<usize> = (0..mesh.full.len())
            .filter(|&i| mesh.mapping[i] == star)
            .collect();
        assert!((weight - members.len() as f64 / mesh.full.len() as f64).abs() < 1e-14);
        // The star is the orbit of any of its members.
        assert_eq!(orbit(&mesh.full, members[0], &zone.rotations), members);
    }
}

#[test]
fn the_square_lattice_zone_is_gamma_x_and_m_with_weights_a_quarter_a_half_a_quarter() {
    let cell = diagonal(2.6, 2.6, 9.);
    let zone = reduce(&cell, &[[0.; 3]], &[1], [2, 2, 1], true);
    assert_eq!(zone.space_group_operations, 16, "P4/mmm");
    assert_eq!(zone.distinct_rotations, 16);
    assert_eq!(zone.rotations.len(), 16);
    let mut stars: Vec<(f64, [f64; 3])> = zone
        .mesh
        .weights
        .iter()
        .copied()
        .zip(zone.mesh.points.iter().copied())
        .collect();
    stars.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(stars.len(), 3);
    assert_eq!(stars[0].0, 0.25);
    assert_eq!(stars[1].0, 0.25);
    assert_eq!(stars[2].0, 0.5);
    check_zone(&zone);
    // 3 x 3 x 1: the zone center, a star of four on the axes and one of four on the diagonals.
    let zone = reduce(&cell, &[[0.; 3]], &[1], [3, 3, 1], false);
    let mut weights = zone.mesh.weights.clone();
    weights.sort_by(f64::total_cmp);
    assert_eq!(weights.len(), 3);
    assert!((weights[0] - 1. / 9.).abs() < 1e-15 && (weights[2] - 4. / 9.).abs() < 1e-15);
    check_zone(&zone);
}

#[test]
fn the_group_of_a_mesh_is_the_subgroup_of_the_lattice_group_that_maps_it_onto_itself() {
    // A 2 x 1 mesh of the square lattice is not carried by the fourfold axis.
    let cell = diagonal(2.6, 2.6, 9.);
    let zone = reduce(&cell, &[[0.; 3]], &[1], [2, 1, 1], true);
    assert_eq!(zone.distinct_rotations, 16);
    assert!(zone.rotations.len() < 16 && zone.rotations.len() >= 4);
    assert_eq!(zone.mesh.points.len(), 2);
    check_zone(&zone);
    // A second species lowers the symmetry of the cell, and the zone grows with it.
    let two = diagonal(2.6, 2.6, 9.);
    let lowered = reduce(&two, &[[0.; 3], [0.5, 0., 0.]], &[1, 2], [2, 2, 1], true);
    assert!(lowered.distinct_rotations < 16);
    assert!(lowered.mesh.points.len() >= 3);
    check_zone(&lowered);
}

#[test]
fn the_hexagonal_and_cubic_zones_have_orbits_that_match_their_weights() {
    let a = 3.0;
    let gamma = 120_f64.to_radians();
    let hex = rows([
        [a, 0., 0.],
        [a * gamma.cos(), a * gamma.sin(), 0.],
        [0., 0., 10.],
    ]);
    let zone = reduce(&hex, &[[0.; 3]], &[6], [3, 3, 1], false);
    assert_eq!(zone.rotations.len(), 24, "P6/mmm");
    // Each star is the orbit of its representative, found by brute force.
    check_zone(&zone);
    let zone = reduce(&diagonal(3., 3., 3.), &[[0.; 3]], &[29], [4, 4, 4], true);
    assert_eq!(zone.rotations.len(), 48, "Pm-3m");
    check_zone(&zone);
    assert_eq!(
        zone.mesh.points.len(),
        10,
        "irreducible points of a 4 x 4 x 4 cubic mesh"
    );
}

#[test]
fn an_orthorhombic_cell_follows_the_increasing_length_order_of_its_convention() {
    // X is the face center along the shortest axis: b1 / 2 when a < b < c.
    let sorted = convention::convention(&diagonal(3.0, 3.9, 5.1), [true; 3]).unwrap();
    assert_eq!(sorted.point("X").unwrap(), [0.5, 0., 0.]);
    assert_eq!(sorted.point("Z").unwrap(), [0., 0., 0.5]);
    // Any other order would put X along a longer axis: refused, as for an unordered tetragonal cell.
    for (a, b, c) in [(5.1, 3.9, 3.0), (3.9, 3.0, 5.1), (3.0, 5.1, 3.9)] {
        let message = convention::convention(&diagonal(a, b, c), [true; 3]).unwrap_err().to_string();
        assert!(message.contains("increasing order"), "{message}");
    }
}
