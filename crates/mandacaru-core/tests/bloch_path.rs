//! The spectral function along a band path: the tight-binding square lattice, whose
//! `A(k, omega)` is one pole at `eps(k) = -2 t (cos kx + cos ky)`, and the symmetry of
//! `A(k, omega)` of an interacting state under the point group and time reversal.
//!
//! The lattice is a 3 x 3 torus of sites (a Born-von Karman supercell with `kpts = (3, 3, 1)`),
//! the Bloch operators are the plane waves of its mesh points, and the path, the zone and the
//! gather are the library's own.
use mandacaru_core::{
    math::linalg::eigen,
    molecular::{periodic::BlochSpectral, transform::Model},
    periodic::{
        kpath::{irreducible_zone, select, BandPath, Metric},
        spectrum,
    },
    quantum::{
        spectral::{lehmann, lorentzian},
        Basis, Encoding, C64,
    },
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

const L: usize = 3;
const PLANE: [bool; 3] = [true, true, false];

fn torus(ctx: &Context, t: f64, u: f64) -> Model {
    let m = L * L;
    let site = |x: usize, y: usize| (x % L) * L + y % L;
    let mut one = ctx.allocate::<C64>(m * m).unwrap();
    for x in 0..L {
        for y in 0..L {
            for (dx, dy) in [(1, 0), (L - 1, 0), (0, 1), (0, L - 1)] {
                one.as_mut_slice()[site(x, y) * m + site(x + dx, y + dy)] += C64::new(-t, 0.);
            }
        }
    }
    let mut two = ctx.allocate::<C64>(m * m * m * m).unwrap();
    for p in 0..m {
        two.as_mut_slice()[((p * m + p) * m + p) * m + p] = C64::new(u, 0.);
    }
    Model::new(ctx, m, [1, 1], one, two, 0., false).unwrap()
}

fn ground(ctx: &Context, model: &Model) -> Vec<(u64, C64)> {
    let encoding = Encoding::new("jordan_wigner", 2 * model.m, model.particles).unwrap();
    let basis = Basis::new(ctx, &encoding, true).unwrap();
    let h = model.operator(ctx, "jordan_wigner").unwrap();
    let d = basis.len();
    let dense = h.dense(ctx, &basis).unwrap();
    let (values, vectors) = eigen(DMatrix::from_row_slice(d, d, dense.as_slice())).unwrap();
    assert!(
        values[1] - values[0] > 1e-6,
        "the test needs a nondegenerate ground state"
    );
    basis
        .indices
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &ket)| (ket, vectors[(i, 0)]))
        .collect()
}

fn plane_wave(k: [f64; 3]) -> Vec<C64> {
    (0..L * L)
        .map(|p| {
            let (x, y) = ((p / L) as f64, (p % L) as f64);
            C64::from_polar(1. / (L as f64), 2. * PI * (k[0] * x + k[1] * y))
        })
        .collect()
}

fn band(t: f64, k: [f64; 3]) -> f64 {
    -2. * t * ((2. * PI * k[0]).cos() + (2. * PI * k[1]).cos())
}

/// `A(k, omega)` of the state at `points`, one row per point.
fn spectral(
    ctx: &Context,
    model: &Model,
    state: &[(u64, C64)],
    points: &[[f64; 3]],
) -> BlochSpectral {
    let operators: Vec<_> = points.iter().map(|&k| plane_wave(k)).collect();
    let parts = lehmann(ctx, model, model.particles, state, &operators, 400).unwrap();
    BlochSpectral {
        kpoints: points.to_vec(),
        n_primitive: 1,
        parts,
    }
}

fn curve(s: &BlochSpectral, row: usize, axis: &[f64], eta: f64) -> Vec<f64> {
    let poles = [
        &s.parts.addition[row][0],
        &s.parts.removal[row][0],
        &s.parts.addition[row][1],
        &s.parts.removal[row][1],
    ];
    lorentzian(&poles, axis, eta).unwrap()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}

#[test]
fn the_noninteracting_square_lattice_has_one_pole_at_its_dispersion_along_the_path() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let t = 0.37;
    let model = torus(&ctx, t, 0.);
    let state = ground(&ctx, &model);
    let cell = Matrix3::from_diagonal(&Vector3::new(2.6, 2.6, 9.));
    let mesh = spectrum::mesh(&ctx, [L, L, 1], false, &[], false).unwrap();
    let path = BandPath::named(&cell, PLANE, Some("GXMG")).unwrap();
    let selected = select(&path, &Metric::new(&cell, PLANE).unwrap(), &mesh.full).unwrap();
    assert_eq!(selected.missing, ["X", "M"], "a 3 x 3 mesh has no X or M");
    let points: Vec<[f64; 3]> = selected.kpoints.clone();
    assert_eq!(points.len(), 4);
    let result = spectral(&ctx, &model, &state, &points);
    for (row, k) in points.iter().enumerate() {
        let occupied = k[0].abs() < 1e-9 && k[1].abs() < 1e-9;
        // part = [addition, removal]: the occupied state is removed, an empty one is added.
        let (kept, empty) = if occupied { (1, 0) } else { (0, 1) };
        // Spin up and down alike: every pole of a determinant is one delta of weight 1.
        for spin in 0..2 {
            let part = [
                &result.parts.addition[row][spin],
                &result.parts.removal[row][spin],
            ];
            assert!(
                (part[kept].weight() - 1.).abs() < 1e-12,
                "{k:?}: weight {}",
                part[kept].weight()
            );
            assert!(part[empty].weight() < 1e-20);
            let at = part[kept].weights.iter().position(|&w| w > 0.5).unwrap();
            let pole = part[kept].energies[at];
            assert!(
                (pole - band(t, *k)).abs() < 1e-12,
                "{k:?}: {pole} vs {}",
                band(t, *k)
            );
        }
    }
    // The tight-binding values: -4t at the zone center, -t at (1/3, 0), 2t at (1/3, 1/3).
    assert!((band(t, points[0]) + 4. * t).abs() < 1e-12);
    assert!((band(t, points[1]) + t).abs() < 1e-12);
    assert!((band(t, points[2]) - 2. * t).abs() < 1e-12);
}

#[test]
fn an_interacting_state_has_the_point_group_and_time_reversal_symmetry_of_its_lattice() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let model = torus(&ctx, 1., 3.);
    let state = ground(&ctx, &model);
    let cell = Matrix3::from_diagonal(&Vector3::new(2.6, 2.6, 9.));
    let zone = irreducible_zone(&ctx, &cell, &[[0.; 3]], &[1], [L, L, 1], false, 1e-5).unwrap();
    let full = zone.mesh.full.clone();
    let all = spectral(&ctx, &model, &state, &full);
    let axis: Vec<f64> = (0..600).map(|i| -14. + 0.04 * i as f64).collect();
    let eta = 0.15;
    // Every point of a star has the spectral function of the star's representative.
    let reps = spectral(&ctx, &model, &state, &zone.mesh.points);
    let mut worst = 0f64;
    for (i, k) in full.iter().enumerate() {
        let star = zone.star_of(*k).unwrap();
        worst = worst.max(distance(
            &curve(&all, i, &axis, eta),
            &curve(&reps, star, &axis, eta),
        ));
    }
    assert!(worst < 1e-9, "point-group invariance: {worst:e}");
    // Time reversal alone: A(k) = A(-k), including the pairs the point group would also join.
    let mut worst = 0f64;
    for (i, k) in full.iter().enumerate() {
        let minus = full
            .iter()
            .position(|q| (0..3).all(|a| ((q[a] + k[a]) - (q[a] + k[a]).round()).abs() < 1e-9))
            .unwrap();
        worst = worst.max(distance(
            &curve(&all, i, &axis, eta),
            &curve(&all, minus, &axis, eta),
        ));
    }
    assert!(worst < 1e-9, "time reversal: {worst:e}");
    // The interaction is on: A at an axis point and at a diagonal point differ.
    let axis_point = full
        .iter()
        .position(|k| k[0] > 0.3 && k[1].abs() < 1e-9)
        .unwrap();
    let diagonal_point = full.iter().position(|k| k[0] > 0.3 && k[1] > 0.3).unwrap();
    assert!(
        distance(
            &curve(&all, axis_point, &axis, eta),
            &curve(&all, diagonal_point, &axis, eta)
        ) > 1e-3
    );
    // The sum rule holds at every point.
    for row in 0..full.len() {
        for spin in 0..2 {
            let total =
                all.parts.addition[row][spin].weight() + all.parts.removal[row][spin].weight();
            assert!((total - 1.).abs() < 1e-12, "sum rule {total}");
        }
    }
}

#[test]
fn a_gathered_spectral_function_repeats_the_rows_it_is_given() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let model = torus(&ctx, 1., 2.);
    let state = ground(&ctx, &model);
    let points = [[0., 0., 0.], [1. / 3., 0., 0.]];
    let computed = spectral(&ctx, &model, &state, &points);
    // A path visits the zone center at both ends: one Lehmann evaluation, two rows.
    let along = computed
        .gather(&[0, 1, 0], vec![[0.; 3], [1. / 3., 0., 0.], [1., 0., 0.]])
        .unwrap();
    assert_eq!(along.kpoints.len(), 3);
    let axis: Vec<f64> = (0..200).map(|i| -10. + 0.1 * i as f64).collect();
    assert_eq!(
        curve(&along, 2, &axis, 0.1),
        curve(&computed, 0, &axis, 0.1)
    );
    assert_eq!(
        curve(&along, 1, &axis, 0.1),
        curve(&computed, 1, &axis, 0.1)
    );
    assert!(computed.gather(&[2], vec![[0.; 3]]).is_err());
    assert!(computed.gather(&[0], vec![]).is_err());
}
