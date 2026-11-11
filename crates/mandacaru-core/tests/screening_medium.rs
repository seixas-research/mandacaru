//! The static polarizability of a mesh of Bloch states (`periodic::screening::medium`) against the
//! Adler-Wiser sum of the empty-lattice (free-electron) crystal, where every transition density is
//! a Kronecker delta and the sum can be written without the code under test.
//!
//! States `psi_{k+G1}(r) = exp(i (k + G1).r) / sqrt(Omega)` of every `G1` in a cube, on a grid whose
//! first node is not the origin, a Gamma-centered mesh, a mesh vector `q` pairing `k` with the
//! reduced `k + q`. The reference loops over band pairs by hand:
//!
//! ```text
//! chi_GG(q) = 1/(N Omega) sum_{k, G1, G2} g (f_a - f_b) / (e_a - e_b) (1 - p_a p_b)  [G2 = G1 + G + G0]
//! ```
//!
//! with `G0` the reciprocal vector that reduces `k + q`. Also the compressibility sum rule
//! `chi_00(q = 0) = -dn/dmu` and the constrained limits (all weight in the target subspace gives
//! `chi = 0`; no weight gives the full RPA).
use mandacaru_core::{
    math::{storage::ComplexSamples, SpatialGrid},
    periodic::{
        screening::{
            bloch::Bloch,
            medium::{within, Medium},
        },
        spectrum::Smearing,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

const SIDE: f64 = 8.;
const SHAPE: usize = 16;
const MESH: usize = 3;
const FERMI: f64 = 0.4;
const WIDTH: f64 = 0.05;

fn fermi(e: f64) -> f64 {
    1. / (1. + ((e - FERMI) / WIDTH).exp())
}
fn slope(e: f64) -> f64 {
    let f = fermi(e);
    -f * (1. - f) / WIDTH
}

struct Set {
    fractional: [f64; 3],
    /// Integer `G1` of each band.
    g1: Vec<[i64; 3]>,
    energies: Vec<f64>,
    target: Vec<f64>,
    samples: ComplexSamples,
}

fn reciprocal() -> Matrix3<f64> {
    Matrix3::identity() * (2. * PI / SIDE)
}

fn grid() -> SpatialGrid {
    let h = SIDE / SHAPE as f64;
    SpatialGrid::new(
        [SHAPE; 3],
        [0.37, -0.21, 0.55],
        [h, 0., 0., 0., h, 0., 0., 0., h],
        true,
    )
    .unwrap()
}

fn set(
    ctx: &Context,
    grid: &SpatialGrid,
    fractional: [f64; 3],
    target: &dyn Fn(f64) -> f64,
) -> Set {
    let k = reciprocal() * Vector3::from(fractional);
    let mut g1 = vec![];
    for x in -1i64..=1 {
        for y in -1i64..=1 {
            for z in -1i64..=1 {
                g1.push([x, y, z]);
            }
        }
    }
    let omega = SIDE.powi(3);
    let n = grid.index.points();
    let mut samples = Vec::with_capacity(g1.len() * n);
    let mut energies = vec![];
    let mut weights = vec![];
    for g in &g1 {
        let p = k + reciprocal() * Vector3::new(g[0] as f64, g[1] as f64, g[2] as f64);
        for x in 0..n {
            let r = grid.point(x);
            samples.push(C64::from_polar(
                omega.powf(-0.5),
                p[0] * r[0] + p[1] * r[1] + p[2] * r[2],
            ));
        }
        energies.push(0.5 * p.norm_squared());
        weights.push(target(0.5 * p.norm_squared()));
    }
    Set {
        fractional,
        g1,
        energies,
        target: weights,
        samples: ComplexSamples::Float64(ctx.copy_slice(&samples).unwrap()),
    }
}

fn bloch<'a>(s: &'a Set, points: usize) -> Bloch<'a> {
    Bloch {
        fractional: s.fractional,
        energies: &s.energies,
        target: s.target.clone(),
        samples: &s.samples,
        projections: None,
        points,
    }
}

fn mesh_points() -> Vec<[usize; 3]> {
    (0..MESH * MESH * MESH)
        .map(|k| [k / (MESH * MESH), (k / MESH) % MESH, k % MESH])
        .collect()
}

fn medium<'a>(ctx: &'a Context, grid: &'a SpatialGrid, constrained: bool) -> Medium<'a> {
    Medium {
        ctx,
        grid,
        volume: SIDE.powi(3),
        fermi: FERMI,
        width: WIDTH,
        degeneracy: 2.,
        smearing: Smearing::FermiDirac,
        constrained,
        paw: None,
        workers: None,
    }
}

/// chi(q) of the mesh pairs `(k, k + q)` from the library, and the Adler-Wiser reference.
fn compare(
    q_cell: [usize; 3],
    target: &dyn Fn(f64) -> f64,
    constrained: bool,
) -> (f64, f64, usize) {
    let ctx = Context::new(Some(1), 1 << 29).unwrap();
    let grid = grid();
    let lattice = Matrix3::identity() * SIDE;
    let q = reciprocal()
        * Vector3::new(
            q_cell[0] as f64 / MESH as f64,
            q_cell[1] as f64 / MESH as f64,
            q_cell[2] as f64 / MESH as f64,
        );
    let g = within(&lattice, q.into(), 2.4);
    let points = mesh_points();
    // the states at every mesh point, indexed like `points`
    let sets: Vec<Set> = points
        .iter()
        .map(|p| {
            set(
                &ctx,
                &grid,
                [
                    p[0] as f64 / MESH as f64,
                    p[1] as f64 / MESH as f64,
                    p[2] as f64 / MESH as f64,
                ],
                target,
            )
        })
        .collect();
    let index = |p: [usize; 3]| (p[0] * MESH + p[1]) * MESH + p[2];
    let n = grid.index.points();
    let views: Vec<Bloch<'_>> = sets.iter().map(|s| bloch(s, n)).collect();
    let pairs: Vec<(&Bloch<'_>, &Bloch<'_>)> = points
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let partner = [
                (p[0] + q_cell[0]) % MESH,
                (p[1] + q_cell[1]) % MESH,
                (p[2] + q_cell[2]) % MESH,
            ];
            (&views[k], &views[index(partner)])
        })
        .collect();
    let (chi, vectors) = medium(&ctx, &grid, constrained)
        .polarizability(q.into(), &g, &pairs, points.len())
        .unwrap();
    assert_eq!(vectors.len(), g.len());
    // the reference
    let omega = SIDE.powi(3);
    let mut miss = 0f64;
    let mut scale = 0f64;
    for (i, gi) in g.iter().enumerate() {
        let mut expected = 0.;
        for (k, p) in points.iter().enumerate() {
            let raw = [
                p[0] as i64 + q_cell[0] as i64,
                p[1] as i64 + q_cell[1] as i64,
                p[2] as i64 + q_cell[2] as i64,
            ];
            let reduced: [usize; 3] =
                std::array::from_fn(|a| raw[a].rem_euclid(MESH as i64) as usize);
            let g0: [i64; 3] =
                std::array::from_fn(|a| (raw[a] - reduced[a] as i64).div_euclid(MESH as i64));
            let (a_set, b_set) = (&sets[k], &sets[index(reduced)]);
            for (a, g1) in a_set.g1.iter().enumerate() {
                let want: [i64; 3] = std::array::from_fn(|c| g1[c] + gi[c] + g0[c]);
                let Some(b) = b_set.g1.iter().position(|g2| *g2 == want) else {
                    continue;
                };
                let (ea, eb) = (a_set.energies[a], b_set.energies[b]);
                let weight = if (ea - eb).abs() < 1e-9 {
                    2. * slope(0.5 * (ea + eb))
                } else {
                    2. * (fermi(ea) - fermi(eb)) / (ea - eb)
                };
                let factor = if constrained {
                    1. - a_set.target[a] * b_set.target[b]
                } else {
                    1.
                };
                expected += weight * factor;
            }
        }
        expected /= points.len() as f64 * omega;
        miss = miss.max((chi[(i, i)] - C64::new(expected, 0.)).norm());
        scale = scale.max(expected.abs());
        // chi is diagonal for plane waves
        for j in 0..g.len() {
            if i != j {
                assert!(
                    chi[(i, j)].norm() < 1e-13,
                    "off-diagonal chi {}",
                    chi[(i, j)].norm()
                );
            }
        }
    }
    (miss, scale, g.len())
}

#[test]
fn the_free_electron_polarizability_equals_the_adler_wiser_sum_at_every_mesh_vector() {
    for q in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 2]] {
        let (miss, scale, count) = compare(q, &|_| 0., false);
        println!("q = {q:?}: {count} vectors, largest |chi| {scale:.6e}, miss {miss:.2e}");
        assert!(
            scale > 1e-5,
            "the reference chi vanishes: the test checks nothing"
        );
        assert!(
            miss < 1e-12 * scale.max(1.),
            "q = {q:?}: miss {miss:e} of {scale:e}"
        );
    }
}

#[test]
fn constrained_rpa_removes_the_target_transitions_and_all_weight_in_the_target_gives_zero() {
    // an arbitrary smooth weight per energy: chi carries (1 - p_a p_b)
    let weight = |e: f64| 0.5 * (1. + (4. * e).sin());
    let (miss, scale, _) = compare([1, 0, 0], &weight, true);
    assert!(
        scale > 1e-5 && miss < 1e-12 * scale.max(1.),
        "miss {miss:e} of {scale:e}"
    );
    // the unconstrained sum of the same states is different
    let (miss_rpa, _, _) = compare([1, 0, 0], &weight, false);
    assert!(miss_rpa < 1e-12);
    // every state in the target subspace: nothing screens
    let ctx = Context::new(Some(1), 1 << 29).unwrap();
    let grid = grid();
    let sets: Vec<Set> = mesh_points()
        .iter()
        .map(|p| {
            set(
                &ctx,
                &grid,
                [
                    p[0] as f64 / MESH as f64,
                    p[1] as f64 / MESH as f64,
                    p[2] as f64 / MESH as f64,
                ],
                &|_| 1.,
            )
        })
        .collect();
    let n = grid.index.points();
    let views: Vec<Bloch<'_>> = sets.iter().map(|s| bloch(s, n)).collect();
    let pairs: Vec<_> = views.iter().map(|v| (v, v)).collect();
    let q = [0.; 3];
    let g = within(&(Matrix3::identity() * SIDE), q, 2.4);
    let (chi, _) = medium(&ctx, &grid, true)
        .polarizability(q, &g, &pairs, views.len())
        .unwrap();
    assert!(chi.norm() == 0., "chi = {}", chi.norm());
}

#[test]
fn the_static_response_at_q_zero_is_minus_the_compressibility() {
    // chi_00(0) = -dn/dmu = (g / N Omega) sum_k,G f'(e) with f' < 0, from the analytic slope
    let (miss, scale, _) = compare([0, 0, 0], &|_| 0., false);
    assert!(miss < 1e-12 * scale);
    let ctx = Context::new(Some(1), 1 << 29).unwrap();
    let grid = grid();
    let sets: Vec<Set> = mesh_points()
        .iter()
        .map(|p| {
            set(
                &ctx,
                &grid,
                [
                    p[0] as f64 / MESH as f64,
                    p[1] as f64 / MESH as f64,
                    p[2] as f64 / MESH as f64,
                ],
                &|_| 0.,
            )
        })
        .collect();
    let n = grid.index.points();
    let views: Vec<Bloch<'_>> = sets.iter().map(|s| bloch(s, n)).collect();
    let pairs: Vec<_> = views.iter().map(|v| (v, v)).collect();
    let g = [[0, 0, 0]];
    let (chi, _) = medium(&ctx, &grid, false)
        .polarizability([0.; 3], &g, &pairs, views.len())
        .unwrap();
    let compressibility: f64 = sets
        .iter()
        .flat_map(|s| s.energies.iter())
        .map(|e| -slope(*e))
        .sum::<f64>()
        * 2.
        / (sets.len() as f64 * SIDE.powi(3));
    println!(
        "chi_00(0) = {:.8e}, -dn/dmu = {:.8e}",
        chi[(0, 0)].re,
        -compressibility
    );
    assert!(compressibility > 1e-4);
    assert!((chi[(0, 0)].re + compressibility).abs() < 1e-12 * compressibility);
    let _: DMatrix<C64> = chi;
}

/// `chi(q)` of the free-electron mesh at `q_cell` on a context of `threads` workers.
fn chi_on(threads: usize, q_cell: [usize; 3]) -> DMatrix<C64> {
    let ctx = Context::new(Some(threads), 1 << 29).unwrap();
    let grid = grid();
    let q = reciprocal()
        * Vector3::new(
            q_cell[0] as f64 / MESH as f64,
            q_cell[1] as f64 / MESH as f64,
            q_cell[2] as f64 / MESH as f64,
        );
    let g = within(&(Matrix3::identity() * SIDE), q.into(), 2.4);
    let points = mesh_points();
    let weight = |e: f64| 0.5 * (1. + (4. * e).sin());
    let sets: Vec<Set> = points
        .iter()
        .map(|p| {
            set(
                &ctx,
                &grid,
                [
                    p[0] as f64 / MESH as f64,
                    p[1] as f64 / MESH as f64,
                    p[2] as f64 / MESH as f64,
                ],
                &weight,
            )
        })
        .collect();
    let n = grid.index.points();
    let views: Vec<Bloch<'_>> = sets.iter().map(|s| bloch(s, n)).collect();
    let pairs: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let at = ((p[0] + q_cell[0]) % MESH * MESH + (p[1] + q_cell[1]) % MESH) * MESH
                + (p[2] + q_cell[2]) % MESH;
            (&views[k], &views[at])
        })
        .collect();
    medium(&ctx, &grid, true)
        .polarizability(q.into(), &g, &pairs, points.len())
        .unwrap()
        .0
}

#[test]
fn the_polarizability_is_bit_identical_for_one_two_and_four_workers() {
    let one = chi_on(1, [1, 1, 0]);
    assert!(one.norm() > 1e-3);
    for threads in [2, 4] {
        let many = chi_on(threads, [1, 1, 0]);
        let same = one
            .iter()
            .zip(many.iter())
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
        assert!(same, "{threads} workers changed chi");
    }
}
