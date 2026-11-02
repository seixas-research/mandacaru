//! Time reversal in the spinor hybrid (SH-1a, gate g2 and unknowns U-1, U-3): the operator of the
//! complex-harmonic basis is `U = i sigma_y (x) Pi_AO` on the basis functions, `Pi_p` on the
//! projectors and a signed permutation on the compensation channels, NOT `i sigma_y (x) 1`.
//! Dataset level first (the Bloch sums, projections, fixed and spin-orbit blocks of the
//! snapshot), then the exchange: `K(-k) = U K(k)^* U^T` for a time-reversal-symmetric ensemble on
//! the complex-k meshes 3x1x1 and 4x1x1. Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_spinor_support;
mod setup_support;

use mandacaru_core::periodic::screened_exchange::{exchange_operators_with, Screening};
use num_complex::Complex64 as C64;
use periodic_hybrid_spinor_support::*;

const FIXTURES: [&str; 2] = ["si_so_tr3.json", "si_so_tr4.json"];
const GATE: f64 = 1e-11;
/// The compensation charges break the covariance on the Nyquist planes of the grid alone (see
/// `the_covariance_breaks_only_on_the_nyquist_planes_of_the_shape_transforms`): measured 5.1e-8
/// (complex pair) and 2.0e-8 (Gamma) of the largest entry of K on this grid, gate 20x the largest.
const COMPENSATED_GATE: f64 = 1e-6;

fn mirror(pi: &Matrix, a: &Matrix) -> Matrix {
    pi * a.conjugate() * pi.transpose()
}

/// The relative miss of each dataset array against its time-reversal image, the worst over the
/// pairs `(k, -k)` that are not the same point, plus the same with `Pi` replaced by the identity.
#[derive(Default, Debug)]
struct Misses {
    psi: f64,
    overlap: f64,
    fixed: f64,
    projections: f64,
    spin_orbit: f64,
    moments: f64,
    psi_identity: f64,
    projections_identity: f64,
    spin_orbit_identity: f64,
}

fn measure(name: &str) -> Misses {
    let ctx = context();
    let v = read(name);
    let library = library(&v, name);
    let (pi, pp) = (pi_ao(&v), pi_projector(&v, &library));
    let c = hybrid(&ctx, name, &|_| {});
    let (m, n, p) = (c.m, c.grid.index.points(), c.p);
    let u = time_reversal(&pi);
    let identity = Matrix::identity(m, m);
    let u_plain = time_reversal(&identity);
    let mut worst = Misses::default();
    let mut paired = 0;
    for i in 0..c.points.len() {
        let j = partner(&c, i);
        if j == i {
            continue;
        }
        paired += 1;
        let (a, b) = (&c.points[i], &c.points[j]);
        let samples = |k: usize| Matrix::from_row_slice(m, n, &c.points[k].psi.read(&ctx).unwrap());
        let (psi_i, psi_j) = (samples(i), samples(j));
        worst.psi = worst.psi.max(relative(&psi_j, &(&pi * psi_i.conjugate())));
        worst.psi_identity = worst.psi_identity.max(relative(&psi_j, &psi_i.conjugate()));
        worst.overlap = worst
            .overlap
            .max(relative(&b.overlap, &mirror(&pi, &a.overlap)));
        worst.fixed = worst.fixed.max(relative(&b.fixed, &mirror(&pi, &a.fixed)));
        let image = &pi * a.projections.conjugate() * &pp;
        worst.projections = worst.projections.max(relative(&b.projections, &image));
        worst.projections_identity = worst
            .projections_identity
            .max(relative(&b.projections, &a.projections.conjugate()));
        let (so_a, so_b) = (
            a.spin_orbit.as_ref().unwrap(),
            b.spin_orbit.as_ref().unwrap(),
        );
        worst.spin_orbit = worst
            .spin_orbit
            .max(relative(so_b, &(&u * so_a.conjugate() * u.transpose())));
        worst.spin_orbit_identity = worst.spin_orbit_identity.max(relative(
            so_b,
            &(&u_plain * so_a.conjugate() * u_plain.transpose()),
        ));
        assert!(
            relative(&b.overlap, &a.overlap) > 1e-3 && relative(so_b, so_a) > 1e-3,
            "{name}: the pair is not a complex-k pair, the relations would hold trivially"
        );
    }
    assert!(paired > 0, "{name} has no complex-k pair");
    // The compensation channels: conj(B_(atom,L,M)) = (-1)^M Pi_p B_(atom,L,-M) Pi_p.
    let labels: Vec<(usize, usize, i32)> = v["channel_labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t[0].as_u64().unwrap() as usize,
                t[1].as_u64().unwrap() as usize,
                t[2].as_i64().unwrap() as i32,
            )
        })
        .collect();
    let blocks = |a: usize| Matrix::from_fn(p, p, |i, j| c.blocks.as_slice()[(a * p + i) * p + j]);
    let mut moments = 0f64;
    let mut populated = 0;
    for (a, &(atom, l, order)) in labels.iter().enumerate() {
        let b = labels
            .iter()
            .position(|&t| t == (atom, l, -order))
            .expect("no partner channel");
        let sign = if order % 2 == 0 { 1. } else { -1. };
        let image = &pp * blocks(b) * &pp * C64::new(sign, 0.);
        let reference = blocks(a).conjugate();
        if reference.norm() > 1e-9 {
            populated += 1;
        }
        moments = moments.max((&reference - &image).norm() / reference.norm().max(1e-300));
    }
    assert!(populated > 0);
    worst.moments = moments;
    worst
}

#[test]
fn the_dataset_obeys_time_reversal_with_the_signed_permutations() {
    for name in FIXTURES {
        let w = measure(name);
        println!("{name}: {w:?}");
        for (what, miss) in [
            ("Bloch sums", w.psi),
            ("overlap", w.overlap),
            ("fixed Hamiltonian", w.fixed),
            ("projections C(-k) = Pi_AO conj(C(k)) Pi_p", w.projections),
            ("spin-orbit block", w.spin_orbit),
            ("moment blocks", w.moments),
        ] {
            assert!(
                miss < GATE,
                "{name}: {what} misses its time-reversal image by {miss:e}"
            );
        }
        // The identity in place of Pi misses by far more than the gate: Pi is not optional.
        for (what, miss) in [
            ("Bloch sums", w.psi_identity),
            ("projections", w.projections_identity),
            ("spin-orbit block", w.spin_orbit_identity),
        ] {
            assert!(
                miss > 100. * GATE,
                "{name}: the control for {what} does not bite ({miss:e})"
            );
        }
    }
}

#[test]
fn the_hybrid_exchange_is_time_reversal_covariant_at_complex_k() {
    for name in FIXTURES {
        let ctx = context();
        let v = read(name);
        let pi = pi_ao(&v);
        let u = time_reversal(&pi);
        let c = hybrid(&ctx, name, &|_| {});
        let d = kramers_densities(&c, &u, 8);
        for i in 0..c.points.len() {
            let j = partner(&c, i);
            assert!(relative(&d[j], &(&u * d[i].conjugate() * u.transpose())) < 1e-13);
        }
        let screening = Screening {
            omega: 0.11,
            fraction: 0.25,
        };
        for exxdiv in [NONE, GB] {
            let k = c.hybrid_exchange(&d, exxdiv).unwrap();
            let wrong_u = [
                (
                    "i sigma_y (x) 1 instead of i sigma_y (x) Pi_AO",
                    time_reversal(&Matrix::identity(c.m, c.m)),
                ),
                ("sigma_x in place of i sigma_y", {
                    let mut x = Matrix::zeros(2 * c.m, 2 * c.m);
                    for s in 0..2 {
                        x.view_mut((s * c.m, (1 - s) * c.m), (c.m, c.m))
                            .copy_from(&pi);
                    }
                    x
                }),
            ];
            let mut worst = 0f64;
            let mut moved = 0f64;
            let mut control = vec![0f64; wrong_u.len() + 1];
            for i in 0..c.points.len() {
                let j = partner(&c, i);
                let want = |u: &Matrix| u * k.operators[i].conjugate() * u.transpose();
                let miss = relative(&k.operators[j], &want(&u));
                println!("{name} {exxdiv:?}: K(k{j}) vs U K(k{i})^* U^T: {miss:.2e}");
                worst = worst.max(miss);
                if i != j {
                    moved = moved.max(relative(&k.operators[j], &k.operators[i]));
                    for (slot, (_, wrong)) in wrong_u.iter().enumerate() {
                        control[slot] = control[slot].max(relative(&k.operators[j], &want(wrong)));
                    }
                    control[wrong_u.len()] = control[wrong_u.len()].max(relative(
                        &k.operators[j],
                        &(&u * &k.operators[i] * u.transpose()),
                    ));
                }
            }
            assert!(
                k.operators.iter().all(|o| o.norm() > 1e-3),
                "the exchange is vacuous"
            );
            assert!(
                moved > 1e-3,
                "K(-k) = K(k): the check cannot see a conjugation error"
            );
            assert!(
                worst < COMPENSATED_GATE,
                "{name} {exxdiv:?}: time-reversal covariance misses by {worst:e}"
            );
            for (slot, miss) in control.iter().enumerate() {
                let what = wrong_u.get(slot).map_or("no conjugation", |t| t.0);
                println!("  control ({what}): {miss:.2e}");
                assert!(
                    *miss > 100. * COMPENSATED_GATE,
                    "control '{what}' does not bite ({miss:e})"
                );
            }
            // The grid pair alone is covariant to round-off: the whole 5e-8 is the compensation.
            let plain = exchange_operators_with(&c, &d, screening, exxdiv, None)
                .unwrap()
                .operators;
            let pair = (0..c.points.len())
                .map(|i| {
                    relative(
                        &plain[partner(&c, i)],
                        &(&u * plain[i].conjugate() * u.transpose()),
                    )
                })
                .fold(0., f64::max);
            println!("  grid pair alone: {pair:.2e}");
            assert!(
                pair < GATE,
                "the grid pair misses its time-reversal image by {pair:e}"
            );
        }
    }
}

#[test]
fn the_covariance_breaks_only_on_the_nyquist_planes_of_the_shape_transforms() {
    use mandacaru_core::periodic::screened_augmentation::Augmentation;
    let name = "si_so_tr3.json";
    let ctx = context();
    let c = hybrid(&ctx, name, &|_| {});
    let rig = periodic_hybrid_spinor_support::rig::Rig::new(name, &c);
    let sites = rig.geometry.sites();
    let aug = Augmentation::new(&c, sites, &rig.shapes).unwrap();
    let n = c.grid.index.points();
    let shape = c.grid.index.shape();
    let labels = &rig.geometry.channels;
    // conj(T_(A,L,M)(Q)) = (-1)^M T_(A,L,-M)(-Q), the identity that makes K time-reversal covariant.
    for kd in [[0.; 3], [1. / 3., 0., 0.]] {
        let plus = aug.pair_shapes(&c, kd).unwrap();
        let minus = aug.pair_shapes(&c, [-kd[0], -kd[1], -kd[2]]).unwrap();
        let (mut inner, mut edge, mut scale) = (0f64, 0f64, 0f64);
        for (a, &(atom, l, order)) in labels.iter().enumerate() {
            let b = labels.iter().position(|&t| t == (atom, l, -order)).unwrap();
            let sign = if order % 2 == 0 { 1. } else { -1. };
            for r in 0..n {
                let idx = [
                    r / (shape[1] * shape[2]),
                    r / shape[2] % shape[1],
                    r % shape[2],
                ];
                let mirrored = (0..3).fold(0, |acc, ax| {
                    acc * shape[ax] + (shape[ax] - idx[ax]) % shape[ax]
                });
                let nyquist = (0..3).any(|ax| 2 * idx[ax] == shape[ax]);
                let miss = (plus.as_slice()[a * n + r].conj()
                    - minus.as_slice()[b * n + mirrored] * sign)
                    .norm();
                scale = scale.max(plus.as_slice()[a * n + r].norm());
                if nyquist {
                    edge = edge.max(miss)
                } else {
                    inner = inner.max(miss)
                }
            }
        }
        println!("k_diff {kd:?}: mirror miss off the Nyquist planes {:.2e}, on them {:.2e} (of the largest transform)", inner / scale, edge / scale);
        // Exact off the planes, a convention (one of the two aliased vectors) on them.
        assert!(inner / scale < 1e-14, "{:e}", inner / scale);
        assert!(
            edge / scale > 1e-3,
            "the Nyquist planes carry the asymmetry ({:e})",
            edge / scale
        );
    }
}
