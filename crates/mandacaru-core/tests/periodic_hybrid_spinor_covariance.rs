//! SU(2) covariance of the spinor exchange with every spin block populated (SH-1a, gates g1 and
//! g7): `K[R D R^+] = R K[D] R^+` for `R = u (x) 1_m` in the spin-major layout, the linearity
//! of K in D and its reciprocity (grid-plus-compensation part), and the controls that must bite (a spin-diagonal K, `u^T` for `u^+`, the
//! interleaved layout). Mesh 3x1x1 with complex k. Needs `MANDACARU_PAW_PATH`.
mod periodic_hybrid_spinor_support;
mod setup_support;

use mandacaru_core::periodic::Crystal;
use num_complex::Complex64 as C64;
use periodic_hybrid_spinor_support::{rig::Rig, *};

const NAME: &str = "si_so_tr3.json";
const GATE: f64 = 1e-11;
const FRACTION: f64 = 0.25;

fn fixture() -> (Crystal, Rig) {
    let c = hybrid(&context(), NAME, &|_| {});
    let rig = Rig::new(NAME, &c);
    (c, rig)
}

fn worst(c: &Crystal, a: &[Matrix], b: &[Matrix]) -> f64 {
    (0..c.points.len())
        .map(|k| relative(&a[k], &b[k]))
        .fold(0., f64::max)
}

/// The exchange energy `-(a/2) sum_k w_k tr(D_k K_k)` of operators `k` and densities `d`.
fn trace_energy(c: &Crystal, d: &[Matrix], k: &[Matrix]) -> f64 {
    -0.5 * FRACTION
        * (0..c.points.len())
            .map(|i| c.points[i].weight * (&d[i] * &k[i]).trace().re)
            .sum::<f64>()
}

/// `max |D_ud| / max |D|`: the cross blocks the gate needs populated.
fn transverse(c: &Crystal, d: &[Matrix]) -> f64 {
    let top = d
        .iter()
        .map(|x| x.iter().map(|v| v.norm()).fold(0., f64::max))
        .fold(0., f64::max);
    let cross = d
        .iter()
        .map(|x| {
            block(x, c.m, 0, 1)
                .iter()
                .map(|v| v.norm())
                .fold(0., f64::max)
        })
        .fold(0., f64::max);
    cross / top
}

#[test]
#[ignore = "blocked by the SH-2 time-reversal guard: a magnetic D on a time-reversal-flagged crystal is refused, and an unflagged spinor hybrid is refused until periodic::fields::evaluate takes the spin layout (SH-2 log, section 4); green in the scratch copy with that edit"]
fn the_exchange_is_su2_covariant_with_every_spin_block_populated() {
    let (c, _) = fixture();
    let m = c.m;
    let (u, w) = (su2(0.0), su2(1.3));
    let d = rotated_polarized(&c, 5, 3, &u);
    assert!(
        transverse(&c, &d) > 1e-2,
        "the cross-spin blocks of D must be populated"
    );
    let r = spin_rotation(&w, m);
    let rotated: Vec<Matrix> = d.iter().map(|x| &r * x * r.adjoint()).collect();
    for exxdiv in [NONE, GB] {
        let k = c.hybrid_exchange(&d, exxdiv).unwrap();
        let kr = c.hybrid_exchange(&rotated, exxdiv).unwrap();
        let image = |r: &Matrix| -> Vec<Matrix> {
            k.operators.iter().map(|x| r * x * r.adjoint()).collect()
        };
        let miss = worst(&c, &kr.operators, &image(&r));
        let energy = (kr.energy - k.energy).abs();
        let cross = k
            .operators
            .iter()
            .map(|x| block(x, m, 0, 1).norm())
            .fold(0., f64::max);
        println!("{exxdiv:?}: K covariance {miss:.2e}, energy change {energy:.2e}, largest cross block {cross:.2e}");
        assert!(cross > 1e-3, "the cross-spin blocks of K are populated");
        assert!(
            miss < GATE && energy < GATE * k.energy.abs().max(1.),
            "covariance {miss:e}, energy {energy:e}"
        );
        // Controls, each compared with the same oracle.
        let transposed = spin_rotation(&w.transpose(), m); // u^T where u^+ belongs
        let interleaved = spin_rotation_interleaved(&w, m);
        let spatial = Matrix::from_fn(2 * m, 2 * m, |i, j| {
            if i / m == j / m {
                w[(0, 0)] * C64::new((i == j) as u8 as f64, 0.)
            } else {
                C64::default()
            }
        });
        let diagonal: Vec<Matrix> = k.operators.iter().map(|x| spin_diagonal(x, m)).collect();
        let diagonal_rotated: Vec<Matrix> =
            kr.operators.iter().map(|x| spin_diagonal(x, m)).collect();
        for (what, got, want) in [
            ("u^T for u^+", kr.operators.clone(), image(&transposed)),
            ("1 (x) u layout", kr.operators.clone(), image(&interleaved)),
            (
                "a phase on the spatial block only",
                kr.operators.clone(),
                image(&spatial),
            ),
            (
                "cross-spin blocks of K zeroed",
                diagonal_rotated,
                diagonal.iter().map(|x| &r * x * r.adjoint()).collect(),
            ),
        ] {
            let bite = worst(&c, &got, &want);
            println!("  control ({what}): {bite:.2e}");
            assert!(
                bite > 100. * GATE.max(1e-9),
                "control '{what}' does not bite ({bite:e})"
            );
        }
    }
}

#[test]
fn the_energy_is_the_trace_and_needs_the_cross_spin_blocks() {
    let (c, rig) = fixture();
    let m = c.m;
    let d1 = rotated_polarized(&c, 5, 3, &su2(0.4));
    let d2 = rotated_polarized(&c, 4, 4, &su2(-0.9));
    for exxdiv in [NONE, GB] {
        use mandacaru_core::periodic::{
            screened_augmentation::Augmentation,
            screened_exchange::{exchange_operators_with, Screening},
        };
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(&c, sites, &rig.shapes).unwrap();
        let screening = Screening {
            omega: 0.11,
            fraction: FRACTION,
        };
        let run =
            |d: &[Matrix]| exchange_operators_with(&c, d, screening, exxdiv, Some(&aug)).unwrap();
        let (a, b) = (run(&d1), run(&d2));
        // K is linear in D (the eigen-factorization of D must not matter), and E = -(a/2) tr(D K).
        let sum: Vec<Matrix> = d1.iter().zip(&d2).map(|(x, y)| x + y).collect();
        let both = run(&sum);
        let linear = (0..c.points.len())
            .map(|k| relative(&both.operators[k], &(&a.operators[k] + &b.operators[k])))
            .fold(0., f64::max);
        assert!(
            a.energy < -1e-6,
            "the exchange energy is vacuous ({:e})",
            a.energy
        );
        // Reciprocity: tr(D1 K[D2]) = tr(D2 K[D1]).
        let trace = |d: &[Matrix], k: &[Matrix]| {
            (0..c.points.len())
                .map(|i| c.points[i].weight * (&d[i] * &k[i]).trace())
                .sum::<C64>()
        };
        let (x, y) = (trace(&d1, &b.operators), trace(&d2, &a.operators));
        let reciprocity = (x - y).norm() / x.norm();
        // Zeroing the cross-spin blocks of K changes the energy: the measured fraction.
        let zeroed: Vec<Matrix> = a.operators.iter().map(|k| spin_diagonal(k, m)).collect();
        let missed = (trace_energy(&c, &d1, &zeroed) - a.energy).abs() / a.energy.abs();
        println!("{exxdiv:?}: K[D1+D2] - K[D1] - K[D2] {linear:.2e}, reciprocity {reciprocity:.2e}, cross blocks carry {:.1} percent of E", 100. * missed);
        assert!(linear < 1e-12, "{linear:e}");
        assert!(reciprocity < 1e-12, "{reciprocity:e}");
        assert!(
            missed > 1e-2,
            "the zeroed-cross control does not bite ({missed:e})"
        );
    }
}

/// The fixture with the grid origin moved by `shift`, and with the atoms (the sphere centers) too
/// when `with_atoms`: the sampled numbers are unchanged, so with the atoms it is the crystal
/// translated by `shift`.
fn translated(shift: [f64; 3], with_atoms: bool) -> Crystal {
    hybrid(&context(), NAME, &|v| {
        for a in 0..3 {
            let origin = v["grid"]["origin"][a].as_f64().unwrap();
            v["grid"]["origin"][a] = serde_json::json!(origin + shift[a]);
            if with_atoms {
                for center in v["sites"]["centers"].as_array_mut().unwrap() {
                    center[a] = serde_json::json!(center[a].as_f64().unwrap() + shift[a]);
                }
            }
        }
    })
}

#[test]
#[ignore = "blocked by the SH-2 time-reversal guard: a magnetic D on a time-reversal-flagged crystal is refused, and an unflagged spinor hybrid is refused until periodic::fields::evaluate takes the spin layout (SH-2 log, section 4); green in the scratch copy with that edit"]
fn translating_the_crystal_leaves_the_complex_k_operator_unchanged() {
    // The Bloch phases of the pair densities and of the compensation charges must cancel (plan g-S6).
    let (c, _) = fixture();
    let shift = [0.37, -0.91, 1.43];
    let moved = translated(shift, true);
    let d = rotated_polarized(&c, 5, 3, &su2(0.2));
    for exxdiv in [NONE, GB] {
        let (a, b) = (
            c.hybrid_exchange(&d, exxdiv).unwrap(),
            moved.hybrid_exchange(&d, exxdiv).unwrap(),
        );
        let miss = worst(&c, &a.operators, &b.operators);
        println!(
            "{exxdiv:?}: crystal translated by {shift:?}: K {miss:.2e}, energy {:.2e}",
            (a.energy - b.energy).abs()
        );
        assert!(miss < GATE && (a.energy - b.energy).abs() < GATE);
    }
    // The control: the Bloch phases moved without the atoms (the compensation charges keep their
    // place against a moved grid) is another crystal at complex k.
    let origin_only = translated(shift, false);
    let miss = worst(
        &c,
        &c.hybrid_exchange(&d, GB).unwrap().operators,
        &origin_only.hybrid_exchange(&d, GB).unwrap().operators,
    );
    println!("  control (grid origin moved, atoms not): {miss:.2e}");
    assert!(miss > 1e-9, "{miss:e}");
}
