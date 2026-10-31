//! Hubbard functional model: the Dudarev fully-localized-limit energy and operator,
//! the exactly solvable 2x2 and dimer models against the frozen analytic values,
//! rotation invariance, Hellmann-Feynman building blocks and refusals.
use mandacaru_core::models::hubbard::{
    self, double_counting, eigenvalue_range, evaluate, interaction, Evaluation, Hubbard, Spin,
};
use mandacaru_core::{Context, Error};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64 as C64;
use serde_json::Value;

fn models() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/hubbard/frozen.json"
    );
    let frozen: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    frozen["cases"]["models"].clone()
}

fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

fn hub(u: f64) -> Hubbard {
    Hubbard::new(u, 0.).unwrap()
}

fn real(x: f64) -> [C64; 1] {
    [C64::new(x, 0.)]
}

fn eval1(h: &Hubbard, l: usize, n: &[C64]) -> Evaluation {
    evaluate(
        &context(),
        h,
        l,
        Spin::Unrestricted,
        &[n, &vec![C64::default(); n.len()]],
    )
    .unwrap()
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
}

fn unitary(p: usize, rng: &mut Lcg) -> DMatrix<C64> {
    let m = DMatrix::from_fn(p, p, |_, _| C64::new(rng.next(), rng.next()));
    m.qr().q()
}

/// Hermitian matrix with the given eigenvalues in a random basis.
fn hermitian(p: usize, eigenvalues: &[f64], rng: &mut Lcg) -> DMatrix<C64> {
    let q = unitary(p, rng);
    let d = DMatrix::from_fn(p, p, |i, j| {
        if i == j {
            C64::new(eigenvalues[i], 0.)
        } else {
            C64::default()
        }
    });
    &q * d * q.adjoint()
}

fn flat(m: &DMatrix<C64>) -> Vec<C64> {
    let p = m.nrows();
    (0..p * p).map(|k| m[(k / p, k % p)]).collect()
}

fn matrix(p: usize, v: &[C64]) -> DMatrix<C64> {
    DMatrix::from_fn(p, p, |i, j| v[i * p + j])
}

fn max_diff(a: &DMatrix<C64>, b: &DMatrix<C64>) -> f64 {
    (a - b).iter().map(|x| x.norm()).fold(0., f64::max)
}

fn json_matrix(v: &Value) -> Vec<C64> {
    let re = v["re"].as_array().unwrap();
    let im = v["im"].as_array().unwrap();
    let p = re.len();
    let mut out = Vec::new();
    for i in 0..p {
        for j in 0..p {
            out.push(C64::new(
                re[i][j].as_f64().unwrap(),
                im[i][j].as_f64().unwrap(),
            ));
        }
    }
    out
}

// ---------------------------------------------------------------- the functional

#[test]
fn the_frozen_occupation_matrix_reproduces_the_stored_energy_and_operator() {
    let pure = &models()["pure"];
    let n = json_matrix(&pure["n"]);
    let u = pure["U"].as_f64().unwrap();
    let e = eval1(&hub(u), 1, &n);
    assert!((e.energy - pure["E"].as_f64().unwrap()).abs() < 1e-14);
    let frozen_v = json_matrix(&pure["V"]);
    for (got, want) in e.potentials[0].as_slice().iter().zip(&frozen_v) {
        assert!((got - want).norm() < 1e-14);
    }
    // The empty second channel contributes nothing to E and V = U/2.
    for k in 0..9 {
        let want = if k % 4 == 0 { 0.5 * u } else { 0. };
        assert!((e.potentials[1].as_slice()[k] - C64::new(want, 0.)).norm() < 1e-15);
    }
}

#[test]
fn the_half_filling_identity_holds() {
    let pure = &models()["pure"]["half_filling_identity"];
    let l = pure["l"].as_u64().unwrap() as usize;
    let p = 2 * l + 1;
    let n: Vec<C64> = (0..p * p)
        .map(|k| C64::new(if k / p == k % p { 0.5 } else { 0. }, 0.))
        .collect();
    let e = eval1(&hub(0.3), l, &n);
    // One filled channel at n = 1/2; the empty one contributes zero.
    assert!((e.energy - pure["expected"].as_f64().unwrap()).abs() < 1e-15);
    assert!((e.energy - 0.3 / 2. * p as f64 / 4.).abs() < 1e-15);
}

#[test]
fn a_full_or_empty_shell_and_a_zero_u_are_the_identity() {
    for l in [0usize, 1, 2, 3] {
        let p = 2 * l + 1;
        let full: Vec<C64> = (0..p * p)
            .map(|k| C64::new(if k / p == k % p { 1. } else { 0. }, 0.))
            .collect();
        let empty = vec![C64::default(); p * p];
        for n in [&full, &empty] {
            let e = evaluate(&context(), &hub(4.), l, Spin::Restricted, &[n]).unwrap();
            assert!(e.energy.abs() < 1e-14);
            assert!(e.per_u_eff.abs() < 1e-15);
        }
    }
    let mut rng = Lcg(7);
    let n = flat(&hermitian(5, &[0.9, 0.7, 0.5, 0.2, 0.05], &mut rng));
    for h in [hub(0.), Hubbard::new(3., 3.).unwrap()] {
        let e = eval1(&h, 2, &n);
        assert!(e.energy == 0.);
        assert!(e
            .potentials
            .iter()
            .all(|v| v.as_slice().iter().all(|x| *x == C64::default())));
        assert!(e.per_u_eff > 0., "dE/dU stays defined when U_eff = 0");
    }
}

#[test]
fn energy_and_potential_follow_the_closed_forms() {
    let mut rng = Lcg(11);
    let eig = [0.95, 0.8, 0.45, 0.3, 0.02];
    let n = hermitian(5, &eig, &mut rng);
    let h = Hubbard::new(5., 0.9).unwrap();
    let e = eval1(&h, 2, &flat(&n));
    let ueff = 4.1;
    let want_e = 0.5 * ueff * eig.iter().map(|x| x * (1. - x)).sum::<f64>();
    assert!((e.energy - want_e).abs() < 1e-13);
    assert!((e.per_u_eff * ueff - e.energy).abs() < 1e-15);
    let half = DMatrix::from_fn(5, 5, |i, j| C64::new(if i == j { 0.5 } else { 0. }, 0.));
    let want_v = (half - &n) * C64::new(ueff, 0.);
    assert!(max_diff(&matrix(5, e.potentials[0].as_slice()), &want_v) < 1e-14);
    // E_Hub - E_dc is E_U (the split is not a second route to the number).
    assert!((e.interaction - e.double_counting - e.energy).abs() < 1e-12);
}

#[test]
fn the_restricted_form_is_two_identical_unrestricted_channels() {
    let mut rng = Lcg(13);
    let n = flat(&hermitian(
        7,
        &[0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.0],
        &mut rng,
    ));
    let h = Hubbard::new(6., 1.).unwrap();
    let r = evaluate(&context(), &h, 3, Spin::Restricted, &[&n]).unwrap();
    let u = evaluate(&context(), &h, 3, Spin::Unrestricted, &[&n, &n]).unwrap();
    assert!((r.energy - u.energy).abs() < 1e-14);
    assert!((r.interaction - u.interaction).abs() < 1e-12);
    assert!((r.double_counting - u.double_counting).abs() < 1e-12);
    assert_eq!(r.potentials.len(), 1);
    for (a, b) in r.potentials[0]
        .as_slice()
        .iter()
        .zip(u.potentials[0].as_slice())
    {
        assert!((a - b).norm() < 1e-15);
    }
}

#[test]
fn the_fully_localized_double_counting_is_the_standard_form() {
    let h = Hubbard::new(5., 0.9).unwrap();
    // U/2 N(N-1) - J/2 sum N_s (N_s - 1) at N_up = 4.7, N_dn = 3.1.
    let want = 2.5 * 7.8 * 6.8 - 0.45 * (4.7 * 3.7 + 3.1 * 2.1);
    assert!((double_counting(&h, 4.7, 3.1) - want).abs() < 1e-12);
    // One electron, J = 0: no self-interaction to remove.
    assert!(double_counting(&hub(4.), 1., 0.).abs() < 1e-15);
    let h0 = hub(2.);
    assert!((interaction(&h0, 1., 1., 0.5, 0.5) - (0.5 * 2. * 4. - 0.5 * 2. * 1.)).abs() < 1e-15);
}

// ---------------------------------------------------------------- rotation invariance

#[test]
fn the_energy_is_invariant_and_the_operator_covariant_under_a_shell_rotation() {
    let mut rng = Lcg(2024);
    for l in [1usize, 2, 3] {
        let p = 2 * l + 1;
        let eig: Vec<f64> = (0..p).map(|k| 0.05 + 0.9 * k as f64 / p as f64).collect();
        let n = hermitian(p, &eig, &mut rng);
        let q = unitary(p, &mut rng);
        let rotated = &q * &n * q.adjoint();
        let h = Hubbard::new(4., 0.5).unwrap();
        let a = eval1(&h, l, &flat(&n));
        let b = eval1(&h, l, &flat(&rotated));
        assert!((a.energy - b.energy).abs() < 1e-14, "l = {l}");
        let va = matrix(p, a.potentials[0].as_slice());
        let vb = matrix(p, b.potentials[0].as_slice());
        assert!(max_diff(&(&q * va * q.adjoint()), &vb) < 1e-14, "l = {l}");
    }
}

#[test]
fn complex_and_real_harmonics_give_the_same_energy() {
    // S_lm = sum U_mm' Y_lm': the unitary that takes the complex basis to the real one.
    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let root = std::f64::consts::FRAC_1_SQRT_2;
        let mut u = DMatrix::<C64>::zeros(p, p);
        u[(l, l)] = C64::new(1., 0.);
        for m in 1..=l {
            let sign = if m % 2 == 0 { 1. } else { -1. };
            u[(l + m, l + m)] = C64::new(sign * root, 0.);
            u[(l + m, l - m)] = C64::new(root, 0.);
            u[(l - m, l - m)] = C64::new(0., root);
            u[(l - m, l + m)] = C64::new(0., -sign * root);
        }
        let mut rng = Lcg(5 + l as u64);
        let eig: Vec<f64> = (0..p).map(|k| 0.1 + 0.8 * k as f64 / p as f64).collect();
        let n = hermitian(p, &eig, &mut rng);
        let real_n = &u * &n * u.adjoint();
        assert!((&u * u.adjoint() - DMatrix::<C64>::identity(p, p)).norm() < 1e-14);
        let h = hub(3.);
        let a = eval1(&h, l, &flat(&n));
        let b = eval1(&h, l, &flat(&real_n));
        assert!((a.energy - b.energy).abs() < 1e-14);
    }
}

// ---------------------------------------------------------------- V = dE/dn

/// Largest relative error of `Re Tr[V delta]` against the central difference of
/// `E` along a random Hermitian direction, for each channel.
fn potential_error(
    h: &Hubbard,
    l: usize,
    channels: &[Vec<C64>],
    potential: impl Fn(usize, &[C64]) -> Vec<C64>,
) -> f64 {
    let p = 2 * l + 1;
    let step = 1e-3;
    let mut rng = Lcg(99);
    let energy = |chs: &[Vec<C64>]| {
        let refs: Vec<&[C64]> = chs.iter().map(|c| c.as_slice()).collect();
        evaluate(&context(), h, l, Spin::Unrestricted, &refs)
            .unwrap()
            .energy
    };
    let mut worst = 0f64;
    for c in 0..channels.len() {
        let g = DMatrix::from_fn(p, p, |_, _| C64::new(rng.next(), rng.next()));
        let delta = (&g + g.adjoint()) * C64::new(0.5, 0.);
        let shifted = |sign: f64| {
            let mut chs = channels.to_vec();
            let m = matrix(p, &chs[c]) + &delta * C64::new(sign * step, 0.);
            chs[c] = flat(&m);
            chs
        };
        let fd = (energy(&shifted(1.)) - energy(&shifted(-1.))) / (2. * step);
        let v = matrix(p, &potential(c, &channels[c]));
        let predicted = (0..p)
            .flat_map(|a| (0..p).map(move |b| (a, b)))
            .map(|(a, b)| (v[(a, b)] * delta[(b, a)]).re)
            .sum::<f64>();
        worst = worst.max((fd - predicted).abs() / predicted.abs().max(1e-30));
    }
    worst
}

#[test]
fn the_potential_is_the_derivative_of_the_energy_and_a_wrong_one_fails() {
    let mut rng = Lcg(31);
    let h = Hubbard::new(5., 0.9).unwrap();
    for l in [1usize, 2, 3] {
        let p = 2 * l + 1;
        let up: Vec<f64> = (0..p).map(|k| 0.9 - 0.8 * k as f64 / p as f64).collect();
        let dn: Vec<f64> = (0..p).map(|k| 0.1 + 0.8 * k as f64 / p as f64).collect();
        let channels = vec![
            flat(&hermitian(p, &up, &mut rng)),
            flat(&hermitian(p, &dn, &mut rng)),
        ];
        let engine = |_: usize, n: &[C64]| {
            let other = vec![C64::default(); n.len()];
            let e = evaluate(&context(), &h, l, Spin::Unrestricted, &[n, &other]).unwrap();
            e.potentials[0].as_slice().to_vec()
        };
        let good = potential_error(&h, l, &channels, engine);
        assert!(good < 1e-8, "l = {l}: relative error {good:e}");
        // Negative controls: the same check must reject each plausible mistake.
        let scaled =
            |c: usize, n: &[C64]| engine(c, n).iter().map(|x| x * 1.01).collect::<Vec<_>>();
        let no_spin_half =
            |c: usize, n: &[C64]| engine(c, n).iter().map(|x| x * 2.).collect::<Vec<_>>();
        let shifted = |c: usize, n: &[C64]| {
            let mut v = engine(c, n);
            for a in 0..p {
                v[a * p + a] += C64::new(0.01, 0.);
            }
            v
        };
        for (name, wrong) in [
            ("scaled by 1.01", potential_error(&h, l, &channels, scaled)),
            ("factor 2", potential_error(&h, l, &channels, no_spin_half)),
            ("diagonal shift", potential_error(&h, l, &channels, shifted)),
        ] {
            assert!(
                wrong > 1e-4,
                "l = {l}: a V {name} passed with error {wrong:e}"
            );
        }
    }
}

#[test]
fn the_restricted_derivative_counts_both_spins() {
    let mut rng = Lcg(3);
    let h = hub(4.);
    let n = hermitian(5, &[0.9, 0.8, 0.3, 0.2, 0.1], &mut rng);
    let delta = hermitian(5, &[0.3, -0.2, 0.1, 0.05, -0.4], &mut rng);
    let step = 1e-3;
    let e = |s: f64| {
        let m = flat(&(&n + &delta * C64::new(s * step, 0.)));
        evaluate(&context(), &h, 2, Spin::Restricted, &[&m])
            .unwrap()
            .energy
    };
    let fd = (e(1.) - e(-1.)) / (2. * step);
    let v = matrix(
        5,
        evaluate(&context(), &h, 2, Spin::Restricted, &[&flat(&n)])
            .unwrap()
            .potentials[0]
            .as_slice(),
    );
    let tr = (0..5)
        .flat_map(|a| (0..5).map(move |b| (a, b)))
        .map(|(a, b)| (v[(a, b)] * delta[(b, a)]).re)
        .sum::<f64>();
    assert!((fd - 2. * tr).abs() < 1e-8 * tr.abs());
}

#[test]
fn the_hellmann_feynman_building_blocks_are_the_derivatives_in_u_and_j() {
    let mut rng = Lcg(17);
    let n = flat(&hermitian(5, &[0.9, 0.8, 0.5, 0.2, 0.1], &mut rng));
    let e = |u: f64, j: f64| eval1(&Hubbard::new(u, j).unwrap(), 2, &n);
    let g = e(4., 0.5).per_u_eff;
    let dh = 1e-3;
    assert!(((e(4. + dh, 0.5).energy - e(4. - dh, 0.5).energy) / (2. * dh) - g).abs() < 1e-11);
    assert!(((e(4., 0.5 + dh).energy - e(4., 0.5 - dh).energy) / (2. * dh) + g).abs() < 1e-11);
    // Eigenvalue shifts are <psi|V|psi> / U_eff = <psi|(1/2 - n)|psi>.
    let v = e(4., 0.5).potentials.remove(0);
    let v2 = e(7., 0.5).potentials.remove(0);
    for (a, b) in v.as_slice().iter().zip(v2.as_slice()) {
        assert!((a * (6.5 / 3.5) - b).norm() < 1e-14);
    }
}

// ---------------------------------------------------------------- exactly solvable models

fn lowest(m: &DMatrix<f64>) -> (f64, f64, f64) {
    let eig = SymmetricEigen::new(m.clone());
    let (lo, hi) = if eig.eigenvalues[0] <= eig.eigenvalues[1] {
        (0, 1)
    } else {
        (1, 0)
    };
    (
        eig.eigenvalues[lo],
        eig.eigenvalues[hi],
        eig.eigenvectors[(0, lo)].powi(2),
    )
}

/// 2x2 d + ligand model, one electron per spin, restricted. Returns (E, n).
fn two_by_two(u: f64, seed: f64) -> (f64, f64) {
    let (ed, el, t) = (-0.3, -0.2, -0.15);
    let h = hub(u);
    let ctx = context();
    let potential = |n: f64| {
        evaluate(&ctx, &h, 0, Spin::Restricted, &[&real(n)])
            .unwrap()
            .potentials[0]
            .as_slice()[0]
            .re
    };
    let mut n = seed;
    for _ in 0..200_000 {
        let v = potential(n);
        let (_, _, next) = lowest(&DMatrix::from_row_slice(2, 2, &[ed + v, t, t, el]));
        let done = (next - n).abs() < 1e-15;
        n = 0.5 * n + 0.5 * next;
        if done {
            break;
        }
    }
    let v = potential(n);
    let (eps, _, _) = lowest(&DMatrix::from_row_slice(2, 2, &[ed + v, t, t, el]));
    let eu = evaluate(&ctx, &h, 0, Spin::Restricted, &[&real(n)])
        .unwrap()
        .energy;
    (2. * eps - 2. * n * v + eu, n)
}

#[test]
fn the_two_by_two_model_reaches_the_analytic_stationary_points_including_both_fixed_points() {
    let frozen = models();
    let runs = frozen["two_by_two"]["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 12);
    let mut fixed_points_at_1p2 = Vec::new();
    for run in runs {
        let u = run["U"].as_f64().unwrap();
        let seed = run["seed_n"].as_f64().unwrap();
        let (e, n) = two_by_two(u, seed);
        let want: f64 = run["matched_E"].as_str().unwrap().parse().unwrap();
        assert!(
            (e - want).abs() < 1e-12,
            "U = {u}, seed {seed}: E {e} vs {want}"
        );
        // The closed form E(n) at the converged n is the SCF energy.
        let closed = 2. * (-0.3 * n - 0.2 * (1. - n) + 2. * -0.15 * (n * (1. - n)).sqrt())
            + u * n * (1. - n);
        assert!((closed - e).abs() < 1e-13);
        let point = run["analytic_stationary_points"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| (p["E"].as_str().unwrap().parse::<f64>().unwrap() - want).abs() < 1e-14)
            .unwrap();
        let want_n: f64 = point["n"].as_str().unwrap().parse().unwrap();
        assert!((n - want_n).abs() < 1e-9, "U = {u}: n {n} vs {want_n}");
        assert!(point["minimum"].as_bool().unwrap());
        if (u - 1.2).abs() < 1e-12 {
            fixed_points_at_1p2.push((seed, e));
        }
    }
    // U = 1.2 has two stable fixed points 0.17 Ha apart; the seed picks one.
    let by_seed = |s: f64| fixed_points_at_1p2.iter().find(|x| x.0 == s).unwrap().1;
    assert!((by_seed(0.95) - -0.6638344263865393).abs() < 1e-12);
    assert!((by_seed(0.5) - -0.6638344263865393).abs() < 1e-12);
    assert!((by_seed(0.05) - -0.4922831674410936).abs() < 1e-12);
    assert!((by_seed(0.5) - by_seed(0.05)).abs() > 0.17);
}

#[test]
fn without_hopping_the_levels_shift_by_minus_and_plus_half_u() {
    for case in models()["level_shift_t0"].as_array().unwrap() {
        let u = case["U"].as_f64().unwrap();
        let h = hub(u);
        let level = |n: f64| {
            -0.3 + evaluate(&context(), &h, 0, Spin::Restricted, &[&real(n)])
                .unwrap()
                .potentials[0]
                .as_slice()[0]
                .re
        };
        assert!((level(1.) - case["eps_d_occ"].as_f64().unwrap()).abs() < 1e-14);
        assert!((level(1.) - case["expect_occ"].as_f64().unwrap()).abs() < 1e-14);
        assert!((level(0.) - case["expect_empty"].as_f64().unwrap()).abs() < 1e-14);
    }
}

/// Hubbard dimer: two 1x1 shells, hopping t, one up and one down electron.
/// Returns (E, m, gap) for the seed n_A,up = a.
fn dimer(t: f64, u: f64, a: f64) -> (f64, f64, f64) {
    let h = hub(u);
    let ctx = context();
    // n[site][spin]
    let mut n = [[a, 1. - a], [1. - a, a]];
    let step = |n: &[[f64; 2]; 2]| {
        let mut out = [[0.; 2]; 2];
        let mut eps = [(0., 0.); 2];
        let mut energy = 0.;
        let mut v = [[0.; 2]; 2];
        for site in 0..2 {
            let e = evaluate(
                &ctx,
                &h,
                0,
                Spin::Unrestricted,
                &[&real(n[site][0]), &real(n[site][1])],
            )
            .unwrap();
            energy += e.energy;
            for s in 0..2 {
                v[site][s] = e.potentials[s].as_slice()[0].re;
            }
        }
        for s in 0..2 {
            let m = DMatrix::from_row_slice(2, 2, &[v[0][s], t, t, v[1][s]]);
            let (lo, hi, weight_a) = lowest(&m);
            eps[s] = (lo, hi);
            out[0][s] = weight_a;
            out[1][s] = 1. - weight_a;
            energy += lo - n[0][s] * v[0][s] - n[1][s] * v[1][s];
        }
        (out, energy, eps)
    };
    for _ in 0..200_000 {
        let (next, _, _) = step(&n);
        let change = (0..2)
            .flat_map(|i| (0..2).map(move |s| (i, s)))
            .map(|(i, s)| (next[i][s] - n[i][s]).abs())
            .fold(0., f64::max);
        for i in 0..2 {
            for s in 0..2 {
                n[i][s] = 0.5 * n[i][s] + 0.5 * next[i][s];
            }
        }
        if change < 1e-15 {
            break;
        }
    }
    let (_, energy, eps) = step(&n);
    (energy, n[0][0] - n[0][1], eps[0].1 - eps[0].0)
}

#[test]
fn the_dimer_reproduces_the_nonmagnetic_and_antiferromagnetic_energies() {
    let frozen = models();
    let t = frozen["dimer"]["t"].as_f64().unwrap();
    assert_eq!(t, -0.5);
    for run in frozen["dimer"]["runs"].as_array().unwrap() {
        let u = run["U"].as_f64().unwrap();
        let analytic = &run["analytic"];
        // Non-magnetic state: n = 1/2 on every site, V = 0 exactly.
        let (e_nm, _, _) = dimer(t, u, 0.5);
        let want_nm = analytic["E_NM"].as_f64().unwrap();
        assert!(
            (want_nm - (-2. * t.abs() + u / 2.)).abs() < 1e-15,
            "frozen E_NM at U = {u}"
        );
        assert!((e_nm - want_nm).abs() < 1e-14, "U = {u}");
        if (u - 2. * t.abs()).abs() < 1e-12 {
            continue; // the critical point converges too slowly to gate on
        }
        let (e, m, gap) = dimer(t, u, 0.9);
        if u > 2. * t.abs() {
            let want_e = analytic["E_AFM"].as_f64().unwrap();
            assert!((want_e - -2. * t * t / u).abs() < 1e-15);
            assert!((e - want_e).abs() < 1e-12, "U = {u}: E {e} vs {want_e}");
            let want_m = (1. - (2. * t / u).powi(2)).sqrt();
            assert!((m - want_m).abs() < 1e-9, "U = {u}: m {m} vs {want_m}");
            assert!((m - analytic["m"].as_f64().unwrap()).abs() < 1e-9);
            assert!((gap - u).abs() < 1e-10, "U = {u}: gap {gap}");
            assert!(
                e < e_nm,
                "the antiferromagnet binds below the non-magnetic state"
            );
        } else {
            // No magnetic solution below U = 2|t|: the seed collapses.
            assert!(m.abs() < 1e-6, "U = {u}: m {m}");
            assert!((e - want_nm).abs() < 1e-12);
        }
    }
}

#[test]
fn a_single_shell_holding_both_sites_makes_n_idempotent_and_e_u_vanishes() {
    // The trap: a shell whose occupation is a projector has E_U = 0, so a dimer
    // is two 1x1 shells. A rank-one projector on p = 3 is the executable check.
    let v = [1. / 3f64.sqrt(); 3];
    let projector: Vec<C64> = (0..9).map(|k| C64::new(v[k / 3] * v[k % 3], 0.)).collect();
    let e = evaluate(&context(), &hub(3.), 1, Spin::Restricted, &[&projector]).unwrap();
    eprintln!("rank-one projector on p = 3: E_U = {:e}", e.energy);
    assert!(e.energy.abs() < 1e-14);
}

// ---------------------------------------------------------------- bounds and refusals

fn invalid(r: Result<impl Sized, Error>) -> String {
    match r {
        Err(Error::Invalid(m)) => m,
        Err(other) => panic!("wrong error kind: {other:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

#[test]
fn bad_parameters_are_errors_not_panics() {
    for (u, j) in [
        (f64::NAN, 0.),
        (4., f64::NAN),
        (f64::INFINITY, 0.),
        (-1., 0.),
        (4., -0.1),
        (1., 2.),
    ] {
        invalid(Hubbard::new(u, j));
    }
    assert!(Hubbard::new(4., 4.).unwrap().is_inert());
    assert!(!Hubbard::new(4., 1.).unwrap().is_inert());
    assert_eq!(Hubbard::new(5., 0.9).unwrap().u_eff(), 5. - 0.9);
}

#[test]
fn bad_occupation_matrices_are_errors_not_panics() {
    let ctx = context();
    let h = hub(4.);
    let eye = |p: usize| -> Vec<C64> {
        (0..p * p)
            .map(|k| C64::new(if k / p == k % p { 0.5 } else { 0. }, 0.))
            .collect()
    };
    invalid(evaluate(&ctx, &h, 4, Spin::Restricted, &[&eye(9)])); // shell size
    invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[&eye(3)])); // length
    invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[&eye(5), &eye(5)])); // channels
    invalid(evaluate(&ctx, &h, 2, Spin::Unrestricted, &[&eye(5)]));
    invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[]));
    let mut nan = eye(5);
    nan[7] = C64::new(f64::NAN, 0.);
    invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[&nan]));
    let mut inf = eye(5);
    inf[0] = C64::new(0., f64::INFINITY);
    invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[&inf]));
    let mut skew = eye(5);
    skew[1] = C64::new(0.1, 0.);
    skew[5] = C64::new(0.2, 0.);
    assert!(invalid(evaluate(&ctx, &h, 2, Spin::Restricted, &[&skew])).contains("Hermitian"));
    let mut conj = eye(3);
    conj[1] = C64::new(0., 0.1);
    conj[3] = C64::new(0., 0.1); // should be -0.1i
    invalid(evaluate(&ctx, &h, 1, Spin::Restricted, &[&conj]));
    conj[3] = C64::new(0., -0.1);
    evaluate(&ctx, &h, 1, Spin::Restricted, &[&conj]).unwrap();
    assert!(hubbard::validate(0, &[C64::new(0.3, 0.)]).is_ok());
    invalid(hubbard::shell_size(4));
    invalid(hubbard::trace(1, &eye(5)));
    invalid(eigenvalue_range(&ctx, 1, &eye(5)));
}

#[test]
fn memory_comes_from_the_context_and_a_refusal_keeps_nothing() {
    let diagonal: Vec<C64> = (0..25)
        .map(|k| C64::new(if k / 5 == k % 5 { 0.5 } else { 0. }, 0.))
        .collect();
    let need = 2 * 25 * std::mem::size_of::<C64>();
    let tight = Context::new(Some(1), need - 1).unwrap();
    let r = evaluate(
        &tight,
        &hub(4.),
        2,
        Spin::Unrestricted,
        &[&diagonal, &diagonal],
    );
    assert!(matches!(r, Err(Error::Budget { .. })));
    assert_eq!(tight.memory_used(), 0);
    let exact = Context::new(Some(1), need).unwrap();
    let kept = evaluate(
        &exact,
        &hub(4.),
        2,
        Spin::Unrestricted,
        &[&diagonal, &diagonal],
    )
    .unwrap();
    assert_eq!(exact.memory_used(), need);
    assert_eq!(exact.memory_peak(), need);
    drop(kept);
    assert_eq!(exact.memory_used(), 0);
    let none = Context::new(Some(1), 1).unwrap();
    assert!(matches!(
        eigenvalue_range(&none, 2, &diagonal),
        Err(Error::Budget { .. })
    ));
}

#[test]
fn the_eigenvalue_range_is_the_spectrum_of_n() {
    let mut rng = Lcg(77);
    let eig = [0.97, 0.6, 0.5, 0.33, 0.01];
    let n = flat(&hermitian(5, &eig, &mut rng));
    let (lo, hi) = eigenvalue_range(&context(), 2, &n).unwrap();
    assert!((lo - 0.01).abs() < 1e-12 && (hi - 0.97).abs() < 1e-12);
    // A measured excess is reported, not clamped.
    let over = flat(&hermitian(5, &[1.02, 0.5, 0.5, 0.5, -0.003], &mut rng));
    let (lo, hi) = eigenvalue_range(&context(), 2, &over).unwrap();
    assert!((hi - 1.02).abs() < 1e-12 && (lo + 0.003).abs() < 1e-12);
    assert!((hubbard::trace(2, &n).unwrap() - eig.iter().sum::<f64>()).abs() < 1e-13);
}
