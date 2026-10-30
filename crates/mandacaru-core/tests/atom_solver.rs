//! Reference-atom solver: parity with the frozen pre-port Python numbers, the Pulay mixer,
//! configuration selection and the relativistic invariants.
use mandacaru_core::{
    models::{
        atom::{self, AtomModel, AtomOptions},
        configuration,
        mixing::{density_change, PulayMixer},
        numerov,
        relativity::{self, Treatment},
        selection,
    },
    Context,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn reference() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/paw/atomic_solver.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// Frozen cases: eigenvalues, `(2j+1)`-resolved levels and total energy against the Python
/// solver, with the largest relative difference per case printed (`--nocapture`).
#[test]
fn matches_the_frozen_python_atoms() {
    let frozen = reference();
    let ctx = context();
    let options = AtomOptions {
        points: frozen["points"].as_u64().unwrap() as usize,
        r_max: frozen["r_max"].as_f64().unwrap(),
        max_iterations: frozen["max_iterations"].as_u64().unwrap() as usize,
        tolerance: frozen["tolerance"].as_f64().unwrap(),
        mixing: frozen["mixing"].as_f64().unwrap(),
    };
    let only = std::env::var("ATOM_PARITY_ONLY").ok();
    let mut worst: BTreeMap<String, f64> = BTreeMap::new();
    for (key, case) in frozen["cases"].as_object().unwrap() {
        if only.as_ref().is_some_and(|o| !key.starts_with(o.as_str())) {
            continue;
        }
        let parts = key.split('/').collect::<Vec<_>>();
        let z: u8 = parts[0].parse().unwrap();
        let model = AtomModel {
            functional: parts[1].to_string(),
            relativity: Treatment::parse(parts[2]).unwrap(),
            polish: frozen["polish"].as_u64().unwrap() as usize,
        };
        let atom = atom::solve_model(&ctx, z, &options, &model, None, None).unwrap();
        assert_eq!(
            atom.converged,
            case["converged"].as_bool().unwrap(),
            "{key}"
        );
        let mut diff = relative(atom.energy, case["total_energy"].as_f64().unwrap());
        let energy_diff = diff;
        let mut eig_diff: f64 = 0.;
        for (i, &(n, l, _)) in atom.occupations.iter().enumerate() {
            let want = case["eigenvalues"][format!("{n},{l}")].as_f64().unwrap();
            eig_diff = eig_diff.max(relative(atom.eigenvalues[i], want));
        }
        for level in &atom.j_states {
            let want = case["eigenvalues_j"][format!("{},{},{}", level.n, level.l, level.kappa)]
                .as_f64()
                .unwrap();
            eig_diff = eig_diff.max(relative(level.eigenvalue, want));
        }
        diff = diff.max(eig_diff);
        println!(
            "{key:>14}  iterations {:>2}/{:>2}  energy {energy_diff:.2e}  eigenvalues {eig_diff:.2e}",
            atom.iterations,
            case["iterations"].as_u64().unwrap()
        );
        worst.insert(key.clone(), diff);
    }
    let max = worst.values().cloned().fold(0., f64::max);
    println!(
        "largest relative difference over {} cases: {max:.3e}",
        worst.len()
    );
    for (key, diff) in &worst {
        assert!(*diff < 1e-6, "{key}: {diff:e}");
    }
}

fn options(points: usize, r_max: f64) -> AtomOptions {
    AtomOptions {
        points,
        r_max,
        max_iterations: 200,
        tolerance: 1e-8,
        mixing: 0.3,
    }
}

fn model(functional: &str, relativity: Treatment) -> AtomModel {
    AtomModel {
        functional: functional.to_string(),
        relativity,
        polish: 12,
    }
}

fn grid(points: usize, r_max: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let step = r_max / (points as f64 + 1.);
    let r = (1..=points).map(|i| i as f64 * step).collect::<Vec<_>>();
    let weight = r.iter().map(|r| 4. * PI * r * r * step).collect();
    (r, weight, step)
}

#[test]
fn density_change_sees_a_valence_change_the_plain_maximum_misses() {
    let (r, _, _) = grid(400, 20.);
    let tail = r.iter().map(|r| 1e-3 * (-r).exp()).collect::<Vec<_>>();
    let moved = r
        .iter()
        .map(|r| 1e-3 * (-0.9 * r).exp())
        .collect::<Vec<_>>();
    let core = r.iter().map(|r| 1e4 * (-40. * r).exp()).collect::<Vec<_>>();
    let sum = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a + b).collect::<Vec<_>>();
    let plain = tail
        .iter()
        .zip(&moved)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(plain < 1e-4);
    assert!(density_change(&sum(&core, &tail), &sum(&core, &moved), &r) > 10. * plain);
    // never weaker than the plain maximum
    let a = r.iter().map(|r| (-r).exp()).collect::<Vec<_>>();
    let b = r.iter().map(|r| (-1.1 * r).exp()).collect::<Vec<_>>();
    let plain = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(density_change(&a, &b, &r) >= plain);
}

/// A contracting linear map with a known positive fixed point.
fn linear_problem(r: &[f64]) -> (Vec<f64>, impl Fn(&[f64]) -> Vec<f64>) {
    let fixed = r
        .iter()
        .map(|r| (-r).exp() + 0.1 * (-0.3 * r).exp())
        .collect::<Vec<_>>();
    let basis = [0.5f64, 1.0, 2.0, 3.0]
        .iter()
        .map(|k| r.iter().map(|r| (-k * r).exp()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let coupling = [
        [0.31, -0.62, 0.45, 0.12],
        [-0.77, 0.2, 0.58, -0.4],
        [0.66, -0.15, -0.83, 0.27],
        [0.05, 0.71, -0.36, 0.5],
    ];
    let reference = fixed.clone();
    let output = move |n: &[f64]| {
        let delta = n
            .iter()
            .zip(&reference)
            .map(|(n, f)| n - f)
            .collect::<Vec<_>>();
        let moments = basis
            .iter()
            .map(|b| b.iter().zip(&delta).map(|(b, d)| b * d).sum::<f64>())
            .collect::<Vec<_>>();
        (0..delta.len())
            .map(|i| {
                let coupled: f64 = (0..4)
                    .map(|a| (0..4).map(|c| coupling[a][c] * moments[c]).sum::<f64>() * basis[a][i])
                    .sum();
                reference[i] + 0.95 * delta[i] + 0.02 * coupled
            })
            .collect()
    };
    (fixed, output)
}

#[test]
fn pulay_mixing_beats_linear_mixing_on_a_linear_fixed_point() {
    let (r, weight, _) = grid(400, 20.);
    let (fixed, output) = linear_problem(&r);
    let run = |history: usize| {
        let mut mixer = PulayMixer::new(weight.clone(), 0.3, 1., history);
        let mut n = fixed.iter().map(|f| 2. * f).collect::<Vec<_>>();
        for _ in 0..60 {
            n = mixer.mix(&n, &output(&n));
        }
        n.iter()
            .zip(&fixed)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max)
    };
    let (pulay, linear) = (run(8), run(1));
    assert!(pulay < 1e-8, "{pulay:e}");
    assert!(pulay < 1e-3 * linear, "{pulay:e} against {linear:e}");
}

#[test]
fn pulay_mixing_conserves_charge_without_renormalizing() {
    let (r, weight, _) = grid(400, 20.);
    let electrons = 3.;
    let normalized = |scale: f64| {
        let mut n = r.iter().map(|r| (-scale * r).exp()).collect::<Vec<_>>();
        let charge = n.iter().zip(&weight).map(|(n, w)| n * w).sum::<f64>();
        n.iter_mut().for_each(|v| *v *= electrons / charge);
        n
    };
    let mut mixer = PulayMixer::new(weight.clone(), 0.3, electrons, 8);
    let mut n = normalized(1.);
    for scale in [1.3, 0.8, 1.1, 0.95] {
        n = mixer.mix(&n, &normalized(scale));
        let charge = n.iter().zip(&weight).map(|(n, w)| n * w).sum::<f64>();
        assert!((charge - electrons).abs() < 1e-12 * electrons);
    }
}

#[test]
fn a_negative_extrapolation_falls_back_to_the_linear_step() {
    // Two inputs whose residuals are nearly parallel drive the coefficients far outside
    // [0, 1]; the combination dips negative, so the mixer must return the linear step.
    let (r, weight, _) = grid(400, 20.);
    let mut mixer = PulayMixer::new(weight, 0.3, 1., 8);
    let a = r.iter().map(|r| (-r).exp()).collect::<Vec<_>>();
    let b = r.iter().map(|r| 0.5 * (-0.5 * r).exp()).collect::<Vec<_>>();
    let first = mixer.mix(&a, &b);
    let linear = a
        .iter()
        .zip(&b)
        .map(|(a, b)| a + 0.3 * (b - a))
        .collect::<Vec<_>>();
    assert_eq!(first, linear);
    let c = r.iter().map(|r| (-1.2 * r).exp()).collect::<Vec<_>>();
    let second = mixer.mix(&first, &c);
    assert!(second.iter().all(|v| *v >= 0.));
}

#[test]
fn aufbau_and_rearrangements_follow_the_filling_rule() {
    let oxygen = configuration::ground_state(8).unwrap();
    assert_eq!(oxygen, vec![(1, 0, 2.), (2, 0, 2.), (2, 1, 4.)]);
    // 2s -> 2p is not a crossing the aufbau order can get wrong: no candidate besides aufbau
    assert_eq!(configuration::rearrangements(8, 3).unwrap().len(), 1);
    let lanthanum = configuration::rearrangements(57, 3).unwrap();
    assert!(lanthanum.len() > 1);
    assert!(lanthanum
        .iter()
        .any(|c| c.contains(&(5, 2, 1.)) && !c.iter().any(|s| (s.0, s.1) == (4, 3))));
    assert_eq!(
        configuration::valence_subshells(&configuration::ground_state(26).unwrap()),
        vec![(3, 2), (4, 0)]
    );
    // j-j filling: the lower j (kappa = +l) fills first
    let split = configuration::split_configuration(&[(2, 1, 4.)]);
    assert_eq!(split, vec![(2, 1, 1, 2.), (2, 1, -2, 2.)]);
    let open = configuration::split_configuration(&[(2, 1, 3.)]);
    assert_eq!(open, vec![(2, 1, 1, 2.), (2, 1, -2, 1.)]);
}

#[test]
fn an_atom_without_an_f_valence_shell_is_not_solved_to_choose_its_configuration() {
    // A 1 MiB context fails any solve at the selection grid: only the table lookup can succeed.
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    for z in [8u8, 26, 29] {
        let chosen =
            selection::relaxed_configuration(&ctx, z, "lda", Treatment::None, 25., None).unwrap();
        assert_eq!(chosen, configuration::ground_state(z).unwrap());
    }
}

#[test]
fn an_unbound_f_shell_moves_one_electron_to_the_d_shell() {
    // At 1500 points in a 60 Bohr box lanthanum's aufbau 4f is held by the box
    // (frozen Python: -0.0006 Ha); the lowest-energy bound rearrangement is 5d^1 6s^2.
    let frozen = reference();
    let want = &frozen["relaxed"]["relax3:57,1500,60.0"];
    let ctx = context();
    let chosen =
        selection::relaxed_configuration(&ctx, 57, "lda", Treatment::None, 60., Some(1500))
            .unwrap();
    for (n, l, q) in &chosen {
        assert_eq!(want[format!("{n},{l}")].as_f64().unwrap(), *q);
    }
    assert_eq!(chosen.len(), want.as_object().unwrap().len());
    assert!(chosen.contains(&(5, 2, 1.)) && !chosen.iter().any(|s| (s.0, s.1) == (4, 3)));
    // the same atom on the selection grid keeps the aufbau filling (4f is bound)
    let default =
        selection::relaxed_configuration(&ctx, 57, "lda", Treatment::None, 25., None).unwrap();
    assert!(default.contains(&(4, 3, 1.)));
}

#[test]
fn the_relativistic_levels_of_a_dirac_atom_average_to_the_scalar_ones() {
    let ctx = context();
    let opts = options(3000, 18.);
    let dirac =
        atom::solve_model(&ctx, 10, &opts, &model("lda", Treatment::Dirac), None, None).unwrap();
    let scalar = atom::solve_model(
        &ctx,
        10,
        &opts,
        &model("lda", Treatment::Scalar),
        None,
        None,
    )
    .unwrap();
    assert!(dirac.converged && scalar.converged);
    // an s level has one j, and kappa = -1 is its physical value: scalar == Dirac
    let s = |a: &atom::Atom, n: usize| {
        a.occupations
            .iter()
            .position(|s| (s.0, s.1) == (n, 0))
            .map(|i| a.eigenvalues[i])
            .unwrap()
    };
    assert!(((s(&dirac, 2) - s(&scalar, 2)) / s(&scalar, 2)).abs() < 1e-6);
    // spin-orbit splits 2p: j = 3/2 (kappa = -2) lies above j = 1/2 (kappa = 1) for a
    // neutral atom's outer shell? no -- below: more binding for the lower j
    let level = |kappa: i32| {
        dirac
            .j_states
            .iter()
            .find(|s| (s.n, s.l, s.kappa) == (2, 1, kappa))
            .unwrap()
            .eigenvalue
    };
    assert!(level(1) < level(-2));
    let p = dirac
        .occupations
        .iter()
        .position(|s| (s.0, s.1) == (2, 1))
        .unwrap();
    let average = relativity::j_average(1, level);
    assert!((dirac.eigenvalues[p] - average).abs() < 1e-12);
    // occupations: 2p^6 fills 2 + 4 electrons
    let filled = dirac
        .j_states
        .iter()
        .filter(|s| (s.n, s.l) == (2, 1))
        .map(|s| s.occupation)
        .sum::<f64>();
    assert_eq!(filled, 6.);
    assert!(scalar.j_states.is_empty());
}

#[test]
fn relativity_lowers_the_total_energy_and_pbe_binds_more_than_lda() {
    let ctx = context();
    let opts = options(2500, 18.);
    let energy = |functional: &str, treatment| {
        atom::solve_model(&ctx, 8, &opts, &model(functional, treatment), None, None)
            .unwrap()
            .energy
    };
    assert!(energy("lda", Treatment::Scalar) < energy("lda", Treatment::None));
    assert!(energy("pbe", Treatment::None) < energy("lda", Treatment::None));
}

#[test]
fn a_hydrogenic_bound_state_agrees_with_the_tridiagonal_level() {
    let ctx = context();
    let (r, _, _) = grid(6000, 40.);
    let z = 2.;
    let potential = r.iter().map(|r| -z / r).collect::<Vec<_>>();
    for (n, l) in [(2usize, 1usize), (3, 2), (3, 0)] {
        let (_, level) = relativity::solve_radial(
            &ctx,
            &r,
            &potential,
            l,
            n - l - 1,
            Treatment::None,
            None,
            z,
            None,
            None,
        )
        .unwrap();
        let exact = -z * z / (2. * (n * n) as f64);
        assert!(
            ((level - exact) / exact).abs() < 2e-4,
            "{n}{l}: {level} against {exact}"
        );
        let (u, shot) = numerov::bound_state(
            &ctx,
            &r,
            &potential,
            l,
            level,
            z,
            numerov::DEFAULT_WINDOW,
            Treatment::None,
            None,
        )
        .unwrap()
        .expect("the shoot brackets a hydrogenic level");
        // Numerov is fourth order: it lands closer to the exact level than the tridiagonal one
        assert!(((shot - exact) / exact).abs() < ((level - exact) / exact).abs());
        let norm = r
            .windows(2)
            .zip(u.windows(2))
            .map(|(r, u)| 0.5 * (r[1] - r[0]) * (u[0] * u[0] + u[1] * u[1]))
            .sum::<f64>();
        assert!((norm - 1.).abs() < 1e-12);
    }
}

#[test]
fn unsupported_requests_raise_instead_of_choosing_another_calculation() {
    let ctx = context();
    let opts = options(500, 15.);
    assert!(Treatment::parse("zora").is_err());
    assert!(atom::solve_model(
        &ctx,
        2,
        &opts,
        &model("r2scan", Treatment::None),
        None,
        None
    )
    .is_err());
    assert!(atom::solve_model(
        &ctx,
        2,
        &opts,
        &model("hse06", Treatment::Scalar),
        None,
        None
    )
    .is_err());
    assert!(atom::solve_model(&ctx, 0, &opts, &model("lda", Treatment::None), None, None).is_err());
    assert!(atom::solve_model(
        &ctx,
        2,
        &opts,
        &model("lda", Treatment::None),
        Some(&[(1, 0, 3.)]),
        None
    )
    .is_err());
    let wall = vec![0.; 10];
    assert!(atom::solve_model(
        &ctx,
        2,
        &opts,
        &model("lda", Treatment::None),
        None,
        Some(&wall)
    )
    .is_err());
    assert!(relativity::solve_radial(
        &ctx,
        &[0.1, 0.2, 0.3, 0.4],
        &[0.; 4],
        0,
        0,
        Treatment::Dirac,
        None,
        1.,
        None,
        None
    )
    .is_err());
}
