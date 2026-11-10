//! The Lehmann spectral function (`quantum::spectral`) against an exact diagonalization of a
//! Hubbard ring, built from the library's own Jordan-Wigner ladder operators.
use mandacaru_core::{
    math::linalg::eigen,
    molecular::transform::Model,
    quantum::{
        spectral::{lehmann, lorentzian, Poles},
        Basis, Encoding, Operator, C64,
    },
    Context,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// A ring of `m` sites, hopping `-t`, on-site `U`.
fn hubbard(ctx: &Context, m: usize, t: f64, u: f64, particles: [usize; 2]) -> Model {
    let mut one = ctx.allocate::<C64>(m * m).unwrap();
    for p in 0..m {
        for q in [(p + 1) % m, (p + m - 1) % m] {
            one.as_mut_slice()[p * m + q] = C64::new(-t, 0.);
        }
    }
    let mut two = ctx.allocate::<C64>(m * m * m * m).unwrap();
    for p in 0..m {
        two.as_mut_slice()[((p * m + p) * m + p) * m + p] = C64::new(u, 0.);
    }
    Model::new(ctx, m, particles, one, two, 0., false).unwrap()
}

/// The ground state of `model` in its sector, as `(occupations, amplitude)` pairs.
fn ground(ctx: &Context, model: &Model) -> (f64, Vec<(u64, C64)>) {
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
    let state = basis
        .indices
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &ket)| (ket, vectors[(i, 0)]))
        .collect();
    (values[0], state)
}

/// Brute-force Lehmann sum: the library's ladder operators on the whole register, the
/// `N +- 1` sector Hamiltonians diagonalized densely.
fn brute(
    ctx: &Context,
    model: &Model,
    state: &[(u64, C64)],
    w: &[C64],
    sigma: usize,
    dagger: bool,
) -> Poles {
    let m = model.m;
    let n = 2 * m;
    let register = Basis::register(ctx, n).unwrap();
    let plain = Encoding::plain("jordan_wigner", n).unwrap();
    let mut op: Option<Operator> = None;
    for (p, &c) in w.iter().enumerate() {
        let ladder = plain.ladder(ctx, p + sigma * m, dagger).unwrap();
        let c = if dagger { c } else { c.conj() };
        op = Some(match op {
            None => Operator::new(ctx, n, &[])
                .unwrap()
                .add(ctx, &ladder, c)
                .unwrap(),
            Some(o) => o.add(ctx, &ladder, c).unwrap(),
        });
    }
    let mut full = vec![C64::default(); 1 << n];
    for &(ket, a) in state {
        full[ket as usize] = a;
    }
    let image = op.unwrap().apply(ctx, &register, &full).unwrap();
    let h = model.operator(ctx, "jordan_wigner").unwrap();
    let e0: f64 = {
        let applied = h.apply(ctx, &register, &full).unwrap();
        full.iter()
            .zip(applied.as_slice())
            .map(|(a, b)| (a.conj() * b).re)
            .sum()
    };
    let mut target = model.particles;
    if dagger {
        target[sigma] += 1;
    } else {
        target[sigma] -= 1;
    }
    let encoding = Encoding::new("jordan_wigner", n, target).unwrap();
    let basis = Basis::new(ctx, &encoding, true).unwrap();
    let d = basis.len();
    let dense = h.dense(ctx, &basis).unwrap();
    let (values, vectors) = eigen(DMatrix::from_row_slice(d, d, dense.as_slice())).unwrap();
    let mut poles = Poles::default();
    for j in 0..d {
        let overlap: C64 = basis
            .indices
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &ket)| vectors[(i, j)].conj() * image.as_slice()[ket as usize])
            .sum();
        poles.energies.push(if dagger {
            values[j] - e0
        } else {
            e0 - values[j]
        });
        poles.weights.push(overlap.norm_sqr());
    }
    poles
}

fn plane_wave(m: usize, k: f64) -> Vec<C64> {
    (0..m)
        .map(|p| C64::from_polar(1. / (m as f64).sqrt(), 2. * PI * k * p as f64))
        .collect()
}

#[test]
fn the_lehmann_sum_matches_a_brute_force_diagonalization_of_a_hubbard_ring() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let model = hubbard(&ctx, 4, 1.0, 4.0, [1, 1]);
    let (energy, state) = ground(&ctx, &model);
    let operators: Vec<_> = (0..4).map(|j| plane_wave(4, j as f64 / 4.)).collect();
    let spectral = lehmann(&ctx, &model, [1, 1], &state, &operators, 400).unwrap();
    assert!((spectral.ground_energy - energy).abs() < 1e-12);
    let axis: Vec<f64> = (0..801).map(|i| -10. + i as f64 * 0.025).collect();
    for (o, w) in operators.iter().enumerate() {
        for sigma in 0..2 {
            for dagger in [true, false] {
                let reference = brute(&ctx, &model, &state, w, sigma, dagger);
                let ours = if dagger {
                    &spectral.addition[o][sigma]
                } else {
                    &spectral.removal[o][sigma]
                };
                assert!((ours.weight() - reference.weight()).abs() < 1e-12);
                let a = lorentzian(&[ours], &axis, 0.05).unwrap();
                let b = lorentzian(&[&reference], &axis, 0.05).unwrap();
                let error = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f64, f64::max);
                assert!(error < 1e-9, "k {o} spin {sigma} add {dagger}: {error:e}");
            }
            // The sum rule {c, c+} = 1 for a normalized operator.
            let total = spectral.addition[o][sigma].weight() + spectral.removal[o][sigma].weight();
            assert!((total - 1.).abs() < 1e-12, "sum rule {total}");
        }
    }
}

#[test]
fn a_determinant_of_the_tight_binding_ring_has_one_pole_per_k_at_its_band_energy() {
    // U = 0: the ground state is a Slater determinant of plane waves and A(k, omega) is a single
    // delta at eps(k) = -2 t cos(2 pi k) relative to the chemical potential of the state.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let m = 6;
    let t = 0.7;
    // Two electrons per spin: k = 0 and one of the k = +-1/6 pair is degenerate, so use one per
    // spin, the closed shell k = 0.
    let model = hubbard(&ctx, m, t, 0., [1, 1]);
    let (energy, state) = ground(&ctx, &model);
    let band = |k: f64| -2. * t * (2. * PI * k).cos();
    assert!((energy - 2. * band(0.)).abs() < 1e-12);
    for j in 0..m {
        let k = j as f64 / m as f64;
        let w = plane_wave(m, k);
        let spectral = lehmann(&ctx, &model, [1, 1], &state, &[w], 64).unwrap();
        let (occupied, empty) = (&spectral.removal[0][0], &spectral.addition[0][0]);
        if j == 0 {
            assert!((occupied.weight() - 1.).abs() < 1e-12 && empty.weight() < 1e-24);
            // Removing the k = 0 electron costs eps(0): the pole sits at E0 - E(N-1) = eps(0).
            let pole = occupied.energies[occupied.weights.iter().position(|&w| w > 0.5).unwrap()];
            assert!((pole - band(0.)).abs() < 1e-12, "{pole}");
        } else {
            assert!((empty.weight() - 1.).abs() < 1e-12 && occupied.weight() < 1e-24);
            let pole = empty.energies[empty.weights.iter().position(|&w| w > 0.5).unwrap()];
            assert!(
                (pole - band(k)).abs() < 1e-12,
                "k {k}: {pole} vs {}",
                band(k)
            );
        }
    }
}

#[test]
fn a_short_krylov_space_keeps_the_sum_rule_and_the_first_moment() {
    // Gauss quadrature with K vectors reproduces the moments of order < 2K exactly.
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let model = hubbard(&ctx, 4, 1.0, 2.5, [1, 1]);
    let (_, state) = ground(&ctx, &model);
    let w = plane_wave(4, 0.25);
    let exact = lehmann(&ctx, &model, [1, 1], &state, std::slice::from_ref(&w), 400).unwrap();
    let short = lehmann(&ctx, &model, [1, 1], &state, &[w], 2).unwrap();
    for sigma in 0..2 {
        for (a, b) in [
            (&exact.addition[0][sigma], &short.addition[0][sigma]),
            (&exact.removal[0][sigma], &short.removal[0][sigma]),
        ] {
            assert!(b.energies.len() <= 2);
            assert!((a.weight() - b.weight()).abs() < 1e-12);
            let moment = |p: &Poles, order: i32| -> f64 {
                p.energies
                    .iter()
                    .zip(&p.weights)
                    .map(|(e, w)| w * e.powi(order))
                    .sum()
            };
            for order in 1..4 {
                assert!(
                    (moment(a, order) - moment(b, order)).abs() < 1e-9,
                    "moment {order}"
                );
            }
        }
    }
}
