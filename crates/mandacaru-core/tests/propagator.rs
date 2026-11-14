//! The orbital propagator of `math::propagator` on models with exact references: a constant and
//! an explicitly time-dependent Hermitian matrix (R1), and an interacting many-level mean-field
//! model whose matrix is the derivative of its energy (stationarity, energy conservation, time
//! reversal, R3, and the stability map against `dt W`, R1b; TDDFT_PLAN.md section 13).
use mandacaru_core::{
    math::{
        linalg::eigen,
        propagator::{evolve, Channel, Evaluation, FockMap, Options, Propagator, Scheme},
        random::Rng,
    },
    Context, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

type Matrix = DMatrix<C64>;

fn context() -> Context {
    Context::new(Some(1), 256 * 1024 * 1024).unwrap()
}

fn symmetric(rng: &mut Rng, m: usize, scale: f64) -> DMatrix<f64> {
    let a = DMatrix::from_fn(m, m, |_, _| rng.normal());
    (&a + a.transpose()) * (0.5 * scale)
}

fn complex(a: &DMatrix<f64>) -> Matrix {
    a.map(|v| C64::new(v, 0.))
}

fn max_abs(a: &Matrix) -> f64 {
    a.iter().map(|v| v.norm()).fold(0., f64::max)
}

const SCHEMES: [Scheme; 3] = [Scheme::Midpoint, Scheme::Aetrs, Scheme::Etrs];

/// `H(t) = H_0 + sin(omega t) B`, independent of the state.
struct Driven {
    h0: Matrix,
    b: Matrix,
    omega: f64,
    amplitude: f64,
}
impl FockMap for Driven {
    fn evaluate(&mut self, time: f64, densities: &[Matrix]) -> Result<Evaluation> {
        let h = &self.h0 + &self.b * C64::new(self.amplitude * (self.omega * time).sin(), 0.);
        let energy = (&h * &densities[0]).trace().re;
        Ok(Evaluation {
            matrices: vec![h],
            energy,
        })
    }
}

fn driven(amplitude: f64) -> (Driven, Channel) {
    let mut rng = Rng(11);
    let m = 6;
    let h0 = complex(&symmetric(&mut rng, m, 1.));
    let mut b = complex(&symmetric(&mut rng, m, 0.7));
    // A complex Hermitian part, so the test does not live in the real subspace.
    let im = DMatrix::from_fn(m, m, |_, _| rng.normal());
    let anti = (&im - im.transpose()) * 0.25;
    b += anti.map(|v| C64::new(0., v));
    let (_, vectors) = eigen(h0.clone()).unwrap();
    let orbitals = vectors.columns(0, 2).into_owned();
    let channel = Channel {
        orbitals,
        occupations: vec![1., 1.],
    };
    (
        Driven {
            h0,
            b,
            omega: 1.3,
            amplitude,
        },
        channel,
    )
}

fn run_driven(scheme: Scheme, dt: f64, duration: f64, amplitude: f64) -> Matrix {
    let ctx = context();
    let (map, channel) = driven(amplitude);
    let mut p = Propagator::new(&ctx, map, vec![channel], 0., Options::new(scheme, dt)).unwrap();
    let steps = (duration / dt).round() as usize;
    for _ in 0..steps {
        p.step().unwrap();
    }
    assert!((p.time() - duration).abs() < 1e-9);
    p.channels()[0].orbitals.clone()
}

#[test]
fn a_constant_matrix_is_propagated_exactly_at_any_step() {
    let ctx = context();
    for scheme in SCHEMES {
        let (map, channel) = driven(0.);
        let exact = evolve(&map.h0, 3.7 * 9., &channel.orbitals).unwrap().0;
        let mut p =
            Propagator::new(&ctx, map, vec![channel], 0., Options::new(scheme, 3.7)).unwrap();
        for _ in 0..9 {
            let step = p.step().unwrap();
            assert!(step.norm_error < 1e-13, "{scheme:?}: {}", step.norm_error);
        }
        let error = max_abs(&(&p.channels()[0].orbitals - &exact));
        assert!(error < 1e-12, "{scheme:?}: {error:.3e}");
    }
}

#[test]
fn every_scheme_is_second_order_for_a_driven_matrix() {
    let duration = 2.;
    let reference = run_driven(Scheme::Etrs, 1e-3, duration, 1.);
    for scheme in SCHEMES {
        let coarse = max_abs(&(run_driven(scheme, 0.1, duration, 1.) - &reference));
        let fine = max_abs(&(run_driven(scheme, 0.05, duration, 1.) - &reference));
        let order = (coarse / fine).log2();
        assert!(
            (order - 2.).abs() < 0.2,
            "{scheme:?}: errors {coarse:.3e} and {fine:.3e}, order {order:.3}"
        );
    }
}

#[test]
fn options_and_states_that_cannot_propagate_are_refused() {
    let ctx = context();
    let (map, channel) = driven(1.);
    assert!(Scheme::parse("crank-nicolson").is_err());
    assert_eq!(Scheme::parse("aetrs").unwrap(), Scheme::Aetrs);
    let mut bad = channel.clone();
    bad.orbitals *= C64::new(1.01, 0.);
    assert!(Propagator::new(&ctx, map, vec![bad], 0., Options::new(Scheme::Aetrs, 0.1)).is_err());
    let (map, channel) = driven(1.);
    assert!(Propagator::new(
        &ctx,
        map,
        vec![channel],
        0.,
        Options::new(Scheme::Aetrs, 0.)
    )
    .is_err());
    // A matrix that is not Hermitian is refused, not symmetrized silently.
    struct Skewed;
    impl FockMap for Skewed {
        fn evaluate(&mut self, _: f64, d: &[Matrix]) -> Result<Evaluation> {
            let m = d[0].nrows();
            let mut h = Matrix::identity(m, m);
            h[(0, 1)] = C64::new(1e-3, 0.);
            Ok(Evaluation {
                matrices: vec![h],
                energy: 0.,
            })
        }
    }
    let (_, channel) = driven(1.);
    assert!(Propagator::new(
        &ctx,
        Skewed,
        vec![channel],
        0.,
        Options::new(Scheme::Aetrs, 0.1)
    )
    .is_err());
}

/// The interacting model: `E[D] = Re tr(h D) + (U/2) (d^T V d - sum_ij V_ij abs(D_ij)^2)` with
/// `d = diag D`, a Hartree and an exchange term on the sites of a lattice-like model, so
/// `F = dE/dD^T = h + U (diag(V d) - V o D)` is Hermitian for any Hermitian `D` and its exchange
/// part responds to `Im D`. After a kick the energy is a constant of the motion.
#[derive(Clone)]
struct Interacting {
    h: Matrix,
    v: DMatrix<f64>,
    coupling: f64,
}
impl Interacting {
    fn fock(&self, d: &Matrix) -> Matrix {
        let m = d.nrows();
        let sites: Vec<f64> = (0..m).map(|i| d[(i, i)].re).collect();
        let mut f = self.h.clone();
        for i in 0..m {
            let hartree: f64 = (0..m).map(|j| self.v[(i, j)] * sites[j]).sum();
            f[(i, i)] += C64::new(self.coupling * hartree, 0.);
            for j in 0..m {
                f[(i, j)] -= d[(i, j)] * (self.coupling * self.v[(i, j)]);
            }
        }
        f
    }
    fn energy(&self, d: &Matrix) -> f64 {
        let m = d.nrows();
        let mut e = (&self.h * d).trace().re;
        for i in 0..m {
            for j in 0..m {
                let v = self.v[(i, j)];
                e += 0.5 * self.coupling * v * (d[(i, i)].re * d[(j, j)].re - d[(i, j)].norm_sqr());
            }
        }
        e
    }
}
impl FockMap for Interacting {
    fn evaluate(&mut self, _: f64, densities: &[Matrix]) -> Result<Evaluation> {
        Ok(Evaluation {
            matrices: vec![self.fock(&densities[0])],
            energy: self.energy(&densities[0]),
        })
    }
}

struct Model {
    map: Interacting,
    ground: Channel,
    kick: Matrix,
    width: f64,
}

/// The model of vera's T6 check (m = 24, 8 occupied, a bare span of 40), its ground state
/// converged to 1e-13, the basis width `W` of `F[D_0]`, and a kick operator: dense (random) or
/// with elements damped by `exp(-abs(e_a - e_b) / 4)` in the eigenbasis of `F[D_0]`.
fn model(seed: u64, coupling: f64, dense: bool) -> Model {
    let m = 24;
    let occupied = 8;
    let span = 40.;
    let mut rng = Rng(seed);
    let mut h = symmetric(&mut rng, m, 0.6);
    for i in 0..m {
        h[(i, i)] += -0.5 * span + span * i as f64 / (m - 1) as f64;
    }
    let v = symmetric(&mut rng, m, 1.).map(|x| x.abs() * 0.5);
    let v = (&v + v.transpose()) * 0.5;
    let map = Interacting {
        h: complex(&h),
        v,
        coupling,
    };
    let mut d = Matrix::zeros(m, m);
    for i in 0..occupied {
        d[(i, i)] = C64::new(1., 0.);
    }
    let mut converged = false;
    for _ in 0..5000 {
        let (_, vectors) = eigen(map.fock(&d)).unwrap();
        let orbitals = vectors.columns(0, occupied).into_owned();
        let next = &orbitals * orbitals.adjoint();
        let change = max_abs(&(&next - &d));
        if change < 1e-13 {
            d = next;
            converged = true;
            break;
        }
        d = &d * C64::new(0.5, 0.) + next * C64::new(0.5, 0.);
    }
    assert!(converged, "the model SCF did not converge");
    let (values, vectors) = eigen(map.fock(&d)).unwrap();
    let orbitals = vectors.columns(0, occupied).into_owned();
    let width = values[m - 1] - values[0];
    let raw = complex(&symmetric(&mut rng, m, 0.3));
    let kick = if dense {
        raw
    } else {
        let mut x = vectors.adjoint() * &raw * &vectors;
        for a in 0..m {
            for b in 0..m {
                x[(a, b)] *= (-(values[a] - values[b]).abs() / 4.).exp();
            }
        }
        let x = &vectors * x * vectors.adjoint();
        x.map(|v| C64::new(v.re, 0.))
    };
    Model {
        map,
        ground: Channel {
            orbitals,
            occupations: vec![1.; occupied],
        },
        kick,
        width,
    }
}

fn kicked(model: &Model, strength: f64) -> Channel {
    Channel {
        orbitals: evolve(&model.kick, strength, &model.ground.orbitals)
            .unwrap()
            .0,
        occupations: model.ground.occupations.clone(),
    }
}

/// The worst energy deviation over `steps` in units of the kick energy, the worst corrector
/// iteration count, whether every corrector converged, and the Fock builds; `None` if the energy
/// leaves `1e3` kick energies (a blow-up).
struct Trajectory {
    drift: Option<f64>,
    iterations: usize,
    converged: bool,
    builds: usize,
}
fn trajectory(model: &Model, scheme: Scheme, dt_w: f64, steps: usize, strength: f64) -> Trajectory {
    trajectory_to(model, scheme, dt_w, steps, strength, 1e-10, 8)
}
fn trajectory_to(
    model: &Model,
    scheme: Scheme,
    dt_w: f64,
    steps: usize,
    strength: f64,
    corrector_tolerance: f64,
    max_corrector_iterations: usize,
) -> Trajectory {
    let ctx = context();
    let start = kicked(model, strength);
    let ground = model.map.energy(&model.ground.density());
    let initial = model.map.energy(&start.density());
    let kick_energy = initial - ground;
    assert!(kick_energy > 0.);
    let dt = dt_w / model.width;
    let mut options = Options::new(scheme, dt);
    options.corrector_tolerance = corrector_tolerance;
    options.max_corrector_iterations = max_corrector_iterations;
    let mut p = Propagator::new(&ctx, model.map.clone(), vec![start], 0., options).unwrap();
    let mut drift = 0f64;
    let mut iterations = 0;
    let mut converged = true;
    for _ in 0..steps {
        let step = p.advance().unwrap();
        iterations = iterations.max(step.corrector_iterations);
        converged &= step.converged;
        drift = drift.max((step.energy - initial).abs() / kick_energy);
        if !(drift < 1e3) {
            return Trajectory {
                drift: None,
                iterations,
                converged,
                builds: p.builds(),
            };
        }
    }
    Trajectory {
        drift: Some(drift),
        iterations,
        converged,
        builds: p.builds(),
    }
}

#[test]
fn the_unkicked_ground_state_of_the_interacting_model_is_stationary() {
    let ctx = context();
    let model = model(1, 1.5, true);
    let d0 = model.ground.density();
    let e0 = model.map.energy(&d0);
    for scheme in SCHEMES {
        let dt = 2. / model.width;
        let mut p = Propagator::new(
            &ctx,
            model.map.clone(),
            vec![model.ground.clone()],
            0.,
            Options::new(scheme, dt),
        )
        .unwrap();
        for _ in 0..50 {
            let step = p.step().unwrap();
            // Round-off over 50 steps on an energy of order 10^2.
            assert!(
                (step.energy - e0).abs() < 1e-10,
                "{scheme:?}: {:.3e}",
                step.energy - e0
            );
        }
        let change = max_abs(&(&p.densities()[0] - &d0));
        assert!(change < 1e-10, "{scheme:?}: density moved by {change:.3e}");
    }
}

#[test]
fn the_kicked_energy_is_conserved_to_second_order_in_the_step() {
    // The kick energy is 4.4e-5 Ha on an energy of -110.8 Ha, so the energy's own round-off is
    // about 1e-6 kick energies. ETRS (corrector converged to 1e-14) reaches that floor by
    // dt W = 0.5 and is measured at larger steps; a fixed duration of 100 / W throughout.
    let model = model(1, 0.4, false);
    for (scheme, dt_w) in [
        (Scheme::Midpoint, 0.5),
        (Scheme::Aetrs, 0.5),
        (Scheme::Etrs, 2.),
    ] {
        let steps = (100. / dt_w) as usize;
        let coarse = trajectory_to(&model, scheme, dt_w, steps, 0.02, 1e-14, 30);
        let fine = trajectory_to(&model, scheme, dt_w / 2., 2 * steps, 0.02, 1e-14, 30);
        assert!(coarse.converged && fine.converged);
        let (coarse, fine) = (coarse.drift.unwrap(), fine.drift.unwrap());
        let order = (coarse / fine).log2();
        eprintln!(
            "{scheme:?} at dt W {dt_w}: drifts {coarse:.3e} and {fine:.3e}, order {order:.3}"
        );
        // At least second order (the extrapolated schemes are still preasymptotic here, near 3).
        assert!(
            order > 1.7,
            "{scheme:?}: drifts {coarse:.3e} and {fine:.3e}, order {order:.3}"
        );
    }
}

#[test]
fn etrs_retraces_its_path_and_aetrs_to_third_order_per_step() {
    let ctx = context();
    let model = model(2, 1.5, true);
    let start = kicked(&model, 0.02);
    let d0 = start.density();
    let mut defects = Vec::new();
    for (scheme, dt_w) in [
        (Scheme::Etrs, 1.),
        (Scheme::Aetrs, 1.),
        (Scheme::Aetrs, 0.5),
    ] {
        let dt = dt_w / model.width;
        let steps = (40. / dt_w) as usize;
        let mut p = Propagator::new(
            &ctx,
            model.map.clone(),
            vec![start.clone()],
            0.,
            Options::new(scheme, dt),
        )
        .unwrap();
        for _ in 0..steps {
            p.step().unwrap();
        }
        p.reverse();
        for _ in 0..steps {
            p.step().unwrap();
        }
        assert!(p.time().abs() < 1e-9);
        defects.push(max_abs(&(&p.densities()[0] - &d0)));
    }
    eprintln!("time-reversal defects (ETRS dt W 1, AETRS 1, AETRS 0.5): {defects:?}");
    // ETRS: the corrector tolerance per step, accumulated over 80 steps.
    assert!(defects[0] < 1e-8, "ETRS: {:.3e}", defects[0]);
    // AETRS: O(dt^3) per step over a fixed duration, so O(dt^2): a factor of about 4.
    assert!(
        defects[1] > 1e2 * defects[0],
        "AETRS is not exactly reversible: {defects:?}"
    );
    let order = (defects[1] / defects[2]).log2();
    assert!(order > 1.5, "AETRS reversal order {order:.3}: {defects:?}");
}

/// The stability thresholds the fast tier pins, from the map below, at the strong coupling
/// (`U = 1.5`) with a damped kick and the production corrector (1e-10, at most 8 iterations):
/// every scheme keeps the energy within 5e-2 kick energies at `dt W = 1` (measured 2e-2 at
/// worst); ETRS converges its corrector and keeps it within 1e-2 at `dt W = 4` (measured 8e-4).
#[test]
fn the_stability_thresholds_hold_on_the_interacting_model() {
    for seed in [1, 2] {
        let model = model(seed, 1.5, false);
        for scheme in SCHEMES {
            let t = trajectory(&model, scheme, 1., 300, 0.02);
            let drift = t.drift.expect("no blow-up at dt W = 1");
            assert!(drift < 5e-2, "seed {seed} {scheme:?} dt W 1: {drift:.3e}");
        }
        let t = trajectory(&model, Scheme::Etrs, 4., 300, 0.02);
        assert!(t.converged, "seed {seed}: ETRS corrector at dt W 4");
        assert!(
            t.drift.unwrap() < 1e-2,
            "seed {seed}: ETRS dt W 4: {:.3e}",
            t.drift.unwrap()
        );
    }
}

/// The stability map (R1b): the worst energy drift over four seeds, in kick energies, for each
/// scheme against `dt W`, at a weak and a strong coupling, for a damped and a dense kick, and the
/// same at equal Fock builds (AETRS at half the ETRS step). Printed for the plan's record.
#[test]
#[ignore = "slow: 186 s; the R1b stability map, printed; run: cargo test --profile release-test -p mandacaru-core --test propagator -- --ignored --nocapture"]
fn stability_map_of_the_interacting_model() {
    for dense in [false, true] {
        for coupling in [0.4, 1.5] {
            eprintln!(
                "kick {}, U = {coupling}: dt W | AETRS  midpoint  ETRS (iterations, failures) | builds/step ETRS | AETRS at dt W / 2",
                if dense { "dense" } else { "damped" }
            );
            for dt_w in [0.5, 1., 2., 4., 8., 12., 16., 24.] {
                let mut worst = [0f64; 4];
                let mut iterations = 0;
                let mut failures = 0;
                let mut builds = 0.;
                for seed in 1..=4 {
                    let model = model(seed, coupling, dense);
                    let rows = [
                        trajectory(&model, Scheme::Aetrs, dt_w, 300, 0.02),
                        trajectory(&model, Scheme::Midpoint, dt_w, 300, 0.02),
                        trajectory_to(&model, Scheme::Etrs, dt_w, 300, 0.02, 1e-10, 30),
                        trajectory(&model, Scheme::Aetrs, dt_w / 2., 600, 0.02),
                    ];
                    for (w, r) in worst.iter_mut().zip(&rows) {
                        *w = w.max(r.drift.unwrap_or(f64::INFINITY));
                    }
                    iterations = iterations.max(rows[2].iterations);
                    failures += usize::from(!rows[2].converged);
                    builds += rows[2].builds as f64 / 300. / 4.;
                }
                eprintln!(
                    "  {dt_w:5.1} | {:9.2e} {:9.2e} {:9.2e} ({iterations}, {failures}) | {builds:4.1} | {:9.2e}",
                    worst[0], worst[1], worst[2], worst[3]
                );
            }
        }
    }
}
