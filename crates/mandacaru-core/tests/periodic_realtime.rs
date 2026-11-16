//! Real-time propagation of a crystal in the velocity gauge (`periodic::optics::realtime`,
//! TDDFT_PLAN.md T6b), checked by identities of the engine's own numbers (section 13.8):
//!
//! - R8: with the Hamiltonian frozen (`Kernel::Independent`), the current equals the exact
//!   spectral evolution under `H_o + A_0 P_o`, at any time step and field;
//! - R10: in the linear limit the independent-particle current is the band sum's cosine series,
//!   and its dielectric function the closed form of the same transitions;
//! - R2: the orbitals stay orthonormal, the energy after a kick is conserved to `O(dt^2)`, and
//!   under a smooth ramp `dE/dt = Omega j . E`;
//! - R9b: a ramp to a static `A` leaves the current `-A (N - N_dia) / Omega` with the band sum's
//!   `N_dia` (a density-only kernel);
//! - R14: the current is bit-identical across `kpts` layouts and thread counts;
//! - R15: the whole-mesh start of a symmetry-reduced SCF has the SCF's density and energy;
//! - Si in R8, R10 and R2; `direction="all"` (the cubic tensor); a magnetic crystal's two
//!   unequal channels; DFT+U with the ramp-measured `N_dia`; r2SCAN with `tau~` (T6c-c); R13
//!   against the RPA head (recorded); and the F1/F2 comparison that sets the default (Q20).
//!
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;
mod wannier_downfold_support;

use berry_support::*;
use mandacaru_core::{
    math::{linalg::eigen, propagator::Channel},
    models::time_spectrum::{dielectric_function, Window},
    periodic::{
        bloch_mesh::MeshIndex,
        optics::{
            momentum::transitions,
            realtime::{
                prepare, propagate, propagate_axes, Field, Formulation, Kernel, Prepared, Request,
            },
        },
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::sync::Arc;

type Matrix = DMatrix<C64>;

/// The current of `prepared` evolved exactly under the frozen `H_o + A . P_o + A^2 / 2` at `times`.
fn exact_current(prepared: &Prepared, strength: f64, times: &[f64]) -> Vec<[f64; 3]> {
    let c = &prepared.crystal;
    let nk = c.points.len();
    let a = prepared.direction.map(|e| strength * e);
    let volume = c.grid.index.dv() * c.grid.index.points() as f64;
    let spectra: Vec<(Vec<f64>, Matrix)> = prepared
        .ground
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            let mut h = h.clone();
            for axis in 0..3 {
                h += &prepared.momenta[idx % nk][axis] * C64::new(a[axis], 0.);
            }
            eigen(h).unwrap()
        })
        .collect();
    times
        .iter()
        .map(|&t| {
            let mut paramagnetic = [0.; 3];
            for (idx, channel) in prepared.channels.iter().enumerate() {
                let (levels, vectors) = &spectra[idx];
                let phases = Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    levels.len(),
                    levels.iter().map(|e| C64::from_polar(1., -e * t)),
                ));
                let orbitals = vectors * phases * vectors.adjoint() * &channel.orbitals;
                let density = Channel {
                    orbitals,
                    occupations: channel.occupations.clone(),
                }
                .density();
                let weight = c.points[idx % nk].weight;
                for (axis, value) in paramagnetic.iter_mut().enumerate() {
                    *value += weight * (&density * &prepared.momenta[idx % nk][axis]).trace().re;
                }
            }
            std::array::from_fn(|axis| -(paramagnetic[axis] + c.electrons * a[axis]) / volume)
        })
        .collect()
}

#[test]
fn a_frozen_hamiltonian_gives_the_exact_spectral_current() {
    // R8: no rebuild, so the exponential is exact at any dt and any A_0 (here far from linear).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    for (time_step, steps) in [(0.2, 30), (0.6, 10)] {
        let mut request = Request::new([1., 0.3, 0.], 0.05, time_step, steps);
        request.kernel = Kernel::Independent;
        let prepared = prepare(&sampler, &request).unwrap();
        let run = prepared.propagate(&sampler, &request).unwrap();
        let exact = exact_current(&prepared, request.strength, &run.times);
        let scale = exact
            .iter()
            .flat_map(|j| j.iter())
            .fold(0f64, |t, v| t.max(v.abs()));
        let error = run
            .current
            .iter()
            .zip(&exact)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0f64, f64::max);
        eprintln!(
            "dt {time_step}: largest |j| {scale:.4e}, error {error:.2e}, norm {:.1e}, drift {:.1e}",
            run.worst_norm_error, run.energy_drift
        );
        assert!(scale > 1e-4 && error < 1e-12 * scale.max(1.), "{error:e}");
        assert!(run.worst_norm_error < 1e-10);
    }
}

/// `j(A_0) - j(-A_0)` over 2: the odd, linear-limit part of the corrected current along `e`.
fn odd_current(
    sampler: &mandacaru_core::periodic::bloch_mesh::Sampler<'_>,
    request: &Request,
) -> (
    Vec<f64>,
    mandacaru_core::periodic::optics::realtime::CrystalRealTime,
) {
    let plus = propagate(sampler, request).unwrap();
    let mut minus = request.clone();
    minus.strength = -request.strength;
    let minus = propagate(sampler, &minus).unwrap();
    let odd = plus
        .corrected
        .iter()
        .zip(&minus.corrected)
        .map(|(a, b)| 0.5 * (a - b))
        .collect();
    (odd, plus)
}

#[test]
fn the_independent_particle_current_is_the_band_sum_series() {
    // R10: j(t) = -(A_0 / Omega) sum_t (2 g w_t / gap_t) cos(gap_t t) after the deficit is
    // removed, and epsilon(z) = 1 + (4 pi / Omega) sum_t a_t / (gap_t^2 - z^2), a_t = 2 g w_t /
    // gap_t: the sign and the prefactor of the real-time dielectric function, from the band
    // sum of `transitions`, which shares only P with the propagation.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let lines = transitions(
        &sampler,
        &MeshIndex::new([2, 1, 1], [0.; 3]).unwrap(),
        [1., 0., 0.],
    )
    .unwrap();
    let amplitudes: Vec<f64> = lines
        .weights
        .iter()
        .zip(&lines.gaps)
        .map(|(w, gap)| 2. * lines.degeneracy * w / gap)
        .collect();
    // The odd part leaves the third order: the error falls as A_0^2 (measured ratio below).
    let mut errors = vec![];
    let mut last = None;
    for strength in [2e-3, 1e-3] {
        let mut request = Request::new([1., 0., 0.], strength, 0.05, 2000);
        request.kernel = Kernel::Independent;
        let (odd, run) = odd_current(&sampler, &request);
        let oracle: Vec<f64> = run
            .times
            .iter()
            .map(|t| {
                -strength / run.volume
                    * amplitudes
                        .iter()
                        .zip(&lines.gaps)
                        .map(|(a, gap)| a * (gap * t).cos())
                        .sum::<f64>()
            })
            .collect();
        let scale = oracle.iter().fold(0f64, |t, v| t.max(v.abs()));
        let error = odd
            .iter()
            .zip(&oracle)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max)
            / scale;
        eprintln!(
            "LiH A_0 {strength}: N {}, N_dia {:.8}, largest |j| {scale:.4e}, relative error \
             {error:.2e}",
            run.electrons, run.diamagnetic
        );
        assert!((run.diamagnetic - lines.diamagnetic(1).total).abs() < 1e-12);
        errors.push(error);
        last = Some((odd, run, request, strength));
    }
    assert!(
        errors[1] < 1e-3 && errors[0] / errors[1] > 3.5,
        "{errors:?}"
    );
    let (odd, run, request, strength) = last.unwrap();

    let ctx = &solved.ctx;
    let damping = 0.1;
    let frequencies: Vec<f64> = (1..=60).map(|i| 0.025 * i as f64).collect();
    let epsilon = dielectric_function(
        ctx,
        &odd,
        request.time_step,
        strength,
        &frequencies,
        damping,
        Window::Exponential,
    )
    .unwrap();
    let mut worst = 0f64;
    let mut largest = 0f64;
    for (value, omega) in epsilon.values.as_slice().iter().zip(&frequencies) {
        let z = C64::new(*omega, damping);
        let closed = C64::new(1., 0.)
            + amplitudes
                .iter()
                .zip(&lines.gaps)
                .map(|(a, gap)| {
                    C64::new(4. * std::f64::consts::PI * a / run.volume, 0.) / (gap * gap - z * z)
                })
                .sum::<C64>();
        largest = largest.max(closed.im.abs());
        worst = worst.max((value - closed).norm());
    }
    eprintln!(
        "epsilon: largest |Im| {largest:.4e}, worst |difference| {worst:.2e}, tail {:.1e}",
        epsilon.tail
    );
    assert!(worst < 1e-3 * largest, "{worst:e} against {largest:e}");
}

#[test]
fn the_kicked_crystal_conserves_its_norm_and_energy() {
    // R2 (crystal): orthonormal orbitals, and an energy drift that falls as dt^2.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mut drifts = vec![];
    for (time_step, steps) in [(0.4, 25), (0.2, 50)] {
        let request = Request::new([1., 0., 0.], 0.01, time_step, steps);
        let run = propagate(&sampler, &request).unwrap();
        eprintln!(
            "dt {time_step}: kick energy {:.4e} (N A^2 / 2 = {:.4e}), drift {:.3e} kick energies, \
             norm {:.1e}, builds {}, dt W {:.3}, j(0) {:.2e}, N_dia {:.6}",
            run.kick_energy,
            0.5 * run.electrons * 1e-4,
            run.energy_drift,
            run.worst_norm_error,
            run.builds,
            run.step_width,
            run.along[0] + 0.01 * run.electrons / run.volume,
            run.diamagnetic
        );
        assert!(run.worst_norm_error < 1e-8);
        assert!((run.kick_energy - 0.5 * run.electrons * 1e-4).abs() < 1e-3 * run.kick_energy);
        drifts.push(run.energy_drift);
    }
    assert!(drifts[1] < drifts[0] / 2.5, "{drifts:?}");
}

#[test]
fn a_ramp_balances_the_power_of_the_field() {
    // R2 (crystal): under A(t) = A_0 s(t) e, E(T) - E(0) = int Omega j . E dt with
    // E_field = -dA/dt (a smooth test field; the integral is the trapezoid of the samples).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let strength = 0.02;
    let mut request = Request::new([1., 0., 0.], strength, 0.1, 200);
    request.field = Field::Ramp { duration: 20. };
    let run = propagate(&sampler, &request).unwrap();
    let rate = |t: f64| strength * request.field.rate(t);
    let power: Vec<f64> = run
        .times
        .iter()
        .zip(&run.along)
        .map(|(t, j)| -run.volume * j * rate(*t))
        .collect();
    let work: f64 = power
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]) * request.time_step)
        .sum();
    let change = run.energy[run.energy.len() - 1] - run.energy[0];
    eprintln!(
        "ramp: energy change {change:.6e}, work {work:.6e}, norm {:.1e}",
        run.worst_norm_error
    );
    assert!(change.abs() > 1e-6);
    assert!(
        (change - work).abs() < 1e-3 * change.abs(),
        "{change:e} vs {work:e}"
    );
}

#[test]
#[ignore = "R9b, slow: two ramps on LiH (~1 min)"]
fn a_static_vector_potential_leaves_the_band_sum_deficit() {
    // R9b for a density-only kernel (LDA): after an adiabatic ramp to a static A the late-time
    // current is -A (N - N_dia) / Omega with the band sum's N_dia (TDDFT_PLAN.md 13.5.3).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let strength = 1e-3;
    let mut request = Request::new([1., 0., 0.], strength, 0.2, 1000);
    request.field = Field::Ramp { duration: 100. };
    let plus = propagate(&sampler, &request).unwrap();
    let mut other = request.clone();
    other.strength = -strength;
    let minus = propagate(&sampler, &other).unwrap();
    let late: Vec<f64> = plus
        .along
        .iter()
        .zip(&minus.along)
        .zip(&plus.times)
        .filter(|(_, t)| **t >= 120.)
        .map(|((a, b), _)| 0.5 * (a - b))
        .collect();
    let mean = late.iter().sum::<f64>() / late.len() as f64;
    let ramp = plus.electrons + plus.volume * mean / strength;
    eprintln!(
        "LiH LDA: N {}, band-sum N_dia {:.6}, ramp N_dia {ramp:.6}",
        plus.electrons, plus.diamagnetic
    );
    assert!((ramp - plus.diamagnetic).abs() < 2e-2 * plus.diamagnetic);
}

#[test]
fn the_current_does_not_depend_on_the_layout() {
    // R14: kpts lanes and threads change no bit of j(t).
    let cell = lih("lda-sr");
    let mut traces = vec![];
    for (threads, kpts) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let ctx = Context::new(Some(threads), 1 << 30)
            .unwrap()
            .with_layout(kpts, 1)
            .unwrap();
        let solved = solve_on(&cell, "raw", "SZ", ctx);
        let sampler = solved.sampler();
        let run = propagate(&sampler, &Request::new([1., 0., 0.], 0.01, 0.2, 10)).unwrap();
        traces.push(run.current);
    }
    for trace in &traces[1..] {
        assert_eq!(trace, &traces[0]);
    }
}

/// Silicon (fcc, 10.26 Bohr) solved on a Gamma-centered 2^3 mesh reduced by its symmetry.
fn symmetric_silicon(
    ctx: &Context,
) -> (Arc<mandacaru_core::periodic::setup::PeriodicSetup>, Solved) {
    let a = 10.26 / 2.;
    let datasets = vec![Arc::new(setup_shells_support::dataset(ctx, "lda-sr", "Si"))];
    let input = SetupInput {
        mesh: Some(([2, 2, 2], true)),
        recipes: None,
        projector_basis: "raw".into(),
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid_shape: [16, 16, 16],
        atoms: [[0.; 3], [0.25; 3]]
            .iter()
            .map(|f| AtomSpec {
                species: 0,
                symbol: "Si".into(),
                fractional: *f,
                moment: 0.,
            })
            .collect(),
        kpoints_frac: vec![],
        weights: vec![],
        functional: "lda".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: false,
        symmetry: true,
        filter_cutoff: None,
        sigma: None,
        dataset_files: None,
        hubbard_u: None,
        charge: None,
    };
    let setup = Arc::new(create_setup(ctx, input, datasets).unwrap());
    let crystal = Arc::new(setup.build(ctx).unwrap());
    let state = Arc::new(
        crystal
            .solve(&Options {
                tolerance: 1e-11,
                density_tolerance: 1e-9,
                width: 1e-3,
                ..Options::default()
            })
            .unwrap(),
    );
    assert!(state.converged);
    (
        setup.clone(),
        Solved {
            ctx: ctx.clone(),
            setup,
            crystal,
            state,
        },
    )
}

#[test]
fn the_whole_mesh_start_of_a_reduced_scf_is_the_scf_state() {
    // R15: the SCF used its operations (fewer k-points than the mesh); the propagation's own
    // whole-mesh start has the SCF's density and energy, and carries no current.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (setup, solved) = symmetric_silicon(&ctx);
    assert!(setup.kpoints_frac.len() < 8, "the SCF mesh was not reduced");
    let sampler = solved.sampler();
    let request = Request::new([0., 0., 1.], 0.01, 0.2, 2);
    let prepared = prepare(&sampler, &request).unwrap();
    assert_eq!(prepared.crystal.points.len(), 8);
    let peak = solved
        .state
        .density
        .as_slice()
        .iter()
        .fold(0f64, |t, v| t.max(v.abs()));
    eprintln!(
        "Si: {} SCF k-points, density change {:.2e} of {peak:.3e}, energy {:.10} vs SCF {:.10}, \
         stationarity {:.1e}",
        setup.kpoints_frac.len(),
        prepared.initial_density_change,
        prepared.ground_energy,
        solved.state.energy,
        prepared.stationarity
    );
    assert!(prepared.initial_density_change < 1e-6 * peak);
    assert!((prepared.ground_energy - solved.state.energy).abs() < 1e-7);
    let run = prepared.propagate(&sampler, &request).unwrap();
    let paramagnetic = run.current[0][2] + 0.01 * run.electrons / run.volume;
    assert!(
        paramagnetic.abs() < 1e-9,
        "the start carries a current {paramagnetic:e}"
    );
}

#[test]
fn unsupported_requests_are_refused_by_name() {
    assert!(Kernel::parse("rpa")
        .unwrap_err()
        .to_string()
        .contains("hartree"));
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let zero = Request::new([1., 0., 0.], 0., 0.2, 1);
    assert!(propagate(&sampler, &zero)
        .unwrap_err()
        .to_string()
        .contains("strength"));
    let fast = Request::new([1., 0., 0.], 0.01, 50., 1);
    assert!(propagate(&sampler, &fast)
        .unwrap_err()
        .to_string()
        .contains("time_step"));
}

#[test]
fn a_spin_polarized_closed_shell_and_pbe_propagate() {
    // A closed shell run spin-polarized (two equal channels) carries the unpolarized current, and
    // PBE (pbe-sr LiH) conserves its energy after a kick like LDA.
    let request = Request::new([1., 0., 0.], 0.01, 0.2, 20);
    let plain = solve(&lih("lda-sr"));
    let mut polarized = lih("lda-sr");
    polarized.spin = true;
    let polarized = solve(&polarized);
    let a = propagate(&plain.sampler(), &request).unwrap();
    let b = propagate(&polarized.sampler(), &request).unwrap();
    let scale = a.along.iter().fold(0f64, |t, v| t.max(v.abs()));
    let difference = a
        .along
        .iter()
        .zip(&b.along)
        .map(|(x, y)| (x - y).abs())
        .fold(0f64, f64::max);
    eprintln!(
        "LiH spin-polarized against unpolarized: |j| {scale:.4e}, difference {difference:.2e}, \
         N_dia {:.8} / {:.8}",
        a.diamagnetic, b.diamagnetic
    );
    assert!(difference < 1e-8 * scale, "{difference:e}");
    assert!((a.diamagnetic - b.diamagnetic).abs() < 1e-8);
    let mut pbe = lih("pbe-sr");
    pbe.functional = "pbe";
    let pbe = solve(&pbe);
    let run = propagate(&pbe.sampler(), &request).unwrap();
    eprintln!(
        "LiH PBE: drift {:.3e} kick energies, norm {:.1e}, N_dia {:.6}",
        run.energy_drift, run.worst_norm_error, run.diamagnetic
    );
    assert!(run.energy_drift < 1e-3 && run.worst_norm_error < 1e-10);
}

#[test]
fn the_independent_particle_spectrum_obeys_kramers_kronig_and_the_f_sum() {
    // R11 on the independent-particle run (LiH): on the line z = omega + i gamma, f = epsilon - 1
    // is analytic above and real-time symmetric, so Re f(omega) = (2 / pi) P int omega' Im f /
    // (omega'^2 - omega^2), and int omega Im epsilon d omega = 2 pi^2 N_dia / Omega for the
    // corrected current (whatever gamma), 2 pi^2 N / Omega for the raw one (its Drude term).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let strength = 1e-3;
    let mut request = Request::new([1., 0., 0.], strength, 0.1, 6000);
    request.kernel = Kernel::Independent;
    let (corrected, run) = odd_current(&sampler, &request);
    let mut minus = request.clone();
    minus.strength = -strength;
    let other = propagate(&sampler, &minus).unwrap();
    let raw: Vec<f64> = run
        .along
        .iter()
        .zip(&other.along)
        .map(|(a, b)| 0.5 * (a - b))
        .collect();
    let (damping, step, top) = (0.05, 0.004, 24.);
    let frequencies: Vec<f64> = (1..=(top / step) as usize)
        .map(|i| i as f64 * step)
        .collect();
    let epsilon = |signal: &[f64]| {
        dielectric_function(
            &solved.ctx,
            signal,
            request.time_step,
            strength,
            &frequencies,
            damping,
            Window::Exponential,
        )
        .unwrap()
        .values
        .as_slice()
        .to_vec()
    };
    let (good, drude) = (epsilon(&corrected), epsilon(&raw));
    let moment = |values: &[C64]| -> f64 {
        frequencies
            .iter()
            .zip(values)
            .map(|(w, e)| w * e.im * step)
            .sum()
    };
    let pi2 = 2. * std::f64::consts::PI.powi(2) / run.volume;
    let (sum_good, sum_drude) = (moment(&good), moment(&drude));
    eprintln!(
        "f-sum: corrected {sum_good:.6e} against 2 pi^2 N_dia / Omega {:.6e}; raw {sum_drude:.6e} \
         against 2 pi^2 N / Omega {:.6e}",
        pi2 * run.diamagnetic,
        pi2 * run.electrons
    );
    assert!((sum_good / (pi2 * run.diamagnetic) - 1.).abs() < 1e-2);
    assert!((sum_drude / (pi2 * run.electrons) - 1.).abs() < 1e-2);
    // Kramers-Kronig at the midpoints between frequency samples (a principal value by symmetry).
    let mut worst = 0f64;
    let mut largest = 0f64;
    for i in (100..frequencies.len() / 4).step_by(50) {
        let omega = frequencies[i] + 0.5 * step;
        let direct = 0.5 * (good[i].re + good[i + 1].re) - 1.;
        let transformed: f64 = frequencies
            .iter()
            .zip(&good)
            .map(|(w, e)| 2. / std::f64::consts::PI * w * e.im / (w * w - omega * omega) * step)
            .sum();
        largest = largest.max(direct.abs());
        worst = worst.max((direct - transformed).abs());
    }
    eprintln!(
        "Kramers-Kronig: largest |Re epsilon - 1| {largest:.4e}, worst difference {worst:.2e}"
    );
    assert!(worst < 2e-2 * largest, "{worst:e} against {largest:e}");
}

#[test]
#[ignore = "slow: three kernels, kick pairs on LiH, 1,000 steps each (about 1.5 min)"]
fn the_three_kernels_conserve_their_energies_and_order_the_static_response() {
    // TD-Hartree (the XC potential frozen, R13's kernel) is the derivative of its own linearized
    // energy, so a kick conserves it as the adiabatic and the frozen kernels do. The low-frequency
    // epsilon_1 of each kernel is printed (R13 compares the Hartree one with the cRPA head; a
    // diagnostic, not a gate).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    for kernel in [Kernel::Independent, Kernel::Hartree, Kernel::Adiabatic] {
        let mut request = Request::new([1., 0., 0.], 1e-3, 0.2, 1000);
        request.kernel = kernel;
        let (odd, run) = odd_current(&sampler, &request);
        let epsilon = dielectric_function(
            &solved.ctx,
            &odd,
            request.time_step,
            request.strength,
            &[0.02],
            0.05,
            Window::Exponential,
        )
        .unwrap();
        eprintln!(
            "{}: drift {:.3e} kick energies, epsilon(0.02 + 0.05 i) = {:.6}",
            kernel.name(),
            run.energy_drift,
            epsilon.values.as_slice()[0]
        );
        assert!(
            run.energy_drift < 1e-3,
            "{}: {:e}",
            kernel.name(),
            run.energy_drift
        );
    }
}

#[test]
fn the_moved_basis_with_a_frozen_hamiltonian_is_exact() {
    // R8 for F2: the transferred bands evolve field-free under the frozen H_o at k + A, so the
    // current equals the exact spectral evolution there (no A in H, none in j), at any dt.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    for (time_step, steps) in [(0.2, 30), (0.6, 10)] {
        let mut request = Request::new([1., 0.3, 0.], 0.05, time_step, steps);
        request.kernel = Kernel::Independent;
        request.formulation = Formulation::Moved;
        let prepared = prepare(&sampler, &request).unwrap();
        let run = prepared.propagate(&sampler, &request).unwrap();
        let exact = exact_current(&prepared, 0., &run.times);
        let scale = exact
            .iter()
            .flat_map(|j| j.iter())
            .fold(0f64, |t, v| t.max(v.abs()));
        let error = run
            .current
            .iter()
            .zip(&exact)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0f64, f64::max);
        eprintln!(
            "F2 dt {time_step}: largest |j| {scale:.4e}, error {error:.2e}, transfer defect \
             {:.2e}, static current {:.3e}",
            run.transfer_defect, run.static_current
        );
        assert!(scale > 1e-5 && error < 1e-12 * scale.max(1.), "{error:e}");
    }
}

#[test]
#[ignore = "slow: F1 and F2 adiabatic kick pairs on LiH, 1,500 steps each (about 2 min)"]
fn the_moved_basis_conserves_energy_and_shares_the_fixed_poles() {
    // F2 with the adiabatic kernel: the energy after the transfer is conserved; its spectrum has
    // the poles of F1 (one Hamiltonian, two bases) and the length-form weights, and its f-sum is
    // printed against F1's N_dia and N.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let strength = 1e-3;
    let frequencies: Vec<f64> = (1..=400).map(|i| 0.005 * i as f64).collect();
    let mut spectra = vec![];
    for formulation in [Formulation::Fixed, Formulation::Moved] {
        let mut request = Request::new([1., 0., 0.], strength, 0.2, 1500);
        request.formulation = formulation;
        let (odd, run) = odd_current(&sampler, &request);
        let epsilon = dielectric_function(
            &solved.ctx,
            &odd,
            request.time_step,
            strength,
            &frequencies,
            0.02,
            Window::Exponential,
        )
        .unwrap();
        let values: Vec<f64> = epsilon.values.as_slice().iter().map(|e| e.im).collect();
        let peak = (0..values.len())
            .max_by(|a, b| values[*a].total_cmp(&values[*b]))
            .unwrap();
        let moment: f64 = frequencies
            .iter()
            .zip(&values)
            .map(|(w, e)| w * e * 0.005)
            .sum::<f64>()
            * run.volume
            / (2. * std::f64::consts::PI.powi(2));
        eprintln!(
            "{}: drift {:.3e} kick energies, kick energy {:.4e}, peak of Im epsilon at {:.3} Ha \
             ({:.4}), f-sum to 2 Ha {moment:.4} (N_dia {:.4}, N {})",
            formulation.name(),
            run.energy_drift,
            run.kick_energy,
            frequencies[peak],
            values[peak],
            run.diamagnetic,
            run.electrons
        );
        assert!(run.energy_drift < 1e-2, "{:e}", run.energy_drift);
        spectra.push(frequencies[peak]);
    }
    assert!((spectra[0] - spectra[1]).abs() < 0.011, "{spectra:?}");
}

/// Silicon (fcc, 10.26 Bohr, two atoms) on 20^3 points and a 4^3 whole mesh.
fn silicon() -> Cell {
    let a = 10.26 / 2.;
    Cell {
        library: "lda-sr",
        symbols: vec!["Si", "Si"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid: [20, 20, 20],
        fractional: vec![[0.; 3], [0.25; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [4, 4, 4],
        hubbard: None,
        width: 1e-3,
    }
}

#[test]
#[ignore = "F1/F2 default decision, slow: Si 4^3 at DZP and QZ2P, both formulations, independent particles"]
fn the_corrected_current_of_both_formulations_against_the_converged_basis() {
    // The comparison Q20's default turns on (DECISIONS.md 2026-11-15): the same observable, the
    // corrected current's f-sum N_eff = Omega / (2 pi^2) int omega Im epsilon and the static
    // Re epsilon(0.02 + 0.05 i Ha), of F1 and F2 at the production and the converged basis.
    let strength = 1e-3;
    let frequencies: Vec<f64> = (1..=1000).map(|i| 0.004 * i as f64).collect();
    // A narrow smearing keeps the QZ2P occupations integral (its 4^3 gap is small).
    let mut cell = silicon();
    cell.width = 1e-4;
    let only = std::env::var("SIZES").ok();
    // Measured 2026-11-15 (N_eff, Re epsilon) for F1 and F2. Their f-sums agree at DZP only by
    // cancellation in a signed sum (vera: line by line they differ by 0.256 at DZP and 0.118 at
    // QZ2P), and the velocity (F1), mixed (F2) and length sums cross between the bases (DZP
    // 9.842 / 9.841 / 10.096, QZ2P 10.482 / 10.194 / 10.024), so nothing here says which form is
    // closer to the complete basis. F1 is the default on operational grounds (positive weights;
    // static fields, DFT+U and meta-GGAs on a -k closed mesh), not on accuracy (DECISIONS.md).
    let pinned = [
        ("DZP", [(9.68479, 19.56355), (9.68400, 19.66706)]),
        ("QZ2P", [(10.25933, 21.69526), (9.99427, 21.25444)]),
    ];
    let mut found = vec![];
    for (size, expected) in pinned {
        if only
            .as_deref()
            .is_some_and(|o| !o.split(',').any(|s| s == size))
        {
            continue;
        }
        let solved = solve_on(&cell, "raw", size, Context::new(Some(2), 3 << 29).unwrap());
        let sampler = solved.sampler();
        for (formulation, (n_eff, static_epsilon)) in [Formulation::Fixed, Formulation::Moved]
            .into_iter()
            .zip(expected)
        {
            let mut request = Request::new([1., 0., 0.], strength, 0.1, 3000);
            request.kernel = Kernel::Independent;
            request.formulation = formulation;
            let (odd, run) = odd_current(&sampler, &request);
            let epsilon = |grid: &[f64]| {
                dielectric_function(
                    &solved.ctx,
                    &odd,
                    request.time_step,
                    strength,
                    grid,
                    0.05,
                    Window::Exponential,
                )
                .unwrap()
                .values
                .as_slice()
                .to_vec()
            };
            let moment: f64 = frequencies
                .iter()
                .zip(epsilon(&frequencies))
                .map(|(w, e)| w * e.im * 0.004)
                .sum::<f64>()
                * run.volume
                / (2. * std::f64::consts::PI.powi(2));
            let low = epsilon(&[0.02])[0];
            eprintln!(
                "Si {size} {}: N_eff {moment:.5}, Re epsilon(0.02 + 0.05 i) {:.5}, band-sum N_dia \
                 {:.5}, dt W {:.3}, static current {:.3e}",
                formulation.name(),
                low.re,
                run.diamagnetic,
                run.step_width,
                run.static_current
            );
            assert!((moment - n_eff).abs() < 1e-4 * n_eff, "{size}: {moment}");
            assert!(
                (low.re - static_epsilon).abs() < 1e-4 * static_epsilon,
                "{size}: {low}"
            );
            found.push((size, moment));
        }
    }
    for pair in found.chunks(2).filter(|p| p[0].0 == "DZP") {
        assert!((pair[0].1 - pair[1].1).abs() < 1e-3 * pair[0].1, "{pair:?}");
    }
}

/// The largest difference of two traces relative to the largest entry of the second.
fn relative_difference(a: &[f64], b: &[f64]) -> (f64, f64) {
    let scale = b.iter().fold(0f64, |t, v| t.max(v.abs()));
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f64, f64::max);
    (error / scale, scale)
}

#[test]
fn silicon_holds_the_frozen_identity_the_band_sum_series_and_the_power_balance() {
    // R8, R10 and R2 on Si: SZ, 16^3 points, the 2^3 whole mesh rebuilt from a symmetry-reduced
    // SCF (the R15 fixture), so the crystal that fails Q20's basis bound passes the identities.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (_, solved) = symmetric_silicon(&ctx);
    let sampler = solved.sampler();
    // R8: the frozen Hamiltonian is the exact spectral evolution at a large field.
    let mut request = Request::new([1., 0.3, 0.], 0.05, 0.4, 15);
    request.kernel = Kernel::Independent;
    let prepared = prepare(&sampler, &request).unwrap();
    let run = prepared.propagate(&sampler, &request).unwrap();
    let exact = exact_current(&prepared, request.strength, &run.times);
    let flat = |rows: &[[f64; 3]]| rows.iter().flatten().copied().collect::<Vec<f64>>();
    let (r8, scale) = relative_difference(&flat(&run.current), &flat(&exact));
    // R10: the odd independent-particle current is the band sum's cosine series.
    let lines = transitions(
        &sampler,
        &MeshIndex::new([2, 2, 2], [0.; 3]).unwrap(),
        [1., 0., 0.],
    )
    .unwrap();
    let strength = 1e-3;
    let mut request = Request::new([1., 0., 0.], strength, 0.2, 300);
    request.kernel = Kernel::Independent;
    let (odd, kicked) = odd_current(&sampler, &request);
    let series: Vec<f64> = kicked
        .times
        .iter()
        .map(|t| {
            -strength / kicked.volume
                * lines
                    .weights
                    .iter()
                    .zip(&lines.gaps)
                    .map(|(w, gap)| 2. * lines.degeneracy * w / gap * (gap * t).cos())
                    .sum::<f64>()
        })
        .collect();
    let (r10, _) = relative_difference(&odd, &series);
    // R2: the adiabatic kick conserves the energy as dt^2, and a ramp's work is the energy change.
    let drift = |time_step: f64, steps: usize| {
        let run = propagate(
            &sampler,
            &Request::new([0., 0., 1.], 0.01, time_step, steps),
        )
        .unwrap();
        assert!(run.worst_norm_error < 1e-8);
        run.energy_drift
    };
    let drifts = [drift(0.4, 25), drift(0.2, 50)];
    let mut ramp = Request::new([1., 1., 0.], 0.02, 0.1, 200);
    ramp.field = Field::Ramp { duration: 20. };
    let ramped = propagate(&sampler, &ramp).unwrap();
    let e = ramped.times.len() - 1;
    let work: f64 = (0..e)
        .map(|i| {
            let power = |i: usize| {
                -ramped.volume * ramped.along[i] * ramp.strength * ramp.field.rate(ramped.times[i])
            };
            0.5 * (power(i) + power(i + 1)) * ramp.time_step
        })
        .sum();
    let change = ramped.energy[e] - ramped.energy[0];
    eprintln!(
        "Si: R8 {r8:.2e} of |j| {scale:.3e}; R10 {r10:.2e} (N_dia {:.6}, N {}); drift {:.3e} / \
         {:.3e} kick energies; ramp energy change {change:.6e}, work {work:.6e}",
        kicked.diamagnetic, kicked.electrons, drifts[0], drifts[1]
    );
    assert!(r8 < 1e-12, "R8 {r8:e}");
    assert!(r10 < 2e-3, "R10 {r10:e}");
    assert!(drifts[1] < drifts[0] / 2.5, "{drifts:?}");
    assert!(
        (change - work).abs() < 1e-3 * change.abs(),
        "{change:e} vs {work:e}"
    );
}

#[test]
fn three_axes_give_the_cubic_tensor_of_silicon() {
    // direction="all": the runs along x, y and z share one start; on cubic Si the diagonal of the
    // corrected-current tensor is isotropic and the off-diagonal vanishes, and the column along x
    // is the single run along x bit for bit.
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (_, solved) = symmetric_silicon(&ctx);
    let sampler = solved.sampler();
    let mut request = Request::new([1., 0., 0.], 1e-3, 0.2, 60);
    request.kernel = Kernel::Independent;
    let runs = propagate_axes(&sampler, &request).unwrap();
    let single = propagate(&sampler, &request).unwrap();
    assert_eq!(runs[0].corrected_current, single.corrected_current);
    let column = |b: usize, a: usize| -> Vec<f64> {
        runs[b].corrected_current.iter().map(|j| j[a]).collect()
    };
    let (mut diagonal, mut off) = (0f64, 0f64);
    let reference = column(0, 0);
    for b in 0..3 {
        for a in 0..3 {
            let (difference, scale) = relative_difference(&column(b, a), &reference);
            assert!(scale > 1e-8);
            if a == b {
                diagonal = diagonal.max(difference);
            } else {
                // |j_ab| relative to the largest |j_xx|.
                let largest = column(b, a).iter().fold(0f64, |t, v| t.max(v.abs()));
                off = off.max(largest / scale);
            }
        }
        assert!((runs[b].diamagnetic_vector[b] - runs[b].diamagnetic).abs() < 1e-12);
    }
    eprintln!(
        "Si tensor: diagonal spread {diagonal:.2e}, off-diagonal {off:.2e} (relative to j_xx); \
         N_dia {:.6} {:.6} {:.6}",
        runs[0].diamagnetic, runs[1].diamagnetic, runs[2].diamagnetic
    );
    assert!(diagonal < 1e-8 && off < 1e-8, "{diagonal:e} {off:e}");
}

/// O2 (2.28 Bohr along x) in an 8 Bohr cube at Gamma, spin-polarized from moments of one per
/// atom: the triplet, 7 up and 5 down valence electrons, a gap in each channel.
fn oxygen_triplet() -> Cell {
    Cell {
        library: "lda-sr",
        symbols: vec!["O", "O"],
        lattice: Matrix3::from_diagonal(&Vector3::new(8., 8., 8.)),
        grid: [24, 24, 24],
        fractional: vec![[0.5 - 1.14 / 8., 0.5, 0.5], [0.5 + 1.14 / 8., 0.5, 0.5]],
        moments: vec![1., 1.],
        spin: true,
        spinor: false,
        functional: "lda",
        mesh: [1, 1, 1],
        hubbard: None,
        width: 1e-3,
    }
}

#[test]
fn a_magnetic_crystal_sums_its_unequal_spin_channels() {
    // A spin-polarized crystal with a moment of two: N_dia = N_dia,up + N_dia,down (vera fix 5),
    // the frozen-Hamiltonian identity R8 per channel, and the adiabatic energy after a kick.
    let solved = solve(&oxygen_triplet());
    let sampler = solved.sampler();
    let lines = transitions(
        &sampler,
        &MeshIndex::new([1, 1, 1], [0.; 3]).unwrap(),
        [1., 0., 0.],
    )
    .unwrap();
    let band = lines.diamagnetic(2);
    let mut request = Request::new([1., 0., 0.], 0.02, 0.1, 20);
    request.kernel = Kernel::Independent;
    let prepared = prepare(&sampler, &request).unwrap();
    let occupied: Vec<usize> = prepared
        .channels
        .iter()
        .map(|c| c.occupations.len())
        .collect();
    let run = prepared.propagate(&sampler, &request).unwrap();
    let exact = exact_current(&prepared, request.strength, &run.times);
    let flat = |rows: &[[f64; 3]]| rows.iter().flatten().copied().collect::<Vec<f64>>();
    let (r8, _) = relative_difference(&flat(&run.current), &flat(&exact));
    let kicked = propagate(&sampler, &Request::new([1., 0., 0.], 0.01, 0.1, 40)).unwrap();
    eprintln!(
        "O2 triplet: occupied {occupied:?}, N_dia,s {:?}, total {:.6} (run {:.6}), R8 {r8:.2e}, \
         adiabatic drift {:.3e} kick energies",
        band.channels, band.total, run.diamagnetic, kicked.energy_drift
    );
    assert_eq!(occupied, vec![7, 5]);
    assert!((band.channels[0] - band.channels[1]).abs() > 1e-3 * band.total);
    assert!((band.total - band.channels.iter().sum::<f64>()).abs() < 1e-12);
    assert!((run.diamagnetic - band.total).abs() < 1e-10);
    assert!(r8 < 1e-12, "R8 {r8:e}");
    assert!(kicked.energy_drift < 1e-2 && kicked.worst_norm_error < 1e-8);
}

#[test]
#[ignore = "slow: the static and ramp routes on LiH LDA, then DFT+U on Zn + H2 with and without U (about 3 min)"]
fn a_hubbard_crystal_corrects_its_current_with_the_static_field_response() {
    // Q21a for DFT+U: its shell occupations respond to Im D, so N_dia is measured by
    // self-consistency at a static field. On LDA both that route and the ramp reproduce the band
    // sum (R9b; the ramp to its own error). Zn sits off every inversion center of the fixture
    // (vera: on a center the first-order Im D of the d shell vanishes by parity, so the shift is
    // zero whatever U is); with U = 3 eV on the closed 3d shell the start is the SCF's, the kick
    // conserves the energy, the run's N_dia is the static one, and the shift from U is set
    // against the same fixture without U.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 10);
    let prepared = prepare(&sampler, &request).unwrap();
    let ramp = prepared.ramp_response(&sampler, &request).unwrap()[0];
    let fixed = prepared.static_response(&request).unwrap();
    let band = prepared.propagate(&sampler, &request).unwrap().diamagnetic;
    let lda = (fixed.vector[0] - band) / band;
    eprintln!(
        "LiH LDA: static N_dia {:.6} ({} iterations), ramp {ramp:.6}, band sum {band:.6} \
         (static {lda:+.2e}, ramp {:+.2e})",
        fixed.vector[0],
        fixed.iterations,
        (ramp - band) / band
    );
    let mut cell = Cell {
        library: "lda-sr",
        symbols: vec!["Zn", "H", "H"],
        lattice: Matrix3::from_diagonal(&Vector3::new(8., 8., 8.)),
        grid: [32, 32, 32],
        fractional: vec![
            [0.5, 0.5, 0.5],
            [0.25 - 0.0875, 0.28, 0.31],
            [0.25 + 0.0875, 0.28, 0.31],
        ],
        moments: vec![0.; 3],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [1, 1, 1],
        hubbard: Some(r#"{"Zn": {"l": "d", "U": 3.0}}"#),
        width: 1e-3,
    };
    let solved = solve(&cell);
    assert!(solved.state.hubbard_energy.is_some(), "the +U term is on");
    let sampler = solved.sampler();
    let shell: f64 = {
        let report = solved.state.hubbard.as_ref().unwrap();
        let n = &report.occupations[0];
        let p = (n.len() as f64).sqrt().round() as usize;
        (0..p).map(|i| n[i * p + i].re).sum()
    };
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 100);
    let prepared = prepare(&sampler, &request).unwrap();
    let run = prepared.propagate(&sampler, &request).unwrap();
    let mut hartree = request.clone();
    hartree.kernel = Kernel::Hartree;
    let refused = propagate(&sampler, &hartree).unwrap_err().to_string();
    // The same fixture without U: the static route against its own band sum.
    cell.hubbard = None;
    let plain = solve(&cell);
    let plain_sampler = plain.sampler();
    let plain_prepared = prepare(&plain_sampler, &request).unwrap();
    let plain_static = plain_prepared.static_response(&request).unwrap().vector[0];
    let plain_band = plain_prepared
        .propagate(&plain_sampler, &request)
        .unwrap()
        .diamagnetic;
    eprintln!(
        "Zn + H2 +U: 3d occupation {shell:.4} per spin, start energy {:.10} against the SCF \
         {:.10}, static N_dia {:.6} against the band sum {:.6} ({:+.3e}), drift {:.3e} kick \
         energies; without U: static {plain_static:.6} against the band sum {plain_band:.6} \
         ({:+.3e})",
        prepared.ground_energy,
        solved.state.energy,
        run.diamagnetic,
        run.band_sum,
        run.diamagnetic - run.band_sum,
        run.energy_drift,
        plain_static - plain_band
    );
    assert!(lda.abs() < 1e-4, "{lda:e}");
    assert!(shell > 4., "the Zn 3d shell is occupied: {shell}");
    assert!((prepared.ground_energy - solved.state.energy).abs() < 1e-7);
    assert!(run.self_consistent && run.energy_drift < 1e-2);
    // Off the inversion center U shifts the static response (vera: +2.1e-4), and without U the
    // static route is the band sum to the A^2 order of the symmetric pair.
    assert!(run.diamagnetic - run.band_sum > 1e-5);
    assert!((plain_static - plain_band).abs() < 1e-5);
    assert!(refused.contains("hartree"), "{refused}");
}

#[test]
#[ignore = "R13, a recorded diagnostic (not a gate): the Wannier H2 chain, TD-Hartree along three axes (about 1 min)"]
fn the_static_hartree_response_against_the_rpa_head() {
    // R13: epsilon_M(omega -> 0) of TD-Hartree (RPA with local fields, q = 0, velocity form)
    // against 1 / eps^-1_00 of the downfold's RPA head (q = HEAD_WAVEVECTOR along each axis,
    // averaged as eps^-1). The two differ by the basis's length/velocity defect; recorded.
    use mandacaru_core::periodic::{
        bloch_mesh::Sampler,
        wannier::downfold::{Request as Downfold, Screening},
    };
    let (_ctx, wannier) = wannier_downfold_support::chain([3, 1, 1]);
    let head = wannier
        .downfold(&Downfold {
            screening: Some(Screening::Rpa),
            ..wannier_downfold_support::request(Some(vec![(0, [0; 3])]))
        })
        .unwrap()
        .problem
        .dielectric_head
        .unwrap();
    let sampler = Sampler::new(&wannier.crystal, &wannier.state, &wannier.setup).unwrap();
    let damping = 0.02;
    let mut inverse = 0.;
    let mut diagonal = vec![];
    for axis in 0..3 {
        let mut request = Request::new(
            std::array::from_fn(|i| if i == axis { 1. } else { 0. }),
            1e-3,
            0.2,
            1500,
        );
        request.kernel = Kernel::Hartree;
        let (odd, _) = odd_current(&sampler, &request);
        let epsilon = dielectric_function(
            &wannier.crystal.context,
            &odd,
            request.time_step,
            request.strength,
            &[0.],
            damping,
            Window::Exponential,
        )
        .unwrap()
        .values
        .as_slice()[0];
        inverse += 1. / epsilon.re / 3.;
        diagonal.push(epsilon.re);
    }
    eprintln!(
        "H2 chain: TD-Hartree Re epsilon(0 + {damping} i) {diagonal:.5?}, mean 1/epsilon \
         {inverse:.5}; RPA head eps^-1_00 {head:.5} (1/head {:.5}); relative difference {:+.3e}",
        1. / head,
        (inverse - head) / head
    );
    assert!(head > 0. && head <= 1.);
    assert!(diagonal.iter().all(|e| *e > 1.));
}

#[test]
fn a_meta_gga_crystal_propagates_with_the_gauge_invariant_kinetic_density() {
    // T6c-c: r2SCAN on LiH with tau~ = tau - |j|^2 / (2 n) from the Bloch gradients. The start is
    // the SCF's (j = 0 there, so tau~ = tau), and the Fock matrix is the derivative of the energy
    // only with the current terms (v_tau |j|^2 / (2 n^2) and F^j): the drift after a kick then
    // falls as dt^2.
    let mut cell = lih("lda-sr");
    cell.functional = "r2scan";
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let mut drifts = vec![];
    for (time_step, steps) in [(0.2, 40), (0.1, 80)] {
        let mut request = Request::new([1., 0.3, 0.], 0.02, time_step, steps);
        // The energy does not read the correction: skip the ramp that would measure it.
        request.statics = Some([0.; 3]);
        let prepared = prepare(&sampler, &request).unwrap();
        assert!((prepared.ground_energy - solved.state.energy).abs() < 1e-7);
        let run = prepared.propagate(&sampler, &request).unwrap();
        eprintln!(
            "LiH r2SCAN dt {time_step}: start {:.10} against the SCF {:.10}, kick energy {:.4e}, \
             drift {:.3e} kick energies, norm {:.1e}",
            prepared.ground_energy,
            solved.state.energy,
            run.kick_energy,
            run.energy_drift,
            run.worst_norm_error
        );
        assert!(run.worst_norm_error < 1e-8);
        drifts.push(run.energy_drift);
    }
    assert!(drifts[1] < drifts[0] / 2.5, "{drifts:?}");
    let mut hartree = Request::new([1., 0., 0.], 0.01, 0.2, 2);
    hartree.kernel = Kernel::Hartree;
    assert!(propagate(&sampler, &hartree)
        .unwrap_err()
        .to_string()
        .contains("meta-GGA"));
}

#[test]
#[ignore = "R16(c) and R9b with r2SCAN, recorded: the static field and two ramps on LiH (about 1 min)"]
fn the_static_response_of_a_meta_gga_crystal() {
    // R9b with r2SCAN: the current term of tau~ responds to Im D, so the static response is not
    // the band sum of the same bands (measured +8.6 % on LiH, toward N: the boost invariance of
    // E_xc[tau~]); the run takes the static-field N_dia, and the ramp cross-checks it (vera: the
    // ramp is right to 2e-4).
    let mut cell = lih("lda-sr");
    cell.functional = "r2scan";
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 10);
    let prepared = prepare(&sampler, &request).unwrap();
    let ramp = prepared.ramp_response(&sampler, &request).unwrap()[0];
    let fixed = prepared.static_response(&request).unwrap();
    let run = prepared.propagate(&sampler, &request).unwrap();
    eprintln!(
        "LiH r2SCAN: N {}, static N_dia {:.6} ({} iterations), ramp {ramp:.6} ({:+.2e}), band \
         sum {:.6} ({:+.3e})",
        run.electrons,
        fixed.vector[0],
        fixed.iterations,
        (ramp - fixed.vector[0]) / fixed.vector[0],
        run.band_sum,
        (fixed.vector[0] - run.band_sum) / run.band_sum
    );
    assert!(run.self_consistent && (run.diamagnetic - fixed.vector[0]).abs() < 1e-12);
    assert!((ramp - fixed.vector[0]).abs() < 1e-3 * fixed.vector[0]);
    assert!(fixed.vector[0] > run.band_sum && fixed.vector[0] < run.electrons);
}

#[test]
#[ignore = "R9b on Si with r2SCAN, recorded: the static field against the ramp and the band sum, and direction=\"all\" (SZ, 2^3; several minutes)"]
fn silicon_r2scan_takes_its_static_response_by_self_consistency() {
    // The static self-consistency on a covalent crystal (vera measured it on LiH only), its
    // cross-check by the ramp, and r2SCAN with direction="all": each axis measures its own static
    // response, and on cubic Si the three are equal.
    let mut cell = silicon();
    cell.functional = "r2scan";
    cell.mesh = [2, 2, 2];
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 10);
    let prepared = prepare(&sampler, &request).unwrap();
    let started = std::time::Instant::now();
    let fixed = prepared.static_response(&request).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let ramp = prepared.ramp_response(&sampler, &request).unwrap()[0];
    let ramp_seconds = started.elapsed().as_secs_f64();
    let runs = propagate_axes(&sampler, &request).unwrap();
    let band = runs[0].band_sum;
    eprintln!(
        "Si r2SCAN SZ 2^3: static N_dia {:.6} ({} iterations, {seconds:.0} s), ramp {ramp:.6} \
         ({:+.2e}, {ramp_seconds:.0} s), band sum {band:.6} ({:+.3e}); all: {:?}",
        fixed.vector[0],
        fixed.iterations,
        (ramp - fixed.vector[0]) / fixed.vector[0],
        (fixed.vector[0] - band) / band,
        runs.iter().map(|r| r.diamagnetic).collect::<Vec<_>>()
    );
    // Measured 2026-11-15: static 10.502205 in 10 iterations (3 s), the ramp 10.462611 (-3.8e-3,
    // 895 s): on Si the ramp's own error is ten times LiH's, so it is a loose cross-check here.
    assert!((ramp - fixed.vector[0]).abs() < 1e-2 * fixed.vector[0]);
    assert!(runs.iter().all(|r| r.self_consistent));
    assert!((runs[0].diamagnetic - fixed.vector[0]).abs() < 1e-10);
    for run in &runs[1..] {
        assert!((run.diamagnetic - runs[0].diamagnetic).abs() < 1e-6 * runs[0].diamagnetic);
    }
}

#[test]
#[ignore = "spin-polarized r2SCAN and DFT+U crystals, recorded: LiH r2SCAN and Zn + H2 +U, each against its unpolarized twin (about 3 min)"]
fn spin_polarized_meta_gga_and_hubbard_crystals_match_their_unpolarized_twins() {
    // The two-channel branches of tau~ (`gauge_invariant_kinetic`) and of the +U shell
    // occupations: a closed shell run spin-polarized has two equal channels, so its static
    // response, its corrected current and its energy conservation are the unpolarized ones.
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 40);
    let mut meta = lih("lda-sr");
    meta.functional = "r2scan";
    let zinc = Cell {
        library: "lda-sr",
        symbols: vec!["Zn", "H", "H"],
        lattice: Matrix3::from_diagonal(&Vector3::new(8., 8., 8.)),
        grid: [32, 32, 32],
        fractional: vec![
            [0.5, 0.5, 0.5],
            [0.25 - 0.0875, 0.28, 0.31],
            [0.25 + 0.0875, 0.28, 0.31],
        ],
        moments: vec![0.; 3],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [1, 1, 1],
        hubbard: Some(r#"{"Zn": {"l": "d", "U": 3.0}}"#),
        width: 1e-3,
    };
    for (name, cell) in [("LiH r2SCAN", meta), ("Zn + H2 +U", zinc)] {
        let plain = solve(&cell);
        let mut polarized = cell.clone();
        polarized.spin = true;
        let polarized = solve(&polarized);
        let a = propagate(&plain.sampler(), &request).unwrap();
        let b = propagate(&polarized.sampler(), &request).unwrap();
        let scale = a.corrected.iter().fold(0f64, |t, v| t.max(v.abs()));
        let difference = a
            .corrected
            .iter()
            .zip(&b.corrected)
            .map(|(x, y)| (x - y).abs())
            .fold(0f64, f64::max);
        eprintln!(
            "{name}: spin-polarized against unpolarized: N_dia {:.8} / {:.8}, corrected current \
             difference {difference:.2e} of {scale:.3e}, drift {:.2e} / {:.2e} kick energies",
            a.diamagnetic, b.diamagnetic, a.energy_drift, b.energy_drift
        );
        assert!(a.self_consistent && b.self_consistent);
        assert!((a.diamagnetic - b.diamagnetic).abs() < 1e-6 * a.diamagnetic);
        assert!(difference < 1e-5 * scale, "{difference:e}");
        assert!(b.energy_drift < 1e-2);
    }
}

#[test]
fn the_band_sum_tensor_is_symmetric_on_a_skew_cell() {
    // The column N_dia e of the corrected current's off-diagonal part: on the skew LiH cell the
    // tensor has nonzero off-diagonal elements, and N_xy from the x column equals N_yx from y.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let column = |e: [f64; 3]| {
        transitions(&sampler, &mesh, e)
            .unwrap()
            .diamagnetic(1)
            .vector
    };
    let (x, y) = (column([1., 0., 0.]), column([0., 1., 0.]));
    eprintln!("LiH N_dia: x column {x:.8?}, y column {y:.8?}");
    assert!(x[1].abs() > 1e-4 * x[0], "{x:?}");
    assert!((x[1] - y[0]).abs() < 1e-12 * x[0], "{x:?} {y:?}");
}
