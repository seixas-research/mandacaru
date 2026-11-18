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
//!   against the RPA head (recorded); and the F1/F2 comparison that sets the default (Q20);
//! - r2SCAN with a partial core in a field: the Fock matrix and the current (with its
//!   exchange-correlation term) are the derivatives of the energy, and `tau~` keeps the core's share.
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

/// R14 with r2SCAN: the meta-GGA's `tau` and current pass, parallel over the k-points and summed in
/// k order, and its exchange-correlation current change no bit of `j(t)` across layouts.
#[test]
fn the_meta_gga_current_does_not_depend_on_the_layout() {
    let mut cell = lih("lda-sr");
    cell.functional = "r2scan";
    let mut traces = vec![];
    for (threads, kpts) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let ctx = Context::new(Some(threads), 1 << 30)
            .unwrap()
            .with_layout(kpts, 1)
            .unwrap();
        let solved = solve_on(&cell, "raw", "SZ", ctx);
        let sampler = solved.sampler();
        let mut request = Request::new([1., 0., 0.], 0.01, 0.2, 5);
        request.statics = Some([0.; 3]);
        traces.push(propagate(&sampler, &request).unwrap().current);
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
    let hartree_run = propagate(&sampler, &hartree).unwrap();
    let mut moved = Request::new([1., 0., 0.], 1e-3, 0.2, 20);
    moved.formulation = Formulation::Moved;
    let moved_run = propagate(&sampler, &moved).unwrap();
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
    assert!(hartree_run.self_consistent && hartree_run.energy_drift < 1e-2);
    assert!(moved_run.energy_drift < 1e-2);
    // Off the inversion center U shifts the static response (vera: +2.1e-4), and without U the
    // static route is the band sum to the A^2 order of the symmetric pair.
    assert!(run.diamagnetic - run.band_sum > 1e-5);
    assert!((plain_static - plain_band).abs() < 1e-5);
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

/// The energy route of the static response (R9b, `StaticResponse::energy`): at a fixed field the
/// insulator's energy is `E(0) + A^2 (N - N_dia) / 2`, flat to second order, so
/// `N - (E(+A_0) + E(-A_0) - 2 E(0)) / A_0^2` is `N_dia` with the current route's. On Si LDA (SZ,
/// 2^3), where the band sum is exact for the density-only kernel, both routes at `A_0` = 5e-3 and
/// 1e-3, extrapolated in `A_0^2`, are the band sum; the energy route's truncation is half the
/// current route's (the `A^4` term of `E` against the `A^3` of the current).
#[test]
fn the_energy_route_of_the_static_response_is_its_current_route() {
    let mut cell = silicon();
    cell.mesh = [2, 2, 2];
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let band = transitions(
        &sampler,
        &MeshIndex::new([2, 2, 2], [0.; 3]).unwrap(),
        [1., 0., 0.],
    )
    .unwrap()
    .diamagnetic(1)
    .total;
    let routes: Vec<(f64, f64)> = [5e-3, 1e-3]
        .iter()
        .map(|&strength| {
            let request = Request::new([1., 0., 0.], strength, 0.2, 10);
            let fixed = prepare(&sampler, &request)
                .unwrap()
                .static_response(&request)
                .unwrap();
            (fixed.vector[0], fixed.energy)
        })
        .collect();
    // `N(A) = N + a A^2`: from 5e-3 and 1e-3, `(25 N(1e-3) - N(5e-3)) / 24`.
    let extrapolate = |coarse: f64, fine: f64| (25. * fine - coarse) / 24.;
    let current = extrapolate(routes[0].0, routes[1].0);
    let energy = extrapolate(routes[0].1, routes[1].1);
    let truncation = [(routes[1].0 - band) / band, (routes[1].1 - band) / band];
    eprintln!(
        "Si LDA SZ 2^3: band sum {band:.8}; at 5e-3 / 1e-3 current {:.8} / {:.8}, energy {:.8} / \
         {:.8}; extrapolated current {current:.8} ({:+.2e}), energy {energy:.8} ({:+.2e}); \
         truncation at 1e-3 {:+.2e} / {:+.2e}",
        routes[0].0,
        routes[1].0,
        routes[0].1,
        routes[1].1,
        (current - band) / band,
        (energy - band) / band,
        truncation[0],
        truncation[1]
    );
    assert!(((current - band) / band).abs() < 1e-6);
    assert!(((energy - band) / band).abs() < 1e-6);
    assert!(
        (truncation[1] / truncation[0] - 0.5).abs() < 0.05,
        "{truncation:?}"
    );
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
    // The energy route (the discriminator of the static value on Si): both routes at 5e-3 and
    // 1e-3, extrapolated in A_0^2.
    let mut coarse_request = request.clone();
    coarse_request.strength = 5e-3;
    let coarse = prepared.static_response(&coarse_request).unwrap();
    let extrapolate = |coarse: f64, fine: f64| (25. * fine - coarse) / 24.;
    let (current, energy) = (
        extrapolate(coarse.vector[0], fixed.vector[0]),
        extrapolate(coarse.energy, fixed.energy),
    );
    eprintln!(
        "Si r2SCAN SZ 2^3: extrapolated current route {current:.8}, energy route {energy:.8} \
         ({:+.2e})",
        (energy - current) / current
    );
    // With `A` in `tau~` the current needs the exchange-correlation term `dE_xc / dA`: without
    // it the routes parted by +1.31e-4 (seventeenth session); with it +4.86e-8.
    assert!(((energy - current) / current).abs() < 2e-7);
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
    // Measured 2026-11-16 (`mandacaru-paw` 8a6e977), with the DIIS static iteration and `A` in
    // `tau~` (eighteenth session): static 10.509847 in 5 iterations (2 s), the energy route
    // +4.86e-8 from the current route, the ramp 10.470218 (-3.77e-3, 517 s): on Si the ramp's own
    // error is ten times LiH's, so it is a loose cross-check. Before `A` entered `tau~`: 10.502460.
    assert!((ramp - fixed.vector[0]).abs() < 1e-2 * fixed.vector[0]);
    assert!(runs.iter().all(|r| r.self_consistent));
    assert!((runs[0].diamagnetic - fixed.vector[0]).abs() < 1e-10);
    for run in &runs[1..] {
        assert!((run.diamagnetic - runs[0].diamagnetic).abs() < 1e-6 * runs[0].diamagnetic);
    }
}

/// A deterministic Hermitian `m x m` matrix (`seed` varies it), for the derivative gates.
fn hermitian(m: usize, seed: f64) -> Matrix {
    DMatrix::from_fn(m, m, |a, b| {
        let (i, j) = (a.min(b) as f64, a.max(b) as f64);
        let real = ((i + 1.) * 0.37 + (j + 1.) * 0.91 + seed).sin() / (1. + i + j);
        let imaginary = ((i + 2.) * 0.53 - (j + 1.) * 0.29 + seed).cos() / (1. + i + j);
        C64::new(
            real,
            if a < b {
                imaginary
            } else if a > b {
                -imaginary
            } else {
                0.
            },
        )
    })
}

/// `exp(i e X) D exp(-i e X)` of every channel of `prepared`'s start, `X` [`hermitian`] per
/// channel: an idempotent state that carries a current.
fn rotated_start(prepared: &Prepared, e: f64) -> Vec<Matrix> {
    prepared
        .channels
        .iter()
        .enumerate()
        .map(|(idx, channel)| {
            let m = channel.orbitals.nrows();
            let (values, vectors) = eigen(hermitian(m, 0.1 * idx as f64)).unwrap();
            let phases = Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
                m,
                values.iter().map(|v| C64::from_polar(1., e * v)),
            ));
            let u = &vectors * phases * vectors.adjoint();
            &u * channel.density() * u.adjoint()
        })
        .collect()
}

/// Item 4 on crystals (TDDFT_PLAN.md 13.6.1): with `A` in `tau~` and a partial core, the crystal's
/// Fock matrix is still the derivative of its energy, and its current `dE / dA` carries the
/// exchange-correlation term `dE_xc / dA = sum_s int v_tau,s (1 - n_v,s / n_s) j_s`. Si r2SCAN
/// (`lda-sr`, silicon's partial core; SZ on the 2^3 whole mesh) at a rotated start
/// `exp(i e X) D exp(-i e X)` that carries a current, at `A = 0` and two nonzero fields: (1)
/// `dE / de` along the tangent `dD = i [Y, D]` against `sum w Re tr(F dD)`; (2) `dE / dA_axis` at
/// fixed `D` against the observable, both by central differences at 1e-3 and 5e-4 (Richardson).
/// Red: the observable without its exchange-correlation term misses by that term.
#[test]
fn the_crystal_fock_matrix_and_current_are_the_derivatives_of_its_energy_in_a_field() {
    let mut cell = silicon();
    cell.functional = "r2scan";
    cell.mesh = [2, 2, 2];
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let request = Request::new([1., 0., 0.], 1e-3, 0.2, 1);
    let prepared = prepare(&sampler, &request).unwrap();
    let nk = prepared.crystal.points.len();
    let d = rotated_start(&prepared, 0.05);
    let tangent: Vec<Matrix> = d
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            let y = hermitian(x.nrows(), 1.3 + 0.2 * idx as f64);
            (&y * x - x * &y) * C64::new(0., 1.)
        })
        .collect();
    let weight = |idx: usize| prepared.crystal.points[idx % nk].weight;
    let richardson = |slope: &dyn Fn(f64) -> f64| (4. * slope(5e-4) - slope(1e-3)) / 3.;
    let mut worst = [0f64; 2];
    for field in [[0.; 3], [0.05, 0., 0.], [0.02, -0.03, 0.04]] {
        let (focks, _, current) = prepared.evaluate(&d, field).unwrap();
        let energy_along = |h: f64| {
            let moved: Vec<Matrix> = d
                .iter()
                .zip(&tangent)
                .map(|(x, t)| x + t * C64::new(h, 0.))
                .collect();
            prepared.evaluate(&moved, field).unwrap().1
        };
        let numeric = richardson(&|h| (energy_along(h) - energy_along(-h)) / (2. * h));
        let analytic: f64 = focks
            .iter()
            .zip(&tangent)
            .enumerate()
            .map(|(idx, (f, t))| weight(idx) * (f * t).trace().re)
            .sum();
        let fock_miss = (numeric - analytic).abs() / analytic.abs();
        worst[0] = worst[0].max(fock_miss);
        // The observable without the exchange-correlation term: the trace and N A.
        let orbital: [f64; 3] = std::array::from_fn(|axis| {
            d.iter()
                .enumerate()
                .map(|(idx, x)| weight(idx) * (x * &prepared.momenta[idx % nk][axis]).trace().re)
                .sum::<f64>()
                + prepared.crystal.electrons * field[axis]
        });
        for axis in 0..3 {
            let energy_in = |h: f64| {
                let mut a = field;
                a[axis] += h;
                prepared.evaluate(&d, a).unwrap().1
            };
            let slope = richardson(&|h| (energy_in(h) - energy_in(-h)) / (2. * h));
            let miss = (slope - current[axis]).abs();
            let exchange = current[axis] - orbital[axis];
            eprintln!(
                "Si r2SCAN SZ 2^3, A {field:?}: dE/de {numeric:.12e} against sum w Re tr(F dD) \
                 {analytic:.12e} ({fock_miss:.2e}); axis {axis}: dE/dA {slope:.12e} against the \
                 observable {:.12e} (miss {miss:.2e}), its exchange-correlation part {exchange:.6e} \
                 (red: without it the miss is {:.2e})",
                current[axis],
                (slope - orbital[axis]).abs()
            );
            worst[1] = worst[1].max(miss / current[axis].abs().max(1e-12));
            assert!(
                miss < 1e-8 * current[axis].abs().max(1.),
                "{field:?} {axis}"
            );
            assert!(exchange.abs() > 1e3 * miss, "{field:?} {axis}: no signal");
        }
        assert!(fock_miss < 1e-7, "{field:?}: {fock_miss:e}");
    }
    eprintln!(
        "worst: Fock {:.2e}, current {:.2e} (relative)",
        worst[0], worst[1]
    );
}

/// Item 4 on crystals: in `A = kappa e_x` the start's kinetic fields are `j = kappa n_v` and
/// `tau + kappa^2 n_v / 2` (its paramagnetic current is zero), so `tau~` shifts by the partial
/// core's share `(1 - n_v / n) n_v kappa^2 / 2` per spin, as for a molecule
/// (`molecular_realtime::in_the_velocity_gauge_tau_tilde_keeps_the_partial_cores_share`). Si r2SCAN
/// (`lda-sr`), SZ on the 2^3 whole mesh. Before the change the shift was zero (the field did not
/// enter `tau~`).
#[test]
fn in_a_crystal_tau_tilde_keeps_the_partial_cores_share_in_the_field() {
    let mut cell = silicon();
    cell.functional = "r2scan";
    cell.mesh = [2, 2, 2];
    let solved = solve(&cell);
    let sampler = solved.sampler();
    let prepared = prepare(&sampler, &Request::new([1., 0., 0.], 1e-3, 0.2, 1)).unwrap();
    let d: Vec<Matrix> = prepared.channels.iter().map(Channel::density).collect();
    let n = prepared.crystal.grid.index.points();
    let spins = prepared.crystal.spins;
    let share = if spins == 1 { 0.5 } else { 1. };
    let ground = prepared.kinetic_fields(&d, [0.; 3]).unwrap();
    for kappa in [0.05, 0.1] {
        let moved = prepared.kinetic_fields(&d, [kappa, 0., 0.]).unwrap();
        let [mut scale, mut core, mut tilde_miss, mut flow, mut flow_miss] = [0f64; 5];
        for s in 0..spins {
            for r in 0..n {
                let valence = ground.valence[s * n + r];
                if valence <= 1e-4 {
                    continue;
                }
                let total = ground.density[s * n + r];
                let boost = 0.5 * kappa * kappa * valence;
                let expected = (1. - valence / total) * boost / share;
                scale += boost / share;
                core += expected;
                tilde_miss += ((moved.gauge_invariant[s * n + r]
                    - ground.gauge_invariant[s * n + r])
                    - expected)
                    .abs();
                flow += kappa * valence;
                flow_miss += (0..3)
                    .map(|axis| {
                        let j = moved.current[(s * 3 + axis) * n + r]
                            - ground.current[(s * 3 + axis) * n + r];
                        (j - if axis == 0 { kappa * valence } else { 0. }).abs()
                    })
                    .sum::<f64>();
            }
        }
        eprintln!(
            "Si r2SCAN SZ 2^3, kappa {kappa}: the core term {:.4e} of int kappa^2 n_v / 2, tau~ \
             misses it by {:.2e} of itself, j misses kappa n_v by {:.2e}",
            core / scale,
            tilde_miss / core,
            flow_miss / flow
        );
        assert!(
            tilde_miss < 1e-8 * core && core > 1e-6 * scale,
            "{tilde_miss:e} of {core:e}"
        );
        assert!(flow_miss < 1e-12 * flow);
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

#[test]
#[ignore = "R13 on Si, a recorded diagnostic (not a gate): Wannier s and p on both atoms, the downfold's RPA head against T5a's local-field static tensor (SZ, 2^3; 9 s)"]
fn the_static_hartree_response_of_silicon_against_the_rpa_head() {
    // R13 on a covalent crystal: `eps_M(0)` with local fields at q = 0 (T5a's frequency route,
    // the static limit of the real-time TD-Hartree run, E2) against `1 / eps^-1_00` of the
    // downfold's RPA head at the finite `HEAD_WAVEVECTOR`. Measured 2026-11-16: T5a 23.58952 on
    // every axis, mean `1/eps` 0.04239, head `eps^-1_00` 0.02995 (1/head 33.38372), +41.5 %. The
    // coarse 2^3 mesh and the SZ basis's length/velocity gap separate the two, as on the H2 chain.
    use mandacaru_core::periodic::{
        bloch_mesh::Sampler,
        optics::dielectric::{dielectric, Kernel as Optical, Request as OpticalRequest},
        wannier::{
            downfold::{Request as Downfold, Screening},
            setup::Wannier,
            Options as Localization,
        },
    };
    let mut cell = silicon();
    cell.mesh = [2, 2, 2];
    let solved = solve(&cell);
    let quarter = 10.26 / 4.;
    let trials = serde_json::json!({
        "size": [2, 2, 2],
        "trials": [
            {"center": [0., 0., 0.], "orbital": "s", "label": "Si0s"},
            {"center": [0., 0., 0.], "orbital": "p", "label": "Si0p"},
            {"center": [quarter, quarter, quarter], "orbital": "s", "label": "Si1s"},
            {"center": [quarter, quarter, quarter], "orbital": "p", "label": "Si1p"},
        ],
    });
    let wannier = Wannier::build(
        solved.crystal.clone(),
        solved.state.clone(),
        solved.setup.clone(),
        &trials.to_string(),
        &Localization::default(),
    )
    .unwrap();
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
    let mut request = OpticalRequest::new(vec![0.], 0.02, Optical::Hartree);
    request.cutoff = 2.;
    let eps = dielectric(
        &sampler,
        &MeshIndex::new([2, 2, 2], [0.; 3]).unwrap(),
        &request,
    )
    .unwrap();
    let diagonal: Vec<f64> = (0..3).map(|i| eps.static_tensor[i][i]).collect();
    let inverse: f64 = diagonal.iter().map(|e| 1. / e / 3.).sum();
    eprintln!(
        "R13 Si SZ 2^3: T5a local-field static eps {diagonal:.5?}, mean 1/eps {inverse:.5}; RPA \
         head eps^-1_00 {head:.5} (1/head {:.5}); relative difference {:+.3e}",
        1. / head,
        (inverse - head) / head
    );
    assert!(head > 0. && head <= 1.);
    assert!(diagonal.iter().all(|e| *e > 1.));
}

#[test]
#[ignore = "R13 on LiF, a recorded diagnostic (not a gate): Wannier functions on every band, the downfold's RPA head against T5a's local-field static tensor (SZ, 2^3; a few minutes at load 15); run: cargo test --profile release-test -p mandacaru-core --test periodic_realtime -- --ignored lithium_fluoride --nocapture"]
fn the_static_hartree_response_of_lithium_fluoride_against_the_rpa_head() {
    // R13 on an ionic crystal, recorded the way Si's is: `eps_M(0)` with local fields at q = 0
    // (T5a's frequency route, the static limit of the TD-Hartree run) against `1 / eps^-1_00` of
    // the downfold's RPA head at the finite `HEAD_WAVEVECTOR`. Rocksalt LiF, 4.03 Angstrom.
    use mandacaru_core::periodic::{
        bloch_mesh::Sampler,
        optics::dielectric::{dielectric, Kernel as Optical, Request as OpticalRequest},
        wannier::{
            downfold::{Request as Downfold, Screening},
            setup::Wannier,
            Options as Localization,
        },
    };
    let a = 4.03 / 0.529177210903 / 2.;
    let cell = Cell {
        library: "lda-sr",
        symbols: vec!["Li", "F"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid: [24, 24, 24],
        fractional: vec![[0.; 3], [0.5; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [2, 2, 2],
        hubbard: None,
        width: 1e-3,
    };
    let solved = solve(&cell);
    let trials = serde_json::json!({
        "size": [2, 2, 2],
        "trials": [
            {"center": [0., 0., 0.], "orbital": "s", "label": "Li s"},
            {"center": [a, a, a], "orbital": "s", "label": "F s"},
            {"center": [a, a, a], "orbital": "p", "label": "F p"},
        ],
    });
    let wannier = Wannier::build(
        solved.crystal.clone(),
        solved.state.clone(),
        solved.setup.clone(),
        &trials.to_string(),
        &Localization::default(),
    )
    .unwrap();
    let head = wannier
        .downfold(&Downfold {
            screening: Some(Screening::Rpa),
            // The fragment is the empty Li s function (0.263 Kohn-Sham electrons measured).
            num_particles: Some([0, 0]),
            ..wannier_downfold_support::request(Some(vec![(0, [0; 3])]))
        })
        .unwrap()
        .problem
        .dielectric_head
        .unwrap();
    let sampler = Sampler::new(&wannier.crystal, &wannier.state, &wannier.setup).unwrap();
    let mut request = OpticalRequest::new(vec![0.], 0.02, Optical::Hartree);
    request.cutoff = 2.;
    let eps = dielectric(
        &sampler,
        &MeshIndex::new([2, 2, 2], [0.; 3]).unwrap(),
        &request,
    )
    .unwrap();
    let diagonal: Vec<f64> = (0..3).map(|i| eps.static_tensor[i][i]).collect();
    let inverse: f64 = diagonal.iter().map(|e| 1. / e / 3.).sum();
    eprintln!(
        "R13 LiF SZ 2^3: T5a local-field static eps {diagonal:.5?}, mean 1/eps {inverse:.5}; RPA \
         head eps^-1_00 {head:.5} (1/head {:.5}); relative difference {:+.3e}",
        1. / head,
        (inverse - head) / head
    );
    assert!(head > 0. && head <= 1.);
    assert!(diagonal.iter().all(|e| *e > 1.));
}

/// Toward the +U insulator with a partly filled shell off inversion (TDDFT_PLAN.md 13.5.3, open
/// in T6b): CuH2 bent and unequal (Cu-H 2.9 and 3.1 Bohr) in a 9 x 8 x 8 Bohr cell at Gamma,
/// spin-polarized (11 and 10 electrons, a gap in each channel), Cu 3d with U = 3 and 6 eV. Its
/// shell is nearly full (4.70 and 4.72 electrons per channel at U = 3 eV); the partly filled
/// fixture is NiO, below. The static self-consistency (Richardson of 2e-3 and 1e-3) departs
/// from the band sum with U and equals it without U, and the energy route agrees with the
/// current route. Recorded: a Cu(II) d9 fixture (CuF2) is a metal in LDA when linear, and bent
/// its gap is too small for the static iteration to converge.
#[test]
#[ignore = "slow: about 64 s, three CuH2 SCFs and their static responses (recorded, 2026-11-16)"]
fn a_nearly_full_hubbard_shell_off_inversion_shifts_the_static_response() {
    let mut shifts = vec![];
    for u in [
        None,
        Some(r#"{"Cu": {"l": "d", "U": 3.0}}"#),
        Some(r#"{"Cu": {"l": "d", "U": 6.0}}"#),
    ] {
        let cell = Cell {
            library: "lda-sr",
            symbols: vec!["Cu", "H", "H"],
            lattice: Matrix3::from_diagonal(&Vector3::new(9., 8., 8.)),
            grid: [36, 32, 32],
            fractional: vec![
                [0.5, 0.5, 0.5],
                [0.5 - 2.9 / 9., 0.52, 0.5],
                [0.5 + 3.1 / 9., 0.49, 0.5],
            ],
            moments: vec![1., 0., 0.],
            spin: true,
            spinor: false,
            functional: "lda",
            mesh: [1, 1, 1],
            hubbard: u,
            width: 1e-3,
        };
        let solved = solve(&cell);
        let sampler = solved.sampler();
        let shell: Vec<f64> = solved.state.hubbard.as_ref().map_or(vec![], |report| {
            report
                .occupations
                .iter()
                .map(|n| {
                    let p = (n.len() as f64).sqrt().round() as usize;
                    (0..p).map(|i| n[i * p + i].re).sum()
                })
                .collect()
        });
        let band = transitions(
            &sampler,
            &MeshIndex::new([1, 1, 1], [0.; 3]).unwrap(),
            [1., 0., 0.],
        )
        .unwrap()
        .diamagnetic(2)
        .total;
        let routes: Vec<(f64, f64)> = [2e-3, 1e-3]
            .iter()
            .map(|&strength| {
                let request = Request::new([1., 0., 0.], strength, 0.2, 10);
                let fixed = prepare(&sampler, &request)
                    .unwrap()
                    .static_response(&request)
                    .unwrap();
                (fixed.vector[0], fixed.energy)
            })
            .collect();
        let extrapolate = |coarse: f64, fine: f64| (4. * fine - coarse) / 3.;
        let current = extrapolate(routes[0].0, routes[1].0);
        let energy = extrapolate(routes[0].1, routes[1].1);
        eprintln!(
            "CuH2 {}: d per channel {shell:.4?}; static N_dia {current:.8}, energy route \
             {energy:.8}, band sum {band:.8}: static - band {:+.4e} ({:+.3e} relative)",
            u.unwrap_or("without U"),
            current - band,
            (current - band) / band
        );
        assert!(((energy - current) / current).abs() < 3e-6);
        shifts.push((current - band) / band);
    }
    // Measured: -5.1e-9 without U, +1.85e-5 and +1.45e-5 with U = 3 and 6 eV.
    assert!(shifts[0].abs() < 1e-7, "{shifts:?}");
    assert!(shifts[1..].iter().all(|s| *s > 1e-5), "{shifts:?}");
}

/// The +U insulator with a partly filled shell off inversion (TDDFT_PLAN.md 13.5.3, open in T6b):
/// the NiO molecule in a 10 x 9 x 9 Bohr cell at Gamma (Ni-O 3.08 Bohr along x, off the cell's
/// mirror planes; spin-polarized, moment 2, a gap of 1.4 to 3.0 eV in each channel), Ni 3d with
/// U = 3 and 6 eV. Its minority shell is partly filled: 3.48 and 3.45 electrons, the occupation
/// eigenvalues from 0.44 and 0.37 to 0.97 (CuH2's shell above holds 4.70, nearly full). The
/// static self-consistency equals the band sum without U and departs from it with U by 15 to 38
/// times the nearly full shell's shift, and the energy route agrees with the current route. The
/// static map of this molecule diverges under plain mixing (a density change of 0.65 after 300
/// iterations, even without U); the DIIS of `static_response` converges it in 9 to 45.
#[test]
#[ignore = "slow: 60 s, three NiO SCFs and their static responses (recorded, 2026-11-16)"]
fn a_partly_filled_hubbard_shell_off_inversion_departs_from_the_band_sum() {
    let mut shifts = vec![];
    for u in [
        None,
        Some(r#"{"Ni": {"l": "d", "U": 3.0}}"#),
        Some(r#"{"Ni": {"l": "d", "U": 6.0}}"#),
    ] {
        let cell = Cell {
            library: "lda-sr",
            symbols: vec!["Ni", "O"],
            lattice: Matrix3::from_diagonal(&Vector3::new(10., 9., 9.)),
            grid: [40, 36, 36],
            fractional: vec![[0.5 - 0.154, 0.52, 0.5], [0.5 + 0.154, 0.49, 0.5]],
            moments: vec![2., 0.],
            spin: true,
            spinor: false,
            functional: "lda",
            mesh: [1, 1, 1],
            hubbard: u,
            width: 1e-3,
        };
        let solved = solve(&cell);
        let sampler = solved.sampler();
        // Each channel's shell occupation eigenvalues.
        let shells: Vec<Vec<f64>> = solved.state.hubbard.as_ref().map_or(vec![], |report| {
            report
                .occupations
                .iter()
                .map(|n| {
                    let p = (n.len() as f64).sqrt().round() as usize;
                    let matrix = DMatrix::from_fn(p, p, |i, j| n[i * p + j]);
                    let mut values: Vec<f64> = ((&matrix + matrix.adjoint()) * C64::new(0.5, 0.))
                        .symmetric_eigen()
                        .eigenvalues
                        .iter()
                        .copied()
                        .collect();
                    values.sort_by(f64::total_cmp);
                    values
                })
                .collect()
        });
        let band = transitions(
            &sampler,
            &MeshIndex::new([1, 1, 1], [0.; 3]).unwrap(),
            [1., 0., 0.],
        )
        .unwrap()
        .diamagnetic(2)
        .total;
        let routes: Vec<(f64, f64)> = [2e-3, 1e-3]
            .iter()
            .map(|&strength| {
                let request = Request::new([1., 0., 0.], strength, 0.2, 10);
                let fixed = prepare(&sampler, &request)
                    .unwrap()
                    .static_response(&request)
                    .unwrap();
                (fixed.vector[0], fixed.energy)
            })
            .collect();
        let extrapolate = |coarse: f64, fine: f64| (4. * fine - coarse) / 3.;
        let current = extrapolate(routes[0].0, routes[1].0);
        let energy = extrapolate(routes[0].1, routes[1].1);
        eprintln!(
            "NiO {}: shell eigenvalues per channel {shells:.4?}; static N_dia {current:.8}, energy \
             route {energy:.8}, band sum {band:.8}: static - band {:+.4e} ({:+.3e} relative)",
            u.unwrap_or("without U"),
            current - band,
            (current - band) / band
        );
        assert!(((energy - current) / current).abs() < 5e-6);
        if let Some(minority) = shells.get(1) {
            assert!(
                minority[0] < 0.5 && minority[minority.len() - 1] > 0.9,
                "{shells:?}"
            );
        }
        shifts.push((current - band) / band);
    }
    // Measured 2026-11-16: -1.7e-10 without U, +2.80e-4 and +5.46e-4 with U = 3 and 6 eV (U = 8
    // eV, whose shell reorders, +3.52e-4); CuH2's nearly full shell +1.85e-5 and +1.45e-5.
    assert!(shifts[0].abs() < 1e-8, "{shifts:?}");
    assert!(shifts[1..].iter().all(|s| *s > 1e-4), "{shifts:?}");
}
