//! The frequency-domain dielectric function of a crystal at `q -> 0`
//! (`periodic::optics::dielectric`, TDDFT_PLAN.md T5a), checked by identities of the engine's
//! own numbers (section 4.4):
//!
//! - E0 (R10): the independent-particle tensor is the closed form of the band sum's transitions,
//!   and equals the real-time `kernel="independent"` spectrum at the same complex frequency;
//! - E2: at `omega = 0` the frequency polarizability's local-field block is the screening code's
//!   static polarizability (the same transitions); the macroscopic static limit against the cRPA
//!   head and the real-time TD-Hartree run is recorded (`#[ignore]`);
//! - E7: the tensor is bit-identical across `kpts` layouts and thread counts;
//! - E8: the first moment (`eps -> 1 - 4 pi N_dia / (Omega z^2)` on the imaginary axis) and
//!   Kramers-Kronig on the real line, with and without local fields;
//! - the refusals by name, and the mesh convergence of Si (recorded, `#[ignore]`).
//!
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;
mod wannier_downfold_support;

use berry_support::*;
use mandacaru_core::{
    models::time_spectrum::{dielectric_function, Window},
    periodic::{
        bloch_mesh::{MeshIndex, Sampler},
        optics::{
            dielectric::{dielectric, pairs, Kernel, Request},
            momentum::transitions,
            realtime::{self, propagate},
        },
        screening::medium::{gamma_polarizability, within},
        spectrum::Smearing,
    },
    quantum::C64,
    Context,
};
use std::f64::consts::PI;

fn axis(a: usize) -> [f64; 3] {
    std::array::from_fn(|i| if i == a { 1. } else { 0. })
}

/// `eps^{ab}(z)` of independent particles from the band sum's transitions:
/// `delta_ab + (8 pi g / Omega) sum w Re[conj(P^a) P^b] / (D (D^2 - z^2))`.
fn closed_form(sampler: &Sampler<'_>, mesh: &MeshIndex, z: C64) -> [[C64; 3]; 3] {
    let lines: Vec<_> = (0..3)
        .map(|a| transitions(sampler, mesh, axis(a)).unwrap())
        .collect();
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let t = &lines[a];
            let mut sum = C64::new(f64::from(a == b), 0.);
            for (gap, cross) in t.gaps.iter().zip(&t.cross) {
                sum += 8. * PI * t.degeneracy * cross[b]
                    / t.volume
                    / (gap * (C64::new(gap * gap, 0.) - z * z));
            }
            sum
        })
    })
}

#[test]
fn the_independent_tensor_is_the_closed_form_of_the_band_sum() {
    // E0, the closed-form half: the same P and the same gaps as `momentum::transitions`, so the
    // tensor and the first moment are the band sum's to round-off.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let frequencies = vec![0., 0.1, 0.25, 0.4];
    let out = dielectric(
        &sampler,
        &mesh,
        &Request::new(frequencies.clone(), 0.05, Kernel::Independent),
    )
    .unwrap();
    let mut worst = 0f64;
    for (omega, tensor) in frequencies.iter().zip(&out.tensor) {
        let expected = closed_form(&sampler, &mesh, C64::new(*omega, 0.05));
        for a in 0..3 {
            for b in 0..3 {
                worst = worst.max((tensor[a][b] - expected[a][b]).norm() / expected[0][0].norm());
            }
        }
    }
    let band_sum: Vec<f64> = (0..3)
        .map(|a| {
            transitions(&sampler, &mesh, axis(a))
                .unwrap()
                .diamagnetic(sampler.channels())
                .total
        })
        .collect();
    eprintln!(
        "LiH: worst tensor difference {worst:.2e}; first moment diagonal {:?} against the band \
         sum {band_sum:?}; eps_xx(0.05 i) {}",
        [0, 1, 2].map(|a| out.sum_rule[a][a]),
        out.tensor[0][0][0]
    );
    assert!(worst < 1e-12, "{worst:e}");
    for a in 0..3 {
        assert!((out.sum_rule[a][a] - band_sum[a]).abs() < 1e-12 * band_sum[a]);
    }
    // The skew cell couples x and y; the tensor is symmetric.
    assert!(out.tensor[0][0][1].norm() > 1e-3);
    assert!((out.tensor[0][0][1] - out.tensor[0][1][0]).norm() < 1e-12);
}

#[test]
fn the_static_local_field_block_is_the_screening_polarizability() {
    // E2, the identity half: at omega = 0 (no damping) the frequency weights of every pair,
    // 1 / (0 - D) and -1 / (0 + D), are the static weights (f_a - f_b) / (e_a - e_b) of the
    // screening code, on the same pair densities; the Gamma-only H2 chain (DZP) is every
    // pair of one point, `gamma_polarizability`'s audit seam.
    let solved = solve_sized(&h2_chain("lda-sr", "lda"), "raw", "DZP");
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([1, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(vec![0.], 0., Kernel::Hartree);
    request.cutoff = 2.;
    let pairs = pairs(&sampler, &mesh, &request).unwrap();
    let chi = pairs.polarizability(C64::new(0., 0.)).unwrap();
    let g: Vec<[i64; 3]> = within(&sampler.lattice, [0.; 3], request.cutoff)
        .into_iter()
        .filter(|g| *g != [0, 0, 0])
        .collect();
    assert_eq!(g.len(), pairs.vectors.len());
    let crystal = &solved.crystal;
    let fermi = solved.state.fermi;
    let reference = gamma_polarizability(
        crystal,
        &solved.state,
        0,
        crystal.reference,
        fermi,
        1e-4,
        Smearing::FermiDirac,
        &g,
    )
    .unwrap();
    let n = g.len();
    let mut worst = 0f64;
    let mut largest = 0f64;
    for i in 0..n {
        for j in 0..n {
            largest = largest.max(reference[(i, j)].norm());
            worst = worst.max((chi[(3 + i, 3 + j)] - reference[(i, j)]).norm());
        }
    }
    let static_tensor = pairs.tensor(C64::new(0., 0.)).unwrap();
    eprintln!(
        "H2 chain DZP: {n} local-field vectors, {} transitions; static block difference \
         {worst:.2e} of {largest:.3e}; eps_M(0) diagonal {:.6} {:.6} {:.6}",
        pairs.energies.len(),
        static_tensor[0][0].re,
        static_tensor[1][1].re,
        static_tensor[2][2].re
    );
    assert!(largest > 0. && worst < 1e-10 * largest, "{worst:e}");
}

#[test]
fn the_first_moment_and_kramers_kronig_hold_with_and_without_local_fields() {
    // E8: (a) on the imaginary axis eps(i Y) - 1 -> 4 pi N_dia / (Omega Y^2): the first moment of
    // epsilon_2 is the band sum, for the head alone and with local fields; (b) on the line
    // z = omega + i eta, Re(eps - 1) is the Kramers-Kronig transform of Im eps.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    for kernel in [Kernel::Independent, Kernel::Hartree] {
        let request = Request::new(vec![0.], 0., kernel);
        let pairs = pairs(&sampler, &mesh, &request).unwrap();
        let top = 2e3;
        let far = pairs.tensor(C64::new(0., top)).unwrap();
        let mut moment = [0f64; 3];
        for (a, m) in moment.iter_mut().enumerate() {
            *m = (far[a][a].re - 1.) * pairs.volume * top * top / (4. * PI);
        }
        let (damping, step, last) = (0.05, 0.004, 24.);
        let frequencies: Vec<f64> = (1..=(last / step) as usize)
            .map(|i| i as f64 * step)
            .collect();
        let line: Vec<C64> = frequencies
            .iter()
            .map(|w| pairs.tensor(C64::new(*w, damping)).unwrap()[0][0])
            .collect();
        let mut worst = 0f64;
        let mut largest = 0f64;
        for i in (100..frequencies.len() / 4).step_by(50) {
            let omega = frequencies[i] + 0.5 * step;
            let direct = 0.5 * (line[i].re + line[i + 1].re) - 1.;
            let transformed: f64 = frequencies
                .iter()
                .zip(&line)
                .map(|(w, e)| 2. / PI * w * e.im / (w * w - omega * omega) * step)
                .sum();
            largest = largest.max(direct.abs());
            worst = worst.max((direct - transformed).abs());
        }
        eprintln!(
            "LiH {}: first moment {moment:.8?} against N_dia {:.8?}; Kramers-Kronig worst \
             {worst:.2e} of {largest:.4e}; eps_xx(0) {:.6}",
            kernel.name(),
            [0, 1, 2].map(|a| pairs.sum_rule[a][a]),
            pairs.tensor(C64::new(0., 0.)).unwrap()[0][0].re
        );
        for (a, m) in moment.iter().enumerate() {
            let expected = pairs.sum_rule[a][a];
            assert!(
                (m - expected).abs() < 1e-4 * expected,
                "{m} against {expected}"
            );
        }
        assert!(worst < 2e-2 * largest, "{worst:e} against {largest:e}");
    }
}

#[test]
fn local_fields_lower_the_static_dielectric_constant() {
    // Local fields (the induced microscopic fields) screen the macroscopic one less: on LiH the
    // RPA eps_M(0) lies below the independent-particle one, and the two approach as the
    // local-field cutoff shrinks to nothing.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let independent = dielectric(
        &sampler,
        &mesh,
        &Request::new(vec![0.], 0., Kernel::Independent),
    )
    .unwrap()
    .tensor[0][0][0]
        .re;
    let mut values = vec![];
    for cutoff in [1., 2., 3.] {
        let mut request = Request::new(vec![0.], 0., Kernel::Hartree);
        request.cutoff = cutoff;
        let out = dielectric(&sampler, &mesh, &request).unwrap();
        values.push((cutoff, out.vectors, out.tensor[0][0][0].re));
    }
    eprintln!("LiH eps_xx(0): independent {independent:.6}; with local fields {values:?}");
    for (_, _, value) in &values {
        assert!(*value > 1. && *value < independent);
    }
}

#[test]
fn the_long_range_kernel_reduces_to_the_local_fields_and_raises_epsilon() {
    // T5c: alpha = 0 is the local-field tensor bit for bit; the fitted alpha is
    // 4.615 / eps_inf - 0.213 of the local-field static tensor; alpha > 0 raises eps_M(0); alpha
    // with another kernel, and alpha >= 4 pi, raise.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let frequencies = vec![0.05, 0.2];
    let hartree = dielectric(&sampler, &mesh, &Request::new(frequencies.clone(), 0.05, Kernel::Hartree)).unwrap();
    let mut request = Request::new(frequencies.clone(), 0.05, Kernel::LongRange);
    request.alpha = Some(0.);
    let zero = dielectric(&sampler, &mesh, &request).unwrap();
    assert_eq!(zero.tensor, hartree.tensor);
    request.alpha = None;
    let fitted = dielectric(&sampler, &mesh, &request).unwrap();
    let eps_inf = (0..3).map(|a| hartree.static_tensor[a][a]).sum::<f64>() / 3.;
    let alpha = fitted.alpha.unwrap();
    assert!((alpha - (4.615 / eps_inf - 0.213)).abs() < 1e-14);
    request.alpha = Some(0.1);
    let weak = dielectric(&sampler, &mesh, &request).unwrap();
    request.alpha = Some(0.5);
    let strong = dielectric(&sampler, &mesh, &request).unwrap();
    eprintln!(
        "T5c LiH: eps_M(0) xx local fields {:.6}, alpha 0.1 {:.6}, alpha 0.5 {:.6}, fitted alpha \
         {alpha:.4} {:.6}",
        hartree.static_tensor[0][0],
        weak.static_tensor[0][0],
        strong.static_tensor[0][0],
        fitted.static_tensor[0][0]
    );
    assert!(strong.static_tensor[0][0] > weak.static_tensor[0][0]);
    assert!(weak.static_tensor[0][0] > hartree.static_tensor[0][0]);
    request.alpha = Some(4. * PI);
    assert!(dielectric(&sampler, &mesh, &request).unwrap_err().to_string().contains("alpha"));
    let mut wrong = Request::new(frequencies, 0.05, Kernel::Hartree);
    wrong.alpha = Some(0.1);
    assert!(dielectric(&sampler, &mesh, &wrong).unwrap_err().to_string().contains("long-range"));
}

#[test]
fn the_tensor_does_not_depend_on_the_layout() {
    // E7: kpts lanes and threads change no bit of the tensor.
    let cell = lih("lda-sr");
    let mut tensors = vec![];
    for (threads, kpts) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let ctx = Context::new(Some(threads), 1 << 30)
            .unwrap()
            .with_layout(kpts, 1)
            .unwrap();
        let solved = solve_on(&cell, "raw", "SZ", ctx);
        let sampler = solved.sampler();
        let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
        let out = dielectric(
            &sampler,
            &mesh,
            &Request::new(vec![0., 0.2, 0.4], 0.05, Kernel::Hartree),
        )
        .unwrap();
        tensors.push(out.tensor);
    }
    for tensor in &tensors[1..] {
        assert_eq!(tensor, &tensors[0]);
    }
}

/// `eps_xx(omega + i damping)` of the real-time run with `kernel` (the odd part of a kick pair
/// of `strength` along x, `steps` of `dt`) on `frequencies`.
fn real_time_epsilon(
    sampler: &Sampler<'_>,
    kernel: realtime::Kernel,
    dt: f64,
    steps: usize,
    frequencies: &[f64],
    damping: f64,
) -> Vec<C64> {
    real_time_run(sampler, kernel, dt, steps, frequencies, damping).0
}

/// [`real_time_epsilon`] with the run's band-sum `N_dia` and the static deficit the corrected odd
/// current still carries, `-Omega mean(j_odd) / A` over the second half of the run.
fn real_time_run(
    sampler: &Sampler<'_>,
    kernel: realtime::Kernel,
    dt: f64,
    steps: usize,
    frequencies: &[f64],
    damping: f64,
) -> (Vec<C64>, f64, f64) {
    let strength = 1e-3;
    let mut request = realtime::Request::new([1., 0., 0.], strength, dt, steps);
    request.kernel = kernel;
    let plus = propagate(sampler, &request).unwrap();
    let mut minus = request.clone();
    minus.strength = -strength;
    let minus = propagate(sampler, &minus).unwrap();
    let odd: Vec<f64> = plus
        .corrected
        .iter()
        .zip(&minus.corrected)
        .map(|(a, b)| 0.5 * (a - b))
        .collect();
    let late = &odd[odd.len() / 2..];
    let residual = -plus.volume * late.iter().sum::<f64>() / late.len() as f64 / strength;
    let values = dielectric_function(
        &sampler.crystal.context,
        &odd,
        dt,
        strength,
        frequencies,
        damping,
        Window::Exponential,
    )
    .unwrap()
    .values
    .as_slice()
    .to_vec();
    (values, plus.diamagnetic, residual)
}

#[test]
fn the_independent_tensor_is_the_real_time_spectrum() {
    // E0 (R10), the cross-engine half: the real-time run with the Hamiltonian frozen, in the
    // linear limit, transformed with the exponential window, is the independent-particle
    // eps(omega + i gamma) of the same P. 600 a.u. at gamma = 0.05 leaves a tail of e^-30.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let damping = 0.05;
    let frequencies = vec![0.02, 0.1, 0.2, 0.3, 0.5];
    let time = real_time_epsilon(
        &sampler,
        realtime::Kernel::Independent,
        0.1,
        6000,
        &frequencies,
        damping,
    );
    let frequency = dielectric(
        &sampler,
        &mesh,
        &Request::new(frequencies.clone(), damping, Kernel::Independent),
    )
    .unwrap();
    let mut worst = 0f64;
    for (t, tensor) in time.iter().zip(&frequency.tensor) {
        worst = worst.max((t - tensor[0][0]).norm() / tensor[0][0].norm());
    }
    eprintln!(
        "LiH independent: worst relative difference {worst:.2e}; eps_xx(0.02 + 0.05 i) real \
         time {:.6}, frequency {:.6}",
        time[0], frequency.tensor[0][0][0]
    );
    assert!(worst < 1e-3, "{worst:e}");
}

#[test]
#[ignore = "E2, recorded (not a gate): local fields against TD-Hartree (LiH) and the cRPA head (H2 chain); about 2 min"]
fn the_static_limit_against_td_hartree_and_the_rpa_head() {
    // The frequency RPA with local fields and the real-time TD-Hartree run treat the same
    // Hartree kernel on different vector sets (|G| <= cutoff against the whole grid, and the PAW
    // one-center Hartree terms only in the second); the cRPA head is the same polarizability at a
    // finite q = 0.02 along each axis, in the length form of the pair densities, so it differs
    // from the k . p head by the basis's length/velocity defect (R13: 1.022 against 1.187).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let damping = 0.05;
    let frequencies = vec![0.02, 0.1, 0.2, 0.3, 0.5];
    let time = real_time_epsilon(
        &sampler,
        realtime::Kernel::Hartree,
        0.1,
        6000,
        &frequencies,
        damping,
    );
    for cutoff in [2., 3., 4.] {
        let mut request = Request::new(frequencies.clone(), damping, Kernel::Hartree);
        request.cutoff = cutoff;
        let frequency = dielectric(&sampler, &mesh, &request).unwrap();
        let rows: Vec<String> = time
            .iter()
            .zip(&frequency.tensor)
            .zip(&frequencies)
            .map(|((t, f), w)| format!("{w:.2}: {t:.5} / {:.5}", f[0][0]))
            .collect();
        eprintln!(
            "LiH hartree, cutoff {cutoff} ({} vectors): real time / frequency {rows:?}",
            frequency.vectors
        );
    }
    let (_ctx, wannier) = wannier_downfold_support::chain([3, 1, 1]);
    use mandacaru_core::periodic::wannier::downfold::{Request as Downfold, Screening};
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
    // The real-time run takes the SCF's mesh, so the frequency side samples that one against it;
    // the cRPA head is made on the Wannier mesh of the fixture (3 x 1 x 1), so the frequency side
    // samples that one against the head.
    let size = realtime::scf_mesh(&wannier.setup).unwrap();
    eprintln!("H2 chain: the SCF mesh is {size:?}, the head's [3, 1, 1]");
    let mesh = MeshIndex::new(size, [0.; 3]).unwrap();
    let head_mesh = MeshIndex::new([3, 1, 1], [0.; 3]).unwrap();
    // The real-time run along the chain at R13's step, over R13's 1,500 steps and four times
    // longer: the static limit of a short run is truncated (its tail is e^-6 at gamma = 0.02).
    for (rt, kernel) in [
        (realtime::Kernel::Independent, Kernel::Independent),
        (realtime::Kernel::Hartree, Kernel::Hartree),
    ] {
        let chain = pairs(&sampler, &mesh, &Request::new(vec![0.], 0., kernel)).unwrap();
        let frequency = chain.tensor(C64::new(0., 0.02)).unwrap()[0][0];
        for steps in [1500, 6000] {
            let (time, band, left) = real_time_run(&sampler, rt, 0.2, steps, &[0.], 0.02);
            eprintln!(
                "H2 chain {}: eps_xx(0 + 0.02 i) real time ({steps} steps of 0.2) {:.5}, \
                 frequency {frequency:.5}; N_dia real time {band:.8}, frequency {:.8}; deficit \
                 left in the corrected current {left:+.3e}",
                kernel.name(),
                time[0],
                chain.sum_rule[0][0]
            );
        }
    }
    for (kernel, z) in [
        (Kernel::Independent, C64::new(0., 0.)),
        (Kernel::Hartree, C64::new(0., 0.)),
        (Kernel::Hartree, C64::new(0., 0.02)),
    ] {
        let pairs = pairs(&sampler, &head_mesh, &Request::new(vec![0.], 0., kernel)).unwrap();
        let tensor = pairs.tensor(z).unwrap();
        let inverse: f64 = (0..3).map(|a| 1. / tensor[a][a].re / 3.).sum();
        eprintln!(
            "H2 chain {} at {z}: eps diagonal {:.5} {:.5} {:.5}, mean 1/eps {inverse:.5}; cRPA \
             head eps^-1_00 {head:.5} (relative {:+.3e})",
            kernel.name(),
            tensor[0][0].re,
            tensor[1][1].re,
            tensor[2][2].re,
            (inverse - head) / head
        );
    }
}

#[test]
fn unsupported_requests_are_refused_by_name() {
    for (name, word) in [("adiabatic", "get_real_time"), ("rpa", "hartree")] {
        let error = Kernel::parse(name).unwrap_err().to_string();
        assert!(error.contains(word), "{name}: {error}");
    }
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(vec![0.], 0.05, Kernel::Independent);
    request.window = Some((9, 1));
    assert!(dielectric(&sampler, &mesh, &request)
        .unwrap_err()
        .to_string()
        .contains("bands"));
    let shifted = MeshIndex::new([3, 1, 1], [0.5, 0., 0.]).unwrap();
    assert!(dielectric(
        &sampler,
        &shifted,
        &Request::new(vec![0.], 0.05, Kernel::Independent)
    )
    .is_err());
    let mut meta = lih("lda-sr");
    meta.functional = "r2scan";
    let meta = solve(&meta);
    assert!(dielectric(&meta.sampler(), &mesh, &Request::new(vec![0.], 0.05, Kernel::Hartree))
        .unwrap_err()
        .to_string()
        .contains("meta-GGA"));
    let mut negative = Request::new(vec![0.], -0.1, Kernel::Independent);
    assert!(dielectric(&sampler, &mesh, &negative)
        .unwrap_err()
        .to_string()
        .contains("damping"));
    negative.damping = 0.05;
    negative.scissor = -100.;
    assert!(dielectric(&sampler, &mesh, &negative)
        .unwrap_err()
        .to_string()
        .contains("scissor"));
}

#[test]
fn a_scissor_moves_the_poles_and_keeps_the_dipoles() {
    // The scissor shifts every transition by the same energy and keeps P / (e_c - e_v): the
    // first moment scales each line by D / (e_c - e_v), and the independent eps_2 moves rigidly.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let shift = 0.05;
    let plain = dielectric(
        &sampler,
        &mesh,
        &Request::new(vec![0.2], 0.02, Kernel::Independent),
    )
    .unwrap();
    let mut request = Request::new(vec![0.2 + shift], 0.02, Kernel::Independent);
    request.scissor = shift;
    let moved = dielectric(&sampler, &mesh, &request).unwrap();
    let lines = transitions(&sampler, &mesh, axis(0)).unwrap();
    let expected: f64 = lines
        .gaps
        .iter()
        .zip(&lines.weights)
        .map(|(g, w)| 2. * lines.degeneracy * w * (g + shift) / (g * g))
        .sum();
    eprintln!(
        "LiH: onset {:.6} -> {:.6}; first moment {:.8} against {expected:.8}; Im eps_xx {:.6} at \
         0.2 and {:.6} at {:.2}",
        plain.onset,
        moved.onset,
        moved.sum_rule[0][0],
        plain.tensor[0][0][0].im,
        moved.tensor[0][0][0].im,
        0.2 + shift
    );
    assert!((moved.onset - plain.onset - shift).abs() < 1e-12);
    assert!((moved.sum_rule[0][0] - expected).abs() < 1e-12 * expected);
    assert!(moved.tensor[0][0][0].im > 0.);
}

/// Silicon (fcc, 10.26 Bohr, two atoms) on 20^3 points, solved on the Gamma-centered `mesh`.
fn silicon(mesh: usize) -> Cell {
    let a = 10.26 / 2.;
    Cell {
        library: "lda-sr",
        symbols: vec!["Si", "Si"],
        lattice: nalgebra::Matrix3::from_columns(&[
            nalgebra::Vector3::new(0., a, a),
            nalgebra::Vector3::new(a, 0., a),
            nalgebra::Vector3::new(a, a, 0.),
        ]),
        grid: [20, 20, 20],
        fractional: vec![[0.; 3], [0.25; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [mesh; 3],
        hubbard: None,
        width: 1e-3,
    }
}

#[test]
#[ignore = "T5a mesh convergence and the Si table, recorded: one SCF per basis (SZ, DZP) on 4^3, the response on 4^3, 6^3, 8^3 (several minutes)"]
fn the_dielectric_function_of_silicon_converges_with_the_mesh() {
    // eps_M(0) of independent particles and with local fields, the onset and the first moment,
    // per basis and response mesh, all from one SCF per basis on the SCF mesh `SCF` (default 4^3):
    // the response samples its own whole mesh (`MESHES`, default 4, 6, 8) from the same converged
    // potential, so only the response's k sampling changes. Then the table eps(E + 0.1 eV i) on the
    // densest response mesh of the last basis.
    let hartree_ev = 27.211386245988;
    let sizes: Vec<String> = std::env::var("SIZES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| vec!["SZ".into(), "DZP".into()]);
    let meshes: Vec<usize> = std::env::var("MESHES")
        .map(|v| v.split(',').map(|m| m.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![4, 6, 8]);
    let scf: usize = std::env::var("SCF").map(|v| v.parse().unwrap()).unwrap_or(4);
    for size in &sizes {
        let solved = solve_sized(&silicon(scf), "raw", size);
        let sampler = solved.sampler();
        for &n in &meshes {
            let started = std::time::Instant::now();
            let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
            let independent =
                pairs(&sampler, &mesh, &Request::new(vec![0.], 0., Kernel::Independent)).unwrap();
            let local = pairs(&sampler, &mesh, &Request::new(vec![0.], 0., Kernel::Hartree)).unwrap();
            let e_ip = independent.tensor(C64::new(0., 0.)).unwrap()[0][0].re;
            let e_lf = local.tensor(C64::new(0., 0.)).unwrap()[0][0].re;
            eprintln!(
                "Si {size} SCF {scf}^3, response {n}^3: eps_M(0) independent {e_ip:.5}, local \
                 fields {e_lf:.5} ({} vectors, {} transitions); onset {:.4} eV; N_dia {:.5} (N {}); \
                 {:.0} s",
                local.vectors.len(),
                local.energies.len(),
                local.onset * hartree_ev,
                local.sum_rule[0][0],
                local.electrons,
                started.elapsed().as_secs_f64()
            );
            if n == *meshes.last().unwrap() && size == sizes.last().unwrap() {
                let damping = 0.1 / hartree_ev;
                for i in 0..=40 {
                    let e = 0.25 * i as f64;
                    let z = C64::new(e / hartree_ev, damping);
                    let a = independent.tensor(z).unwrap()[0][0];
                    let b = local.tensor(z).unwrap()[0][0];
                    eprintln!(
                        "Si {size} {n}^3 table: {e:5.2} eV: independent {:9.4} {:9.4}  local \
                         fields {:9.4} {:9.4}",
                        a.re, a.im, b.re, b.im
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "Q20/R9a on converged response meshes, recorded: Si's velocity, mixed and length band sums streamed per k-point (several minutes per mesh)"]
fn the_three_band_sums_of_silicon_converge_with_the_mesh() {
    // vera's no-propagation recipe, one k-point at a time: at each k the velocity sum
    // `g sum_vc 2 |P^x_vc|^2 / (e_c - e_v)`; the occupied bands transferred to `k +- A` by
    // `exp(i A . r)` (`Sampler::overlap`, orthonormalized, F2's transfer); the length sum from their
    // energy lift `N_len = 2 lift / A^2` and the mixed sum from their current against the moved
    // ground state's, `N_mix = g sum_v [<v~|P(k + A)|v~> - P_vv(k + A)] / A`, each the mean of `+A`
    // and `-A`. On 4^3 these are vera's 9.842146 / 9.841003 / 10.096240 (DZP).
    use mandacaru_core::{
        math::linalg::eigen,
        periodic::{
            optics::momentum::{band_momenta, one_center_momentum},
            setup::pulay::Mesh,
            wannier::sampling::Frame,
        },
    };
    type Matrix = nalgebra::DMatrix<C64>;
    let sizes: Vec<String> = std::env::var("SIZES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| vec!["DZP".into()]);
    let meshes: Vec<usize> = std::env::var("MESHES")
        .map(|v| v.split(',').map(|m| m.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![4]);
    let strength = 1e-3;
    for size in &sizes {
        let mut cell = silicon(4);
        cell.width = 1e-4;
        // QZ2P's SCF needs more than the fixture's 1 GB budget (vera's 2.5 GB).
        let solved = solve_on(&cell, "raw", size, Context::new(Some(2), 5 << 29).unwrap());
        let sampler = solved.sampler();
        let (ctx, crystal, setup) = (&solved.ctx, &*solved.crystal, &*solved.setup);
        let one_center = one_center_momentum(ctx, setup).unwrap();
        let grid_mesh = Mesh::new(ctx, &setup.grid).unwrap();
        let bands: Vec<usize> = (0..crystal.bands()).collect();
        let o = (crystal.electrons / 2.).round() as usize;
        let g = sampler.degeneracy();
        let shift = sampler.reciprocal.try_inverse().unwrap() * nalgebra::Vector3::new(strength, 0., 0.);
        for &n in &meshes {
            let started = std::time::Instant::now();
            let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
            let w = 1. / mesh.points() as f64;
            let (mut velocity, mut length, mut mixed) = (0., 0., 0.);
            let mut defect = 0f64;
            let chunk = 8;
            for first in (0..mesh.points()).step_by(chunk) {
                let last = (first + chunk).min(mesh.points());
                let mut fractional = vec![];
                for i in first..last {
                    let k = mesh.point(i);
                    fractional.push(k);
                    for sign in [1., -1.] {
                        fractional.push(std::array::from_fn(|a| k[a] + sign * shift[a]));
                    }
                }
                let weights = vec![1. / fractional.len() as f64; fractional.len()];
                let points = setup.matrices_at(ctx, &fractional, &weights).unwrap();
                for j in 0..last - first {
                    let cart = |f: [f64; 3]| -> [f64; 3] { (sampler.reciprocal * nalgebra::Vector3::from(f)).into() };
                    let here = Frame::from_point(crystal, &solved.state, &points[3 * j], 0, &bands).unwrap();
                    let p = &band_momenta(ctx, setup, &grid_mesh, &here, cart(fractional[3 * j]), &one_center)
                        .unwrap()
                        .matrices[0];
                    let e = here.energies.as_slice();
                    for v in 0..o {
                        for c in o..bands.len() {
                            velocity += w * g * 2. * p[(v, c)].norm_sqr() / (e[c] - e[v]);
                        }
                    }
                    for (s, sign) in [1f64, -1.].iter().enumerate() {
                        let at = 3 * j + 1 + s;
                        let there = Frame::from_point(crystal, &solved.state, &points[at], 0, &bands).unwrap();
                        let overlap = sampler.overlap(&here, &there, [sign * strength, 0., 0.]).unwrap();
                        let amplitudes = Matrix::from_fn(bands.len(), o, |m, v| overlap[(v, m)].conj());
                        let (values, vectors) = eigen(amplitudes.adjoint() * &amplitudes).unwrap();
                        for value in &values {
                            defect = defect.max((1. - value).abs());
                        }
                        let root = &vectors
                            * Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
                                o,
                                values.iter().map(|v| C64::new(1. / v.sqrt(), 0.)),
                            ))
                            * vectors.adjoint();
                        let moved = &amplitudes * root;
                        let levels = there.energies.as_slice();
                        let pt = &band_momenta(ctx, setup, &grid_mesh, &there, cart(fractional[at]), &one_center)
                            .unwrap()
                            .matrices[0];
                        let current = (moved.adjoint() * pt * &moved).trace().re;
                        let ground: f64 = (0..o).map(|v| pt[(v, v)].re).sum();
                        mixed += 0.5 * w * g * (current - ground) / (sign * strength);
                        let mut lift = 0.;
                        for v in 0..o {
                            for m in 0..bands.len() {
                                lift += moved[(m, v)].norm_sqr() * (levels[m] - levels[v]);
                            }
                        }
                        length += 0.5 * w * g * 2. * lift / (strength * strength);
                    }
                }
            }
            eprintln!(
                "Si {size} SCF 4^3, response {n}^3: velocity {velocity:.6}, mixed {mixed:.6}, length \
                 {length:.6} (N 8; transfer defect {defect:.2e}); {:.0} s",
                started.elapsed().as_secs_f64()
            );
        }
    }
}
