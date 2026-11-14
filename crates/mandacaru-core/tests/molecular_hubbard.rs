//! The molecular DFT+U shell (`molecular::hubbard`) hands the model occupations in the real
//! (Cartesian) harmonics of the projector, while the Liechtenstein tensor is written in the
//! complex ones. These tests hold the bridge (`unitary_real_to_complex`, the rotation in
//! `shell_response` and `shell_linear`) to three independent references:
//!
//! - the harmonics themselves, pointwise (`math::angular`);
//! - the real-harmonic Slater tensor (`Basis::Real`) and a transcription of the paper's energy
//!   that uses nothing of the model's evaluation;
//! - physics: a molecule's energy and force do not depend on its orientation, and with an
//!   imaginary Hermitian occupation change (which carries no density) the singlet and triplet
//!   `A - B` kernels are equal and equal the second difference of the energy.
//!
//! The Dudarev form is invariant under the basis change and must not move by one bit.
mod paw_molecular_support;

use mandacaru_core::{
    math::{angular, linalg::real_eigen},
    models::{
        hubbard::{self as model, Form, Hubbard, Spin as Channels},
        hubbard_liechtenstein::evaluate_with_basis,
        hubbard_slater::{Basis, Coupling, Ratios},
    },
    molecular::{
        excitations::{Coupling as Approximation, Response, Spin, Unrestricted},
        hubbard::{unitary_real_to_complex, MolecularHubbard, MolecularHubbardShell},
        MeanField, MolecularIntegrals, ScfOptions,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, DVector};
use paw_molecular_support::{analytic_shells, dataset, molecule_with};

const EV: f64 = 27.211386245988;

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
fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-12,
        max_iterations: 400,
        ..Default::default()
    }
}
fn flat(m: &DMatrix<C64>) -> Vec<C64> {
    let p = m.nrows();
    (0..p * p).map(|i| m[(i / p, i % p)]).collect()
}
fn unflat(v: &[C64], p: usize) -> DMatrix<C64> {
    DMatrix::from_fn(p, p, |a, b| v[a * p + b])
}
/// A real symmetric matrix with the spectrum `eigs` in a random basis.
fn random_symmetric(eigs: &[f64], seed: u64) -> DMatrix<f64> {
    let p = eigs.len();
    let mut r = Lcg(seed);
    let a = DMatrix::from_fn(p, p, |_, _| r.next());
    let (_, q) = real_eigen((&a + a.transpose()) * 0.5).unwrap();
    &q * DMatrix::from_diagonal(&DVector::from_column_slice(eigs)) * q.transpose()
}
fn spectrum(p: usize, top: f64) -> Vec<f64> {
    (0..p).map(|k| top - 0.8 * k as f64 / p as f64).collect()
}

// ------------------------------------------------------------------ the harmonics

#[test]
fn the_real_to_complex_unitary_is_the_relation_of_the_engines_harmonics() {
    // `Y_lm = sum_m' U[m, m'] S_lm'` at random directions, with `Y = angular::complex` and
    // `S = angular::cartesian`, the harmonics the projectors of the molecular shell are built on.
    let mut rng = Lcg(2026);
    for l in [2usize, 3] {
        let u = unitary_real_to_complex(l);
        let identity = (&u * u.adjoint()) - DMatrix::<C64>::identity(2 * l + 1, 2 * l + 1);
        assert!(
            identity.iter().all(|z| z.norm() < 1e-15),
            "l = {l}: not unitary"
        );
        let mut worst = 0f64;
        for _ in 0..12 {
            let point = [rng.next() + 1.5, rng.next(), rng.next() - 0.5];
            for m in -(l as i32)..=(l as i32) {
                let row = (m + l as i32) as usize;
                let rotated: C64 = (-(l as i32)..=(l as i32))
                    .map(|k| u[(row, (k + l as i32) as usize)] * angular::cartesian(l, k, point))
                    .sum();
                worst = worst.max((rotated - angular::complex(l, m, point)).norm());
            }
        }
        println!("l = {l}: |Y - U S| = {worst:.3e}");
        assert!(worst < 1e-14, "l = {l}: {worst:e}");
    }
}

// ------------------------------------------------------------------ the shell bridge

/// The paper's Liechtenstein energy of the occupations `ns` (one or two spin channels) in the
/// orbital basis of `coupling`, written without the model's evaluation.
fn paper_energy(coupling: &Coupling, u: f64, j: f64, ns: &[DMatrix<f64>]) -> f64 {
    let p = coupling.p();
    let slater = coupling
        .integrals(u, j, &Ratios::default_for(coupling.l()).unwrap())
        .unwrap();
    let mut v = vec![0.; p.pow(4)];
    for (i, &k) in coupling.orders().iter().enumerate() {
        let t = coupling.tensor(k).unwrap();
        for (x, v) in v.iter_mut().enumerate() {
            *v += slater[i] * t[x];
        }
    }
    let spins: Vec<&DMatrix<f64>> = if ns.len() == 1 {
        vec![&ns[0], &ns[0]]
    } else {
        vec![&ns[0], &ns[1]]
    };
    let total = spins[0] + spins[1];
    let mut e = 0.;
    for (a, b, c, d) in (0..p.pow(4)).map(|x| (x / p.pow(3), x / p.pow(2) % p, x / p % p, x % p)) {
        let vv = v[((a * p + b) * p + c) * p + d];
        e += vv * total[(a, c)] * total[(b, d)];
        for s in &spins {
            e -= vv * s[(a, d)] * s[(b, c)];
        }
    }
    let (up, down) = (spins[0].trace(), spins[1].trace());
    let n = up + down;
    0.5 * e - (0.5 * u * n * (n - 1.) - 0.5 * j * (up * (up - 1.) + down * (down - 1.)))
}

fn shell_of(form: Form, l: usize, u: f64, j: f64) -> (MolecularHubbard, MolecularHubbardShell) {
    let p = 2 * l + 1;
    let shell = MolecularHubbardShell {
        atom: 0,
        symbol: "Cu".into(),
        l,
        u,
        j,
        weights: vec![1.],
        q_ao: DMatrix::zeros(1, p),
        q: DMatrix::zeros(1, p),
    };
    (
        MolecularHubbard {
            form,
            shells: vec![shell.clone()],
        },
        shell,
    )
}

#[test]
fn the_liechtenstein_energy_of_real_harmonic_occupations_is_the_real_basis_energy() {
    // G-M4 of `dev_plan/DFTU_PLAN.md`: `shell_response` at occupations in the real harmonics
    // equals the real-harmonic tensor of the model and the paper's formula on that tensor, energy
    // and potential, for d and f, restricted and unrestricted, with and without J.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let real = Coupling::new(&ctx, l, Basis::Real).unwrap();
        for (u, j) in [(0.7, 0.0), (0.7, 0.1), (0.4, 0.4)] {
            let (hubbard, shell) = shell_of(Form::LiechtensteinFll, l, u, j);
            for channels in [1usize, 2] {
                let ns: Vec<DMatrix<f64>> = (0..channels)
                    .map(|c| random_symmetric(&spectrum(p, 0.9 - 0.1 * c as f64), 7 + c as u64))
                    .collect();
                let (energy, potentials) = hubbard.shell_response(&ctx, &shell, &ns).unwrap();
                let spin = if channels == 1 {
                    Channels::Restricted
                } else {
                    Channels::Unrestricted
                };
                let lifted: Vec<Vec<C64>> = ns
                    .iter()
                    .map(|n| flat(&n.map(|x| C64::new(x, 0.))))
                    .collect();
                let refs: Vec<&[C64]> = lifted.iter().map(|v| v.as_slice()).collect();
                let reference =
                    evaluate_with_basis(&ctx, u, j, l, Basis::Real, spin, &refs).unwrap();
                let paper = paper_energy(&real, u, j, &ns);
                println!(
                    "l = {l} U = {u} J = {j} channels = {channels}: E {energy:.12e}, \
                     |E - E_real tensor| {:.2e}, |E - paper| {:.2e}",
                    (energy - reference.energy).abs(),
                    (energy - paper).abs()
                );
                assert!(
                    (energy - reference.energy).abs() < 1e-12,
                    "l = {l}, J = {j}"
                );
                assert!((energy - paper).abs() < 1e-12, "l = {l}, J = {j}");
                for (c, v) in potentials.iter().enumerate() {
                    let expected = unflat(reference.potentials[c].as_slice(), p);
                    assert!(
                        (v.map(|x| C64::new(x, 0.)) - &expected).camax() < 1e-12,
                        "l = {l}, J = {j}, channel {c}: potential"
                    );
                }
                // E is nothing if V is not its derivative: a symmetric step of the first channel.
                let step = random_symmetric(&spectrum(p, 0.3), 31);
                let h = 1e-5;
                let moved = |s: f64| {
                    let mut moved = ns.clone();
                    moved[0] += &step * s;
                    hubbard.shell_response(&ctx, &shell, &moved).unwrap().0
                };
                let derivative = (moved(h) - moved(-h)) / (2. * h);
                // A restricted matrix stands for both spins: `E` counts it twice.
                let analytic = (potentials[0].transpose() * &step).trace() * (3 - channels) as f64;
                assert!(
                    (derivative - analytic).abs() < 1e-8 * analytic.abs().max(1.),
                    "l = {l}, J = {j}: dE = {derivative:e} against tr[V dn] = {analytic:e}"
                );
            }
        }
    }
}

/// The matrix `D` of the real harmonics under the rotation `r`: `S_k(r u) = sum_k' D[k, k'] S_k'(u)`,
/// fitted on random directions with the engine's own `angular::cartesian`.
fn harmonic_rotation(l: usize, r: &[[f64; 3]; 3]) -> DMatrix<f64> {
    let p = 2 * l + 1;
    let mut rng = Lcg(77);
    let points: Vec<[f64; 3]> = (0..4 * p)
        .map(|_| [rng.next() + 0.3, rng.next() - 0.2, rng.next() + 0.1])
        .collect();
    let value = |u: [f64; 3], k: usize| angular::cartesian(l, k as i32 - l as i32, u);
    let basis = DMatrix::from_fn(points.len(), p, |i, k| value(points[i], k));
    let turned = DMatrix::from_fn(points.len(), p, |i, k| {
        let u = points[i];
        value(
            [0, 1, 2].map(|a| (0..3).map(|b| r[a][b] * u[b]).sum::<f64>()),
            k,
        )
    });
    // Least squares: D^T = (B^T B)^-1 B^T T.
    let d = (basis.transpose() * &basis).try_inverse().unwrap() * basis.transpose() * &turned;
    let d = d.transpose();
    let check = (&turned - &basis * d.transpose()).amax();
    assert!(check < 1e-12, "l = {l}: the fit of D leaves {check:e}");
    d
}

#[test]
fn the_liechtenstein_shell_is_invariant_under_a_rotation_of_the_molecule() {
    // A rotation of space turns the real-harmonic occupations by `D n D^T` and the potential the
    // same way. A wrong sign or order of any `m` breaks this for a tilted axis, which the mirror
    // planes of a bond along z or x never probe. Before the rotation `E` moved by 1.5e-2 here (l = 2).
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    // R = Rz(0.7) Ry(1.1) Rx(-0.4): a generic rotation, no axis kept.
    let (a, b, c) = (0.7f64, 1.1f64, -0.4f64);
    let rz = [
        [a.cos(), -a.sin(), 0.],
        [a.sin(), a.cos(), 0.],
        [0., 0., 1.],
    ];
    let ry = [
        [b.cos(), 0., b.sin()],
        [0., 1., 0.],
        [-b.sin(), 0., b.cos()],
    ];
    let rx = [
        [1., 0., 0.],
        [0., c.cos(), -c.sin()],
        [0., c.sin(), c.cos()],
    ];
    let mul = |x: [[f64; 3]; 3], y: [[f64; 3]; 3]| {
        [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| x[i][k] * y[k][j]).sum::<f64>()))
    };
    let r = mul(rz, mul(ry, rx));
    for l in [2usize, 3] {
        let p = 2 * l + 1;
        let d = harmonic_rotation(l, &r);
        assert!((&d * d.transpose() - DMatrix::<f64>::identity(p, p)).amax() < 1e-12);
        let (hubbard, shell) = shell_of(Form::LiechtensteinFll, l, 0.7, 0.1);
        let ns: Vec<DMatrix<f64>> = (0..2)
            .map(|c| random_symmetric(&spectrum(p, 0.9 - 0.1 * c as f64), 17 + c as u64))
            .collect();
        let turned: Vec<DMatrix<f64>> = ns.iter().map(|n| &d * n * d.transpose()).collect();
        let (e, v) = hubbard.shell_response(&ctx, &shell, &ns).unwrap();
        let (e_turned, v_turned) = hubbard.shell_response(&ctx, &shell, &turned).unwrap();
        let moved = (e - e_turned).abs();
        let potential = v
            .iter()
            .zip(&v_turned)
            .map(|(v, w)| (&d * v * d.transpose() - w).amax())
            .fold(0., f64::max);
        println!("l = {l}: |E - E(rotated)| {moved:.3e}, |D V D^T - V(rotated)| {potential:.3e}");
        assert!(moved < 1e-12, "l = {l}: E moved by {moved:e}");
        assert!(potential < 1e-12, "l = {l}: V moved by {potential:e}");
    }
}

#[test]
fn the_dudarev_shell_is_evaluated_on_the_matrices_as_they_are() {
    // The Dudarev form is basis invariant: no rotation, no bit moves.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for l in [1usize, 2, 3] {
        let p = 2 * l + 1;
        let (hubbard, shell) = shell_of(Form::DudarevFll, l, 0.7, 0.1);
        let ns = vec![
            random_symmetric(&spectrum(p, 0.9), 3),
            random_symmetric(&spectrum(p, 0.8), 4),
        ];
        let (energy, potentials) = hubbard.shell_response(&ctx, &shell, &ns).unwrap();
        let lifted: Vec<Vec<C64>> = ns
            .iter()
            .map(|n| flat(&n.map(|x| C64::new(x, 0.))))
            .collect();
        let refs: Vec<&[C64]> = lifted.iter().map(|v| v.as_slice()).collect();
        let direct = model::evaluate(
            &ctx,
            &Hubbard::with_form(0.7, 0.1, Form::DudarevFll).unwrap(),
            l,
            Channels::Unrestricted,
            &refs,
        )
        .unwrap();
        assert_eq!(energy.to_bits(), direct.energy.to_bits(), "l = {l}");
        for (c, v) in potentials.iter().enumerate() {
            for a in 0..p {
                for b in 0..p {
                    let expected = direct.potentials[c].as_slice()[a * p + b].re;
                    assert_eq!(v[(a, b)].to_bits(), expected.to_bits(), "l = {l}");
                }
            }
        }
    }
}

// ------------------------------------------------------------------ a molecule: CuH

/// CuH along `direction`, centered in the 17 Bohr cell (bond length 2.76 Bohr).
fn copper_hydride(
    ctx: &Context,
    hubbard: Option<(&str, f64, f64)>,
    direction: [f64; 3],
) -> (MolecularIntegrals, usize) {
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let copper = dataset::read(ctx, "lda-sr", "Cu");
    let electrons = copper.valence_charge.round() as usize + 1;
    let copper = dataset::species(ctx, &mut arrays, &copper);
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "lda-sr", "H"));
    let mut extra = vec![("allow_heavy_dft", serde_json::Value::Bool(true))];
    if let Some((form, u, j)) = hubbard {
        let request = format!(r#"{{"Cu": {{"l": "d", "U": {u}, "J": {j}, "form": "{form}"}}}}"#);
        extra.push(("hubbard", serde_json::Value::String(request)));
    }
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    let at = |sign: f64| direction.map(|x| 8.5 + sign * 1.38 * x / norm);
    let integrals = molecule_with(
        ctx,
        &[29, 1],
        &[at(-1.), at(1.)],
        17.,
        0.5,
        (vec![copper, hydrogen], vec![0, 1], arrays),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &extra,
    );
    (integrals, electrons)
}
fn converged(integrals: &MolecularIntegrals, electrons: usize) -> MeanField {
    let state = integrals
        .kohn_sham("lda", [electrons / 2, electrons / 2], false, &options())
        .unwrap();
    assert!(state.converged);
    state
}

/// The spread, over `directions`, of the total energy (Ha) and of the force on H along the bond
/// (eV/A), and the largest force component across the bond (eV/A), of CuH with `hubbard`.
fn orientation_spread(
    ctx: &Context,
    hubbard: (&str, f64, f64),
    directions: &[[f64; 3]],
) -> (f64, f64, f64) {
    let (mut energies, mut bonds, mut across) = (Vec::new(), Vec::new(), 0f64);
    for direction in directions {
        let (integrals, electrons) = copper_hydride(ctx, Some(hubbard), *direction);
        let state = converged(&integrals, electrons);
        energies.push(integrals.kohn_sham_energy("lda", &state.densities).unwrap());
        let gradient = integrals.gradient(&state).unwrap();
        let total = gradient.total.as_slice();
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        let force: Vec<f64> = (0..3)
            .map(|a| -total[3 + a] * EV / 0.529177210903)
            .collect();
        let along: f64 = (0..3).map(|a| force[a] * direction[a] / norm).sum();
        let transverse = (force.iter().map(|f| f * f).sum::<f64>() - along * along)
            .max(0.)
            .sqrt();
        println!(
            "{} {direction:?}: E {:.10} Ha, F on H along {along:.6}, across {transverse:.2e} eV/A",
            hubbard.0,
            energies.last().unwrap()
        );
        bonds.push(along);
        across = across.max(transverse);
    }
    let top = |v: &[f64]| {
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    (top(&energies), top(&bonds), across)
}

#[test]
#[ignore = "slow: 34 s (measured 2026-11-12, 33.6 s); run: cargo test --profile release-test -p mandacaru-core --test molecular_hubbard -- --ignored"]
fn the_energy_and_force_of_a_molecule_do_not_depend_on_its_axis_at_j_not_zero() {
    // Two sets of axes that the cubic grid maps onto each other: z and x, and the face diagonals
    // (1, 1, 0), (0, 1, 1), (1, 0, 1), which mix xy, yz and xz and the m = +-1 harmonics that z
    // and x never do. The spread inside a set is the grid's, which the Dudarev form, basis
    // invariant, measures on the same geometries; before the rotation the Liechtenstein spread
    // was 3.0e-4 Ha for z against x (U = 20, J = 3 eV). Bonds along a general direction put the
    // atoms off the 0.5 Bohr grid by more than any set's symmetry covers (an egg-box of 0.2 Ha for
    // Dudarev too), so the rotation of every `m` is also gated on the model
    // (`..._is_invariant_under_a_rotation_of_the_molecule`).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let axes = [[0., 0., 1.], [1., 0., 0.]];
    let diagonals = [[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]];
    for (name, set) in [("axes", &axes[..]), ("face diagonals", &diagonals[..])] {
        let dudarev = orientation_spread(&ctx, ("dudarev-fll", 20., 0.), set);
        let liechtenstein = orientation_spread(&ctx, ("liechtenstein-fll", 20., 3.), set);
        println!(
            "{name}: energy spread, bond force spread, transverse (Ha, eV/A): \
             Dudarev {:.3e} {:.3e} {:.3e}, Liechtenstein {:.3e} {:.3e} {:.3e}",
            dudarev.0, dudarev.1, dudarev.2, liechtenstein.0, liechtenstein.1, liechtenstein.2
        );
        assert!(
            liechtenstein.0 < 3. * dudarev.0 + 1e-7,
            "{name}: energy spread {:e} Ha against the Dudarev {:e}",
            liechtenstein.0,
            dudarev.0
        );
        assert!(
            liechtenstein.1 < 3. * dudarev.1 + 1e-5,
            "{name}: bond force spread {:e} against the Dudarev {:e} eV/A",
            liechtenstein.1,
            dudarev.1
        );
        // The force across the bond is the grid's alone, and only the on-grid axes can pin it.
        if name == "axes" {
            assert!(
                liechtenstein.2 < 3. * dudarev.2 + 1e-5,
                "{name}: transverse force {:e} against the Dudarev {:e} eV/A",
                liechtenstein.2,
                dudarev.2
            );
        }
    }
}

// ------------------------------------------------------------------ the +U kernel

/// `d^2 E / ds^2` of the paper's energy (real-harmonic tensor, occupations as they are) along the
/// Hermitian changes `dn` of the real-harmonic occupations `n0`, one matrix per spin channel.
fn second_difference(
    coupling: &Coupling,
    (u, j): (f64, f64),
    n0: &[DMatrix<f64>],
    dn: &[DMatrix<C64>],
    s: f64,
) -> f64 {
    // `E` of a complex Hermitian occupation: the paper's expression with `n^s_ab` complex. The
    // energy is quadratic, so the transcription below keeps the real part of each product.
    let at = |t: f64| -> f64 {
        let ns: Vec<DMatrix<C64>> = n0
            .iter()
            .zip(dn)
            .map(|(a, b)| a.map(|x| C64::new(x, 0.)) + b * C64::new(t, 0.))
            .collect();
        complex_paper_energy(coupling, u, j, &ns)
    };
    (at(s) + at(-s) - 2. * at(0.)) / (s * s)
}
fn complex_paper_energy(coupling: &Coupling, u: f64, j: f64, ns: &[DMatrix<C64>]) -> f64 {
    let p = coupling.p();
    let slater = coupling
        .integrals(u, j, &Ratios::default_for(coupling.l()).unwrap())
        .unwrap();
    let mut v = vec![0.; p.pow(4)];
    for (i, &k) in coupling.orders().iter().enumerate() {
        let t = coupling.tensor(k).unwrap();
        for (x, v) in v.iter_mut().enumerate() {
            *v += slater[i] * t[x];
        }
    }
    let spins: Vec<&DMatrix<C64>> = if ns.len() == 1 {
        vec![&ns[0], &ns[0]]
    } else {
        vec![&ns[0], &ns[1]]
    };
    let total = spins[0] + spins[1];
    let mut e = 0.;
    for (a, b, c, d) in (0..p.pow(4)).map(|x| (x / p.pow(3), x / p.pow(2) % p, x / p % p, x % p)) {
        let vv = v[((a * p + b) * p + c) * p + d];
        e += vv * (total[(a, c)] * total[(b, d)]).re;
        for s in &spins {
            e -= vv * (s[(a, d)] * s[(b, c)]).re;
        }
    }
    let (up, down) = (spins[0].trace().re, spins[1].trace().re);
    let n = up + down;
    0.5 * e - (0.5 * u * n * (n - 1.) - 0.5 * j * (up * (up - 1.) + down * (down - 1.)))
}
fn trial(n: usize, seed: u64) -> DVector<f64> {
    let mut r = Lcg(seed);
    DVector::from_fn(n, |_, _| r.next())
}
fn dense(response: &Response<'_>) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = response.dimension();
    let identity = DMatrix::identity(n, n);
    (
        response.sum(&identity).unwrap(),
        response.difference(&identity).unwrap(),
    )
}
/// The smallest `A - B` entry (Hartree) a +U kernel test accepts: measurably nonzero.
const MIN_HUBBARD_KERNEL: f64 = 1e-5;

#[test]
fn the_liechtenstein_antisymmetric_kernel_is_the_second_difference_of_the_physical_energy() {
    // An imaginary Hermitian change of the occupations carries no density: with a pure
    // functional the whole of `A - B - Delta` is the +U term, so the singlet and the triplet
    // kernels are equal, and the bilinear form `4 y1 (A - B - Delta) y2` is the mixed second
    // difference of the Liechtenstein energy of the real-harmonic tensor, a formula that shares
    // nothing with the rotation under test. CuH, U = 20 eV, J = 3 eV (before the rotation:
    // singlet against triplet 1.1e-3 Ha, the oracle off by 23 to 100 percent).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let (integrals, electrons) =
        copper_hydride(&ctx, Some(("liechtenstein-fll", 20., 3.)), [0., 0., 1.]);
    let hubbard = integrals
        .corrections
        .as_ref()
        .unwrap()
        .hubbard
        .as_ref()
        .unwrap()
        .clone();
    let shell = &hubbard.shells[0];
    let real = Coupling::new(&ctx, 2, Basis::Real).unwrap();
    let state = converged(&integrals, electrons);
    let mut singlet = None;
    for spin in [Spin::Singlet, Spin::Triplet] {
        let response = Response::new(&integrals, &state, spin, Approximation::Adiabatic).unwrap();
        let (o, v) = (response.occupied(), response.virtual_orbitals());
        let c = response.orbitals();
        let (po, pv) = (
            c.columns(0, o).transpose() * &shell.q,
            c.columns(o, v).transpose() * &shell.q,
        );
        let (_, difference) = dense(&response);
        let delta = DMatrix::from_diagonal(&DVector::from_column_slice(response.delta()));
        let kernel = &difference - &delta;
        assert!(
            kernel.amax() > MIN_HUBBARD_KERNEL,
            "{spin:?}: the +U kernel vanishes"
        );
        match &singlet {
            None => singlet = Some(kernel.clone()),
            Some(first) => {
                let gap = (first - &kernel).amax();
                println!(
                    "singlet against triplet A - B - Delta: {gap:.3e} of {:.3e}",
                    kernel.amax()
                );
                assert!(
                    gap < 1e-12 * kernel.amax(),
                    "singlet and triplet kernels differ by {gap:e}"
                );
            }
        }
        let n0 = shell.q.transpose() * state.densities[0].map(|z| z.re) * &shell.q;
        let (mut worst, mut scale) = (0f64, 0f64);
        for pair in 0..6 {
            let (y1, y2) = (trial(o * v, 100 + pair), trial(o * v, 200 + pair));
            let change = |y: &DVector<f64>| -> Vec<DMatrix<C64>> {
                let d = po.transpose() * response.matrix(y.as_slice()) * &pv;
                let dn = (&d - d.transpose()).map(|x| C64::new(0., x));
                match spin {
                    Spin::Singlet => vec![dn],
                    _ => vec![dn.clone(), -&dn],
                }
            };
            let n0s = match spin {
                Spin::Singlet => vec![n0.clone()],
                _ => vec![n0.clone(), n0.clone()],
            };
            let q = |y: &DVector<f64>| {
                second_difference(&real, (shell.u, shell.j), &n0s, &change(y), 0.05)
            };
            let (sum, dif) = (&y1 + &y2, &y1 - &y2);
            let bilinear = 0.25 * (q(&sum) - q(&dif));
            let column = DMatrix::from_column_slice(o * v, 1, y2.as_slice());
            let engine = 4. * y1.dot(&response.antisymmetric_kernel(&column).unwrap().column(0));
            scale = scale.max(engine.abs());
            worst = worst.max((engine - bilinear).abs());
        }
        println!("{spin:?}: |kernel - oracle| {worst:.3e} of {scale:.3e}");
        assert!(
            scale > MIN_HUBBARD_KERNEL * 1e-2,
            "{spin:?}: kernel scale {scale:e}"
        );
        assert!(
            worst < 1e-9 * scale.max(1e-3),
            "{spin:?}: {worst:e} against a scale {scale:e}"
        );
    }
    // The unrestricted CuH+, spin channels apart.
    let cation = integrals
        .kohn_sham("lda", [electrons / 2, electrons / 2 - 1], true, &options())
        .unwrap();
    assert!(cation.converged);
    let response = Unrestricted::new(&integrals, &cation, Approximation::Adiabatic).unwrap();
    let n0: Vec<DMatrix<f64>> = (0..2)
        .map(|s| shell.q.transpose() * cation.densities[s].map(|z| z.re) * &shell.q)
        .collect();
    let (mut worst, mut scale) = (0f64, 0f64);
    for pair in 0..6 {
        let (y1, y2) = (
            trial(response.dimension(), 300 + pair),
            trial(response.dimension(), 400 + pair),
        );
        let change = |y: &DVector<f64>| -> Vec<DMatrix<C64>> {
            let blocks = response.matrices(y.as_slice());
            (0..2)
                .map(|s| {
                    let (o, v) = response.channel(s);
                    let c = response.orbitals(s);
                    let (po, pv) = (
                        c.columns(0, o).transpose() * &shell.q,
                        c.columns(o, v).transpose() * &shell.q,
                    );
                    let d = po.transpose() * &blocks[s] * &pv;
                    (&d - d.transpose()).map(|x| C64::new(0., x))
                })
                .collect()
        };
        let q =
            |y: &DVector<f64>| second_difference(&real, (shell.u, shell.j), &n0, &change(y), 0.05);
        let bilinear = 0.25 * (q(&(&y1 + &y2)) - q(&(&y1 - &y2)));
        let column = DMatrix::from_column_slice(y2.len(), 1, y2.as_slice());
        let kernel = response.antisymmetric_kernel(&column).unwrap();
        let (b1, bk) = (
            response.matrices(y1.as_slice()),
            response.matrices(kernel.column(0).as_slice()),
        );
        let engine = 2. * (0..2).map(|s| b1[s].dot(&bk[s])).sum::<f64>();
        scale = scale.max(engine.abs());
        worst = worst.max((engine - bilinear).abs());
    }
    println!("CuH+: |kernel - oracle| {worst:.3e} of {scale:.3e}");
    assert!(
        scale > MIN_HUBBARD_KERNEL * 1e-2,
        "CuH+: kernel scale {scale:e}"
    );
    assert!(
        worst < 1e-9 * scale.max(1e-3),
        "CuH+: {worst:e} against a scale {scale:e}"
    );
}
