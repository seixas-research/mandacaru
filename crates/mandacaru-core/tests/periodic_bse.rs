//! The Bethe-Salpeter excitons of a crystal (`periodic::optics::bse`, TDDFT_PLAN.md T5b), checked
//! by the gates of section 4.4:
//!
//! - E0: without kernel the excitonic tensor is T5a's independent-particle tensor;
//! - E1: the full (coupling) exchange-only equation is T5a's Dyson equation with local fields,
//!   by the Woodbury oracle and by the production coupling form; with the direct term the
//!   coupling form's eigenpairs are its resolvent;
//! - E3: a two-band Wannier-Mott model through the same head average, contraction and solvers
//!   gives the hydrogenic 1s binding `mu / (2 eps^2)` and a 2s near a quarter of it;
//! - E4, E5: a bound singlet below the continuum's onset in LiH and LiF on two meshes, the triplet
//!   at or below it, the Wannier-Mott estimate `mu / (2 eps_M^2)` from the onset's band masses
//!   beside it (recorded, `#[ignore]`);
//! - E6: `H` Hermitian, Davidson equal to the dense roots, Haydock equal to the eigenvector sum;
//! - E7: the operator does not depend on the layout.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::{
    periodic::{
        bloch_mesh::MeshIndex,
        optics::{
            bse::{
                self, contract, coupled_tensor, operator, solve_davidson, solve_dense, Request,
                Spin,
            },
            dielectric::{dielectric, pairs, Kernel, Request as Optical},
        },
        screening::interaction::{cell_average, folded, head_average, vector},
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

const HARTREE_EV: f64 = 27.211386245988;

fn worst(a: &[[[C64; 3]; 3]], b: &[[[C64; 3]; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            (0..3)
                .flat_map(move |i| (0..3).map(move |j| (x[i][j] - y[i][j]).norm() / y[0][0].norm()))
        })
        .fold(0f64, f64::max)
}

#[test]
fn without_a_kernel_the_excitons_are_the_independent_transitions() {
    // E0: H = D, so every exciton is one transition and the tensor is T5a's independent one.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let op = operator(
        &sampler,
        &mesh,
        &Request::new(Kernel::Independent, Spin::Singlet),
    )
    .unwrap();
    let all = op.dense().unwrap();
    let frequencies = [0., 0.1, 0.25, 0.4];
    let reference = dielectric(
        &sampler,
        &mesh,
        &Optical::new(frequencies.to_vec(), 0.05, Kernel::Independent),
    )
    .unwrap();
    let ours: Vec<_> = frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let difference = worst(&ours, &reference.tensor);
    let moment = bse::first_moment(&op);
    eprintln!(
        "E0 LiH: {} transitions, tensor difference {difference:.2e}, first moment {:?} against {:?}",
        op.energies.len(),
        [0, 1, 2].map(|a| moment[a][a]),
        [0, 1, 2].map(|a| reference.sum_rule[a][a])
    );
    assert!(difference < 1e-12, "{difference:e}");
    assert_eq!(op.hermiticity, 0.);
    for a in 0..3 {
        assert!((moment[a][a] - reference.sum_rule[a][a]).abs() < 1e-12 * reference.sum_rule[0][0]);
    }
}

#[test]
fn the_coupled_exchange_equation_is_the_local_field_dyson_equation() {
    // E1: the full (resonant and antiresonant, coupled) equation with the exchange alone is T5a's
    // Dyson equation on the same columns, by the Woodbury identity; the Tamm-Dancoff tensor of the
    // operator differs from it by the coupling only (recorded).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut optical = Optical::new(vec![0., 0.1, 0.25, 0.4], 0.05, Kernel::Hartree);
    optical.cutoff = 2.;
    let columns = pairs(&sampler, &mesh, &optical).unwrap();
    let reference = dielectric(&sampler, &mesh, &optical).unwrap();
    let coupled: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| coupled_tensor(&columns, C64::new(*w, 0.05)).unwrap())
        .collect();
    let difference = worst(&coupled, &reference.tensor);
    let mut request = Request::new(Kernel::Hartree, Spin::Singlet);
    request.cutoff = 2.;
    let op = operator(&sampler, &mesh, &request).unwrap();
    let all = op.dense().unwrap();
    let tda: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    eprintln!(
        "E1 LiH ({} transitions, {} vectors): coupled against Dyson {difference:.2e}; Tamm-Dancoff \
         against Dyson {:.2e}; eps_xx(0.05 i) Dyson {:.6}, coupled {:.6}, TDA {:.6}; independent \
         {:.6}",
        op.energies.len(),
        op.vectors,
        worst(&tda, &reference.tensor),
        reference.tensor[0][0][0],
        coupled[0][0][0],
        tda[0][0][0],
        dielectric(&sampler, &mesh, &Optical::new(vec![0.], 0.05, Kernel::Independent))
            .unwrap()
            .tensor[0][0][0]
    );
    assert!(difference < 1e-10, "{difference:e}");
}

#[test]
fn the_coupling_form_with_the_exchange_alone_is_the_dyson_equation() {
    // E1 for the production coupling form: with the exchange alone its tensor, summed over every
    // eigenpair of `L^dagger diag(1, -1) L`, is T5a's Dyson equation with local fields and the
    // Woodbury oracle `coupled_tensor`; its lowest root lies at or below the Tamm-Dancoff one.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut optical = Optical::new(vec![0., 0.1, 0.25, 0.4], 0.05, Kernel::Hartree);
    optical.cutoff = 2.;
    let reference = dielectric(&sampler, &mesh, &optical).unwrap();
    let columns = pairs(&sampler, &mesh, &optical).unwrap();
    let mut request = Request::new(Kernel::Hartree, Spin::Singlet);
    request.cutoff = 2.;
    request.coupling = true;
    let op = operator(&sampler, &mesh, &request).unwrap();
    let all = op.coupled().unwrap();
    let ours: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let oracle: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| coupled_tensor(&columns, C64::new(*w, 0.05)).unwrap())
        .collect();
    request.coupling = false;
    let tda = operator(&sampler, &mesh, &request)
        .unwrap()
        .dense()
        .unwrap();
    let symmetry = op.coupling.as_ref().unwrap().symmetry;
    eprintln!(
        "E1 coupling form, LiH ({} transitions): against Dyson {:.2e}, against the Woodbury \
         oracle {:.2e}; B symmetric to {symmetry:.1e}; lowest {:.10} Ha (Tamm-Dancoff {:.10})",
        op.energies.len(),
        worst(&ours, &reference.tensor),
        worst(&ours, &oracle),
        all.energies[0],
        tda.energies[0]
    );
    assert!(worst(&ours, &reference.tensor) < 1e-9);
    assert!(worst(&ours, &oracle) < 1e-9);
    assert!(all.energies[0] <= tda.energies[0] + 1e-12);
}

#[test]
fn the_coupling_form_with_the_direct_term_is_its_resolvent() {
    // The coupling form with exchange and the screened direct term on the H2 chain DZP (Gamma):
    // `M` Hermitian, `B` symmetric (time reversal and `W(r, r') = W(r', r)`), the eigenpair
    // tensor equal to the resolvent `s h^T (z - diag(1, -1) M)^-1 diag(1, -1) h^*` by one dense
    // solve per frequency, `X^dagger X - Y^dagger Y = 1` on every root, and the lowest root at
    // or below the Tamm-Dancoff one. LiH SZ on 2 x 1 x 1 (two mesh points, so `B` reads the
    // direct term across `k' != k`): `B` symmetric, and since its coarse-mesh head gives a
    // negative Tamm-Dancoff root, `M` is not positive definite and refuses by name.
    let unstable = solve(&lih("lda-sr"));
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    request.coupling = true;
    let across = operator(
        &unstable.sampler(),
        &MeshIndex::new([2, 1, 1], [0.; 3]).unwrap(),
        &request,
    )
    .unwrap();
    let symmetry = across.coupling.as_ref().unwrap().symmetry;
    let refusal = across.coupled().unwrap_err().to_string();
    eprintln!("LiH 2 x 1 x 1: B symmetric to {symmetry:.1e}; {refusal}");
    assert!(symmetry < 1e-8, "{symmetry:e}");
    assert!(refusal.contains("positive definite"), "{refusal}");
    let solved = solve_sized(&h2_chain("lda-sr", "lda"), "raw", "DZP");
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([1, 1, 1], [0.; 3]).unwrap();
    let op = operator(&sampler, &mesh, &request).unwrap();
    let coupling = op.coupling.as_ref().unwrap();
    let n = op.energies.len();
    let all = op.coupled().unwrap();
    let (a, b, a2) = (&op.hamiltonian, &coupling.resonant, &coupling.antiresonant);
    let signed = DMatrix::from_fn(2 * n, 2 * n, |i, j| {
        let value = match (i < n, j < n) {
            (true, true) => a[(i, j)],
            (true, false) => b[(i, j - n)],
            (false, true) => b[(j, i - n)].conj(),
            (false, false) => a2[(i - n, j - n)],
        };
        if i < n {
            value
        } else {
            -value
        }
    });
    let heads = |axis: usize| -> Vec<C64> {
        op.dipoles
            .iter()
            .chain(&coupling.dipoles)
            .map(|h| h[axis])
            .collect()
    };
    let frequencies = [0., 0.1, 0.2, 0.3];
    let resolvent: Vec<[[C64; 3]; 3]> = frequencies
        .iter()
        .map(|w| {
            let z = C64::new(*w, 0.05);
            let system = DMatrix::from_fn(2 * n, 2 * n, |i, j| {
                C64::new(f64::from(u8::from(i == j)), 0.) * z - signed[(i, j)]
            });
            let right = DMatrix::from_fn(2 * n, 3, |i, axis| {
                let h = heads(axis)[i].conj();
                if i < n {
                    h
                } else {
                    -h
                }
            });
            let solved = system.lu().solve(&right).unwrap();
            std::array::from_fn(|x| {
                std::array::from_fn(|y| {
                    let chi: C64 = heads(x)
                        .iter()
                        .enumerate()
                        .map(|(i, h)| h * solved[(i, y)])
                        .sum();
                    C64::new(f64::from(u8::from(x == y)), 0.) - chi * op.scale * (4. * PI)
                })
            })
        })
        .collect();
    let ours: Vec<_> = frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let y = all.antiresonant.as_ref().unwrap();
    let norms = (0..n)
        .map(|l| (all.vectors.column(l).norm_squared() - y.column(l).norm_squared() - 1.).abs())
        .fold(0f64, f64::max);
    request.coupling = false;
    let tda = operator(&sampler, &mesh, &request)
        .unwrap()
        .dense()
        .unwrap();
    eprintln!(
        "coupling form with the direct term, H2 chain DZP ({n} transitions): Hermiticity {:.2e}, B \
         symmetric to {:.1e}; eigenpairs against the resolvent {:.2e}; X^dagger X - Y^dagger Y - 1 \
         {norms:.1e}; lowest {:.10} Ha (Tamm-Dancoff {:.10}), largest |B| {:.3e} Ha",
        op.hermiticity,
        coupling.symmetry,
        worst(&ours, &resolvent),
        all.energies[0],
        tda.energies[0],
        b.iter().map(|v| v.norm()).fold(0f64, f64::max)
    );
    assert!(coupling.symmetry < 1e-8, "{:e}", coupling.symmetry);
    assert!(worst(&ours, &resolvent) < 1e-9);
    assert!(norms < 1e-9, "{norms:e}");
    assert!(all.energies[0] <= tda.energies[0] + 1e-12);
}

/// The screened-exchange energy `E_W = -(s / 2) sum_{k k'} sum_{i j occupied} M_{jk, ik'}^dagger
/// W(k' - k) M_{jk, ik'}` (`s = 1 / (N_k Omega)`, per spin or per spinor) of the occupied
/// orbitals moved to `phi_vk + eps sum_c Z_{vck} phi_ck`, unnormalized, so that `E_W(eps)` is a
/// quartic: the pair densities of the moved orbitals are the bilinear combinations
/// `sum_ab conj(U_k[j, a]) U_k'[i, b] M_{ak, bk'}` of every band pair's.
fn screened_exchange(
    parts: &bse::DirectParts,
    transitions: &[bse::Transition],
    z: &[C64],
    eps: C64,
    scale: f64,
) -> f64 {
    let (bands, occupied) = (parts.bands, parts.occupied[0]);
    let nk = transitions.iter().map(|t| t.k).max().unwrap() + 1;
    let identity = |j: usize, a: usize| C64::new(f64::from(u8::from(j == a)), 0.);
    let mut u: Vec<DMatrix<C64>> = (0..nk)
        .map(|_| DMatrix::from_fn(occupied, bands, identity))
        .collect();
    for (t, z) in transitions.iter().zip(z) {
        u[t.k][(t.valence, t.conduction)] += eps * z;
    }
    let mut energy = 0.;
    for (block, per_k) in parts.blocks.iter().zip(&parts.densities[0]) {
        for (k, kp, m) in per_k {
            let coefficients = DMatrix::from_fn(bands * bands, occupied * occupied, |ab, ji| {
                let (a, b) = (ab / bands, ab % bands);
                let (j, i) = (ji / occupied, ji % occupied);
                u[*k][(j, a)].conj() * u[*kp][(i, b)]
            });
            let moved = m * coefficients;
            let screened = &block.screened * &moved;
            energy += (0..moved.ncols())
                .map(|c| moved.column(c).dotc(&screened.column(c)).re)
                .sum::<f64>();
        }
    }
    -0.5 * scale * energy
}

/// `beta = Z^dagger B Z^*` of the direct part of `B` from the screened-exchange energy alone: the
/// `e^{-2i theta}` part of `d^2 E_W / d eps^2` along `e^{i theta} Z` is `2 Re(e^{-2i theta}
/// beta)`; every other part (the direct term of `A` and the screened exchange of the moved
/// conduction orbitals) does not depend on `theta`. Central differences at 1e-2 and 5e-3,
/// Richardson (exact for a quartic).
fn screened_exchange_coupling(
    parts: &bse::DirectParts,
    transitions: &[bse::Transition],
    z: &[C64],
    scale: f64,
) -> C64 {
    let zero = screened_exchange(parts, transitions, z, C64::new(0., 0.), scale);
    let second = |theta: f64| {
        let phase = C64::from_polar(1., theta);
        let at = |h: f64| {
            (screened_exchange(parts, transitions, z, phase * h, scale)
                + screened_exchange(parts, transitions, z, -phase * h, scale)
                - 2. * zero)
                / (h * h)
        };
        (4. * at(5e-3) - at(1e-2)) / 3.
    };
    C64::new(
        second(0.) - second(PI / 2.),
        second(PI / 4.) - second(3. * PI / 4.),
    ) / 4.
}

/// The direct part of the coupling block `B` of `solved` on `mesh` (cutoff 2): the triplet
/// operator's `B` (no exchange), or the singlet's less that of `kernel='hartree'` (its exchange).
fn direct_coupling(solved: &Solved, spin: Spin, mesh: &MeshIndex) -> (bse::Operator, DMatrix<C64>) {
    let mut request = Request::new(Kernel::Bse, spin);
    request.cutoff = 2.;
    request.coupling = true;
    let op = operator(&solved.sampler(), mesh, &request).unwrap();
    let mut b = op.coupling.as_ref().unwrap().resonant.clone();
    if spin == Spin::Singlet {
        request.kernel = Kernel::Hartree;
        let exchange = operator(&solved.sampler(), mesh, &request).unwrap();
        b -= &exchange.coupling.as_ref().unwrap().resonant;
    }
    (op, b)
}

#[test]
fn the_direct_part_of_the_coupling_block_is_the_second_derivative_of_the_screened_exchange() {
    // The independent check of the direct part of `B`: the coupling form's direct term is the
    // second derivative of one functional, the screened exchange `E_W` of the occupied orbitals
    // in the operator's own `W`, and its `B` part is the phase-2 part of that derivative along a
    // complex rotation of valence into conduction (`screened_exchange_coupling`). The oracle
    // reads `W` and the pair densities of every band pair (`bse::direct_parts`), not the
    // operator's band-index map, its `(k, k')` order, its conjugations, its sign or its
    // prefactor. LiH DZP on 2 x 1 x 1 (`B` across `k' != k`, nine conduction bands), the
    // triplet operator: every element of the symmetric part of `B` from unit and paired `Z`.
    // The spinor path with spin-orbit coupling is the slow Xe case below.
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let solved = solve_sized(&lih("lda-sr"), "raw", "DZP");
    let (op, b) = direct_coupling(&solved, Spin::Triplet, &mesh);
    let parts = bse::direct_parts(&solved.sampler(), &mesh, 2.).unwrap();
    let n = op.transitions.len();
    let scale = 1. / (mesh.points() as f64 * op.volume);
    let unit = |t: usize, u: usize| -> Vec<C64> {
        (0..n)
            .map(|i| C64::new(f64::from(u8::from(i == t) + u8::from(i == u && u != t)), 0.))
            .collect()
    };
    let diagonal: Vec<C64> = (0..n)
        .map(|t| screened_exchange_coupling(&parts, &op.transitions, &unit(t, t), scale))
        .collect();
    let mut oracle = DMatrix::<C64>::zeros(n, n);
    for t in 0..n {
        oracle[(t, t)] = diagonal[t];
        for u in t + 1..n {
            let paired = screened_exchange_coupling(&parts, &op.transitions, &unit(t, u), scale);
            oracle[(t, u)] = (paired - diagonal[t] - diagonal[u]) / 2.;
            oracle[(u, t)] = oracle[(t, u)];
        }
    }
    let symmetric = (&b + b.transpose()) * C64::new(0.5, 0.);
    let largest = symmetric.iter().map(|v| v.norm()).fold(0f64, f64::max);
    let miss = (&oracle - &symmetric)
        .iter()
        .map(|v| v.norm())
        .fold(0f64, f64::max)
        / largest;
    let crossing = (0..n)
        .flat_map(|t| (0..n).map(move |u| (t, u)))
        .filter(|&(t, u)| op.transitions[t].k != op.transitions[u].k)
        .map(|(t, u)| symmetric[(t, u)].norm())
        .fold(0f64, f64::max);
    eprintln!(
        "LiH DZP 2 x 1 x 1 ({n} transitions, {} bands, {} occupied): direct B against the \
         screened-exchange energy {miss:.2e} of its largest {largest:.3e} Ha (largest across \
         k' != k {crossing:.3e} Ha)",
        parts.bands, parts.occupied[0]
    );
    // Measured 2026-11-16: 9.89e-11 of 3.152e-2 Ha (across k' != k 2.332e-2 Ha). Red checks:
    // `c` and `c'` swapped in the block's index map 6.1e-1, the `(c v')` column read as
    // `(v c')` 1.9.
    assert!(miss < 1e-8, "{miss:e}");
    assert!(crossing > 1e-6 * largest, "{crossing:e}");
}

#[test]
#[ignore = "slow: about 11 s, the fcc Xe spin-orbit SCF and two coupling operators; run: cargo test --profile release-test -p mandacaru-core --test periodic_bse -- --ignored the_direct_part_of_a_spin_orbit"]
fn the_direct_part_of_a_spin_orbit_crystals_coupling_block_is_the_second_derivative_of_the_screened_exchange(
) {
    // The same oracle on spinor transitions with spin-orbit coupling: fcc Xe (`lda-fr`, SZP,
    // Gamma; the 5p split by 1.36 eV), `B` of the singlet operator less that of
    // `kernel='hartree'`, along three complex `Z`.
    let gamma = MeshIndex::new([1, 1, 1], [0.; 3]).unwrap();
    let solved = solve_scaled_spin_orbit(&xenon("lda-fr", true), "SZP", 1.);
    let (op, b) = direct_coupling(&solved, Spin::Singlet, &gamma);
    let parts = bse::direct_parts(&solved.sampler(), &gamma, 2.).unwrap();
    let scale = 1. / op.volume;
    let n = op.transitions.len();
    let largest = b.iter().map(|v| v.norm()).fold(0f64, f64::max);
    let misses: Vec<f64> = (1..=3)
        .map(|seed| {
            let z: Vec<C64> = (0..n)
                .map(|t| {
                    let x = (seed * 31 + t) as f64;
                    C64::new((1.3 * x + 0.7).sin(), (2.1 * x + 0.4).cos())
                })
                .collect();
            let expected: C64 = (0..n)
                .flat_map(|t| (0..n).map(move |u| (t, u)))
                .map(|(t, u)| z[t].conj() * b[(t, u)] * z[u].conj())
                .sum();
            let found = screened_exchange_coupling(&parts, &op.transitions, &z, scale);
            (found - expected).norm() / expected.norm()
        })
        .collect();
    eprintln!(
        "Xe SZP with spin-orbit coupling, Gamma ({n} transitions, {} spinor bands, {} occupied): \
         direct B against the screened-exchange energy {:.2e}, {:.2e}, {:.2e} (relative, three \
         Z; largest |B| {largest:.3e} Ha)",
        parts.bands, parts.occupied[0], misses[0], misses[1], misses[2]
    );
    // Measured 2026-11-16: 2.07e-11, 2.10e-9, 1.59e-10 (largest |B| 4.861e-2 Ha).
    assert!(misses.iter().all(|m| *m < 1e-8), "{misses:?}");
}

/// The model's direct element of every mesh vector `p` through the crate's head average and
/// contraction (one pair density, `G = 0` only), and the band energies `D_k = 1 + k^2`.
fn wannier_mott_terms(n: usize, a: f64, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let reciprocal = Matrix3::identity() * (2. * PI / a);
    let nk = mesh.points();
    let one = DMatrix::from_element(1, 1, C64::new(1., 0.));
    let dielectric = Matrix3::identity() * eps;
    let direct = (0..nk)
        .map(|p| {
            // The mean of 4 pi / (eps q^2) over each mesh vector's mini-zone cell, as the crystal's
            // W takes it (the self-similar head at p = 0).
            let w = if p == 0 {
                head_average(&reciprocal, &mesh, &dielectric, 8).unwrap()
            } else {
                cell_average(&reciprocal, &mesh, folded(&mesh, p), &dielectric, 8).unwrap()
            };
            let screened = DMatrix::from_element(1, 1, C64::new(w, 0.));
            contract(&one, &screened, &one, 1, 1).unwrap()[(0, 0)].re / (nk as f64 * a * a * a)
        })
        .collect();
    let energies = (0..nk)
        .map(|k| 1. + Vector3::from(vector(&reciprocal, &mesh, folded(&mesh, k))).norm_squared())
        .collect();
    (energies, direct)
}

/// The explicit model operator `H_{kk'} = D_k delta - w(k' - k)`.
fn wannier_mott_matrix(n: usize, a: f64, eps: f64) -> DMatrix<C64> {
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let (energies, direct) = wannier_mott_terms(n, a, eps);
    let nk = mesh.points();
    DMatrix::from_fn(nk, nk, |i, j| {
        let (at, bt) = (mesh.indices(i), mesh.indices(j));
        let p = mesh.index(std::array::from_fn(|x| (bt[x] + n - at[x]) % n));
        C64::new(if i == j { energies[i] } else { 0. } - direct[p], 0.)
    })
}

/// The same operator applied by a circular convolution on the mesh (`w` is even in `p`).
struct WannierMott {
    energies: Vec<f64>,
    transform: Vec<C64>,
    fft: std::sync::Mutex<mandacaru_core::math::fft::Fft3>,
}

impl mandacaru_core::math::davidson::Symmetric for WannierMott {
    fn dimension(&self) -> usize {
        self.energies.len()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.energies.clone()
    }
    fn apply(&self, block: &DMatrix<f64>) -> mandacaru_core::Result<DMatrix<f64>> {
        let mut out = DMatrix::zeros(block.nrows(), block.ncols());
        let mut fft = self.fft.lock().unwrap();
        for j in 0..block.ncols() {
            let mut work: Vec<C64> = block.column(j).iter().map(|v| C64::new(*v, 0.)).collect();
            fft.transform(&mut work, false)?;
            for (x, w) in work.iter_mut().zip(&self.transform) {
                *x *= w;
            }
            fft.transform(&mut work, true)?;
            for i in 0..block.nrows() {
                out[(i, j)] = self.energies[i] * block[(i, j)] - work[i].re;
            }
        }
        Ok(out)
    }
}

/// The lowest roots `(binding 1 - E, |sum_k A_k|^2)` of the matrix-free model on an `n^3` mesh.
fn wannier_mott(n: usize, a: f64, eps: f64, roots: usize) -> Vec<(f64, f64)> {
    let ctx = Context::new(Some(1), 1 << 31).unwrap();
    let (energies, direct) = wannier_mott_terms(n, a, eps);
    let mut fft = mandacaru_core::math::fft::Fft3::new(&ctx, [n; 3]).unwrap();
    let mut transform: Vec<C64> = direct.iter().map(|w| C64::new(*w, 0.)).collect();
    fft.transform(&mut transform, false).unwrap();
    let op = WannierMott {
        energies,
        transform,
        fft: std::sync::Mutex::new(fft),
    };
    let mut options = mandacaru_core::math::davidson::Options::new(roots);
    options.tolerance = 1e-8;
    options.max_iterations = 400;
    let solved = mandacaru_core::math::davidson::lowest(&op, &options).unwrap();
    assert!(solved.converged);
    (0..roots)
        .map(|l| {
            let sum: f64 = solved.vectors.column(l).iter().sum();
            (1. - solved.values[l], sum * sum)
        })
        .collect()
}

#[test]
fn the_wannier_mott_model_binds_a_hydrogenic_series() {
    // E3: masses 1 and 1 (mu = 1/2) and eps = 2: the 1s binding mu / (2 eps^2) = 1/16 Ha, the 2s
    // at a quarter of it and its weight |sum_k A_k|^2 (the s amplitude at r = 0) at an eighth;
    // Bohr radius eps / mu = 4, lattice constant 3 (the zone reaches k a_B = 4 pi / 3; the
    // supercell n a reaches 30 a_B at n = 40).
    let exact = 0.5 / (2. * 2. * 2.);
    // The explicit operator through the crate's dense and Davidson solvers, and the matrix-free
    // one, on the smallest mesh.
    let h = wannier_mott_matrix(8, 3., 2.);
    let dense = solve_dense(&h).unwrap();
    let iterative = solve_davidson(&h, 6, 1e-9, 400).unwrap();
    let free = wannier_mott(8, 3., 2., 6);
    for l in 0..6 {
        assert!((dense.energies[l] - iterative.energies[l]).abs() < 1e-9);
        assert!((1. - dense.energies[l] - free[l].0).abs() < 1e-8);
    }
    let mut errors = vec![];
    for n in [16usize, 24, 32, 40] {
        let roots = wannier_mott(n, 3., 2., 6);
        let first = roots[0];
        let second = roots[1..]
            .iter()
            .copied()
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap();
        eprintln!(
            "E3 n = {n}: 1s binding {:.6} Ha ({:+.3} % of mu / 2 eps^2 = {exact:.6}), 2s {:.6} Ha \
             (ratio {:.4} against 0.25), s weights ratio {:.4} against 0.125",
            first.0,
            100. * (first.0 / exact - 1.),
            second.0,
            second.0 / first.0,
            second.1 / first.1,
        );
        errors.push((
            (first.0 / exact - 1.).abs(),
            (second.0 / first.0 / 0.25 - 1.).abs(),
        ));
    }
    // The zone's truncation: the same n with a smaller lattice constant (the zone reaches
    // k a_B = 2 pi instead of 4 pi / 3; the supercell is 20 a_B instead of 30).
    let finer = wannier_mott(40, 2., 2., 1)[0].0;
    eprintln!(
        "E3 n = 40, a = 2: 1s binding {finer:.6} Ha ({:+.3} %)",
        100. * (finer / exact - 1.)
    );
    let last = errors[errors.len() - 1];
    assert!(last.0 < errors[0].0 && last.0 < 0.03, "{errors:?}");
    assert!(last.1 < errors[0].1 && last.1 < 0.1, "{errors:?}");
}

#[test]
fn the_operator_is_hermitian_and_its_solvers_agree() {
    // E6 on LiH 2 x 1 x 1 with every band and the direct term: H Hermitian to round-off,
    // Davidson on the real form equal to the dense roots, the Haydock spectrum (the whole Krylov
    // space) equal to the eigenvector sum.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    let op = operator(&sampler, &mesh, &request).unwrap();
    let n = op.energies.len();
    let all = op.dense().unwrap();
    let roots = 3.min(n);
    let iterative = solve_davidson(&op.hamiltonian, roots, 1e-9, 400).unwrap();
    let parity = (0..roots)
        .map(|l| (iterative.energies[l] - all.energies[l]).abs())
        .fold(0f64, f64::max);
    let haydock = op.haydock(&solved.ctx, n).unwrap();
    let frequencies = [0., 0.1, 0.2, 0.3];
    let exact: Vec<_> = frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let lanczos: Vec<_> = frequencies
        .iter()
        .map(|w| haydock.tensor(C64::new(*w, 0.05)))
        .collect();
    let spectra = worst(&lanczos, &exact);
    eprintln!(
        "E6 LiH ({n} transitions): Hermiticity {:.2e}; Davidson against dense {parity:.2e}; \
         Haydock against the eigenvector sum {spectra:.2e}; head {:.4} Ha Bohr^3 (eps_M {:.4}); \
         lowest {:?} Ha against onset {:.6}",
        op.hermiticity,
        op.head,
        op.static_tensor[0][0],
        &all.energies[..roots],
        op.onset
    );
    assert!(op.hermiticity < 1e-10, "{:e}", op.hermiticity);
    assert!(parity < 1e-8, "{parity:e}");
    assert!(spectra < 1e-9, "{spectra:e}");
}

#[test]
fn the_operator_does_not_depend_on_the_layout() {
    // E7: threads and kpts lanes change no bit of H.
    let cell = lih("lda-sr");
    let mut operators = vec![];
    for (threads, kpts) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let ctx = Context::new(Some(threads), 1 << 30)
            .unwrap()
            .with_layout(kpts, 1)
            .unwrap();
        let solved = solve_on(&cell, "raw", "SZ", ctx);
        let sampler = solved.sampler();
        let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
        let mut request = Request::new(Kernel::Bse, Spin::Singlet);
        request.cutoff = 2.;
        let op = operator(&sampler, &mesh, &request).unwrap();
        eprintln!(
            "E7 ({threads}, {kpts}): head {:e}, eps_M {:e}",
            op.head, op.static_tensor[0][0]
        );
        let exchange = operator(&sampler, &mesh, &{
            let mut r = request.clone();
            r.kernel = Kernel::Hartree;
            r
        })
        .unwrap()
        .hamiltonian;
        operators.push((op.hamiltonian, exchange));
    }
    for (h, x) in &operators[1..] {
        let direct = (h - &operators[0].0)
            .iter()
            .map(|v| v.norm())
            .fold(0f64, f64::max);
        let exchange = (x - &operators[0].1)
            .iter()
            .map(|v| v.norm())
            .fold(0f64, f64::max);
        eprintln!("E7: largest change of H {direct:e}, of the exchange-only H {exchange:e}");
    }
    for (h, x) in &operators[1..] {
        assert_eq!(x, &operators[0].1);
        assert_eq!(h, &operators[0].0);
    }
}

/// A rock-salt crystal (fcc, cubic constant `a` Bohr) with the cation at the origin.
fn rocksalt(symbols: [&'static str; 2], a: f64, grid: usize, mesh: usize) -> Cell {
    let h = a / 2.;
    Cell {
        library: "lda-sr",
        symbols: symbols.to_vec(),
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., h, h),
            Vector3::new(h, 0., h),
            Vector3::new(h, h, 0.),
        ]),
        grid: [grid; 3],
        fractional: vec![[0.; 3], [0.5; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [mesh; 3],
        hubbard: None,
        width: 1e-3,
    }
}

/// E5's band masses `(m_e, m_h)` (electron masses): the curvature of the conduction and valence
/// bands of the onset's transition at its mesh point, `1 / m = d^2 e / dk^2` by central
/// differences of 0.01 / Bohr along x, y and z, averaged over the three axes (an isotropic
/// estimate; at a degenerate band edge it is the curvature of the band the onset names).
fn band_masses(solved: &Solved, op: &bse::Operator, mesh: &MeshIndex) -> (f64, f64) {
    let sampler = solved.sampler();
    let t = (0..op.energies.len())
        .min_by(|&a, &b| op.energies[a].total_cmp(&op.energies[b]))
        .unwrap();
    let onset = op.transitions[t];
    let k = Vector3::from(mesh.point(onset.k));
    let to_fractional = sampler.reciprocal.try_inverse().unwrap();
    let bands = vec![onset.valence, onset.conduction];
    let at = |shift: Vector3<f64>| -> [f64; 2] {
        let f = k + to_fractional * shift;
        let point = MeshIndex::new([1, 1, 1], [f[0], f[1], f[2]]).unwrap();
        let frames = sampler
            .frames(&point, &[bands.clone()], |_, _| Ok(()))
            .unwrap();
        let e = frames[0][0].energies.as_slice();
        [e[0], e[1]]
    };
    let step = 0.01;
    let centre = at(Vector3::zeros());
    let mut inverse = [0f64; 2];
    for a in 0..3 {
        let mut d = Vector3::zeros();
        d[a] = step;
        let (plus, minus) = (at(d), at(-d));
        for s in 0..2 {
            inverse[s] += (plus[s] + minus[s] - 2. * centre[s]) / (step * step) / 3.;
        }
    }
    (1. / inverse[1], -1. / inverse[0])
}

/// The lowest singlet and triplet of `solved` on an `n^3` response mesh: binding against the
/// onset, participation ratio, and the Wannier-Mott estimate `mu / (2 eps_M^2)` from the band
/// masses of the onset (E5, a diagnostic).
fn excitons(
    solved: &Solved,
    n: usize,
    window: Option<(usize, usize)>,
    label: &str,
) -> (f64, f64, f64) {
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let started = std::time::Instant::now();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.window = window;
    let singlet = operator(&sampler, &mesh, &request).unwrap();
    let lowest = singlet.lowest(4, 1e-7, 400).unwrap();
    request.spin = Spin::Triplet;
    let triplet = operator(&sampler, &mesh, &request).unwrap();
    let lowest_triplet = triplet.lowest(1, 1e-7, 400).unwrap();
    // E6 on a window with a degenerate root and a complex H: Davidson on the real form at the
    // binding's default tolerance against the dense roots.
    let iterative = solve_davidson(&singlet.hamiltonian, 4, 1e-5, 400).unwrap();
    let parity = (0..4)
        .map(|l| (iterative.energies[l] - lowest.energies[l]).abs())
        .fold(0f64, f64::max);
    eprintln!("{label} {n}^3: Davidson (1e-5) against dense {parity:.2e} Ha");
    assert!(parity < 1e-8, "{parity:e}");
    assert!(singlet.hermiticity < 1e-10 && triplet.hermiticity < 1e-10);
    let a = lowest.vectors.column(0);
    let participation = 1. / a.iter().map(|x| x.norm_sqr().powi(2)).sum::<f64>();
    let binding = singlet.onset - lowest.energies[0];
    let eps = singlet.static_tensor[0][0];
    let (electron, hole) = band_masses(solved, &singlet, &mesh);
    let reduced = 1. / (1. / electron + 1. / hole);
    eprintln!(
        "{label} {n}^3 E5: m_e {electron:.4}, m_h {hole:.4}, mu {reduced:.4}; Wannier-Mott \
         mu / (2 eps_M^2) = {:.4} eV against the binding {:.4} eV",
        reduced / (2. * eps * eps) * HARTREE_EV,
        (singlet.onset - lowest.energies[0]) * HARTREE_EV
    );
    eprintln!(
        "{label} {n}^3 ({} transitions, window {window:?}): onset {:.4} eV; singlets {:?} eV; \
         binding {:.4} eV; triplet {:.4} eV (binding {:.4}); participation {:.1} of {}; eps_M \
         {eps:.4}; head {:.3}; Hermiticity {:.1e}; {:.0} s",
        singlet.energies.len(),
        singlet.onset * HARTREE_EV,
        lowest
            .energies
            .iter()
            .map(|e| (e * HARTREE_EV * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        binding * HARTREE_EV,
        lowest_triplet.energies[0] * HARTREE_EV,
        (triplet.onset - lowest_triplet.energies[0]) * HARTREE_EV,
        participation,
        singlet.energies.len(),
        singlet.head,
        singlet.hermiticity,
        started.elapsed().as_secs_f64()
    );
    (
        binding,
        lowest_triplet.energies[0] - lowest.energies[0],
        participation / singlet.energies.len() as f64,
    )
}

#[test]
#[ignore = "E4/E5 on LiH DZP, recorded: the 4^3 and 6^3 meshes (both hold X; about 10 minutes)"]
fn lithium_hydride_binds_an_exciton_below_the_continuum() {
    // E4 on LiH (rock salt, 4.084 A): the lowest singlet below the onset on two meshes, by more
    // than the change between them; the triplet at or below it; the weight concentrated.
    // DZP: an SZ basis gives LiH one conduction band, and an eps_M of 1.2.
    let solved = solve_sized(&rocksalt(["Li", "H"], 7.718, 24, 3), "raw", "DZP");
    let (b3, t3, r3) = excitons(&solved, 4, Some((1, 4)), "E4 LiH DZP");
    let (b4, t4, r4) = excitons(&solved, 6, Some((1, 4)), "E4 LiH DZP");
    eprintln!("E4 LiH: bindings {:.4} / {:.4} eV, triplet offsets {t3:.2e} / {t4:.2e}, participation fractions {r3:.3} / {r4:.3}", b3 * HARTREE_EV, b4 * HARTREE_EV);
    assert!(b4 > 0. && b4 > (b4 - b3).abs(), "{b3} {b4}");
    assert!(t3 <= 1e-10 && t4 <= 1e-10);
}

#[test]
#[ignore = "E4/E5 on LiF, recorded: two meshes (several minutes)"]
fn lithium_fluoride_binds_a_strong_exciton() {
    let size = std::env::var("SIZES").unwrap_or("SZ".into());
    let meshes: Vec<usize> = std::env::var("MESHES")
        .map(|v| v.split(',').map(|m| m.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![3, 4]);
    let solved = solve_sized(&rocksalt(["Li", "F"], 7.608, 28, 4), "raw", &size);
    let mut out = vec![];
    for n in meshes {
        out.push(excitons(
            &solved,
            n,
            Some((3, 3)),
            &format!("E4 LiF {size}"),
        ));
    }
    let b: Vec<f64> = out.iter().map(|o| o.0 * HARTREE_EV).collect();
    eprintln!("E4 LiF: bindings {b:?} eV");
    assert!(out.iter().all(|o| o.0 > 0.));
}

#[test]
#[ignore = "T5c against T5b, recorded: LiH DZP, LiF DZP and Si SZ on 4^3 (several minutes)"]
fn the_long_range_kernel_against_the_bethe_salpeter_equation() {
    // eps_M(0) and the first peak of Im eps (0 to 15 eV, damping 0.1 eV) for independent particles,
    // local fields, the long-range kernel with its fitted alpha, and the Bethe-Salpeter equation,
    // each on the same window and mesh.
    let cases = [
        (
            "LiH DZP",
            rocksalt(["Li", "H"], 7.718, 24, 4),
            "DZP",
            (1, 4),
        ),
        (
            "LiF DZP",
            rocksalt(["Li", "F"], 7.608, 28, 4),
            "DZP",
            (3, 3),
        ),
        (
            "Si SZ",
            {
                let mut si = rocksalt(["Si", "Si"], 10.26, 20, 4);
                si.fractional = vec![[0.; 3], [0.25; 3]];
                si
            },
            "SZ",
            (4, 4),
        ),
    ];
    let only = std::env::var("CASES").unwrap_or_default();
    let energies: Vec<f64> = (0..=300).map(|i| 0.05 * i as f64 / HARTREE_EV).collect();
    for (label, cell, size, window) in cases {
        if !only.is_empty() && !only.split(',').any(|c| label.starts_with(c)) {
            continue;
        }
        let started = std::time::Instant::now();
        let solved = solve_sized(&cell, "raw", size);
        let sampler = solved.sampler();
        let mesh = MeshIndex::new([4; 3], [0.; 3]).unwrap();
        let mut line = format!("T5c {label} 4^3 window {window:?}:");
        for kernel in [
            Kernel::Independent,
            Kernel::Hartree,
            Kernel::LongRange,
            Kernel::Bse,
        ] {
            let mut request = Optical::new(energies.clone(), 0.1 / HARTREE_EV, kernel);
            request.window = Some(window);
            let out = dielectric(&sampler, &mesh, &request).unwrap();
            let along = out.along(None).unwrap();
            let (peak, height) = energies
                .iter()
                .zip(&along)
                .map(|(e, v)| (e * HARTREE_EV, v.im))
                .fold(
                    (0., f64::NEG_INFINITY),
                    |best, x| if x.1 > best.1 { x } else { best },
                );
            let first = energies
                .iter()
                .zip(&along)
                .position(|(_, v)| v.im > 0.5 * height)
                .map(|i| energies[i] * HARTREE_EV)
                .unwrap_or(f64::NAN);
            line += &format!(
                " {}: eps_M(0) {:.4}, peak {peak:.2} eV ({height:.2}), half-height onset {first:.2} eV{};",
                kernel.name(),
                (out.static_tensor[0][0] + out.static_tensor[1][1] + out.static_tensor[2][2]) / 3.,
                out.alpha.map(|a| format!(", alpha {a:.4}")).unwrap_or_default()
            );
        }
        // The Tamm-Dancoff exchange-only tensor: separates the excitonic effect from the TDA's.
        let mut request = Request::new(Kernel::Hartree, Spin::Singlet);
        request.window = Some(window);
        let op = operator(&sampler, &mesh, &request).unwrap();
        let all = op.dense().unwrap();
        let tda: Vec<C64> = energies
            .iter()
            .map(|e| {
                let t = op.tensor(&all, C64::new(*e, 0.1 / HARTREE_EV));
                (t[0][0] + t[1][1] + t[2][2]) / 3.
            })
            .collect();
        let (peak, height) = energies
            .iter()
            .zip(&tda)
            .map(|(e, v)| (e * HARTREE_EV, v.im))
            .fold(
                (0., f64::NEG_INFINITY),
                |best, x| if x.1 > best.1 { x } else { best },
            );
        let t0 = op.tensor(&all, C64::new(0., 0.));
        line += &format!(
            " hartree (TDA): eps_M(0) {:.4}, peak {peak:.2} eV ({height:.2});",
            (t0[0][0].re + t0[1][1].re + t0[2][2].re) / 3.
        );
        eprintln!("{line} {:.0} s", started.elapsed().as_secs_f64());
    }
}

#[test]
fn the_lithium_hydride_sz_momentum_is_hermitian_where_it_vanishes_by_symmetry() {
    use mandacaru_core::periodic::optics::momentum::{one_center_momentum, transitions};
    // Rock-salt LiH in SZ has two s-like bands: at the inversion-invariant points of the mesh
    // `P` vanishes (largest element 4e-16) and was refused for 5.6e-2 of that, rounding, though
    // its grid part (A^dagger diag(w) A, real symbols) and its sphere part are Hermitian.
    let solved = solve_sized(&rocksalt(["Li", "H"], 7.718, 16, 4), "raw", "SZ");
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([4; 3], [0.; 3]).unwrap();
    let one_center = one_center_momentum(&solved.ctx, &solved.setup).unwrap();
    let block = one_center
        .iter()
        .map(|b| (b - b.adjoint()).camax() / b.camax().max(1e-300))
        .fold(0f64, f64::max);
    let all: Vec<Vec<usize>> = (0..sampler.channels())
        .map(|_| (0..solved.crystal.bands()).collect())
        .collect();
    let mut sphere = 0f64;
    sampler
        .visit(&mesh, &all, |_, frame| {
            for b in &one_center {
                let part = &frame.projections * b * frame.projections.adjoint();
                sphere = sphere.max((&part - part.adjoint()).camax());
            }
            Ok(())
        })
        .unwrap();
    let residual = transitions(&sampler, &mesh, [1., 0., 0.]).unwrap().residual;
    eprintln!(
        "LiH SZ grid 16^3, 4^3 mesh: Delta p anti-Hermitian {block:.2e} (relative), sphere part \
         {sphere:.2e} (absolute), P residual {residual:.2e}"
    );
    assert!(
        block < 1e-10 && sphere < 1e-12 && residual < 1e-6,
        "{residual:e}"
    );
}

/// T8 crystals, the collinear limit of the Bethe-Salpeter equation: LiH on `lda-fr` solved with
/// two-component states and its spin-orbit term zeroed (Kramers-unpolarized). Its operator, exchange
/// and direct terms on every spinor transition, has the collinear singlets once and the triplets
/// three times as its roots, and the collinear singlets' excitonic tensor; `spin='triplet'`
/// refuses by name.
#[test]
fn a_spinor_crystal_without_spin_orbit_coupling_has_the_collinear_excitons() {
    let collinear = solve_without_spin_orbit(&lih("lda-fr"));
    let spinor = solve_without_spin_orbit(&Cell {
        spinor: true,
        ..lih("lda-fr")
    });
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    let singlet = operator(&collinear.sampler(), &mesh, &request).unwrap();
    let mut triplet_request = request.clone();
    triplet_request.spin = Spin::Triplet;
    let triplet = operator(&collinear.sampler(), &mesh, &triplet_request).unwrap();
    let two = operator(&spinor.sampler(), &mesh, &request).unwrap();
    let (s, t, b) = (
        singlet.dense().unwrap(),
        triplet.dense().unwrap(),
        two.dense().unwrap(),
    );
    let mut expected: Vec<f64> = s.energies.clone();
    for e in &t.energies {
        expected.extend([*e; 3]);
    }
    expected.sort_by(f64::total_cmp);
    let roots = (0..expected.len())
        .map(|j| (expected[j] - b.energies[j]).abs())
        .fold(0f64, f64::max);
    let frequencies = [0., 0.1, 0.2, 0.3];
    let a: Vec<_> = frequencies
        .iter()
        .map(|w| singlet.tensor(&s, C64::new(*w, 0.05)))
        .collect();
    let c: Vec<_> = frequencies
        .iter()
        .map(|w| two.tensor(&b, C64::new(*w, 0.05)))
        .collect();
    let spectra = worst(&a, &c);
    eprintln!(
        "LiH BSE: {} spinor roots against {} singlets and 3 x {} triplets, worst {roots:.2e} Ha; \
         tensor {spectra:.2e}; head {:.6} / {:.6} Ha Bohr^3; Hermiticity {:.1e}; lowest {:?} / \
         singlet {:?} triplet {:?}",
        b.energies.len(),
        s.energies.len(),
        t.energies.len(),
        singlet.head,
        two.head,
        two.hermiticity,
        &b.energies[..4.min(b.energies.len())],
        &s.energies[..2.min(s.energies.len())],
        &t.energies[..2.min(t.energies.len())]
    );
    assert_eq!(b.energies.len(), expected.len());
    assert!(roots < 1e-8, "{roots:e}");
    assert!(spectra < 1e-8, "{spectra:e}");
    let mut refused = request.clone();
    refused.spin = Spin::Triplet;
    assert!(operator(&spinor.sampler(), &mesh, &refused)
        .err()
        .is_some_and(|e| e.to_string().contains("spin='singlet'")));
}

/// T5b, spin-polarized crystals, the collinear limit: LiH solved spin-polarized at zero moment has
/// the two equal channels of the closed shell. Its two-channel operator (the exchange across both
/// channels, the direct term within each) has the singlets and the triplets, each once, as its
/// roots and the singlets' excitonic tensor; `spin='triplet'` refuses by name.
#[test]
fn a_spin_polarized_crystal_has_the_singlets_and_the_triplets() {
    let closed = solve(&lih("lda-sr"));
    let polarized = solve(&Cell {
        spin: true,
        ..lih("lda-sr")
    });
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    let singlet = operator(&closed.sampler(), &mesh, &request).unwrap();
    let mut triplet_request = request.clone();
    triplet_request.spin = Spin::Triplet;
    let triplet = operator(&closed.sampler(), &mesh, &triplet_request).unwrap();
    let two = operator(&polarized.sampler(), &mesh, &request).unwrap();
    let (s, t, b) = (
        singlet.dense().unwrap(),
        triplet.dense().unwrap(),
        two.dense().unwrap(),
    );
    let mut expected: Vec<f64> = s.energies.iter().chain(&t.energies).copied().collect();
    expected.sort_by(f64::total_cmp);
    let roots = (0..expected.len())
        .map(|j| (expected[j] - b.energies[j]).abs())
        .fold(0f64, f64::max);
    let frequencies = [0., 0.1, 0.2, 0.3];
    let a: Vec<_> = frequencies
        .iter()
        .map(|w| singlet.tensor(&s, C64::new(*w, 0.05)))
        .collect();
    let c: Vec<_> = frequencies
        .iter()
        .map(|w| two.tensor(&b, C64::new(*w, 0.05)))
        .collect();
    let spectra = worst(&a, &c);
    eprintln!(
        "LiH two channels: {} roots against {} singlets and {} triplets, worst {roots:.2e} Ha; \
         tensor {spectra:.2e}; head {:.6} / {:.6} Ha Bohr^3; Hermiticity {:.1e}; channels of the \
         transitions {:?}",
        b.energies.len(),
        s.energies.len(),
        t.energies.len(),
        singlet.head,
        two.head,
        two.hermiticity,
        two.transitions
            .iter()
            .map(|x| x.channel)
            .collect::<Vec<_>>()
    );
    assert_eq!(b.energies.len(), expected.len());
    assert!(roots < 1e-8, "{roots:e}");
    assert!(spectra < 1e-8, "{spectra:e}");
    assert!(operator(&polarized.sampler(), &mesh, &triplet_request)
        .err()
        .is_some_and(|e| e.to_string().contains("spin='singlet'")));
}

/// T5b, spin-polarized crystals with unequal channels: an H3 radical in a cell (Gamma only, SZ, 2
/// up and 1 down: windows (2, 1) and (1, 2)). Its two-channel operator is the spin-conserving
/// block of the same cell solved as a spinor crystal magnetized along z without spin-orbit
/// coupling (the spin-flip transitions decouple from it), so every two-channel root is a root of
/// the spinor operator.
#[test]
fn the_two_channel_excitons_are_the_spin_conserving_ones_of_the_spinor_crystal() {
    let cell = Cell {
        symbols: vec!["H", "H", "H"],
        fractional: vec![
            [1.4 / 6., 0.5, 0.5],
            [3.0 / 6., 0.55, 0.5],
            [4.6 / 6., 0.5, 0.52],
        ],
        moments: vec![1., 0., 0.],
        ..h2_chain("lda-fr", "lda")
    };
    let polarized = solve_without_spin_orbit(&Cell {
        spin: true,
        ..cell.clone()
    });
    let spinor = solve_without_spin_orbit(&Cell {
        spinor: true,
        ..cell
    });
    let mesh = MeshIndex::new([1, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    let two = operator(&polarized.sampler(), &mesh, &request).unwrap();
    let whole = operator(&spinor.sampler(), &mesh, &request).unwrap();
    let (a, b) = (two.dense().unwrap(), whole.dense().unwrap());
    let worst = a
        .energies
        .iter()
        .map(|e| {
            b.energies
                .iter()
                .map(|f| (e - f).abs())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0f64, f64::max);
    let channels: Vec<usize> = two.transitions.iter().map(|t| t.channel).collect();
    eprintln!(
        "H3 cell: two-channel roots {:?} (transitions by channel {channels:?}, windows of the \
         columns {:?}); spinor roots {:?}; worst distance to a spinor root {worst:.2e} Ha; heads \
         {:.6} / {:.6}",
        a.energies, two.window, b.energies, two.head, whole.head
    );
    assert_eq!(channels.iter().filter(|&&c| c == 0).count(), 2);
    assert_eq!(channels.iter().filter(|&&c| c == 1).count(), 2);
    assert!(worst < 1e-7, "{worst:e}");
}
