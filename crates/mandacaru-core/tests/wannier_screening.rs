//! The screened downfold of a Wannier fragment (`periodic::wannier::downfold`, stage S2) on an H2
//! molecule in a chain cell (a mechanics state, a 3 x 1 x 1 mesh), through the production path.
//!
//! The polarizability's own correctness (the Adler-Wiser sum, the compressibility sum rule, the
//! constrained limits) is gated without a crystal in `screening_medium.rs`; here the PAW-augmented
//! chain is checked by what it must satisfy:
//!
//! * the transition density at `Q = 0` is the overlap: `rho_ab(0) = <a|S|b> = delta_ab` to 5e-5, which
//!   holds only if the smooth product, the compensation charges and the remainder are consistent;
//! * `chi(q)` is Hermitian and time-reversal symmetric (`chi_{GG'}(q)^* = chi_{-G,-G'}(-q)`);
//! * the limits of the screened interaction: every state in the correlated subspace screens nothing
//!   (`W = v`: the constrained downfold equals the bare one), no states in it gives the full RPA
//!   (the polarizabilities agree), and `U_RPA <= U_cRPA <= U_bare` for a fragment of one function.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;
mod wannier_downfold_support;

use mandacaru_core::{
    periodic::{
        screening::medium::gamma_polarizability,
        wannier::downfold::{Request, Screening},
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use wannier_downfold_support::{chain, chain_on, chain_with, request, supercell};

fn screened(screening: Screening, functions: Option<Vec<(usize, [i64; 3])>>) -> Request {
    Request {
        screening: Some(screening),
        ..request(functions)
    }
}

#[test]
fn the_transition_density_at_zero_wave_vector_is_the_overlap_of_the_states() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    for k in 0..3 {
        let norms = wannier.transition_norms(0, k, None).unwrap();
        let miss = (&norms - DMatrix::<C64>::identity(2, 2)).norm();
        println!("k = {k}: |rho(0) - 1| = {miss:.2e}");
        // the smooth product and the compensation charges give the overlap to 1e-8 (the S1 gate);
        // what is left is the remainder's monopole at Q = 0, the radial quadrature of the dataset's
        // partial waves (1e-5)
        assert!(miss < 5e-5, "k = {k}: {norms}");
    }
}

#[test]
fn the_polarizability_is_hermitian_and_time_reversal_symmetric() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    for screening in [Screening::Rpa, Screening::Crpa] {
        let plus = wannier
            .polarizability([1, 0, 0], screening, &[0], Some(2.0), None, None)
            .unwrap();
        let minus = wannier
            .polarizability([2, 0, 0], screening, &[0], Some(2.0), None, None)
            .unwrap();
        let scale = plus.chi.norm();
        assert!(
            scale > 1e-4,
            "an insulator's chi should not vanish: {scale}"
        );
        assert!((&plus.chi - plus.chi.adjoint()).norm() < 1e-10 * scale);
        // chi(q)_{GG'}^* = chi(-q)_{-G,-G'}: match the vectors Q' = -Q
        let find = |vectors: &[[f64; 3]], target: [f64; 3]| {
            vectors
                .iter()
                .position(|v| (0..3).all(|a| (v[a] - target[a]).abs() < 1e-9))
        };
        let mut matched = 0;
        for (i, qi) in plus.vectors.iter().enumerate() {
            let Some(mi) = find(&minus.vectors, [-qi[0], -qi[1], -qi[2]]) else {
                continue;
            };
            for (j, qj) in plus.vectors.iter().enumerate() {
                let Some(mj) = find(&minus.vectors, [-qj[0], -qj[1], -qj[2]]) else {
                    continue;
                };
                assert!(
                    (plus.chi[(i, j)].conj() - minus.chi[(mi, mj)]).norm() < 1e-9 * scale,
                    "{screening:?} ({i},{j})"
                );
                matched += 1;
            }
        }
        assert!(
            matched > plus.vectors.len(),
            "too few matched vectors: {matched}"
        );
    }
}

#[test]
fn an_empty_correlated_subspace_is_the_full_rpa_and_a_full_one_screens_nothing() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    let rpa = wannier
        .polarizability([1, 0, 0], Screening::Rpa, &[0, 1], Some(2.0), None, None)
        .unwrap();
    // cRPA with no function in the target: the weights are zero
    let empty = wannier
        .polarizability([1, 0, 0], Screening::Crpa, &[], Some(2.0), None, None)
        .unwrap();
    assert!((&rpa.chi - &empty.chi).norm() < 1e-14 * rpa.chi.norm());
    // both functions are all the bands: every transition is inside the subspace
    let full = wannier
        .polarizability([1, 0, 0], Screening::Crpa, &[0, 1], Some(2.0), None, None)
        .unwrap();
    assert!(
        full.chi.norm() < 1e-12 * rpa.chi.norm(),
        "chi = {}",
        full.chi.norm()
    );
    let bare = wannier.downfold(&request(None)).unwrap().problem;
    let constrained = wannier
        .downfold(&screened(Screening::Crpa, None))
        .unwrap()
        .problem;
    let miss = bare.wannier.two_body[0]
        .iter()
        .zip(&constrained.wannier.two_body[0])
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    assert!(
        miss < 1e-10,
        "W = v should leave the integrals alone: {miss}"
    );
    assert!((constrained.dielectric_head.unwrap() - 1.).abs() < 1e-10);
}

#[test]
fn a_fragment_of_one_function_is_screened_between_the_rpa_and_the_bare_interaction() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    let one = Some(vec![(0, [0; 3])]);
    let bare = wannier.downfold(&request(one.clone())).unwrap().problem;
    let rpa = wannier
        .downfold(&screened(Screening::Rpa, one.clone()))
        .unwrap()
        .problem;
    let crpa = wannier
        .downfold(&screened(Screening::Crpa, one))
        .unwrap()
        .problem;
    let ev = 27.211386;
    let u = |p: &mandacaru_core::periodic::wannier::downfold::problem::Problem| {
        p.wannier.two_body[0][0].re * ev
    };
    println!(
        "U bare {:.4} eV, cRPA {:.4} eV, RPA {:.4} eV; heads cRPA {:.5}, RPA {:.5}",
        u(&bare),
        u(&crpa),
        u(&rpa),
        crpa.dielectric_head.unwrap(),
        rpa.dielectric_head.unwrap()
    );
    assert!(u(&rpa) < u(&crpa) && u(&crpa) < u(&bare), "ordering of U");
    for head in [crpa.dielectric_head.unwrap(), rpa.dielectric_head.unwrap()] {
        assert!(
            head > 0. && head <= 1.,
            "the inverse dielectric head {head}"
        );
    }
    // the screened one-body part keeps the Fock identity: F[gamma] = h^KS
    let n = rpa.n;
    let g = &rpa.wannier.two_body[0];
    let gamma = &rpa.density_matrix[0];
    let mut fock = rpa.wannier.one_body[0].clone();
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    fock[(p, q)] += gamma[(r, s)]
                        * (g[((p * n + r) * n + q) * n + s]
                            - 0.5 * g[((p * n + r) * n + s) * n + q]);
                }
            }
        }
    }
    assert!((&fock - &rpa.kohn_sham[0]).norm() < 1e-10);
}

#[test]
fn the_screened_interaction_keeps_the_pair_symmetries_and_a_translated_fragment() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    let home = wannier
        .downfold(&screened(Screening::Crpa, None))
        .unwrap()
        .problem;
    let moved = wannier
        .downfold(&screened(
            Screening::Crpa,
            Some(vec![(0, [1, 0, 0]), (1, [1, 0, 0])]),
        ))
        .unwrap()
        .problem;
    let n = home.n;
    let g = &home.wannier.two_body[0];
    let scale = g.iter().map(|v| v.norm()).fold(0., f64::max);
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let at = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
                    assert!((g[at(p, q, r, s)] - g[at(q, p, s, r)]).norm() < 1e-12 * scale);
                    assert!((g[at(p, q, r, s)] - g[at(r, s, p, q)].conj()).norm() < 1e-12 * scale);
                }
            }
        }
    }
    let miss = g
        .iter()
        .zip(&moved.wannier.two_body[0])
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    assert!(
        miss < 1e-8 * scale,
        "translation moved the screened integrals by {miss}"
    );
}

#[test]
fn the_mesh_polarizability_equals_the_gamma_polarizability_of_the_supercell_crystal() {
    // The three-cell mesh of the chain is the Gamma point of its three-cell supercell: the same
    // Hamiltonian (the primitive potential tiled), the same states, the same transitions, counted
    // without the pairing of k with k + q. The augmentation (compensation charges and remainder
    // of every atom copy at Q != 0) and the Bloch phases of the frames must agree between the two.
    let (_ctx, wannier) = chain([3, 1, 1]);
    let (crystal, state) = supercell(&wannier, 3);
    // The levels on the primitive crystal's scale: the raw eigenvalues plus ITS reference (a
    // frame carries its own crystal's reference, which is taken out).
    let shift = wannier.crystal.reference;
    println!(
        "references: primitive {:.6}, supercell {:.6} Ha",
        wannier.crystal.reference, crystal.reference
    );
    let width = wannier.state.options.width;
    // How far the two Hamiltonians are apart: the sorted levels of the supercell's Gamma point
    // against the union of the mesh's. They differ by a nearly constant 1e-3 Ha (the setups of a
    // three-cell and a one-cell crystal do not share every discretization constant) that moves
    // each level by a spread (largest minus smallest difference) of 3e-4 Ha; the response inherits that spread through the energy
    // denominators, about 3e-4 of itself for a 0.5 Ha gap, which is the agreement asked of it.
    let spread = |a: &[f64], b: &[f64]| {
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        d.iter().cloned().fold(f64::MIN, f64::max) - d.iter().cloned().fold(f64::MAX, f64::min)
    };
    let bands: Vec<usize> = (0..crystal.bands()).collect();
    let frame = mandacaru_core::periodic::wannier::sampling::Frame::from_point(
        &crystal,
        &state,
        &crystal.points[0],
        0,
        &bands,
    )
    .unwrap();
    let mut levels: Vec<f64> = frame
        .energies
        .as_slice()
        .iter()
        .map(|e| e - crystal.reference + shift)
        .collect();
    let mut mesh_levels: Vec<f64> = wannier.channels[0]
        .frames
        .iter()
        .flat_map(|f| f.energies.as_slice().iter().copied())
        .collect();
    levels.sort_by(f64::total_cmp);
    mesh_levels.sort_by(f64::total_cmp);
    let level_miss = levels
        .iter()
        .zip(&mesh_levels)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let level_spread = spread(&levels, &mesh_levels);
    println!(
        "levels: supercell - mesh {level_miss:.2e} Ha (spread {level_spread:.2e}) over {} states",
        levels.len()
    );
    assert!(
        level_spread < 3e-4,
        "the Hamiltonians differ by more than the setups' constants"
    );
    let mut worst = 0f64;
    for q in [[0, 0, 0], [1, 0, 0], [2, 0, 0]] {
        let mesh = wannier
            .polarizability(q, Screening::Rpa, &[], Some(2.0), None, None)
            .unwrap();
        // Q = q + G of the mesh is B_s m with m = size * G + q in the supercell's lattice
        let m: Vec<[i64; 3]> = mesh
            .g
            .iter()
            .map(|g| [3 * g[0] + q[0], g[1] + q[1], g[2] + q[2]])
            .collect();
        let gamma = gamma_polarizability(
            &crystal,
            &state,
            0,
            shift,
            wannier.state.fermi,
            width,
            wannier.state.options.smearing,
            &m,
        )
        .unwrap();
        let scale = mesh.chi.norm();
        let miss = (&mesh.chi - &gamma).norm();
        println!(
            "q = {q:?}: {} vectors, |chi| {scale:.4e}, mesh - supercell {miss:.2e}",
            m.len()
        );
        assert!(scale > 1e-4);
        worst = worst.max(miss / scale);
    }
    assert!(
        worst < 5e-4,
        "the mesh and the supercell disagree by {worst:.2e}"
    );
}

/// The Fock matrix of channel `s` built from the physicists' integrals without the code under test:
/// `h_s + sum_t sum_rs gamma^t_rs <p r|q s>^{st} - sum_rs gamma^s_rs <p r|s q>^{ss}` (a restricted
/// problem's one block holds both spins: its exchange is halved).
fn fock(
    problem: &mandacaru_core::periodic::wannier::downfold::problem::Problem,
    s: usize,
) -> DMatrix<C64> {
    let n = problem.n;
    let spins = problem.spins;
    let at = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    let mut out = problem.wannier.one_body[s].clone();
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for u in 0..n {
                    for t in 0..spins {
                        out[(p, q)] += problem.density_matrix[t][(r, u)]
                            * problem.wannier.two_body[s * spins + t][at(p, r, q, u)];
                    }
                    let exchange = problem.density_matrix[s][(r, u)]
                        * problem.wannier.two_body[s * spins + s][at(p, r, u, q)];
                    out[(p, q)] -= if spins == 1 { 0.5 * exchange } else { exchange };
                }
            }
        }
    }
    out
}

#[test]
fn two_spin_channels_are_the_restricted_problem_without_moments_and_keep_the_fock_identity_with_them(
) {
    let (_a, restricted) = chain([3, 1, 1]);
    let (_b, nonmagnetic) = chain_with([3, 1, 1], Some([0., 0.]));
    for screening in [None, Some(Screening::Rpa)] {
        let ask = |w: &mandacaru_core::periodic::wannier::setup::Wannier| {
            w.downfold(&Request {
                screening,
                cutoff: screening.map(|_| 2.0),
                ..request(None)
            })
            .unwrap()
            .problem
        };
        let (one, two) = (ask(&restricted), ask(&nonmagnetic));
        assert_eq!((one.spins, two.spins), (1, 2));
        let scale = one.wannier.two_body[0]
            .iter()
            .map(|v| v.norm())
            .fold(0., f64::max);
        for s in 0..2 {
            // gamma^s is half the restricted (both-spin) matrix, the double counting and the
            // one-body part are the same operator
            assert!(
                (&two.density_matrix[s] * C64::new(2., 0.) - &one.density_matrix[0]).norm() < 1e-8
            );
            assert!((&two.kohn_sham[s] - &one.kohn_sham[0]).norm() < 1e-8);
            assert!(
                (&two.double_counting[s] - &one.double_counting[0]).norm() < 1e-8,
                "{screening:?}"
            );
            assert!((&two.wannier.one_body[s] - &one.wannier.one_body[0]).norm() < 1e-8);
            for t in 0..2 {
                let miss = two.wannier.two_body[s * 2 + t]
                    .iter()
                    .zip(&one.wannier.two_body[0])
                    .map(|(a, b)| (a - b).norm())
                    .fold(0., f64::max);
                assert!(miss < 1e-8 * scale, "block ({s},{t}) {screening:?}: {miss}");
            }
        }
    }
    // a magnetic start (the molecule falls back to its singlet, so the two channels stay equal
    // here; the antiferromagnetic case with different channels is the slow NiO test in
    // `test/integrations/test_wannier_screening.py`): each spin's Fock matrix is its Kohn-Sham
    // one, and swapping the electrons swaps the spin blocks
    let (_c, magnetic) = chain_with([3, 1, 1], Some([1., -1.]));
    let problem = magnetic
        .downfold(&Request {
            num_particles: Some([1, 1]),
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(Some(vec![(0, [0; 3])]))
        })
        .unwrap()
        .problem;
    assert_eq!(problem.spins, 2);
    for s in 0..2 {
        assert!(
            (fock(&problem, s) - &problem.kohn_sham[s]).norm() < 1e-10,
            "spin {s}"
        );
    }
    let n = problem.n;
    let scale = problem.wannier.two_body[0]
        .iter()
        .map(|v| v.norm())
        .fold(0., f64::max);
    for (s, t) in [(0usize, 1usize), (1, 0)] {
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for u in 0..n {
                        let at =
                            |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
                        let miss = (problem.wannier.two_body[s * 2 + t][at(p, q, r, u)]
                            - problem.wannier.two_body[t * 2 + s][at(q, p, u, r)])
                        .norm();
                        assert!(miss < 1e-12 * scale, "({s},{t}) {miss}");
                    }
                }
            }
        }
    }
}

#[test]
fn the_screened_downfold_is_bit_identical_for_one_two_and_four_workers() {
    // One crystal on a four-worker context (its SCF and bands are what they are: the solver's own
    // reductions depend on the thread count at rounding level, which is not this gate's subject),
    // the screening run on 1, 2 and 4 of its workers: PAW-augmented transition densities, the
    // head's shifted states (six points, one eigensolve each) and the contraction. chi, every integral and the head are the same
    // bits, because each transition density is computed alone and added in a fixed order.
    let (_ctx, wannier) = chain_on([6, 1, 1], None, 4);
    let run = |workers: usize| {
        let chi = wannier
            .polarizability(
                [1, 0, 0],
                Screening::Crpa,
                &[0],
                Some(2.0),
                None,
                Some(workers),
            )
            .unwrap()
            .chi;
        let problem = wannier
            .downfold(&Request {
                screening: Some(Screening::Crpa),
                cutoff: Some(2.0),
                workers: Some(workers),
                ..request(Some(vec![(0, [0; 3])]))
            })
            .unwrap()
            .problem;
        (chi, problem)
    };
    let (chi, problem) = run(1);
    let bits =
        |a: &C64, b: &C64| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits();
    assert!(chi.norm() > 1e-4);
    for workers in [2, 4] {
        let (other_chi, other) = run(workers);
        assert!(
            chi.iter().zip(other_chi.iter()).all(|(a, b)| bits(a, b)),
            "chi, {workers} workers"
        );
        assert!(
            problem.wannier.two_body[0]
                .iter()
                .zip(&other.wannier.two_body[0])
                .all(|(a, b)| bits(a, b)),
            "<pq|rs>, {workers} workers"
        );
        assert_eq!(
            problem.dielectric_head.unwrap().to_bits(),
            other.dielectric_head.unwrap().to_bits(),
            "head, {workers} workers"
        );
    }
}
