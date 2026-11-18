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
use wannier_downfold_support::{
    chain, chain_on, chain_set, chain_with, hcl_spinor, request, supercell,
};

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

#[test]
fn the_periodic_setup_rebuilt_from_a_snapshot_matches_the_screened_downfold_head() {
    use mandacaru_core::periodic::{
        setup::PeriodicSetup,
        wannier::{setup::Wannier, Options as WannierOptions},
    };
    let (_ctx, direct) = chain([3, 1, 1]);
    let expected = direct
        .downfold(&Request {
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(Some(vec![(0, [0; 3])]))
        })
        .unwrap();

    let rebuilt = std::sync::Arc::new(PeriodicSetup::from_crystal(&direct.crystal).unwrap());
    let req_json = serde_json::json!({
        "size": [3, 1, 1],
        "trials": [
            {"center": [2.3, 4., 4.], "orbital": "s", "label": "H0"},
            {"center": [3.7, 4., 4.], "orbital": "s", "label": "H1"},
        ],
    })
    .to_string();
    let from_snapshot = Wannier::build(
        direct.crystal.clone(),
        direct.state.clone(),
        rebuilt,
        &req_json,
        &WannierOptions::default(),
    )
    .unwrap();
    let got = from_snapshot
        .downfold(&Request {
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(Some(vec![(0, [0; 3])]))
        })
        .unwrap();

    assert!(
        (got.problem.dielectric_head.unwrap() - expected.problem.dielectric_head.unwrap()).abs()
            < 1e-13
    );
    assert_eq!(
        got.problem.dielectric_head.unwrap().to_bits(),
        expected.problem.dielectric_head.unwrap().to_bits()
    );
    let worst = got.problem.wannier.two_body[0]
        .iter()
        .zip(&expected.problem.wannier.two_body[0])
        .map(|(a, b)| (a - b).norm())
        .fold(0f64, f64::max);
    assert!(worst < 1e-13, "snapshot rebuilt cRPA diff: {worst:e}");
}

#[test]
fn spinor_crpa_downfold_matches_restricted_and_handles_soc_fragment() {
    let fock_spinor = |problem: &mandacaru_core::periodic::wannier::downfold::problem::Problem| {
        let n = problem.n;
        let g = &problem.wannier.two_body[0];
        let gamma = &problem.density_matrix[0];
        let mut f = problem.wannier.one_body[0].clone();
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        f[(p, q)] += gamma[(r, s)] * (g[idx(p, r, q, s)] - g[idx(p, r, s, q)]);
                    }
                }
            }
        }
        (&f - &problem.kohn_sham[0]).norm()
    };

    // Part A: zero-SOC spinor H2 (`lda-fr`) matches scalar-relativistic H2 (`lda-sr`) closely
    // and keeps exact spin symmetry + generalized spin-orbital Fock identity.
    let (_ctx_r, restricted_w) = chain_set([2, 1, 1], "lda-sr", false);
    let (_ctx_s, spinor_w) = chain_set([2, 1, 1], "lda-fr", true);
    let r_crpa = restricted_w
        .downfold(&Request {
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(None)
        })
        .unwrap()
        .problem;
    let s_crpa = spinor_w
        .downfold(&Request {
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(None)
        })
        .unwrap()
        .problem;
    assert!(s_crpa.spinor && s_crpa.n == 4);
    assert!(
        (s_crpa.dielectric_head.unwrap() - r_crpa.dielectric_head.unwrap()).abs() < 5e-3,
        "head mismatch: spinor {} vs restricted {}",
        s_crpa.dielectric_head.unwrap(),
        r_crpa.dielectric_head.unwrap()
    );
    assert!(
        (s_crpa.wannier.two_body[0][0] - r_crpa.wannier.two_body[0][0]).norm() < 5e-3,
        "cRPA U mismatch: spinor {} vs restricted {}",
        s_crpa.wannier.two_body[0][0],
        r_crpa.wannier.two_body[0][0]
    );
    let n = s_crpa.n;
    let idx = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    assert!(
        (s_crpa.wannier.two_body[0][idx(0, 0, 0, 0)]
            - s_crpa.wannier.two_body[0][idx(0, 1, 0, 1)])
        .norm()
            < 1e-8
    );
    assert!(fock_spinor(&s_crpa) < 1e-10, "zero-SOC cRPA Fock identity");

    // Part B: Spin-orbit-coupled H 1s + Cl 3p fragment in HCl (`lda-fr`) runs bare + cRPA end-to-end.
    let (_ctx_hcl, hcl) = hcl_spinor([2, 1, 1]);
    for k in 0..2 {
        let m = hcl.channels[0].frames[k].energies.as_slice().len();
        let norms = hcl.transition_norms(0, k, None).unwrap();
        assert!(
            (&norms - DMatrix::<C64>::identity(m, m)).norm() < 5e-4,
            "spinor transition_norms at k={k}: {}",
            (&norms - DMatrix::<C64>::identity(m, m)).norm()
        );
    }
    let hcl_bare = hcl
        .downfold(&Request {
            num_particles: Some([3, 3]),
            ..request(None)
        })
        .unwrap()
        .problem;
    let hcl_crpa = hcl
        .downfold(&Request {
            num_particles: Some([3, 3]),
            screening: Some(Screening::Crpa),
            cutoff: Some(2.0),
            ..request(None)
        })
        .unwrap()
        .problem;
    assert!(hcl_crpa.spinor && hcl_crpa.n == 8);
    assert!(fock_spinor(&hcl_crpa) < 1e-9, "SOC cRPA Fock identity");

    // Kramers pairs in the screened Kohn-Sham Hamiltonian & screening lowers on-site U.
    let mut evals: Vec<f64> = hcl_crpa
        .kohn_sham[0]
        .clone()
        .symmetric_eigenvalues()
        .as_slice()
        .to_vec();
    evals.sort_by(f64::total_cmp);
    for pair in 0..4 {
        assert!(
            (evals[2 * pair] - evals[2 * pair + 1]).abs() < 1e-6,
            "Kramers pair {pair}: {} vs {}",
            evals[2 * pair],
            evals[2 * pair + 1]
        );
    }
    let head = hcl_crpa.dielectric_head.unwrap();
    assert!(head > 0. && head <= 1., "HCl head = {head}");
    assert!(
        hcl_crpa.wannier.two_body[0][0].re < hcl_bare.wannier.two_body[0][0].re,
        "cRPA {} must screen bare {}",
        hcl_crpa.wannier.two_body[0][0].re,
        hcl_bare.wannier.two_body[0][0].re
    );
}

#[test]
fn the_paw_remainder_at_finite_wave_vector_matches_independent_quadrature() {
    use mandacaru_core::{
        math::{angular, radial},
        paw::{
            multipole::shape_at,
            multipole_pairs::{pair_cut, radial_moments, smooth_waves, ChannelWaves, Pair},
            onecenter::{Basis, Projector},
        },
        periodic::setup::{
            remainder::Remainder,
            shapes::{Atom, Shapes, Sites},
        },
        Context,
    };
    use nalgebra::Matrix3;
    use setup_shells_support::dataset;
    use std::f64::consts::PI;

    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let ds = dataset(&ctx, "lda-sr", "Cl");
    let center = [0.3, -0.4, 0.5];
    let atoms = [Atom {
        center,
        charge: ds.valence_charge,
        compensation_radius: ds.compensation_radius,
    }];
    let channels = [
        (0, 0, 0),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 2, -2),
        (0, 2, -1),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
    ];
    let sites = Sites {
        atoms: &atoms,
        channels: &channels,
        lattice: Matrix3::identity() * 10.0,
        sigma: 0.4,
    };
    let shapes = Shapes::new(&ctx, &sites, 13.0).unwrap();
    let projectors = [
        Projector { l: 0, m: 0, index: 0, real: false },
        Projector { l: 1, m: 0, index: 0, real: false },
        Projector { l: 1, m: 1, index: 0, real: false },
    ];
    let remainder =
        Remainder::new(&ctx, &ds, &shapes, Basis::Dual, &projectors, center).unwrap();

    // Small-Q finite-field derivative around Q = 0: after factoring out exp(-i Q.R), both L = 0
    // monopoles and L = 1 dipoles are completely absorbed by the compensation charges, so
    // dE/dQ_alpha|_0 = 0 to O(h^2).
    let h = 0.025;
    let q_vecs = [
        [0.0, 0.0, 0.0],
        [h, 0.0, 0.0],
        [-h, 0.0, 0.0],
        [0.0, h, 0.0],
        [0.0, -h, 0.0],
        [0.0, 0.0, h],
        [0.0, 0.0, -h],
        [0.6, -0.4, 0.9],
    ];
    let n = projectors.len();
    let mut table = vec![C64::default(); n * n * q_vecs.len()];
    remainder.transforms(&q_vecs, &mut table).unwrap();

    let strip_phase = |v: usize, ab: usize| -> C64 {
        let q = q_vecs[v];
        let arg = q[0] * center[0] + q[1] * center[1] + q[2] * center[2];
        table[ab * q_vecs.len() + v] * C64::new(arg.cos(), arg.sin())
    };
    for ab in 0..n * n {
        // Monopole at Q = 0 is zero up to the dataset's radial overlap quadrature (~1e-5).
        assert!(
            strip_phase(0, ab).norm() < 5e-5,
            "Q=0 monopole remainder nonzero: {}",
            strip_phase(0, ab)
        );
        for axis in 0..3 {
            let plus = strip_phase(1 + 2 * axis, ab);
            let minus = strip_phase(2 + 2 * axis, ab);
            let deriv = (plus - minus) / (2.0 * h);
            assert!(
                deriv.norm() < 1e-3,
                "dipole finite-field dE/dQ_{axis} not cancelled for pair {ab}: {deriv}"
            );
        }
    }

    // Independent 3D spherical quadrature at finite Q = [0.6, -0.4, 0.9] without Bessel tables.
    let q_test = q_vecs[7];
    let r = ds.r.as_slice();
    let mut r_weights = vec![0.0; r.len()];
    for i in 0..r.len() - 1 {
        let dr = r[i + 1] - r[i];
        r_weights[i] += 0.5 * dr;
        r_weights[i + 1] += 0.5 * dr;
    }
    let ch0 = &ds.channels[&0];
    let ch1 = &ds.channels[&1];
    let rebuild_ps = |ch: &mandacaru_core::paw::record::Channel, l: usize| -> Vec<f64> {
        let ae: Vec<f64> = ch.ae_waves.iter().flat_map(|b| b.as_slice().iter().copied()).collect();
        let coeff: Vec<f64> = ch.wave_coefficients.iter().flat_map(|a| a.data().iter().copied()).collect();
        let wv: Vec<f64> = ch.wavevectors.iter().flat_map(|a| a.data().iter().copied()).collect();
        smooth_waves(
            r,
            &ChannelWaves {
                l,
                r_cut: ch.r_cut,
                ae: &ae,
                coefficients: &coeff,
                wavevectors: &wv,
                waves: ch.ae_waves.len(),
            },
        )
        .unwrap()
    };
    let ps0 = rebuild_ps(ch0, 0);
    let ps1 = rebuild_ps(ch1, 1);
    let ae_ps = |p: Projector, i: usize| -> (f64, f64) {
        if p.l == 0 {
            (ch0.ae_waves[p.index].as_slice()[i], ps0[p.index * r.len() + i])
        } else {
            (ch1.ae_waves[p.index].as_slice()[i], ps1[p.index * r.len() + i])
        }
    };
    let gl_mu = radial::gauss_legendre(16).unwrap();
    let n_phi = 32usize;
    let d_phi = 2.0 * PI / n_phi as f64;
    let r_cut = pair_cut(ch0.r_cut, ch1.r_cut).max(ds.compensation_radius);
    let shape_l: [Vec<f64>; 3] =
        std::array::from_fn(|l| r.iter().map(|&rv| shape_at(rv, ds.compensation_radius, l)).collect());

    for (a, &pa) in projectors.iter().enumerate() {
        for (b, &pb) in projectors.iter().enumerate() {
            let mut comp_lm: Vec<(usize, i32, f64)> = Vec::new();
            for l in 0..=2usize {
                if (pa.l + pb.l) < l || l < pa.l.abs_diff(pb.l) || (pa.l + pb.l + l) % 2 != 0 {
                    continue;
                }
                let ae_a: Vec<f64> = (0..r.len()).map(|i| ae_ps(pa, i).0).collect();
                let ae_b: Vec<f64> = (0..r.len()).map(|i| ae_ps(pb, i).0).collect();
                let ps_a: Vec<f64> = (0..r.len()).map(|i| ae_ps(pa, i).1).collect();
                let ps_b: Vec<f64> = (0..r.len()).map(|i| ae_ps(pb, i).1).collect();
                let pair = Pair {
                    r,
                    r_cut: pair_cut(ds.channels[&(pa.l as u32)].r_cut, ds.channels[&(pb.l as u32)].r_cut),
                    ae: [&ae_a, &ae_b],
                    smooth: [&ps_a, &ps_b],
                    waves: [1, 1],
                };
                let delta = radial_moments(&pair, l).unwrap()[0];
                let m = pb.m - pa.m;
                if m.abs() <= l as i32 {
                    let c_m = gl_mu
                        .iter()
                        .flat_map(|&(mu, w_mu)| {
                            (0..n_phi).map(move |ip| {
                                let phi = ip as f64 * d_phi;
                                let s = (1.0 - mu * mu).max(0.0).sqrt();
                                let u = [s * phi.cos(), s * phi.sin(), mu];
                                let val = angular::complex(l, m, u).conj()
                                    * angular::complex(pa.l, pa.m, u).conj()
                                    * angular::complex(pb.l, pb.m, u);
                                (w_mu * d_phi) * val.re
                            })
                        })
                        .sum::<f64>();
                    comp_lm.push((l, m, delta * c_m));
                }
            }

            let mut direct = C64::default();
            for &(mu, w_mu) in &gl_mu {
                let s = (1.0 - mu * mu).max(0.0).sqrt();
                for ip in 0..n_phi {
                    let phi = ip as f64 * d_phi;
                    let u = [s * phi.cos(), s * phi.sin(), mu];
                    let ang_ab =
                        angular::complex(pa.l, pa.m, u).conj() * angular::complex(pb.l, pb.m, u);
                    let mut ang_comp_l = [C64::default(); 3];
                    for &(l, m, w_lm) in &comp_lm {
                        ang_comp_l[l] += angular::complex(l, m, u) * w_lm;
                    }
                    let q_dot_u = q_test[0] * u[0] + q_test[1] * u[1] + q_test[2] * u[2];
                    let w_ang = w_mu * d_phi;
                    for i in 0..r.len() {
                        if r[i] > r_cut {
                            break;
                        }
                        let (ae_a, ps_a) = ae_ps(pa, i);
                        let (ae_b, ps_b) = ae_ps(pb, i);
                        let arg = -q_dot_u * r[i];
                        let exp_q = C64::new(arg.cos(), arg.sin());
                        let ae_part = (ae_a * ae_b - ps_a * ps_b) * ang_ab;
                        let comp_part = shape_l[0][i] * ang_comp_l[0]
                            + shape_l[1][i] * ang_comp_l[1]
                            + shape_l[2][i] * ang_comp_l[2];
                        direct += w_ang * (r_weights[i] * r[i] * r[i]) * exp_q * (ae_part - comp_part);
                    }
                }
            }
            let got = strip_phase(7, a * n + b);
            let miss = (got - direct).norm();
            assert!(
                miss < 2e-3,
                "finite-Q remainder pair ({a},{b}): Remainder={got} vs direct 3D={direct}, miss={miss:e}"
            );
        }
    }
}
