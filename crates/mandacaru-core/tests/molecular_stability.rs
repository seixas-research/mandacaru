//! Stability following of the Hartree-Fock SCF (`molecular::scf`): a converged RHF, UHF or GHF
//! determinant is never a saddle of its own surface. Molecular PAW-LCAO `lda-sr` SZ in a cubic
//! box of 8 A (grid_spacing 0.25 A, 32 nodes per axis), raw projectors, Cholesky 1e-10; energies
//! electronic, in Hartree; bond lengths in Bohr. Needs `MANDACARU_PAW_PATH`.
mod paw_supercell_support;
mod setup_shells_support;

use mandacaru_core::{
    math::davidson::lowest_root,
    molecular::{scf::STABILITY_TOLERANCE, MeanField, MolecularIntegrals, ScfOptions},
    Context,
};
use paw_supercell_support::{molecular_integrals, molecule_in_a_box};

/// The 8 A box in Bohr and its grid.
const SIDE: f64 = 8. / 0.529_177_210_903;
const NODES: usize = 32;

/// The molecular PAW-LCAO Hamiltonian of the diatomic `a b` at `bond` Bohr along z.
fn diatomic(ctx: &Context, a: &'static str, b: &'static str, bond: f64) -> MolecularIntegrals {
    let setup = molecule_in_a_box(
        ctx,
        &[(a, [0., 0., -bond / 2.]), (b, [0., 0., bond / 2.])],
        SIDE,
        NODES,
    );
    molecular_integrals(ctx, &setup)
}

/// The lowest root of `state`'s orbital Hessian, dense and by seeded Davidson.
fn roots(integrals: &MolecularIntegrals, state: &MeanField) -> (f64, f64) {
    let hessian = integrals.orbital_hessian(state).unwrap().unwrap();
    let dense = lowest_root(&hessian, usize::MAX, 1e-7).unwrap().unwrap();
    let davidson = lowest_root(&hessian, 0, 1e-7).unwrap().unwrap();
    assert!(dense.dense && !davidson.dense && davidson.converged);
    (dense.value, davidson.value)
}

/// The energy's curvature along the lowest unit mode of `state`, by central differences of the
/// Hartree-Fock energy of the rotated densities (step 2e-3 rad), over the root: 4 for RHF and 2
/// for UHF (`OrbitalHessian`).
fn curvature_over_root(integrals: &MolecularIntegrals, state: &MeanField) -> f64 {
    let hessian = integrals.orbital_hessian(state).unwrap().unwrap();
    let root = lowest_root(&hessian, usize::MAX, 1e-7).unwrap().unwrap();
    let step = 2e-3;
    let energy = |angle: f64| {
        let d = hessian.rotated_densities(&root.vector, angle).unwrap();
        integrals.hartree_fock_energy(&state.method, &d).unwrap()
    };
    (energy(step) - 2. * energy(0.) + energy(-step)) / (step * step) / root.value
}

fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-10,
        ..ScfOptions::default()
    }
}

/// vera's defect (review 2026-11-15, 1b): at 3.4 Bohr the Davidson search from unit vectors
/// returned +9.2e-5 while the dense lowest root is -4.79e-2, and the saddle -26.3386383731 came
/// back converged. The lowest root is now found densely (and by seeded Davidson alike), so the
/// state is followed to the minimum below.
#[test]
fn stretched_nitrogen_is_not_returned_converged_on_a_saddle() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let integrals = diatomic(&ctx, "N", "N", 3.4);
    let n = integrals.neutral_electrons() / 2;
    let rhf = integrals.mean_field("rhf", [n, n], &options()).unwrap();
    let (dense, davidson) = roots(&integrals, &rhf);
    let ratio = curvature_over_root(&integrals, &rhf);
    println!(
        "N2 3.4: RHF {:.10} converged {}; lowest root dense {dense:.6e}, seeded Davidson {davidson:.6e}; curvature / root {ratio:.6}",
        rhf.electronic_energy, rhf.converged
    );
    assert!(
        rhf.converged
            && !integrals
                .orbital_hessian(&rhf)
                .unwrap()
                .unwrap()
                .is_complex()
    );
    assert!(dense >= -STABILITY_TOLERANCE, "{dense:e}");
    assert!((dense - davidson).abs() < 1e-6, "{dense:e} {davidson:e}");
    // The minimum's lowest root is a soft rotation of the pi shells (9.1e-5), so the quartic term
    // of the 2e-3 rad difference shows at 1e-3 of it (UHF pins the factor on a stiff mode).
    assert!((ratio - 4.).abs() < 1e-2, "{ratio}");
    // The saddle the unit-vector search accepted lies 5.4e-2 Ha above.
    assert!(
        rhf.electronic_energy < -26.3386383731 - 4e-2,
        "{:.10}",
        rhf.electronic_energy
    );
    // vera's dense-root emulation reached -26.3925378507.
    assert!(
        (rhf.electronic_energy + 26.3925378507).abs() < 1e-8,
        "{:.10}",
        rhf.electronic_energy
    );
}

/// vera 1g: the singlet O2 cold-start UHF stopped 1.208e-4 Ha above the followed RHF, on the
/// saddle the RHF left. UHF is followed with the UHF Hessian under the same root rules; a UHF
/// state is never above the RHF one, and the UHF normalization (curvature = 2 root) is pinned.
#[test]
fn singlet_oxygen_uhf_is_followed_below_the_rhf() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let integrals = diatomic(&ctx, "O", "O", 2.28);
    let n = integrals.neutral_electrons() / 2;
    let rhf = integrals.mean_field("rhf", [n, n], &options()).unwrap();
    let uhf = integrals.mean_field("uhf", [n, n], &options()).unwrap();
    let (dense, davidson) = roots(&integrals, &uhf);
    let ratio = curvature_over_root(&integrals, &uhf);
    println!(
        "O2 2.28 singlet: RHF {:.10} ({}), UHF {:.10} ({}); UHF lowest root dense {dense:.6e}, Davidson {davidson:.6e}; curvature / root {ratio:.6}",
        rhf.electronic_energy, rhf.converged, uhf.electronic_energy, uhf.converged
    );
    assert!(
        rhf.converged
            && uhf.converged
            && !integrals
                .orbital_hessian(&uhf)
                .unwrap()
                .unwrap()
                .is_complex()
    );
    assert!(uhf.electronic_energy <= rhf.electronic_energy + 1e-8);
    assert!(dense >= -STABILITY_TOLERANCE && (dense - davidson).abs() < 1e-6);
    assert!((ratio - 2.).abs() < 1e-3, "{ratio}");
}

/// GHF of the triplet O2: the global spin rotations about x and y are exact zero modes of a
/// collinear determinant and are projected out (two of them; the rotation about z leaves it
/// unchanged), so the state is neither followed along them nor rejected; it is not above UHF.
#[test]
fn triplet_oxygen_ghf_projects_its_spin_rotations() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let integrals = diatomic(&ctx, "O", "O", 2.28);
    let n = integrals.neutral_electrons() / 2;
    let uhf = integrals
        .mean_field("uhf", [n + 1, n - 1], &options())
        .unwrap();
    let ghf = integrals
        .mean_field("ghf", [n + 1, n - 1], &options())
        .unwrap();
    let hessian = integrals.orbital_hessian(&ghf).unwrap().unwrap();
    let root = integrals.stability(&ghf).unwrap().unwrap();
    println!(
        "O2 2.28 triplet: UHF {:.10} ({}), GHF {:.10} ({}); GHF zero modes {}, lowest root {root:.6e}",
        uhf.electronic_energy, uhf.converged, ghf.electronic_energy, ghf.converged, hessian.zero_modes()
    );
    assert!(uhf.converged && ghf.converged && hessian.is_complex());
    assert_eq!(hessian.zero_modes(), 2);
    assert!(root >= -STABILITY_TOLERANCE, "{root:e}");
    assert!(ghf.electronic_energy <= uhf.electronic_energy + 1e-8);
}

/// A channel without an occupied (or an empty) orbital has no rotation: the UHF H2+ ([1, 0]) has
/// one alpha rotation and an empty beta channel, which adds nothing and is not diagonalized; the
/// SZ hydrogen atom (one function) has no rotation at all.
#[test]
fn a_channel_without_occupied_orbitals_has_no_rotation() {
    use mandacaru_core::math::davidson::Symmetric;
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let ion = diatomic(&ctx, "H", "H", 2.0);
    let uhf = ion.mean_field("uhf", [1, 0], &options()).unwrap();
    let hessian = ion.orbital_hessian(&uhf).unwrap().unwrap();
    let root = ion.stability(&uhf).unwrap().unwrap();
    println!(
        "H2+ UHF {:.10} ({}); {} rotation(s), lowest root {root:.6e}",
        uhf.electronic_energy,
        uhf.converged,
        hessian.dimension()
    );
    assert!(uhf.converged && root >= -STABILITY_TOLERANCE);
    assert_eq!(hessian.dimension(), ion.m() - 1);
    let setup = molecule_in_a_box(&ctx, &[("H", [0., 0., 0.])], SIDE, NODES);
    let atom = molecular_integrals(&ctx, &setup);
    let state = atom.mean_field("uhf", [1, 0], &options()).unwrap();
    assert!(state.converged && atom.m() == 1);
    assert!(
        atom.orbital_hessian(&state).unwrap().is_none()
            && atom.stability(&state).unwrap().is_none()
    );
}

/// `continue_mean_field` follows the same state and never leaves it for another: the stable N2
/// minimum at 2.08 Bohr continued onto the 3.4 Bohr Hamiltonian comes back either not converged
/// or stable (a saddle it lands on is reported, not followed).
#[test]
fn a_continued_state_is_never_a_converged_saddle() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let near = diatomic(&ctx, "N", "N", 2.08);
    let far = diatomic(&ctx, "N", "N", 3.4);
    let n = near.neutral_electrons() / 2;
    let start = near.mean_field("rhf", [n, n], &options()).unwrap();
    let continued = far.continue_mean_field(&start, &options()).unwrap();
    let (dense, _) = roots(&far, &continued);
    println!(
        "N2 2.08 -> 3.4 continued: {:.10} converged {}; lowest root {dense:.6e} (the followed minimum -26.3925378507)",
        continued.electronic_energy, continued.converged
    );
    assert!(start.converged);
    assert!(
        !continued.converged || dense >= -STABILITY_TOLERANCE,
        "{dense:e}"
    );
}

/// Following through complex rotations, end to end: the stretched equilateral H3 (sides 3.0
/// Bohr, doublet `[2, 1]`) is spin-frustrated, so its collinear UHF minimum is a saddle of the
/// GHF surface; the GHF SCF follows it (the GHF Hessian's complex rotations, the restart and the
/// descent) to a noncollinear state below the UHF one, converged with no negative root.
#[test]
fn a_frustrated_triangle_is_followed_off_its_collinear_ghf_saddle() {
    use mandacaru_core::math::davidson::Symmetric;
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let side = 3.0;
    let radius = side / 3f64.sqrt();
    let atoms: Vec<(&'static str, [f64; 3])> = (0..3)
        .map(|i| {
            let angle = 2. * std::f64::consts::PI * i as f64 / 3.;
            ("H", [radius * angle.cos(), radius * angle.sin(), 0.])
        })
        .collect();
    let integrals = molecular_integrals(&ctx, &molecule_in_a_box(&ctx, &atoms, SIDE, NODES));
    let uhf = integrals.mean_field("uhf", [2, 1], &options()).unwrap();
    let ghf = integrals.mean_field("ghf", [2, 1], &options()).unwrap();
    let hessian = integrals.orbital_hessian(&ghf).unwrap().unwrap();
    let root = integrals.stability(&ghf).unwrap().unwrap();
    println!(
        "H3 equilateral {side} Bohr: UHF {:.10} ({}, {} iterations), GHF {:.10} ({}, {} iterations), GHF below UHF by {:.3e}; GHF rotations {}, zero modes {}, lowest root {root:.6e}",
        uhf.electronic_energy, uhf.converged, uhf.iterations.len(), ghf.electronic_energy, ghf.converged, ghf.iterations.len(),
        uhf.electronic_energy - ghf.electronic_energy, hessian.dimension(), hessian.zero_modes()
    );
    assert!(uhf.converged && ghf.converged && hessian.is_complex());
    assert!(root >= -STABILITY_TOLERANCE, "{root:e}");
    assert!(
        ghf.electronic_energy < uhf.electronic_energy - 1e-6,
        "the noncollinear state was not reached"
    );
}

/// Following above [`DENSE_HESSIAN_LIMIT`] rotations, end to end (the seeded Davidson root, not
/// the dense one): a stretched 4 x 4 x 2 lattice of hydrogen atoms (2.6 Bohr apart, 32 SZ
/// functions, UHF `[16, 16]`: 512 rotations), whose spin-restricted solution is a saddle of the
/// UHF surface. UHF comes back converged below RHF with no negative root, and the Davidson root
/// of the result equals the dense one.
#[test]
fn a_stretched_hydrogen_lattice_is_followed_above_the_dense_limit() {
    use mandacaru_core::{math::davidson::Symmetric, molecular::scf::DENSE_HESSIAN_LIMIT};
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let spacing = 2.6;
    let mut atoms: Vec<(&'static str, [f64; 3])> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..2 {
                atoms.push((
                    "H",
                    [
                        (i as f64 - 1.5) * spacing,
                        (j as f64 - 1.5) * spacing,
                        (k as f64 - 0.5) * spacing,
                    ],
                ));
            }
        }
    }
    let integrals = molecular_integrals(&ctx, &molecule_in_a_box(&ctx, &atoms, SIDE, NODES));
    let rhf = integrals.mean_field("rhf", [16, 16], &options()).unwrap();
    let uhf = integrals.mean_field("uhf", [16, 16], &options()).unwrap();
    let hessian = integrals.orbital_hessian(&uhf).unwrap().unwrap();
    let (dense, davidson) = roots(&integrals, &uhf);
    println!(
        "H32 lattice {spacing} Bohr: RHF {:.10} ({}), UHF {:.10} ({}, {} iterations), UHF below RHF by {:.3e}; {} rotations; lowest root dense {dense:.6e}, seeded Davidson {davidson:.6e}",
        rhf.electronic_energy, rhf.converged, uhf.electronic_energy, uhf.converged, uhf.iterations.len(),
        rhf.electronic_energy - uhf.electronic_energy, hessian.dimension()
    );
    assert!(hessian.dimension() > DENSE_HESSIAN_LIMIT);
    assert!(uhf.converged && dense >= -STABILITY_TOLERANCE, "{dense:e}");
    assert!((dense - davidson).abs() < 1e-6, "{dense:e} {davidson:e}");
    assert!(uhf.electronic_energy < rhf.electronic_energy - 1e-4);
}

/// The reduced diatomic scan of vera's review (1b, 1d, 1e): the points that came back converged
/// on a saddle or not converged, and stable controls. Per point: RHF and UHF energy, converged,
/// the lowest root dense and by seeded Davidson. Gates: no converged state has a root below
/// `-STABILITY_TOLERANCE`; Davidson = dense; UHF is not above RHF.
#[test]
#[ignore = "slow: about 20 diatomics; run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test molecular_stability -- --ignored --nocapture"]
fn the_stretched_diatomic_scan_has_no_converged_saddle() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let points: [(&str, &str, f64); 22] = [
        ("H", "H", 1.4),
        ("Li", "H", 3.015),
        ("C", "O", 2.13),
        ("F", "F", 2.68),
        ("N", "N", 2.08),
        ("N", "N", 3.4),
        ("N", "N", 4.6),
        ("O", "O", 3.0),
        ("O", "O", 3.6),
        ("O", "O", 3.8),
        ("C", "C", 3.6),
        ("C", "C", 3.8),
        ("C", "C", 4.0),
        ("C", "C", 5.2),
        ("C", "C", 5.6),
        ("B", "N", 2.6),
        ("B", "N", 2.8),
        ("B", "N", 3.2),
        ("B", "B", 3.0),
        ("B", "B", 3.2),
        ("B", "B", 3.4),
        ("C", "O", 6.0),
    ];
    let mut failures = Vec::new();
    for (a, b, bond) in points {
        let integrals = diatomic(&ctx, a, b, bond);
        let n = integrals.neutral_electrons() / 2;
        let start = std::time::Instant::now();
        let rhf = integrals.mean_field("rhf", [n, n], &options()).unwrap();
        let uhf = integrals.mean_field("uhf", [n, n], &options()).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let (r_dense, r_davidson) = roots(&integrals, &rhf);
        let (u_dense, u_davidson) = roots(&integrals, &uhf);
        println!(
            "{a}{b} {bond:.3}: RHF {:.10} {} root {r_dense:+.4e} (Davidson {r_davidson:+.4e}) | UHF {:.10} {} root {u_dense:+.4e} (Davidson {u_davidson:+.4e}) | UHF - RHF {:+.3e} | {seconds:.1} s",
            rhf.electronic_energy,
            rhf.converged,
            uhf.electronic_energy,
            uhf.converged,
            uhf.electronic_energy - rhf.electronic_energy
        );
        for (name, state, dense, davidson) in [
            ("RHF", &rhf, r_dense, r_davidson),
            ("UHF", &uhf, u_dense, u_davidson),
        ] {
            if state.converged && dense < -STABILITY_TOLERANCE {
                failures.push(format!(
                    "{a}{b} {bond} {name}: converged on a saddle, root {dense:e}"
                ));
            }
            if (dense - davidson).abs() > 1e-6 {
                failures.push(format!(
                    "{a}{b} {bond} {name}: Davidson {davidson:e} against dense {dense:e}"
                ));
            }
        }
        if rhf.converged && uhf.converged && uhf.electronic_energy > rhf.electronic_energy + 1e-8 {
            failures.push(format!("{a}{b} {bond}: UHF above RHF"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
