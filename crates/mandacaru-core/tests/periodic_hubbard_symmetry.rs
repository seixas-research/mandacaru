//! DFT+U on a symmetry-reduced k mesh (HU-4): the occupation matrices completed over the kept
//! operations and time reversal equal those of the full mesh.
//!
//! The independent reference of every gate is the same crystal built with `symmetry = false`
//! and the whole Monkhorst-Pack mesh listed k-point by k-point with equal weights: it bypasses
//! both the space-group reduction and the time-reversal halving of the setup. Every crystal is
//! built from the PAW-LCAO datasets of `MANDACARU_PAW_PATH` (`lda-sr`), serial, under 1 GiB.
//!
//! Gates (the wedge against the full mesh of the same crystal): the shell occupations, the
//! total energy and the linear-response U matrix. The algebra of the completion (a projector,
//! covariant under every operation, with a wrong convention failing) is checked without an SCF,
//! to 1e-12.
//!
//! The gates depend on the discretization. The product quadrature of the projector spheres is
//! exactly invariant under rotations about the z axis and under mirrors, so the hexagonal and
//! trigonal cells (hcp Ni, three Ni around a threefold axis) and the control meet the 1e-10 and
//! 1e-8 eV gates (measured 2.4e-11, 1.9e-12 and 1.5e-13 in the occupations). It is not under the
//! cubic operations that mix the axes: Ni fcc and rocksalt NiO differ by about 1e-7 in the
//! occupations and 6e-7 eV in the energy of PLAIN DFT (Ni fcc, 3 x 3 x 3: 5.95e-7 eV with no +U
//! at all, 1.7e-7 eV on 2 x 2 x 2), so their gates sit above that floor. A wrong convention of
//! the completion changes the occupations by 1e-3 or more (see the algebra test).

mod periodic_symmetry_support;

use mandacaru_core::{
    periodic::{
        hubbard::{Seed, Shell, Spec},
        hubbard_response::hubbard_response,
        hubbard_symmetry::Averager,
        setup::symmetry::rotation_tables,
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use periodic_symmetry_support::*;
use serde_json::json;

const RESPONSE_GATE_EV: f64 = 1e-6;

// --------------------------------------------------------------------------------------------
// The algebra of the completion, without an SCF.
// --------------------------------------------------------------------------------------------

fn dummy_shell(atom: usize) -> Shell {
    Shell {
        atom,
        symbol: "Ni".into(),
        l: 2,
        u: 0.1,
        j: 0.,
        projectors: (0..5).map(|m| 5 * atom + m).collect(),
        weights: vec![1.],
        basis: None,
        seed: Seed::Moment {
            n_electrons: 8.,
            moment: 0.,
        },
    }
}

/// Reproducible Hermitian matrices without a generator: a linear congruence.
fn hermitian(seed: u64) -> Vec<C64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut n = vec![C64::default(); 25];
    for a in 0..5 {
        for b in a..5 {
            let z = C64::new(next(), if a == b { 0. } else { next() });
            n[a * 5 + b] = z;
            n[b * 5 + a] = z.conj();
        }
    }
    n
}

fn spec(atoms: usize) -> Spec {
    Spec {
        form: mandacaru_core::models::hubbard::Form::DudarevFll,
        shells: (0..atoms).map(dummy_shell).collect(),
    }
}

fn distance(a: &[Vec<C64>], b: &[Vec<C64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).norm()))
        .fold(0., f64::max)
}

#[test]
fn the_completion_is_a_projector_covariant_under_every_operation_and_a_wrong_convention_is_not() {
    let cell = nickel_triangle();
    let group = group_of(&cell);
    assert_eq!(
        group.len(),
        12,
        "D3h: a threefold axis, three vertical mirrors, the plane of the atoms"
    );
    let atoms = cell.atoms.len();
    let spec = spec(atoms);
    let averager = Averager::new(&spec, Some(&group), true).unwrap();
    let random: Vec<Vec<C64>> = (1..=atoms as u64).map(hermitian).collect();
    let mut once = random.clone();
    averager.apply(&spec, &mut once);
    let mut twice = once.clone();
    averager.apply(&spec, &mut twice);
    assert!(
        distance(&once, &twice) < 1e-13,
        "the completion is not idempotent"
    );
    assert!(
        distance(&once, &random) > 1e-2,
        "the random input must not be symmetric already"
    );

    // Covariance: n -> g.n with n'^{g(a)} = T^dagger n^a T completes to the same matrices.
    for (rotation, map) in group.rotations().iter().zip(group.atom_maps()) {
        let table = &rotation_tables(rotation, 2, 1e-6).unwrap()[2];
        let mut moved = vec![vec![C64::default(); 25]; atoms];
        for atom in 0..atoms {
            let n = DMatrix::from_fn(5, 5, |a, b| random[atom][a * 5 + b]);
            let image = table.adjoint() * n * table;
            moved[map[atom]] = (0..25).map(|i| image[(i / 5, i % 5)]).collect();
        }
        let mut completed = moved;
        averager.apply(&spec, &mut completed);
        assert!(
            distance(&completed, &once) < 1e-12,
            "not covariant under an operation"
        );
    }

    // A wrong convention (the same atom map with T^dagger on the left) is not a projector.
    let tables: Vec<_> = group
        .rotations()
        .iter()
        .map(|r| rotation_tables(r, 2, 1e-6).unwrap().remove(2))
        .collect();
    let wrong = |n: &[Vec<C64>]| -> Vec<Vec<C64>> {
        (0..atoms)
            .map(|atom| {
                let mut sum = DMatrix::<C64>::zeros(5, 5);
                for (table, map) in tables.iter().zip(group.atom_maps()) {
                    let n = DMatrix::from_fn(5, 5, |a, b| n[map[atom]][a * 5 + b]);
                    sum += table.adjoint() * n * table;
                }
                sum /= C64::new(tables.len() as f64, 0.);
                (0..25).map(|i| sum[(i / 5, i % 5)]).collect()
            })
            .collect()
    };
    let (first, second) = (wrong(&random), wrong(&wrong(&random)));
    assert!(
        distance(&first, &second) > 1e-3,
        "the negative control must fail the projector gate"
    );
}

#[test]
fn time_reversal_completion_is_a_projector_and_keeps_real_matrices_in_the_real_basis() {
    let spec = spec(1);
    let averager = Averager::new(&spec, None, true).unwrap();
    let mut n = vec![hermitian(7)];
    averager.apply(&spec, &mut n);
    let mut again = n.clone();
    averager.apply(&spec, &mut again);
    assert!(distance(&n, &again) < 1e-14);
    // (-1)^{m+m'} conj(n_{-m,-m'}) = n_{m m'}
    for a in 0..5 {
        for b in 0..5 {
            let sign = if (a + b) % 2 == 0 { 1. } else { -1. };
            assert!((n[0][a * 5 + b] - n[0][(4 - a) * 5 + (4 - b)].conj() * sign).norm() < 1e-14);
        }
    }
    // The trace and the Hermiticity survive.
    assert!(((0..5).map(|a| n[0][a * 5 + a]).sum::<C64>().im).abs() < 1e-14);
}

#[test]
fn a_table_the_operations_cannot_complete_raises_by_name() {
    let group = group_of(&nickel_hcp());
    let mut one_atom = spec(2);
    one_atom.shells.pop();
    let message = Averager::new(&one_atom, Some(&group), false)
        .err()
        .expect("must raise")
        .to_string();
    assert!(
        message.contains("equivalent") && message.contains("symmetry=False"),
        "{message}"
    );
    let mut unequal = spec(2);
    unequal.shells[1].u = 0.2;
    let message = Averager::new(&unequal, Some(&group), false)
        .err()
        .expect("must raise")
        .to_string();
    assert!(message.contains("same Hubbard shell"), "{message}");
}

// --------------------------------------------------------------------------------------------
// The wedge against the whole mesh.
// --------------------------------------------------------------------------------------------

#[test]
fn nickel_fcc_on_the_wedge_equals_the_whole_odd_mesh_restricted() {
    // 3 x 3 x 3 is not closed under k -> -k: the time-reversal completion is exercised too.
    let (wedge, whole) = compare(&nickel_fcc(), dudarev(4.), false, [3, 3, 3], CUBIC);
    assert_eq!(whole, 27);
    assert!(wedge < whole, "the symmetry must reduce the mesh");
}

#[test]
#[ignore = "slow: 17 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored a_time_reversal_halved_mesh_without_symmetry_equals_the_whole_mesh"]
fn a_time_reversal_halved_mesh_without_symmetry_equals_the_whole_mesh() {
    // symmetry = false keeps only the halving by time reversal of the odd 3 x 3 x 3 mesh. The
    // triangle has no inversion centre, so its occupations at one k are not time-reversal
    // invariant (inversion alone would make them so) and the halved sum needs the completion.
    let cell = nickel_triangle();
    let (halved, whole) = compare_with(&cell, dudarev(4.), false, [3, 3, 3], false, EXACT);
    assert_eq!(whole, 27);
    assert!(halved < whole);
}

#[test]
#[ignore = "slow: 10 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored nickel_fcc_spin_polarized_wedge_equals_the_whole_mesh"]
fn nickel_fcc_spin_polarized_wedge_equals_the_whole_mesh() {
    compare(&nickel_fcc(), dudarev(4.), true, [3, 3, 3], CUBIC);
}

#[test]
#[ignore = "slow: 10 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored the_liechtenstein_form_on_the_wedge_equals_the_whole_mesh"]
fn the_liechtenstein_form_on_the_wedge_equals_the_whole_mesh() {
    let form = json!({"Ni": {"l": "d", "U": 5., "J": 0.8, "form": "liechtenstein-fll"}});
    compare(&nickel_fcc(), form, true, [3, 3, 3], CUBIC);
}

#[test]
#[ignore = "slow: 14 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored rocksalt_nickel_oxide_on_the_wedge_equals_the_whole_mesh"]
fn rocksalt_nickel_oxide_on_the_wedge_equals_the_whole_mesh() {
    let (wedge, whole) = compare(&nickel_oxide(), dudarev(6.), false, [3, 3, 3], CUBIC);
    assert!(wedge < whole);
}

#[test]
#[ignore = "slow: 16 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored nickel_oxide_with_the_type_ii_cell_swaps_its_two_nickel_atoms_and_equals_the_whole_mesh"]
fn nickel_oxide_with_the_type_ii_cell_swaps_its_two_nickel_atoms_and_equals_the_whole_mesh() {
    let cell = nickel_oxide_antiferromagnetic_cell();
    let group = group_of(&cell);
    assert!(
        group.atom_maps().iter().any(|map| map[0] == 1),
        "an operation must send the first Ni atom to the second"
    );
    // 24 operations, 2 x 2 x 2: 4 points against 8. The energy floor of this cell is 3.15e-5 eV
    // with no +U at all (the cubic operations mix the axes), so the energy gate sits above it;
    // E_U, which the floor does not enter at first order, is gated like the others.
    let (wedge, whole) = compare(&cell, dudarev(6.), false, [2, 2, 2], (3e-6, 1e-4, 2e-8));
    assert!(wedge < whole);
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored hcp_nickel_whose_screw_axis_swaps_the_two_hubbard_atoms_equals_the_whole_mesh"]
fn hcp_nickel_whose_screw_axis_swaps_the_two_hubbard_atoms_equals_the_whole_mesh() {
    let (wedge, whole) = compare(&nickel_hcp(), dudarev(4.), false, [3, 3, 2], EXACT);
    assert!(wedge < whole);
}

#[test]
#[ignore = "slow: 11 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored three_atoms_cyclically_permuted_by_a_threefold_axis_equal_the_whole_mesh"]
fn three_atoms_cyclically_permuted_by_a_threefold_axis_equal_the_whole_mesh() {
    let (wedge, whole) = compare(&nickel_triangle(), dudarev(4.), false, [2, 2, 2], EXACT);
    assert!(wedge < whole);
}

#[test]
fn a_cell_with_no_symmetry_beyond_time_reversal_keeps_the_whole_even_mesh() {
    // Two Ni atoms at generic fractional positions in a triclinic cell: the group is the
    // identity, and the 2 x 2 x 2 Gamma-centered mesh is time-reversal invariant, so the
    // wedge is the whole mesh.
    let b = ANGSTROM;
    let cell = Cell {
        lattice: [
            4.1 * b,
            0.4 * b,
            0.3 * b,
            0.,
            4.6 * b,
            0.5 * b,
            0.,
            0.,
            5.2 * b,
        ],
        grid: [8, 8, 10],
        atoms: vec![("Ni", 0, [0.05, 0.1, 0.0]), ("Ni", 0, [0.57, 0.46, 0.53])],
        symbols: vec!["Ni"],
        moments: vec![],
    };
    let (wedge, whole) = compare(&cell, dudarev(4.), false, [2, 2, 2], EXACT);
    assert_eq!((wedge, whole), (8, 8));
}

// --------------------------------------------------------------------------------------------
// The linear response.
// --------------------------------------------------------------------------------------------

#[test]
#[ignore = "slow: 13 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored the_response_on_the_wedge_equals_the_whole_mesh_when_every_hubbard_atom_is_fixed"]
fn the_response_on_the_wedge_equals_the_whole_mesh_when_every_hubbard_atom_is_fixed() {
    let (reduced, whole) = nickel_oxide().crystals(dudarev(6.), false, [2, 2, 2]);
    assert!(reduced.points.len() < whole.points.len());
    let (a, b) = (solve(&reduced, CUBIC.2), solve(&whole, CUBIC.2));
    let alphas = [0.05 / HARTREE_EV, 0.1 / HARTREE_EV];
    let (ra, rb) = (
        hubbard_response(&reduced, &a, &alphas).unwrap(),
        hubbard_response(&whole, &b, &alphas).unwrap(),
    );
    let worst = (ra.u_matrix[(0, 0)] - rb.u_matrix[(0, 0)]).abs() * HARTREE_EV;
    eprintln!(
        "U = {:.6} eV (wedge) against {:.6} eV (whole), {} against {} k-points",
        ra.u_matrix[(0, 0)] * HARTREE_EV,
        rb.u_matrix[(0, 0)] * HARTREE_EV,
        reduced.points.len(),
        whole.points.len()
    );
    assert!(
        worst < RESPONSE_GATE_EV,
        "the response U differs by {worst:.2e} eV"
    );
    assert!(
        rb.u_matrix[(0, 0)] * HARTREE_EV > 0.1,
        "U must be a real number"
    );
}

#[test]
#[ignore = "slow: 14 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_hubbard_symmetry -- --ignored the_response_refuses_a_hubbard_atom_the_operations_move"]
fn the_response_refuses_a_hubbard_atom_the_operations_move() {
    let (reduced, _) = nickel_hcp().crystals(dudarev(4.), false, [2, 2, 2]);
    let state = solve(&reduced, EXACT.2);
    let message = hubbard_response(&reduced, &state, &[0.05 / HARTREE_EV])
        .err()
        .expect("the shift of one atom breaks the symmetry")
        .to_string();
    assert!(
        message.contains("fixed by every kept operation") && message.contains("symmetry=False"),
        "{message}"
    );
}
