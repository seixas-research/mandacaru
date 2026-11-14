//! Lattice geometry on plain arrays (`geometry`): fractional coordinates, supercells and neighbor
//! vectors, against closed forms on a triclinic cell (a symmetric cell hides a row/column swap).
use mandacaru_core::geometry::{
    complete_cell, fractional_coordinates, neighbor_vectors, repeat_cell,
};

const TRICLINIC: [[f64; 3]; 3] = [[3.0, 0.2, -0.1], [0.5, 2.7, 0.3], [0.4, -0.6, 4.1]];
const ALL: [bool; 3] = [true; 3];

fn cartesian(cell: &[[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|j| f[0] * cell[0][j] + f[1] * cell[1][j] + f[2] * cell[2][j])
}

#[test]
fn fractional_coordinates_invert_the_cartesian_combination_of_the_rows() {
    let f = [[0.25, 0.5, 0.75], [-0.3, 1.4, 2.2], [0.0, 0.0, 0.0]];
    let positions: Vec<[f64; 3]> = f.iter().map(|&x| cartesian(&TRICLINIC, x)).collect();
    let back = fractional_coordinates(&TRICLINIC, &positions, ALL, false).unwrap();
    for (a, b) in back.iter().zip(&f) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-14, "{a:?} {b:?}");
        }
    }
}

#[test]
fn wrapping_takes_periodic_axes_into_the_half_open_unit_interval_and_leaves_the_others() {
    let f = [[-0.3, 1.4, 2.2], [1.0, -1.0, 0.5]];
    let positions: Vec<[f64; 3]> = f.iter().map(|&x| cartesian(&TRICLINIC, x)).collect();
    let pbc = [true, true, false];
    let wrapped = fractional_coordinates(&TRICLINIC, &positions, pbc, true).unwrap();
    for (w, original) in wrapped.iter().zip(&f) {
        assert!(
            (0.0..1.0).contains(&w[0]) && (0.0..1.0).contains(&w[1]),
            "{w:?}"
        );
        assert!(
            (w[2] - original[2]).abs() < 1e-14,
            "a vacuum axis is not wrapped"
        );
    }
    assert!((wrapped[0][0] - 0.7).abs() < 1e-13 && (wrapped[0][1] - 0.4).abs() < 1e-13);
}

#[test]
fn a_coordinate_that_rounds_up_to_one_goes_back_to_zero_on_the_second_pass() {
    // x = -1e-17 + 1 is 1.0 in floating point; ASE's second modulo takes it to 0 (a diagonal cell
    // solves exactly, so the fractional coordinate is the position over the length).
    let cell = [[2.0, 0., 0.], [0., 2.0, 0.], [0., 0., 2.0]];
    let wrapped = fractional_coordinates(&cell, &[[-2e-17, 1.0, 0.5]], ALL, true).unwrap();
    assert_eq!(wrapped[0][0], 0.0);
    assert_eq!(wrapped[0][1], 0.5);
    assert_eq!(wrapped[0][2], 0.25);
}

#[test]
fn a_missing_cell_vector_is_completed_and_two_missing_ones_are_refused() {
    let flat = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0; 3]];
    assert_eq!(complete_cell(&flat).unwrap()[2], [0.0, 0.0, 1.0]);
    assert_eq!(
        complete_cell(&[[0.0; 3]; 3]).unwrap(),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
    assert!(complete_cell(&[[2.0, 0.0, 0.0], [0.0; 3], [0.0; 3]]).is_err());
    let f = fractional_coordinates(&flat, &[[1.0, 1.5, 0.7]], [true, true, false], false).unwrap();
    assert_eq!(f[0], [0.5, 0.5, 0.7]);
}

#[test]
fn a_supercell_repeats_the_atoms_with_the_first_repeat_slowest() {
    let positions = [[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]];
    let r = repeat_cell(&TRICLINIC, &positions, &[8, 1], [2, 1, 3]).unwrap();
    assert_eq!(r.numbers, vec![8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1]);
    assert_eq!(r.source, vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    // copy index = m0 * 3 + m2 (m1 = 0): the second block is the a2... c shift of the first
    for (copy, (m0, m2)) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        .iter()
        .enumerate()
    {
        let shift = cartesian(&TRICLINIC, [*m0 as f64, 0.0, *m2 as f64]);
        for atom in 0..2 {
            let got = r.positions[copy * 2 + atom];
            for k in 0..3 {
                assert!((got[k] - (positions[atom][k] + shift[k])).abs() < 1e-14);
            }
        }
    }
    assert_eq!(r.cell[0], [6.0, 0.4, -0.2]);
    assert_eq!(r.cell[1], TRICLINIC[1]);
    assert_eq!(r.cell[2][2], 3.0 * 4.1);
    assert!(repeat_cell(&TRICLINIC, &positions, &[8, 1], [0, 1, 1]).is_err());
}

#[test]
fn a_cubic_lattice_has_six_nearest_neighbors_then_twelve_and_never_itself() {
    let a = 2.0;
    let cell = [[a, 0., 0.], [0., a, 0.], [0., 0., a]];
    let near = neighbor_vectors(&cell, ALL, &[[0.3, 0.3, 0.3]], 1.01 * a).unwrap();
    assert_eq!(near[0].len(), 6);
    assert!(near[0]
        .iter()
        .all(|d| (d.iter().map(|x| x * x).sum::<f64>().sqrt() - a).abs() < 1e-12));
    let next = neighbor_vectors(&cell, ALL, &[[0.3, 0.3, 0.3]], 1.5 * a).unwrap();
    assert_eq!(next[0].len(), 18, "six at a, twelve at a sqrt 2");
    // An open axis has no images: a free pair at 1.2 and one atom across the cell.
    let open = neighbor_vectors(
        &cell,
        [false; 3],
        &[[0., 0., 0.], [1.2, 0., 0.], [3.9, 0., 0.]],
        2.5,
    )
    .unwrap();
    assert_eq!(open.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1, 0]);
    assert!(neighbor_vectors(&cell, ALL, &[[0.; 3]], 0.0).is_err());
}

#[test]
fn the_neighbors_of_a_triclinic_cell_do_not_depend_on_where_the_atoms_sit_in_it() {
    let positions = [[0.2, 0.3, 0.4], [1.1, 1.3, 2.0]];
    let wrapped: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| {
            let f = fractional_coordinates(&TRICLINIC, &[*p], ALL, true).unwrap()[0];
            cartesian(&TRICLINIC, f)
        })
        .collect();
    let norms = |found: Vec<Vec<[f64; 3]>>| {
        let mut all: Vec<f64> = found[0]
            .iter()
            .map(|d| d.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        all.sort_by(f64::total_cmp);
        all
    };
    let a = norms(neighbor_vectors(&TRICLINIC, ALL, &positions, 5.0).unwrap());
    let b = norms(neighbor_vectors(&TRICLINIC, ALL, &wrapped, 5.0).unwrap());
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-12);
    }
}
