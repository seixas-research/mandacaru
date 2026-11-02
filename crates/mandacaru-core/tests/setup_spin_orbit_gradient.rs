//! The atom derivative of the spin-orbit block between the spinor Bloch sums of Si `lda-fr`, SZ
//! (`periodic::setup::spin_orbit::matrix_derivative`, CS-1) on the real union projectors. The
//! projections' derivative is the existing `projection_slopes` with the spin-orbit projector list
//! as its projector set (no second derivative code); the gates are central differences of the
//! Rust block `matrices` with the atom's functions and spin-orbit projectors (and their
//! quadrature spheres) moved together, on every (atom, axis) move, plus the value invariance of
//! the block under a rigid shift of every center. There is no Python derivative to freeze: the
//! block itself is pinned to the frozen Python matrix by `setup_spin_orbit.rs`, so the
//! differences are anchored to it. Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;
mod setup_spin_orbit_silicon_support;

use mandacaru_core::periodic::setup::spin_orbit::{matrix_derivative, Block};
use setup_spin_orbit_silicon_support::*;
use std::collections::BTreeMap;

/// The measured floor of the difference is 2.3e-13 (h^2 spread 1.1e-12); the block derivative is
/// 9e-5 at its largest (a Si spin-orbit splitting is tens of meV), so 1e-10 is 500 x the floor
/// and 1e6 below the derivative, and every control misses by more than 100 x the gate.
const GATE: f64 = 1e-10;

/// One (atom, axis) move: the analytic block derivative against the Richardson difference at both
/// k-points, the sign control, and the functions-only / projectors-only split (each misses the
/// whole by more than 100 x the gate and they add up to it). Each move is its own test: it rebuilds
/// the block twelve times, and the six together are over the 180 s budget.
fn check_move(atom: usize, axis: usize) {
    let t = silicon();
    let (fd, spread) = t.difference(atom, axis, true, true);
    let (rows, _) = t.difference(atom, axis, true, false);
    let (columns, _) = t.difference(atom, axis, false, true);
    let mut split = 0;
    for k in 0..KS.len() {
        let span = k * SIZE..(k + 1) * SIZE;
        let (want, part, rest) = (&fd[span.clone()], &rows[span.clone()], &columns[span]);
        let got = t.analytic(k, atom, axis);
        let error = gap(&got, want);
        eprintln!(
            "atom {atom} axis {axis} k{k}: |analytic - FD| {error:e}, h^2 spread {spread:e}, peak {:e}",
            peak(want)
        );
        assert!(peak(want) > 1e-5, "non-vacuous");
        assert!(error < GATE, "{error:e}");
        let minus: Vec<_> = got.iter().map(|x| -x).collect();
        assert!(gap(&minus, want) > 100. * GATE);
        // The functions alone and the projectors alone each miss by far more than the gate, and
        // add up to the whole: the projector-moving term cannot be dropped.
        assert!(gap(&got, part) > 100. * GATE && gap(&got, rest) > 100. * GATE);
        let sum: Vec<_> = part.iter().zip(rest).map(|(a, b)| a + b).collect();
        assert!(gap(&got, &sum) < GATE);
        split += 1;
    }
    assert_eq!(split, KS.len(), "the split control ran at every k-point");
}

macro_rules! moves {
    ($($name:ident: $atom:expr, $axis:expr;)*) => {$(
        #[test]
        fn $name() {
            check_move($atom, $axis);
        }
    )*};
}

moves! {
    the_derivative_of_the_block_equals_a_central_difference_atom0_x: 0, 0;
    the_derivative_of_the_block_equals_a_central_difference_atom0_y: 0, 1;
    the_derivative_of_the_block_equals_a_central_difference_atom0_z: 0, 2;
    the_derivative_of_the_block_equals_a_central_difference_atom1_x: 1, 0;
    the_derivative_of_the_block_equals_a_central_difference_atom1_y: 1, 1;
    the_derivative_of_the_block_equals_a_central_difference_atom1_z: 1, 2;
}

/// Translation invariance in its physical form: the block `h^SO(k)` depends on the basis and the
/// projectors through their relative positions only, so moving every center by one vector leaves
/// it unchanged (and with it `E_SO = Re tr[D h]` for any `D`). Compared at a finite shift, not
/// differentiated: this is what the vanishing atom-sum of the force term stands for.
#[test]
fn a_rigid_shift_of_every_center_leaves_the_block_unchanged() {
    let t = silicon();
    let base = t.block(None, 0, 0., true, true);
    let scale = peak(&base);
    assert!(scale > 1e-4, "non-vacuous: {scale:e}");
    // Measured 2e-16 .. 2.5e-16 absolute (4e-13 of the peak 5.8e-4: round-off of the quadrature);
    // the gate is 1e-10 of the peak, 230 x the measured defect.
    let gate = 1e-10 * scale;
    for (axis, delta) in [(0, 0.37), (1, -0.52), (2, 0.81)] {
        let shifted = t.block(None, axis, delta, true, true);
        let defect = gap(&shifted, &base);
        eprintln!("shift {delta} along {axis}: |h' - h| = {defect:e}, peak {scale:e}");
        assert!(defect < gate, "{defect:e}");
    }
    // Moving the basis alone IS visible: the block is not insensitive to positions.
    let basis_only = t.block(None, 0, 0.37, true, false);
    let visible = gap(&basis_only, &base);
    eprintln!("basis alone shifted: |h' - h| = {visible:e}");
    assert!(visible > 100. * gate);
}

#[test]
fn a_crystal_without_the_term_and_a_bad_request_are_handled() {
    let t = silicon();
    let empty = BTreeMap::new();
    let block = Block {
        couplings: &empty,
        ..t.block_of()
    };
    let none = matrix_derivative(&t.ctx, &block, KS[0], 0, 0).unwrap();
    assert_eq!(none.as_slice().len(), SIZE);
    assert!(none.as_slice().iter().all(|v| v.norm() == 0.));
    let bad = |owner: &[usize], axis: usize| {
        let block = Block {
            row_owner: owner,
            ..t.block_of()
        };
        matrix_derivative(&t.ctx, &block, KS[0], 0, axis)
    };
    assert!(bad(&t.row_owner[1..], 0).is_err());
    assert!(bad(&t.row_owner, 3).is_err());
}
