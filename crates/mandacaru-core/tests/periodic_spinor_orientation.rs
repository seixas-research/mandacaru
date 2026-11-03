//! F2b: the orientation of a magnetic spinor state, the nearly neutral mode that stalled the SCF.
//!
//! A high-spin N atom next to a Ne atom in a box (`nne_ldafr`, the `lda-fr` datasets, spin-orbit
//! coupling on) is a magnetic spinor whose spin-orbit anisotropy is about `7e-7` Hartree. The
//! rotation of the whole moment against the lattice is a mode of the SCF map with an eigenvalue
//! `1 - O(1e-5)`: the Pulay mixer, and plain linear mixing at any weight, left a transverse density
//! residual of `4e-7` (L1) for 1500 iterations while the energy was converged to `1e-13`. The
//! solver puts the output state at the spin-orbit-optimal rigid orientation (closed form, the
//! Procrustes problem of the anisotropy tensor; every other term of the energy is invariant under a
//! global spin rotation), which turns the mode into one the mixer contracts at once.
//!
//! * the SCF reaches the density tolerance 1e-9 from three start directions (`z`, `x`, `y`) with
//!   the coupling on, the converged moment is asserted magnetic and tilted away from the start by
//!   the anisotropy, and the orientation of the fixed point is already optimal (the rotation the
//!   solver would apply is the identity);
//! * the sign convention of the rotation is pinned apart from the solver: a converged state turned
//!   by a known `U0` is brought back to the original density matrices by the rotation the solver
//!   chooses (`<= 1e-9`), and the spin-orbit energy goes down;
//! * without a spin-orbit matrix (the zero-coupling twin) or for a time-reversal-symmetric crystal
//!   no rotation is ever applied.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::spinor::{optimal_rotation, rotate_density, spin_orbit_energy},
    quantum::C64,
};
use periodic_gradient_spinor_support::{moment, norm, Axis, Spinor};

const CASE: &str = "nne_ldafr";

/// The largest entry of `a - b`.
fn gap(a: &nalgebra::DMatrix<C64>, b: &nalgebra::DMatrix<C64>) -> f64 {
    (a - b).iter().map(|v| v.norm()).fold(0., f64::max)
}

/// `u` and `-u` are the same rotation: the closer of the two to `v`.
fn same_rotation(u: &[[C64; 2]; 2], v: &[[C64; 2]; 2]) -> f64 {
    let plus = (0..2)
        .flat_map(|i| (0..2).map(move |j| (u[i][j] - v[i][j]).norm()))
        .fold(0., f64::max);
    let minus = (0..2)
        .flat_map(|i| (0..2).map(move |j| (u[i][j] + v[i][j]).norm()))
        .fold(0., f64::max);
    plus.min(minus)
}

#[test]
fn a_spin_orbit_magnetic_state_converges_from_three_start_directions() {
    for (axis, label) in [(None, "z"), (Some(Axis::X), "x"), (Some(Axis::Y), "y")] {
        let start = std::time::Instant::now();
        let spinor = match axis {
            None => Spinor::load(CASE),
            Some(axis) => Spinor::rotated(CASE, axis, true),
        };
        // The oracle tolerances of the manifest: density 1e-9, free energy 1e-11 Ha.
        assert_eq!(spinor.data.options.density_tolerance, 1e-9);
        let result = spinor.data.solve();
        let last = result.history.last().unwrap();
        let m = moment(&spinor.data, &result);
        let size = norm(m);
        let index = match label {
            "z" => 2,
            "x" => 0,
            _ => 1,
        };
        let tilt = (1. - (m[index] / size).powi(2)).max(0.).sqrt();
        eprintln!(
            "{label}: {} iterations, density residual {:.2e}, moment {m:.5?} (tilt {tilt:.3} rad of the start), free energy {:.12}, spin-orbit energy {:.6e}, {:.0} s",
            result.history.len(),
            last.residual,
            result.free_energy,
            result.spin_orbit_energy.unwrap(),
            start.elapsed().as_secs_f64()
        );
        assert!(
            last.residual < 1e-9,
            "{label}: residual {:e}",
            last.residual
        );
        // A high-spin N atom: 3 Bohr magnetons, mostly along the start.
        assert!((size - 3.0).abs() < 0.01, "{label}: {m:?}");
        assert!(
            m[index] / size > 0.9,
            "{label}: the state left the start {m:?}"
        );
        // The fixed point sits at its optimal orientation: the rotation the solver would apply is
        // the identity (None, or a rotation below the noise of the anisotropy tensor).
        let c = &spinor.data.crystal;
        match optimal_rotation(c, &result.matrices).unwrap() {
            None => {}
            Some(u) => {
                let identity = [
                    [C64::new(1., 0.), C64::default()],
                    [C64::default(), C64::new(1., 0.)],
                ];
                assert!(
                    same_rotation(&u, &identity) < 1e-6,
                    "{label}: the converged state is not at its optimal orientation: {u:?}"
                );
            }
        }
    }
}

#[test]
fn the_rotation_the_solver_chooses_brings_a_turned_state_back() {
    let spinor = Spinor::load(CASE);
    let result = spinor.data.solve();
    let c = &spinor.data.crystal;
    let energy = spin_orbit_energy(c, &result.matrices).unwrap();
    // A rotation of 0.05 rad about the axis (1, 2, 2) / 3: u = cos(t/2) - i sin(t/2) n.sigma.
    let (angle, axis) = (0.05f64, [1. / 3., 2. / 3., 2. / 3.]);
    let (w, s) = ((angle / 2.).cos(), (angle / 2.).sin());
    let i = C64::i();
    let (x, y, z) = (axis[0] * s, axis[1] * s, axis[2] * s);
    let turn = [
        [C64::new(w, 0.) - i * z, -i * x - C64::new(y, 0.)],
        [-i * x + C64::new(y, 0.), C64::new(w, 0.) + i * z],
    ];
    let turned: Vec<_> = result
        .matrices
        .iter()
        .map(|d| {
            mandacaru_core::math::storage::StoredMatrix::new(
                &c.context,
                rotate_density(&d.matrix(&c.context).unwrap(), c.m, &turn),
                false,
            )
            .unwrap()
        })
        .collect();
    let moved = spin_orbit_energy(c, &turned).unwrap();
    // The turned state has a higher spin-orbit energy (the converged one is its minimum) and the
    // chosen rotation undoes it.
    eprintln!(
        "spin-orbit energy: converged {energy:.9e}, turned by {angle} rad {moved:.9e} (change {:.3e} Ha)",
        moved - energy
    );
    assert!(moved - energy > 1e-10, "the turned state must cost energy");
    let u = optimal_rotation(c, &turned)
        .unwrap()
        .expect("a turned state has a rotation to apply");
    let mut worst = 0f64;
    for (d, t) in result.matrices.iter().zip(&turned) {
        let back = rotate_density(&t.matrix(&c.context).unwrap(), c.m, &u);
        worst = worst.max(gap(&back, &d.matrix(&c.context).unwrap()));
    }
    eprintln!("turned state rotated back against the converged density matrices: {worst:.2e}");
    assert!(worst < 1e-6, "{worst:e}");
    // A wrong sign (the inverse rotation applied instead) does not bring it back.
    let inverse = [
        [u[0][0].conj(), u[1][0].conj()],
        [u[0][1].conj(), u[1][1].conj()],
    ];
    let wrong = result
        .matrices
        .iter()
        .zip(&turned)
        .map(|(d, t)| {
            gap(
                &rotate_density(&t.matrix(&c.context).unwrap(), c.m, &inverse),
                &d.matrix(&c.context).unwrap(),
            )
        })
        .fold(0., f64::max);
    eprintln!("control, the inverse rotation: {wrong:.2e}");
    assert!(wrong > 100. * worst.max(1e-9), "{wrong:e}");
}

#[test]
fn no_rotation_is_applied_without_a_spin_orbit_anisotropy_to_align() {
    // The zero-coupling twin of a magnetic crystal has a zero spin-orbit matrix: nothing to align.
    let twin = Spinor::magnetic_twin("beh_lda", [1., 2., 2.]);
    let c = &twin.data.crystal;
    assert!(c.points.iter().all(|p| p.spin_orbit.is_some()));
    // Any density matrices: with a zero anisotropy tensor the answer does not depend on them.
    let densities: Vec<_> = c
        .points
        .iter()
        .map(|_| {
            mandacaru_core::math::storage::StoredMatrix::new(
                &c.context,
                nalgebra::DMatrix::<C64>::identity(2 * c.m, 2 * c.m) * C64::new(0.5, 0.),
                false,
            )
            .unwrap()
        })
        .collect();
    assert_eq!(densities.len(), c.points.len());
    assert!(optimal_rotation(c, &densities).unwrap().is_none());
}
