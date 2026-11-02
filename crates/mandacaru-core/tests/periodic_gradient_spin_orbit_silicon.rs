//! The spin-orbit force term (`periodic::gradient::spin_orbit`, CS-2) on the real union projectors
//! of Si `lda-fr`, SZ: the contraction of `SpinOrbit::add_k` (real `C^SO`, `G^SO` from the
//! scalar projection code with the spin-orbit projector list) equals `Re tr[D dh]` of the
//! block derivative `matrix_derivative`, which `setup_spin_orbit_gradient.rs` anchors to central
//! differences of the block. The toy file `periodic_gradient_spin_orbit.rs` holds the engine-only
//! gates. Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;
mod setup_spin_orbit_silicon_support;

use mandacaru_core::periodic::{
    gradient::spin_orbit::{operator, Gradient, SpinOrbit},
    setup::{
        pulay::{projection, projection_gradient},
        spin_orbit::{channels, matrices},
    },
};
use num_complex::Complex64;
use setup_spin_orbit_silicon_support::*;

/// The gate of the force term against the trace of the block derivative (measured 1e-18).
const FORCE_GATE: f64 = 1e-11;

fn spin_orbit(t: &Silicon) -> SpinOrbit {
    let found = channels(&t.ctx, &t.projectors, &t.couplings).unwrap();
    SpinOrbit::new(
        operator(&t.ctx, t.projectors.len(), &found).unwrap(),
        t.row_owner.clone(),
        t.projectors.iter().map(|p| p.atom).collect(),
        t.atoms(),
    )
    .unwrap()
}

#[test]
fn the_force_term_equals_the_trace_of_the_block_derivative_and_its_parts() {
    let t = silicon();
    let m = t.shells.functions();
    let so = spin_orbit(&t);
    let d = hermitian(2 * m);
    for (k, &point) in KS.iter().enumerate() {
        let c = projection(&t.ctx, &t.shells, &t.lattice, point, &t.projectors).unwrap();
        let g = projection_gradient(&t.ctx, &t.shells, &t.lattice, point, &t.projectors).unwrap();
        // h = (I x C) B (I x C)^+ is the block of `matrices`.
        let block = matrices(
            &t.ctx,
            &t.shells,
            &t.lattice,
            &[point],
            &t.projectors,
            &t.couplings,
        )
        .unwrap();
        let h = so.matrix(&c);
        let direct: Vec<Complex64> = (0..4 * m * m)
            .map(|x| h[(x / (2 * m), x % (2 * m))])
            .collect();
        assert!(gap(&direct, block.as_slice()) < 1e-12 * peak(&direct).max(1.));
        let mut gradient = Gradient::new(t.atoms());
        so.add_k(&mut gradient, 1., &c, &g, &d).unwrap();
        let force = gradient.total();
        let mut wrong_g = Gradient::new(t.atoms());
        so.add_k(&mut wrong_g, 1., &c, &[c.clone(), c.clone(), c.clone()], &d)
            .unwrap();
        for (atom, row) in force.iter().enumerate() {
            for (axis, &term) in row.iter().enumerate() {
                let want = trace(&d, &t.analytic(k, atom, axis));
                let error = (term - want).abs();
                eprintln!(
                    "k{k} atom {atom} axis {axis}: |term - tr[D dh]| {error:e}, term {term:e}"
                );
                assert!(error < FORCE_GATE, "atom {atom} axis {axis}: {error:e}");
                // Measured 4e-6 to 1.7e-4 on all twelve (k, atom, axis) components: every control
                // applies everywhere, none is skipped under a threshold.
                assert!(want.abs() > 1e-6, "non-vacuous: {want:e}");
                assert!((wrong_g.total()[atom][axis] - want).abs() > 100. * FORCE_GATE);
                assert!((gradient.rows[atom][axis] - want).abs() > 100. * FORCE_GATE);
            }
        }
        // Structural: the rows and the columns carry opposite signs, so the atoms sum to zero.
        for axis in 0..3 {
            let sum: f64 = force.iter().map(|x| x[axis]).sum();
            assert!(sum.abs() < 1e-12, "axis {axis}: {sum:e}");
        }
        let scale = force.iter().flatten().fold(0f64, |a, v| a.max(v.abs()));
        assert!(scale > 1e-5, "non-vacuous: {scale:e}");
    }
}
