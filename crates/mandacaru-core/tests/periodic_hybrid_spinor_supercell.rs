//! The supercell fold of the spinor hybrid (SH-1a, gate g5): the orthorhombic 4-atom Si cell on
//! the 1x2x1 Gamma-centered mesh (the point `(0, 1/2, 0)` carries complex Bloch phases in the
//! projections and the compensation charges) against the same crystal doubled along y at Gamma,
//! pbe-fr SZ, grid_spacing 0.8 A. The per-axis grids are independent, so the two are one
//! problem and the energies per 4-atom cell must agree to the SCF tolerance. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_hybrid_spinor_support;
mod setup_support;

use mandacaru_core::periodic::Options;
use periodic_hybrid_spinor_support::*;

/// The spinor fold measured 7.1e-15 Ha (`none`) and 6.4e-14 Ha (Gygi-Baldereschi) with 49/59 and
/// 38/43 iterations; the scalar one, 1.2e-11 Ha (W-E4P / P2). Gate: 100 x the largest, rounded up.
const GATE_HA: f64 = 1e-11;

#[test]
fn the_four_atom_mesh_equals_the_eight_atom_gamma_hybrid() {
    for exxdiv in [NONE, GB] {
        let options = Options {
            tolerance: 1e-11,
            density_tolerance: 1e-9,
            max_iterations: 300,
            exxdiv_treatment: exxdiv,
            ..Options::default()
        };
        let small = hybrid(&context(), "si4_y2.json", &|_| {});
        let large = hybrid(&context(), "si8_gamma.json", &|_| {});
        assert_eq!((small.points.len(), large.points.len()), (2, 1));
        assert_eq!((small.m, large.m), (16, 32));
        assert!(small.spinor && large.spinor);
        let (a, b) = (
            small.solve(&options).unwrap(),
            large.solve(&options).unwrap(),
        );
        assert!(a.converged && b.converged);
        let (e4, e8) = (a.energy, 0.5 * b.energy);
        // The same bands, each of the 4-atom mesh points twice in the 8-atom Gamma list.
        let mut folded: Vec<f64> = a.eigenvalues.iter().flatten().copied().collect();
        folded.sort_by(f64::total_cmp);
        let mut doubled: Vec<f64> = b.eigenvalues[0].clone();
        doubled.sort_by(f64::total_cmp);
        let bands = folded
            .iter()
            .zip(&doubled)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        println!("{exxdiv:?}: E(4 atoms, 1x2x1) {e4:.12}, E(8 atoms)/2 {e8:.12}: {:.2e} Ha ({:.2e} eV); bands {bands:.2e}; {} and {} iterations", (e4 - e8).abs(), (e4 - e8).abs() * 27.211386245988, a.history.len(), b.history.len());
        assert!(
            (e4 - e8).abs() < GATE_HA,
            "{exxdiv:?}: the fold misses by {:e}",
            (e4 - e8).abs()
        );
        assert!(bands < 1e-7, "{bands:e}");
        // The hybrid is not the semilocal energy: the gate sees the exchange.
        let pbe = hybrid(&context(), "si4_y2.json", &|v| {
            v["functional"] = serde_json::json!("pbe");
            v["sites"] = serde_json::Value::Null;
        });
        let semilocal = pbe
            .solve(&Options {
                exxdiv_treatment: exxdiv,
                ..options.clone()
            })
            .unwrap();
        assert!((semilocal.energy - e4).abs() > 100. * GATE_HA);
    }
}
