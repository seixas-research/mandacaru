//! The crystal momentum matrix and the diamagnetic band sum (`periodic::optics::momentum`,
//! TDDFT_PLAN.md T6b0): projector-basis invariance, Hermiticity, the `-k` closure of the mesh, and
//! the band sums of small insulators. Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::periodic::{
    bloch_mesh::MeshIndex,
    optics::momentum::{check_inversion_closed, diamagnetic_sum, one_center_momentum},
};

/// The band sums of `cell` solved in `basis`, along each of `directions`, with the electron
/// count and the largest anti-Hermitian residual of `P`.
fn sums(cell: &Cell, basis: &str, mesh: [usize; 3], directions: &[[f64; 3]]) -> Vec<(f64, f64, f64)> {
    let solved = solve_in(cell, basis);
    let sampler = solved.sampler();
    let index = MeshIndex::new(mesh, [0.; 3]).unwrap();
    directions
        .iter()
        .map(|d| {
            let out = diamagnetic_sum(&sampler, &index, *d).unwrap();
            (out.total, out.electrons, out.residual)
        })
        .collect()
}

#[test]
fn the_one_center_momentum_is_hermitian_in_both_projector_bases() {
    for basis in ["raw", "dual"] {
        let solved = solve_in(&h2_chain("lda-sr", "lda"), basis);
        let blocks = one_center_momentum(&solved.ctx, &solved.setup).unwrap();
        let scale = blocks.iter().map(|b| b.camax()).fold(0f64, f64::max);
        assert!(scale > 1e-6, "{basis}: the one-center momentum vanishes");
        for b in &blocks {
            let defect = (b - b.adjoint()).camax();
            assert!(defect < 1e-10 * scale, "{basis}: not Hermitian, {defect:e}");
        }
    }
}

#[test]
fn the_diamagnetic_sum_does_not_depend_on_the_projector_basis() {
    // Raw and dual projectors are two coordinates of one PAW transform: the band sum (a physical
    // observable of the converged bands) must agree, which checks the move of Delta p to the raw
    // basis. H2 along x at Gamma; LiH on a 2 x 1 x 1 mesh along its bond.
    for (cell, mesh, label) in [
        (h2_chain("lda-sr", "lda"), [1, 1, 1], "H2"),
        (lih("lda-sr"), [2, 1, 1], "LiH"),
    ] {
        let directions = [[1., 0., 0.], [0., 1., 0.]];
        let raw = sums(&cell, "raw", mesh, &directions);
        let dual = sums(&cell, "dual", mesh, &directions);
        for ((direction, (raw, electrons, residual)), (dual, _, _)) in
            directions.iter().zip(raw).zip(dual)
        {
            eprintln!(
                "{label} along {direction:?}: N_dia raw {raw:.8}, dual {dual:.8}, N {electrons}, \
                 anti-Hermitian residual {residual:.2e}"
            );
            // A minimal s basis has no transition across the H2 axis: the sum is zero there.
            assert!(
                raw >= 0. && (raw - dual).abs() <= 1e-6 * raw.abs().max(1e-12),
                "{label}: {raw} vs {dual}"
            );
        }
    }
}

#[test]
fn a_mesh_not_closed_under_inversion_is_refused() {
    assert!(check_inversion_closed(&MeshIndex::new([2, 2, 1], [0.; 3]).unwrap()).is_ok());
    assert!(check_inversion_closed(&MeshIndex::new([3, 1, 1], [0.; 3]).unwrap()).is_ok());
    let shifted = MeshIndex::new([2, 1, 1], [0.1, 0., 0.]).unwrap();
    let message = check_inversion_closed(&shifted).unwrap_err().to_string();
    assert!(message.contains("-k"), "{message}");
}
