//! The streamed grid terms of the force (`periodic::setup::pulay::cross_streamed`,
//! `kinetic_setup`) against the whole-mesh routines they replace (`grid_cross`, `kinetic_cross` on
//! the `KSamples` of the whole mesh), on the Si fixture, at two k-points and with a block much
//! smaller than the mesh. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;

use mandacaru_core::periodic::setup::pulay::{
    cross_streamed_blocks, grid_cross, kinetic_cross, kinetic_setup, KSamples, Ket, Mesh,
};
use setup_gradients_support::context;
use setup_pulay_support::*;

const IDENTITY: f64 = 1e-11;

#[test]
#[ignore = "slow: whole-mesh samples of Si (run with --ignored)"]
fn the_streamed_grid_terms_are_the_whole_mesh_ones() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    let mesh = Mesh::new(&ctx, &fx.grid).unwrap();
    let n = mesh.len();
    let v: Vec<f64> = (0..n)
        .map(|i| 0.3 + ((i * 7919) % 101) as f64 / 101.)
        .collect();
    for k in [KPOINT, [0.; 3]] {
        let samples = KSamples::new(&ctx, &fx.shells, &fx.lattice, &mesh, k).unwrap();
        let overlap = grid_cross(&ctx, &mesh, &samples, None).unwrap();
        let local = grid_cross(&ctx, &mesh, &samples, Some(&v)).unwrap();
        let kinetic = kinetic_cross(&ctx, &fx.grid, &mesh, &samples).unwrap();
        let setup =
            kinetic_setup(&ctx, &fx.grid, &mesh, &fx.shells, &fx.lattice, k, false).unwrap();
        // Blocks of 777 points: many blocks, the last one short.
        let got = cross_streamed_blocks(
            &ctx,
            &fx.shells,
            &fx.lattice,
            &mesh,
            k,
            &[
                Ket::Weighted(None),
                Ket::Weighted(Some(&v)),
                Ket::Columns(&setup.ket),
            ],
            777,
        )
        .unwrap();
        for axis in 0..3 {
            for (name, want, have) in [
                ("overlap", &overlap[axis], &got[0][axis]),
                ("local", &local[axis], &got[1][axis]),
                ("kinetic", &kinetic[axis], &got[2][axis]),
            ] {
                assert!(peak(want) > 1e-4, "{name}: non-vacuous");
                let error = gap(want, have);
                assert!(
                    error < IDENTITY,
                    "{name} {k:?} axis {axis}: differs by {error:.2e}"
                );
            }
        }
    }
}
