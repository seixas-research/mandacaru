//! The image-resolved projection table (`periodic::setup::projection_table`): `C_k`, its
//! row-center derivative and the strain tangent of every k-point from one set of quadrature
//! sweeps. The value and the center derivative are compared with the per-k-point routines
//! (`pulay::projection`, `pulay::projection_gradient`); the strain tangent is linear in the
//! direction, and its central-difference gate is `strain_spheres.rs`. The short-range sphere table
//! (`periodic::setup::sphere_table`) is compared with `pulay::short_range_gradients`, its strain
//! gate is `strain_spheres_short_range.rs`. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;

use mandacaru_core::periodic::setup::{
    basis_gradient::ShortRange,
    projection_table::ProjectionTable,
    pulay::{projection, projection_gradient, short_range_gradients, Matrix},
    sphere_table::SphereTable,
};
use setup_gradients_support::context;
use setup_pulay_support::*;

const IDENTITY: f64 = 1e-12;
const SECOND_K: [f64; 3] = [-0.0912, 0.2371, 0.1405];

fn unit(a: usize, b: usize) -> [[f64; 3]; 3] {
    let mut e = [[0.; 3]; 3];
    e[a][b] = 1.;
    e
}

fn table_of(name: &'static str) {
    let ctx = context();
    let fx = Fixture::new(&ctx, name);
    let table = ProjectionTable::new(
        &ctx,
        &fx.shells,
        &fx.lattice,
        &fx.projectors,
        &[None, Some(0), Some(1), Some(2)],
    )
    .unwrap();
    for k in [KPOINT, SECOND_K, [0.; 3]] {
        let c = projection(&ctx, &fx.shells, &fx.lattice, k, &fx.projectors).unwrap();
        let g = projection_gradient(&ctx, &fx.shells, &fx.lattice, k, &fx.projectors).unwrap();
        assert!(peak(&c) > 1e-3 && peak(&g[0]) > 1e-3, "{name}: non-vacuous");
        let error = gap(&table.value(k).unwrap(), &c);
        assert!(error < IDENTITY, "{name} {k:?}: C differs by {error:.2e}");
        let moved = table.gradient(k).unwrap();
        for axis in 0..3 {
            let error = gap(&moved[axis], &g[axis]);
            assert!(
                error < IDENTITY,
                "{name} {k:?} axis {axis}: G_C differs by {error:.2e}"
            );
        }
        // The tangent is linear in the direction: a symmetric combination of unit tangents.
        let e = [[0.7, -0.3, 0.2], [-0.3, 1.1, 0.5], [0.2, 0.5, -0.4]];
        let along = &table.strain(k, &[e]).unwrap()[0];
        let units: Vec<Matrix> = (0..9)
            .flat_map(|i| table.strain(k, &[unit(i / 3, i % 3)]).unwrap())
            .collect();
        let mut sum = Matrix::zeros(along.nrows(), along.ncols());
        for a in 0..3 {
            for b in 0..3 {
                sum += &units[a * 3 + b] * num_complex::Complex64::new(e[a][b], 0.);
            }
        }
        assert!(
            peak(&sum) > 1e-3,
            "{name}: the strain tangent is not vacuous"
        );
        let error = gap(along, &sum);
        assert!(
            error < IDENTITY,
            "{name} {k:?}: strain along E differs by {error:.2e}"
        );
    }
}

#[test]
#[ignore = "slow: quadrature of the projector spheres of Si (run with --ignored)"]
fn the_table_reproduces_the_per_k_point_projections_of_si() {
    table_of("si");
}

#[test]
#[ignore = "slow: quadrature of the projector spheres of Al2 (run with --ignored)"]
fn the_table_reproduces_the_per_k_point_projections_of_al2() {
    table_of("al2");
}

fn spheres_of(name: &'static str) {
    let ctx = context();
    let fx = Fixture::new(&ctx, name);
    let spheres: Vec<ShortRange> = fx
        .centers
        .iter()
        .enumerate()
        .map(|(atom, center)| ShortRange {
            center: *center,
            potential: &fx.potentials[atom],
            sigma: fx.sigma,
        })
        .collect();
    let m = fx.shells.functions();
    for k in [KPOINT, SECOND_K, [0.; 3]] {
        let want = short_range_gradients(&ctx, &fx.shells, &fx.lattice, k, &spheres).unwrap();
        for (atom, sphere) in spheres.iter().enumerate() {
            let got = SphereTable::new(&ctx, &fx.shells, &fx.lattice, sphere)
                .unwrap()
                .gradient(k, m);
            for axis in 0..3 {
                assert!(
                    peak(&want[atom][axis]) > 1e-4 || axis > 0,
                    "{name}: non-vacuous"
                );
                let error = gap(&got[axis], &want[atom][axis]);
                assert!(
                    error < IDENTITY,
                    "{name} {k:?} atom {atom} axis {axis}: G differs by {error:.2e}"
                );
            }
        }
    }
}

#[test]
#[ignore = "slow: quadrature of the short-range spheres of Si (run with --ignored)"]
fn the_sphere_table_reproduces_the_per_k_point_gradients_of_si() {
    spheres_of("si");
}

#[test]
#[ignore = "slow: quadrature of the short-range spheres of Al2 (run with --ignored)"]
fn the_sphere_table_reproduces_the_per_k_point_gradients_of_al2() {
    spheres_of("al2");
}
