//! The Hirshfeld shares of the superposed smooth valence densities
//! (`periodic::setup::fields::valence_shares`, the WP-X port of `PeriodicPAW.valence_shares`)
//! against the frozen Python (ideal diamond Si, two atoms, and primitive fcc Al, one atom),
//! and the identities of a partition: the shares of a point sum to one, a lone atom owns
//! everything, and equal atoms split a point by their distance. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;

use mandacaru_core::{
    math::SpatialGrid,
    periodic::setup::fields::{superposed_valence, valence_shares, Placed, Tables},
};
use nalgebra::Matrix3;
use setup_gradients_support::*;

fn grid(case: &Case) -> SpatialGrid {
    let shape: Vec<usize> = case.inputs()["grid_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let lattice = Matrix3::from_row_slice(&Array::read(&case.inputs()["lattice_bohr"]).re);
    let step = Matrix3::from_fn(|r, c| lattice[(r, c)] / shape[c] as f64);
    // The step as the grid takes it: row-major.
    let flat: [f64; 9] = std::array::from_fn(|i| step[(i / 3, i % 3)]);
    SpatialGrid::new([shape[0], shape[1], shape[2]], [0.; 3], flat, true).unwrap()
}

#[test]
fn the_shares_equal_the_frozen_python_and_partition_the_density() {
    for name in ["si_sym", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let g = grid(&case);
        let centers = triples(&Array::read(&case.inputs()["centers_bohr"]).re);
        let datasets: Vec<_> = (0..centers.len()).map(|a| case.dataset(&ctx, a)).collect();
        let tables: Vec<Tables> = datasets
            .iter()
            .map(|d| Tables {
                radius: d.r.as_slice(),
                valence_density: d.valence_density.as_slice(),
                core: None,
            })
            .collect();
        let atoms: Vec<Placed> = centers
            .iter()
            .enumerate()
            .map(|(species, c)| Placed {
                species,
                center: *c,
            })
            .collect();
        let shares = valence_shares(&ctx, &g, &tables, &atoms).unwrap();
        let n = g.index.points();
        let want = case.quantity("valence_shares").re;
        let miss = relative_real(shares.as_slice(), &want);
        let gate = case.gate("valence_shares");
        // The shares of every point with density sum to one.
        let total = |i: usize| {
            (0..atoms.len())
                .map(|a| shares.as_slice()[a * n + i])
                .sum::<f64>()
        };
        let worst = (0..n)
            .filter(|&i| total(i) > 0.)
            .map(|i| (total(i) - 1.).abs())
            .fold(0., f64::max);
        println!("{name}: shares {miss:.2e} (gate {gate:.1e}), partition of one {worst:.1e}");
        assert!(miss <= gate && worst < 1e-14);
        if atoms.len() == 1 {
            assert!(shares
                .as_slice()
                .iter()
                .all(|s| *s == 0. || (*s - 1.).abs() < 1e-15));
        } else {
            // The two equal atoms of the ideal cell split the cell: each share spans (0, 1), and the
            // share of an atom is its density over the pair's (control: the plain density is no share).
            let own = superposed_valence(&ctx, &g, &tables, &atoms, &[1., 0.]).unwrap();
            let other = superposed_valence(&ctx, &g, &tables, &atoms, &[0., 1.]).unwrap();
            let spread = shares.as_slice()[..n].iter().cloned().fold(0., f64::max);
            assert!(spread > 0.6 && spread <= 1.);
            let by_hand = (0..n)
                .filter(|&i| own.as_slice()[i] + other.as_slice()[i] > 0.)
                .map(|i| {
                    (shares.as_slice()[i]
                        - own.as_slice()[i] / (own.as_slice()[i] + other.as_slice()[i]))
                        .abs()
                })
                .fold(0., f64::max);
            assert!(by_hand < 1e-15);
            assert!(relative_real(own.as_slice(), &want[..n]) > 1e3 * gate);
        }
    }
}
