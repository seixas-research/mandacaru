//! H-3 and the unit-charge pair: the strain derivative of the pair shapes `T_a(G) = hat_a(Q)
//! e^{iG.r0} / dv` (`Augmentation::pair_shape_strain`, `-delta_ab T_a - Q_a d_b f_a`) against the
//! central difference of `Augmentation::pair_shapes` over the strained cell at FIXED fractional
//! `k_diff`, entry by entry (nine, both orders), and the derivative of the pair at `G = 0`, which
//! for `k = q` is the overlap `c_m^+ S c_n / dv` (the sum rule of the pair densities), against the
//! central difference of that overlap. Needs `MANDACARU_PAW_PATH`.
//!
//! The shape tables are piecewise linear with a node every 0.01 Bohr^-1: a frequency whose
//! `|Q|` crosses a node between `-h` and `+h` carries the kink and not the slope the analytic
//! term has. Such a frequency is recognized by the disagreement of the central differences at
//! `h` and `h / 2` (a smooth entry agrees to `O(h^2)`), counted, and held out of the entry-wise
//! comparison; the count is asserted small. The table is not touched.
mod periodic_gradient_crystals_support;
mod periodic_stress_hybrid_support;

use mandacaru_core::{
    periodic::{screened_exchange::ExxDivTreatment, stress_terms::hybrid::Tangents},
    quantum::C64,
};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use periodic_stress_hybrid_support::*;

const STEP: f64 = 2e-6;

/// The Cartesian `k_diff` of the strained cell at fixed fractional `k_diff`.
fn carried(eps: &Eps, k: [f64; 3]) -> [f64; 3] {
    let map = Matrix3::identity() + Matrix3::from_fn(|a, b| eps[a][b]);
    (map.transpose().try_inverse().unwrap() * Vector3::from(k)).into()
}

/// `pair_shapes` of the crystal strained by `eps` in the cell.
fn shapes(rig: &mut Rig, eps: &Eps, k: [f64; 3]) -> Vec<C64> {
    let k = carried(eps, k);
    rig.strained(eps, &CELL, |c, aug| {
        aug.pair_shapes(c, k).unwrap().as_slice().to_vec()
    })
}

fn analytic(rig: &mut Rig, k: [f64; 3], a: usize, b: usize) -> (Vec<C64>, Vec<C64>) {
    rig.strained(&entry(0, 0, 0.), &CELL, |c, aug| {
        let t = aug.pair_shapes(c, k).unwrap();
        let form = aug
            .pair_shape_strain(c, k)
            .unwrap()
            .form(&c.context, a, b)
            .unwrap();
        (t.as_slice().to_vec(), form.as_slice().to_vec())
    })
}

fn check(k: [f64; 3]) {
    let mut rig = Rig::new("lih_hse06", ExxDivTreatment::None);
    let (mut worst, mut transposed) = (0f64, 0f64);
    let (mut held, mut total) = (0usize, 0usize);
    let mut peak = 0f64;
    let forms: Vec<Vec<(Vec<C64>, Vec<C64>)>> = (0..3)
        .map(|a| (0..3).map(|b| analytic(&mut rig, k, a, b)).collect())
        .collect();
    for a in 0..3 {
        for b in 0..3 {
            let central = |rig: &mut Rig, h: f64| -> Vec<C64> {
                let (p, m) = (
                    shapes(rig, &entry(a, b, h), k),
                    shapes(rig, &entry(a, b, -h), k),
                );
                p.iter().zip(&m).map(|(p, m)| (p - m) / (2. * h)).collect()
            };
            let (fine, coarse) = (central(&mut rig, STEP / 2.), central(&mut rig, STEP));
            let (t, form) = &forms[a][b];
            // The transposed order `-Q_b d_a` is a different tensor.
            let swapped = &forms[b][a].1;
            for (v, ((f, c), t)) in fine.iter().zip(&coarse).zip(t.iter()).enumerate() {
                let delta = if a == b { -*t } else { C64::default() };
                let want = delta + form[v];
                peak = peak.max(want.norm());
                total += 1;
                // A kink: the two differences disagree far beyond the quadratic remainder.
                if (f - c).norm() > 1e-6 * t.norm().max(1.) {
                    held += 1;
                    continue;
                }
                worst = worst.max((f - want).norm());
                transposed = transposed.max((f - (delta + swapped[v])).norm());
            }
        }
    }
    println!(
        "  k_diff {k:?}: {held} of {total} entries held out as kinks, worst miss {worst:.2e} of peak {peak:.2e}; the transposed order misses by {transposed:.2e}"
    );
    assert!(
        held * 100 < total,
        "too many kink entries: {held} of {total}"
    );
    assert!(
        worst < 1e-8 * peak,
        "the shape strain misses the difference by {worst:e} of {peak:e}"
    );
    // At the head pair (and any pair with the symmetric part dominating) the control bites
    // through the p and d channels: it must miss by more than 100 times the right miss.
    assert!(
        transposed > 100. * worst,
        "the transposed index order bites only {:.1} times the right miss",
        transposed / worst
    );
}

#[test]
fn the_pair_shapes_strain_matches_the_difference_at_the_head_pair() {
    check([0.; 3]);
}

#[test]
fn the_pair_shapes_strain_matches_the_difference_at_a_generic_pair() {
    check([0.173205, -0.141421, 0.0866025]);
}

/// Two fixed states `c_m`, `c_n` (coefficient vectors over the basis).
fn states(m: usize) -> (DVector<C64>, DVector<C64>) {
    (
        DVector::from_fn(m, |i, _| C64::new(seq(i, 31), seq(i, 37))),
        DVector::from_fn(m, |i, _| C64::new(seq(i, 41), seq(i, 43))),
    )
}

/// The pair at `G = 0` of the states `c_m`, `c_n` at the Gamma point as the exchange builds it,
/// `F(0) = sum_r conj(psi_m) psi_n + sum_i conj(c_mi) sum_a mom[i, a] T_a(0)`, on the crystal
/// strained by `eps`, and the overlap `c_m^+ (G + C q C^+ / dv) c_n` it must equal (the sum
/// rule of the pair densities: the grid sum plus the unit-charge compensation is the overlap).
fn unit_charge(rig: &mut Rig, eps: &Eps) -> (C64, C64) {
    let (cm, cn) = states(rig.data.crystal.m);
    let q = rig.data.augmentation.clone();
    rig.strained(eps, &ALL, |c, aug| {
        let (m, n) = (c.m, c.grid.index.points());
        let point = &c.points[0];
        let samples = point.psi.read(&c.context).unwrap();
        let gram = DMatrix::from_fn(m, m, |i, j| {
            (0..n)
                .map(|r| samples[i * n + r].conj() * samples[j * n + r])
                .sum::<C64>()
        });
        let moments = aug.moments(c, point, &aug.project(c, point, cn.as_slice()));
        let t0 = aug.pair_shapes(c, [0.; 3]).unwrap();
        let compensation: C64 = (0..m)
            .map(|i| {
                cm[i].conj()
                    * (0..c.channels)
                        .map(|a| moments[(i, a)] * t0.as_slice()[a * n])
                        .sum::<C64>()
            })
            .sum();
        let pair = (cm.adjoint() * &gram * &cn)[(0, 0)] + compensation;
        let projections = &point.projections;
        let overlap = (cm.adjoint() * &gram * &cn)[(0, 0)]
            + (cm.adjoint() * projections * &q * projections.adjoint() * &cn)[(0, 0)]
                / C64::new(c.grid.index.dv(), 0.);
        (pair, overlap)
    })
}

#[test]
fn the_unit_charge_pair_is_the_overlap_and_so_are_their_strain_derivatives() {
    let mut rig = Rig::new("lih_hse06", ExxDivTreatment::None);
    let (pair, overlap) = unit_charge(&mut rig, &entry(0, 0, 0.));
    println!("  F(0) {pair:.12}, overlap {overlap:.12}");
    assert!((pair - overlap).norm() < 1e-10 * overlap.norm().max(1.));
    // The analytic derivative of F(0) from the tangents of the setup: the pair of the grid
    // samples, `-delta_ab` on the compensation (the `1 / dv` of `T_a`), the form factors vanish
    // at `Q = 0`, and the moments through `dC`.
    let analytic: Vec<Vec<C64>> = {
        let (cm, cn) = states(rig.data.crystal.m);
        let cores = rig.data.cores();
        let inp = periodic_gradient_crystals_support::inputs(&rig.data, &cores);
        let tangents = Tangents::new(&rig.data.crystal.context, &inp).unwrap();
        let c = &rig.data.crystal;
        let (m, n, p) = (c.m, c.grid.index.points(), c.p);
        let samples = c.points[0].psi.read(&c.context).unwrap();
        let projections = &c.points[0].projections;
        let dc = tangents.projection_strain(0).unwrap();
        let blocks: Vec<DMatrix<C64>> = (0..c.channels)
            .map(|a| DMatrix::from_fn(p, p, |i, j| c.blocks.as_slice()[(a * p + i) * p + j]))
            .collect();
        let aug = c.hybrid_augmentation().unwrap();
        let t0 = aug.pair_shapes(c, [0.; 3]).unwrap();
        let a_n = projections.adjoint() * &cn;
        (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| {
                        let tangent = tangents.samples(0, a, b).unwrap();
                        let dg = DMatrix::from_fn(m, m, |i, j| {
                            (0..n)
                                .map(|r| {
                                    tangent.as_slice()[i * n + r].conj() * samples[j * n + r]
                                        + samples[i * n + r].conj() * tangent.as_slice()[j * n + r]
                                })
                                .sum::<C64>()
                        });
                        let mut derivative = (cm.adjoint() * &dg * &cn)[(0, 0)];
                        for (channel, block) in blocks.iter().enumerate() {
                            let b_a = block * &a_n;
                            let moment = projections * &b_a;
                            let dmoment =
                                &dc[a][b] * &b_a + projections * block * (dc[a][b].adjoint() * &cn);
                            let shift = if a == b {
                                -moment
                            } else {
                                moment.map(|_| C64::default())
                            };
                            derivative += (cm.adjoint() * (dmoment + shift))[(0, 0)]
                                * t0.as_slice()[channel * n];
                        }
                        derivative
                    })
                    .collect()
            })
            .collect()
    };
    let mut worst = 0f64;
    let mut largest = 0f64;
    for a in 0..3 {
        for b in 0..3 {
            // The central differences of F(0) and of the overlap.
            let step = 2e-6;
            let (fp, sp) = unit_charge(&mut rig, &entry(a, b, step));
            let (fm, sm) = unit_charge(&mut rig, &entry(a, b, -step));
            let fd_pair = (fp - fm) / (2. * step);
            let fd_overlap = (sp - sm) / (2. * step);
            worst = worst.max((analytic[a][b] - fd_pair).norm());
            worst = worst.max((fd_pair - fd_overlap).norm());
            largest = largest.max(fd_pair.norm());
        }
    }
    println!("  worst miss of dF(0)/d eps over the nine entries {worst:.2e} of {largest:.2e}");
    assert!(largest > 1e-3, "the derivative is not live");
    assert!(worst < 1e-8 * largest.max(1.), "{worst:e}");
}

/// The pins of the hybrid terms that are not tangents: the shape tables span `Shapes::top`, a
/// number the strained cell must not exceed (`max(q_max, 15, 120)`: a table is built once, at the
/// reference cell), and the augmentation lattice is the cell of the grid (`Augmentation::new`
/// refuses anything else, so a strained rig that built one holds it).
#[test]
fn the_shape_tables_cover_every_cell_the_oracle_strains() {
    let rig = Rig::new("lih_hse06", ExxDivTreatment::None);
    let ctx = &rig.data.crystal.context;
    let top = rig.data.shapes.top();
    assert_eq!(top, 120., "the table span of the reference cell");
    // The largest strain of any frozen family (4e-3 per entry) and a compression twice as deep.
    for e in [-8e-3, -4e-3, 0., 4e-3] {
        for a in 0..3 {
            for b in 0..3 {
                let map = Matrix3::identity() + Matrix3::from_fn(|i, j| entry(a, b, e)[i][j]);
                let reference = &rig.data.crystal.grid;
                let step = map * reference.step;
                let mut rows = [0.; 9];
                for (i, v) in rows.iter_mut().enumerate() {
                    *v = step[(i / 3, i % 3)];
                }
                let grid = mandacaru_core::math::SpatialGrid::new(
                    reference.index.shape(),
                    (map * Vector3::from(reference.origin)).into(),
                    rows,
                    true,
                )
                .unwrap();
                let q_max =
                    mandacaru_core::periodic::setup::electrostatics::max_wavenumber(ctx, &grid)
                        .unwrap();
                assert!(
                    q_max < top,
                    "the strained cell reaches {q_max} past the tables"
                );
            }
        }
    }
}
