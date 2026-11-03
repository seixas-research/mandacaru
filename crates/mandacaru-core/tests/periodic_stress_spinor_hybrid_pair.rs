//! AS-10, the unit-charge identity of the SPINOR pair: for two spinor states `c_m`, `c_n` at
//! Gamma the spin-summed pair at `G = 0` as the exchange builds it (the grid sum over the two
//! components plus the compensation of the `2m` rows) is the spin-summed overlap
//! `sum_sigma c_{m sigma}^+ (G + C q C^+ / dv) c_{n sigma}`, and so are their strain derivatives:
//! the analytic one from the tangents of the setup (`A_ab` per component, `-delta_ab` on the
//! compensation, `dC` through the moments of each component) against the central differences of
//! both, all nine entries. The states have both spin components populated (the second is asserted
//! above a tenth of the first). Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_hybrid_support;
mod periodic_stress_spinor_hybrid_support;

use mandacaru_core::{
    periodic::{screened_exchange::ExxDivTreatment, stress_terms::hybrid::Tangents},
    quantum::C64,
};
use nalgebra::{DMatrix, DVector};
use periodic_stress_hybrid_support::{entry, seq, Eps, Rig, ALL};
use periodic_stress_spinor_hybrid_support::rig;

fn states(m2: usize) -> (DVector<C64>, DVector<C64>) {
    (
        DVector::from_fn(m2, |i, _| C64::new(seq(i, 31), seq(i, 37))),
        DVector::from_fn(m2, |i, _| C64::new(seq(i, 41), seq(i, 43))),
    )
}

/// The component `s` (`m` entries) of a spinor vector.
fn part(v: &DVector<C64>, s: usize, m: usize) -> DVector<C64> {
    v.rows(s * m, m).into_owned()
}

/// `(pair, overlap, pair of the first component's rows alone)`.
fn unit_charge(rig: &mut Rig, eps: &Eps) -> (C64, C64, C64) {
    let m = rig.data.crystal.m;
    let (cm, cn) = states(2 * m);
    let q = rig.data.augmentation.clone();
    rig.strained(eps, &ALL, |c, aug| {
        let n = c.grid.index.points();
        let point = &c.points[0];
        let samples = point.psi.read(&c.context).unwrap();
        let gram = DMatrix::from_fn(m, m, |i, j| {
            (0..n)
                .map(|r| samples[i * n + r].conj() * samples[j * n + r])
                .sum::<C64>()
        });
        let moments = aug.moments(c, point, &aug.project(c, point, cn.as_slice()));
        assert_eq!(moments.nrows(), 2 * m, "a spinor state has 2m moment rows");
        let t0 = aug.pair_shapes(c, [0.; 3]).unwrap();
        let compensation_of = |rows: std::ops::Range<usize>| -> C64 {
            rows.map(|i| {
                cm[i].conj()
                    * (0..c.channels)
                        .map(|a| moments[(i, a)] * t0.as_slice()[a * n])
                        .sum::<C64>()
            })
            .sum()
        };
        let compensation = compensation_of(0..2 * m);
        let (mut grid, mut charge) = (C64::default(), C64::default());
        for s in 0..2 {
            let (ms, ns) = (part(&cm, s, m), part(&cn, s, m));
            grid += (ms.adjoint() * &gram * &ns)[(0, 0)];
            charge += (ms.adjoint() * &point.projections * &q * point.projections.adjoint() * &ns)
                [(0, 0)]
                / C64::new(c.grid.index.dv(), 0.);
        }
        let up =
            (part(&cm, 0, m).adjoint() * &gram * part(&cn, 0, m))[(0, 0)] + compensation_of(0..m);
        (grid + compensation, grid + charge, up)
    })
}

#[test]
fn the_spinor_unit_charge_pair_is_the_overlap_and_so_are_their_strain_derivatives() {
    let mut rig = rig("lih_hse06", ExxDivTreatment::None, false);
    let m = rig.data.crystal.m;
    let (cm, cn) = states(2 * m);
    assert!(
        part(&cn, 1, m).norm() > 0.1 * part(&cn, 0, m).norm()
            && part(&cm, 1, m).norm() > 0.1 * part(&cm, 0, m).norm(),
        "both spin components must be populated"
    );
    let (pair, overlap, up) = unit_charge(&mut rig, &entry(0, 0, 0.));
    println!("  F(0) {pair:.12}, overlap {overlap:.12}");
    assert!((pair - overlap).norm() < 1e-10 * overlap.norm().max(1.));
    let analytic: Vec<Vec<C64>> = {
        let cores = rig.data.cores();
        let inp = periodic_gradient_crystals_support::inputs(&rig.data, &cores);
        let tangents = Tangents::new(&rig.data.crystal.context, &inp).unwrap();
        let c = &rig.data.crystal;
        let (n, p) = (c.grid.index.points(), c.p);
        let samples = c.points[0].psi.read(&c.context).unwrap();
        let projections = &c.points[0].projections;
        let dc = tangents.projection_strain(0).unwrap();
        let blocks: Vec<DMatrix<C64>> = (0..c.channels)
            .map(|a| DMatrix::from_fn(p, p, |i, j| c.blocks.as_slice()[(a * p + i) * p + j]))
            .collect();
        let aug = c.hybrid_augmentation().unwrap();
        let t0 = aug.pair_shapes(c, [0.; 3]).unwrap();
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
                        let mut derivative = C64::default();
                        for s in 0..2 {
                            let (ms, ns) = (part(&cm, s, m), part(&cn, s, m));
                            derivative += (ms.adjoint() * &dg * &ns)[(0, 0)];
                            let a_n = projections.adjoint() * &ns;
                            for (channel, block) in blocks.iter().enumerate() {
                                let b_a = block * &a_n;
                                let moment = projections * &b_a;
                                let dmoment = &dc[a][b] * &b_a
                                    + projections * block * (dc[a][b].adjoint() * &ns);
                                let shift = if a == b {
                                    -moment
                                } else {
                                    moment.map(|_| C64::default())
                                };
                                derivative += (ms.adjoint() * (dmoment + shift))[(0, 0)]
                                    * t0.as_slice()[channel * n];
                            }
                        }
                        derivative
                    })
                    .collect()
            })
            .collect()
    };
    let (mut worst, mut largest) = (0f64, 0f64);
    for a in 0..3 {
        for b in 0..3 {
            let step = 2e-6;
            let (fp, sp, _) = unit_charge(&mut rig, &entry(a, b, step));
            let (fm, sm, _) = unit_charge(&mut rig, &entry(a, b, -step));
            let (fd_pair, fd_overlap) = ((fp - fm) / (2. * step), (sp - sm) / (2. * step));
            worst = worst.max((analytic[a][b] - fd_pair).norm());
            worst = worst.max((fd_pair - fd_overlap).norm());
            largest = largest.max(fd_pair.norm());
        }
    }
    println!("  worst miss of dF(0)/d eps over the nine entries {worst:.2e} of {largest:.2e}");
    assert!(largest > 1e-3, "the derivative is not live");
    assert!(worst < 1e-8 * largest.max(1.), "{worst:e}");
    // The control: the rows of the first component alone (a scalar stand-in) miss the identity.
    let bite = (pair - up).norm() / (1e-10 * overlap.norm().max(1.));
    println!("  control first component only: bites {bite:.2e} gates");
    assert!(bite > 100., "{bite:e}");
}
