//! The pair densities of the periodic screened exact exchange with compensation charges
//! (W-E4P / P1): the pair moments against the solver's multipole moments, the shapes against the
//! setup's alias-averaged hats, and the one-center seam.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::{
    periodic::{
        screened_augmentation::{Augmentation, OneCenter, OneCenterTerms},
        screened_exchange::exchange_operators_with,
    },
    Context, Result,
};
use num_complex::Complex64;
use periodic_screened_augmentation_support::{reference::Compensation, *};

#[test]
fn at_the_same_point_the_shapes_are_the_setups_alias_averaged_hats_up_to_the_box_edge() {
    // At k = q the wave vectors are the setup's G. The setup averages every shape over the
    // aliases the grid's lattice operations pick (a skewed grid cuts the star at many indices);
    // `pair_shapes` takes the one wrapped vector, which keeps K exactly translation invariant.
    // The two agree where no operation wraps; the difference elsewhere is the box-edge tail of
    // the shapes, and it moves K by the amount printed.
    for (fixture, gamma, occupied) in [
        ("si_crystal.json", true, vec![4usize]),
        ("al_fcc_gamma.json", true, vec![2]),
    ] {
        let rig = Rig::new(fixture, gamma, &occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let n = c.grid.index.points();
        let mine = aug.pair_shapes(c, [0.; 3]).unwrap();
        let hats = c.hats.read(&c.context).unwrap();
        let peak = hats.iter().map(|v| v.norm()).fold(0., f64::max);
        let (mut agree, mut differ, mut differing) = (0f64, 0f64, 0usize);
        for at in 0..n {
            let norm = |a: usize| {
                (mine.as_slice()[a * n + at] * c.grid.index.dv() * c.phase.as_slice()[at]
                    - hats[a * n + at])
                    .norm()
            };
            let worst = (0..c.channels).map(norm).fold(0., f64::max);
            if worst < 1e-12 * peak {
                agree = agree.max(worst);
            } else {
                differ = differ.max(worst);
                differing += 1;
            }
        }
        let alias: Vec<Complex64> = hats.iter().copied().collect();
        let comp = Compensation {
            sites: &sites,
            shapes: &rig.shapes,
            alias_hats: Some(&alias),
        };
        let plain = reference::exchange(
            c,
            &rig.d,
            0,
            0,
            0.11,
            GB,
            Some(&Compensation {
                alias_hats: None,
                ..comp
            }),
        );
        let mean = reference::exchange(c, &rig.d, 0, 0, 0.11, GB, Some(&comp));
        let none = reference::exchange(c, &rig.d, 0, 0, 0.11, GB, None);
        println!(
            "{fixture}: {differing} of {n} indices differ from the alias mean (up to {:.2e} of the peak {peak:.3}); \
             K moves by {:.2e} of itself against {:.2e} for the whole compensation",
            differ / peak,
            relative(&plain, &mean),
            relative(&plain, &none)
        );
        assert!(differ < 0.05 * peak, "{fixture}: {differ:e}");
        assert!(relative(&plain, &mean) < 0.05 * relative(&plain, &none));
    }
}

#[test]
fn the_pair_moments_are_the_solvers_multipole_moments() {
    // For the density of one state the diagonal pair moment is `q_a = tr(P B_a)` with
    // `P = C^+ D C`, the contraction the solver's `density` makes: all L, M, complex harmonics.
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let (mut worst, mut higher) = (0f64, 0f64);
        for point in &c.points {
            let states = coefficients(point);
            for state in 0..c.m {
                let column: Vec<Complex64> = states.column(state).iter().copied().collect();
                let moments = aug.moments(c, point, &aug.project(c, point, &column));
                let d = states.column(state) * states.column(state).adjoint();
                let projected = point.projections.adjoint() * d * &point.projections;
                for a in 0..c.channels {
                    let q: Complex64 = (0..c.p)
                        .flat_map(|i| (0..c.p).map(move |j| (i, j)))
                        .map(|(i, j)| {
                            projected[(i, j)] * c.blocks.as_slice()[(a * c.p + j) * c.p + i]
                        })
                        .sum();
                    let mine: Complex64 = (0..c.m)
                        .map(|mu| column[mu].conj() * moments[(mu, a)])
                        .sum();
                    worst = worst.max((q - mine).norm());
                    if rig.geometry.channels[a].1 > 0 {
                        higher = higher.max(q.norm());
                    }
                }
            }
        }
        println!("{fixture}: pair moments against the solver's q_a: {worst:.2e} (largest L > 0 moment {higher:.2e})");
        assert!(worst < 1e-12);
        // The hydrogen fixtures have s orbitals only: their L > 0 moments vanish.
        assert!(fixture.starts_with('h') || higher > 1e-6);
    }
}

/// A one-center model that records the projected density it was given.
struct Toy {
    shift: f64,
    energy: f64,
    seen: std::cell::RefCell<Vec<Matrix>>,
}
impl OneCenter for Toy {
    fn collinear(&self, _ctx: &Context, projected: &[Matrix]) -> Result<OneCenterTerms> {
        self.seen.borrow_mut().extend(projected.iter().cloned());
        Ok(OneCenterTerms {
            energy: self.energy * projected.len() as f64,
            operators: projected
                .iter()
                .map(|p| Matrix::identity(p.nrows(), p.ncols()) * Complex64::new(self.shift, 0.))
                .collect(),
        })
    }
    fn spinor(&self, _ctx: &Context, projected: &Matrix) -> Result<OneCenterTerms> {
        self.seen.borrow_mut().push(projected.clone());
        Ok(OneCenterTerms {
            energy: self.energy,
            operators: vec![
                Matrix::identity(projected.nrows(), projected.ncols()) * Complex64::new(self.shift, 0.),
            ],
        })
    }
}

#[test]
fn a_one_center_model_plugs_in_through_its_projected_density() {
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let toy = Toy {
            shift: 0.3,
            energy: 0.125,
            seen: Default::default(),
        };
        let bare = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let plugged = Augmentation::new(c, sites, &rig.shapes)
            .unwrap()
            .with_one_center(&toy);
        let base = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&bare)).unwrap();
        let with = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&plugged)).unwrap();
        let mut worst = 0f64;
        for spin in 0..c.spins {
            let mut expected = Matrix::zeros(c.p, c.p);
            for (k, point) in c.points.iter().enumerate() {
                expected += point.projections.adjoint()
                    * &rig.d[spin * c.points.len() + k]
                    * &point.projections
                    * Complex64::new(point.weight, 0.);
                let at = spin * c.points.len() + k;
                let added = &with.operators[at] - &base.operators[at];
                let want =
                    &point.projections * point.projections.adjoint() * Complex64::new(0.3, 0.);
                worst = worst.max((added - want).iter().map(|v| v.norm()).fold(0., f64::max));
            }
            let seen = toy.seen.borrow()[spin].clone();
            worst = worst.max(
                (seen - expected)
                    .iter()
                    .map(|v| v.norm())
                    .fold(0., f64::max),
            );
        }
        println!(
            "{fixture}: one-center operator and projected density: {worst:.2e}; energy +{:.3}",
            with.energy - base.energy
        );
        assert!(worst < 1e-13);
        assert!((with.energy - base.energy - 0.125 * c.spins as f64).abs() < 1e-14);
    }
}

#[test]
fn the_shape_tail_beyond_the_box_is_small_against_the_compensation() {
    // The Fourier sum stops at the FFT box; the compensation shapes beyond it would add
    // `w_q sum_{n, a} tail_a Q^a_{in} Q^a_{jn}^*` with the on-site tail of the Hartree `U`
    // (`Shapes::tail`, `8 int F_L^2 dq`), the screened kernel being the bare one there.
    for (fixture, occupied) in [("si_crystal.json", 4usize), ("al_fcc_gamma.json", 2)] {
        let rig = Rig::new(fixture, true, &[occupied]);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let b = mandacaru_core::math::reciprocal::reciprocal_cell(&c.grid).unwrap();
        let shape = c.grid.index.shape();
        let cutoff = (0..3)
            .map(|a| b.column(a).norm() * shape[a] as f64 / 2.)
            .fold(f64::INFINITY, f64::min);
        let point = &c.points[0];
        let states = coefficients(point);
        let mut tail = Matrix::zeros(c.m, c.m);
        for state in 0..occupied {
            let column: Vec<Complex64> = states.column(state).iter().copied().collect();
            let moments = aug.moments(c, point, &aug.project(c, point, &column));
            for (a, &(atom, l, _)) in rig.geometry.channels.iter().enumerate() {
                let weight = rig
                    .shapes
                    .tail(rig.geometry.atoms[atom].compensation_radius, l, cutoff)
                    .unwrap();
                for i in 0..c.m {
                    for j in 0..c.m {
                        tail[(i, j)] += moments[(i, a)] * moments[(j, a)].conj() * weight;
                    }
                }
            }
        }
        let with = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&aug)).unwrap();
        let without = exchange_operators_with(c, &rig.d, screening(0.11), GB, None).unwrap();
        let scale = with.operators[0]
            .iter()
            .map(|v| v.norm())
            .fold(0., f64::max);
        let tail_size = tail.iter().map(|v| v.norm()).fold(0., f64::max);
        let effect = (&with.operators[0] - &without.operators[0])
            .iter()
            .map(|v| v.norm())
            .fold(0., f64::max);
        println!(
            "{fixture}: shape tail beyond q = {cutoff:.2}: {:.1e} of K, {:.1e} of the compensation's own effect",
            tail_size / scale,
            tail_size / effect
        );
        assert!(tail_size < 1e-3 * scale);
    }
}
