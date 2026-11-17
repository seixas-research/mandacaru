//! The pair densities of the periodic screened exact exchange with compensation charges
//! (W-E4P / P1): the pair moments against the solver's multipole moments, the shapes against the
//! setup's alias-averaged hats, the shapes of a pair and of its reverse (reciprocity, with the
//! faces of the box), and the one-center seam.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::{
    math::{fft::frequency, reciprocal},
    periodic::{
        screened_augmentation::{Augmentation, OneCenter, OneCenterTerms},
        screened_exchange::exchange_operators_with,
    },
    Context, Result,
};
use nalgebra::Vector3;
use num_complex::Complex64;
use periodic_screened_augmentation_support::{reference::Compensation, *};

#[test]
fn at_the_same_point_the_shapes_are_the_setups_alias_averaged_hats_up_to_the_box_edge() {
    // At k = q the wave vectors are the setup's G. The setup averages every shape over the
    // aliases the grid's lattice operations pick (a skewed grid cuts the star at many indices);
    // `pair_shapes` takes the same aliases, each with its own origin phase (the hats share one),
    // which keeps K exactly translation invariant.
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

/// The shapes of `rig` at `k_diff` as the one-sided wrap read them: every index at its wrapped
/// frequency alone, a face of the box (`-N/2`) read at `-N/2` whichever way it was reached. The
/// control of the reciprocity test below.
fn one_sided_shapes(rig: &Rig, k_diff: [f64; 3]) -> Vec<Complex64> {
    let c = &rig.crystal;
    let (n, shape, dv) = (
        c.grid.index.points(),
        c.grid.index.shape(),
        c.grid.index.dv(),
    );
    let b = reciprocal::reciprocal_cell(&c.grid).unwrap();
    let kappa = b.try_inverse().unwrap() * Vector3::from(k_diff);
    let sites = rig.geometry.sites();
    let mut out = vec![Complex64::default(); c.channels * n];
    for at in 0..n {
        let index = [
            at / (shape[1] * shape[2]),
            at / shape[2] % shape[1],
            at % shape[2],
        ];
        let wrapped = Vector3::from_fn(|a, _| {
            let size = shape[a] as f64;
            let x = frequency(index[a], shape[a]) + kappa[a];
            x - size * ((x + 0.5 * size + 1e-9) / size).floor()
        });
        let q = b * wrapped;
        let g = q - Vector3::from(k_diff);
        let phase = Complex64::from_polar(
            1. / dv,
            g[0] * c.grid.origin[0] + g[1] * c.grid.origin[1] + g[2] * c.grid.origin[2],
        );
        let mut column = vec![Complex64::default(); c.channels];
        rig.shapes
            .transforms(&sites, &[q.into()], &mut column, 1)
            .unwrap();
        for (a, hat) in column.iter().enumerate() {
            out[a * n + at] = hat * phase;
        }
    }
    out
}

#[test]
fn the_shapes_of_a_pair_and_of_its_reverse_are_conjugates_on_the_faces_of_the_box_too() {
    // The pair (q, k) is the complex conjugate of the pair (k, q) at the negated frequency, and
    // the channel (L, M) of one is the channel (L, -M) of the other, with `(-1)^M`:
    // `T_{LM}(k_diff, v)^* = (-1)^M T_{L -M}(-k_diff, -v)`. On a face of the box the frequency
    // `N/2` is its own negative and a single wrapped vector cannot satisfy it: the exchange then
    // fails `tr(D1 K[D2]) = tr(D2 K[D1])` (1.98e-12 on the Si `lda-fr` spinor fixture). The shapes
    // take the mean of their aliases, both faces among them; the one-sided wrap is the control
    // that must break it.
    let mut faces_seen = 0usize;
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let (n, shape) = (c.grid.index.points(), c.grid.index.shape());
        let b = reciprocal::reciprocal_cell(&c.grid).unwrap();
        let partner = |a: usize| {
            let (atom, l, m) = rig.geometry.channels[a];
            rig.geometry
                .channels
                .iter()
                .position(|x| *x == (atom, l, -m))
                .unwrap()
        };
        let sign = |a: usize| {
            if rig.geometry.channels[a].2 % 2 == 0 {
                1.
            } else {
                -1.
            }
        };
        let negated = |at: usize| {
            let index = [
                at / (shape[1] * shape[2]),
                at / shape[2] % shape[1],
                at % shape[2],
            ];
            let minus: [usize; 3] = std::array::from_fn(|a| (shape[a] - index[a]) % shape[a]);
            (minus[0] * shape[1] + minus[1]) * shape[2] + minus[2]
        };
        let (mut worst, mut control, mut peak, mut faces) = (0f64, 0f64, 0f64, 0usize);
        for source in &c.points {
            for target in &c.points {
                let kd: [f64; 3] = std::array::from_fn(|a| target.k[a] - source.k[a]);
                let reverse = kd.map(|x| -x);
                let (t, tr) = (
                    aug.pair_shapes(c, kd).unwrap(),
                    aug.pair_shapes(c, reverse).unwrap(),
                );
                let (one, one_reverse) =
                    (one_sided_shapes(&rig, kd), one_sided_shapes(&rig, reverse));
                // The indices on a face: a frequency N/2 of an axis whose shift is a whole number.
                let kappa = b.try_inverse().unwrap() * Vector3::from(kd);
                let on_face = |at: usize| {
                    let index = [
                        at / (shape[1] * shape[2]),
                        at / shape[2] % shape[1],
                        at % shape[2],
                    ];
                    (0..3).any(|a| {
                        shape[a] % 2 == 0
                            && index[a] == shape[a] / 2
                            && (kappa[a] - kappa[a].round()).abs() < 1e-9
                    })
                };
                for at in 0..n {
                    for a in 0..c.channels {
                        let (x, y) = (
                            t.as_slice()[a * n + at].conj(),
                            tr.as_slice()[partner(a) * n + negated(at)] * sign(a),
                        );
                        let (u, v) = (
                            one[a * n + at].conj(),
                            one_reverse[partner(a) * n + negated(at)] * sign(a),
                        );
                        peak = peak.max(x.norm());
                        worst = worst.max((x - y).norm());
                        if on_face(at) {
                            control = control.max((u - v).norm());
                        } else {
                            // Off the faces the two conventions are one and the same.
                            assert!((u - v).norm() < 1e-12 * peak.max(1.), "{fixture}");
                        }
                    }
                }
                faces += (0..n).filter(|&at| on_face(at)).count();
            }
        }
        println!(
            "{fixture}: {faces} face entries; reciprocity of the pair shapes {:.2e} of the peak {peak:.3}; the one-sided wrap misses by {:.2e}",
            worst / peak,
            control / peak
        );
        assert!(worst < 1e-13 * peak, "{fixture}: {worst:e}");
        if faces > 0 {
            assert!(
                control > 1e-3 * peak,
                "{fixture}: the control does not bite ({control:e})"
            );
        }
        faces_seen += faces;
    }
    assert!(faces_seen > 0, "no fixture has a face of the box to test");
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
    seen: std::sync::Mutex<Vec<Matrix>>,
}
impl OneCenter for Toy {
    fn collinear(&self, _ctx: &Context, projected: &[Matrix]) -> Result<OneCenterTerms> {
        self.seen.lock().unwrap().extend(projected.iter().cloned());
        Ok(OneCenterTerms {
            energy: self.energy * projected.len() as f64,
            operators: projected
                .iter()
                .map(|p| Matrix::identity(p.nrows(), p.ncols()) * Complex64::new(self.shift, 0.))
                .collect(),
        })
    }
    fn spinor(&self, _ctx: &Context, projected: &Matrix) -> Result<OneCenterTerms> {
        self.seen.lock().unwrap().push(projected.clone());
        Ok(OneCenterTerms {
            energy: self.energy,
            operators: vec![
                Matrix::identity(projected.nrows(), projected.ncols())
                    * Complex64::new(self.shift, 0.),
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
            let seen = toy.seen.lock().unwrap()[spin].clone();
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
