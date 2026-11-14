mod berry_support;
mod setup_shells_support;

use berry_support::{lih, solve, solve_in, Solved};
use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::wannier::{
        model,
        sampling::{self, Augmentation, Frame, Trial},
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

#[test]
fn neighbor_shells_reproduce_the_metric_of_skew_meshes_and_wrap_indices() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    for cell in [
        Matrix3::identity() * 8.,
        Matrix3::new(0., 5., 5., 5., 0., 5., 5., 5., 0.),
        Matrix3::new(7., 1., 0.2, 0., 9., 1.1, 0.3, 0., 5.),
    ] {
        let size = [4, 3, 2];
        let shells = sampling::neighbor_shells(cell, size).unwrap();
        let mut metric = Matrix3::zeros();
        let b = 2. * PI * cell.try_inverse().unwrap().transpose();
        for ((v, w), step) in shells
            .vectors
            .iter()
            .zip(&shells.weights)
            .zip(&shells.steps)
        {
            let vector = Vector3::from(*v);
            metric += *w * vector * vector.transpose();
            assert!(
                (vector
                    - b * Vector3::from(std::array::from_fn::<_, 3, _>(
                        |a| step[a] as f64 / size[a] as f64
                    )))
                .norm()
                    < 1e-13
            );
        }
        assert!((metric - Matrix3::identity()).norm() < 1e-8);
        let neighbors = sampling::neighbor_indices(&ctx, size, &shells.steps).unwrap();
        for k in 0..24 {
            let origin = sampling::mesh_point(size, k).unwrap();
            for (d, step) in shells.steps.iter().enumerate() {
                let other =
                    sampling::mesh_point(size, neighbors.as_slice()[k * shells.steps.len() + d])
                        .unwrap();
                for a in 0..3 {
                    let delta = other[a] - origin[a] - step[a] as f64 / size[a] as f64;
                    assert!((delta - delta.round()).abs() < 1e-14);
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn archived_bloch_links_match_independent_ao_contractions_and_boundary_sewing() {
    let solved = solve(&lih("lda-sr"));
    let (ctx, c, state) = (&solved.ctx, &*solved.crystal, &*solved.state);
    {
        let augmentation = Augmentation::from_setup(&solved.setup, c).unwrap();
        let bands: Vec<usize> = (0..c.bands()).collect();
        let left = Frame::from_point(c, state, &c.points[0], 0, &bands).unwrap();
        let right = Frame::from_point(c, state, &c.points[1], 0, &bands).unwrap();
        let b: [f64; 3] = std::array::from_fn(|a| c.points[1].k[a] - c.points[0].k[a]);
        let native = sampling::link(c, &left, &right, b, &augmentation).unwrap();
        let n = c.grid.index.points();
        let first = c.points[0].psi.read(ctx).unwrap();
        let second = c.points[1].psi.read(ctx).unwrap();
        let mut ao = DMatrix::<C64>::zeros(c.m, c.m);
        for i in 0..c.m {
            for j in 0..c.m {
                ao[(i, j)] = (0..n)
                    .map(|p| {
                        let r = c.grid.point(p);
                        first[i * n + p].conj()
                            * second[j * n + p]
                            * C64::from_polar(
                                c.grid.index.dv(),
                                -(0..3).map(|a| r[a] * b[a]).sum::<f64>(),
                            )
                    })
                    .sum();
            }
        }
        ao += &c.points[0].projections
            * augmentation.operator(c, b).unwrap()
            * c.points[1].projections.adjoint();
        let expected = left.coefficients.adjoint() * ao * &right.coefficients;
        assert!((&native - expected).norm() < 1e-12);
        let zero = sampling::link(c, &left, &left, [0.; 3], &augmentation).unwrap();
        assert!((zero - DMatrix::identity(bands.len(), bands.len())).norm() < 1e-10);
        let reciprocal = 2.
            * PI
            * sampling::lattice(&c.grid)
                .try_inverse()
                .unwrap()
                .transpose();
        let sewn =
            sampling::link(c, &left, &left, reciprocal.column(0).into(), &augmentation).unwrap();
        assert!((sewn - DMatrix::identity(bands.len(), bands.len())).norm() > 0.1);
        // The frame is the solver's state: equal levels, and equal projectors of every group of
        // degenerate bands (a single band of a degenerate group is a gauge choice).
        let vectors = state.coefficients[0].matrix(ctx).unwrap();
        let levels = left.energies.as_slice();
        let mut lo = 0;
        while lo < bands.len() {
            let mut hi = lo + 1;
            while hi < bands.len() && (levels[hi] - levels[lo]).abs() < 1e-8 {
                hi += 1;
            }
            for i in lo..hi {
                assert!((levels[i] - state.eigenvalues[0][bands[i]]).abs() < 1e-10);
            }
            let frame = left.coefficients.columns(lo, hi - lo);
            let solver = DMatrix::from_fn(c.bands(), hi - lo, |r, j| vectors[(r, bands[lo + j])]);
            assert!(
                (&frame * frame.adjoint() - &solver * solver.adjoint()).norm() < 1e-9,
                "the projector of bands {lo}..{hi} differs from the solver's"
            );
            lo = hi;
        }
    }
}

/// `int [phi_i* phi_j - phi~_i* phi~_j] e^{-ib.r} d^3r` for every projector pair of `atom`, by a
/// product rule on the sphere and Simpson's rule on the dataset's radial grid (no Bessel
/// expansion), in the dual (partial-wave) projector basis.
fn quadrature_one_center(
    setup: &mandacaru_core::periodic::setup::PeriodicSetup,
    atom: usize,
    b: [f64; 3],
) -> Vec<((usize, usize), C64)> {
    use mandacaru_core::{
        math::spline::simpson,
        paw::{
            multipole::{angular_grid, directions, harmonic},
            multipole_pairs::{smooth_waves, ChannelWaves},
        },
    };
    let ds = &setup.datasets[setup.atoms_species[atom]];
    let r = ds.r.as_slice();
    let waves = |l: usize, index: usize| {
        let ch = &ds.channels[&(l as u32)];
        let ae: Vec<f64> = ch
            .ae_waves
            .iter()
            .flat_map(|w| w.as_slice().iter().copied())
            .collect();
        let coefficients: Vec<f64> = ch
            .wave_coefficients
            .iter()
            .flat_map(|a| a.data().iter().copied())
            .collect();
        let wavevectors: Vec<f64> = ch
            .wavevectors
            .iter()
            .flat_map(|a| a.data().iter().copied())
            .collect();
        let smooth = smooth_waves(
            r,
            &ChannelWaves {
                l,
                r_cut: ch.r_cut,
                ae: &ae,
                coefficients: &coefficients,
                wavevectors: &wavevectors,
                waves: ch.ae_waves.len(),
            },
        )
        .unwrap();
        let n = r.len();
        (
            ae[index * n..(index + 1) * n].to_vec(),
            smooth[index * n..(index + 1) * n].to_vec(),
            ch.r_cut,
        )
    };
    let (theta, phi, weight) = angular_grid(24, 48).unwrap();
    let dirs = directions(&theta, &phi);
    let mut out = vec![];
    for (i, pi) in setup
        .projectors
        .iter()
        .enumerate()
        .filter(|(_, p)| p.atom == atom)
    {
        for (j, pj) in setup
            .projectors
            .iter()
            .enumerate()
            .filter(|(_, p)| p.atom == atom)
        {
            let (ae_i, ps_i, cut_i) = waves(pi.l, pi.index);
            let (ae_j, ps_j, cut_j) = waves(pj.l, pj.index);
            // The production nodes: `r <= max(r_cut_i, r_cut_j)`, where the pair density ends.
            let inside = r.partition_point(|&x| x <= cut_i.max(cut_j));
            let r = &r[..inside];
            let mut value = C64::default();
            for (direction, w) in dirs.iter().zip(&weight) {
                let angular = harmonic(pi.l, pi.m, *direction, false).conj()
                    * harmonic(pj.l, pj.m, *direction, false);
                if angular.norm() < 1e-300 {
                    continue;
                }
                let projection = (0..3).map(|a| b[a] * direction[a]).sum::<f64>();
                let real: Vec<f64> = (0..r.len())
                    .map(|k| {
                        (ae_i[k] * ae_j[k] - ps_i[k] * ps_j[k])
                            * r[k]
                            * r[k]
                            * (projection * r[k]).cos()
                    })
                    .collect();
                let imaginary: Vec<f64> = (0..r.len())
                    .map(|k| {
                        -(ae_i[k] * ae_j[k] - ps_i[k] * ps_j[k])
                            * r[k]
                            * r[k]
                            * (projection * r[k]).sin()
                    })
                    .collect();
                let radial = C64::new(simpson(r, &real).unwrap(), simpson(r, &imaginary).unwrap());
                value += angular * radial * *w;
            }
            out.push(((i, j), value));
        }
    }
    out
}

#[test]
fn the_exact_one_center_plane_wave_matches_a_direct_quadrature_and_the_first_order_term_does_not() {
    // D1 of dev_plan/WANNIER_PLAN.md: the one-center part of <psi_k| e^{-ib.r} |psi_k+b> needs the
    // whole plane wave, not its first order, for the second moments of a spread.
    let solved = solve_in(&lih("lda-sr"), "dual");
    let c = &*solved.crystal;
    let augmentation = Augmentation::from_setup(&solved.setup, c).unwrap();
    let zero = augmentation.operator(c, [0.; 3]).unwrap();
    let at_zero: Vec<Vec<((usize, usize), C64)>> = (0..augmentation.centers.len())
        .map(|atom| quadrature_one_center(&solved.setup, atom, [0.; 3]))
        .collect();
    let mut worst_exact = 0f64;
    let mut worst_first = 0f64;
    for b in [[0.31, -0.12, 0.07], [0.05, 0.6, -0.2], [1e-3, 0., 0.]] {
        let exact = augmentation.operator(c, b).unwrap();
        // The first-order operator: monopole and dipole of the crystal's multipole blocks.
        let mut first = DMatrix::<C64>::zeros(c.p, c.p);
        for (channel, [atom, l, m]) in augmentation.labels.iter().enumerate() {
            let radial = match (*l, *m) {
                (0, 0) => C64::new((4. * PI).sqrt(), 0.),
                (1, 0) => C64::new(0., -b[2] * (4. * PI / 3.).sqrt()),
                (1, -1) => C64::new(-b[1], -b[0]) * (8. * PI / 3.).sqrt() / 2.,
                (1, 1) => C64::new(-b[1], b[0]) * (8. * PI / 3.).sqrt() / 2.,
                _ => C64::default(),
            };
            let phase = C64::from_polar(
                1.,
                -(0..3)
                    .map(|a| b[a] * augmentation.centers[*atom as usize][a])
                    .sum::<f64>(),
            );
            for i in 0..c.p {
                for j in 0..c.p {
                    first[(i, j)] +=
                        phase * radial * c.blocks.as_slice()[(channel * c.p + i) * c.p + j];
                }
            }
        }
        for atom in 0..augmentation.centers.len() {
            let phase = C64::from_polar(
                1.,
                -(0..3)
                    .map(|a| b[a] * augmentation.centers[atom][a])
                    .sum::<f64>(),
            );
            // The L = 0 diagonal block is the dataset's overlap correction, which the quadrature
            // of its own waves reproduces only to the dataset's accuracy: compare the b-dependent
            // part, the difference from b = 0.
            for (((i, j), value), (_, origin)) in quadrature_one_center(&solved.setup, atom, b)
                .into_iter()
                .zip(&at_zero[atom])
            {
                let expected = phase * (value - origin);
                let scale = 1f64.max(value.norm());
                worst_exact = worst_exact
                    .max(((exact[(i, j)] - phase * zero[(i, j)]) - expected).norm() / scale);
                worst_first = worst_first
                    .max(((first[(i, j)] - phase * zero[(i, j)]) - expected).norm() / scale);
            }
        }
    }
    assert!(
        worst_exact < 1e-8,
        "exact one-center operator off the quadrature by {worst_exact:e}"
    );
    assert!(
        worst_first > 1e-5,
        "the first-order operator is already exact ({worst_first:e}): the gate is vacuous"
    );
}

#[test]
fn the_exact_one_center_plane_wave_is_one_operator_in_the_raw_and_dual_projector_bases() {
    // The remainders carry the same Vanderbilt transform as the multipole blocks: between the AO
    // Bloch sums, `<chi_k|p> O(b) <p|chi_k'>` does not depend on the projector basis. b = 0 checks
    // the blocks alone (the monopole), the others the whole plane wave.
    let raw = solve_in(&lih("lda-sr"), "raw");
    let dual = solve_in(&lih("lda-sr"), "dual");
    let between = |s: &Solved, b: [f64; 3]| {
        let c = &*s.crystal;
        let augmentation = Augmentation::from_setup(&s.setup, c).unwrap();
        &c.points[0].projections
            * augmentation.operator(c, b).unwrap()
            * c.points[1].projections.adjoint()
    };
    // b = 0 is the round-off floor of the Vanderbilt transform of the existing blocks (1e-10 on
    // LiH); the remainders must sit at the same level, far below their own size.
    for b in [[0.; 3], [0.31, -0.12, 0.07], [0.05, 0.6, -0.2]] {
        let (x, y) = (between(&raw, b), between(&dual, b));
        let miss = (&x - &y).norm() / x.norm().max(1.);
        eprintln!("b = {b:?}: raw against dual, relative {miss:e}");
        assert!(
            miss < 1e-8,
            "b = {b:?}: raw and dual one-center matrices differ by {miss:e}"
        );
    }
}

#[test]
fn shells_are_exact_single_shells_without_parallel_steps() {
    // D2 of dev_plan/WANNIER_PLAN.md: a mesh whose shortest steps are along the axes takes one
    // shell per independent direction, weight 1 / (2 |b|^2) for a +-b pair, and no step parallel
    // to an accepted one (the minimum-norm solve used to give 80% of the x weight to 2b).
    for (cell, size) in [
        (Matrix3::identity() * 10., [8, 2, 2]),
        (Matrix3::from_diagonal(&Vector3::new(6., 8., 8.)), [3, 1, 1]),
    ] {
        let shells = sampling::neighbor_shells(cell, size).unwrap();
        let reciprocal = 2. * PI * cell.try_inverse().unwrap().transpose();
        for ((v, w), step) in shells
            .vectors
            .iter()
            .zip(&shells.weights)
            .zip(&shells.steps)
        {
            assert_eq!(
                step.iter().map(|s| s.abs()).sum::<i32>(),
                1,
                "a non-axis step {step:?}"
            );
            let axis = step.iter().position(|s| *s != 0).unwrap();
            let length = reciprocal.column(axis).norm() / size[axis] as f64;
            assert!((Vector3::from(*v).norm() - length).abs() < 1e-12);
            // The x pair is one shell; y and z share one (equal lengths): 1/(2b^2) per pair and
            // per Cartesian direction.
            assert!(
                (w - 1. / (2. * length * length)).abs() < 1e-9 / (length * length),
                "{w}"
            );
        }
        assert_eq!(shells.vectors.len(), 6);
    }
}

#[test]
fn gaussian_trial_radials_are_normalized_and_p_harmonics_point_along_their_axes() {
    for l in 0..5 {
        let trial = Trial {
            center: [0.; 3],
            angular: vec![(l, 0, 1.)],
            width: 0.7,
            spin: None,
            label: format!("l={l}"),
        };
        trial.validate().unwrap();
        let step = 0.0005;
        // At the polar axis |Y_l0|^2=(2l+1)/(4pi), recovering the radial
        // normalization independently of the angular implementation.
        let radial_norm = (0..8400)
            .map(|i| {
                let r = (i as f64 + 0.5) * step;
                trial.value([0., 0., r]).powi(2) * r * r * step * 4. * PI / (2 * l + 1) as f64
            })
            .sum::<f64>();
        assert!((radial_norm - 1.).abs() < 2e-10, "l={l}: {radial_norm}");
    }
    let px = Trial {
        center: [0.; 3],
        angular: vec![(1, 1, 1.)],
        width: 1.,
        spin: None,
        label: "px".into(),
    };
    assert!(px.value([1., 0., 0.]) > 0.);
    assert_eq!(px.value([0., 1., 0.]), 0.);
    assert_eq!(px.value([-1., 0., 0.]), -px.value([1., 0., 0.]));
}

#[test]
fn fourier_hoppings_recover_complex_bands_and_remain_hermitian_off_mesh() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    {
        let size = [4, 2, 2];
        let h = |k: [f64; 3]| {
            let phase = 2. * PI * k[0];
            let off = C64::new(1., 0.) + C64::new(0., 0.2) * C64::from_polar(1., -phase);
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    C64::new(2. * phase.cos(), 0.),
                    off,
                    off.conj(),
                    C64::new(-phase.cos(), 0.),
                ],
            )
        };
        let values: Vec<_> = (0..16)
            .flat_map(|k| {
                let matrix = h(sampling::mesh_point(size, k).unwrap());
                (0..4).map(move |a| matrix[(a / 2, a % 2)])
            })
            .collect();
        let blocks = model::fourier_blocks(&ctx, size, 2, &values).unwrap();
        let cell = Matrix3::new(6., 0.4, 0., 0., 7., 0.5, 0.1, 0., 8.);
        let hopping = model::nearest_hoppings(
            &ctx,
            cell,
            size,
            &[[0.; 3], [0.2, 0., 0.]],
            blocks.as_slice(),
        )
        .unwrap();
        let kpoints: Vec<_> = (0..16)
            .map(|k| sampling::mesh_point(size, k).unwrap())
            .chain([[0.123, 0.293, 0.017], [0.481, 0.1, 0.32]])
            .collect();
        let interpolated = hopping.interpolate(&ctx, &kpoints).unwrap();
        for (k, point) in kpoints.iter().enumerate() {
            let (e, _) = mandacaru_core::math::linalg::eigen(h(*point)).unwrap();
            for i in 0..2 {
                assert!((interpolated.as_slice()[2 * k + i] - e[i]).abs() < 1e-12);
            }
        }
        assert!(model::polar(&DMatrix::from_element(3, 2, C64::new(1., 0.))).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
