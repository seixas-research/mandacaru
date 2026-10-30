use mandacaru_core::math::spline::simpson;
use mandacaru_core::paw::checks::interpolate_linear;
use mandacaru_core::paw::multipole::shape_function;
use mandacaru_core::paw::onecenter::{
    one_center_tensor, Basis, Channel, Couplings, Projector, Sphere,
};
use mandacaru_core::paw::onecenter_hybrid::{
    exact_atom, exact_channel_derivatives, exact_channels, expand_blocks, ChannelData,
    ChannelStack, Factors, Hybrid, HybridAtom,
};
use mandacaru_core::paw::onecenter_radial::{
    legendre, long_range_radial_kernel, long_range_shape_potential, radial_coulomb,
    three_j_squared, MAX_THREE_J,
};
use mandacaru_core::paw::onecenter_terms::{
    core_extent, core_orbital_tables, exact_one_center_terms, frozen_exchange_terms, CoreOrbital,
    Occupied, ReferenceAtom,
};
use mandacaru_core::Context;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn the_slater_integral_of_a_uniform_sphere_is_exact_for_even_and_odd_grids() {
    // int int r_>^-1 r^2 r'^2 over a unit-density sphere is 2 R^5 / 15.
    let radius: f64 = 2.;
    for (n, tolerance) in [(4000, 1e-6), (4001, 1e-6), (600, 1e-5), (601, 1e-5)] {
        let r = linspace(0., radius, n);
        let rho = vec![1.; n];
        let value = radial_coulomb(&r, &rho, &rho, 0).unwrap();
        assert!(
            (value / (2. * radius.powi(5) / 15.) - 1.).abs() < tolerance,
            "n = {n}: {value}"
        );
    }
    // The two electrons commute.
    let r = linspace(0., 6., 6001);
    let a = r.iter().map(|r| (-r * r).exp()).collect::<Vec<_>>();
    let b = r.iter().map(|r| r * (-0.5 * r).exp()).collect::<Vec<_>>();
    for l in 0..3 {
        let (x, y) = (
            radial_coulomb(&r, &a, &b, l).unwrap(),
            radial_coulomb(&r, &b, &a, l).unwrap(),
        );
        assert!((x - y).abs() < 1e-6 * x.abs(), "L = {l}");
    }
    assert!(radial_coulomb(&r, &a, &b[..10], 0).is_err());
}

#[test]
fn the_slater_integral_refuses_fewer_than_two_points() {
    // An empty grid used to index `above[n - 1]` and panic; one point has no interval.
    for n in [0usize, 1] {
        let (r, rho) = (vec![0.5; n], vec![1.; n]);
        match radial_coulomb(&r, &rho, &rho, 0) {
            Err(mandacaru_core::Error::Invalid(message)) => {
                assert!(message.contains("at least two grid points"), "{message}");
            }
            other => panic!("n = {n}: expected Error::Invalid, got {other:?}"),
        }
    }
    // Two points is the smallest supported grid (a trapezoid).
    let value = radial_coulomb(&[0., 1.], &[1., 1.], &[1., 1.], 0).unwrap();
    assert!(value.is_finite());
}

#[test]
fn the_squared_three_j_symbols() {
    let t = |a, b, c| three_j_squared(a, b, c).unwrap();
    assert!((t(1, 1, 0) - 1. / 3.).abs() < 1e-15);
    assert!((t(1, 1, 2) - 2. / 15.).abs() < 1e-15);
    assert!((t(2, 2, 2) - 2. / 35.).abs() < 1e-15);
    assert_eq!(t(1, 1, 1), 0.);
    assert_eq!(t(0, 0, 2), 0.);
    // sum_c (2c + 1) (a b c; 0 0 0)^2 = 1.
    for (a, b) in [(0usize, 2usize), (1, 1), (2, 3), (3, 3)] {
        let total: f64 = (a.abs_diff(b)..=a + b)
            .map(|c| (2 * c + 1) as f64 * t(a, b, c))
            .sum();
        assert!((total - 1.).abs() < 1e-13, "({a}, {b}): {total}");
    }
}

#[test]
fn the_squared_three_j_symbols_hold_over_the_stated_range() {
    // Exact rational values (fractions of factorials, evaluated in Python), rounded to f64.
    // The factorial form returned 0 at (60, 60, 60) and NaN at (100, 100, 100).
    let t = |a, b, c| three_j_squared(a, b, c).unwrap();
    for (abc, exact) in [
        ((40, 40, 40), 0.00022405672539895654),
        ((60, 60, 60), 0.00010041201423327862),
        ((100, 100, 100), 3.638974495877624e-05),
        (
            (MAX_THREE_J, MAX_THREE_J, MAX_THREE_J),
            2.2259126946201172e-05,
        ),
        ((MAX_THREE_J, MAX_THREE_J, 2), 0.0009728068322712406),
        ((MAX_THREE_J, MAX_THREE_J, 0), 0.0038910505836575876),
        ((64, 100, MAX_THREE_J), 4.9859849768677637e-05),
        ((7, 5, 12), 0.01279310808991789),
    ] {
        let value = t(abc.0, abc.1, abc.2);
        assert!(
            ((value - exact) / exact).abs() < 1e-12,
            "{abc:?}: {value} against {exact}"
        );
    }
    // The orthogonality sum still closes at large angular momenta.
    let (a, b) = (50usize, 70usize);
    let total: f64 = (a.abs_diff(b)..=a + b)
        .map(|c| (2 * c + 1) as f64 * t(a, b, c))
        .sum();
    assert!((total - 1.).abs() < 1e-11, "{total}");
    // Zero (not an error) for an odd sum or a broken triangle inside the range.
    assert_eq!(t(MAX_THREE_J, MAX_THREE_J, 1), 0.);
    assert_eq!(t(MAX_THREE_J, 3, 5), 0.);
    // One past the range is refused, whatever the other two are.
    for abc in [
        (MAX_THREE_J + 1, 1, MAX_THREE_J + 2),
        (0, MAX_THREE_J + 1, MAX_THREE_J + 1),
        (MAX_THREE_J + 1, MAX_THREE_J + 1, MAX_THREE_J + 1),
        (1, 1, 100_000),
    ] {
        assert!(
            matches!(
                three_j_squared(abc.0, abc.1, abc.2),
                Err(mandacaru_core::Error::Invalid(_))
            ),
            "{abc:?}"
        );
    }
}

#[test]
fn legendre_polynomials() {
    for x in [-0.7, 0.1, 0.9] {
        assert!((legendre(2, x) - 0.5 * (3. * x * x - 1.)).abs() < 1e-15);
        assert!((legendre(3, x) - 0.5 * (5. * x * x * x - 3. * x)).abs() < 1e-15);
    }
}

#[test]
fn the_screened_kernel_has_the_series_and_the_coulomb_limits() {
    let ctx = context();
    // Small omega: 2w/sqrt(pi) - (2w^3 / 3 sqrt(pi)) (r^2 + r'^2) at L = 0, and the
    // dipole term (4 w^3 / 3 sqrt(pi)) r r' at L = 1, up to w^5 R^4 (< 3e-6 here).
    let omega = 0.05;
    let r = linspace(0., 1.5, 7);
    let f = long_range_radial_kernel(&ctx, &r, &r, &[0, 1, 2], omega, None).unwrap();
    let c = 2. * omega.powi(3) / (3. * PI.sqrt());
    for (i, a) in r.iter().enumerate() {
        for (j, b) in r.iter().enumerate() {
            let k = i * r.len() + j;
            assert!((f[0][k] - (2. * omega / PI.sqrt() - c * (a * a + b * b))).abs() < 3e-6);
            assert!((f[1][k] - 2. * c * a * b).abs() < 3e-6);
            assert!(f[2][k].abs() < 3e-6);
        }
    }
    // Large omega, away from r = r': r_<^L / r_>^(L+1).
    let (r1, r2) = ([0.5, 1.], [2., 3.]);
    let f = long_range_radial_kernel(&ctx, &r1, &r2, &[0, 1, 2], 20., Some(64)).unwrap();
    for (l, level) in f.iter().enumerate() {
        for (i, a) in r1.iter().enumerate() {
            for (j, b) in r2.iter().enumerate() {
                let expected = a.powi(l as i32) / b.powi(l as i32 + 1);
                assert!((level[i * 2 + j] - expected).abs() < 1e-6 * expected);
            }
        }
    }
    // The rule grows with omega r: 100 Bohr^-1 still integrates to the Coulomb kernel.
    let wide = long_range_radial_kernel(&ctx, &[1.], &[1.4], &[0], 100., None).unwrap();
    assert!((wide[0][0] * 1.4 - 1.).abs() < 1e-3);
    // Beyond the 4096 nodes of the Gauss-Legendre rule the request is refused, not degraded.
    assert!(long_range_radial_kernel(&ctx, &[1.], &[1.4], &[0], 1000., None).is_err());
    // omega = 0 and no levels give zeros.
    assert!(
        long_range_radial_kernel(&ctx, &r, &r, &[0], 0., None).unwrap()[0]
            .iter()
            .all(|v| *v == 0.)
    );
    assert!(long_range_radial_kernel(&ctx, &r, &r, &[0], -1., None).is_err());
}

#[test]
fn the_screened_shape_potential_tends_to_its_monopole() {
    let ctx = context();
    let far = long_range_shape_potential(&ctx, &[20., 30.], 1.45, 0, 0.11).unwrap();
    for (v, r) in far.iter().zip([20., 30.]) {
        let expected = 4. * PI * libm::erf(0.11 * r) / r;
        assert!((v - expected).abs() < 1e-4 * expected);
    }
    let omega = 1e-4;
    let near = long_range_shape_potential(&ctx, &[0., 1.], 1.45, 0, omega).unwrap();
    for v in near {
        assert!((v - 8. * PI.sqrt() * omega).abs() < 1e-6 * 8. * PI.sqrt() * omega);
    }
    assert!(long_range_shape_potential(&ctx, &[1.], 1.45, 1, 0.).unwrap()[0] == 0.);
    assert!(long_range_shape_potential(&ctx, &[1.], -1., 0, 0.1).is_err());
}

/// A one-channel `s` sphere: all-electron wave `exp(-r)`, smooth wave given.
struct Synthetic {
    r: Vec<f64>,
    ae: Vec<f64>,
    ps: Vec<f64>,
    vanderbilt: Vec<f64>,
}

impl Synthetic {
    fn new(smooth: impl Fn(f64) -> f64) -> Self {
        let r = (1..=3000).map(|i| i as f64 * 0.005).collect::<Vec<_>>();
        Self {
            ae: r.iter().map(|r| (-r).exp()).collect(),
            ps: r.iter().map(|r| smooth(*r)).collect(),
            vanderbilt: vec![1.],
            r,
        }
    }
    fn sphere(&self) -> Sphere<'_> {
        Sphere {
            r: &self.r,
            channels: vec![Channel {
                l: 0,
                waves: 1,
                ae: &self.ae,
                ps: &self.ps,
                vanderbilt: &self.vanderbilt,
                r_cut: 1.5,
            }],
            compensation_radius: 1.2,
        }
    }
}

const S: Projector = Projector {
    l: 0,
    m: 0,
    index: 0,
    real: true,
};

#[test]
fn the_tensor_is_the_slater_difference_and_vanishes_for_identical_waves() {
    let ctx = context();
    let identical = Synthetic::new(|r| (-r).exp());
    let tensor = one_center_tensor(&ctx, &identical.sphere(), Basis::Raw, &[S], 0.).unwrap();
    assert_eq!(tensor.data.len(), 1);
    assert!(tensor.data[0].norm() < 1e-12);

    let smooth = Synthetic::new(|r| 0.5 * (-0.3 * r * r).exp());
    let tensor = one_center_tensor(&ctx, &smooth.sphere(), Basis::Raw, &[S], 0.).unwrap();
    // The same number from the radial integrals alone.
    let r = linspace(0., 1.5, 600);
    let ae = interpolate_linear(&r, &smooth.r, &smooth.ae).unwrap();
    let ps = interpolate_linear(&r, &smooth.r, &smooth.ps).unwrap();
    let density = ae.iter().map(|a| a * a).collect::<Vec<_>>();
    let pair = ps.iter().map(|p| p * p).collect::<Vec<_>>();
    let moment = simpson(
        &r,
        &(0..600)
            .map(|i| (density[i] - pair[i]) * r[i] * r[i])
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let shape = shape_function(&r, 1.2, 0).unwrap();
    let compensated = (0..600)
        .map(|i| pair[i] + moment * shape[i])
        .collect::<Vec<_>>();
    let expected = radial_coulomb(&r, &density, &density, 0).unwrap()
        - radial_coulomb(&r, &compensated, &compensated, 0).unwrap();
    assert!(expected.abs() > 1e-3);
    assert!((tensor.data[0].re - expected).abs() < 1e-12 * expected.abs());
    assert!(tensor.real && tensor.data[0].im == 0.);

    // A complex harmonic gives the same monopole; the dual basis the same for B = 1.
    let complex = Projector { real: false, ..S };
    let other = one_center_tensor(&ctx, &smooth.sphere(), Basis::Dual, &[complex], 0.).unwrap();
    assert!(!other.real && (other.data[0].re - expected).abs() < 1e-12);
}

#[test]
fn a_screened_kernel_leaves_less_than_the_bare_one_and_the_exchange_symmetry_holds() {
    let ctx = context();
    let smooth = Synthetic::new(|r| 0.5 * (-0.3 * r * r).exp());
    let projectors = [S, Projector { l: 1, m: 0, ..S }];
    // A p channel is needed for two projectors.
    let p_ae = smooth.ae.clone();
    let p_ps = smooth.ps.iter().map(|v| 0.9 * v).collect::<Vec<_>>();
    let ae = [smooth.ae.clone(), p_ae].concat();
    let ps = [smooth.ps.clone(), p_ps].concat();
    let sphere = Sphere {
        r: &smooth.r,
        channels: [0, 1]
            .map(|l| Channel {
                l,
                waves: 1,
                ae: &ae[l * 3000..(l + 1) * 3000],
                ps: &ps[l * 3000..(l + 1) * 3000],
                vanderbilt: &smooth.vanderbilt,
                r_cut: 1.5,
            })
            .to_vec(),
        compensation_radius: 1.2,
    };
    let bare = one_center_tensor(&ctx, &sphere, Basis::Raw, &projectors, 0.).unwrap();
    let screened = one_center_tensor(&ctx, &sphere, Basis::Raw, &projectors, 20.).unwrap();
    assert!(screened.data[0].re.abs() < 0.2 * bare.data[0].re.abs());
    let symmetric = bare.exchange_symmetrized();
    let n = 2;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let (x, y) = (
                        symmetric.data[((a * n + b) * n + c) * n + d],
                        symmetric.data[((c * n + d) * n + a) * n + b],
                    );
                    assert!((x - y).norm() < 1e-15);
                }
            }
        }
    }
    assert!(one_center_tensor(&ctx, &sphere, Basis::Raw, &projectors, -1.).is_err());

    // The pair and radial caches are accounted: two projectors, three levels,
    // 600 radial points reserve 2^2 x 3 x (2 x 600 x 8 + 160) + 2^4 x 3 x 160 bytes.
    let fresh = context();
    one_center_tensor(&fresh, &sphere, Basis::Raw, &projectors, 0.).unwrap();
    assert!(
        fresh.memory_peak() >= 124_800 + 256,
        "{}",
        fresh.memory_peak()
    );
    assert_eq!(fresh.memory_used(), 0);
    let tight = Context::new(Some(1), 100_000).unwrap();
    assert!(one_center_tensor(&tight, &sphere, Basis::Raw, &projectors, 0.).is_err());
    let missing = Projector { l: 2, ..S };
    assert!(one_center_tensor(&ctx, &sphere, Basis::Raw, &[missing], 0.).is_err());
    assert!(Basis::parse("other").is_err());
}

#[test]
fn angular_couplings_follow_the_addition_theorem() {
    let mut couplings = Couplings::new().unwrap();
    let s = Projector { l: 0, m: 0, ..S };
    let p = Projector { l: 1, m: 0, ..S };
    assert!((couplings.angular(0, &s, &s, &s, &s).re - 1.).abs() < 1e-12);
    assert!(couplings.angular(1, &s, &s, &s, &s).norm() < 1e-12);
    for real in [true, false] {
        let (s, p) = (Projector { real, ..s }, Projector { real, ..p });
        assert!((couplings.angular(1, &s, &p, &s, &p).re - 1. / 3.).abs() < 1e-12);
        assert!(couplings.angular(0, &s, &p, &s, &p).norm() < 1e-12);
    }
}

#[test]
fn the_core_extent_and_the_core_orbital_tables() {
    // The frozen Python outputs for these two synthetic orbitals.
    let r = linspace(0.005, 30., 6000);
    let orbitals = vec![
        CoreOrbital {
            l: 0,
            occupation: 2.,
            radial: r.iter().map(|r| 2. * (-3. * r).exp()).collect(),
        },
        CoreOrbital {
            l: 0,
            occupation: 2.,
            radial: r
                .iter()
                .map(|r| 1.2 * (-0.9 * r).exp() * (1. - r / 3.))
                .collect(),
        },
    ];
    assert_eq!(core_extent(&r, &orbitals, 1e-14), 4863);
    assert_eq!(core_extent(&r, &orbitals, 1e-6), 2580);
    assert_eq!(core_extent(&r, &[], 1e-14), 0);

    // u / r, the density it makes, and the electrons by which it differs from a stored core.
    let u = r.iter().map(|r| r * (-r).exp()).collect::<Vec<_>>();
    let stored = r
        .iter()
        .map(|r| 2. * (-2. * r).exp() / (4. * PI))
        .collect::<Vec<_>>();
    let (radial, error) =
        core_orbital_tables(&r, Some(&r), &[(0, 2., u.clone())], &stored).unwrap();
    assert!((radial[0][100] - (-r[100]).exp()).abs() < 1e-14);
    assert!(error < 1e-12);
    // One electron in place of two leaves int r^2 exp(-2r) dr = 1/4 electron.
    let (_none, error) = core_orbital_tables(&r, None, &[(0, 1., u)], &stored).unwrap();
    assert!((error - 0.25).abs() < 1e-3, "{error}");
    assert_eq!(core_orbital_tables(&r, None, &[], &stored).unwrap().1, 0.);
    // A different atom grid is interpolated.
    let coarse = linspace(0.005, 30., 3000);
    let u = coarse.iter().map(|r| r * (-r).exp()).collect::<Vec<_>>();
    let (radial, _) = core_orbital_tables(&r, Some(&coarse), &[(1, 2., u)], &stored).unwrap();
    assert!((radial[0][2000] - (-r[2000]).exp()).abs() < 1e-5);
}

fn occupied<'a>(s: &'a Synthetic, occupations: &'a [f64]) -> Occupied<'a> {
    Occupied {
        channel: s.sphere().channels[0],
        occupations,
        projector_radius: 1.5,
    }
}

#[test]
fn identical_smooth_and_all_electron_waves_freeze_nothing() {
    let ctx = context();
    let s = Synthetic::new(|r| (-r).exp());
    let occupations = [1.];
    let channels = [occupied(&s, &occupations)];
    let (constant, blocks) =
        frozen_exchange_terms(&ctx, &s.r, &channels, 0.11, Basis::Raw).unwrap();
    assert!(constant.abs() < 1e-12);
    assert!(blocks[0].matrix.iter().all(|v| v.abs() < 1e-12));

    // The exact terms: V vanishes when only the core differs, and K is the Slater
    // integral of the core orbital against the wave.
    let density = s.ae.iter().map(|w| w * w / (4. * PI)).collect::<Vec<_>>();
    let core_density =
        s.r.iter()
            .map(|r| 2. * (-4. * r).exp() / (4. * PI))
            .collect::<Vec<_>>();
    let core = [CoreOrbital {
        l: 0,
        occupation: 2.,
        radial: s.r.iter().map(|r| (-2. * r).exp()).collect(),
    }];
    let atom = ReferenceAtom {
        r: &s.r,
        channels: vec![occupied(&s, &occupations)],
        core_density: &core_density,
        valence_density: &density,
        smooth_core: &core_density,
        functional: "lda",
        relativistic: false,
        core: &core,
    };
    let terms = exact_one_center_terms(&ctx, &atom, Basis::Raw).unwrap();
    let (l, v, k, d0) = &terms[0];
    assert_eq!(*l, 0);
    assert!(v[(0, 0)].abs() < 1e-12);
    assert!((d0[(0, 0)] - 1.).abs() < 1e-15);
    let extent = core_extent(&s.r, &core, 1e-14);
    let left = (0..extent)
        .map(|i| s.ae[i] * core[0].radial[i])
        .collect::<Vec<_>>();
    // f_c = 2 / (2 (2 l_c + 1)) = 1 per spin, (0 0 0; 0 0 0)^2 = 1.
    let expected = -radial_coulomb(&s.r[..extent], &left, &left, 0).unwrap();
    assert!((k[(0, 0)] - expected).abs() < 1e-12 * expected.abs());
    let pbe = ReferenceAtom {
        functional: "pbe",
        ..atom
    };
    assert!(exact_one_center_terms(&ctx, &pbe, Basis::Raw).unwrap()[0].1[(0, 0)].abs() < 1e-9);
    let bad = ReferenceAtom {
        functional: "lda",
        core_density: &core_density[..10],
        ..pbe
    };
    assert!(exact_one_center_terms(&ctx, &bad, Basis::Raw).is_err());
}

/// A random real tensor with the 8-fold symmetry of a real basis.
fn symmetric_tensor(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let mut raw = vec![0.; n * n * n * n];
    for v in raw.iter_mut() {
        *v = next();
    }
    let at = |a: usize, b: usize, c: usize, d: usize| raw[((a * n + b) * n + c) * n + d];
    let mut out = vec![0.; n * n * n * n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    out[((a * n + b) * n + c) * n + d] = [
                        at(a, b, c, d),
                        at(b, a, c, d),
                        at(a, b, d, c),
                        at(b, a, d, c),
                        at(c, d, a, b),
                        at(d, c, a, b),
                        at(c, d, b, a),
                        at(d, c, b, a),
                    ]
                    .iter()
                    .sum::<f64>()
                        / 8.;
                }
            }
        }
    }
    out
}

fn random_hermitian(p: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let raw = (0..p * p)
        .map(|_| Complex64::new(next(), next()))
        .collect::<Vec<_>>();
    (0..p * p)
        .map(|k| 0.5 * (raw[k] + raw[(k % p) * p + k / p].conj()))
        .collect()
}

fn hybrid(real: bool) -> Hybrid {
    let tensor = |seed| {
        symmetric_tensor(3, seed)
            .into_iter()
            .map(|v| Complex64::new(v, 0.))
            .collect::<Vec<_>>()
    };
    let _ = real;
    let layout = [(0, 0, 0), (1, 0, 0), (1, 1, 0)];
    let block0 = DMatrix::from_row_slice(1, 1, &[0.3]);
    let block1 = DMatrix::from_row_slice(1, 1, &[-0.2]);
    let v = expand_blocks(&layout, &[(0, &block0), (1, &block1)]).unwrap();
    let atoms = vec![
        HybridAtom {
            members: vec![0, 2, 3],
            tensor: tensor(1),
            v: v.clone(),
        },
        HybridAtom {
            members: vec![1, 4, 5],
            tensor: tensor(2),
            v,
        },
    ];
    Hybrid::new(atoms, 6, 0.25, 0.7).unwrap()
}

#[test]
fn the_hybrid_operator_is_the_derivative_of_its_energy() {
    let ctx = context();
    let h = hybrid(true);
    assert!((h.constant() + 0.25 * 0.7).abs() < 1e-15);
    let p = 6;
    let step = 1e-5;
    for spins in [1, 2] {
        let densities = (0..spins)
            .map(|s| random_hermitian(p, 10 + s as u64))
            .collect::<Vec<_>>();
        let direction = (0..spins)
            .map(|s| random_hermitian(p, 20 + s as u64))
            .collect::<Vec<_>>();
        let shifted = |sign: f64| {
            densities
                .iter()
                .zip(&direction)
                .map(|(d, x)| {
                    d.iter()
                        .zip(x)
                        .map(|(a, b)| a + b * (sign * step))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let energy = |sign: f64| {
            let moved = shifted(sign);
            let refs = moved.iter().map(|d| d.as_slice()).collect::<Vec<_>>();
            h.evaluate(&ctx, &refs).unwrap().energy
        };
        let refs = densities.iter().map(|d| d.as_slice()).collect::<Vec<_>>();
        let terms = h.evaluate(&ctx, &refs).unwrap();
        let derivative = (energy(1.) - energy(-1.)) / (2. * step);
        // dE = sum Re (O^T . dD).
        let analytic: f64 = terms
            .operators
            .iter()
            .zip(&direction)
            .map(|(o, x)| {
                (0..p * p)
                    .map(|k| (o.as_slice()[(k % p) * p + k / p] * x[k]).re)
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (derivative - analytic).abs() < 1e-8 * analytic.abs().max(1e-3),
            "{spins} spin(s): {derivative} against {analytic}"
        );
        assert!((terms.energy - terms.exact_exchange - terms.semilocal).abs() < 1e-14);
    }
    // The spinor form with vanishing off-diagonal blocks is the spin-channel form.
    let (up, down) = (random_hermitian(p, 31), random_hermitian(p, 32));
    let zero = vec![Complex64::new(0., 0.); p * p];
    let spinor = h
        .evaluate_spinor(&ctx, [[&up, &zero], [&zero, &down]])
        .unwrap();
    let channels = h.evaluate(&ctx, &[&up, &down]).unwrap();
    assert!((spinor.energy - channels.energy).abs() < 1e-13);
    for (s, offset) in [(0, 0), (1, p)] {
        for i in 0..p {
            for j in 0..p {
                let a = spinor.operators[0].as_slice()[(offset + i) * 2 * p + offset + j];
                let b = channels.operators[s].as_slice()[i * p + j];
                assert!((a - b).norm() < 1e-13);
            }
        }
    }
    assert!(h.evaluate(&ctx, &[]).is_err());
    assert!(h.evaluate(&ctx, &[&up[..5]]).is_err());
}

#[test]
fn no_fraction_no_hybrid() {
    let ctx = context();
    let layout = [(0, 0, 0)];
    let block = DMatrix::from_row_slice(1, 1, &[1.]);
    let v = expand_blocks(&layout, &[(0, &block)]).unwrap();
    let atom = HybridAtom {
        members: vec![0],
        tensor: vec![Complex64::new(1., 0.)],
        v,
    };
    let h = Hybrid::new(vec![atom], 1, 0., 3.).unwrap();
    let d = vec![Complex64::new(0.7, 0.)];
    let terms = h.evaluate(&ctx, &[&d]).unwrap();
    assert_eq!(terms.energy, 0.);
    assert!(terms.operators[0].as_slice()[0].norm() == 0.);
    assert!(expand_blocks(&[(2, 0, 0)], &[(0, &block)]).is_err());
}

#[test]
fn the_exact_factors_rebuild_the_tensor_and_the_channels_are_the_derivative() {
    let n = 3;
    let tensor = symmetric_tensor(n, 7);
    let layout = [(0, 0, 0), (1, 0, 0), (1, 1, 0)];
    let ones = DMatrix::from_row_slice(1, 1, &[1.]);
    let blocks = [(0, &ones), (1, &ones)];
    let zero = DMatrix::zeros(n, n);
    let d0 = expand_blocks(&layout, &blocks).unwrap();
    let atom = exact_atom(n, &tensor, zero.clone(), zero.clone(), d0.clone()).unwrap();
    // sum_k lambda_k U^k (x) U^k is the tensor.
    for ab in 0..n * n {
        for cd in 0..n * n {
            let rebuilt: f64 = atom
                .values
                .iter()
                .zip(&atom.factors)
                .map(|(lambda, u)| lambda * u[(ab / n, ab % n)] * u[(cd / n, cd % n)])
                .sum();
            assert!((rebuilt - tensor[ab * n * n + cd]).abs() < 1e-12);
        }
    }
    // coupling = -V - sym(W . D0) + K.
    for i in 0..n {
        for j in 0..n {
            let h = |i: usize, j: usize| -> f64 {
                (0..n * n)
                    .map(|kl| tensor[(i * n + j) * n * n + kl] * d0[(kl / n, kl % n)])
                    .sum()
            };
            assert!((atom.coupling[(i, j)] + 0.5 * (h(i, j) + h(j, i))).abs() < 1e-13);
        }
    }
    assert!(exact_atom(n, &tensor[..10], zero.clone(), zero.clone(), d0).is_err());

    // Channels of two spheres, and their derivative along dC.
    let (rows, width) = (4, 6);
    let c = random_hermitian(6, 5)[..rows * width].to_vec();
    let dc = random_hermitian(6, 6)[..rows * width].to_vec();
    let members = [vec![0, 2, 3], vec![1, 4, 5]];
    let atoms = members
        .iter()
        .map(|m| Factors {
            members: m,
            factors: &atom.factors,
        })
        .collect::<Vec<_>>();
    let ctx = context();
    let stack = exact_channels(&ctx, &c, rows, width, &atoms, false).unwrap();
    assert_eq!(stack.count, 2 * atom.factors.len());
    let g = complex_stack(&stack);
    let derivative = complex_stack(
        &exact_channel_derivatives(&ctx, &c, &dc, rows, width, &atoms, false).unwrap(),
    )
    .to_vec();
    let step = 1e-6;
    let moved = |sign: f64| {
        let shifted = c
            .iter()
            .zip(&dc)
            .map(|(a, b)| a + b * (sign * step))
            .collect::<Vec<_>>();
        complex_stack(&exact_channels(&ctx, &shifted, rows, width, &atoms, false).unwrap()).to_vec()
    };
    let (plus, minus) = (moved(1.), moved(-1.));
    for e in 0..plus.len() {
        let numeric = (plus[e] - minus[e]) / (2. * step);
        assert!(
            (numeric - derivative[e]).norm() < 1e-7,
            "factor {}",
            e / (rows * rows)
        );
    }
    // The real stack is the real part of the complex one.
    let real = exact_channels(&ctx, &c, rows, width, &atoms, true).unwrap();
    match &real.data {
        ChannelData::Real(b) => {
            assert!(b.as_slice().iter().zip(g).all(|(r, z)| *r == z.re));
        }
        ChannelData::Complex(_) => panic!("a real stack was asked for"),
    }
    drop(real);
    drop(stack);
    assert_eq!(ctx.memory_used(), 0);
    // A zero dC gives zeros.
    let still = vec![Complex64::new(0., 0.); rows * width];
    let zeros = exact_channel_derivatives(&ctx, &c, &still, rows, width, &atoms, false).unwrap();
    assert!(complex_stack(&zeros).iter().all(|v| v.norm() == 0.));
    // Malformed shapes are refused, not indexed.
    assert!(exact_channels(&ctx, &c[1..], rows, width, &atoms, false).is_err());
    assert!(exact_channel_derivatives(&ctx, &c, &still[1..], rows, width, &atoms, false).is_err());
    let outside = [Factors {
        members: &[0, 2, 6],
        factors: &atom.factors,
    }];
    assert!(exact_channels(&ctx, &c, rows, width, &outside, false).is_err());
}

fn complex_stack(channels: &ChannelStack) -> &[Complex64] {
    match &channels.data {
        ChannelData::Complex(b) => b.as_slice(),
        ChannelData::Real(_) => panic!("a complex stack was asked for"),
    }
}

/// Bytes the hybrid evaluation reserves: the operators and the exchange scratch.
fn hybrid_bytes(spins: usize, p: usize, n_max: usize) -> usize {
    (spins * p * p + 2 * n_max * n_max) * 16
}

#[test]
fn the_hybrid_evaluation_reserves_what_it_allocates() {
    let h = hybrid(true);
    let p = 6;
    let (up, down) = (random_hermitian(p, 41), random_hermitian(p, 42));
    let zero = vec![Complex64::new(0., 0.); p * p];
    // Each sphere has three members.
    for spins in [1usize, 2] {
        let ctx = context();
        let densities = [up.as_slice(), down.as_slice()];
        let terms = h.evaluate(&ctx, &densities[..spins]).unwrap();
        // Operators stay charged while the result lives; the scratch is gone.
        assert_eq!(ctx.memory_used(), spins * p * p * 16);
        assert_eq!(ctx.memory_peak(), hybrid_bytes(spins, p, 3));
        drop(terms);
        assert_eq!(ctx.memory_used(), 0);
    }
    let ctx = context();
    let terms = h
        .evaluate_spinor(&ctx, [[&up, &zero], [&zero, &down]])
        .unwrap();
    assert_eq!(ctx.memory_used(), 4 * p * p * 16);
    assert_eq!(ctx.memory_peak(), hybrid_bytes(4, p, 3));
    drop(terms);
    assert_eq!(ctx.memory_used(), 0);

    // A budget below the operators refuses with the memory error and keeps nothing.
    for limit in [spins_bytes(1, p) - 1, spins_bytes(1, p) + 16 * 17 - 1] {
        let small = Context::new(Some(1), limit).unwrap();
        assert!(matches!(
            h.evaluate(&small, &[&up]),
            Err(mandacaru_core::Error::Budget { .. })
        ));
        assert_eq!(small.memory_used(), 0);
        assert!(matches!(
            h.evaluate_spinor(&small, [[&up, &zero], [&zero, &down]]),
            Err(mandacaru_core::Error::Budget { .. })
        ));
        assert_eq!(small.memory_used(), 0);
    }
}

fn spins_bytes(spins: usize, p: usize) -> usize {
    spins * p * p * 16
}

#[test]
fn the_exact_channels_reserve_the_stack_and_the_scratch() {
    let n = 3;
    let tensor = symmetric_tensor(n, 7);
    let layout = [(0, 0, 0), (1, 0, 0), (1, 1, 0)];
    let ones = DMatrix::from_row_slice(1, 1, &[1.]);
    let d0 = expand_blocks(&layout, &[(0, &ones), (1, &ones)]).unwrap();
    let zero = DMatrix::zeros(n, n);
    let atom = exact_atom(n, &tensor, zero.clone(), zero, d0).unwrap();
    let (rows, width) = (4, 6);
    let c = random_hermitian(6, 5)[..rows * width].to_vec();
    let dc = random_hermitian(6, 6)[..rows * width].to_vec();
    let members = [vec![0, 2, 3], vec![1, 4, 5]];
    let atoms = members
        .iter()
        .map(|m| Factors {
            members: m,
            factors: &atom.factors,
        })
        .collect::<Vec<_>>();
    let count = 2 * atom.factors.len();
    let square = rows * rows;
    for (real, entry) in [(false, 16usize), (true, 8)] {
        let stack = count * square * entry;
        // Gathered columns and `left` (rows x n each), and the rows x rows square, 16 bytes.
        let channels_scratch = (2 * rows * n + square) * 16;
        let derivatives_scratch = (3 * rows * n + square) * 16;
        let ctx = context();
        let out = exact_channels(&ctx, &c, rows, width, &atoms, real).unwrap();
        assert_eq!(ctx.memory_used(), stack);
        assert_eq!(ctx.memory_peak(), stack + channels_scratch);
        drop(out);
        assert_eq!(ctx.memory_used(), 0);
        let ctx = context();
        let out = exact_channel_derivatives(&ctx, &c, &dc, rows, width, &atoms, real).unwrap();
        assert_eq!(ctx.memory_used(), stack);
        assert_eq!(ctx.memory_peak(), stack + derivatives_scratch);
        drop(out);
        assert_eq!(ctx.memory_used(), 0);

        // Room for the stack but not for the scratch, and not even for the stack.
        for limit in [stack + channels_scratch - 1, stack - 1] {
            let small = Context::new(Some(1), limit).unwrap();
            assert!(matches!(
                exact_channels(&small, &c, rows, width, &atoms, real),
                Err(mandacaru_core::Error::Budget { .. })
            ));
            assert_eq!(small.memory_used(), 0);
        }
        let small = Context::new(Some(1), stack + derivatives_scratch - 1).unwrap();
        assert!(matches!(
            exact_channel_derivatives(&small, &c, &dc, rows, width, &atoms, real),
            Err(mandacaru_core::Error::Budget { .. })
        ));
        assert_eq!(small.memory_used(), 0);
    }
}
