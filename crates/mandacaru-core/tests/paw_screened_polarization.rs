//! The spin dependence of the spheres' semilocal short-range exchange
//! (`paw::screened_polarization`), on a species given as a bare
//! `SpeciesView`: no molecule, no setup.
//!
//! The polarization term is `P = (F[2 n_up] + F[2 n_dn]) / 2 - F[n_up + n_dn]`,
//! so it vanishes for equal spin densities, is even and quadratic in the
//! magnetization, reproduces the frozen unpolarized exchange of the reference
//! atom through the same density, and has a potential that is its derivative.
use mandacaru_core::{
    models::xc::exchange_hole_batch,
    paw::{
        onecenter_radial::{gradient_first_edge, trapezoid},
        pair_tables::{layout, ChannelView, Projector, SpeciesView},
        screened_atom::screened,
        screened_polarization::Polarization,
    },
    Context,
};
use nalgebra::DMatrix;

const RADIUS: f64 = 1.;
const OMEGA: f64 = 0.11;

/// Smooth waves on a uniform grid; the smooth ones are the all-electron ones
/// outside the sphere and a softened copy inside it.
struct Waves {
    knots: Vec<f64>,
    ae: Vec<Vec<Vec<f64>>>,
    ps: Vec<Vec<Vec<f64>>>,
    reference: Vec<DMatrix<f64>>,
}
fn waves() -> Waves {
    let knots: Vec<f64> = (1..=700).map(|k| 0.02 * k as f64).collect();
    let (mut ae, mut ps) = (Vec::new(), Vec::new());
    for l in 0..2usize {
        let channel: Vec<Vec<f64>> = vec![
            knots
                .iter()
                .map(|&r| 1.5 * r.powi(l as i32) * (-r).exp())
                .collect(),
            knots
                .iter()
                .map(|&r| 1.1 * r.powi(l as i32) * (-0.6 * r).exp())
                .collect(),
        ];
        ps.push(
            channel
                .iter()
                .map(|wave| {
                    knots
                        .iter()
                        .zip(wave)
                        .map(|(&r, &w)| {
                            let x = (r / RADIUS).min(1.);
                            w * (1. - 0.6 * (1. - x * x).powi(2))
                        })
                        .collect()
                })
                .collect(),
        );
        ae.push(channel);
    }
    // A reference density matrix per `m`, with off-diagonal elements.
    let reference = vec![
        DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.1, 0.15]),
        DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.2]) / 3.,
    ];
    Waves {
        knots,
        ae,
        ps,
        reference,
    }
}
fn view(w: &Waves) -> SpeciesView<'_> {
    SpeciesView {
        knots: &w.knots,
        compensation_radius: RADIUS,
        channels: (0..2)
            .map(|l| ChannelView {
                l,
                radius: RADIUS,
                ae: w.ae[l].iter().map(Vec::as_slice).collect(),
                ps: w.ps[l].iter().map(Vec::as_slice).collect(),
                reference: &w.reference[l],
            })
            .collect(),
    }
}
/// The reference density matrix placed in the projector layout.
fn reference_layout(w: &Waves, projectors: &[Projector]) -> DMatrix<f64> {
    let p = projectors.len();
    DMatrix::from_fn(p, p, |a, b| {
        let (pa, pb) = (&projectors[a], &projectors[b]);
        if pa.channel == pb.channel && pa.m == pb.m {
            w.reference[pa.channel][(pa.radial, pb.radial)]
        } else {
            0.
        }
    })
}
fn context() -> Context {
    Context::new(Some(1), 512 << 20).unwrap()
}

#[test]
fn the_unpolarized_exchange_of_the_reference_density_is_the_frozen_one() {
    // X0 = constant + V . D0 is what the frozen term adds at the reference. The
    // polarization module rebuilds the sphere's density from a density matrix in
    // the projector layout; at D0 it must give the same short-range exchange of
    // the all-electron minus the smooth density, i.e. the same frame.
    let ctx = context();
    let w = waves();
    let view = view(&w);
    let projectors = layout(&view);
    let atom = screened(&ctx, &view, &projectors, OMEGA).unwrap();
    let d0 = reference_layout(&w, &projectors);
    let x0 = atom.constant + (&atom.v * &d0).trace();
    let half = &d0 * 0.5;
    let terms = atom.polarization.evaluate(&ctx, &half, &half).unwrap();
    // The same energy from first principles: the density of the reference atom
    // on the full grid, the hole model, the trapezoid rule.
    let direct = |waves: &Vec<Vec<Vec<f64>>>| -> f64 {
        let n: Vec<f64> = (0..w.knots.len())
            .map(|k| {
                let mut density = 0.;
                for (l, channel) in waves.iter().enumerate() {
                    for i in 0..2 {
                        for j in 0..2 {
                            density += (2 * l + 1) as f64
                                * w.reference[l][(i, j)]
                                * channel[i][k]
                                * channel[j][k]
                                / (4. * std::f64::consts::PI);
                        }
                    }
                }
                density
            })
            .collect();
        let slope = gradient_first_edge(&w.knots, &n).unwrap();
        let values: Vec<f64> = n
            .iter()
            .zip(&slope)
            .flat_map(|(n, s)| [*n, s * s])
            .collect();
        let hole = exchange_hole_batch(&ctx, &values, OMEGA).unwrap();
        let integrand: Vec<f64> = (0..n.len())
            .map(|k| hole.as_slice()[3 * k] * 4. * std::f64::consts::PI * w.knots[k] * w.knots[k])
            .collect();
        trapezoid(&w.knots, &integrand)
    };
    let independent = direct(&w.ae) - direct(&w.ps);
    println!("independent F_ae - F_ps over the full grid: {independent:.12e}");
    println!(
        "X0 = {x0:.12e}, F[n(D0)] = {:.12e}, difference {:.3e}",
        terms.unpolarized,
        terms.unpolarized - x0
    );
    assert!(x0.abs() > 1e-4, "the test needs a nontrivial X0: {x0}");
    assert!(
        (terms.unpolarized - x0).abs() < 1e-9 * (1. + x0.abs()),
        "{} vs {x0}",
        terms.unpolarized
    );
}

#[test]
fn equal_spin_densities_have_no_polarization() {
    // A closed shell: the energy and both potentials are exactly zero, whatever
    // the density matrix is (here an asymmetric mixture of the reference).
    let ctx = context();
    let w = waves();
    let view = view(&w);
    let projectors = layout(&view);
    let polarization = Polarization::new(&view, &projectors, OMEGA).unwrap();
    let d0 = reference_layout(&w, &projectors);
    for scale in [0.3, 1., 1.7] {
        let d = &d0 * scale;
        let terms = polarization.evaluate(&ctx, &d, &d).unwrap();
        assert_eq!(terms.energy, 0., "scale {scale}");
        assert!(terms.up.iter().chain(terms.down.iter()).all(|v| *v == 0.));
    }
    let zero = DMatrix::zeros(projectors.len(), projectors.len());
    let terms = polarization.evaluate(&ctx, &zero, &zero).unwrap();
    assert_eq!((terms.energy, terms.unpolarized), (0., 0.));
}

#[test]
fn the_polarization_is_even_and_quadratic_in_the_magnetization() {
    // D_up,dn = D0 (1 +- t) / 2: the magnetization is t times the density.
    let ctx = context();
    let w = waves();
    let view = view(&w);
    let projectors = layout(&view);
    let polarization = Polarization::new(&view, &projectors, OMEGA).unwrap();
    let d0 = reference_layout(&w, &projectors);
    let at = |t: f64| {
        polarization
            .evaluate(&ctx, &(&d0 * (0.5 * (1. + t))), &(&d0 * (0.5 * (1. - t))))
            .unwrap()
            .energy
    };
    let (small, large) = (at(0.02), at(0.04));
    assert_eq!(at(-0.04), large, "P is even in the magnetization");
    println!(
        "P(0.02) = {small:.6e}, P(0.04) = {large:.6e}, ratio {:.5}",
        large / small
    );
    assert!(small.abs() > 1e-12);
    assert!((large / small - 4.).abs() < 0.1, "ratio {}", large / small);
}

#[test]
fn a_single_spin_follows_the_spin_scaling_relation() {
    // P(D, 0) = F[2 n]/2 - F[n], and F[2 n] is the unpolarized exchange of the
    // pair (D, D): the same quantity through the other code path.
    let ctx = context();
    let w = waves();
    let view = view(&w);
    let projectors = layout(&view);
    let polarization = Polarization::new(&view, &projectors, OMEGA).unwrap();
    let d = reference_layout(&w, &projectors);
    let zero = DMatrix::zeros(projectors.len(), projectors.len());
    let one = polarization.evaluate(&ctx, &d, &zero).unwrap();
    let two = polarization.evaluate(&ctx, &d, &d).unwrap();
    let expected = 0.5 * two.unpolarized - one.unpolarized;
    println!(
        "P(D, 0) = {:.9e}, F[2n]/2 - F[n] = {expected:.9e}",
        one.energy
    );
    assert!((one.energy - expected).abs() < 1e-13 * (1. + expected.abs()));
    assert!(one.energy.abs() > 1e-6);
}

#[test]
fn the_potentials_are_the_derivatives_of_the_energy() {
    // Perturb both spin densities along the reference's own pattern with a
    // different weight on each element; the central difference of P is
    // V_up . G_up + V_dn . G_dn.
    let ctx = context();
    let w = waves();
    let view = view(&w);
    let projectors = layout(&view);
    let polarization = Polarization::new(&view, &projectors, OMEGA).unwrap();
    let d0 = reference_layout(&w, &projectors);
    let p = projectors.len();
    let weights = |seed: f64| {
        DMatrix::from_fn(p, p, |a, b| {
            let (i, j) = (a.min(b) as f64, a.max(b) as f64);
            d0[(a, b)] * (0.5 + (seed * (3. * i + 7. * j + 1.)).sin())
        })
    };
    let (up, down) = (&d0 * 0.8, &d0 * 0.3);
    let (g_up, g_down) = (weights(0.37), weights(0.91));
    let at = |e: f64| {
        polarization
            .evaluate(&ctx, &(&up + &g_up * e), &(&down + &g_down * e))
            .unwrap()
            .energy
    };
    let e = 1e-5;
    let numeric = (at(e) - at(-e)) / (2. * e);
    let terms = polarization.evaluate(&ctx, &up, &down).unwrap();
    let analytic =
        (terms.up.component_mul(&g_up)).sum() + (terms.down.component_mul(&g_down)).sum();
    println!("dP numeric {numeric:.9e}, analytic {analytic:.9e}");
    assert!(analytic.abs() > 1e-6);
    assert!(
        (numeric - analytic).abs() < 1e-7 * analytic.abs(),
        "{numeric:e} vs {analytic:e}"
    );
}
