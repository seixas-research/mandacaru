//! Confined pseudo-atomic orbitals on a synthetic PAW channel: a harmonic well
//! with an inert projector, whose levels are known in closed form.
use mandacaru_core::basis::{confinement as wall, confinement::Potential, pao as radial};
use mandacaru_core::paw::{
    confinement::{self, ChannelData, Prepared, Radius},
    pao,
};
use mandacaru_core::Context;

const OMEGA: f64 = 1.;

struct Channel {
    table: Vec<f64>,
    local: Vec<f64>,
    projector: Vec<f64>,
    wave: Vec<f64>,
}
impl Channel {
    fn new() -> Self {
        let table: Vec<f64> = (1..=6000).map(|i| i as f64 * 0.005).collect();
        let local = table.iter().map(|r| 0.5 * OMEGA * OMEGA * r * r).collect();
        let projector = table.iter().map(|r| (-r * r).exp()).collect();
        // The 3D oscillator's l = 0 ground state, positive near the origin.
        let wave = table.iter().map(|r| (-0.5 * OMEGA * r * r).exp()).collect();
        Self {
            table,
            local,
            projector,
            wave,
        }
    }
    fn data(&self) -> ChannelData<'_> {
        ChannelData {
            table: &self.table,
            local: &self.local,
            projectors: &self.projector,
            rank: 1,
            // D = q = 0: the projector is inert and the problem is the bare well.
            coupling: &[0.],
            overlap: &[0.],
            wave: &self.wave,
            l: 0,
            r_cut: 1.5,
        }
    }
}
fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}
const DEFAULT: Potential = Potential {
    amplitude: wall::AMPLITUDE,
    inner_fraction: wall::INNER_FRACTION,
};

#[test]
fn the_confining_potential_is_flat_inside_and_diverges_at_the_wall() {
    let r: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
    let (r_c, inner) = (8., 0.6);
    let v = wall::potential(&r, r_c, 12., inner);
    for (x, v) in r.iter().zip(&v) {
        if *x <= inner * r_c {
            assert_eq!(*v, 0.);
        } else if *x < r_c {
            let expected = 12. / (r_c - x) * (-(r_c - inner * r_c) / (x - inner * r_c)).exp();
            assert!((v - expected).abs() <= 1e-15 * expected.abs().max(1.));
            assert!(*v >= 0.);
        } else {
            assert!(v.is_infinite() && *v > 0.);
        }
    }
    // It is smooth where it switches on: tiny just above the inner radius.
    assert!(v[48] == 0. && v[49] > 0. && v[49] < 1e-3);
}

#[test]
fn the_wall_sits_on_a_node_of_both_grids() {
    for r_c in [3.3, 7.0, 12.7] {
        let fine = wall::grid(r_c, 0);
        let coarse = wall::grid(r_c, 1);
        assert_eq!(fine.len() + 1, 2 * (coarse.len() + 1));
        let h = r_c / (fine.len() + 1) as f64;
        assert!((fine[0] - h).abs() < 1e-15);
        assert!((r_c - fine[fine.len() - 1] - h).abs() < 1e-12);
        assert!((r_c - coarse[coarse.len() - 1] - 2. * h).abs() < 1e-12);
        assert!(h <= 0.01 + 1e-12);
    }
    // A short wall still gets eight intervals on the coarse grid.
    assert_eq!(wall::grid(0.05, 1).len(), 7);
}

#[test]
fn the_free_level_is_the_oscillator_ground_state() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let free = confinement::free_energy(&ctx, &prepared, 0).unwrap();
    assert!((free - 1.5 * OMEGA).abs() < 1e-6, "{free}");
    // The second state of the l = 0 channel is 3.5 omega.
    let second =
        confinement::energy(&ctx, &prepared, 0.95 * 30., &Potential::default(), 1).unwrap();
    assert!((second - 3.5 * OMEGA).abs() < 1e-5, "{second}");
}

#[test]
fn a_nearer_wall_raises_the_level_monotonically() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let mut previous = f64::INFINITY;
    for r_c in [2.5, 3.5, 5., 7.] {
        let e = confinement::energy(&ctx, &prepared, r_c, &DEFAULT, 0).unwrap();
        assert!(e > 1.5 * OMEGA && e < previous);
        previous = e;
    }
    assert!(previous - 1.5 * OMEGA < 5e-3);
}

#[test]
fn the_radius_realizes_the_requested_shift() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let free = confinement::free_energy(&ctx, &prepared, 0).unwrap();
    let mut radii = Vec::new();
    for shift_ev in [0.05, 0.1, 0.3, 1.0] {
        let found =
            confinement::radius(&ctx, &prepared, free, shift_ev, &DEFAULT, false, 0).unwrap();
        let Radius::Found(r_c) = found else {
            panic!("{found:?}")
        };
        let level = confinement::energy(&ctx, &prepared, r_c, &DEFAULT, 0).unwrap();
        let achieved = (level - free) * mandacaru_core::units::HARTREE_TO_EV;
        assert!((achieved - shift_ev).abs() < 1e-3 * shift_ev, "{achieved}");
        radii.push(r_c);
    }
    // A larger shift is a nearer wall.
    assert!(radii.windows(2).all(|w| w[1] < w[0]));
}

#[test]
fn a_shift_out_of_reach_is_reported_not_guessed() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let free = confinement::free_energy(&ctx, &prepared, 0).unwrap();
    // Far larger than a wall at r_cut / inner can give: refused with the largest usable.
    match confinement::radius(&ctx, &prepared, free, 5000., &DEFAULT, false, 0).unwrap() {
        Radius::TooLarge { largest_ev } => assert!(largest_ev > 0. && largest_ev < 5000.),
        other => panic!("{other:?}"),
    }
    // ... or, for a deep channel, the tightest wall the sphere allows.
    match confinement::radius(&ctx, &prepared, free, 5000., &DEFAULT, true, 0).unwrap() {
        Radius::Tightest(r) => assert!((r - 1.5 / 0.6).abs() < 1e-12),
        other => panic!("{other:?}"),
    }
    // A level the table cannot reach (a free level 1 mHa too deep, as for a
    // reference that is not the dataset's own) is a reach, not a radius.
    match confinement::radius(&ctx, &prepared, free - 1e-3, 0.01, &DEFAULT, false, 0).unwrap() {
        Radius::TooSmall { reach } => assert!((reach - 0.95 * 30.).abs() < 1e-9),
        other => panic!("{other:?}"),
    }
    assert!(confinement::radius(&ctx, &prepared, free, -1., &DEFAULT, false, 0).is_err());
    assert!(confinement::radius(&ctx, &prepared, free, f64::NAN, &DEFAULT, false, 0).is_err());
}

#[test]
fn the_eigenstate_is_normalized_signed_and_vanishes_at_the_wall() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let r_c = 4.2;
    let (r, u, e) = confinement::eigenstate(&ctx, &prepared, r_c, &DEFAULT, 0).unwrap();
    let h = r[1] - r[0];
    // q = 0: the metric is the identity, int u^2 dr = 1.
    let norm: f64 = u.iter().map(|v| v * v).sum::<f64>() * h;
    assert!((norm - 1.).abs() < 1e-3, "{norm}");
    let overlap: f64 = u
        .iter()
        .zip(&r)
        .map(|(u, r)| u * (-0.5 * OMEGA * r * r).exp() * r)
        .sum::<f64>();
    assert!(overlap > 0.);
    assert!(u[u.len() - 1].abs() < 1e-4);
    assert!((e - confinement::energy(&ctx, &prepared, r_c, &DEFAULT, 0).unwrap()).abs() < 1e-12);
    // The wall moves out by one grid, the sign does not flip.
    assert!(u[2] > 0.);
}

#[test]
fn the_orbital_is_tabulated_on_the_dataset_grid() {
    let ctx = context();
    let channel = Channel::new();
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    let orbital = confinement::orbital(&ctx, &prepared, 0.1, &DEFAULT, 0, None)
        .unwrap()
        .unwrap();
    assert_eq!(orbital.radial.len(), channel.table.len());
    for (r, v) in channel.table.iter().zip(&orbital.radial) {
        if *r >= orbital.r_c {
            assert_eq!(*v, 0.);
        }
    }
    assert!(orbital.radial[100] > 0.);
    let achieved = (orbital.energy - orbital.free_energy) * mandacaru_core::units::HARTREE_TO_EV;
    assert!((achieved - 0.1).abs() < 1e-3);
    // The free level may be supplied; the result is identical.
    let again = confinement::orbital(&ctx, &prepared, 0.1, &DEFAULT, 0, Some(orbital.free_energy))
        .unwrap()
        .unwrap();
    assert_eq!(again.r_c, orbital.r_c);
    assert_eq!(again.radial, orbital.radial);
    // A shift the table cannot reach returns the reach as the error value.
    let reach = confinement::orbital(
        &ctx,
        &prepared,
        0.01,
        &DEFAULT,
        0,
        Some(orbital.free_energy - 1e-3),
    )
    .unwrap()
    .unwrap_err();
    assert!((reach - 0.95 * 30.).abs() < 1e-9);
}

#[test]
fn bad_arrays_and_potentials_are_refused() {
    let ctx = context();
    let channel = Channel::new();
    let mut data = channel.data();
    data.coupling = &[0., 0.];
    assert!(Prepared::new(&ctx, data).is_err());
    let prepared = Prepared::new(&ctx, channel.data()).unwrap();
    for bad in [
        Potential {
            amplitude: -1.,
            inner_fraction: 0.6,
        },
        Potential {
            amplitude: 12.,
            inner_fraction: 1.,
        },
    ] {
        assert!(confinement::energy(&ctx, &prepared, 5., &bad, 0).is_err());
    }
    // A wall outside the table is not a confinement.
    assert!(confinement::energy(&ctx, &prepared, 31., &DEFAULT, 0).is_err());
    assert!(confinement::energy(&ctx, &prepared, f64::NAN, &DEFAULT, 0).is_err());
    // A budget too small for the splines is refused, not allocated.
    let tiny = Context::new(Some(1), 1 << 10).unwrap();
    assert!(Prepared::new(&tiny, channel.data()).is_err());
}

#[test]
fn the_bound_state_of_an_empty_channel_is_found_in_its_box() {
    let ctx = context();
    let channel = Channel::new();
    let level = pao::bound_level(&ctx, channel.data(), 12.).unwrap();
    assert!((level - 1.5 * OMEGA).abs() < 1e-5, "{level}");
    let (radial, level_again) = pao::bound_radial(&ctx, channel.data(), 12.).unwrap();
    assert!((level - level_again).abs() < 1e-12);
    // R = u / r follows exp(-omega r^2 / 2) up to the normalization.
    let at = |r: f64| radial[(r / 0.005) as usize - 1];
    let ratio = at(1.0) / at(0.5);
    assert!(
        (ratio - (-0.5 * OMEGA * (1. - 0.25f64)).exp()).abs() < 1e-5,
        "{ratio}"
    );
    for (r, v) in channel.table.iter().zip(&radial) {
        if *r >= 12. {
            assert_eq!(*v, 0.);
        }
    }
    assert!(pao::bound_level(&ctx, channel.data(), 0.001).is_err());
}

#[test]
fn the_polarization_shells_are_normalized_and_cut() {
    let table: Vec<f64> = (1..=4000).map(|i| i as f64 * 0.005).collect();
    let alpha = 0.7;
    let r_cut = 6.;
    let q = radial::quasi_gaussian(&table, alpha, r_cut);
    for (r, v) in table.iter().zip(&q) {
        if *r >= r_cut {
            assert_eq!(*v, 0.);
        }
    }
    // Value and slope vanish at the cutoff.
    let edge = table.iter().rposition(|r| *r < r_cut).unwrap();
    assert!(q[edge].abs() < 1e-8 * q[0].abs());
    assert!(((q[edge] - q[edge - 1]) / 0.005).abs() < 1e-5);
    let radial = radial::polarization_radial(&table, 1, r_cut, 1. / alpha.sqrt()).unwrap();
    assert!((radial::radial_norm(&table, &radial) - 1.).abs() < 1e-12);
    assert!(radial[0] > 0. && radial[1] / radial[0] > 1.5);
    let outermost: Vec<f64> = table.iter().map(|r| (-r).exp()).collect();
    for power in [1, 2] {
        let shape = radial::power_shape(&table, &outermost, power).unwrap();
        assert!((radial::radial_norm(&table, &shape) - 1.).abs() < 1e-12);
    }
    assert!(radial::power_shape(&table, &outermost[1..], 1).is_err());
    assert!(radial::polarization_radial(&table, 1, -1., 1.).is_err());
    assert!(radial::polarization_radial(&table[..2], 1, 1., 1.).is_err());
}
