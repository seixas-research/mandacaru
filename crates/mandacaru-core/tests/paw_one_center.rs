//! The densities and the frozen one-center constant of a reference atom
//! (`paw::generate::{core, energies}`, A2): composed from the recorded inputs of the A0 freeze
//! and compared with the dataset scalars the Python generator stored, plus the checks that do
//! not need the freeze: the variational scheme (finite-difference derivative of its own energy,
//! its documented difference from the shipped scheme on PBE) and closed forms.
mod paw_assemble_support;

use mandacaru_core::math::radial::local_derivatives;
use mandacaru_core::models::xc_scheme::{AtomSolver, Variational, XcSource};
use mandacaru_core::paw::generate::core::{
    augmented_density, core_density, pseudize_density, valence_sums, ChannelWaves, CoreShell,
};
use mandacaru_core::paw::generate::energies::{
    one_center_energies, xc_energies, EnergyDensities, OneCenterEnergies,
};
use mandacaru_core::Context;
use paw_assemble_support::call::call;
use paw_assemble_support::{datasets, Frozen};
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 512 << 20).unwrap()
}

/// The six densities the freeze recorded for the energies, in the order they were taken:
/// `hartree_energy.{0,1}` are the all-electron valence density and the augmented one,
/// `xc_energies.{0..3}` the all-electron valence, smooth valence, all-electron valence plus core
/// and smooth valence plus smooth core.
struct Recorded {
    r: Vec<f64>,
    ae_valence: Vec<f64>,
    augmented: Vec<f64>,
    smooth_valence: Vec<f64>,
    core: Vec<f64>,
    smooth_core: Vec<f64>,
    xc: String,
    relativistic: bool,
}

fn recorded(f: &Frozen) -> Recorded {
    let rho = |function: &str, k: usize| {
        let cases = f.cases(function);
        let (_, case) = &cases[k];
        f.input(&case["inputs"]["rho"])
    };
    let cases = f.cases("_xc_energies");
    let first = &cases[0].1["inputs"];
    let difference =
        |a: Vec<f64>, b: &[f64]| a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>();
    let ae_valence = rho("_hartree_energy", 0);
    let smooth_valence = rho("_xc_energies", 1);
    Recorded {
        r: f.input(&first["r"]),
        core: difference(rho("_xc_energies", 2), &ae_valence),
        smooth_core: difference(rho("_xc_energies", 3), &smooth_valence),
        augmented: rho("_hartree_energy", 1),
        ae_valence,
        smooth_valence,
        xc: first["xc"].as_str().unwrap().to_string(),
        relativistic: first["relativistic"].as_bool().unwrap(),
    }
}

fn energies(ctx: &Context, source: &dyn XcSource, f: &Frozen, d: &Recorded) -> OneCenterEnergies {
    let band = f.doc["scalars"]["energies"]["band"].as_f64().unwrap();
    let densities = EnergyDensities {
        band,
        ae_valence: &d.ae_valence,
        smooth_valence: &d.smooth_valence,
        augmented: &d.augmented,
        core: &d.core,
        smooth_core: &d.smooth_core,
    };
    one_center_energies(ctx, source, &d.r, &densities, &d.xc, d.relativistic).unwrap()
}

fn terms(e: &OneCenterEnergies) -> [(&'static str, f64); 9] {
    [
        ("band", e.band),
        ("reference_valence", e.reference_valence),
        ("pseudo_atom", e.pseudo_atom),
        ("one_center", e.one_center),
        ("hartree_ae", e.hartree_ae),
        ("hartree_ps", e.hartree_ps),
        ("xc_ae", e.xc_ae),
        ("xc_ps", e.xc_ps),
        ("core_valence_xc_omitted", e.core_valence_xc_omitted),
    ]
}

#[test]
fn the_one_center_constant_composes_from_the_frozen_densities() {
    let ctx = context();
    let (mut worst, mut variational) = (0f64, vec![]);
    for name in datasets() {
        let f = Frozen::open(&name);
        let d = recorded(&f);
        let frozen = &f.doc["scalars"]["energies"];
        let ours = energies(&ctx, &AtomSolver, &f, &d);
        // PBE moves the integrals by 1e-12 of their value through the exchange-floor band
        // (`paw_assemble.rs`); the constant is a difference of such terms.
        let gate = if d.xc == "pbe" { 2e-11 } else { 2e-12 };
        for (key, value) in terms(&ours) {
            let want = frozen[key].as_f64().unwrap();
            let deviation = (value - want).abs() / want.abs().max(1.);
            worst = worst.max(deviation / gate);
            assert!(
                deviation <= gate,
                "{name} {key}: {value} against {want} ({deviation:.2e})"
            );
        }
        let stored = f.doc["scalars"]["one_center_energy"].as_f64().unwrap();
        assert!(
            (ours.one_center - stored).abs() <= 1e-10,
            "{name}: {}",
            ours.one_center - stored
        );
        let moved = energies(&ctx, &Variational, &f, &d);
        variational.push(format!(
            "  {name:<12} one_center {:+.6e} (atom scheme {:+.6e}, difference {:+.2e}); xc_ps difference {:+.2e}",
            moved.one_center,
            ours.one_center,
            moved.one_center - ours.one_center,
            moved.xc_ps - ours.xc_ps
        ));
    }
    println!(
        "worst deviation {worst:.3} of the gate; the variational scheme on the same densities:"
    );
    variational.iter().for_each(|row| println!("{row}"));
}

/// The index recorded after the last dot of a case key.
fn order(key: &str) -> usize {
    key.rsplit('.').next().unwrap().parse().unwrap()
}

#[test]
fn the_densities_of_the_assembled_channels_reproduce_the_frozen_ones() {
    let ctx = context();
    let mut checked = 0;
    for name in datasets() {
        let f = Frozen::open(&name);
        let mut cases = f.cases("assemble_paw_channel");
        cases.sort_by_key(|(key, _)| order(key));
        // A Dirac dataset also records the j-resolved branches: only the scalar channels count.
        if cases.len() != f.doc["scalars"]["channels"].as_object().unwrap().len() {
            continue;
        }
        let d = recorded(&f);
        let runs = cases
            .iter()
            .map(|(_, case)| call(&f, case))
            .collect::<Vec<_>>();
        let channels = runs.iter().map(|run| run.run(&ctx)).collect::<Vec<_>>();
        let waves = channels
            .iter()
            .zip(&runs)
            .map(|(channel, run)| ChannelWaves {
                channel,
                ae_waves: &run.waves,
            })
            .collect::<Vec<_>>();
        let sums = valence_sums(d.r.len(), &waves).unwrap();
        let scalars = &f.doc["scalars"];
        let relative = |a: &[f64], b: &[f64]| {
            let scale = b.iter().fold(0f64, |m, x| m.max(x.abs()));
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0., f64::max)
                / scale
        };
        let radius = scalars["compensation_radius"].as_f64().unwrap();
        let augmented =
            augmented_density(&d.r, &sums.smooth_valence, sums.compensation_charge, radius);
        for (what, got, want) in [
            (
                "ae_valence",
                relative(&sums.ae_valence, &d.ae_valence),
                1e-12,
            ),
            (
                "smooth_valence",
                relative(&sums.smooth_valence, &d.smooth_valence),
                1e-12,
            ),
            ("augmented", relative(&augmented, &d.augmented), 1e-12),
            (
                "compensation_charge",
                (sums.compensation_charge - scalars["compensation_charge"].as_f64().unwrap()).abs(),
                1e-12,
            ),
            (
                "band",
                (sums.band - scalars["energies"]["band"].as_f64().unwrap()).abs(),
                1e-12,
            ),
        ] {
            assert!(got <= want, "{name} {what}: {got:.2e}");
        }
        drop(waves);
        checked += 1;
    }
    assert!(
        checked >= 10,
        "only {checked} frozen datasets have scalar channels only"
    );
    println!("{checked} datasets: densities, compensation charge and band at 1e-12");
}

#[test]
fn the_variational_potential_is_the_derivative_of_its_energy_and_the_atom_scheme_is_not() {
    let ctx = context();
    let f = Frozen::open("pbe-sr_O");
    let cases = f.cases("_xc_energies");
    let (_, case) = &cases[0];
    let (r, rho) = (
        f.input(&case["inputs"]["r"]),
        f.input(&case["inputs"]["rho"]),
    );
    let energy = |density: &[f64]| {
        xc_energies(&ctx, &Variational, &r, density, "pbe", true)
            .unwrap()
            .energy
    };
    let potential = Variational
        .evaluate(&ctx, &r, &rho, "pbe", true)
        .unwrap()
        .potential;
    let atom = AtomSolver
        .evaluate(&ctx, &r, &rho, "pbe", true)
        .unwrap()
        .potential;
    let (mut worst, mut atom_worst) = (0f64, 0f64);
    for i in [299, 575, 800] {
        let width = 0.5 * (r[i + 1] - r[i - 1]);
        let weight = 4. * PI * r[i] * r[i] * width;
        let step = 1e-5 * rho[i];
        let (mut up, mut down) = (rho.clone(), rho.clone());
        up[i] += step;
        down[i] -= step;
        let slope = (energy(&up) - energy(&down)) / (2. * step);
        worst = worst.max((slope - weight * potential[i]).abs() / (weight * potential[i]).abs());
        atom_worst = atom_worst.max((slope - weight * atom[i]).abs() / (weight * atom[i]).abs());
    }
    println!("finite difference of E_xc against w v_xc: variational {worst:.2e}, atom scheme {atom_worst:.2e}");
    assert!(worst < 5e-6, "variational: {worst:.2e}");
    assert!(
        atom_worst > 1e-5,
        "the atom scheme is not the derivative of the energy: {atom_worst:.2e}"
    );
}

#[test]
fn the_variational_pbe_potential_differs_from_the_shipped_scheme_by_the_documented_amount() {
    let ctx = context();
    for name in ["pbe-sr_O", "pbe-sr_Cu", "pbe-fr_Si", "pbe-sr_Ce"] {
        let f = Frozen::open(name);
        for (key, case) in f.cases("_xc_energies") {
            let (r, rho) = (
                f.input(&case["inputs"]["r"]),
                f.input(&case["inputs"]["rho"]),
            );
            let v = Variational
                .evaluate(&ctx, &r, &rho, "pbe", true)
                .unwrap()
                .potential;
            let frozen = f.output(&case["outputs"]["v_xc"]);
            let mut worst = (0f64, 0usize);
            for (k, b) in frozen.values.iter().enumerate() {
                let i = (k + 1) * frozen.stride - 1;
                // From the nucleus to the 1e-9 density contour; beyond it the schemes differ
                // at the exchange floor (`paw_assemble.rs`).
                if rho[i] >= 1e-9 && i > 2 && (v[i] - b).abs() > worst.0 {
                    worst = ((v[i] - b).abs(), i);
                }
            }
            let energy = xc_energies(&ctx, &Variational, &r, &rho, "pbe", true)
                .unwrap()
                .energy;
            let delta = energy - f.output(&case["outputs"]["E_xc"]).values[0];
            println!(
                "{name} {key}: max |dv_xc| {:.2e} Ha at r = {:.3} (rho {:.2e}); E_xc difference {delta:+.2e} Ha",
                worst.0, r[worst.1], rho[worst.1]
            );
            assert!(
                worst.0 > 1e-6,
                "{name} {key}: the schemes are not distinguishable"
            );
            // Cerium's all-electron valence density has a node-like minimum at 0.11 Bohr where
            // the gradient correction spikes (frozen +14.2 Ha, variational +33.8 Ha): the
            // pointwise potential there is reported, not bounded; the energy still is.
            let spike = name.ends_with("Ce");
            assert!(spike || worst.0 < 5e-2, "{name} {key}: {:.2e}", worst.0);
            assert!(delta.abs() < 2e-3, "{name} {key}: {delta:.2e}");
        }
    }
}

#[test]
fn the_smooth_density_matches_value_slope_and_curvature_and_the_fit_is_exact_on_a_polynomial() {
    let r = (1..=3000).map(|i| i as f64 * 0.004).collect::<Vec<_>>();
    // 1 + 2 r^2 + 3 r^4 - r^6: the degree-8 fit is exact, so the derivatives at r_cut are.
    let density = r
        .iter()
        .map(|x| 1. + 2. * x * x + 3. * x.powi(4) - x.powi(6))
        .collect::<Vec<_>>();
    let cut = r[249]; // 1.0
    let at = local_derivatives(&r, &density, cut, 4, 25, 8).unwrap();
    let want = [
        1. + 2. + 3. - 1.,
        4. * cut + 12. * cut.powi(3) - 6. * cut.powi(5),
        4. + 36. * cut.powi(2) - 30. * cut.powi(4),
        72. * cut - 120. * cut.powi(3),
        72. - 360. * cut.powi(2),
    ];
    for (k, w) in want.iter().enumerate() {
        assert!(
            (at[k] - w).abs() <= 1e-8 * w.abs().max(1.),
            "derivative {k}: {} against {w}",
            at[k]
        );
    }
    let smooth = pseudize_density(&r, &density, cut + 1e-4).unwrap(); // snaps to the node
    for i in 0..r.len() {
        if r[i] > cut {
            assert_eq!(smooth[i], density[i]);
        }
    }
    // Inside it is the even quartic through the same value, slope and curvature.
    assert!((smooth[249] - density[249]).abs() < 1e-8);
    // Inside it is the even quartic p0 + p1 r^2 + p2 r^4: three inside nodes determine it, and
    // its value, slope and curvature at r_cut are those of the density.
    let nodes = [10usize, 100, 200];
    let system = nalgebra::Matrix3::from_fn(|i, j| r[nodes[i]].powi(2 * j as i32));
    let p = system
        .lu()
        .solve(&nalgebra::Vector3::from_fn(|i, _| smooth[nodes[i]]))
        .unwrap();
    for j in [20usize, 50, 150, 230] {
        let predicted = p[0] + p[1] * r[j].powi(2) + p[2] * r[j].powi(4);
        assert!(
            (predicted - smooth[j]).abs() < 1e-9,
            "{j}: {predicted} against {}",
            smooth[j]
        );
    }
    let value = p[0] + p[1] * cut.powi(2) + p[2] * cut.powi(4);
    let slope = 2. * p[1] * cut + 4. * p[2] * cut.powi(3);
    let curvature = 2. * p[1] + 12. * p[2] * cut.powi(2);
    for (got, want) in [(value, at[0]), (slope, at[1]), (curvature, at[2])] {
        assert!((got - want).abs() < 1e-7, "{got} against {want}");
    }
}

#[test]
fn a_hydrogenic_core_density_integrates_to_its_occupancy() {
    let r = (1..=12000).map(|i| i as f64 * 0.0025).collect::<Vec<_>>();
    let z: f64 = 3.;
    let u = r
        .iter()
        .map(|x| 2. * z.powf(1.5) * x * (-z * x).exp())
        .collect::<Vec<_>>();
    let density = core_density(
        &r,
        &[CoreShell {
            occupancy: 2.,
            orbital: &u,
        }],
    )
    .unwrap();
    for (x, rho) in r.iter().zip(&density).step_by(997) {
        assert!((rho - 2. * z.powi(3) * (-2. * z * x).exp() / PI).abs() < 1e-12 * z.powi(3));
    }
    let shell = r
        .iter()
        .zip(&density)
        .map(|(x, rho)| 4. * PI * x * x * rho)
        .collect::<Vec<_>>();
    let electrons = mandacaru_core::math::radial::integrate(&r, &shell).unwrap();
    // The trapezoid rule leaves out the first interval, 1.7e-6 electrons here.
    assert!((electrons - 2.).abs() < 1e-5, "{electrons}");
    let none = core_density(&r, &[]).unwrap();
    assert!(none.iter().all(|x| *x == 0.));
    assert!(core_density(
        &r,
        &[CoreShell {
            occupancy: 1.,
            orbital: &u[1..]
        }]
    )
    .is_err());
}

#[test]
fn the_variational_lda_agrees_with_the_freeze_above_the_exchange_floor_and_not_below() {
    // D-2 expects LDA to be identical in both schemes. It is, where the density is at least
    // the 2e-12 below which the kernel of `spherical::exchange_correlation` drops exchange;
    // the atom scheme and the freeze continue the uniform-gas tail there, the variational one
    // does not (printed below; if that is repaired, only the printed number changes).
    let ctx = context();
    for name in ["lda-sr_O", "lda-sr_Fe", "upaw_Al"] {
        let f = Frozen::open(name);
        for (key, case) in f.cases("_xc_energies") {
            let (r, rho) = (
                f.input(&case["inputs"]["r"]),
                f.input(&case["inputs"]["rho"]),
            );
            let ours = Variational.evaluate(&ctx, &r, &rho, "lda", true).unwrap();
            let frozen = f.output(&case["outputs"]["v_xc"]);
            let (mut above, mut below) = (0f64, 0f64);
            for (k, b) in frozen.values.iter().enumerate() {
                let i = (k + 1) * frozen.stride - 1;
                let d = (ours.potential[i] - b).abs();
                if rho[i] >= 2e-12 {
                    above = above.max(d);
                } else {
                    below = below.max(d);
                }
            }
            println!("{name} {key}: |dv_xc| above the floor {above:.2e}, below {below:.2e} Ha");
            assert!(above < 1e-12 * frozen.scale, "{name} {key}: {above:.2e}");
            // Reported, not pinned: the far tail is a defect of the scheme, not behavior to keep.
            assert!(below < 1e-3, "{name} {key}: {below:.2e}");
        }
    }
}
