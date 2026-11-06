//! The density-independent fields of the periodic setup against the frozen Si
//! crystal (T1 = 1e-10 of the array maximum), plus an independent image sum of
//! the superposed valence density that settles the support-threshold and spline
//! parity items of the oracle.
mod setup_support;

use mandacaru_core::{
    math::spline::Cubic,
    periodic::setup::{
        fields::{
            core, core_table, initial_charges, initial_density, initial_magnetization, kerker,
            phase, reference, valence_field, Placed, Tables,
        },
        shapes::Sites,
    },
    Context, Error,
};
use nalgebra::Vector3;
use setup_support::*;

const T1: f64 = 1e-10;

#[test]
fn valence_cutoff_does_not_truncate_the_spline_boundary_data() {
    let ctx = Context::new(Some(1), 16 << 20).unwrap();
    let r = [0., 1., 2., 3., 4., 5., 6., 7.];
    let values = [1., 2., 3., 4., 5., 1., 0., 0.];
    let grid = mandacaru_core::math::SpatialGrid::new(
        [1, 1, 1], [0.; 3], [1., 0., 0., 0., 1., 0., 0., 0., 1.], false,
    ).unwrap();
    let actual = valence_field(&ctx, &grid, &r, &values, [-4.5, 0., 0.]).unwrap();
    let full = Cubic::new(&ctx, ctx.copy_slice(&r).unwrap(), ctx.copy_slice(&values).unwrap()).unwrap();
    let truncated = Cubic::new(&ctx, ctx.copy_slice(&r[..6]).unwrap(), ctx.copy_slice(&values[..6]).unwrap()).unwrap();
    let expected = full.value_slope(4.5).0;
    assert!((actual.as_slice()[0] - expected).abs() < 1e-14);
    assert!((truncated.value_slope(4.5).0 - expected).abs() > 1e-3);
}

fn setup(si: &Silicon, shipped: &Shipped) -> (Vec<Placed>, bool) {
    let atoms = si
        .centers
        .iter()
        .map(|&center| Placed { species: 0, center })
        .collect();
    (atoms, shipped.nlcc_applied)
}

#[test]
fn the_grid_arrays_equal_the_frozen_crystal() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let k = kerker(&ctx, &si.grid).unwrap();
        let p = phase(&ctx, &si.grid).unwrap();
        let (ke, pe) = (
            relative(k.as_slice(), &reals(&si.json["kerker"])),
            relative(p.as_slice(), &complexes(&si.json["phase"])),
        );
        let sites = Sites {
            atoms: &si.atoms,
            channels: &si.channels,
            lattice: si.lattice,
            sigma: si.sigma,
        };
        let r = reference(&sites);
        let want = si.json["reference"].as_f64().unwrap();
        println!("kerker {ke:.2e}, phase {pe:.2e}, reference {r:.17} against {want:.17}");
        assert!(ke < T1 && pe < T1 && (r - want).abs() / want < 1e-14);
        assert_eq!(k.as_slice()[0], 1.);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_symmetric_wave_vectors_equal_the_frozen_xc_wave() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    {
        let wave = mandacaru_core::math::reciprocal::symmetric_wavevectors(&ctx, &si.grid).unwrap();
        let want = reals(&si.json["xc_wave"]);
        assert_eq!(want.len(), 3000);
        let error = relative(wave.as_slice(), &want);
        // Control: the plain FFT box differs on this fcc grid.
        let b = mandacaru_core::math::reciprocal::reciprocal_cell(&si.grid).unwrap();
        let plain: Vec<f64> = (0..3)
            .flat_map(|a| {
                let b = &b;
                (0..1000)
                    .map(move |at| mandacaru_core::math::reciprocal::box_vector(b, [10; 3], at)[a])
            })
            .collect();
        let control = relative(&plain, &want);
        println!("xc_wave: symmetric wave vectors {error:.2e} from the frozen array; the box alone {control:.2e}");
        assert!(error < T1 && control > 1e-3);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_core_the_initial_density_and_the_monopoles_equal_the_frozen_crystal() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        let (atoms, applied) = setup(&si, &shipped);
        let smooth = core_table(applied, &shipped.nlcc_source, &shipped.smooth_core).unwrap();
        let tables = [Tables {
            radius: &shipped.radius,
            valence_density: &shipped.valence_density,
            core: smooth,
        }];
        let c = core(&ctx, &si.grid, &tables, &atoms, false).unwrap();
        let rho = initial_density(&ctx, &si.grid, &tables, &atoms).unwrap();
        let m =
            initial_magnetization(&ctx, &si.grid, &tables, &atoms, &[4., 4.], &[0., 0.]).unwrap();
        let (ce, re) = (
            relative(c.as_slice(), &reals(&si.json["core"])),
            relative(rho.as_slice(), &reals(&si.json["initial"])),
        );
        println!(
            "core {ce:.2e}, initial density {re:.2e}; core table applied={applied} source={:?}",
            shipped.nlcc_source
        );
        assert!(ce < T1 && re < T1);
        assert!(m.as_slice().iter().all(|v| *v == 0.));
        assert_eq!(m.as_slice().len(), 1000);
        let sites = Sites {
            atoms: &si.atoms,
            channels: &si.channels,
            lattice: si.lattice,
            sigma: si.sigma,
        };
        let q = initial_charges(&sites, &[shipped.compensation_charge; 2]).unwrap();
        let qe = relative(&q, &complexes(&si.json["initial_q"]));
        println!("initial_q {qe:.2e}");
        assert!(qe < 1e-13);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The oracle's own algorithm, written for the test: a not-a-knot spline of the
/// whole table, every atom and every image within `support`, support = the last
/// radius above `threshold` of the peak.
fn image_sum(
    si: &Silicon,
    ctx: &Context,
    radius: &[f64],
    values: &[f64],
    threshold: f64,
) -> Vec<f64> {
    let peak = values.iter().fold(0f64, |m, v| m.max(v.abs()));
    let support = radius
        .iter()
        .zip(values)
        .filter(|(_, v)| v.abs() > threshold * peak)
        .map(|(r, _)| *r)
        .next_back()
        .unwrap();
    let spline = Cubic::new(
        ctx,
        ctx.copy_slice(radius).unwrap(),
        ctx.copy_slice(values).unwrap(),
    )
    .unwrap();
    (0..1000)
        .map(|at| {
            let p = Vector3::from(si.grid.point(at));
            let mut total = 0.;
            for center in &si.centers {
                for i in -5..=5 {
                    for j in -5..=5 {
                        for k in -5..=5 {
                            let image = Vector3::from(*center)
                                + si.lattice * Vector3::new(i as f64, j as f64, k as f64);
                            let d = (p - image).norm();
                            if d <= support {
                                total += spline.value_slope(d).0;
                            }
                        }
                    }
                }
            }
            total
        })
        .collect()
}

#[test]
fn the_support_threshold_and_the_spline_of_the_placement_reproduce_the_oracle() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        let (atoms, _) = setup(&si, &shipped);
        let tables = [Tables {
            radius: &shipped.radius,
            valence_density: &shipped.valence_density,
            core: None,
        }];
        let rho = initial_density(&ctx, &si.grid, &tables, &atoms).unwrap();
        let frozen = reals(&si.json["initial"]);
        let wide = image_sum(&si, &ctx, &shipped.radius, &shipped.valence_density, 1e-12);
        let tight = image_sum(&si, &ctx, &shipped.radius, &shipped.valence_density, 1e-14);
        let (rust_frozen, wide_frozen, tight_frozen, rust_wide) = (
            relative(rho.as_slice(), &frozen),
            relative(&wide, &frozen),
            relative(&tight, &frozen),
            relative(rho.as_slice(), &wide),
        );
        println!(
            "initial density against the frozen array: Rust {rust_frozen:.2e}; an independent image sum at the oracle's 1e-12 support {wide_frozen:.2e}, at the placement's 1e-14 support {tight_frozen:.2e}; Rust against the 1e-12 sum {rust_wide:.2e}"
        );
        // The independent sum at the oracle's threshold IS the frozen number; the
        // 1e-14 support is not (5e-11 away); the Rust profile follows the oracle.
        assert!(wide_frozen < 1e-13 && tight_frozen > 1e-11);
        assert!(rust_frozen < 1e-11 && rust_wide < 1e-11);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_initial_magnetization_is_the_clamped_moment_fraction_of_each_atom() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        let (atoms, _) = setup(&si, &shipped);
        let tables = [Tables {
            radius: &shipped.radius,
            valence_density: &shipped.valence_density,
            core: None,
        }];
        let first = initial_density(&ctx, &si.grid, &tables, &atoms[..1]).unwrap();
        let second = initial_density(&ctx, &si.grid, &tables, &atoms[1..]).unwrap();
        let charges = [4., 4.];
        let m =
            initial_magnetization(&ctx, &si.grid, &tables, &atoms, &charges, &[1., -9.]).unwrap();
        let want: Vec<f64> = first
            .as_slice()
            .iter()
            .zip(second.as_slice())
            .map(|(a, b)| 0.25 * a - b)
            .collect();
        let error = relative(m.as_slice(), &want);
        println!("moments (1, -9): clamped fractions (1/4, -1): {error:.2e}");
        assert!(error < 1e-12);
        assert!(
            initial_magnetization(&ctx, &si.grid, &tables, &atoms, &charges[..1], &[0., 0.])
                .is_err()
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_kinetic_core_is_the_von_weizsaecker_density_and_other_sources_raise() {
    let si = Silicon::load();
    let shipped = Shipped::read("lda-sr", "Si");
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    {
        let (atoms, applied) = setup(&si, &shipped);
        assert!(
            applied,
            "the lda-sr Si dataset is expected to carry a smooth core"
        );
        let smooth = core_table(applied, &shipped.nlcc_source, &shipped.smooth_core).unwrap();
        let tables = [Tables {
            radius: &shipped.radius,
            valence_density: &shipped.valence_density,
            core: smooth,
        }];
        let density = core(&ctx, &si.grid, &tables, &atoms, false).unwrap();
        let tau = core(&ctx, &si.grid, &tables, &atoms, true).unwrap();
        assert!(tau.as_slice().iter().all(|t| t.is_finite() && *t >= 0.));
        // The oracle's definition, built from the Cubic and the image sum of the
        // test: tau(r) = n'(r)^2 / (8 n(r)) at the knots (zero under 1e-12 of the
        // peak), splined, summed over the images.
        let spline = Cubic::new(
            &ctx,
            ctx.copy_slice(&shipped.radius).unwrap(),
            ctx.copy_slice(&shipped.smooth_core).unwrap(),
        )
        .unwrap();
        let floor = shipped.smooth_core.iter().fold(0f64, |m, v| m.max(v.abs())) * 1e-12;
        let radial: Vec<f64> = shipped
            .radius
            .iter()
            .map(|&r| {
                let (n, slope) = spline.value_slope(r);
                if n > floor {
                    slope * slope / (8. * n)
                } else {
                    0.
                }
            })
            .collect();
        let want = image_sum(&si, &ctx, &shipped.radius, &radial, 1e-14);
        let error = relative(tau.as_slice(), &want);
        println!("kinetic core against the independent image sum of n'^2/(8n): {error:.2e}; peak tau {:.4e}", want.iter().cloned().fold(0., f64::max));
        assert!(error < 1e-12);
        let plain = image_sum(&si, &ctx, &shipped.radius, &shipped.smooth_core, 1e-14);
        assert!(relative(density.as_slice(), &plain) < 1e-12);
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(core_table(false, "anything", &[1.]).unwrap().is_none());
    assert!(core_table(true, "smooth_core", &[0., 0.])
        .unwrap()
        .is_none());
    assert!(matches!(
        core_table(true, "re-pseudized", &[1.]),
        Err(Error::Unsupported(_))
    ));
}
