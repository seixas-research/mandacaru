use mandacaru_core::{
    math::fft::Fft3,
    periodic::{fields, Crystal, Options},
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;

fn crystal(context: &Context) -> Crystal {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5/si_crystal.json");
    Crystal::load(context, path.to_str().unwrap()).unwrap()
}

#[test]
fn periodic_meta_and_nonlocal_potentials_are_derivatives_of_the_same_energy() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let mut c = crystal(&ctx);
        let n = c.grid.index.points();
        c.core_tau = Some(ctx.allocate::<f64>(n).unwrap());
        let rho: Vec<_> = (0..n)
            .map(|k| 0.03 * (1. + 0.1 * (k as f64 * 0.03).sin()))
            .collect();
        let tau: Vec<_> = (0..n)
            .map(|k| 0.06 * (1. + 0.03 * (k as f64 * 0.05).cos()))
            .collect();
        let direction: Vec<_> = (0..n).map(|k| 0.01 * (k as f64 * 0.07).sin()).collect();
        for name in ["r2scan", "pbe+rvv10", "r2scan+rvv10"] {
            c.functional = name.into();
            let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
            let terms = fields::evaluate(&c, &mut fft, &rho, Some(&tau)).unwrap();
            for varying_tau in [false, true] {
                if varying_tau && terms.kinetic_potential.is_none() {
                    continue;
                }
                let h = 1e-5;
                let shifted = |sign: f64| {
                    let moved: Vec<_> = (if varying_tau { &tau } else { &rho })
                        .iter()
                        .zip(&direction)
                        .map(|(v, d)| v + sign * h * d)
                        .collect();
                    let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                    fields::evaluate(
                        &c,
                        &mut fft,
                        if varying_tau { &rho } else { &moved },
                        Some(if varying_tau { &moved } else { &tau }),
                    )
                    .unwrap()
                    .energy
                };
                let derivative = (shifted(1.) - shifted(-1.)) / (2. * h);
                let potential = if varying_tau {
                    terms.kinetic_potential.as_ref().unwrap()
                } else {
                    &terms.potential
                };
                let contraction = potential
                    .as_slice()
                    .iter()
                    .zip(&direction)
                    .map(|(v, d)| v * d)
                    .sum::<f64>()
                    * c.grid.index.dv();
                assert!(
                    (derivative - contraction).abs() < 2e-8,
                    "{name}, tau={varying_tau}: {derivative} vs {contraction}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn bloch_kinetic_matrix_contracts_the_actual_complex_orbital_density() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let mut c = crystal(&ctx);
        c.permutations.clear();
        let n = c.grid.index.points();
        let potential: Vec<_> = (0..n)
            .map(|k| 0.3 * (1. + 0.2 * (k as f64 * 0.03).cos()))
            .collect();
        let matrices: Vec<_> = c
            .points
            .iter()
            .map(|_| {
                DMatrix::from_fn(c.m, c.m, |a, b| {
                    C64::new(
                        if a == b { 1. } else { 0.01 },
                        0.001 * (a as f64 - b as f64),
                    )
                })
            })
            .collect();
        let tau = fields::kinetic_density(&c, &matrices).unwrap();
        let direct = tau
            .as_slice()
            .iter()
            .zip(&potential)
            .map(|(t, v)| t * v)
            .sum::<f64>()
            * c.grid.index.dv();
        let mut projected = 0.;
        for (k, point) in c.points.iter().enumerate() {
            let matrix = fields::kinetic_operator(&c, point, &potential).unwrap();
            for a in 0..c.m {
                for b in 0..c.m {
                    projected +=
                        point.weight * (matrices[k][(a, b)] * matrix.as_slice()[b + a * c.m]).re;
                }
            }
        }
        assert!(
            (direct - projected).abs() < 1e-10,
            "{direct} vs {projected}"
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn periodic_meta_scf_updates_its_kinetic_density_and_roundtrips_core_input() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    {
        let mut c = crystal(&ctx);
        c.functional = "r2scan".into();
        c.core_tau = Some(ctx.allocate::<f64>(c.grid.index.points()).unwrap());
        let result = c
            .solve(&Options {
                max_iterations: 300,
                ..Default::default()
            })
            .unwrap();
        assert!(
            result.converged,
            "meta-SCF did not converge after {} iterations; density {}, kinetic {}",
            result.history.len(),
            result.history.last().unwrap().residual,
            result.history.last().unwrap().kinetic_residual
        );
        let rebuilt = fields::kinetic_density(&c, &result.matrices).unwrap();
        assert_eq!(
            rebuilt.as_slice(),
            result.kinetic_density.as_ref().unwrap().as_slice()
        );
        assert!(result
            .kinetic_potential
            .as_ref()
            .unwrap()
            .as_slice()
            .iter()
            .any(|v| v.abs() > 1e-6));
        let path =
            std::env::temp_dir().join(format!("mandacaru-meta-input-{}.json", std::process::id()));
        c.save(path.to_str().unwrap()).unwrap();
        let loaded = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded.functional, "r2scan");
        assert_eq!(
            loaded.core_tau.as_ref().unwrap().as_slice(),
            c.core_tau.as_ref().unwrap().as_slice()
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
