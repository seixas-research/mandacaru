//! A synthetic hydrogen-like PAW species and the molecular integrals and states
//! built on it for the force tests: smooth analytic tables, so no dataset file is
//! needed. It exercises projectors, overlap and compensation augmentation, the
//! signed one-center Coulomb factors and the local potential. Real datasets
//! (partial core, closed-1s core exchange) are covered in `paw_forces_dataset`.
#![allow(dead_code)]
use mandacaru_core::{
    math::{storage::Precision, SpatialGrid},
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    paw::molecular::{build, Arrays},
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};
use std::f64::consts::PI;

pub const SPACING: f64 = 0.5;
pub const CELL: f64 = 14.;
pub const RADIUS: f64 = 1.;

pub fn bessel(l: usize, x: f64) -> f64 {
    match l {
        0 => x.sin() / x,
        _ => x.sin() / (x * x) - x.cos() / x,
    }
}
pub fn smooth_cutoff(r: f64, edge: f64) -> f64 {
    if r >= edge {
        0.
    } else {
        (1. - (r / edge).powi(2)).powi(3)
    }
}
pub fn table(ctx: &Context, arrays: &mut Arrays, values: &[f64]) -> Value {
    let key = format!("a{}", arrays.len());
    arrays.insert(key.clone(), ctx.copy_slice(values).unwrap());
    Value::String(key)
}
/// A hydrogen-like species with s and p channels, two partial waves each.
pub fn species(ctx: &Context, arrays: &mut Arrays, functional: &str) -> Value {
    let r: Vec<f64> = (1..=700).map(|k| 0.02 * k as f64).collect();
    let n = r.len();
    let mut channels = Vec::new();
    for l in 0..2usize {
        let power = |r: f64| r.powi(l as i32);
        let ae: Vec<Vec<f64>> = vec![
            r.iter().map(|&r| 1.5 * power(r) * (-r).exp()).collect(),
            r.iter()
                .map(|&r| 1.1 * power(r) * (-0.6 * r).exp())
                .collect(),
        ];
        let projectors: Vec<Vec<f64>> = (0..2)
            .map(|i| {
                r.iter()
                    .map(|&r| {
                        let x = r / RADIUS;
                        if x >= 1. {
                            0.
                        } else {
                            (1.2 + 0.5 * i as f64 * x) * power(r) * (1. - x * x).powi(2)
                        }
                    })
                    .collect()
            })
            .collect();
        let mut wavevectors = Vec::new();
        let mut coefficients = Vec::new();
        for wave in &ae {
            let q: Vec<f64> = (1..=8).map(|z| PI * z as f64 / RADIUS).collect();
            let points: Vec<usize> = (0..n).filter(|&k| r[k] <= RADIUS && k % 2 == 0).collect();
            let design =
                DMatrix::from_fn(points.len(), q.len(), |a, z| bessel(l, q[z] * r[points[a]]));
            let target = DVector::from_fn(points.len(), |a, _| wave[points[a]]);
            let fit = design
                .svd(true, true)
                .solve(&target, 1e-10)
                .unwrap()
                .as_slice()
                .to_vec();
            wavevectors.push(table(ctx, arrays, &q));
            coefficients.push(table(ctx, arrays, &fit));
        }
        let flat = |rows: &[Vec<f64>]| rows.iter().flatten().copied().collect::<Vec<_>>();
        channels.push(json!({
            "l": l,
            "principal_n": l + 1,
            "radius": RADIUS,
            "occupations": if l == 0 { vec![1., 0.] } else { vec![0., 0.] },
            "ae": table(ctx, arrays, &flat(&ae)),
            "projectors": table(ctx, arrays, &flat(&projectors)),
            "coupling": table(ctx, arrays, &[-0.08, -0.15, -0.15, -0.30]),
            "overlap": table(ctx, arrays, &[0.014, 0.026, 0.026, 0.050]),
            "vanderbilt": table(ctx, arrays, &[-0.063, -0.138, -0.125, -0.276]),
            "wavevectors": wavevectors,
            "wave_coefficients": coefficients,
        }));
    }
    let ionic: Vec<f64> = r.iter().map(|&r| -libm::erf(r / 0.35) / r).collect();
    let valence: Vec<f64> = r.iter().map(|&r| (-2. * r).exp() / PI).collect();
    json!({
        "atomic_number": 1,
        "spin_orbit": false,
        "frozen_subshells": [],
        "semicore_subshells": [],
        "r": table(ctx, arrays, &r),
        "ionic": table(ctx, arrays, &ionic),
        "core": table(ctx, arrays, &vec![0.; n]),
        "valence": table(ctx, arrays, &valence),
        "xc_core": table(ctx, arrays, &vec![0.; n]),
        "charge": 1.0,
        "local_radius": 0.9,
        "compensation_radius": 1.0,
        "functional": functional,
        "one_center_energy": 0.002,
        "relativistic_exchange": false,
        "channels": channels,
    })
}
pub fn integrals(ctx: &Context, positions: &[[f64; 3]], functional: &str) -> MolecularIntegrals {
    integrals_in(ctx, positions, functional, Precision::Float64)
}
pub fn integrals_in(
    ctx: &Context,
    positions: &[[f64; 3]],
    functional: &str,
    precision: Precision,
) -> MolecularIntegrals {
    let geometry = Geometry::new(
        vec![1; positions.len()],
        positions.to_vec(),
        [[CELL, 0., 0.], [0., CELL, 0.], [0., 0., CELL]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [SPACING; 3], false, false).unwrap();
    let mut arrays = Arrays::new();
    let species = species(ctx, &mut arrays, functional);
    let mut basis = Vec::new();
    let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
    let tables: Vec<(usize, Value, Value)> = [0usize, 1]
        .into_iter()
        .map(|l| {
            let values: Vec<f64> = radii
                .iter()
                .map(|&r| r.powi(l as i32) * (-1.1 * r).exp() * smooth_cutoff(r, 7.))
                .collect();
            (
                l,
                table(ctx, &mut arrays, &radii),
                table(ctx, &mut arrays, &values),
            )
        })
        .collect();
    // The s function on every atom shares one table; the p set is added so
    // that the basis can polarize, which the forces need.
    for atom in 0..positions.len() {
        for (l, r, values) in &tables {
            for m in -(*l as i32)..=*l as i32 {
                basis.push(json!({"atom": atom, "l": l, "m": m, "r": r, "values": values}));
            }
        }
    }
    // Radial keys are consumed on first use; later atoms hit the value cache.
    let metadata = json!({
        "format": "molecular-paw-input",
        "version": 1,
        "length_unit": "bohr",
        "energy_unit": "Ha",
        "projector_basis": "dual",
        "atoms": vec![0; positions.len()],
        "species": [species],
        "basis": basis,
    });
    build(
        ctx,
        geometry,
        grid,
        &metadata.to_string(),
        arrays,
        "spectral",
        1e-12,
        precision,
    )
    .unwrap()
}
pub fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-12,
        max_iterations: 400,
        ..Default::default()
    }
}
#[derive(Clone, Copy)]
pub enum Method {
    Hf(&'static str),
    Ks(&'static str),
}
pub fn solve(integrals: &MolecularIntegrals, method: Method, particles: [usize; 2]) -> MeanField {
    let state = match method {
        Method::Hf(kind) => integrals.mean_field(kind, particles, &options()).unwrap(),
        Method::Ks(name) => integrals
            .kohn_sham(name, particles, particles[0] != particles[1], &options())
            .unwrap(),
    };
    assert!(state.converged, "SCF did not converge");
    state
}
pub fn displaced(
    base: &MolecularIntegrals,
    atom: usize,
    axis: usize,
    delta: f64,
) -> MolecularIntegrals {
    let mut positions = base.geometry.positions().to_vec();
    positions[atom][axis] += delta;
    let geometry = Geometry::new(
        base.geometry.numbers().to_vec(),
        positions,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap();
    base.at_geometry(geometry).unwrap()
}
/// Richardson-extrapolated derivative of independently converged total
/// energies on the same fixed grid, in Hartree per Bohr.
pub fn energy_derivative(
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    atom: usize,
    axis: usize,
) -> f64 {
    let energy = |delta: f64| {
        let moved = displaced(base, atom, axis, delta);
        solve(&moved, method, particles).electronic_energy + moved.nuclear
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    (4. * fine - coarse) / 3.
}
pub const H2: [[f64; 3]; 2] = [[0., 0., -0.8], [0., 0., 0.8]];
