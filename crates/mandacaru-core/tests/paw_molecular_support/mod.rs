//! Builders shared by the `paw_molecular_*` test binaries: a synthetic
//! hydrogen-like PAW species (smooth analytic tables, so no dataset file is
//! needed) and a molecule built from any species JSON with a small analytic
//! atomic-orbital basis. [`dataset`] turns a dataset of the shipped library into
//! such a species.
#![allow(dead_code)]

pub mod dataset;

use mandacaru_core::{
    math::{fft::Poisson, storage::Precision, SpatialGrid},
    molecular::MolecularIntegrals,
    paw::molecular::{build, Arrays},
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};
use std::f64::consts::PI;

pub const CELL: f64 = 12.;
pub const RADIUS: f64 = 1.;
pub const H2: [[f64; 3]; 2] = [[0., 0., -0.8], [0., 0., 0.8]];

fn bessel(l: usize, x: f64) -> f64 {
    match l {
        0 => x.sin() / x,
        _ => x.sin() / (x * x) - x.cos() / x,
    }
}
pub fn table(ctx: &Context, arrays: &mut Arrays, values: &[f64]) -> Value {
    let key = format!("a{}", arrays.len());
    arrays.insert(key.clone(), ctx.copy_slice(values).unwrap());
    Value::String(key)
}
/// A hydrogen-like species with s and p channels and two partial waves each,
/// from smooth analytic tables.
pub fn species(ctx: &Context, arrays: &mut Arrays) -> Value {
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
            "raw_projectors": table(ctx, arrays, &flat(&projectors)),
            "coupling": table(ctx, arrays, &[-0.08, -0.15, -0.15, -0.30]),
            "overlap": table(ctx, arrays, &[0.014, 0.026, 0.026, 0.050]),
            "vanderbilt": table(ctx, arrays, &[1.0, 0.2, 0.2, 0.8]),
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
        "functional": "pbe",
        "one_center_energy": 0.002,
        "relativistic_exchange": false,
        "channels": channels,
    })
}

/// A radial table `(l, r, R(r))` of an atomic-orbital shell.
pub type Shell = (usize, Vec<f64>, Vec<f64>);

/// The analytic shells `r^l exp(-1.1 r)`, cut smoothly at 7 Bohr, for each `l`.
pub fn analytic_shells(shells: &[usize]) -> Vec<Shell> {
    let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
    shells
        .iter()
        .map(|&l| {
            let values: Vec<f64> = radii
                .iter()
                .map(|&r| {
                    let cut = if r >= 7. {
                        0.
                    } else {
                        (1. - (r / 7.).powi(2)).powi(3)
                    };
                    r.powi(l as i32) * (-1.1 * r).exp() * cut
                })
                .collect();
            (l, radii.clone(), values)
        })
        .collect()
}

/// A molecule of `numbers` at `positions` (Bohr) in a cubic cell of `cell`, with
/// one species JSON per entry of `species` (`atoms` maps each atom to one) and
/// the radial `shells` on every atom.
pub fn molecule(
    ctx: &Context,
    numbers: &[u8],
    positions: &[[f64; 3]],
    cell: f64,
    spacing: f64,
    (species, atoms, mut arrays): (Vec<Value>, Vec<usize>, Arrays),
    basis_kind: &str,
    shells: &[Shell],
) -> MolecularIntegrals {
    let geometry = Geometry::new(
        numbers.to_vec(),
        positions.to_vec(),
        [[cell, 0., 0.], [0., cell, 0.], [0., 0., cell]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [spacing; 3], false, false).unwrap();
    let tables: Vec<(usize, Value, Value)> = shells
        .iter()
        .map(|(l, radii, values)| {
            (
                *l,
                table(ctx, &mut arrays, radii),
                table(ctx, &mut arrays, values),
            )
        })
        .collect();
    let mut basis = Vec::new();
    for atom in 0..positions.len() {
        for (l, r, values) in &tables {
            for m in -(*l as i32)..=*l as i32 {
                basis.push(json!({"atom": atom, "l": l, "m": m, "r": r, "values": values}));
            }
        }
    }
    let metadata = json!({
        "format": "molecular-paw-input",
        "version": 1,
        "length_unit": "bohr",
        "energy_unit": "Ha",
        "projector_basis": basis_kind,
        "atoms": atoms,
        "species": species,
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
        Precision::Float64,
    )
    .unwrap()
}

/// The synthetic hydrogen-like species at `positions`.
pub fn integrals(ctx: &Context, positions: &[[f64; 3]], basis_kind: &str) -> MolecularIntegrals {
    let mut arrays = Arrays::new();
    let species = species(ctx, &mut arrays);
    molecule(
        ctx,
        &vec![1; positions.len()],
        positions,
        CELL,
        0.5,
        (vec![species], vec![0; positions.len()], arrays),
        basis_kind,
        &analytic_shells(&[0, 1]),
    )
}

/// `(ab|cd)` of the smooth pair densities on the grid under `erfc(omega r)/r`.
pub fn grid_pairs(ctx: &Context, base: &MolecularIntegrals, omega: f64) -> Vec<f64> {
    let (m, g) = (base.m(), base.grid.index.points());
    let dv = base.grid.index.dv();
    let mut orbitals = vec![0.; m * g];
    for p in 0..m {
        base.orthonormal_samples
            .widen_range(p * g, &mut orbitals[p * g..(p + 1) * g])
            .unwrap();
    }
    let mut solver = Poisson::screened(ctx, &base.grid, omega).unwrap();
    let mut out = vec![0.; m * m * m * m];
    let mut source = vec![C64::new(0., 0.); g];
    let mut potential = vec![C64::new(0., 0.); g];
    for a in 0..m {
        for b in a..m {
            for k in 0..g {
                source[k] = C64::new(orbitals[a * g + k] * orbitals[b * g + k], 0.);
            }
            solver.solve_into(&source, &mut potential).unwrap();
            for c in 0..m {
                for d in 0..m {
                    let value: f64 = (0..g)
                        .map(|k| orbitals[c * g + k] * orbitals[d * g + k] * potential[k].re)
                        .sum::<f64>()
                        * dv;
                    out[((a * m + b) * m + c) * m + d] = value;
                    out[((b * m + a) * m + c) * m + d] = value;
                }
            }
        }
    }
    out
}
