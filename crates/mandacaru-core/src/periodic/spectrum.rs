//! Metallic occupations, reciprocal meshes and spectral observables.
use super::invalid;
pub use crate::math::smearing::{fermi, Smearing};
use crate::{finite, product, Buffer, Context, Reservation, Result};
use std::{collections::HashMap, f64::consts::PI};
pub struct Mesh {
    pub full: Vec<[f64; 3]>,
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub mapping: Vec<usize>,
    _memory: Reservation,
}
fn key(v: [f64; 3]) -> [i64; 3] {
    v.map(|x| (((x - x.floor()) * 1e10).round() as i64).rem_euclid(10_000_000_000))
}
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
pub fn mesh(
    ctx: &Context,
    size: [usize; 3],
    gamma: bool,
    operations: &[[i32; 9]],
    time_reversal: bool,
) -> Result<Mesh> {
    let signed: Vec<([i32; 9], bool)> = operations.iter().map(|&op| (op, false)).collect();
    mesh_signed(ctx, size, gamma, &signed, time_reversal)
}

/// Reduces a Monkhorst-Pack mesh by magnetic operations `(W, anti_unitary)` without assuming
/// global time reversal: `anti_unitary = false` maps `k -> k W`, `anti_unitary = true` maps
/// `k -> -k W`.
pub fn magnetic_mesh(
    ctx: &Context,
    size: [usize; 3],
    gamma: bool,
    operations: &[([i32; 9], bool)],
) -> Result<Mesh> {
    mesh_signed(ctx, size, gamma, operations, false)
}

fn mesh_signed(
    ctx: &Context,
    size: [usize; 3],
    gamma: bool,
    operations: &[([i32; 9], bool)],
    time_reversal: bool,
) -> Result<Mesh> {
    if size.contains(&0) {
        return Err(invalid("k-mesh dimensions must be positive"));
    }
    for (operation, _) in operations {
        let a = nalgebra::Matrix3::from_row_slice(&operation.map(|v| v as f64));
        if (a.determinant().abs() - 1.).abs() > 1e-6 {
            return Err(invalid(
                "k-point rotations must be integer unimodular matrices",
            ));
        }
    }
    let n = product(&size)?;
    let memory = ctx.reserve(product(&[n, 192])?)?;
    let mut full = Vec::new();
    full.try_reserve_exact(n)
        .map_err(|_| crate::Error::Allocation)?;
    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                let indices = [i, j, k];
                full.push(std::array::from_fn(|a| {
                    (2. * indices[a] as f64 + 1. - size[a] as f64) / (2. * size[a] as f64)
                        + if gamma && size[a] % 2 == 0 {
                            0.5 / size[a] as f64
                        } else {
                            0.
                        }
                }));
            }
        }
    }
    let lookup: HashMap<_, _> = full.iter().enumerate().map(|(i, p)| (key(*p), i)).collect();
    let mut parent: Vec<_> = (0..n).collect();
    for (i, p) in full.iter().enumerate() {
        for (operation, anti_unitary) in operations {
            let sign = if *anti_unitary { -1. } else { 1. };
            let transformed = std::array::from_fn(|a| {
                sign * (0..3)
                    .map(|b| p[b] * operation[3 * b + a] as f64)
                    .sum::<f64>()
            });
            let &j = lookup.get(&key(transformed)).ok_or_else(|| {
                invalid("a supplied symmetry operation does not preserve the k-mesh")
            })?;
            let a = find(&mut parent, i);
            let b = find(&mut parent, j);
            parent[a.max(b)] = a.min(b);
        }
        if time_reversal {
            let &j = lookup
                .get(&key(p.map(|x| -x)))
                .ok_or_else(|| invalid("k-mesh is not time-reversal invariant"))?;
            let a = find(&mut parent, i);
            let b = find(&mut parent, j);
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut representatives = HashMap::new();
    let mut points = vec![];
    let mut counts = vec![];
    let mut mapping = vec![];
    for i in 0..n {
        let root = find(&mut parent, i);
        let next = points.len();
        let index = *representatives.entry(root).or_insert_with(|| {
            points.push(full[root]);
            counts.push(0usize);
            next
        });
        counts[index] += 1;
        mapping.push(index);
    }
    let weights = counts.iter().map(|v| *v as f64 / n as f64).collect();
    Ok(Mesh {
        full,
        points,
        weights,
        mapping,
        _memory: memory,
    })
}
pub fn broaden(
    ctx: &Context,
    energies: &[f64],
    eps: &[Vec<f64>],
    weights: &[f64],
    width: f64,
    degeneracy: f64,
    projections: Option<(&[f64], usize)>,
) -> Result<Buffer<f64>> {
    finite(energies, "DOS energy grid")?;
    if !width.is_finite()
        || width <= 0.
        || !degeneracy.is_finite()
        || degeneracy <= 0.
        || eps.is_empty()
        || eps.len() != weights.len()
    {
        return Err(invalid("invalid DOS dimensions or width"));
    }
    let bands = eps[0].len();
    if bands == 0
        || eps
            .iter()
            .any(|e| e.len() != bands || e.iter().any(|v| !v.is_finite()))
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
    {
        return Err(invalid("invalid DOS bands or weights"));
    }
    let (projection, columns) = projections.unwrap_or((&[], 1));
    if columns == 0
        || (projections.is_some() && projection.len() != product(&[eps.len(), columns, bands])?)
        || projection.iter().any(|p| !p.is_finite() || *p < 0.)
    {
        return Err(invalid("invalid projected DOS weights"));
    }
    let mut out = ctx.allocate::<f64>(product(&[energies.len(), columns])?)?;
    for (k, levels) in eps.iter().enumerate() {
        for (b, &level) in levels.iter().enumerate() {
            for (i, &energy) in energies.iter().enumerate() {
                let peak = degeneracy * weights[k] / (width * (2. * PI).sqrt())
                    * (-0.5 * ((energy - level) / width).powi(2)).exp();
                for column in 0..columns {
                    out.as_mut_slice()[i * columns + column] += peak
                        * if projections.is_some() {
                            projection[(k * columns + column) * bands + b]
                        } else {
                            1.
                        };
                }
            }
        }
    }
    Ok(out)
}

/// The occupations `degeneracy f((e - mu) / width)` of every eigenvalue and the entropy term
/// `degeneracy sum_k w_k sum_n s((e_kn - mu) / width)` at the Fermi level `mu`.
pub fn occupations_and_entropy(
    eigenvalues: &[Vec<f64>],
    weights: &[f64],
    mu: f64,
    method: Smearing,
    width: f64,
    degeneracy: f64,
) -> (Vec<Vec<f64>>, f64) {
    let occupations = eigenvalues
        .iter()
        .map(|e| {
            e.iter()
                .map(|e| degeneracy * method.occupation((e - mu) / width))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let entropy = degeneracy
        * eigenvalues
            .iter()
            .zip(weights)
            .map(|(e, w)| {
                w * e
                    .iter()
                    .map(|e| method.entropy((e - mu) / width))
                    .sum::<f64>()
            })
            .sum::<f64>();
    (occupations, entropy)
}
