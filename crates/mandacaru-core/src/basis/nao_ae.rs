//! All-electron numerical species: confined LDA reference, radial tiers and
//! three-pass channel Gram–Schmidt. No precomputed atomic orbital tables.
use super::{invalid, radial::Table};
use crate::models::atom::{self, AtomOptions};
use crate::{kernels, Buffer, Context, Result};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};
struct Function {
    l: usize,
    u: Buffer<f64>,
    energy: f64,
    minimal: bool,
}
fn trap(values: impl Iterator<Item = f64>, n: usize, h: f64) -> f64 {
    values
        .enumerate()
        .map(|(i, v)| v * if i == 0 || i == n - 1 { 0.5 } else { 1. })
        .sum::<f64>()
        * h
}
fn wall(r: f64, onset: f64, width: f64, scale: f64) -> f64 {
    if r <= onset {
        0.
    } else if r >= onset + width {
        1e5
    } else {
        (scale * (-width / (r - onset)).exp() / (onset + width - r).powi(2)).min(1e5)
    }
}
fn radial_state(
    ctx: &Context,
    r: &[f64],
    potential: &[f64],
    l: usize,
    n: usize,
    radius: f64,
) -> Result<(Buffer<f64>, f64)> {
    let keep = r.partition_point(|&v| v < radius);
    let h = r[1] - r[0];
    if keep < 3 || n <= l || n - l > keep || l > 8 {
        return Err(invalid(
            "radial state is not resolved inside the NAO-AE wall",
        ));
    }
    let mut diag = ctx.allocate::<f64>(keep)?;
    let mut off = ctx.allocate::<f64>(keep - 1)?;
    off.as_mut_slice().fill(-0.5 / (h * h));
    for i in 0..keep {
        diag.as_mut_slice()[i] =
            1. / (h * h) + potential[i] + (l * (l + 1)) as f64 / (2. * r[i] * r[i]);
    }
    let (energy, u) = kernels::eigenpair(ctx, diag.as_slice(), off.as_slice(), n - l - 1)?;
    let norm = trap(u.as_slice().iter().map(|v| v * v), keep, h)
        .sqrt()
        .copysign(u.as_slice()[0]);
    let mut full = ctx.allocate::<f64>(r.len())?;
    for i in 0..keep {
        full.as_mut_slice()[i] = u.as_slice()[i] / norm;
    }
    Ok((full, energy))
}
#[allow(clippy::too_many_arguments)]
fn hydrogenic(
    ctx: &Context,
    r: &[f64],
    n: usize,
    l: usize,
    z: f64,
    onset: f64,
    width: f64,
    scale: f64,
    tail: Option<f64>,
) -> Result<Function> {
    if !z.is_finite() || z <= 0. {
        return Err(invalid(
            "NAO-AE extra effective charges must be positive and finite",
        ));
    }
    let h = r[1] - r[0];
    let mut potential = ctx.allocate::<f64>(r.len())?;
    let mut solve = |r0| {
        for (i, &radius) in r.iter().enumerate() {
            potential.as_mut_slice()[i] = -z / radius + wall(radius, r0, width, scale);
        }
        radial_state(ctx, r, potential.as_slice(), l, n, r0 + width)
    };
    let (mut u, mut energy) = solve(onset)?;
    if let Some(tail) = tail.filter(|t| *t > 0.) {
        let norm = trap(u.as_slice().iter().map(|v| v * v), r.len(), h);
        let mut inner = 0.;
        let mut tail_radius = *r.last().unwrap();
        for (i, &radius) in r.iter().enumerate() {
            if i > 0 {
                inner += 0.5 * (u.as_slice()[i - 1].powi(2) + u.as_slice()[i].powi(2)) * h;
            }
            if 1. - inner / norm < tail {
                tail_radius = radius;
                break;
            }
        }
        if tail_radius < onset {
            (u, energy) = solve(tail_radius)?;
        }
    }
    Ok(Function {
        l,
        u,
        energy,
        minimal: false,
    })
}
fn real(spec: &Value, name: &str) -> Result<f64> {
    spec[name]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("NAO-AE {name} must be finite")))
}
pub fn build(ctx: &Context, z: u8, spec: &Value) -> Result<Vec<(usize, Arc<Table>)>> {
    let points = spec["points"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("NAO-AE points must be an integer"))?;
    let r_max = real(spec, "r_max")?;
    let onset = real(spec, "onset")?;
    let width = real(spec, "width")?;
    let scale = real(spec, "scale")?;
    let end = real(spec, "wall_radius")?;
    let tolerance = real(spec, "linear_dependence_tol")?;
    let tier = spec["tier"]
        .as_u64()
        .filter(|v| *v <= 2)
        .ok_or_else(|| invalid("NAO-AE tier must be 0, 1 or 2"))?;
    let tail = if spec["tail_norm"].is_null() {
        None
    } else {
        Some(real(spec, "tail_norm")?)
    };
    if points < 4
        || points > i32::MAX as usize
        || r_max <= 0.
        || onset < 0.
        || width <= 0.
        || scale < 0.
        || onset + width > r_max
        || end <= 0.
        || end > r_max
        || tolerance <= 0.
        || tail.is_some_and(|v| !(0. ..1.).contains(&v))
    {
        return Err(invalid("invalid NAO-AE radial grid, wall or norm options"));
    }
    let h = r_max / (points as f64 + 1.);
    let mut confinement = ctx.allocate::<f64>(points)?;
    for i in 0..points {
        confinement.as_mut_slice()[i] = wall((i + 1) as f64 * h, onset, width, scale);
    }
    let atom = atom::solve(
        ctx,
        z,
        &AtomOptions {
            points,
            r_max,
            max_iterations: 200,
            tolerance: 1e-6,
            mixing: 0.3,
        },
        None,
        Some(confinement.as_slice()),
    )?;
    if !atom.converged {
        return Err(invalid(
            "the native confined LDA reference atom did not converge",
        ));
    }
    let r = atom.r.as_slice();
    let mut potential = ctx.allocate::<f64>(points)?;
    for i in 0..points {
        potential.as_mut_slice()[i] = atom.effective.as_slice()[i] + confinement.as_slice()[i];
    }
    let mut functions = vec![];
    for &(n, l, _) in &atom.occupations {
        let (u, energy) = radial_state(ctx, r, potential.as_slice(), l, n, onset + width)?;
        functions.push(Function {
            l,
            u,
            energy,
            minimal: true,
        });
    }
    let outer = atom.occupations.iter().map(|s| s.0).max().unwrap();
    let mut valence = BTreeMap::new();
    for (i, &(n, l, _)) in atom.occupations.iter().enumerate() {
        if n == outer || (l == 2 && n + 1 == outer) || (l == 3 && n + 2 == outer) {
            let u = atom.orbitals[i].as_slice();
            let radius = trap(u.iter().zip(r).map(|(u, r)| u * u * r), points, h)
                / trap(u.iter().map(|u| u * u), points, h);
            valence.insert(l, (n, radius));
        }
    }
    let l_max = *valence.keys().last().unwrap();
    let outer_radius = valence.values().map(|s| s.1).fold(0f64, f64::max);
    let mut extras = vec![];
    if tier >= 1 {
        extras.push((l_max + 2, l_max + 1, outer_radius));
        for (&l, &(n, radius)) in &valence {
            extras.push((n + 1, l, 2. * radius));
        }
    }
    if tier >= 2 {
        extras.push((l_max + 3, l_max + 2, outer_radius));
        extras.push((l_max + 3, l_max + 1, 1.5 * outer_radius));
        for (&l, &(n, radius)) in &valence {
            extras.push((n, l, 0.6 * radius));
        }
    }
    for (n, l, radius) in extras {
        let charge = (3. * (n * n) as f64 - (l * (l + 1)) as f64) / (2. * radius);
        functions.push(hydrogenic(ctx, r, n, l, charge, onset, width, scale, tail)?);
    }
    if let Some(extra) = spec.get("extra").filter(|v| !v.is_null()) {
        for item in extra
            .as_array()
            .ok_or_else(|| invalid("NAO-AE extra must be triples"))?
        {
            let values = item
                .as_array()
                .filter(|v| v.len() == 3)
                .ok_or_else(|| invalid("NAO-AE extra must be triples"))?;
            let n = values[0]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid("invalid extra principal number"))?;
            let l = values[1]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid("invalid extra angular number"))?;
            let charge = values[2]
                .as_f64()
                .ok_or_else(|| invalid("invalid extra charge"))?;
            functions.push(hydrogenic(ctx, r, n, l, charge, onset, width, scale, tail)?);
        }
    }
    functions.sort_by_key(|f| {
        (
            !f.minimal,
            if f.minimal {
                0
            } else {
                f.u.as_slice()
                    .iter()
                    .rposition(|v| v.abs() > 1e-6)
                    .map_or(0, |i| i + 1)
            },
        )
    });
    let mut accepted: Vec<Function> = vec![];
    for mut f in functions {
        let mut dropped = false;
        for pass in 0..3 {
            for g in accepted.iter().filter(|g| g.l == f.l) {
                let overlap = trap(
                    g.u.as_slice()
                        .iter()
                        .zip(f.u.as_slice())
                        .map(|(a, b)| a * b),
                    points,
                    h,
                );
                for (u, g) in f.u.as_mut_slice().iter_mut().zip(g.u.as_slice()) {
                    *u -= overlap * g;
                }
            }
            let norm2 = trap(f.u.as_slice().iter().map(|v| v * v), points, h);
            if pass == 0 && norm2 < tolerance {
                if f.minimal {
                    return Err(invalid("minimal NAO-AE function is linearly dependent"));
                }
                dropped = true;
                break;
            }
            if !norm2.is_finite() || norm2 <= 0. {
                return Err(invalid("invalid NAO-AE residual norm"));
            }
            for value in f.u.as_mut_slice() {
                *value /= norm2.sqrt();
            }
        }
        if !dropped {
            let pivot =
                f.u.as_slice()
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .unwrap()
                    .0;
            if f.u.as_slice()[pivot] < 0. {
                for value in f.u.as_mut_slice() {
                    *value = -*value;
                }
            }
            accepted.push(f);
        }
    }
    let keep = r.partition_point(|&r| r < end);
    if keep < 2 {
        return Err(invalid("NAO-AE wall has too few grid points"));
    }
    let mut tables = vec![];
    for f in accepted {
        let mut values = ctx.allocate::<f64>(keep + 2)?;
        for (i, r) in r.iter().enumerate().take(keep) {
            values.as_mut_slice()[i + 1] = f.u.as_slice()[i] / r;
        }
        if f.l == 0 {
            values.as_mut_slice()[0] = 2. * values.as_slice()[1] - values.as_slice()[2];
        }
        tables.push((
            f.l,
            Arc::new(Table::with_radius(ctx, h, values, f.energy, f.l, end)?),
        ));
    }
    Ok(tables)
}
