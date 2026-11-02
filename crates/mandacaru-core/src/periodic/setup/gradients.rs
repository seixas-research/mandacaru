//! Derivatives of the crystal's lattice sums with respect to the atoms (the explicit terms E1-E4
//! of the crystal force): the erfc pair sum of the Gaussian ions (E4), the short-range ion
//! potential on the compensation charges (E3, `linear`), the dense reciprocal sums `U`, `U_ion`
//! and `E_ion` (E3, E2) and the contractions of the moving hats and ions with a density on the
//! grid (E1, E2).
//!
//! A compensation channel moves with its owner, so every position dependence enters through the
//! atom phases `exp(-i G.R)` (reciprocal sums) or through the owner-to-other distance
//! (real-space sums). Derivatives are `d / d R_atom^axis` per Bohr, never forces.
use super::{
    electrostatics::coulomb,
    ions::{LocalPotential, ION_PAIR_REACH, SPHERE_AZIMUTHAL, SPHERE_POLAR, SPHERE_RADIAL},
    shapes::{visit_lattice, Shapes, Sites, Transient},
};
use crate::{
    math::{angular, reciprocal, SpatialGrid},
    paw::{
        local_split::{angular_rule, radial_rule, short_range_potential_derivative},
        multipole::shape_function,
    },
    periodic::invalid,
    Context, Result,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use std::f64::consts::PI;

const ZERO: Complex64 = Complex64::new(0., 0.);
/// `-i`, the factor a phase `exp(-i G.R)` brings down when it is differentiated along `R`.
const MINUS_I: Complex64 = Complex64::new(0., -1.);

/// `dE/dR_A` of [`super::ions::ion_constants`], analytic. A pair term `1/2 Z_A Z_B f(d)`,
/// `f = erfc(d / 2 sigma) / d`, `d = R_B + R - R_A`, pulls `A` by `-f' d^` and `B` by `+f' d^`;
/// the pairs inside one atom's own images cancel.
pub fn ion_constants_gradient(sites: &Sites) -> Result<Vec<[f64; 3]>> {
    sites.check()?;
    let sigma = sites.sigma;
    let reach = ION_PAIR_REACH * sigma;
    let mut out = vec![[0.; 3]; sites.atoms.len()];
    for (a, first) in sites.atoms.iter().enumerate() {
        for (b, second) in sites.atoms.iter().enumerate() {
            if a == b {
                continue;
            }
            let d0 = Vector3::from(second.center) - Vector3::from(first.center);
            let mut pull = Vector3::zeros();
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let vector = d0 + Vector3::from(*image);
                        let d = vector.norm();
                        if d < 1e-10 || d > reach {
                            continue;
                        }
                        let x = d / (2. * sigma);
                        let slope =
                            -(libm::erfc(x) / (d * d) + (-x * x).exp() / (PI.sqrt() * sigma * d));
                        pull += vector * (slope / d);
                    }
                    Ok(())
                },
            )?;
            pull *= 0.5 * first.charge * second.charge;
            for axis in 0..3 {
                out[a][axis] -= pull[axis];
                out[b][axis] += pull[axis];
            }
        }
    }
    Ok(out)
}

/// `dE/d eps_ab` of [`super::ions::ion_constants`] under the strain `x -> (1 + eps) x` of the
/// lattice and the atoms at a fixed Gaussian width (the pinned discretization of the stress):
/// `1/2 sum Z_A Z_B f'(d) v_a v_b / d` over the pair vectors `v` of the unstrained cell. The
/// self term does not depend on the cell. Symmetric; an independent analytic cross-check of a
/// differenced stress, not a stress route of its own.
pub fn ion_constants_strain(sites: &Sites) -> Result<[[f64; 3]; 3]> {
    sites.check()?;
    let sigma = sites.sigma;
    let reach = ION_PAIR_REACH * sigma;
    let mut out = Matrix3::zeros();
    for first in sites.atoms {
        for second in sites.atoms {
            let d0 = Vector3::from(second.center) - Vector3::from(first.center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let vector = d0 + Vector3::from(*image);
                        let d = vector.norm();
                        if d < 1e-10 || d > reach {
                            continue;
                        }
                        let x = d / (2. * sigma);
                        let slope =
                            -(libm::erfc(x) / (d * d) + (-x * x).exp() / (PI.sqrt() * sigma * d));
                        out += vector
                            * vector.transpose()
                            * (0.5 * first.charge * second.charge * slope / d);
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(std::array::from_fn(|a| {
        std::array::from_fn(|b| out[(a, b)])
    }))
}

/// `d / d R_atom^axis` of [`super::ions::short_range_ion_compensation`]: `out[channel][atom][axis]`.
/// A shape moves with its owner (the quadrature rigidly) and sees every other atom's image
/// through the slope of `v^sr`: the owner gets `+sum w g v' x^`, the atom it looks at the
/// opposite; an atom's own images cancel.
pub fn short_range_ion_compensation_gradient(
    ctx: &Context,
    sites: &Sites,
    potentials: &[LocalPotential],
) -> Result<Vec<Vec<[Complex64; 3]>>> {
    sites.check()?;
    let atoms = sites.atoms.len();
    if potentials.len() != atoms {
        return Err(invalid("one local potential per atom"));
    }
    let sigma = sites.sigma;
    let reaches = potentials
        .iter()
        .map(|p| p.reach(sigma))
        .collect::<Result<Vec<_>>>()?;
    let mut out = vec![vec![[ZERO; 3]; atoms]; sites.channels.len()];
    let (axes, angular_weights) = angular_rule(SPHERE_POLAR, SPHERE_AZIMUTHAL)?;
    let directions: Vec<[f64; 3]> = (0..angular_weights.len())
        .map(|j| [axes[0][j], axes[1][j], axes[2][j]])
        .collect();
    let harmonics = |l: usize, m: i32| -> Vec<Complex64> {
        directions
            .iter()
            .map(|d| angular::complex(l, m, *d))
            .collect()
    };
    for owner in 0..atoms {
        let rows: Vec<usize> = (0..out.len())
            .filter(|&c| sites.channels[c].0 == owner)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let radius = sites.atoms[owner].compensation_radius;
        let (r, radial_weights) = radial_rule(radius, 0., SPHERE_RADIAL)?;
        let points = r.len() * directions.len();
        let _scratch = ctx.reserve(crate::product(&[points, 8, 8])?)?;
        let center = Vector3::from(sites.atoms[owner].center);
        let at = |p: usize| {
            center + Vector3::from(directions[p % directions.len()]) * r[p / directions.len()]
        };
        for other in (0..atoms).filter(|&o| o != owner) {
            let reach = reaches[other] + radius;
            let d0 = center - Vector3::from(sites.atoms[other].center);
            // sum over images of (v^sr)' (x - R_other) / |x - R_other|, per point
            let mut scale = vec![Vector3::zeros(); points];
            let mut distance = vec![0.; points];
            let mut sampled = vec![0.; points];
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let shift = Vector3::from(*image);
                        if (d0 - shift).norm() > reach {
                            continue;
                        }
                        let origin = Vector3::from(sites.atoms[other].center) + shift;
                        for p in 0..points {
                            distance[p] = (at(p) - origin).norm();
                            sampled[p] = potentials[other].slope(distance[p]);
                        }
                        let slope = short_range_potential_derivative(
                            &sampled,
                            potentials[other].charge,
                            sigma,
                            &distance,
                        )?;
                        for p in 0..points {
                            scale[p] += (at(p) - origin) * (slope[p] / distance[p].max(1e-300));
                        }
                    }
                    Ok(())
                },
            )?;
            for &row in &rows {
                let (_, l, m) = sites.channels[row];
                let shape = shape_function(&r, radius, l)?;
                let harmonic = harmonics(l, m);
                let mut pull = [ZERO; 3];
                for (i, &weight) in radial_weights.iter().enumerate() {
                    for (j, y) in harmonic.iter().enumerate() {
                        let w = *y * (weight * angular_weights[j] * shape[i]);
                        let s = scale[i * directions.len() + j];
                        for axis in 0..3 {
                            pull[axis] += w * s[axis];
                        }
                    }
                }
                for axis in 0..3 {
                    out[row][owner][axis] += pull[axis];
                    out[row][other][axis] -= pull[axis];
                }
            }
        }
    }
    Ok(out)
}

/// The derivative of the dense sums with respect to one coordinate of one atom: `dU`
/// (`channels x channels`, row-major), `dU_ion` per channel and `dE_ion`.
#[derive(Clone, Debug)]
pub struct DenseGradient {
    pub u: Vec<Complex64>,
    pub u_ion: Vec<Complex64>,
    pub e_ion: f64,
}

/// [`DenseGradient`] of every atom and axis.
#[derive(Debug)]
pub struct DenseGradients {
    items: Vec<DenseGradient>,
}
impl DenseGradients {
    pub fn get(&self, atom: usize, axis: usize) -> &DenseGradient {
        &self.items[atom * 3 + axis]
    }
}

/// The derivative of [`super::electrostatics::couplings`] (same cutoff, same `G` set): with
/// `g_a(G) ~ exp(-i G.R_owner(a))`, `dU[a,b] = (1/V) sum K (i G) g_a^* g_b (delta_aA - delta_bA)`,
/// `dU_ion[a]` the same against the ions with the ion of `A` differentiated, and
/// `dE_ion = (1/V) sum K Re[ion^* (-i G) ion_A]`. The on-site tail beyond the cutoff does not move.
pub fn dense_gradients(
    ctx: &Context,
    shapes: &Shapes,
    sites: &Sites,
    cutoff: f64,
) -> Result<DenseGradients> {
    sites.check()?;
    let (rows, atoms) = (sites.channels.len(), sites.atoms.len());
    let volume = sites.volume();
    let sigma2 = sites.sigma * sites.sigma;
    let block = (Transient::of(ctx).dense_bytes / ((rows + atoms + 2) * 16 + 64)).max(1);
    let _scratch = ctx.reserve(crate::product(&[block, (rows + atoms + 2) * 16 + 64])?)?;
    let mut f = vec![ZERO; rows * block];
    let mut ion_of = vec![ZERO; atoms * block];
    let mut mx = vec![vec![ZERO; rows * rows]; 3];
    let mut px = vec![vec![ZERO; rows]; 3];
    let mut qx = vec![vec![ZERO; rows * atoms]; 3];
    let mut ex = vec![vec![0.; atoms]; 3];
    visit_lattice(&sites.reciprocal()?, cutoff, true, block, &mut |vectors| {
        let m = vectors.len();
        shapes.transforms(sites, vectors, &mut f[..rows * m], m)?;
        let mut k = vec![0.; m];
        let mut ion = vec![ZERO; m];
        for (g, v) in vectors.iter().enumerate() {
            k[g] = coulomb(*v);
            let envelope = (-0.5 * sigma2 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])).exp();
            for (a, atom) in sites.atoms.iter().enumerate() {
                let angle =
                    -(v[0] * atom.center[0] + v[1] * atom.center[1] + v[2] * atom.center[2]);
                let value = Complex64::new(angle.cos(), angle.sin()) * (-atom.charge * envelope);
                ion_of[a * m + g] = value;
                ion[g] += value;
            }
        }
        for axis in 0..3 {
            // `-i G_axis` of every vector: the derivative of a phase `exp(-i G.R)`.
            let down: Vec<Complex64> = vectors.iter().map(|v| MINUS_I * v[axis]).collect();
            for a in 0..rows {
                let weighted: Vec<Complex64> = (0..m).map(|g| f[a * m + g].conj() * k[g]).collect();
                let raised: Vec<Complex64> = (0..m).map(|g| weighted[g] * -down[g]).collect();
                for b in 0..rows {
                    mx[axis][a * rows + b] +=
                        (0..m).map(|g| raised[g] * f[b * m + g]).sum::<Complex64>();
                }
                px[axis][a] += (0..m).map(|g| raised[g] * ion[g]).sum::<Complex64>();
                for atom in 0..atoms {
                    qx[axis][a * atoms + atom] += (0..m)
                        .map(|g| weighted[g] * down[g] * ion_of[atom * m + g])
                        .sum::<Complex64>();
                }
            }
            for atom in 0..atoms {
                ex[axis][atom] += (0..m)
                    .map(|g| k[g] * (ion[g].conj() * down[g] * ion_of[atom * m + g]).re)
                    .sum::<f64>();
            }
        }
        Ok(())
    })?;
    let mut items = Vec::with_capacity(atoms * 3);
    for atom in 0..atoms {
        for axis in 0..3 {
            let owns = |channel: usize| f64::from(u8::from(sites.channels[channel].0 == atom));
            let mut u = vec![ZERO; rows * rows];
            for a in 0..rows {
                for b in 0..rows {
                    u[a * rows + b] = mx[axis][a * rows + b] * ((owns(a) - owns(b)) / volume);
                }
            }
            let u_ion = (0..rows)
                .map(|a| (px[axis][a] * owns(a) + qx[axis][a * atoms + atom]) / volume)
                .collect();
            items.push(DenseGradient {
                u,
                u_ion,
                e_ion: ex[axis][atom] / volume,
            });
        }
    }
    Ok(DenseGradients { items })
}

/// `sum_G weights(G) (-i G_axis) f_row(G)` for a family of `rows` functions of the wave vector,
/// on the grid's reciprocal set with each frequency the mean over its aliases: the derivative
/// of an alias-averaged phase `f ~ exp(-i G.R)` carries the alias' own `G`. `evaluate` fills
/// `f[row * count + v]` for a block of vectors.
fn derivative_sum(
    ctx: &Context,
    grid: &SpatialGrid,
    weights: &[Complex64],
    rows: usize,
    evaluate: &dyn Fn(&[[f64; 3]], &mut [Complex64]) -> Result<()>,
) -> Result<Vec<[Complex64; 3]>> {
    let n = grid.index.points();
    if weights.len() != n {
        return Err(invalid("one weight per grid point"));
    }
    let mut out = vec![[ZERO; 3]; rows];
    if rows == 0 {
        return Ok(out);
    }
    let shape = grid.index.shape();
    let b = reciprocal::reciprocal_cell(grid)?;
    let block = (Transient::of(ctx).dense_bytes / ((2 * rows + 1) * 16 + 64)).max(1);
    let _scratch = ctx.reserve(crate::product(&[block, (2 * rows + 1) * 16 + 64])?)?;
    let mut f = vec![ZERO; rows * block];
    let mut accumulate =
        |vectors: &[[f64; 3]], values: &[Complex64], weight: &[Complex64], sign: f64| {
            let m = vectors.len();
            for row in 0..rows {
                for (v, g) in vectors.iter().enumerate() {
                    let value = values[row * m + v] * weight[v] * sign;
                    for axis in 0..3 {
                        out[row][axis] += MINUS_I * g[axis] * value;
                    }
                }
            }
        };
    for start in (0..n).step_by(block) {
        let vectors: Vec<[f64; 3]> = (start..(start + block).min(n))
            .map(|at| reciprocal::box_vector(&b, shape, at))
            .collect();
        let m = vectors.len();
        evaluate(&vectors, &mut f[..rows * m])?;
        accumulate(&vectors, &f[..rows * m], &weights[start..start + m], 1.);
    }
    let table = reciprocal::alias_table(ctx, grid)?;
    let mut alias = vec![ZERO; rows * block.min(table.len()).max(1)];
    for range in table.blocks(block) {
        let m = range.len();
        if alias.len() < rows * m {
            alias.resize(rows * m, ZERO);
        }
        let aliased = &table.vectors[range.clone()];
        let boxed: Vec<[f64; 3]> = range
            .clone()
            .map(|e| reciprocal::box_vector(&b, shape, table.index[e]))
            .collect();
        let scaled: Vec<Complex64> = range
            .clone()
            .map(|e| weights[table.index[e]] * table.weights[e])
            .collect();
        evaluate(aliased, &mut alias[..rows * m])?;
        accumulate(aliased, &alias[..rows * m], &scaled, 1.);
        let mut at_box = vec![ZERO; rows * m];
        evaluate(&boxed, &mut at_box)?;
        accumulate(&boxed, &at_box, &scaled, -1.);
    }
    Ok(out)
}

/// E1: per channel and axis, `sum_G weights(G) d hat_a(G) / d R_owner(a)^axis` (a channel does
/// not move with any other atom). With `weights = K rho^* / V`, `Re sum_a q_a out[a]` is the
/// derivative of the hats' share of the electrostatic energy.
pub fn hat_overlaps(
    ctx: &Context,
    grid: &SpatialGrid,
    shapes: &Shapes,
    sites: &Sites,
    weights: &[Complex64],
) -> Result<Vec<[Complex64; 3]>> {
    sites.check()?;
    derivative_sum(ctx, grid, weights, sites.channels.len(), &|vectors, out| {
        shapes.transforms(sites, vectors, out, vectors.len())
    })
}

/// E2: per atom and axis, `sum_G weights(G) d ion_A(G) / d R_A^axis` of the Gaussian ions
/// (electron sign, `ion_A = -Z_A exp(-sigma^2 G^2 / 2) exp(-i G.R_A)`).
pub fn ion_overlaps(
    ctx: &Context,
    grid: &SpatialGrid,
    sites: &Sites,
    weights: &[Complex64],
) -> Result<Vec<[Complex64; 3]>> {
    sites.check()?;
    derivative_sum(ctx, grid, weights, sites.atoms.len(), &|vectors, out| {
        let m = vectors.len();
        for (v, g) in vectors.iter().enumerate() {
            let envelope =
                (-0.5 * sites.sigma * sites.sigma * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]))
                    .exp();
            for (a, atom) in sites.atoms.iter().enumerate() {
                let angle =
                    -(g[0] * atom.center[0] + g[1] * atom.center[1] + g[2] * atom.center[2]);
                out[a * m + v] =
                    Complex64::new(angle.cos(), angle.sin()) * (-atom.charge * envelope);
            }
        }
        Ok(())
    })
}
