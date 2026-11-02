//! Strain tangents of the Bloch sums on a quadrature sphere, and the strain derivatives of the
//! sphere quantities built on them: the projections `C = <chi | p>`, the short-range local
//! matrix `V_sr^A` of an atom and the short-range ion-compensation sum `linear`.
//!
//! A sphere is rigid about its atom: its points are `x_p = R_A + s_p` with `s_p` fixed, and so
//! are the projector shape `p(s_p)`, the radial potential `v(|s_p|)` and the quadrature weights.
//! Under `x -> (1 + eps) x` the atom, the centers `c_mu` and the image translations `T` move with
//! the cell, so the argument of an image of `chi_mu` changes from `y = x_p - c_mu - T` to
//! `y' = s_p - (1 + eps)(c_mu + T - R_A)` and `d y / d eps_ab = -e_a Delta_b` with
//! `Delta = c_mu + T - R_A`. With `y_b - s_b = -Delta_b`:
//!
//! ```text
//! d chi_mu(x_p) / d eps_ab = sum_T e^{i k.T} (d_a phi)(y) (y_b - s_b) = A_ab(x_p) - s_b G_a(x_p)
//! ```
//!
//! `A_ab` is the weighted Bloch sum of [`super::strain_samples`] (the grid nodes are the special
//! case `s = 0`) and `G_a` the plain summed derivative. This is the identity of the +U gradient
//! (`dC/d eps_ab = sum_T e^{-ik.T} G^a_T Delta^b_T`) written image by image.
//!
//! * [`projection_strain`]: `dC_ab = sum_x w_x p_p(s_x) conj(d chi_mu(x))`.
//! * [`short_range_strain`]: `V_sr^A` is atom `A`'s OWN radial potential on its rigid sphere, so
//!   the potential does not move (`du_A = 0`) and
//!   `dV_ab = sum_x w_x v(s_x) [conj(d chi_mu) chi_nu + conj(chi_mu) d chi_nu]`: only the two
//!   functions change. The image potentials of OTHER atoms do not enter this matrix.
//! * [`linear_strain`]: the short-range ion-compensation sum evaluates the potentials of every
//!   atom and image at the points of a compensation sphere, and `y = s_p - Delta` with
//!   `Delta = R_B + T - R_A` moves, own images `T != 0` included (under an atom displacement
//!   they cancel, under strain `Delta = T` does not): the explicit image-potential term.
use super::{
    basis_gradient::ShortRange,
    ions::{LocalPotential, SPHERE_AZIMUTHAL, SPHERE_POLAR, SPHERE_RADIAL},
    pulay::Matrix,
    shapes::{visit_lattice, Sites},
    shells::{DenseRows, Sampling, Shells},
    spheres::{
        chunk_size, evaluate, Projector, Sphere, SHORT_RANGE_AZIMUTHAL, SHORT_RANGE_POLAR,
        SHORT_RANGE_RADIAL,
    },
    strain_samples::weighted_derivatives,
};
use crate::{
    math::angular,
    paw::{
        local_split::{
            angular_rule, radial_rule, short_range_potential, short_range_potential_derivative,
        },
        multipole::shape_function,
    },
    periodic::invalid,
    product, Buffer, Context, Result,
};
use nalgebra::{DMatrixView, Matrix3, Vector3};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// `G_a(x_p)`, the summed Cartesian derivative of every function of `shells` at the points of a
/// chunk, `(function, point)`, for the one k-point of `sampling`.
pub fn sphere_gradients(
    ctx: &Context,
    shells: &Shells,
    sampling: &Sampling<'_>,
    points: [&[f64]; 3],
) -> Result<[Buffer<Complex64>; 3]> {
    one_k(sampling)?;
    let (m, n) = (shells.functions(), points[0].len());
    let mut out = Vec::with_capacity(3);
    for axis in 0..3 {
        let mut rows = ctx.allocate::<Complex64>(product(&[m, n])?)?;
        shells.derivatives(
            ctx,
            sampling,
            points,
            axis,
            &mut DenseRows::new(rows.as_mut_slice(), m, n),
        )?;
        out.push(rows);
    }
    Ok(out.try_into().ok().expect("three axes"))
}

fn one_k(sampling: &Sampling<'_>) -> Result<()> {
    if sampling.kpoints.len() == 1 {
        Ok(())
    } else {
        Err(invalid(
            "a strain tangent is formed for one k-point at a time",
        ))
    }
}

/// The sphere form `A_ab - s_b G_a` of every function at the points of a chunk of a sphere about
/// `center` (`s = point - center`), `(function, point)`; `gradients` is [`sphere_gradients`] of
/// the same shells, points and k-point.
pub fn sphere_form(
    ctx: &Context,
    shells: &Shells,
    sampling: &Sampling<'_>,
    points: [&[f64]; 3],
    center: [f64; 3],
    gradients: &[Buffer<Complex64>; 3],
    (a, b): (usize, usize),
) -> Result<Buffer<Complex64>> {
    one_k(sampling)?;
    let (m, n) = (shells.functions(), points[0].len());
    let mut out = ctx.allocate::<Complex64>(product(&[m, n])?)?;
    weighted_derivatives(
        ctx,
        shells,
        sampling,
        points,
        a,
        b,
        &mut DenseRows::new(out.as_mut_slice(), m, n),
    )?;
    let slope = gradients[a].as_slice();
    for (row, g) in out.as_mut_slice().chunks_mut(n).zip(slope.chunks(n)) {
        for ((value, g), x) in row.iter_mut().zip(g).zip(points[b]) {
            *value -= g * (x - center[b]);
        }
    }
    Ok(out)
}

/// `dC_ab` for a list of projectors at the k-point `k`: `[a][b]` is the `m x p` matrix
/// `d C[mu, p] / d eps_ab` with the spheres, the projector shapes and the quadrature weights
/// rigid about their atoms, `C[mu, p] = sum_x w_x conj(chi_mu(x)) p_p(x)`. The list is the scalar
/// projectors or the spin-orbit union; projectors of one atom with the same cutoff share a sphere.
/// One chunk of a sphere is formed at a time (three gradients and one tangent of the functions
/// that reach the sphere).
pub fn projection_strain(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
) -> Result<[[Matrix; 3]; 3]> {
    let (functions, count) = (shells.functions(), projectors.len());
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(functions, count)));
    let mut spheres: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for (p, projector) in projectors.iter().enumerate() {
        if !(projector.r_cut.is_finite() && projector.r_cut > 0.) {
            return Err(invalid("a projector needs a positive cutoff radius"));
        }
        spheres
            .entry((projector.atom, (projector.r_cut * 1e12).round() as i64))
            .or_default()
            .push(p);
    }
    for members in spheres.values() {
        let first = &projectors[members[0]];
        let region = super::images::Region {
            center: first.center,
            radius: first.r_cut,
        };
        let reached = shells.reaching(ctx, lattice, &region)?;
        let near = reached.rows.len();
        if near == 0 {
            continue;
        }
        let sphere = Sphere::projector(ctx, first.center, first.r_cut)?;
        let sampling = Sampling {
            lattice,
            region,
            kpoints: &[k],
        };
        for range in sphere.chunks(chunk_size(1, 4 * near)) {
            let points = sphere.points(range.clone());
            let weights = sphere.weights(range.clone());
            let n = weights.len();
            let mut columns = ctx.allocate::<Complex64>(product(&[members.len(), n])?)?;
            for (row, &p) in columns.as_mut_slice().chunks_mut(n).zip(members) {
                evaluate(ctx, &projectors[p], points, row)?;
                for (value, w) in row.iter_mut().zip(weights) {
                    *value *= *w;
                }
            }
            let bra_columns = DMatrixView::from_slice(columns.as_slice(), n, members.len());
            let gradients = sphere_gradients(ctx, &reached.shells, &sampling, points)?;
            for a in 0..3 {
                for b in 0..3 {
                    let tangent = sphere_form(
                        ctx,
                        &reached.shells,
                        &sampling,
                        points,
                        first.center,
                        &gradients,
                        (a, b),
                    )?;
                    let rows = DMatrixView::from_slice(tangent.as_slice(), n, near);
                    let block = rows.adjoint() * bra_columns;
                    for (i, &mu) in reached.rows.iter().enumerate() {
                        for (j, &p) in members.iter().enumerate() {
                            out[a][b][(mu, p)] += block[(i, j)];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `dV_ab` of every short-range sphere at the k-point `k`: `[a][b]` is the Hermitian `m x m`
/// matrix `d V_sr^A[mu, nu] / d eps_ab` of that atom's sphere (the same rule, radial potential
/// and weights as [`super::basis_gradient::short_range_cross`]), zero for the functions that do
/// not reach it. The sphere and its potential are rigid about the atom: only the functions move.
pub fn short_range_strain(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    spheres: &[ShortRange<'_>],
) -> Result<Vec<[[Matrix; 3]; 3]>> {
    let m = shells.functions();
    spheres
        .iter()
        .map(|sphere| {
            let mut out: [[Matrix; 3]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, m)));
            let rule = Sphere::rule(
                ctx,
                sphere.center,
                sphere.potential.reach(sphere.sigma)?,
                SHORT_RANGE_RADIAL,
                SHORT_RANGE_POLAR,
                SHORT_RANGE_AZIMUTHAL,
                sphere.potential.cutoff,
            )?;
            let all = 0..rule.len();
            let radii = rule.radii(all.clone());
            let sampled: Vec<f64> = radii.iter().map(|r| sphere.potential.value(*r)).collect();
            let potential =
                short_range_potential(&sampled, sphere.potential.charge, sphere.sigma, radii)?;
            let weights: Vec<f64> = rule
                .weights(all)
                .iter()
                .zip(&potential)
                .map(|(w, v)| w * v)
                .collect();
            let region = rule.region();
            let reached = shells.reaching(ctx, lattice, &region)?;
            let near = reached.rows.len();
            if near == 0 {
                return Ok(out);
            }
            let sampling = Sampling {
                lattice,
                region,
                kpoints: &[k],
            };
            // `T_ab[i, j] = sum_x w v conj(d chi_i) chi_j`, and `dV = T + T^+`.
            let mut half: [[Matrix; 3]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(near, near)));
            for range in rule.chunks(chunk_size(1, 6 * near)) {
                let points = rule.points(range.clone());
                let n = range.len();
                let _held = ctx.reserve(product(&[n, near, 16])?)?;
                let mut values = ctx.allocate::<Complex64>(product(&[near, n])?)?;
                reached.shells.values(
                    ctx,
                    &sampling,
                    points,
                    &mut DenseRows::new(values.as_mut_slice(), near, n),
                )?;
                let w = &weights[range];
                let ket = Matrix::from_fn(n, near, |x, j| values.as_slice()[j * n + x] * w[x]);
                let gradients = sphere_gradients(ctx, &reached.shells, &sampling, points)?;
                for a in 0..3 {
                    for b in 0..3 {
                        let tangent = sphere_form(
                            ctx,
                            &reached.shells,
                            &sampling,
                            points,
                            sphere.center,
                            &gradients,
                            (a, b),
                        )?;
                        let rows = DMatrixView::from_slice(tangent.as_slice(), n, near);
                        half[a][b] += rows.adjoint() * &ket;
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    for (i, &mu) in reached.rows.iter().enumerate() {
                        for (j, &nu) in reached.rows.iter().enumerate() {
                            out[a][b][(mu, nu)] += half[a][b][(i, j)] + half[a][b][(j, i)].conj();
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect()
}

/// `d linear[channel] / d eps_ab` of [`super::ions::linear`], `out[channel][a][b]`. Only the
/// short-range ion-compensation sum depends on the cell (the on-site long-range integral is a
/// radial integral at a pinned radius and width). A compensation sphere about atom `A` is rigid;
/// the ion `B` in the image `T` sits at `Delta = R_B + T - R_A` from its center, so
/// `d v^sr(|y|) / d eps_ab = -(v^sr)'(|y|) y_a Delta_b / |y|` with `y = s_p - Delta`. The images
/// are those of the energy (`B = A, T = 0` skipped, the reach test unchanged), the own images
/// `T != 0` included.
pub fn linear_strain(
    ctx: &Context,
    sites: &Sites,
    potentials: &[LocalPotential],
) -> Result<Vec<[[Complex64; 3]; 3]>> {
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
    let mut out = vec![[[Complex64::default(); 3]; 3]; sites.channels.len()];
    let (axes, angular_weights) = angular_rule(SPHERE_POLAR, SPHERE_AZIMUTHAL)?;
    let directions: Vec<[f64; 3]> = (0..angular_weights.len())
        .map(|j| [axes[0][j], axes[1][j], axes[2][j]])
        .collect();
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
        let _scratch = ctx.reserve(product(&[points, 8, 12])?)?;
        let center = Vector3::from(sites.atoms[owner].center);
        let at = |p: usize| {
            center + Vector3::from(directions[p % directions.len()]) * r[p / directions.len()]
        };
        // `sum_{B,T} (v^sr)'(|y|) y_a Delta_b / |y|` at every point, `[a * 3 + b]`.
        let mut tensor = vec![[0f64; 9]; points];
        let mut distance = vec![0.; points];
        let mut sampled = vec![0.; points];
        for (other, species) in potentials.iter().enumerate() {
            let reach = reaches[other] + radius;
            let d0 = center - Vector3::from(sites.atoms[other].center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let shift = Vector3::from(*image);
                        if other == owner && shift.norm() < 1e-10 {
                            continue;
                        }
                        if (d0 - shift).norm() > reach {
                            continue;
                        }
                        let origin = Vector3::from(sites.atoms[other].center) + shift;
                        let delta = origin - center;
                        for p in 0..points {
                            distance[p] = (at(p) - origin).norm();
                            sampled[p] = species.slope(distance[p]);
                        }
                        let slope = short_range_potential_derivative(
                            &sampled,
                            species.charge,
                            sigma,
                            &distance,
                        )?;
                        for p in 0..points {
                            let y = at(p) - origin;
                            let scale = slope[p] / distance[p].max(1e-300);
                            for a in 0..3 {
                                for b in 0..3 {
                                    tensor[p][a * 3 + b] += scale * y[a] * delta[b];
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )?;
        }
        for &row in &rows {
            let (_, l, m) = sites.channels[row];
            let shape = shape_function(&r, radius, l)?;
            for (p, t) in tensor.iter().enumerate() {
                let (i, j) = (p / directions.len(), p % directions.len());
                let weight = radial_weights[i] * angular_weights[j] * shape[i];
                let y = angular::complex(l, m, directions[j]) * weight;
                for a in 0..3 {
                    for b in 0..3 {
                        out[row][a][b] -= y * t[a * 3 + b];
                    }
                }
            }
        }
    }
    Ok(out)
}
