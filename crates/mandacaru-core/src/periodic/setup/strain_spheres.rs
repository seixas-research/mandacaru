//! Strain tangents of the Bloch sums on a quadrature sphere, and the strain derivatives of the
//! sphere quantities built on them: the projections `C = <chi | p>`, the short-range local
//! matrix `V_sr^A` of an atom and the short-range ion-compensation sum `linear`.
//!
//! A sphere is rigid about its atom: its points are `x_p = R_A + s_p` with `s_p` fixed, and so
//! are the projector shape `p(s_p)`, the radial potential `v(|s_p|)` and the quadrature weights.
//! Under `x -> (1 + eps) x` the atom, the centers `c_mu` and the image translations `T` move with
//! the cell, so the argument of an image of `chi_mu` changes from `y = x_p - c_mu - T` to
//! `y' = s_p - (1 + eps)(c_mu + T - R_A)` and `d y / d eps_ab = -e_a Delta_b` with
//! `Delta = c_mu + T - R_A`: the tangent of an image is its derivative `(d_a phi)` weighted by
//! `-Delta_b`, a constant of the image. The projections and the short-range sphere are therefore
//! formed image by image once for every k-point and strain direction
//! ([`super::projection_table`], [`super::sphere_table`]); this module keeps the explicit
//! image-potential term of the compensation sum and the nine-entry form of the projections.
//!
//! * [`projection_strain`]: `dC_ab = sum_x w_x p_p(s_x) conj(d chi_mu(x))`.
//! * [`linear_strain`]: the short-range ion-compensation sum evaluates the potentials of every
//!   atom and image at the points of a compensation sphere, and `y = s_p - Delta` with
//!   `Delta = R_B + T - R_A` moves, own images `T != 0` included (under an atom displacement
//!   they cancel, under strain `Delta = T` does not): the explicit image-potential term.
use super::{
    ions::{LocalPotential, SPHERE_AZIMUTHAL, SPHERE_POLAR, SPHERE_RADIAL},
    projection_table::ProjectionTable,
    pulay::Matrix,
    shapes::{visit_lattice, Sites},
    shells::Shells,
    spheres::Projector,
};
use crate::{
    math::angular,
    paw::{
        local_split::{lattice_angular_rule, radial_rule, short_range_potential_derivative},
        multipole::shape_function,
    },
    periodic::invalid,
    product, Context, Result,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;

/// `dC_ab` for a list of projectors at the k-point `k`: `[a][b]` is the `m x p` matrix
/// `d C[mu, p] / d eps_ab` with the spheres, the projector shapes and the quadrature weights
/// rigid about their atoms, `C[mu, p] = sum_x w_x conj(chi_mu(x)) p_p(x)`. The list is the scalar
/// projectors or the spin-orbit union; projectors of one atom with the same cutoff share a sphere.
/// This is [`ProjectionTable::strain`] along the nine unit directions of one k-point: the stress
/// itself reads the table of all its k-points and the directions of its symmetry.
pub fn projection_strain(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
) -> Result<[[Matrix; 3]; 3]> {
    let table = ProjectionTable::new(
        ctx,
        shells,
        lattice,
        projectors,
        &[Some(0), Some(1), Some(2)],
    )?;
    let directions: Vec<[[f64; 3]; 3]> = (0..9)
        .map(|index| {
            let mut e = [[0.; 3]; 3];
            e[index / 3][index % 3] = 1.;
            e
        })
        .collect();
    let mut tangents = table.strain(k, &directions)?.into_iter();
    Ok(std::array::from_fn(|_| {
        std::array::from_fn(|_| tangents.next().expect("nine tangents"))
    }))
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
    let (axes, angular_weights) =
        lattice_angular_rule(ctx, &sites.lattice, SPHERE_POLAR, SPHERE_AZIMUTHAL)?;
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
