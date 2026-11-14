//! The one-center momentum of a PAW sphere, shared by the molecule (the velocity form of the
//! linear response and the velocity gauge of real time) and the crystal (the momentum matrix
//! `P(k)` of `periodic::optics`). `p = -i grad`, so `Delta p_ij = -i G_ij` with `G` the
//! antisymmetric one-center gradient blocks below; the PAW velocity is
//! `p~ + sum |p~_i> Delta p_ij <p~_j|`, with no separate nonlocal term.
use crate::{
    math::{angular::cartesian_gradient, spline::simpson},
    paw::pair_tables::{angular_rule, harmonic, Projector},
    Context, Error, Result,
};
use nalgebra::DMatrix;

/// The partial waves of one channel on the species' radial knots: the sphere radius and the
/// all-electron and smooth radial tables, one per radial index.
pub struct SphereWaves<'a> {
    pub radius: f64,
    pub ae: &'a [Vec<f64>],
    pub ps: &'a [Vec<f64>],
}

/// The one-center part of the gradient operator of one sphere, in its projector basis:
/// `G^x_ij = <phi_i| d/dx |phi_j> - <phi~_i| d/dx |phi~_j>` for the partial waves
/// `phi_i = R_i(r) Y_i(r^)`, from `d/dx (R Y) = R' Y x^ + (R / r) [grad_t Y]_x`:
///
/// ```text
/// G^x_ij = int r^2 (R_i R_j' - R~_i R~_j') dr  A^x_ij + int r (R_i R_j - R~_i R~_j) dr  B^x_ij,
/// A^x_ij = int Y_i Y_j x^ dOmega,   B^x_ij = int Y_i [grad_t Y_j]_x dOmega,
/// ```
///
/// up to two nodes past the largest radius of the pair (where the all-electron and smooth
/// waves agree).
/// In the continuum `G_ij + G_ji` is the vanishing surface term `[r^2 (R_i R_j - R~_i R~_j)]`;
/// the quadrature's symmetric residual is removed, so the blocks are exactly antisymmetric.
pub fn one_center_gradient(
    ctx: &Context,
    knots: &[f64],
    channels: &[SphereWaves<'_>],
    projectors: &[Projector],
) -> Result<[DMatrix<f64>; 3]> {
    let angular = angular_rule(24, 48)?;
    let p = projectors.len();
    // The radial derivative of every partial wave, all-electron and smooth.
    let mut slopes = Vec::with_capacity(channels.len());
    // The slope of the natural cubic spline through each table (third order in the knot
    // spacing, where a three-point difference is second order).
    let slope_of = |w: &Vec<f64>| -> Result<Vec<f64>> {
        let spline = crate::math::spline::Cubic::new(
            ctx,
            ctx.copy_slice(&knots[..w.len()])?,
            ctx.copy_slice(w)?,
        )?;
        Ok(knots[..w.len()]
            .iter()
            .map(|&r| spline.value_slope(r).1)
            .collect())
    };
    for channel in channels {
        let ae = channel
            .ae
            .iter()
            .map(slope_of)
            .collect::<Result<Vec<_>>>()?;
        let ps = channel
            .ps
            .iter()
            .map(slope_of)
            .collect::<Result<Vec<_>>>()?;
        slopes.push((ae, ps));
    }
    let mut blocks = [
        DMatrix::zeros(p, p),
        DMatrix::zeros(p, p),
        DMatrix::zeros(p, p),
    ];
    for (a, pa) in projectors.iter().enumerate() {
        for (b, pb) in projectors.iter().enumerate() {
            // Only l_b = l_a +- 1 couples through x^ and the tangential gradient.
            if pa.l.abs_diff(pb.l) != 1 {
                continue;
            }
            let (ca, cb) = (&channels[pa.channel], &channels[pb.channel]);
            // Two nodes past the sphere: the waves agree from the first node beyond it,
            // so the surface term of the integration by parts vanishes at the end.
            let inside = (knots.partition_point(|r| *r <= ca.radius.max(cb.radius)) + 2)
                .min(ca.ae[pa.radial].len())
                .min(cb.ae[pb.radial].len())
                .min(ca.ps[pa.radial].len())
                .min(cb.ps[pb.radial].len());
            if inside < 3 {
                return Err(Error::Invalid(
                    "a PAW sphere needs at least three radial nodes".into(),
                ));
            }
            let (d_ae, d_ps) = (
                &slopes[pb.channel].0[pb.radial],
                &slopes[pb.channel].1[pb.radial],
            );
            let (ra, rb, sa, sb) = (
                &ca.ae[pa.radial],
                &cb.ae[pb.radial],
                &ca.ps[pa.radial],
                &cb.ps[pb.radial],
            );
            // Both integrands carry a power of r and vanish at the origin: the node r = 0
            // (zero integrand) adds the interval below the first knot.
            let origin = knots[0] > 0.;
            let nodes: Vec<f64> = origin
                .then_some(0.)
                .into_iter()
                .chain(knots[..inside].iter().copied())
                .collect();
            let slope: Vec<f64> = origin
                .then_some(0.)
                .into_iter()
                .chain(
                    (0..inside).map(|k| knots[k] * knots[k] * (ra[k] * d_ae[k] - sa[k] * d_ps[k])),
                )
                .collect();
            let product: Vec<f64> = origin
                .then_some(0.)
                .into_iter()
                .chain((0..inside).map(|k| knots[k] * (ra[k] * rb[k] - sa[k] * sb[k])))
                .collect();
            let (radial_slope, radial_product) =
                (simpson(&nodes, &slope)?, simpson(&nodes, &product)?);
            let mut a_x = [0.; 3];
            let mut b_x = [0.; 3];
            for (direction, weight) in &angular {
                let (ya, yb) = (
                    harmonic(pa.l, pa.m, *direction),
                    harmonic(pb.l, pb.m, *direction),
                );
                let tangential = cartesian_gradient(pb.l, pb.m, *direction);
                for axis in 0..3 {
                    a_x[axis] += weight * ya * yb * direction[axis];
                    b_x[axis] += weight * ya * tangential[axis];
                }
            }
            for axis in 0..3 {
                blocks[axis][(a, b)] = radial_slope * a_x[axis] + radial_product * b_x[axis];
            }
        }
    }
    // The operator is antisymmetric; the radial quadrature leaves a small symmetric residual
    // (on the library's water, 4e-7 of the largest element of the molecular matrix with the
    // origin interval, 4e-5 without it), which is projected out.
    for block in &mut blocks {
        *block = (&*block - block.transpose()) * 0.5;
    }
    Ok(blocks)
}
