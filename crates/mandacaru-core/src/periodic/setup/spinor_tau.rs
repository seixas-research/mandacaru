//! The kinetic-energy density matrix of spinor density matrices, as its trace and its vector
//! part: `tau_{st} = 1/2 sum_k w_k sum_{ab} P^k_{a s, b t} grad chi_a . grad chi_b^*`, the trace
//! `tau_uu + tau_dd` and `(2 Re tau_ud, -2 Im tau_ud, tau_uu - tau_dd)` (the magnetization's
//! construction). The whole mesh only: a vector part is axial, so a mesh reduced by operations
//! does not average to it.
use crate::{periodic::invalid, product, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// `tau` (`points`) and `tau_m` (`3 x points`, row-major), both real.
pub struct SpinorKinetic {
    pub tau: Vec<f64>,
    pub tau_m: Vec<f64>,
}

/// `weights[k]`, `matrices[k]` (`2m x 2m` row-major, the alpha block first) and `gradients[k]`
/// (`3 x m x points` row-major, the Cartesian derivatives of the Bloch sums at `k`).
pub fn spinor_kinetic_density(
    ctx: &Context,
    m: usize,
    points: usize,
    weights: &[f64],
    matrices: &[&[Complex64]],
    gradients: &[&[Complex64]],
) -> Result<SpinorKinetic> {
    let nk = weights.len();
    if nk == 0
        || matrices.len() != nk
        || gradients.len() != nk
        || matrices.iter().any(|p| p.len() != 4 * m * m)
        || gradients.iter().any(|g| g.len() != 3 * m * points)
    {
        return Err(invalid(
            "a spinor kinetic density needs, per k-point, a weight, a 2m x 2m matrix and (3, m, points) gradients",
        ));
    }
    let _scratch = ctx.reserve(product(&[m, points, 2 * 16])?)?;
    let mut uu = vec![0.; points];
    let mut dd = vec![0.; points];
    let mut ud = vec![Complex64::default(); points];
    for ((&w, matrix), gradient) in weights.iter().zip(matrices).zip(gradients) {
        let block = |row: usize, column: usize| {
            DMatrix::from_fn(m, m, |a, b| matrix[(row * m + a) * 2 * m + column * m + b])
        };
        let (p_uu, p_dd, p_ud) = (block(0, 0), block(1, 1), block(0, 1));
        for axis in 0..3 {
            // `d` is (m, points); the products below are `sum_a d_a (P d^*)_a` per point.
            let d = DMatrix::from_fn(m, points, |a, x| gradient[(axis * m + a) * points + x]);
            let conjugate = d.map(|v| v.conj());
            for (target, p) in [(&mut uu, &p_uu), (&mut dd, &p_dd)] {
                let product = p * &conjugate;
                for x in 0..points {
                    let sum: Complex64 = (0..m).map(|a| d[(a, x)] * product[(a, x)]).sum();
                    target[x] += 0.5 * w * sum.re;
                }
            }
            let product = &p_ud * &conjugate;
            for x in 0..points {
                let sum: Complex64 = (0..m).map(|a| d[(a, x)] * product[(a, x)]).sum();
                ud[x] += sum * (0.5 * w);
            }
        }
    }
    let tau = uu.iter().zip(&dd).map(|(u, d)| u + d).collect();
    let mut tau_m = Vec::with_capacity(3 * points);
    tau_m.extend(ud.iter().map(|v| 2. * v.re));
    tau_m.extend(ud.iter().map(|v| -2. * v.im));
    tau_m.extend(uu.iter().zip(&dd).map(|(u, d)| u - d));
    Ok(SpinorKinetic { tau, tau_m })
}
