//! The kinetic-energy density and smooth density contractions for periodic setups:
//! - Spinor kinetic density (`PeriodicPAW.spinor_kinetic_energy_density`)
//! - Scalar kinetic density (`PeriodicPAW.kinetic_energy_density`)
//! - Smooth magnetization (`PeriodicPAW.magnetization`)
//! - Smooth valence density and projected density matrix (`PeriodicPAW.density`)
use crate::{periodic::invalid, product, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;

const CHUNK: usize = 4096;

/// `tau` (`points`) and `tau_m` (`3 x points`, row-major), both real.
pub struct SpinorKinetic {
    pub tau: Vec<f64>,
    pub tau_m: Vec<f64>,
}

/// Smooth valence density `rho` and projected density matrix `R = sum_k w_k C(k)^+ P^k C(k)`.
pub struct DensityAndProjected {
    pub rho: Vec<f64>,
    pub r_matrix: Vec<Complex64>,
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
    let chunk = points.min(CHUNK);
    let _scratch = ctx.reserve(product(&[m, chunk, 2 * 16])?)?;
    let mut uu = vec![0.; points];
    let mut dd = vec![0.; points];
    let mut ud = vec![Complex64::default(); points];
    for ((&w, matrix), gradient) in weights.iter().zip(matrices).zip(gradients) {
        let block = |row: usize, column: usize| {
            DMatrix::from_fn(m, m, |a, b| matrix[(row * m + a) * 2 * m + column * m + b])
        };
        let (p_uu, p_dd, p_ud) = (block(0, 0), block(1, 1), block(0, 1));
        for axis in 0..3 {
            let g_axis = &gradient[axis * m * points..(axis + 1) * m * points];
            for x0 in (0..points).step_by(CHUNK) {
                let x_len = (points - x0).min(CHUNK);
                let d = DMatrix::from_fn(m, x_len, |a, j| g_axis[a * points + (x0 + j)]);
                let conjugate = d.map(|v| v.conj());
                for (target, p) in [(&mut uu, &p_uu), (&mut dd, &p_dd)] {
                    let product = p * &conjugate;
                    for j in 0..x_len {
                        let sum: Complex64 = (0..m).map(|a| d[(a, j)] * product[(a, j)]).sum();
                        target[x0 + j] += 0.5 * w * sum.re;
                    }
                }
                let product = &p_ud * &conjugate;
                for j in 0..x_len {
                    let sum: Complex64 = (0..m).map(|a| d[(a, j)] * product[(a, j)]).sum();
                    ud[x0 + j] += sum * (0.5 * w);
                }
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

/// Scalar kinetic-energy density: `tau(x) = 1/2 sum_k w_k sum_axis sum_{ab} P^k_{ab} grad chi_a . grad chi_b^*`.
pub fn scalar_kinetic_density(
    ctx: &Context,
    m: usize,
    points: usize,
    weights: &[f64],
    matrices: &[&[Complex64]],
    gradients: &[&[Complex64]],
) -> Result<Vec<f64>> {
    let nk = weights.len();
    if nk == 0
        || matrices.len() != nk
        || gradients.len() != nk
        || matrices.iter().any(|p| p.len() != m * m)
        || gradients.iter().any(|g| g.len() != 3 * m * points)
    {
        return Err(invalid(
            "a kinetic density needs, per k-point, a weight, an m x m matrix and (3, m, points) gradients",
        ));
    }
    let chunk = points.min(CHUNK);
    let _scratch = ctx.reserve(product(&[m, chunk, 16])?)?;
    let mut tau = vec![0.; points];
    for ((&w, matrix), gradient) in weights.iter().zip(matrices).zip(gradients) {
        let p_mat = DMatrix::from_fn(m, m, |a, b| matrix[a * m + b]);
        for axis in 0..3 {
            let g_axis = &gradient[axis * m * points..(axis + 1) * m * points];
            for x0 in (0..points).step_by(CHUNK) {
                let x_len = (points - x0).min(CHUNK);
                let d = DMatrix::from_fn(m, x_len, |a, j| g_axis[a * points + (x0 + j)]);
                let conjugate = d.map(|v| v.conj());
                let prod = &p_mat * &conjugate;
                for j in 0..x_len {
                    let sum: Complex64 = (0..m).map(|a| d[(a, j)] * prod[(a, j)]).sum();
                    tau[x0 + j] += 0.5 * w * sum.re;
                }
            }
        }
    }
    Ok(tau)
}

/// Smooth magnetization of spinor density matrices: `(3, points)` row-major,
/// `m_x = 2 Re(ud)`, `m_y = -2 Im(ud)`, `m_z = uu - dd`.
pub fn spinor_magnetization(
    ctx: &Context,
    m: usize,
    points: usize,
    weights: &[f64],
    matrices: &[&[Complex64]],
    psi_list: &[&[Complex64]],
) -> Result<Vec<f64>> {
    let nk = weights.len();
    if nk == 0
        || matrices.len() != nk
        || psi_list.len() != nk
        || matrices.iter().any(|p| p.len() != 4 * m * m)
        || psi_list.iter().any(|psi| psi.len() != m * points)
    {
        return Err(invalid(
            "spinor magnetization needs, per k-point, a weight, a 2m x 2m matrix and (m, points) psi samples",
        ));
    }
    let chunk = points.min(CHUNK);
    let _scratch = ctx.reserve(product(&[m, chunk, 16])?)?;
    let mut z = vec![0.; points];
    let mut ud = vec![Complex64::default(); points];
    for ((&w, matrix), psi) in weights.iter().zip(matrices).zip(psi_list) {
        let block = |row: usize, column: usize| {
            DMatrix::from_fn(m, m, |a, b| matrix[(row * m + a) * 2 * m + column * m + b])
        };
        let (p_uu, p_dd, p_ud) = (block(0, 0), block(1, 1), block(0, 1));
        let p_z = p_uu - p_dd;
        for x0 in (0..points).step_by(CHUNK) {
            let x_len = (points - x0).min(CHUNK);
            let psi_mat = DMatrix::from_fn(m, x_len, |a, j| psi[a * points + (x0 + j)]);
            let conjugate = psi_mat.map(|v| v.conj());
            let prod_z = &p_z * &conjugate;
            let prod_ud = &p_ud * &conjugate;
            for j in 0..x_len {
                let sum_z: Complex64 = (0..m).map(|a| psi_mat[(a, j)] * prod_z[(a, j)]).sum();
                z[x0 + j] += w * sum_z.re;
                let sum_ud: Complex64 = (0..m).map(|a| psi_mat[(a, j)] * prod_ud[(a, j)]).sum();
                ud[x0 + j] += sum_ud * w;
            }
        }
    }
    let mut m_vec = Vec::with_capacity(3 * points);
    m_vec.extend(ud.iter().map(|v| 2. * v.re));
    m_vec.extend(ud.iter().map(|v| -2. * v.im));
    m_vec.extend(z);
    Ok(m_vec)
}

/// Computes the smooth valence density `rho` and projected density matrix `R = sum_k w_k C(k)^+ P^k C(k)`.
pub fn density_and_projected(
    ctx: &Context,
    m: usize,
    points: usize,
    p_proj: usize,
    weights: &[f64],
    matrices: &[&[Complex64]],
    psi_list: &[&[Complex64]],
    projections: &[&[Complex64]],
) -> Result<DensityAndProjected> {
    let nk = weights.len();
    if nk == 0
        || matrices.len() != nk
        || psi_list.len() != nk
        || projections.len() != nk
        || matrices.iter().any(|p| p.len() != m * m)
        || psi_list.iter().any(|psi| psi.len() != m * points)
        || projections.iter().any(|c| c.len() != m * p_proj)
    {
        return Err(invalid(
            "density_and_projected needs, per k-point, a weight, an m x m matrix, (m, points) psi and (m, p) C",
        ));
    }
    let chunk = points.min(CHUNK);
    let _scratch = ctx.reserve(product(&[m, chunk.max(p_proj), 16])?)?;
    let mut rho = vec![0.; points];
    let mut r_mat = DMatrix::zeros(p_proj, p_proj);
    for (((&w, matrix), psi), c_slice) in weights.iter().zip(matrices).zip(psi_list).zip(projections) {
        let p_mat = DMatrix::from_fn(m, m, |a, b| matrix[a * m + b]);
        for x0 in (0..points).step_by(CHUNK) {
            let x_len = (points - x0).min(CHUNK);
            let psi_mat = DMatrix::from_fn(m, x_len, |a, j| psi[a * points + (x0 + j)]);
            let conjugate = psi_mat.map(|v| v.conj());
            let prod = &p_mat * &conjugate;
            for j in 0..x_len {
                let sum: Complex64 = (0..m).map(|a| psi_mat[(a, j)] * prod[(a, j)]).sum();
                rho[x0 + j] += w * sum.re;
            }
        }
        let c_mat = DMatrix::from_fn(m, p_proj, |a, p| c_slice[a * p_proj + p]);
        let r_k = c_mat.adjoint() * &p_mat * &c_mat;
        for i in 0..p_proj {
            for j in 0..p_proj {
                r_mat[(i, j)] += Complex64::new(w, 0.) * r_k[(i, j)];
            }
        }
    }
    let r_vec = (0..p_proj)
        .flat_map(|i| (0..p_proj).map(move |j| (i, j)))
        .map(|(i, j)| r_mat[(i, j)])
        .collect();
    Ok(DensityAndProjected {
        rho,
        r_matrix: r_vec,
    })
}
