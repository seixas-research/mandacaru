//! Assembly of per-kpoint matrices (overlap S, kinetic T, fixed Hamiltonian, projections C, Lowdin transform).
//!
//! Evaluates the Bloch sums on the cell grid, projects them on the augmentation spheres,
//! and forms the matrices required by [`crate::periodic::Crystal`]:
//!
//! ```text
//! S(k)     = (psi_k^H @ psi_k) * dV + C(k) @ q_overlap @ C(k)^H
//! fixed(k) = T(k) + V_sr(k) + C(k) @ D_ion @ C(k)^H
//! ```
//!
//! The wave function samples `psi` are written once at the requested precision ([`Precision`]).

use super::{
    basis_gradient::{short_range_cross, ShortRange},
    ions::LocalPotential,
    provider::Operators,
    pulay::{kinetic_form, Mesh},
    shells::{DenseRows, Sampling, Shells},
    spheres::{self, Projector},
    spin_orbit::{self, Coupling},
};
use crate::{
    math::{
        linalg::eigen,
        storage::{ComplexSamples, Precision},
        SpatialGrid,
    },
    periodic::input::KPoint,
    product, Context, Result,
};
use nalgebra::{DMatrix, DVector, Matrix3};
use num_complex::{Complex32, Complex64};
use std::collections::BTreeMap;

/// Parameters for assembling the k-point matrices.
pub struct MatrixAssembly<'a> {
    pub grid: &'a SpatialGrid,
    pub shells: &'a Shells,
    pub projectors: &'a [Projector],
    pub potentials: &'a [LocalPotential],
    pub centers: &'a [[f64; 3]],
    pub lattice: &'a Matrix3<f64>,
    pub kpoints_cart: &'a [[f64; 3]],
    pub kpoints_frac: &'a [[f64; 3]],
    pub weights: &'a [f64],
    pub operators: &'a Operators,
    pub spin_orbit_couplings: Option<&'a BTreeMap<(usize, usize), Coupling>>,
    pub sigma: f64,
    pub precision: Precision,
}

/// Compute all k-point matrices and package them as [`KPoint`]s for [`crate::periodic::Crystal`].
pub fn kpoint_matrices(ctx: &Context, assembly: &MatrixAssembly<'_>) -> Result<Vec<KPoint>> {
    let nk = assembly.kpoints_cart.len();
    let m = assembly.shells.functions();
    let p = assembly.projectors.len();
    let mesh = Mesh::new(ctx, assembly.grid)?;
    let n_points = mesh.len();
    let d_v = mesh.dv;

    let points_slice = [
        mesh.points()[0],
        mesh.points()[1],
        mesh.points()[2],
    ];

    let so_buffer = if let Some(couplings) = assembly.spin_orbit_couplings {
        if !couplings.is_empty() {
            Some(spin_orbit::matrices(
                ctx,
                assembly.shells,
                assembly.lattice,
                assembly.kpoints_cart,
                assembly.projectors,
                couplings,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    let mut kpoints = Vec::with_capacity(nk);

    for k_idx in 0..nk {
        let k = assembly.kpoints_cart[k_idx];
        let frac = assembly.kpoints_frac[k_idx];
        let weight = assembly.weights[k_idx];

        // 1. Evaluate Bloch sums psi on the grid
        let sampling = Sampling {
            lattice: assembly.lattice,
            region: mesh.region(),
            kpoints: &[k],
        };
        let mut psi_buf = ctx.allocate::<Complex64>(product(&[m, n_points])?)?;
        assembly.shells.values(
            ctx,
            &sampling,
            points_slice,
            &mut DenseRows::new(psi_buf.as_mut_slice(), m, n_points),
        )?;

        // 2. Projections C of basis onto augmentation projectors: (m, p)
        let c_buf = spheres::projections(
            ctx,
            assembly.shells,
            assembly.lattice,
            &[k],
            assembly.projectors,
        )?;
        let c_mat = DMatrix::from_row_slice(m, p, c_buf.as_slice());

        // 3. Grid overlap S_grid = dV * sum_x conj(psi_mu(x)) psi_nu(x)
        let psi_slice = psi_buf.as_slice();
        let mut s_mat = DMatrix::zeros(m, m);
        for mu in 0..m {
            let row_mu = &psi_slice[mu * n_points..(mu + 1) * n_points];
            for nu in 0..m {
                let row_nu = &psi_slice[nu * n_points..(nu + 1) * n_points];
                let mut dot = Complex64::new(0., 0.);
                for x in 0..n_points {
                    dot += row_mu[x].conj() * row_nu[x];
                }
                s_mat[(mu, nu)] = dot * d_v;
            }
        }

        // Add PAW augmentation overlap correction: C @ q_overlap @ C^H
        let q_overlap = &assembly.operators.augmentation;
        s_mat += &c_mat * q_overlap * c_mat.adjoint();

        // Symmetrize / Hermitianize S = 0.5 * (S + S^H)
        s_mat = (&s_mat + s_mat.adjoint()) * Complex64::new(0.5, 0.);

        // Lowdin orthonormalization matrix S^{-1/2}
        let (eigenvalues, eigenvectors) = eigen(s_mat.clone())?;
        if eigenvalues.iter().any(|&e| e <= 1e-10) {
            return Err(crate::Error::Invalid(
                "periodic overlap is singular or indefinite".into(),
            ));
        }
        let inv_sqrt_diag = DMatrix::from_diagonal(&DVector::from_iterator(
            m,
            eigenvalues.iter().map(|&e| Complex64::new(1.0 / e.sqrt(), 0.0)),
        ));
        let lowdin = &eigenvectors * inv_sqrt_diag * eigenvectors.adjoint();

        // 4. Kinetic matrix T
        let mut t_mat = kinetic_form(
            ctx,
            assembly.grid,
            &mesh,
            k,
            (psi_slice, m),
            (psi_slice, m),
        )?;
        t_mat = (&t_mat + t_mat.adjoint()) * Complex64::new(0.5, 0.);

        // 5. Short-range local potential matrix V_sr summed over atoms
        let mut v_sr = DMatrix::zeros(m, m);
        for (a_idx, pot) in assembly.potentials.iter().enumerate() {
            let sr = ShortRange {
                center: assembly.centers[a_idx],
                potential: pot,
                sigma: assembly.sigma,
            };
            let cross = short_range_cross(
                ctx,
                assembly.lattice,
                &[k],
                &sr,
                assembly.shells,
                None,
                assembly.shells,
            )?;
            let cross_mat = DMatrix::from_row_slice(m, m, cross.as_slice());
            v_sr += cross_mat;
        }

        // 6. Nonlocal ionic term C @ D_ion @ C^H
        let d_ion = &assembly.operators.ion_operator;
        let h_ion = &c_mat * d_ion * c_mat.adjoint();

        // Total fixed Hamiltonian: fixed = T + V_sr + C @ D_ion @ C^H
        let mut fixed = t_mat + v_sr + h_ion;
        fixed = (&fixed + fixed.adjoint()) * Complex64::new(0.5, 0.);

        // 7. Spin-orbit matrix if present
        let spin_orbit = if let Some(ref so_buf) = so_buffer {
            let stride = 4 * m * m;
            let so_slice = &so_buf.as_slice()[k_idx * stride..(k_idx + 1) * stride];
            Some(DMatrix::from_row_slice(2 * m, 2 * m, so_slice))
        } else {
            None
        };

        // 8. Convert psi to ComplexSamples with requested precision and offload rule
        let sample_bytes = product(&[
            m,
            n_points,
            if assembly.precision == Precision::Float32 { 8 } else { 16 },
        ])?;
        let psi_samples = if ctx.offloads(sample_bytes) {
            ComplexSamples::archived_copy(ctx, psi_slice, assembly.precision)?
        } else {
            match assembly.precision {
                Precision::Float64 => ComplexSamples::Float64(psi_buf),
                Precision::Float32 => {
                    let mut f32_buf = ctx.allocate::<Complex32>(psi_slice.len())?;
                    for (dst, src) in f32_buf.as_mut_slice().iter_mut().zip(psi_slice) {
                        *dst = Complex32::new(src.re as f32, src.im as f32);
                    }
                    ComplexSamples::Float32(f32_buf)
                }
            }
        };

        kpoints.push(KPoint {
            k,
            fractional: frac,
            weight,
            psi: psi_samples,
            overlap: s_mat,
            fixed,
            projections: c_mat,
            lowdin,
            spin_orbit,
        });
    }

    Ok(kpoints)
}
