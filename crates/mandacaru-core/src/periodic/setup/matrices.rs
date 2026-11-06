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
    pulay::{kinetic_self, Mesh},
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
    /// The spin-orbit dataset projectors; distinct from scalar raw projectors.
    pub spin_orbit_projectors: Option<&'a [Projector]>,
    pub sigma: f64,
    pub precision: Precision,
}

/// Grid overlap of row-major orbital samples, without a conjugated grid copy.
pub fn grid_overlap(samples: &[Complex64], functions: usize, points: usize, dv: f64) -> Result<DMatrix<Complex64>> {
    if functions == 0 || points == 0 || samples.len() != product(&[functions, points])?
        || !dv.is_finite() || dv <= 0. {
        return Err(crate::Error::Invalid("invalid orbital grid or volume element".into()));
    }
    crate::math::gemm::complex_gram(samples, points, functions, dv)
}

/// Compute `C Q C^H` and `C D C^H`. The accessors permit both real and
/// complex, row- or column-major operators without duplicating their storage.
pub fn projector_products(
    ctx: &Context, m: usize, p: usize,
    c: impl Fn(usize, usize) -> Complex64,
    q: impl Fn(usize, usize) -> Complex64,
    d: impl Fn(usize, usize) -> Complex64,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>)> {
    product(&[m, m])?;
    // PAW operators are normally block diagonal. Find each row's occupied
    // interval once so every orbital avoids scanning the zero-filled tails.
    // Interior exact zeros are still skipped; no tolerance is introduced.
    let mut ranges = ctx.allocate::<usize>(product(&[2, p])?)?;
    let tiled = m >= 16 && p >= m.saturating_mul(2);
    let mut hermitian = true;
    for k in 0..p {
        let (mut first, mut end) = (p, 0);
        for j in 0..p {
            let (qkj, dkj) = (q(k, j), d(k, j));
            if !qkj.re.is_finite() || !qkj.im.is_finite() || !dkj.re.is_finite() || !dkj.im.is_finite() {
                return Err(crate::Error::Invalid("PAW projector operators must be finite".into()));
            }
            if !tiled && j <= k && (qkj != q(j, k).conj() || dkj != d(j, k).conj()) {
                hermitian = false;
            }
            if qkj != Complex64::new(0., 0.) || dkj != Complex64::new(0., 0.) {
                first = first.min(j);
                end = j + 1;
            }
        }
        ranges.as_mut_slice()[2 * k] = first;
        ranges.as_mut_slice()[2 * k + 1] = end;
    }
    if tiled {
        for i in 0..m {
            for k in 0..p {
                let value = c(i, k);
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(crate::Error::Invalid("PAW projections must be finite".into()));
                }
            }
        }
        let mut scratch = TiledScratch::new(ctx, m, p)?;
        let overlap = tiled_projector_product(m, p, &c, &q, ranges.as_slice(), &mut scratch)?;
        let ionic = tiled_projector_product(m, p, &c, &d, ranges.as_slice(), &mut scratch)?;
        return Ok((overlap, ionic));
    }
    let mut overlap = DMatrix::zeros(m, m);
    let mut ionic = DMatrix::zeros(m, m);
    let mut work = ctx.allocate::<Complex64>(product(&[2, p])?)?;
    let (qrow, drow) = work.as_mut_slice().split_at_mut(p);
    for i in 0..m {
        qrow.fill(Complex64::new(0., 0.));
        drow.fill(Complex64::new(0., 0.));
        for k in 0..p {
            let value = c(i, k);
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(crate::Error::Invalid("PAW projections must be finite".into()));
            }
            for j in ranges.as_slice()[2 * k]..ranges.as_slice()[2 * k + 1] {
                let (qkj, dkj) = (q(k, j), d(k, j));
                if qkj != Complex64::new(0., 0.) { qrow[j] += value * qkj; }
                if dkj != Complex64::new(0., 0.) { drow[j] += value * dkj; }
            }
        }
        for j in (if hermitian { i } else { 0 })..m {
            let mut sq = Complex64::new(0., 0.);
            let mut sd = Complex64::new(0., 0.);
            for k in 0..p {
                let right = c(j, k).conj();
                sq += qrow[k] * right;
                sd += drow[k] * right;
            }
            overlap[(i, j)] = sq;
            ionic[(i, j)] = sd;
            if hermitian && i != j {
                overlap[(j, i)] = sq.conj();
                ionic[(j, i)] = sd.conj();
            }
        }
    }
    Ok((overlap, ionic))
}

struct TiledScratch {
    tile: usize,
    transformed: crate::Buffer<f64>,
    right: crate::Buffer<f64>,
    result: crate::Buffer<f64>,
}
impl TiledScratch {
    fn new(ctx: &Context, m: usize, p: usize) -> Result<Self> {
        let tile = (m / 2).min(32).max(1);
        Ok(Self {
            tile,
            transformed: ctx.allocate::<f64>(product(&[2, m, p])?)?,
            right: ctx.allocate::<f64>(product(&[2, p, tile])?)?,
            result: ctx.allocate::<f64>(product(&[2, m, tile])?)?,
        })
    }
}

/// Tiled real GEMMs for a complex product. Only one transformed projection
/// and a bounded block of conjugate projections are held at a time.
fn tiled_projector_product(
    m: usize, p: usize,
    c: &impl Fn(usize, usize) -> Complex64,
    op: &impl Fn(usize, usize) -> Complex64,
    ranges: &[usize],
    scratch: &mut TiledScratch,
) -> Result<DMatrix<Complex64>> {
    use crate::math::gemm::{multiply, MatrixMut, MatrixRef};
    let tile = scratch.tile;
    let mp = product(&[m, p])?;
    let pt = product(&[p, tile])?;
    let mt = product(&[m, tile])?;
    scratch.transformed.as_mut_slice().fill(0.);
    let (ar, ai) = scratch.transformed.as_mut_slice().split_at_mut(mp);
    for i in 0..m {
        for k in 0..p {
            let value = c(i, k);
            for j in ranges[2 * k]..ranges[2 * k + 1] {
                let coefficient = op(k, j);
                if coefficient != Complex64::new(0., 0.) {
                    let z = value * coefficient;
                    ar[i + j * m] += z.re;
                    ai[i + j * m] += z.im;
                }
            }
        }
    }
    let (br, bi) = scratch.right.as_mut_slice().split_at_mut(pt);
    let (rr, ri) = scratch.result.as_mut_slice().split_at_mut(mt);
    let mut result = DMatrix::zeros(m, m);
    for first in (0..m).step_by(tile) {
        let count = tile.min(m - first);
        for j in 0..count {
            for k in 0..p {
                let z = c(first + j, k);
                br[k + j * p] = z.re;
                bi[k + j * p] = z.im;
            }
        }
        let a = |values| MatrixRef { values, rows: m, columns: p, leading: m };
        let b = |values| MatrixRef { values, rows: p, columns: count, leading: p };
        multiply(a(ar), b(br), MatrixMut { values: rr, rows: m, columns: count, leading: m }, 1., 0.)?;
        multiply(a(ai), b(bi), MatrixMut { values: rr, rows: m, columns: count, leading: m }, 1., 1.)?;
        multiply(a(ai), b(br), MatrixMut { values: ri, rows: m, columns: count, leading: m }, 1., 0.)?;
        multiply(a(ar), b(bi), MatrixMut { values: ri, rows: m, columns: count, leading: m }, -1., 1.)?;
        for j in 0..count {
            for i in 0..m {
                result[(i, first + j)] = Complex64::new(rr[i + j * m], ri[i + j * m]);
            }
        }
    }
    Ok(result)
}

/// Apply PAW corrections and Hermitianize the two energy matrices in place.
pub fn assemble_paw_matrices(
    ctx: &Context, m: usize, p: usize, grid_overlap: &[Complex64],
    kinetic: &[Complex64], local: &[Complex64],
    c: impl Fn(usize, usize) -> Complex64,
    q: impl Fn(usize, usize) -> Complex64,
    d: impl Fn(usize, usize) -> Complex64,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>)> {
    let size = product(&[m, m])?;
    if m == 0 || grid_overlap.len() != size || kinetic.len() != size || local.len() != size {
        return Err(crate::Error::Invalid("PAW matrix dimensions disagree".into()));
    }
    if grid_overlap.iter().chain(kinetic).chain(local).any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(crate::Error::Invalid("PAW input matrices must be finite".into()));
    }
    let (augmentation, ionic) = projector_products(ctx, m, p, c, q, d)?;
    let mut overlap = DMatrix::from_row_slice(m, m, grid_overlap) + augmentation;
    let mut fixed = DMatrix::from_row_slice(m, m, kinetic)
        + DMatrix::from_row_slice(m, m, local) + ionic;
    for matrix in [&mut overlap, &mut fixed] {
        for i in 0..m {
            for j in i..m {
                let value = (matrix[(i, j)] + matrix[(j, i)].conj()) * 0.5;
                matrix[(i, j)] = value;
                matrix[(j, i)] = value.conj();
            }
        }
    }
    Ok((overlap, fixed))
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
                assembly.spin_orbit_projectors.ok_or_else(|| {
                    crate::Error::Invalid("spin-orbit couplings need their own projectors".into())
                })?,
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
        let mut s_mat = grid_overlap(psi_slice, m, n_points, d_v)?;

        // Add PAW augmentation overlap correction: C @ q_overlap @ C^H
        let (augmentation, h_ion) = projector_products(ctx, m, p,
            |i, j| c_mat[(i, j)],
            |i, j| assembly.operators.augmentation[(i, j)],
            |i, j| assembly.operators.ion_operator[(i, j)])?;
        s_mat += augmentation;

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
        let mut t_mat = kinetic_self(
            ctx,
            assembly.grid,
            &mesh,
            k,
            psi_slice,
            m,
        )?;
        t_mat = (&t_mat + t_mat.adjoint()) * Complex64::new(0.5, 0.);

        // The full-precision samples are no longer needed for assembly. Archive
        // or convert them now, before the short-range quadrature allocates its
        // own work buffers, so those peaks do not overlap.
        let sample_bytes = product(&[
            m,
            n_points,
            if assembly.precision == Precision::Float32 { 8 } else { 16 },
        ])?;
        let psi_samples = if ctx.offloads(sample_bytes) {
            let archived = ComplexSamples::archived_copy(ctx, psi_slice, assembly.precision)?;
            drop(psi_buf);
            archived
        } else {
            match assembly.precision {
                Precision::Float64 => ComplexSamples::Float64(psi_buf),
                Precision::Float32 => {
                    let mut f32_buf = ctx.allocate::<Complex32>(psi_slice.len())?;
                    for (dst, src) in f32_buf.as_mut_slice().iter_mut().zip(psi_slice) {
                        *dst = Complex32::new(src.re as f32, src.im as f32);
                    }
                    drop(psi_buf);
                    ComplexSamples::Float32(f32_buf)
                }
            }
        };

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
