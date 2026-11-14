//! The meta-GGA (r2SCAN) grid part of the response kernel.
//!
//! The molecular meta-GGA sees the orbitals' own density gradient `g_s = 2 sum_ab D_ab phi_a grad
//! phi_b` and kinetic density `tau_s = 1/2 sum_ab D_ab grad phi_a . grad phi_b`, and its Fock matrix
//! is `sum_r [phi_a phi_b v_s + 1/2 v_tau grad phi_a . grad phi_b + F_s . (phi_a grad phi_b + grad
//! phi_a phi_b)] dv` with the flux `F_s = df / dg_s` (`molecular::dft::evaluate_with`). For a trial
//! block `y` (`dD = sum y_ia (|i><a| + |a><i|)`) the changes are `dn = 2 sum y phi_i phi_a`, `dg =
//! 2 sum y (phi_i grad phi_a + grad phi_i phi_a)`, `dtau = sum y grad phi_i . grad phi_a`; the
//! kernel ([`MetaKernel`]) turns them into the changes of `(v, v_tau, F)`, and [`elements`] takes
//! those back to the occupied-empty block with the same contraction as the Fock matrix.
use super::{invalid, response::view};
use crate::{
    math::storage::SampleBlocks,
    models::xc_kernel::MetaKernel,
    molecular::{dft, MolecularIntegrals},
    product,
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;

const BLOCK: usize = 2048;

/// The gradients of the canonical orbitals on the grid, `[axis][orbital][point]`.
pub(super) fn gradient_samples(
    integrals: &MolecularIntegrals,
    orbitals: &DMatrix<f64>,
) -> Result<Buffer<f64>> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let gradients = dft::orbital_gradients(integrals)?;
    let mut out = ctx.allocate::<f64>(product(&[3, n, m])?)?;
    let mut blocks = SampleBlocks::new(ctx, &gradients, 3 * m, n)?;
    let stride = blocks.stride();
    for start in (0..n).step_by(stride) {
        let count = blocks.load(start)?;
        for axis in 0..3 {
            let samples = nalgebra::DMatrixView::from_slice_with_strides_generic(
                &blocks.values()[axis * m * stride..],
                nalgebra::Dyn(count),
                nalgebra::Dyn(m),
                nalgebra::Const::<1>,
                nalgebra::Dyn(stride),
            );
            let mo = samples * orbitals;
            for j in 0..m {
                for i in 0..count {
                    out.as_mut_slice()[(axis * m + j) * n + start + i] = mo[(i, j)];
                }
            }
        }
    }
    Ok(out)
}

/// The r2SCAN kernel at the spin density matrices `densities` (one restricted or two), with the
/// valence grid densities `valence` (`[spin][point]`) of the same evaluation: the densities,
/// kinetic densities and gradients exactly as the potential saw them (the smooth core added half
/// to each spin, its gradient by the grid stencil).
pub(super) fn ground_kernel(
    integrals: &MolecularIntegrals,
    densities: &[DMatrix<C64>],
    valence: &[f64],
) -> Result<MetaKernel> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let gradients = dft::orbital_gradients(integrals)?;
    let mut tau = dft::grid_kinetic(integrals, densities, &gradients)?;
    let mut gradient = dft::grid_density_gradients(integrals, densities, &gradients)?;
    let mut rho = ctx.copy_slice(valence)?;
    if let Some(paw) = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref()) {
        let mut core = ctx.allocate::<f64>(n)?;
        for s in 0..2 {
            for k in 0..n {
                rho.as_mut_slice()[s * n + k] += 0.5 * paw.core.as_slice()[k];
                tau.as_mut_slice()[s * n + k] += 0.5 * paw.core_tau.as_slice()[k];
            }
        }
        for axis in 0..3 {
            crate::models::xc::derivative(
                &integrals.grid,
                paw.core.as_slice(),
                axis,
                core.as_mut_slice(),
            )?;
            for s in 0..2 {
                for k in 0..n {
                    gradient.as_mut_slice()[(s * 3 + axis) * n + k] += 0.5 * core.as_slice()[k];
                }
            }
        }
    }
    MetaKernel::new(ctx, rho.as_slice(), tau.as_slice(), gradient.as_slice())
}

/// The changes `(dn, dg, dtau)` of one spin for the trial block `y` (occupied x empty), from the
/// orbital values `samples` (`[orbital][point]`) and gradients `gradients` (`[axis][orbital][point]`)
/// of `n` points and `m` orbitals: `density` (`n`), `gradient` (`3 n`, axis-major), `tau` (`n`).
#[allow(clippy::too_many_arguments)]
pub(super) fn changes(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    o: usize,
    y: &DMatrix<f64>,
    density: &mut [f64],
    gradient: &mut [f64],
    tau: &mut [f64],
) -> Result<()> {
    let v = m - o;
    if y.shape() != (o, v) || density.len() != n || gradient.len() != 3 * n || tau.len() != n {
        return Err(invalid("a meta-GGA change does not match the grid"));
    }
    let yt = y.transpose();
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let po = view(samples, start, count, o, n);
        let pv = view(samples, o * n + start, count, v, n);
        let t = pv * &yt;
        for i in 0..count {
            density[start + i] = 2. * (0..o).map(|j| po[(i, j)] * t[(i, j)]).sum::<f64>();
            tau[start + i] = 0.;
        }
        for axis in 0..3 {
            let gpo = view(gradients, axis * m * n + start, count, o, n);
            let gpv = view(gradients, (axis * m + o) * n + start, count, v, n);
            let tg = gpv * &yt;
            for i in 0..count {
                let mut g = 0.;
                let mut k = 0.;
                for j in 0..o {
                    g += po[(i, j)] * tg[(i, j)] + gpo[(i, j)] * t[(i, j)];
                    k += gpo[(i, j)] * tg[(i, j)];
                }
                gradient[axis * n + start + i] = 2. * g;
                tau[start + i] += k;
            }
        }
    }
    Ok(())
}

/// `sum_r [phi_i phi_a local + 1/2 tau grad phi_i . grad phi_a + flux . (phi_i grad phi_a + grad
/// phi_i phi_a)] dv`: the occupied-empty block of the meta-GGA potential changes of one spin.
#[allow(clippy::too_many_arguments)]
pub(super) fn elements(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    o: usize,
    local: &[f64],
    tau: &[f64],
    flux: &[f64],
    dv: f64,
) -> Result<DMatrix<f64>> {
    let v = m - o;
    if local.len() != n || tau.len() != n || flux.len() != 3 * n {
        return Err(invalid("meta-GGA potential changes do not match the grid"));
    }
    let mut out = DMatrix::zeros(o, v);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let po = view(samples, start, count, o, n);
        let pv = view(samples, o * n + start, count, v, n);
        out += po.transpose() * DMatrix::from_fn(count, v, |i, a| pv[(i, a)] * local[start + i]);
        for axis in 0..3 {
            let gpo = view(gradients, axis * m * n + start, count, o, n);
            let gpv = view(gradients, (axis * m + o) * n + start, count, v, n);
            let f = &flux[axis * n..(axis + 1) * n];
            out += gpo.transpose()
                * DMatrix::from_fn(count, v, |i, a| {
                    0.5 * gpv[(i, a)] * tau[start + i] + pv[(i, a)] * f[start + i]
                });
            out += po.transpose() * DMatrix::from_fn(count, v, |i, a| gpv[(i, a)] * f[start + i]);
        }
    }
    Ok(out * dv)
}
