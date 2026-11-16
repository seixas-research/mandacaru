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
//!
//! The antisymmetric channel: with complex orbitals the functional sees the gauge-invariant
//! `tau~_s = tau_s - |j_s|^2 / (2 n_s)` (`molecular::dft`, the real-time path), whose second
//! variation in `Im D` at the real ground state (`j = 0`) adds
//! `(A - B)_{ia,jb} += -2 int (v_tau,s / n_s) j^{ia} . j^{jb}`, `j^{ia} = (phi_i grad phi_a -
//! phi_a grad phi_i) / 2`, same spin only ([`current_elements`], TDDFT_PLAN.md 13.6.1, vera fix
//! 1); `A + B` is untouched.
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

/// The fields the ground-state meta-GGA potential saw at the spin density matrices `densities`
/// (one restricted or two), with the valence grid densities `valence` (`[spin][point]`) of the
/// same evaluation: the densities, kinetic densities and gradients (the smooth core added half to
/// each spin, its gradient by the grid stencil).
fn ground_fields(
    integrals: &MolecularIntegrals,
    densities: &[DMatrix<C64>],
    valence: &[f64],
) -> Result<(Buffer<f64>, Buffer<f64>, Buffer<f64>)> {
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
    Ok((rho, tau, gradient))
}

/// The r2SCAN kernel at the spin density matrices `densities` (one restricted or two), with the
/// valence grid densities `valence` of the same evaluation ([`ground_fields`]).
pub(super) fn ground_kernel(
    integrals: &MolecularIntegrals,
    densities: &[DMatrix<C64>],
    valence: &[f64],
) -> Result<MetaKernel> {
    let (rho, tau, gradient) = ground_fields(integrals, densities, valence)?;
    MetaKernel::new(
        &integrals.context,
        rho.as_slice(),
        tau.as_slice(),
        gradient.as_slice(),
    )
}

/// `v_tau,s / n_s` of the ground state of `functional` (`[spin][point]`, zero where the density
/// is not positive): the weight of the current-density term that the gauge-invariant kinetic
/// density `tau~ = tau - |j|^2 / (2 n)` adds to `A - B` (TDDFT_PLAN.md 13.6.1, vera fix 1), from
/// the fields [`ground_kernel`] linearizes. The ground state carries no current, so `A + B` and the
/// ground state are untouched.
pub(super) fn current_weights(
    integrals: &MolecularIntegrals,
    functional: &str,
    densities: &[DMatrix<C64>],
    valence: &[f64],
) -> Result<Buffer<f64>> {
    let (rho, tau, gradient) = ground_fields(integrals, densities, valence)?;
    weights_of(integrals, functional, &rho, &tau, &gradient)
}

/// `v_tau,s / n_s` of `functional` at the ground fields ([`current_weights`]).
fn weights_of(
    integrals: &MolecularIntegrals,
    functional: &str,
    rho: &Buffer<f64>,
    tau: &Buffer<f64>,
    gradient: &Buffer<f64>,
) -> Result<Buffer<f64>> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let terms = crate::models::xc::evaluate_fields_relativistic(
        ctx,
        &integrals.grid,
        rho.as_slice(),
        Some(tau.as_slice()),
        Some(gradient.as_slice()),
        functional,
        false,
    )?;
    let v_tau = terms
        .tau_potentials
        .ok_or_else(|| invalid("a meta-GGA evaluation returned no kinetic potential"))?;
    let mut out = ctx.allocate::<f64>(product(&[2, n])?)?;
    for (k, value) in out.as_mut_slice().iter_mut().enumerate() {
        let density = rho.as_slice()[k];
        *value = if density > 0. {
            v_tau.as_slice()[k] / density
        } else {
            0.
        };
    }
    Ok(out)
}

/// The ground state of a meta-GGA reference as the excited-state gradient differentiates it: the
/// fields its potential saw ([`ground_fields`]: densities and kinetic densities `[spin][point]`,
/// density gradients `[spin][axis][point]`, the smooth core included), its kernel and the
/// current-density weights `v_tau / n` ([`current_weights`]).
pub(super) struct MetaGround {
    pub rho: Buffer<f64>,
    pub tau: Buffer<f64>,
    pub gradient: Buffer<f64>,
    pub kernel: MetaKernel,
    pub weights: Buffer<f64>,
}

impl MetaGround {
    pub(super) fn new(
        integrals: &MolecularIntegrals,
        functional: &str,
        densities: &[DMatrix<C64>],
        valence: &[f64],
    ) -> Result<Self> {
        let (rho, tau, gradient) = ground_fields(integrals, densities, valence)?;
        let kernel = MetaKernel::new(
            &integrals.context,
            rho.as_slice(),
            tau.as_slice(),
            gradient.as_slice(),
        )?;
        let weights = weights_of(integrals, functional, &rho, &tau, &gradient)?;
        Ok(Self {
            rho,
            tau,
            gradient,
            kernel,
            weights,
        })
    }
}

/// `-2 sum_r w(r) j^{ia}(r) . J(r) dv` with `J = sum_jb y_jb j^{jb}` and `j^{ia} = (phi_i grad
/// phi_a - phi_a grad phi_i) / 2`: the current-density part of `(A - B) y` of one spin, with the
/// weights `w = v_tau / n` of [`current_weights`] (`n` points).
#[allow(clippy::too_many_arguments)]
pub(super) fn current_elements(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    o: usize,
    y: &DMatrix<f64>,
    weights: &[f64],
    dv: f64,
) -> Result<DMatrix<f64>> {
    let v = m - o;
    if y.shape() != (o, v) || weights.len() != n {
        return Err(invalid("a current-density kernel does not match the grid"));
    }
    let yt = y.transpose();
    let mut out = DMatrix::zeros(o, v);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let po = view(samples, start, count, o, n);
        let pv = view(samples, o * n + start, count, v, n);
        let t = pv * &yt;
        for axis in 0..3 {
            let gpo = view(gradients, axis * m * n + start, count, o, n);
            let gpv = view(gradients, (axis * m + o) * n + start, count, v, n);
            let tg = gpv * &yt;
            // w J_axis, J_axis = 1/2 sum_j (phi_j (dphi_v y^T)_j - dphi_j (phi_v y^T)_j).
            let wj: Vec<f64> = (0..count)
                .map(|i| {
                    0.5 * weights[start + i]
                        * (0..o)
                            .map(|j| po[(i, j)] * tg[(i, j)] - gpo[(i, j)] * t[(i, j)])
                            .sum::<f64>()
                })
                .collect();
            // -2 w J . (phi_i dphi_a - phi_a dphi_i) / 2.
            out -= po.transpose() * DMatrix::from_fn(count, v, |i, a| wj[i] * gpv[(i, a)]);
            out += gpo.transpose() * DMatrix::from_fn(count, v, |i, a| wj[i] * pv[(i, a)]);
        }
    }
    Ok(out * dv)
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

/// The changes `(dn, dg, dtau)` of one spin for a general symmetric orbital matrix `x` (`m x m`):
/// `dn = sum_pq x_pq phi_p phi_q`, `dg = 2 sum_pq x_pq phi_p grad phi_q`,
/// `dtau = 1/2 sum_pq x_pq grad phi_p . grad phi_q` ([`changes`] for every block).
#[allow(clippy::too_many_arguments)]
pub(super) fn general_changes(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    x: &DMatrix<f64>,
    density: &mut [f64],
    gradient: &mut [f64],
    tau: &mut [f64],
) -> Result<()> {
    if x.shape() != (m, m) || density.len() != n || gradient.len() != 3 * n || tau.len() != n {
        return Err(invalid("a meta-GGA change does not match the grid"));
    }
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let phi = view(samples, start, count, m, n);
        let px = phi * x;
        for i in 0..count {
            density[start + i] = (0..m).map(|q| px[(i, q)] * phi[(i, q)]).sum::<f64>();
            tau[start + i] = 0.;
        }
        for axis in 0..3 {
            let g = view(gradients, axis * m * n + start, count, m, n);
            let gx = g * x;
            for i in 0..count {
                let (mut slope, mut kinetic) = (0., 0.);
                for q in 0..m {
                    slope += px[(i, q)] * g[(i, q)];
                    kinetic += gx[(i, q)] * g[(i, q)];
                }
                gradient[axis * n + start + i] = 2. * slope;
                tau[start + i] += 0.5 * kinetic;
            }
        }
    }
    Ok(())
}

/// `sum_r [phi_p phi_q local + 1/2 tau grad phi_p . grad phi_q + flux . (phi_p grad phi_q + grad
/// phi_p phi_q)] dv`: every block of the meta-GGA potential parts of one spin ([`elements`] for
/// every block).
#[allow(clippy::too_many_arguments)]
pub(super) fn general_elements(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    local: &[f64],
    tau: &[f64],
    flux: &[f64],
    dv: f64,
) -> Result<DMatrix<f64>> {
    if local.len() != n || tau.len() != n || flux.len() != 3 * n {
        return Err(invalid("meta-GGA potential parts do not match the grid"));
    }
    let mut out = DMatrix::zeros(m, m);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let phi = view(samples, start, count, m, n);
        out += phi.transpose() * DMatrix::from_fn(count, m, |i, q| phi[(i, q)] * local[start + i]);
        for axis in 0..3 {
            let g = view(gradients, axis * m * n + start, count, m, n);
            let f = &flux[axis * n..(axis + 1) * n];
            let mixed =
                g.transpose() * DMatrix::from_fn(count, m, |i, q| phi[(i, q)] * f[start + i]);
            out += &mixed + mixed.transpose();
            out +=
                g.transpose() * DMatrix::from_fn(count, m, |i, q| 0.5 * g[(i, q)] * tau[start + i]);
        }
    }
    Ok(out * dv)
}

/// The paramagnetic current `j = sum_pq x_qp phi_p grad phi_q` (`[axis][point]`) of the imaginary
/// part `x` (antisymmetric, `m x m`) of one spin's density matrix, as `molecular::dft` sees it.
pub(super) fn current(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    x: &DMatrix<f64>,
    out: &mut [f64],
) -> Result<()> {
    if x.shape() != (m, m) || out.len() != 3 * n {
        return Err(invalid("a current does not match the grid"));
    }
    let xt = x.transpose();
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let px = view(samples, start, count, m, n) * &xt;
        for axis in 0..3 {
            let g = view(gradients, axis * m * n + start, count, m, n);
            for i in 0..count {
                out[axis * n + start + i] = (0..m).map(|q| px[(i, q)] * g[(i, q)]).sum::<f64>();
            }
        }
    }
    Ok(())
}

/// `sum_r field . (phi_p grad phi_q - phi_q grad phi_p) dv` (`m x m`, antisymmetric): the
/// imaginary Fock matrix of a current field (`field = (v_tau / 2 n) j`, `[axis][point]`).
pub(super) fn current_fock(
    samples: &[f64],
    gradients: &[f64],
    n: usize,
    m: usize,
    field: &[f64],
    dv: f64,
) -> Result<DMatrix<f64>> {
    if field.len() != 3 * n {
        return Err(invalid("a current field does not match the grid"));
    }
    let mut out = DMatrix::zeros(m, m);
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let phi = view(samples, start, count, m, n);
        for axis in 0..3 {
            let g = view(gradients, axis * m * n + start, count, m, n);
            let f = &field[axis * n..(axis + 1) * n];
            let mixed =
                phi.transpose() * DMatrix::from_fn(count, m, |i, q| g[(i, q)] * f[start + i]);
            out += &mixed - mixed.transpose();
        }
    }
    Ok(out * dv)
}

/// The actions `A_p = u . grad chi_p + div(u chi_p)` (`[orbital][point]`), `chi_p = sum_q x_qp
/// phi_q`, whose contraction `sum_p int A_p dphi_p` with the basis derivative of the orbitals is
/// the explicit derivative of `int u . j` at fixed `x` (`j` of [`current`], the grid derivative
/// antisymmetric): the force of a current-density term.
#[allow(clippy::too_many_arguments)]
pub(super) fn current_actions(
    integrals: &MolecularIntegrals,
    samples: &[f64],
    gradients: &[f64],
    m: usize,
    x: &DMatrix<f64>,
    u: &[f64],
) -> Result<Buffer<f64>> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    if x.shape() != (m, m) || u.len() != 3 * n {
        return Err(invalid("a current action does not match the grid"));
    }
    let mut out = ctx.allocate::<f64>(product(&[m, n])?)?;
    let mut chi = ctx.allocate::<f64>(product(&[m, n])?)?;
    for start in (0..n).step_by(BLOCK) {
        let count = BLOCK.min(n - start);
        let px = view(samples, start, count, m, n) * x;
        for p in 0..m {
            for i in 0..count {
                chi.as_mut_slice()[p * n + start + i] = px[(i, p)];
            }
        }
        for axis in 0..3 {
            let gx = view(gradients, axis * m * n + start, count, m, n) * x;
            for p in 0..m {
                for i in 0..count {
                    out.as_mut_slice()[p * n + start + i] += u[axis * n + start + i] * gx[(i, p)];
                }
            }
        }
    }
    let mut product_field = ctx.allocate::<f64>(n)?;
    let mut divergence = ctx.allocate::<f64>(n)?;
    for p in 0..m {
        for axis in 0..3 {
            for k in 0..n {
                product_field.as_mut_slice()[k] = u[axis * n + k] * chi.as_slice()[p * n + k];
            }
            crate::models::xc::derivative(
                &integrals.grid,
                product_field.as_slice(),
                axis,
                divergence.as_mut_slice(),
            )?;
            for k in 0..n {
                out.as_mut_slice()[p * n + k] += divergence.as_slice()[k];
            }
        }
    }
    Ok(out)
}
