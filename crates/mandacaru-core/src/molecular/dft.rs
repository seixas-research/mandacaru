//! Restricted and collinear unrestricted Kohn–Sham with native XC and Poisson.
use super::{
    invalid, occupation,
    scf::{self, Candidate, Diis, Iteration},
    MeanField, MolecularIntegrals, ScfOptions,
};
use crate::math::fft::Poisson;
use crate::math::gemm::{self, MatrixMut, MatrixRef};
use crate::math::storage::{with_rows, RealSamples, SampleBlocks};
use crate::models::xc;
use crate::{product, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;
type Matrix = DMatrix<C64>;

pub(super) fn grid_densities(integrals: &MolecularIntegrals, d: &[Matrix]) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let mut rho = integrals.context.allocate::<f64>(2 * n)?;
    let mut blocks = SampleBlocks::new(&integrals.context, &integrals.orthonormal_samples, m, n)?;
    // A small point tile bounds the extra workspace while allowing a BLAS
    // contraction of the density matrix with all orbitals in that tile.
    let tile = 256.min(blocks.block());
    let mut contracted = integrals.context.allocate::<f64>(product(&[tile, m])?)?;
    let mut density = integrals
        .context
        .allocate::<f64>(product(&[m, m, d.len()])?)?;
    for (spin, matrix) in d.iter().enumerate() {
        for b in 0..m {
            for a in 0..m {
                density.as_mut_slice()[spin * m * m + a + b * m] = matrix[(a, b)].re;
            }
        }
    }
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (phi, w) = (blocks.values(), blocks.stride());
        // A restricted state stores one spin density; evaluating it twice
        // gives exactly the same values and doubles the matrix contraction.
        for s in 0..if d.len() == 1 { 1 } else { 2 } {
            let matrix = &density.as_slice()[s * m * m..(s + 1) * m * m];
            for offset in (0..count).step_by(tile) {
                let width = tile.min(count - offset);
                gemm::multiply(
                    MatrixRef {
                        values: &phi[offset..],
                        rows: width,
                        columns: m,
                        leading: w,
                    },
                    MatrixRef {
                        values: matrix,
                        rows: m,
                        columns: m,
                        leading: m,
                    },
                    MatrixMut {
                        values: contracted.as_mut_slice(),
                        rows: width,
                        columns: m,
                        leading: tile,
                    },
                    1.,
                    0.,
                )?;
                for i in 0..width {
                    let value = (0..m)
                        .map(|a| phi[a * w + offset + i] * contracted.as_slice()[a * tile + i])
                        .sum::<f64>();
                    // Roundoff in a positive semidefinite density may be slightly negative.
                    rho.as_mut_slice()[s * n + start + offset + i] = value.max(0.);
                }
            }
        }
        if d.len() == 1 {
            for i in 0..count {
                rho.as_mut_slice()[n + start + i] = rho.as_slice()[start + i];
            }
        }
    }
    Ok(rho)
}
/// What one start of [`MolecularIntegrals::iterate_start`] reached.
pub(crate) struct StartRun {
    pub coefficients: Vec<Matrix>,
    pub densities: Vec<Matrix>,
    pub history: Vec<Iteration>,
    pub converged: bool,
    /// Occupations (0 to 1) of each channel's columns at the last iteration.
    pub occupations: Vec<Vec<f64>>,
    /// `-width S` of the occupations that built `densities` (0 without a [`Filling`]): the
    /// Mermin free energy is the Kohn-Sham energy plus this.
    pub entropy: f64,
}
/// Fermi-Dirac filling of one start of [`MolecularIntegrals::iterate_start`]: the levels of the
/// start's orbitals per channel, the width (Hartree), and the electrons per orbital a channel
/// stands for in the entropy (2 for a spin-restricted channel, 1 otherwise).
pub(crate) struct Filling<'a> {
    pub levels: &'a [Vec<f64>],
    pub width: f64,
    pub degeneracy: f64,
}
struct Evaluation {
    f: Vec<Matrix>,
    energy: f64,
    hartree: f64,
    xc: f64,
    fields: Option<Fields>,
    hubbard_energy: Option<f64>,
    hubbard_report: Option<Vec<crate::molecular::hubbard::MolecularHubbardReport>>,
}
/// The grid fields behind one evaluation that a derivative of the energy needs:
/// the valence spin densities, the Hartree potential of their sum and the
/// exchange-correlation potentials (`terms`) of the densities with the core.
pub(super) struct Fields {
    pub(super) valence: Buffer<f64>,
    pub(super) hartree: Buffer<f64>,
    pub(super) terms: xc::Xc,
}
/// The Fock matrices and grid fields of orthonormal-frame spin densities.
pub(super) fn response(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    name: &str,
) -> Result<(Vec<Matrix>, Fields)> {
    let (fock, fields, _) = response_with_energy(integrals, d, name)?;
    Ok((fock, fields))
}
/// The same evaluation also provides the energy, without repeating the SCF work.
pub(super) fn response_with_energy(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    name: &str,
) -> Result<(Vec<Matrix>, Fields, f64)> {
    let (fock, fields, energy, _) = response_with_parts(integrals, d, name)?;
    Ok((fock, fields, energy))
}
/// [`response_with_energy`] and the Hartree energy of the same evaluation.
pub(super) fn response_with_parts(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    name: &str,
) -> Result<(Vec<Matrix>, Fields, f64, f64)> {
    let screened = integrals.screening(name)?;
    let gradients = if crate::models::functional(name)?.requires_tau {
        Some(orbital_gradients(integrals)?)
    } else {
        None
    };
    response_with_shared(integrals, d, name, gradients.as_ref(), screened.as_ref())
}
/// [`response_with_parts`] with the orbital gradients of a meta-GGA and the screening of a
/// hybrid built by the caller, once for every evaluation that shares them.
pub(super) fn response_with_shared(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    name: &str,
    gradients: Option<&RealSamples>,
    screened: Option<&Screening>,
) -> Result<(Vec<Matrix>, Fields, f64, f64)> {
    let mut poisson = Poisson::isolated(&integrals.context, &integrals.grid)?;
    let evaluation = evaluate_with(
        integrals,
        &integrals.kohn_sham_core(),
        d,
        name,
        &mut poisson,
        gradients,
        screened,
        true,
        None,
    )?;
    let fields = evaluation
        .fields
        .ok_or_else(|| invalid("the evaluation retained no grid fields"))?;
    Ok((evaluation.f, fields, evaluation.energy, evaluation.hartree))
}
/// Orbital gradients `[axis][orbital][point]`, stored in the calculation's
/// precision. Each row is differentiated in float64 from the widened stored
/// orbital and rounded once.
pub(super) fn orbital_gradients(integrals: &MolecularIntegrals) -> Result<RealSamples> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let ctx = &integrals.context;
    let mut gradients = RealSamples::zeros(ctx, integrals.precision(), product(&[3, m, n])?)?;
    let mut phi = ctx.allocate::<f64>(n)?;
    let mut row = ctx.allocate::<f64>(n)?;
    for a in 0..m {
        integrals
            .orthonormal_samples
            .widen_range(a * n, phi.as_mut_slice())?;
        for axis in 0..3 {
            xc::derivative(&integrals.grid, phi.as_slice(), axis, row.as_mut_slice())?;
            gradients.fill_row(axis * m + a, n, row.as_slice())?;
        }
    }
    Ok(gradients)
}
/// The real parts of `d`, column-major, one `m x m` block each.
fn column_major(ctx: &crate::Context, d: &[Matrix], m: usize) -> Result<Buffer<f64>> {
    let mut out = ctx.allocate::<f64>(product(&[m, m, d.len()])?)?;
    for (c, matrix) in d.iter().enumerate() {
        for b in 0..m {
            for a in 0..m {
                out.as_mut_slice()[c * m * m + a + b * m] = matrix[(a, b)].re;
            }
        }
    }
    Ok(out)
}
/// `out(i, b) = sum_a left(i, a) matrix(a, b)` for the `width` points of a tile: `left` holds
/// `m` sample rows of stride `w`, `matrix` is column-major and `out` has the leading stride `tile`.
fn tile_product(
    left: &[f64],
    w: usize,
    width: usize,
    m: usize,
    matrix: &[f64],
    tile: usize,
    out: &mut [f64],
) -> Result<()> {
    gemm::multiply(
        MatrixRef {
            values: left,
            rows: width,
            columns: m,
            leading: w,
        },
        MatrixRef {
            values: matrix,
            rows: m,
            columns: m,
            leading: m,
        },
        MatrixMut {
            values: out,
            rows: width,
            columns: m,
            leading: tile,
        },
        1.,
        0.,
    )
}
pub(super) fn grid_kinetic(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &RealSamples,
) -> Result<Buffer<f64>> {
    let spins = [&d[0], &d[if d.len() == 1 { 0 } else { 1 }]].map(|m| m.clone());
    let mut tau = kinetic_channels(integrals, &spins, gradients)?;
    for value in tau.as_mut_slice() {
        *value = value.max(0.);
    }
    Ok(tau)
}
/// `tau_c = 1/2 sum_ab Re M_c[a, b] grad phi_a . grad phi_b` of each matrix of `d`, one block of the
/// grid each, unclamped (a spinor's magnetization channels take either sign).
pub(super) fn kinetic_channels(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &RealSamples,
) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    if gradients.len() != product(&[3, m, n])? {
        return Err(invalid("orbital gradients have inconsistent dimensions"));
    }
    let ctx = &integrals.context;
    let mut tau = ctx.allocate::<f64>(product(&[d.len(), n])?)?;
    let mut blocks = SampleBlocks::new(ctx, gradients, 3 * m, n)?;
    let tile = 256.min(blocks.block());
    let mut contracted = ctx.allocate::<f64>(product(&[tile, m])?)?;
    let matrices = column_major(ctx, d, m)?;
    for start in (0..n).step_by(blocks.block()) {
        let count = blocks.load(start)?;
        let (gradients, w) = (blocks.values(), blocks.stride());
        for c in 0..d.len() {
            let matrix = &matrices.as_slice()[c * m * m..(c + 1) * m * m];
            for offset in (0..count).step_by(tile) {
                let width = tile.min(count - offset);
                let out =
                    &mut tau.as_mut_slice()[c * n + start + offset..c * n + start + offset + width];
                out.fill(0.);
                for axis in 0..3 {
                    let slope = &gradients[axis * m * w + offset..];
                    // `(G D)(i, b)` for the tile, then `1/2 sum_b G(i, b) (G D)(i, b)`.
                    tile_product(slope, w, width, m, matrix, tile, contracted.as_mut_slice())?;
                    for (i, out) in out.iter_mut().enumerate() {
                        *out += 0.5
                            * (0..m)
                                .map(|b| slope[b * w + i] * contracted.as_slice()[b * tile + i])
                                .sum::<f64>();
                    }
                }
            }
        }
    }
    Ok(tau)
}
/// Valence density gradient `2 sum_ab D_ab phi_a grad phi_b` of both spin
/// channels (`[spin][axis][point]`) from the orbitals' own gradients: the
/// product rule that makes `|grad rho|^2 / 8 rho` of one orbital equal the
/// `tau` of `grid_kinetic`. The grid derivative of the sampled density does not
/// obey the product rule where the density is not resolved, and the mismatch
/// drives the iso-orbital indicator of a meta-GGA negative.
pub(super) fn grid_density_gradients(
    integrals: &MolecularIntegrals,
    d: &[Matrix],
    gradients: &RealSamples,
) -> Result<Buffer<f64>> {
    let m = integrals.m();
    let n = integrals.grid.index.points();
    if gradients.len() != product(&[3, m, n])? {
        return Err(invalid("orbital gradients have inconsistent dimensions"));
    }
    let ctx = &integrals.context;
    let mut field = ctx.allocate::<f64>(product(&[6, n])?)?;
    let mut orbitals = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
    let mut slopes = SampleBlocks::new(ctx, gradients, 3 * m, n)?;
    let tile = 256.min(orbitals.block());
    let mut contracted = ctx.allocate::<f64>(product(&[tile, m])?)?;
    let matrices = column_major(ctx, d, m)?;
    for start in (0..n).step_by(orbitals.block()) {
        let count = orbitals.load(start)?;
        slopes.load(start)?;
        let (phi, gradients, w) = (orbitals.values(), slopes.values(), orbitals.stride());
        for offset in (0..count).step_by(tile) {
            let width = tile.min(count - offset);
            for spin in 0..2 {
                let c = if d.len() == 1 { 0 } else { spin };
                let matrix = &matrices.as_slice()[c * m * m..(c + 1) * m * m];
                // `(phi D)(i, b)` for the tile, then `2 sum_b (phi D)(i, b) grad phi_b(i)`.
                tile_product(
                    &phi[offset..],
                    w,
                    width,
                    m,
                    matrix,
                    tile,
                    contracted.as_mut_slice(),
                )?;
                for axis in 0..3 {
                    let slope = &gradients[axis * m * w + offset..];
                    let at = (spin * 3 + axis) * n + start + offset;
                    for (i, out) in field.as_mut_slice()[at..at + width].iter_mut().enumerate() {
                        *out = 2.
                            * (0..m)
                                .map(|b| slope[b * w + i] * contracted.as_slice()[b * tile + i])
                                .sum::<f64>();
                    }
                }
            }
        }
    }
    Ok(field)
}
fn evaluate(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
    orbital_gradients: Option<&RealSamples>,
    screened: Option<&Screening>,
) -> Result<Evaluation> {
    evaluate_with(
        integrals,
        h,
        d,
        name,
        poisson,
        orbital_gradients,
        screened,
        false,
        None,
    )
}
fn evaluate_scratch(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
    orbital_gradients: Option<&RealSamples>,
    screened: Option<&Screening>,
    scratch: Option<(&mut [C64], &mut [C64])>,
) -> Result<Evaluation> {
    evaluate_with(
        integrals,
        h,
        d,
        name,
        poisson,
        orbital_gradients,
        screened,
        false,
        scratch,
    )
}
#[allow(clippy::too_many_arguments)]
fn evaluate_with(
    integrals: &MolecularIntegrals,
    h: &Matrix,
    d: &[Matrix],
    name: &str,
    poisson: &mut Poisson,
    orbital_gradients: Option<&RealSamples>,
    screened: Option<&Screening>,
    retain: bool,
    scratch: Option<(&mut [C64], &mut [C64])>,
) -> Result<Evaluation> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let mut rho = grid_densities(integrals, d)?;
    let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
    let (mut total_buf, mut hartree_buf);
    let (total_slice, hartree_slice) = match scratch {
        Some((t, h)) => (&mut t[..], &mut h[..]),
        None => {
            total_buf = ctx.allocate::<C64>(n)?;
            hartree_buf = ctx.allocate::<C64>(n)?;
            (total_buf.as_mut_slice(), hartree_buf.as_mut_slice())
        }
    };
    for k in 0..n {
        total_slice[k] = C64::new(rho.as_slice()[k] + rho.as_slice()[n + k], 0.);
    }
    poisson.solve_into(total_slice, hartree_slice)?;
    let mut kinetic = orbital_gradients
        .map(|gradient| grid_kinetic(integrals, d, gradient))
        .transpose()?;
    let valence = if retain {
        Some(ctx.copy_slice(rho.as_slice())?)
    } else {
        None
    };
    if let Some(paw) = paw {
        for s in 0..2 {
            for k in 0..n {
                rho.as_mut_slice()[s * n + k] += 0.5 * paw.core.as_slice()[k];
                if let Some(tau) = kinetic.as_mut() {
                    tau.as_mut_slice()[s * n + k] += 0.5 * paw.core_tau.as_slice()[k];
                }
            }
        }
    }
    // A meta-GGA sees the orbitals' own density gradient, valence by the
    // product rule and the partial core (half per spin) by the grid.
    let mut gradient = orbital_gradients
        .filter(|_| crate::models::functional(name).is_ok_and(|f| f.requires_tau))
        .map(|orbital| grid_density_gradients(integrals, d, orbital))
        .transpose()?;
    if let (Some(gradient), Some(paw)) = (gradient.as_mut(), paw) {
        let mut core = ctx.allocate::<f64>(n)?;
        for axis in 0..3 {
            xc::derivative(
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
    let terms = xc::evaluate_fields_relativistic(
        ctx,
        &integrals.grid,
        rho.as_slice(),
        kinetic.as_ref().map(|v| v.as_slice()),
        gradient.as_ref().map(|v| v.as_slice()),
        name,
        paw.is_some_and(|p| p.relativistic_exchange)
            && matches!(crate::models::functional(name)?.semilocal, "lda" | "pbe"),
    )?;
    let mut eh = 0.5
        * total_slice
            .iter()
            .zip(hartree_slice.iter())
            .map(|(r, v)| (r * v).re)
            .sum::<f64>()
        * dv;
    let mut augmentation = Matrix::zeros(m, m);
    if let Some(paw) = paw {
        for (factor, weight) in paw.pairs.factors.iter().zip(&paw.pairs.weights) {
            let moment = d
                .iter()
                .map(|density| {
                    (0..m)
                        .map(|a| {
                            (0..m)
                                .map(|b| density[(a, b)].re * factor[(b, a)])
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                * if d.len() == 1 { 2. } else { 1. };
            eh += 0.5 * weight * moment * moment;
            for a in 0..m {
                for b in 0..m {
                    augmentation[(a, b)] += weight * moment * factor[(a, b)];
                }
            }
        }
    }
    let mut f = vec![];
    let mut exact_energy = 0.;
    // The local matrix elements accumulate over point blocks widened to
    // float64; each (a, b) sum runs over the points in ascending order.
    let mut sums = vec![DMatrix::<f64>::zeros(m, m); d.len()];
    {
        let mut orbitals = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, n)?;
        let mut slopes = orbital_gradients
            .map(|gradients| SampleBlocks::new(ctx, gradients, 3 * m, n))
            .transpose()?;
        for start in (0..n).step_by(orbitals.block()) {
            let count = orbitals.load(start)?;
            if let Some(slopes) = slopes.as_mut() {
                slopes.load(start)?;
            }
            let (phi, w) = (orbitals.values(), orbitals.stride());
            let gradient = slopes.as_ref().map(|s| s.values());
            if gradient.is_none() {
                let mut w_buf = vec![0.0; count];
                for (s, sum) in sums.iter_mut().enumerate() {
                    let pot = &terms.potentials.as_slice()[s * n..];
                    for b in 0..m {
                        let phi_b = &phi[b * w..b * w + count];
                        for i in 0..count {
                            let k = start + i;
                            w_buf[i] = phi_b[i] * (hartree_slice[k].re + pot[k]);
                        }
                        for a in 0..=b {
                            let phi_a = &phi[a * w..a * w + count];
                            let mut dot = 0.0;
                            for i in 0..count {
                                dot += phi_a[i] * w_buf[i];
                            }
                            sum[(a, b)] += dot;
                            if a != b {
                                sum[(b, a)] += dot;
                            }
                        }
                    }
                }
            } else {
                for (s, sum) in sums.iter_mut().enumerate() {
                    for a in 0..m {
                        for b in 0..m {
                            let mut value = sum[(a, b)];
                            for i in 0..count {
                                let k = start + i;
                                let local = phi[a * w + i]
                                    * phi[b * w + i]
                                    * (hartree_slice[k].re
                                        + terms.potentials.as_slice()[s * n + k]);
                                let meta = if let (Some(gradient), Some(tau)) =
                                    (gradient, terms.tau_potentials.as_ref())
                                {
                                    0.5 * tau.as_slice()[s * n + k]
                                        * (0..3)
                                            .map(|axis| {
                                                gradient[(axis * m + a) * w + i]
                                                    * gradient[(axis * m + b) * w + i]
                                            })
                                            .sum::<f64>()
                                } else {
                                    0.
                                };
                                // dE/dD_ab through the product-rule density gradient.
                                let flux = if let (Some(gradient), Some(flux)) =
                                    (gradient, terms.gradient_potentials.as_ref())
                                {
                                    (0..3)
                                        .map(|axis| {
                                            flux.as_slice()[(s * 3 + axis) * n + k]
                                                * (phi[a * w + i]
                                                    * gradient[(axis * m + b) * w + i]
                                                    + gradient[(axis * m + a) * w + i]
                                                        * phi[b * w + i])
                                        })
                                        .sum::<f64>()
                                } else {
                                    0.
                                };
                                value += local + meta + flux;
                            }
                            sum[(a, b)] = value;
                        }
                    }
                }
            }
        }
    }
    // An open shell's spheres carry the polarization of the short-range semilocal
    // exchange that the hybrid replaces; a restricted density has none.
    let polarization = match screened {
        Some(Screening {
            spin: Some(spin), ..
        }) if d.len() == 2 => {
            let (up, down) = (d[0].map(|v| v.re), d[1].map(|v| v.re));
            Some(spin.polarization(ctx, &up, &down)?)
        }
        _ => None,
    };
    let hubbard_eval = if let Some(hubbard) = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
    {
        Some(hubbard.evaluate(ctx, d)?)
    } else {
        None
    };
    for s in 0..d.len() {
        let v = Matrix::from_fn(m, m, |a, b| C64::new(sums[s][(a, b)] * dv, 0.));
        let mut fock = h + v + &augmentation;
        if let Some((_, delta_focks, _)) = &hubbard_eval {
            fock += Matrix::from_fn(m, m, |a, b| C64::new(delta_focks[s][(a, b)], 0.));
        }
        if let Some(screening) = screened {
            let spins = if d.len() == 1 { 2. } else { 1. };
            let (_, exchange) = scf::jk(&screening.eri, m, &d[s]);
            exact_energy -=
                0.5 * screening.fraction * (d[s].clone() * &exchange).trace().re * spins;
            fock -= exchange * C64::new(screening.fraction, 0.);
            if let Some(frozen) = &screening.one_body {
                let frozen = frozen.map(|v| C64::new(v, 0.));
                exact_energy -= screening.fraction * (d[s].clone() * &frozen).trace().re * spins;
                fock -= frozen * C64::new(screening.fraction, 0.);
            }
            if let Some((_, spheres)) = &polarization {
                fock -= spheres[s].map(|v| C64::new(v, 0.)) * C64::new(screening.fraction, 0.);
            }
        }
        f.push(scf::hermitian(fock));
    }
    if let Some(screening) = screened {
        exact_energy -= screening.fraction * screening.constant;
        if let Some((energy, _)) = &polarization {
            exact_energy -= screening.fraction * energy;
        }
    }
    let eone =
        d.iter().map(|d| (d * h).trace().re).sum::<f64>() * if d.len() == 1 { 2. } else { 1. };
    let hubbard_energy = hubbard_eval.as_ref().map(|(e, _, _)| *e);
    let mut energy = eone + eh + terms.energy + exact_energy + paw.map_or(0., |p| p.constant);
    if let Some(e_u) = hubbard_energy {
        energy += e_u;
    }
    let (hartree_energy, xc_energy) = (eh, terms.energy + exact_energy);
    let fields = match valence {
        Some(valence) => {
            let mut potential = ctx.allocate::<f64>(n)?;
            for (to, &from) in potential
                .as_mut_slice()
                .iter_mut()
                .zip(hartree_slice.iter())
            {
                *to = from.re;
            }
            Some(Fields {
                valence,
                hartree: potential,
                terms,
            })
        }
        None => None,
    };
    Ok(Evaluation {
        f,
        energy,
        hartree: hartree_energy,
        xc: xc_energy,
        fields,
        hubbard_energy,
        hubbard_report: hubbard_eval.map(|(_, _, r)| r),
    })
}
/// A screened hybrid's exact exchange: the short-range pair integrals, the
/// exact-exchange fraction and, on PAW-LCAO, the frozen short-range semilocal
/// exchange of the spheres (a one-body matrix and a constant).
pub(super) struct Screening {
    pub(super) eri: std::sync::Arc<super::integrals::Eri>,
    pub(super) fraction: f64,
    pub(super) one_body: Option<DMatrix<f64>>,
    pub(super) constant: f64,
    /// The spin dependence of the spheres' semilocal exchange (PAW-LCAO only).
    pub(super) spin: Option<crate::paw::molecular::SpinTerms>,
    /// How many correction factors of `eri` are the one-center tensor.
    pub(super) one_center_factors: usize,
}
impl MolecularIntegrals {
    /// The screened exact exchange of `name`, or `None` for a semilocal functional.
    /// On PAW-LCAO the pair integrals carry the compensation charges and the
    /// one-center tensor under the screened kernel, never the bare one.
    pub(super) fn screening(&self, name: &str) -> Result<Option<Screening>> {
        let Some((omega, fraction)) = crate::models::functional(name)?.screened_exchange else {
            return Ok(None);
        };
        let (ctx, m) = (&self.context, self.m());
        let mut solver = Poisson::screened(ctx, &self.grid, omega)?;
        let base = std::sync::Arc::new(with_rows!(self.orthonormal_samples.rows(), |samples| {
            super::integrals::cholesky_from_grid(
                ctx,
                &self.grid,
                samples,
                m,
                1e-12,
                &mut solver,
                None,
            )
        })?);
        let Some(_) = self.corrections.as_ref() else {
            return Ok(Some(Screening {
                eri: base,
                fraction,
                one_body: None,
                constant: 0.,
                spin: None,
                one_center_factors: 0,
            }));
        };
        let setup = self.paw_setup.as_ref().ok_or_else(|| {
            invalid("a screened hybrid needs the PAW setup that rebuilds its geometry; custom molecular corrections have none")
        })?;
        let screened = setup.screened_exchange(self, omega)?;
        Ok(Some(Screening {
            eri: std::sync::Arc::new(super::integrals::Eri::Augmented {
                m,
                base,
                correction: screened.pairs,
            }),
            fraction,
            one_body: Some(screened.one_body),
            constant: screened.constant,
            spin: Some(screened.spin),
            one_center_factors: screened.one_center_factors,
        }))
    }
}
impl MolecularIntegrals {
    /// One-electron matrix of the Kohn-Sham model: the shared core operator
    /// plus the frozen atomic one-center Kohn-Sham difference of a PAW setup.
    pub(super) fn kohn_sham_core(&self) -> Matrix {
        let m = self.m();
        Matrix::from_fn(m, m, |a, b| {
            C64::new(
                self.one.as_slice()[a * m + b]
                    + self
                        .corrections
                        .as_ref()
                        .and_then(|c| c.dft.as_ref())
                        .map_or(0., |p| p.one[(a, b)]),
                0.,
            )
        })
    }
    /// Electronic Kohn-Sham energy of orthonormal-frame spin densities, which
    /// need not be idempotent. Nuclear repulsion is not included. A screened
    /// hybrid's exact exchange and its PAW spheres' terms are included.
    pub fn kohn_sham_energy(&self, functional: &str, d: &[Matrix]) -> Result<f64> {
        let definition = crate::models::functional(functional)?;
        let m = self.m();
        if d.is_empty() || d.len() > 2 || d.iter().any(|d| d.shape() != (m, m)) {
            return Err(invalid("Kohn-Sham energy needs one or two spin densities"));
        }
        let ctx = &self.context;
        let screened = self.screening(functional)?;
        let mut poisson = Poisson::isolated(ctx, &self.grid)?;
        let gradients = if definition.requires_tau {
            Some(orbital_gradients(self)?)
        } else {
            None
        };
        Ok(evaluate(
            self,
            &self.kohn_sham_core(),
            d,
            functional,
            &mut poisson,
            gradients.as_ref(),
            screened.as_ref(),
        )?
        .energy)
    }
    /// One start of a Kohn-Sham SCF, for any state shape: `start` are the coefficient matrices (one
    /// per spin channel, or one 2m x 2m for a spinor) holding `electrons` per channel, and `step`
    /// evaluates a density list to its Fock list and energy. Plain DIIS first; a start that does
    /// not converge is retried with a level shift switched on by the first rise of the energy
    /// (a degenerate open shell can flip its occupied orbital between iterations).
    ///
    /// With a [`Filling`] every channel holds Fermi-Dirac occupations of the levels of the matrix
    /// just diagonalized; without one, every iteration fills by aufbau. Under a level shift
    /// those levels are the shifted ones, which keeps the occupied set from flipping (the
    /// purpose of the shift; filling from the unshifted levels instead was measured to undo
    /// it, all-electron OH at a 0.5 Angstrom grid oscillating for 300 iterations). The shift
    /// moves a fractional level by `s (1 - f)`, so a shifted state is checked against the
    /// unshifted Fock afterwards (the aufbau test of `kohn_sham_smeared`).
    pub(crate) fn iterate_start(
        &self,
        start: &[Matrix],
        electrons: &[usize],
        filling: Option<Filling<'_>>,
        options: &ScfOptions,
        mut step: impl FnMut(&[Matrix]) -> Result<(Vec<Matrix>, f64)>,
    ) -> Result<StartRun> {
        let fill = |levels: Option<&[Vec<f64>]>, c: &[Matrix]| -> Result<Vec<Vec<f64>>> {
            c.iter()
                .zip(electrons)
                .enumerate()
                .map(|(k, (c, &n))| match (&filling, levels) {
                    (Some(filling), Some(levels)) => {
                        occupation::fermi_dirac(&levels[k], n, filling.width)
                    }
                    _ => Ok(occupation::aufbau(c.ncols(), n)),
                })
                .collect()
        };
        // `-width S`, both spins of a restricted channel included.
        let entropy_term = |filled: &[Vec<f64>]| -> f64 {
            filling.as_ref().map_or(0., |filling| {
                -filling.width
                    * filling.degeneracy
                    * filled.iter().map(|f| occupation::entropy(f)).sum::<f64>()
            })
        };
        let density = |c: &[Matrix], f: &[Vec<f64>]| -> Vec<Matrix> {
            c.iter()
                .zip(f)
                .map(|(c, f)| occupation::density(c, f))
                .collect()
        };
        let start_levels = filling.as_ref().map(|f| f.levels);
        let mut c = start.to_vec();
        let mut filled = fill(start_levels, &c)?;
        let mut d = density(&c, &filled);
        let mut history = vec![];
        let mut converged = false;
        for retry in [false, true] {
            if retry {
                c = start.to_vec();
                filled = fill(start_levels, &c)?;
                d = density(&c, &filled);
                history = vec![];
            }
            let mut previous: Option<f64> = None;
            let mut previous_free: Option<f64> = None;
            let mut mixer = Diis::new();
            let mut shift = 0.;
            for iteration in 1..=options.max_iterations {
                let (f, energy) = step(&d)?;
                let free = energy + entropy_term(&filled);
                let gradient = scf::stationarity(&f, &d);
                if retry && previous_free.is_some_and(|p| free > p + 1e-10) && shift == 0. {
                    shift = 0.5;
                }
                let mut effective = if options.diis {
                    mixer.extrapolate(&f, &d)
                } else {
                    f
                };
                if shift > 0. {
                    for (f, d) in effective.iter_mut().zip(&d) {
                        *f += (Matrix::identity(f.nrows(), f.nrows()) - d) * C64::new(shift, 0.);
                    }
                }
                let solved = effective
                    .into_iter()
                    .map(crate::math::linalg::eigen)
                    .collect::<Result<Vec<_>>>()?;
                let levels: Vec<Vec<f64>> = solved.iter().map(|(e, _)| e.clone()).collect();
                c = solved.into_iter().map(|(_, c)| c).collect();
                let levels = filling.as_ref().map(|_| levels);
                filled = fill(levels.as_deref(), &c)?;
                let next = density(&c, &filled);
                let residual = d
                    .iter()
                    .zip(&next)
                    .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).norm()))
                    .fold(0f64, f64::max);
                let delta = previous.map(|p| energy - p);
                let delta_free = previous_free.map(|p| free - p);
                history.push(Iteration {
                    iteration,
                    energy,
                    delta,
                    residual,
                });
                d = next;
                previous = Some(energy);
                previous_free = Some(free);
                if delta.is_some_and(|de: f64| de.abs() < options.tolerance)
                    && delta_free.is_some_and(|df: f64| df.abs() < options.tolerance)
                    && residual < options.tolerance
                    && gradient < options.tolerance
                {
                    converged = true;
                    break;
                }
            }
            if converged {
                break;
            }
        }
        let entropy = entropy_term(&filled);
        Ok(StartRun {
            coefficients: c,
            densities: d,
            history,
            converged,
            occupations: filled,
            entropy,
        })
    }

    /// Restricted or collinear unrestricted Kohn-Sham with Fermi-Dirac occupations at
    /// [`occupation::DEFAULT_WIDTH`] ([`Self::kohn_sham_at_width`]).
    pub fn kohn_sham(
        &self,
        functional: &str,
        particles: [usize; 2],
        unrestricted: bool,
        options: &ScfOptions,
    ) -> Result<MeanField> {
        self.kohn_sham_smeared(
            functional,
            particles,
            unrestricted,
            options,
            "fermi-dirac",
            occupation::DEFAULT_WIDTH,
        )
    }
    /// Restricted or collinear unrestricted Kohn-Sham whose channels hold `smearing` occupations
    /// at `width` Hartree, one Fermi level per spin ([`occupation`]). A molecule takes
    /// Fermi-Dirac only: its zero-width limit is the ensemble of a degenerate shell, while the
    /// Gaussian and Methfessel-Paxton schemes are built for a continuous density of states.
    pub fn kohn_sham_smeared(
        &self,
        functional: &str,
        particles: [usize; 2],
        unrestricted: bool,
        options: &ScfOptions,
        smearing: &str,
        width: f64,
    ) -> Result<MeanField> {
        if self.paw_setup.as_ref().is_some_and(|s| s.has_spin_orbit()) {
            return Err(crate::Error::Unsupported(
                "a molecule with spin-orbit datasets needs spinor Kohn-Sham (spinor_kohn_sham); \
                 restricted and unrestricted Kohn-Sham cannot carry the spin-orbit term"
                    .into(),
            ));
        }
        self.collinear_kohn_sham(functional, particles, unrestricted, options, smearing, width)
    }
    /// Collinear Kohn-Sham without the spin-orbit refusal: the scalar-relativistic part of a
    /// spin-orbit dataset, for the collinear stage of a noncollinear start.
    pub(crate) fn collinear_kohn_sham(
        &self,
        functional: &str,
        particles: [usize; 2],
        unrestricted: bool,
        options: &ScfOptions,
        smearing: &str,
        width: f64,
    ) -> Result<MeanField> {
        let scheme = crate::periodic::spectrum::Smearing::parse(smearing)?;
        if scheme != crate::periodic::spectrum::Smearing::FermiDirac {
            return Err(crate::Error::Unsupported(format!(
                "a molecule takes Fermi-Dirac occupations only, not {} smearing",
                scheme.name()
            )));
        }
        if !width.is_finite() || width <= 0. {
            return Err(invalid("the Fermi-Dirac width must be positive and finite"));
        }
        if self.corrections.as_ref().is_some_and(|c| c.dft.is_none()) {
            return Err(invalid(
                "native molecular PAW Kohn-Sham augmentation is not yet implemented",
            ));
        }
        let definition = crate::models::functional(functional)?;
        self.context.run(|| {
            let m = self.m();
            let ctx = &self.context;
            let screened = self.screening(functional)?;
            let orbital_gradients = if definition.requires_tau {
                Some(orbital_gradients(self)?)
            } else {
                None
            };
            if particles.iter().any(|&v| v > m)
                || (!unrestricted && particles[0] != particles[1])
                || options.max_iterations == 0
                || options.guesses == 0
                || !options.tolerance.is_finite()
                || options.tolerance <= 0.
            {
                return Err(invalid(
                    "invalid Kohn-Sham occupation, iteration budget or tolerance",
                ));
            }
            let workspace = ctx.reserve(
                product(&[2 * m, 2 * m, 16, 160])?
                    .checked_add(product(&[options.max_iterations, 128, 4])?)
                    .ok_or(crate::Error::Overflow)?,
            )?;
            let h = self.kohn_sham_core();
            let (core_levels, core) = crate::math::linalg::eigen(h.clone())?;
            let channels = if unrestricted { 2 } else { 1 };
            // Each start with the levels of its orbitals, from which its first occupations are
            // filled.
            let mut starts: Vec<(Vec<Matrix>, Vec<Vec<f64>>)> = vec![(
                vec![core.clone(); channels],
                vec![core_levels.clone(); channels],
            )];
            let occupations = if unrestricted {
                particles.to_vec()
            } else {
                vec![particles[0]]
            };
            let mut poisson = Poisson::isolated(ctx, &self.grid)?;
            if options.guesses >= 2 {
                let d = starts[0]
                    .0
                    .iter()
                    .zip(&occupations)
                    .zip(&starts[0].1)
                    .map(|((c, &n), levels)| {
                        Ok(occupation::density(
                            c,
                            &occupation::fermi_dirac(levels, n, width)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let f = evaluate(
                    self,
                    &h,
                    &d,
                    functional,
                    &mut poisson,
                    orbital_gradients.as_ref(),
                    screened.as_ref(),
                )?
                .f;
                let solved = f
                    .into_iter()
                    .map(|f| crate::math::linalg::eigen((&h + f) * C64::new(0.5, 0.)))
                    .collect::<Result<Vec<_>>>()?;
                let levels = solved.iter().map(|(e, _)| e.clone()).collect();
                starts.push((solved.into_iter().map(|(_, c)| c).collect(), levels));
            }
            if unrestricted
                && options.guesses >= 3
                && particles[0] == particles[1]
                && particles[0] > 0
                && particles[0] < m
            {
                let i = particles[0] - 1;
                let a = particles[0];
                let mut pair = vec![core.clone(), core.clone()];
                for (spin, c) in pair.iter_mut().enumerate() {
                    let theta = if spin == 0 { 0.3f64 } else { -0.3f64 };
                    for k in 0..m {
                        c[(k, i)] = core[(k, i)] * theta.cos() + core[(k, a)] * theta.sin();
                        c[(k, a)] = -core[(k, i)] * theta.sin() + core[(k, a)] * theta.cos();
                    }
                }
                // The deliberately broken-symmetry start: its levels are those of the core
                // Hamiltonian in the rotated orbitals.
                let levels = pair
                    .iter()
                    .map(|c| {
                        (0..m)
                            .map(|j| (c.column(j).adjoint() * &h * c.column(j))[(0, 0)].re)
                            .collect()
                    })
                    .collect();
                starts.push((pair, levels));
            }
            type Best = (
                Candidate,
                f64,
                f64,
                Option<(f64, Vec<crate::molecular::hubbard::MolecularHubbardReport>)>,
                Vec<Vec<f64>>,
            );
            // Converged and aufbau, then converged, then the lower free energy (what the SCF
            // minimizes).
            let healthy = |c: &Candidate| c.converged && c.aufbau_inversion.is_none();
            let better = |c: &Candidate, b: &Candidate| {
                let rank = |c: &Candidate| (c.converged, healthy(c));
                rank(c) > rank(b)
                    || (rank(c) == rank(b) && c.energy + c.entropy < b.energy + b.entropy - 1e-12)
            };
            // Every start at one width; the best candidate, with the levels of its columns.
            let solve = |starts: Vec<(Vec<Matrix>, Vec<Vec<f64>>)>,
                         width: f64,
                         poisson: &mut Poisson|
             -> Result<Best> {
                let mut best: Option<Best> = None;
                for (start, levels) in starts {
                    let n_pts = self.grid.index.points();
                    let mut total_scratch = ctx.allocate::<C64>(n_pts)?;
                    let mut hartree_scratch = ctx.allocate::<C64>(n_pts)?;
                    let run = self.iterate_start(
                        &start,
                        &occupations,
                        Some(Filling {
                            levels: &levels,
                            width,
                            degeneracy: if unrestricted { 1. } else { 2. },
                        }),
                        options,
                        |d: &[Matrix]| {
                            let value = evaluate_scratch(
                                self,
                                &h,
                                d,
                                functional,
                                poisson,
                                orbital_gradients.as_ref(),
                                screened.as_ref(),
                                Some((
                                    total_scratch.as_mut_slice(),
                                    hartree_scratch.as_mut_slice(),
                                )),
                            )?;
                            Ok((value.f, value.energy))
                        },
                    )?;
                    let value = evaluate(
                        self,
                        &h,
                        &run.densities,
                        functional,
                        poisson,
                        orbital_gradients.as_ref(),
                        screened.as_ref(),
                    )?;
                    // The aufbau test: a level holding more than a lower one (unshifted Fock), which
                    // Fermi-Dirac occupations never do; a level shift can hold such a state. It is
                    // kept, labeled, and a candidate that passes is preferred.
                    let mut inversion: Option<f64> = None;
                    let mut column_levels = Vec::with_capacity(run.occupations.len());
                    // Occupations in the order of the levels, which is the order of `energies`.
                    let mut ordered = Vec::with_capacity(run.occupations.len());
                    for ((f, c), filled) in
                        value.f.iter().zip(&run.coefficients).zip(&run.occupations)
                    {
                        let levels: Vec<f64> = (0..c.ncols())
                            .map(|j| (c.column(j).adjoint() * f * c.column(j))[(0, 0)].re)
                            .collect();
                        if let Some((i, a)) =
                            occupation::non_aufbau(&levels, filled, scf::AUFBAU_TOLERANCE)
                        {
                            inversion = Some(inversion.unwrap_or(0f64).max(levels[i] - levels[a]));
                        }
                        let mut order: Vec<usize> = (0..levels.len()).collect();
                        order.sort_by(|&a, &b| levels[a].total_cmp(&levels[b]));
                        ordered.push(order.into_iter().map(|j| filled[j]).collect::<Vec<f64>>());
                        column_levels.push(levels);
                    }
                    let fractional = !ordered.iter().all(|f| occupation::is_integer(f));
                    let energies = value
                        .f
                        .into_iter()
                        .map(|f| crate::math::linalg::eigen(f).map(|(e, _)| e))
                        .collect::<Result<_>>()?;
                    let candidate = Candidate {
                        energy: value.energy,
                        converged: run.converged,
                        history: run.history,
                        coefficients: run.coefficients,
                        densities: run.densities,
                        energies,
                        unstable_pair: None,
                        occupations: fractional.then_some(ordered),
                        aufbau_inversion: inversion,
                        entropy: run.entropy,
                    };
                    let hubbard_data = match (value.hubbard_energy, value.hubbard_report) {
                        (Some(e), Some(r)) => Some((e, r)),
                        _ => None,
                    };
                    let candidate = (
                        candidate,
                        value.hartree,
                        value.xc,
                        hubbard_data,
                        column_levels,
                    );
                    if best.as_ref().is_none_or(|b| better(&candidate.0, &b.0)) {
                        best = Some(candidate);
                    }
                    if best.as_ref().is_some_and(|b| healthy(&b.0)) && !unrestricted {
                        break;
                    }
                }
                Ok(best.expect("at least one start"))
            };
            // The occupations at the target width, from every start. A stiff frontier (a
            // degenerate or nearly degenerate shell: the occupations respond to its splitting
            // as 1 / width) can defeat DIIS there and end unconverged or held by the level shift
            // on a state that is not aufbau. Then the occupations are annealed (Rabuck and
            // Scuseria, J. Chem. Phys. 110, 695 (1999)): converged at a wide, stable width, then
            // narrowed in steps of three, each step started from the last. The better of the two
            // states is kept.
            let mut best = solve(starts.clone(), width, &mut poisson)?;
            if !healthy(&best.0) && width < occupation::ANNEALING_START {
                let mut stage = occupation::ANNEALING_START;
                let mut current = solve(starts, stage, &mut poisson)?;
                while current.0.converged && stage > width {
                    stage = (stage / 3.).max(width);
                    let start = (current.0.coefficients.clone(), current.4.clone());
                    current = solve(vec![start], stage, &mut poisson)?;
                }
                if stage == width && better(&current.0, &best.0) {
                    best = current;
                }
            }
            let (candidate, eh, exc, hubbard_data, _) = best;
            let mut state = self.finish(
                if unrestricted { "uhf" } else { "rhf" },
                particles,
                options,
                candidate,
                workspace,
                Some((functional, eh, exc)),
                hubbard_data,
            )?;
            state.smearing_width = Some(width);
            Ok(state)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::SpatialGrid, Context, Geometry, LengthUnit};
    #[test]
    fn paw_core_and_signed_compensation_fock_differentiate_the_same_energy() {
        use crate::molecular::integrals::{MolecularCorrections, PairCorrection, PawKohnSham};
        use std::sync::Arc;
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        {
            let geometry = Geometry::new(
                vec![1, 1],
                vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
                [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
                [false; 3],
                LengthUnit::Angstrom,
            )
            .unwrap();
            let grid = SpatialGrid::from_geometry(&geometry, [1.; 3], false, false).unwrap();
            let n = grid.index.points();
            let mut integrals = MolecularIntegrals::new(
                &ctx,
                geometry,
                grid,
                r#"{"name":"HAO"}"#,
                "spectral",
                None,
                Default::default(),
            )
            .unwrap();
            let m = integrals.m();
            let mut core = ctx.allocate::<f64>(n).unwrap();
            let mut tau = ctx.allocate::<f64>(n).unwrap();
            for k in 0..n {
                let point = integrals.grid.point(k);
                let center = integrals.geometry.positions()[0];
                let r2 = (0..3).map(|a| (point[a] - center[a]).powi(2)).sum::<f64>();
                core.as_mut_slice()[k] = 0.2 * (-r2).exp();
                tau.as_mut_slice()[k] = 0.5 * r2 * core.as_slice()[k];
            }
            let one = DMatrix::from_row_slice(m, m, &[0.1, 0.03, 0.03, -0.2]);
            integrals.corrections = Some(MolecularCorrections {
                external: ctx.allocate(n).unwrap(),
                overlap: DMatrix::zeros(m, m),
                potential: DMatrix::zeros(m, m),
                charges: vec![1., 1.],
                electrons: 2,
                core_supported: true,
                hubbard: None,
                projections: None,
                electrostatic: None,
                ionic: None,
                pairs: Arc::new(PairCorrection {
                    factors: vec![],
                    weights: vec![],
                    memory: ctx.reserve(0).unwrap(),
                }),
                dft: Some(PawKohnSham {
                    one: one.clone(),
                    pairs: PairCorrection {
                        factors: vec![DMatrix::from_row_slice(m, m, &[0.2, 0.1, 0.1, -0.3])],
                        weights: vec![-0.4],
                        memory: ctx.reserve(32).unwrap(),
                    },
                    core,
                    core_tau: tau,
                    constant: 0.17,
                    relativistic_exchange: true,
                    memory: ctx.reserve(32).unwrap(),
                }),
            });
            let h = Matrix::from_fn(m, m, |a, b| {
                C64::new(integrals.one.as_slice()[a * m + b] + one[(a, b)], 0.)
            });
            let density = Matrix::from_row_slice(
                m,
                m,
                &[
                    C64::new(0.45, 0.),
                    C64::new(0.05, 0.01),
                    C64::new(0.05, -0.01),
                    C64::new(0.55, 0.),
                ],
            );
            let direction = Matrix::from_row_slice(
                m,
                m,
                &[
                    C64::new(0.3, 0.),
                    C64::new(0.2, 0.1),
                    C64::new(0.2, -0.1),
                    C64::new(-0.3, 0.),
                ],
            );
            let gradients = orbital_gradients(&integrals).unwrap();
            let mut poisson = Poisson::isolated(&ctx, &integrals.grid).unwrap();
            for functional in ["lda", "pbe", "r2scan"] {
                for unrestricted in [false, true] {
                    let densities = if unrestricted {
                        vec![density.clone(), &density * C64::new(0.7, 0.)]
                    } else {
                        vec![density.clone()]
                    };
                    let evaluated = evaluate(
                        &integrals,
                        &h,
                        &densities,
                        functional,
                        &mut poisson,
                        Some(&gradients),
                        None,
                    )
                    .unwrap();
                    let expected = (&evaluated.f[0] * &direction).trace().re
                        * if unrestricted { 1. } else { 2. };
                    let step = 1e-6;
                    let mut plus = densities.clone();
                    let mut minus = densities;
                    plus[0] += &direction * C64::new(step, 0.);
                    minus[0] -= &direction * C64::new(step, 0.);
                    let ep = evaluate(
                        &integrals,
                        &h,
                        &plus,
                        functional,
                        &mut poisson,
                        Some(&gradients),
                        None,
                    )
                    .unwrap()
                    .energy;
                    let em = evaluate(
                        &integrals,
                        &h,
                        &minus,
                        functional,
                        &mut poisson,
                        Some(&gradients),
                        None,
                    )
                    .unwrap()
                    .energy;
                    assert!(
                        ((ep - em) / (2. * step) - expected).abs() < 2e-8,
                        "{functional} unrestricted={unrestricted}"
                    );
                }
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
    #[test]
    fn meta_gga_fock_is_the_derivative_of_its_discrete_density_energy() {
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
            [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
            [false; 3],
            LengthUnit::Angstrom,
        )
        .unwrap();
        let grid = super::super::SpatialGrid::from_geometry(
            &geometry,
            [0.5 / 0.529177210903; 3],
            false,
            false,
        )
        .unwrap();
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )
        .unwrap();
        let m = integrals.m();
        let gradients = orbital_gradients(&integrals).unwrap();
        let h = Matrix::from_fn(m, m, |a, b| {
            C64::new(integrals.one.as_slice()[a * m + b], 0.)
        });
        let density = Matrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            C64::new(0.45, 0.),
            C64::new(0.55, 0.),
        ]));
        let direction = Matrix::from_row_slice(
            2,
            2,
            &[
                C64::new(0.3, 0.),
                C64::new(0.2, 0.),
                C64::new(0.2, 0.),
                C64::new(-0.3, 0.),
            ],
        );
        let mut poisson = Poisson::isolated(&ctx, &integrals.grid).unwrap();
        let evaluate_density = |d: Matrix, p: &mut Poisson| {
            evaluate(&integrals, &h, &[d], "r2scan", p, Some(&gradients), None).unwrap()
        };
        let base = evaluate_density(density.clone(), &mut poisson);
        let step = 1e-5;
        let plus = evaluate_density(&density + &direction * C64::new(step, 0.), &mut poisson);
        let minus = evaluate_density(&density - &direction * C64::new(step, 0.), &mut poisson);
        let finite = (plus.energy - minus.energy) / (2. * step);
        let analytic = 2. * (&direction * &base.f[0]).trace().re;
        assert!((finite - analytic).abs() < 2e-7, "{finite} vs {analytic}");
    }
    #[test]
    fn a_one_orbital_density_is_iso_orbital_with_the_orbital_product_rule_gradient() {
        let ctx = Context::new(Some(1), 64 << 20).unwrap();
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
            [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
            [false; 3],
            LengthUnit::Angstrom,
        )
        .unwrap();
        let grid = super::super::SpatialGrid::from_geometry(
            &geometry,
            [0.5 / 0.529177210903; 3],
            false,
            false,
        )
        .unwrap();
        let integrals = MolecularIntegrals::new(
            &ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )
        .unwrap();
        let n = integrals.grid.index.points();
        // One orbital per spin: rho = psi^2, so tau = |grad psi|^2 / 2 and the
        // Weizsaecker kinetic density |grad rho|^2 / 8 rho must equal it.
        let orbital = [0.6, 0.8];
        let density = Matrix::from_fn(2, 2, |a, b| C64::new(orbital[a] * orbital[b], 0.));
        let d = [density];
        let gradients = orbital_gradients(&integrals).unwrap();
        let tau = grid_kinetic(&integrals, &d, &gradients).unwrap();
        let product_rule = grid_density_gradients(&integrals, &d, &gradients).unwrap();
        let rho = grid_densities(&integrals, &d).unwrap();
        let mut stencil = vec![0.; 3 * n];
        for axis in 0..3 {
            xc::derivative(
                &integrals.grid,
                &rho.as_slice()[..n],
                axis,
                &mut stencil[axis * n..(axis + 1) * n],
            )
            .unwrap();
        }
        let weizsaecker = |gradient: &dyn Fn(usize) -> f64, k: usize| {
            (0..3)
                .map(|axis| gradient(axis * n + k).powi(2))
                .sum::<f64>()
                / (8. * rho.as_slice()[k])
        };
        let (mut worst, mut violated, mut counted) = (0f64, 0, 0);
        for k in 0..n {
            if rho.as_slice()[k] < 1e-12 {
                continue;
            }
            counted += 1;
            let t = tau.as_slice()[k];
            worst = worst.max((weizsaecker(&|i| product_rule.as_slice()[i], k) - t).abs() / t);
            if t - weizsaecker(&|i| stencil[i], k) < -1e-3 * t {
                violated += 1;
            }
        }
        assert!(counted > 1000);
        assert!(worst < 1e-9, "product rule: |tau_W - tau| / tau = {worst}");
        // The grid derivative of the sampled density breaks that identity
        // where the density is not resolved: the indicator would be negative.
        assert!(violated > 0, "the stencil should violate tau_W <= tau");
    }
}
