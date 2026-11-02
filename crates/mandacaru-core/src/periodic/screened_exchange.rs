//! Periodic screened exact exchange operator for hybrid functionals (HSE06).
//! Evaluates the non-singular reciprocal Fock exchange matrix K(k)
//! contracted with the self-consistent density matrices over the k-mesh.
//! Fully supports restricted, unrestricted collinear, and 2-component spinor calculations.
//!
//! The pair densities are the smooth pair on the FFT grid and, when an
//! [`Augmentation`] is given, the PAW compensation charges added in reciprocal space before the
//! kernel weight (so the Gygi-Baldereschi head acts on the unit-charge pair). The `*_with`
//! entries take the range separation explicitly and the augmentation, with the one-center model
//! attached; the SCF, the bands and the Wannier frames reach them through `Crystal::hybrid_exchange`
//! and `Crystal::hybrid_exchange_at_k` (`hybrid_setup.rs`), which hold the crystal's spheres.

use super::{
    invalid,
    screened_augmentation::{add_compensation, Augmentation, OneCenter, OneCenterTerms},
    Crystal, KPoint,
};
use crate::{
    math::{fft::Fft3, reciprocal, storage::MatrixSource},
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

/// Divergence treatment scheme for periodic exact exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExxDivTreatment {
    /// Standard non-singular point evaluation at Q = 0: lim_{Q->0} v^omega(Q) = pi / omega^2.
    #[default]
    None,
    /// Gygi-Baldereschi spherical Brillouin-zone integration:
    /// analytical spherical average of v^omega(Q) over the equivalent k-point Voronoi sphere.
    GygiBaldereschi,
}

impl ExxDivTreatment {
    pub fn parse(s: &str) -> Result<Self> {
        // The two canonical spellings only: a shorthand would be an alias.
        match s {
            "none" => Ok(Self::None),
            "gygi-baldereschi" => Ok(Self::GygiBaldereschi),
            other => Err(crate::Error::Invalid(format!(
                "unknown exxdiv_treatment: '{other}', expected 'none' or 'gygi-baldereschi'"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::GygiBaldereschi => "gygi-baldereschi",
        }
    }
}

/// Reciprocal screened Coulomb kernel v^omega(Q) = 4 pi / Q^2 * (1 - exp(-Q^2 / (4 omega^2))).
/// Strictly finite as Q -> 0: lim_{Q -> 0} v^omega(Q) = pi / omega^2.
#[inline]
pub fn screened_kernel(q2: f64, omega: f64) -> f64 {
    if q2 > 1e-12 {
        4.0 * PI * (1.0 - (-q2 / (4.0 * omega * omega)).exp()) / q2
    } else {
        PI / (omega * omega)
    }
}

/// Gygi-Baldereschi divergence treatment for the screened Coulomb interaction.
/// Evaluates the analytic spherical average of v^omega(Q) over the k-point Voronoi sphere
/// of radius q_c = (6 pi^2 w_q / Omega_cell)^{1/3}:
///
///   v_bar^omega = 12 pi / q_c^2 * [ 1 - omega sqrt(pi) / q_c * erf(q_c / (2 omega)) ]
#[inline]
pub fn gygi_baldereschi_screened_kernel(qc: f64, omega: f64) -> f64 {
    if qc <= 1e-12 {
        return if omega > 1e-12 {
            PI / (omega * omega)
        } else {
            0.0
        };
    }
    if omega <= 1e-12 {
        // Bare Coulomb limit: integral of 4 pi / Q^2 over sphere of radius qc
        return 12.0 * PI / (qc * qc);
    }
    let u = qc / (2.0 * omega);
    if u < 1e-3 {
        // High-order Taylor series for small u = q_c / (2 omega) to avoid cancellation:
        // 1 - sqrt(pi)/(2u) erf(u) = u^2/3 - u^4/10 + u^6/42 - u^8/216 + u^10/1320
        // Multiplying by 12 pi / q_c^2 = 3 pi / (omega^2 u^2):
        // v_bar = pi / omega^2 * (1 - 3/10 u^2 + 9/70 u^4 - 1/24 u^6 + 9/440 u^8)
        let u2 = u * u;
        let u4 = u2 * u2;
        let u6 = u4 * u2;
        let u8 = u4 * u4;
        let w2 = omega * omega;
        (PI / w2) * (1.0 - 0.3 * u2 + (9.0 / 70.0) * u4 - (1.0 / 24.0) * u6 + (9.0 / 440.0) * u8)
    } else {
        let erf_val = libm::erf(u);
        let pref = 12.0 * PI / (qc * qc);
        pref * (1.0 - (omega * PI.sqrt() / qc) * erf_val)
    }
}

/// Range separation and exact-exchange fraction of a hybrid. The public entries without an
/// explicit one read them from the functional registry ([`Screening::of`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Screening {
    pub omega: f64,
    pub fraction: f64,
}

impl Screening {
    pub fn of(c: &Crystal) -> Result<Self> {
        let (omega, fraction) = c.require_screened_exchange()?;
        Ok(Self { omega, fraction })
    }
}

/// The exchange matrices of every (spin channel, k-point) of the mesh and the exact exchange
/// energy `E_x = -1/2 fraction sum_k w_k tr(D_k K_k)`.
pub struct Exchange {
    pub operators: Vec<Matrix>,
    pub energy: f64,
}

/// The pieces of the exchange energy of an augmented pair density `rho = rho_grid + rho_comp`:
/// the quadratic form of `|rho|^2` splits into the grid pair alone, the cross term and the
/// compensation charges alone. `augmentation()` is what the compensation adds to the grid.
#[derive(Clone, Copy, Debug)]
pub struct EnergyParts {
    pub pseudo: f64,
    pub cross: f64,
    pub compensation: f64,
}

impl EnergyParts {
    pub fn total(&self) -> f64 {
        self.pseudo + self.cross + self.compensation
    }
    pub fn augmentation(&self) -> f64 {
        self.cross + self.compensation
    }
}

/// Which part of the augmented pair density a build keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Part {
    Pseudo,
    Compensation,
    Both,
}

pub(super) type Stage<'a> = Option<(&'a Augmentation<'a>, Part)>;

/// Occupied Bloch states at a given q-point for one spin channel or spinor.
pub(super) struct OccupiedStates {
    pub(super) weight: f64,
    // For each occupied state: [up] for collinear, [up, down] for spinor
    pub(super) wavefunctions: Vec<Vec<Buffer<C64>>>,
    // For each occupied state: its coefficients over the basis, the square root of the
    // occupation folded in (`components * m`).
    pub(super) coefficients: Vec<Vec<C64>>,
    // For each occupied state: `<p_j | psi>` per component, filled only for an augmented build.
    pub(super) projected: Vec<Vec<C64>>,
}

/// Extracts occupied Bloch wavefunctions on the real-space grid for a k-point q.
fn extract_occupied(
    c: &Crystal,
    q_idx: usize,
    d_mat: &Matrix,
    spin_scale: f64,
) -> Result<OccupiedStates> {
    let m = c.m;
    let n = c.grid.index.points();
    let point = &c.points[q_idx];
    let spinor = c.spinor;

    // Scale density matrix by spin_scale (0.5 for closed shell, 1.0 for unrestricted or spinor)
    let d_spin = d_mat * C64::new(spin_scale, 0.);
    let (eigenvals, eigenvecs) = crate::math::linalg::eigen(d_spin)?;

    // Read basis samples at point q
    let q_samples = point.psi.read(&c.context)?;

    // Precompute phase array: exp(-i q . r)
    let mut phase = c.context.allocate::<C64>(n)?;
    for (k, val) in phase.as_mut_slice().iter_mut().enumerate() {
        let r = c.grid.point(k);
        let phase_arg = -(0..3).map(|a| point.k[a] * r[a]).sum::<f64>();
        *val = C64::from_polar(1.0, phase_arg);
    }

    let mut wavefunctions = Vec::new();
    let mut coefficients = Vec::new();
    let n_spin_components = if spinor { 2 } else { 1 };

    for (j, &val) in eigenvals.iter().enumerate() {
        if val <= 1e-10 {
            continue;
        }
        let occ_sqrt = val.sqrt();
        let mut components = Vec::with_capacity(n_spin_components);
        let mut folded = Vec::with_capacity(n_spin_components * m);

        for comp in 0..n_spin_components {
            let mut psi = c.context.allocate::<C64>(n)?;
            for mu in 0..m {
                let coeff = eigenvecs[(comp * m + mu, j)] * C64::new(occ_sqrt, 0.);
                folded.push(coeff);
                if coeff.norm_sqr() < 1e-24 {
                    continue;
                }
                let row = &q_samples[mu * n..(mu + 1) * n];
                for k in 0..n {
                    psi.as_mut_slice()[k] += coeff * row[k] * phase.as_slice()[k];
                }
            }
            components.push(psi);
        }
        wavefunctions.push(components);
        coefficients.push(folded);
    }

    Ok(OccupiedStates {
        weight: point.weight,
        wavefunctions,
        coefficients,
        projected: Vec::new(),
    })
}

/// The occupied states of every k-point of one spin channel, with their projections when the
/// build is augmented.
pub(super) fn occupied_states<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    spin: usize,
    spin_scale: f64,
    stage: Stage,
) -> Result<Vec<OccupiedStates>> {
    let nk = c.points.len();
    let mut all = Vec::with_capacity(nk);
    for q in 0..nk {
        let d = matrices[spin * nk + q].matrix(&c.context)?;
        let mut occ = extract_occupied(c, q, &d, spin_scale)?;
        if let Some((augmentation, _)) = stage {
            occ.projected = occ
                .coefficients
                .iter()
                .map(|coefficients| augmentation.project(c, &c.points[q], coefficients))
                .collect();
        }
        all.push(occ);
    }
    Ok(all)
}

/// The Coulomb weight of one (k, q) pair on the grid: `sqrt(w_q dv / N v^omega(Q))`.
pub(super) fn kernel_weights(
    c: &Crystal,
    k_diff: [f64; 3],
    w_q: f64,
    omega: f64,
    exxdiv: ExxDivTreatment,
) -> Result<Buffer<f64>> {
    let n = c.grid.index.points();
    let dv = c.grid.index.dv();
    let cell_volume = dv * n as f64;
    let symbols = reciprocal::bloch(&c.context, &c.grid, k_diff)?;
    let mut sqrt_kernel_weight = c.context.allocate::<f64>(n)?;
    for at in 0..n {
        let q2 = symbols.squares.as_slice()[at];
        let v = if q2 > 1e-12 {
            screened_kernel(q2, omega)
        } else {
            match exxdiv {
                ExxDivTreatment::GygiBaldereschi => {
                    let qc = (6.0 * PI * PI * w_q / cell_volume).cbrt();
                    gygi_baldereschi_screened_kernel(qc, omega)
                }
                ExxDivTreatment::None => screened_kernel(0.0, omega),
            }
        };
        let factor = (w_q * dv / n as f64) * v;
        sqrt_kernel_weight.as_mut_slice()[at] = factor.max(0.0).sqrt();
    }
    Ok(sqrt_kernel_weight)
}

/// The periodic parts `u_mu(r) = chi_mu(r) exp(-i k.r)` of Bloch sums `samples` (`m * n`,
/// function-major) at wave vector `k`, written to `out`.
pub(super) fn periodic_part(c: &Crystal, k: [f64; 3], samples: &[C64], out: &mut [C64]) {
    let n = c.grid.index.points();
    for r_idx in 0..n {
        let r = c.grid.point(r_idx);
        let phase_arg = -(0..3).map(|a| k[a] * r[a]).sum::<f64>();
        let phase = C64::from_polar(1.0, phase_arg);
        for mu in 0..out.len() / n {
            out[mu * n + r_idx] = samples[mu * n + r_idx] * phase;
        }
    }
}

/// What one occupied state contributes to the pair densities of every basis function of a target
/// point: the periodic parts `u` of the target basis, the state's wave function `psi` per
/// spin component and, for an augmented build, its `moments` (`n_basis x channels`) against the
/// pair `shapes` (`channels x n`).
pub(super) struct PairInputs<'a> {
    pub u_k: &'a [C64],
    pub psi: &'a [Buffer<C64>],
    pub moments: Option<&'a DMatrix<C64>>,
    pub shapes: Option<&'a [C64]>,
    pub kernel: &'a [f64],
    pub part: Part,
}

/// `W_i = sqrt(kernel) (FFT[conj(u_i) psi] + sum_a moments[i, a] shapes[a])` for every basis
/// function `i` of the target (`n_basis * n`, row-major), the one assembly of the exchange
/// matrix and of its derivatives (the gradient calls it with tangent inputs).
pub(super) fn weighted_pairs(
    c: &Crystal,
    fft: &mut Fft3,
    pair_buf: &mut Buffer<C64>,
    input: &PairInputs,
    out: &mut [C64],
) -> Result<()> {
    let (m, n) = (c.m, c.grid.index.points());
    for i in 0..out.len() / n {
        let (comp, mu) = (i / m, i % m);
        if input.part == Part::Compensation {
            pair_buf.as_mut_slice().fill(C64::default());
        } else {
            let u_mu = &input.u_k[mu * n..(mu + 1) * n];
            let psi_comp = input.psi[comp].as_slice();
            for r_idx in 0..n {
                pair_buf.as_mut_slice()[r_idx] = u_mu[r_idx].conj() * psi_comp[r_idx];
            }
            fft.transform(pair_buf.as_mut_slice(), false)?;
        }
        if let (Some(moments), Some(shapes)) = (input.moments, input.shapes) {
            if input.part != Part::Pseudo {
                let row: Vec<C64> = moments.row(i).iter().copied().collect();
                add_compensation(pair_buf.as_mut_slice(), &row, shapes);
            }
        }
        for at in 0..n {
            out[i * n + at] = pair_buf.as_slice()[at] * input.kernel[at];
        }
    }
    Ok(())
}

/// `K(k)` of one target point against the occupied states of every point of the mesh (one spin
/// channel): `K_ij = sum_{q, n} sum_Q v(Q) rho_{i n}(Q) rho_{j n}(Q)^*`.
#[allow(clippy::too_many_arguments)]
fn target_matrix(
    c: &Crystal,
    occupied: &[OccupiedStates],
    target_point: &KPoint,
    omega: f64,
    exxdiv: ExxDivTreatment,
    fft: &mut Fft3,
    stage: Stage,
) -> Result<Matrix> {
    let m = c.m;
    let n = c.grid.index.points();
    let spinor = c.spinor;
    let n_basis = if spinor { 2 * m } else { m };
    let part = stage.map_or(Part::Pseudo, |(_, part)| part);

    // Periodic basis functions u_{mu, k}(r) = psi_{mu, k}(r) * exp(-i k . r)
    let mut u_k = c.context.allocate::<C64>(m * n)?;
    if part != Part::Compensation {
        let k_samples = target_point.psi.read(&c.context)?;
        periodic_part(c, target_point.k, &k_samples, u_k.as_mut_slice());
    }

    let mut k_mat = Matrix::zeros(n_basis, n_basis);
    let mut pair_buf = c.context.allocate::<C64>(n)?;

    // Loop over source q-points
    for (q, occ_q) in occupied.iter().enumerate() {
        if occ_q.wavefunctions.is_empty() {
            continue;
        }
        let source_point = &c.points[q];
        let k_diff = [
            source_point.k[0] - target_point.k[0],
            source_point.k[1] - target_point.k[1],
            source_point.k[2] - target_point.k[2],
        ];
        let sqrt_kernel_weight = kernel_weights(c, k_diff, occ_q.weight, omega, exxdiv)?;
        let shapes = match stage {
            Some((augmentation, _)) => Some(augmentation.pair_shapes(c, k_diff)?),
            None => None,
        };

        // Loop over occupied states j at q
        for (state, psi_j) in occ_q.wavefunctions.iter().enumerate() {
            let mut transformed_pairs = c.context.allocate::<C64>(n_basis * n)?;
            let moments = stage.map(|(augmentation, _)| {
                augmentation.moments(c, target_point, &occ_q.projected[state])
            });

            let input = PairInputs {
                u_k: u_k.as_slice(),
                psi: psi_j,
                moments: moments.as_ref(),
                shapes: shapes.as_ref().map(|s| s.as_slice()),
                kernel: sqrt_kernel_weight.as_slice(),
                part,
            };
            weighted_pairs(c, fft, &mut pair_buf, &input, transformed_pairs.as_mut_slice())?;

            // Accumulate into K: K_{i, j} += sum_{at} W_i[at] * W_j[at]^*
            for i in 0..n_basis {
                let w_i = &transformed_pairs.as_slice()[i * n..(i + 1) * n];
                for j in 0..=i {
                    let w_j = &transformed_pairs.as_slice()[j * n..(j + 1) * n];
                    let dot: C64 = (0..n).map(|at| w_i[at] * w_j[at].conj()).sum();
                    k_mat[(i, j)] += dot;
                    if j != i {
                        k_mat[(j, i)] += dot.conj();
                    }
                }
            }
        }
    }

    // Symmetrize K to ensure strict hermiticity
    Ok((&k_mat + k_mat.adjoint()) * C64::new(0.5, 0.0))
}

/// The mesh loop: `K_s(k)` for every spin channel and point, and the exchange energy.
fn build<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    stage: Stage,
) -> Result<Exchange> {
    let nk = c.points.len();
    let spinor = c.spinor;
    let n_spin_channels = if spinor { 1 } else { c.spins };
    let spin_scale = if !spinor && c.spins == 1 { 0.5 } else { 1.0 };
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;

    let mut operators = Vec::with_capacity(n_spin_channels * nk);
    let mut total_exchange_energy = 0.0;

    for s in 0..n_spin_channels {
        // 1. Extract occupied wavefunctions for all q-points
        let all_occupied = occupied_states(c, matrices, s, spin_scale, stage)?;

        // 2. Loop over target k-points
        for k in 0..nk {
            let target_point = &c.points[k];
            let k_herm = target_matrix(
                c,
                &all_occupied,
                target_point,
                screening.omega,
                exxdiv,
                &mut fft,
                stage,
            )?;

            // Exact exchange energy contribution
            let d_mat = matrices[s * nk + k].matrix(&c.context)?;
            let d_spin = &*d_mat * C64::new(spin_scale, 0.);
            let tr = (d_spin * &k_herm).trace().re;
            let mult = if !spinor && c.spins == 1 { 2.0 } else { 1.0 };
            total_exchange_energy -= 0.5 * screening.fraction * target_point.weight * tr * mult;

            operators.push(k_herm);
        }
    }

    Ok(Exchange {
        operators,
        energy: total_exchange_energy,
    })
}

/// The one-center model's terms for the density `matrices`: the projected density
/// `P_s = sum_k w_k C(k)^+ D_s(k) C(k)` (the four spin blocks for a spinor) handed to the model.
pub(super) fn one_center<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    model: &dyn OneCenter,
) -> Result<OneCenterTerms> {
    let (nk, p, m) = (c.points.len(), c.p, c.m);
    if c.spinor {
        let mut projected = Matrix::zeros(2 * p, 2 * p);
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[k].matrix(&c.context)?;
            for s in 0..2 {
                for t in 0..2 {
                    let block = point.projections.adjoint()
                        * d.view((s * m, t * m), (m, m))
                        * &point.projections
                        * C64::new(point.weight, 0.);
                    let mut view = projected.view_mut((s * p, t * p), (p, p));
                    view += block;
                }
            }
        }
        return model.spinor(&c.context, &projected);
    }
    let mut projected = vec![Matrix::zeros(p, p); c.spins];
    for (s, total) in projected.iter_mut().enumerate() {
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[s * nk + k].matrix(&c.context)?;
            *total +=
                point.projections.adjoint() * &*d * &point.projections * C64::new(point.weight, 0.);
        }
    }
    model.collinear(&c.context, &projected)
}

/// `C(k) O C(k)^+`, with `O` of a spin channel (`p x p`) or of a spinor (`2p x 2p`, block by block).
pub(super) fn lift(c: &Crystal, point: &KPoint, operator: &Matrix) -> Matrix {
    if !c.spinor {
        return &point.projections * operator * point.projections.adjoint();
    }
    let (m, p) = (c.m, c.p);
    let mut out = Matrix::zeros(2 * m, 2 * m);
    for s in 0..2 {
        for t in 0..2 {
            let block = &point.projections
                * operator.view((s * p, t * p), (p, p))
                * point.projections.adjoint();
            out.view_mut((s * m, t * m), (m, m)).copy_from(&block);
        }
    }
    out
}

/// The screened exact exchange matrices `K_s(k)` of every point of the mesh and the exchange
/// energy, with the compensation charges of `augmentation` in the pair densities. `None` keeps
/// the smooth pair densities alone (the operator before the augmentation existed). An attached
/// [`OneCenter`] model adds `C(k) O C(k)^+` to each matrix and its energy to the total.
pub fn exchange_operators_with<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    augmentation: Option<&Augmentation>,
) -> Result<Exchange> {
    let mut out = build(
        c,
        matrices,
        screening,
        exxdiv,
        augmentation.map(|a| (a, Part::Both)),
    )?;
    if let Some(model) = augmentation.and_then(|a| a.one_center()) {
        let nk = c.points.len();
        let terms = one_center(c, matrices, model)?;
        if terms.operators.len() != if c.spinor { 1 } else { c.spins } {
            return Err(invalid("the one-center model returned the wrong number of operators"));
        }
        for (s, operator) in terms.operators.iter().enumerate() {
            for (k, point) in c.points.iter().enumerate() {
                out.operators[s * nk + k] += lift(c, point, operator);
            }
        }
        out.energy += terms.energy;
    }
    Ok(out)
}

/// The energy of the grid pair alone, the cross term and the compensation charges alone.
pub fn energy_parts<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    augmentation: &Augmentation,
) -> Result<EnergyParts> {
    let energy =
        |part| build(c, matrices, screening, exxdiv, Some((augmentation, part))).map(|e| e.energy);
    let (pseudo, compensation, both) = (
        energy(Part::Pseudo)?,
        energy(Part::Compensation)?,
        energy(Part::Both)?,
    );
    Ok(EnergyParts {
        pseudo,
        cross: both - pseudo - compensation,
        compensation,
    })
}

/// Evaluates the screened exact exchange matrix K(k_path) at an arbitrary off-mesh k-point
/// using converged SCF density matrices, with the compensation charges of `augmentation`.
pub fn exchange_operator_at_k_with<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    target_point: &KPoint,
    spin: usize,
    screening: Screening,
    exxdiv: ExxDivTreatment,
    augmentation: Option<&Augmentation>,
) -> Result<Matrix> {
    let spin_scale = if !c.spinor && c.spins == 1 { 0.5 } else { 1.0 };
    let stage = augmentation.map(|a| (a, Part::Both));
    let all_occupied = occupied_states(c, matrices, spin, spin_scale, stage)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    let mut k_mat = target_matrix(
        c,
        &all_occupied,
        target_point,
        screening.omega,
        exxdiv,
        &mut fft,
        stage,
    )?;
    if let Some(model) = augmentation.and_then(|a| a.one_center()) {
        let terms = one_center(c, matrices, model)?;
        let operator = terms
            .operators
            .get(spin)
            .ok_or_else(|| invalid("the spin channel has no one-center operator"))?;
        k_mat += lift(c, target_point, operator);
    }
    Ok(k_mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gygi_baldereschi_limits_and_consistency() {
        let omega = 0.11;
        // As qc -> 0, it must converge to pi / omega^2
        let at_origin = PI / (omega * omega);
        let val_small = gygi_baldereschi_screened_kernel(1e-6, omega);
        assert!((val_small - at_origin).abs() < 1e-8);

        // Bare Coulomb limit (omega -> 0): 12 pi / qc^2
        let qc = 0.35;
        let bare = gygi_baldereschi_screened_kernel(qc, 0.0);
        let expected_bare = 12.0 * PI / (qc * qc);
        assert!((bare - expected_bare).abs() < 1e-12);

        // Consistency at Taylor transition boundary u = 1e-3
        let u_trans = 1e-3;
        let qc_trans = 2.0 * omega * u_trans;
        let val_below = gygi_baldereschi_screened_kernel(qc_trans * 0.999, omega);
        let val_above = gygi_baldereschi_screened_kernel(qc_trans * 1.001, omega);
        assert!((val_below - val_above).abs() < 1e-6);

        // Parsing tests
        assert_eq!(
            ExxDivTreatment::parse("none").unwrap(),
            ExxDivTreatment::None
        );
        assert_eq!(
            ExxDivTreatment::parse("gygi-baldereschi").unwrap(),
            ExxDivTreatment::GygiBaldereschi
        );
        // No aliases: every other spelling raises.
        for spelling in [
            "gb",
            "gygi_baldereschi",
            "off",
            "on",
            "true",
            "false",
            "",
            " none",
            "None",
            "unknown_mode",
        ] {
            assert!(ExxDivTreatment::parse(spelling).is_err(), "{spelling:?}");
        }
    }
}
