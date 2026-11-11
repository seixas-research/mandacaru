//! The parameter-space half of the Lagrangian (Z-vector) force correction.
//!
//! A variational state `psi(theta)` that stopped at `theta~ = theta* + delta` has a parameter
//! gradient `g = dE/dtheta` that is not zero. With `H` the parameter Hessian at fixed geometry,
//! `delta = H^-1 g` to first order, and the force of the optimized surface is the geometry
//! derivative of `E(theta~, R) - z . g(theta~, R)` with `H z = g` held fixed. This module finds `g`
//! and `z` and the tangent `psi' = sum_k z_k d psi / d theta_k`, so that
//! `z . g(R) = 2 Re <psi'| H(R) |psi>` at any geometry `R`. That is a quadratic form: with
//! `phi± = psi ± s psi'` (not normalized), `z . g(R) = (E[phi+](R) - E[phi-](R)) / (2 s)` exactly,
//! and the molecular response route contracts the reduced density matrices of `phi±` against the
//! Hamiltonian of every geometry.
//!
//! The Hessian of a product-form state (every ADAPT-VQE state and every Trotterized one) is exact:
//! one forward sweep stores every partial state, one reverse sweep every adjoint, and each
//! derivative state `d psi / d theta_j` is propagated once to the end, collecting the second
//! derivatives on the way; one Hamiltonian action per parameter gives the rest. The summed UCC
//! exponential differences its analytic gradients (its Frechet recurrence has no second order).
//! The truncated solve of [`symmetric_solve`] inverts either.
//!
//! References (`utils/bibliography.py`): the z-vector, Handy and Schaefer, J. Chem. Phys. 81,
//! 5031 (1984); the Lagrangian formulation, Helgaker and Jorgensen, Theor. Chim. Acta 75, 111
//! (1989); variational quantum states, Mitarai, Nakagawa and Mizukami, Phys. Rev. Research 2,
//! 013129 (2020).
use super::amplitude::{inner_wide, stack, Amplitude};
use super::evolution::{exponential, inner};
use super::{solver::CachedProblem, Basis, Operator, State, C64};
use crate::{math::storage::Precision, math::symmetric_solve, Error, Result};
use nalgebra::DMatrix;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// The optimized ansatz a state came from: the operator sequence, its parameters, and whether
/// the state was prepared in product form (every ADAPT-VQE state and every Trotterized one).
#[derive(Clone, Copy, Debug)]
pub struct Ansatz<'a> {
    pub indices: &'a [usize],
    pub parameters: &'a [f64],
    pub product_form: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// A parameter gradient with `|g|` below this is a stationary state: no solve, no correction.
    pub floor: f64,
    /// Length of the central step along each parameter that differences the analytic gradients
    /// into the Hessian of the summed exponential (a product-form Hessian is exact).
    pub hessian_step: f64,
    /// The truncation of the flat directions of the Hessian (see [`symmetric_solve`]).
    pub solver: symmetric_solve::Options,
    /// More parameters than this refuse: the exact Hessian holds two states per generator.
    pub max_parameters: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            floor: 1e-10,
            hessian_step: 1e-4,
            solver: symmetric_solve::Options::default(),
            max_parameters: 1500,
        }
    }
}

/// How the Hessian was built and inverted: its spectrum, the noise that limits it and what was
/// dropped.
#[derive(Clone, Debug)]
pub struct Solve {
    pub parameters: usize,
    /// Whether the Hessian is exact (product form) or differenced analytic gradients (the summed
    /// exponential).
    pub exact: bool,
    /// Gradient evaluations spent on the Hessian (`2 n` for the summed exponential, none when exact).
    pub gradient_evaluations: usize,
    /// Hamiltonian actions spent on the Hessian (one per generator when exact).
    pub hamiltonian_actions: usize,
    /// `|H - H^T|_F`: the measured error of the Hessian.
    pub noise: f64,
    pub cutoff: f64,
    pub lowest: f64,
    pub highest: f64,
    pub smallest_kept: Option<f64>,
    /// Eigen-directions with `|eigenvalue| <= cutoff` carry no correction (the minimum-norm solve).
    pub dropped: usize,
    /// The norm of `g` along the dropped directions, and `|g_dropped| / cutoff`, a floor on the
    /// `|z|` they would add.
    pub dropped_gradient: f64,
    pub neglected_scale: f64,
}

/// The solved Z vector and the two states whose energy difference is `z . g`.
pub struct ZVector {
    pub z: Vec<f64>,
    pub z_norm: f64,
    /// `z . g`, the first-order energy lowering a Newton step would bring.
    pub z_dot_g: f64,
    pub solve: Solve,
    /// `phi± = psi ± scale psi'`, `psi'` the exact tangent along `z` (not normalized).
    pub scale: f64,
    pub plus: State,
    pub minus: State,
}

pub enum Residual {
    /// `|g|` is under the floor: the state is stationary and the correction is skipped.
    Stationary {
        gradient_norm: f64,
    },
    Corrected {
        gradient_norm: f64,
        z: ZVector,
    },
}
impl Residual {
    pub fn gradient_norm(&self) -> f64 {
        match self {
            Self::Stationary { gradient_norm } | Self::Corrected { gradient_norm, .. } => {
                *gradient_norm
            }
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The generators and angles a product-form state applies, in order, and how each angle
/// derives from the parameters (`links[j] = (parameter, weight)`; one to one without a
/// Hamiltonian-variational expansion).
struct Product<'a> {
    ops: Vec<&'a Operator>,
    angles: Vec<f64>,
    links: Vec<(usize, f64)>,
}

/// `out = G input` for one generator.
fn act(generator: &Operator, basis: &Basis, input: &[C64], out: &mut [C64]) {
    generator.apply_into(basis, input, out);
}

impl CachedProblem {
    fn product<'a>(&'a self, indices: &[usize], parameters: &[f64]) -> Result<Product<'a>> {
        if let Some(hva) = &self.hva {
            let expanded = hva.expand(&self.pool, indices, parameters)?;
            return Ok(Product {
                ops: expanded.operators,
                angles: expanded.parameters,
                links: expanded.links,
            });
        }
        let ops = indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Product {
            ops,
            angles: parameters.to_vec(),
            links: (0..parameters.len()).map(|i| (i, 1.)).collect(),
        })
    }

    /// `out = H input` in float64.
    fn hamiltonian_into(&self, basis: &Basis, input: &[C64], out: &mut [C64]) {
        match &self.sparse {
            Some(sparse) => sparse.apply_into(input, out),
            None => self.hamiltonian.apply_into(basis, input, out),
        }
    }

    /// The state `psi` and its tangent `psi' = sum_k v_k d psi / d theta_k` of a product-form
    /// ansatz, by one forward sweep: `psi'_k = U_k psi'_{k-1} + v_k G_k psi_k` (two rows).
    fn forward_tangent(&self, basis: &Basis, product: &Product<'_>, v: &[f64]) -> Result<crate::Buffer<C64>> {
        let ctx = &self.context;
        let d = basis.len();
        let mut rows = stack::<C64>(ctx, 2, d)?;
        rows.as_mut_slice()[basis.reference] = C64::new(1., 0.);
        let mut acc = ctx.allocate::<C64>(C64::accumulator_len(d))?;
        let mut action = ctx.allocate::<C64>(d)?;
        for (j, (g, &angle)) in product.ops.iter().zip(&product.angles).enumerate() {
            exponential(ctx, basis, &[*g], &[angle], rows.as_mut_slice(), 2, &[None], acc.as_mut_slice())?;
            let (i, w) = product.links[j];
            let direction = w * v[i];
            if direction != 0. {
                let (psi, tangent) = rows.as_mut_slice().split_at_mut(d);
                act(g, basis, psi, action.as_mut_slice());
                for (t, a) in tangent.iter_mut().zip(action.as_slice()) {
                    *t += a * direction;
                }
            }
        }
        Ok(rows)
    }

    /// The exact Hessian of a product-form ansatz, `psi = U_{N-1} ... U_0 |ref>`, `U_j = exp(a_j G_j)`,
    /// in its angles `a`: `H_jk = 2 Re [<d_j psi| (H - E) |d_k psi> + <lambda| d_j d_k psi>]`,
    /// `lambda = (H - E) psi` (the form is exact because `d_j d_k <psi|psi> = 0`). A forward sweep
    /// stores the partial states `psi_j`, a reverse sweep `mu_k = G_k^+ lambda_k`; each
    /// `v = G_j psi_j` is then carried through `U_{j+1} .. U_{N-1}`, `<mu_k| v>` after `U_k` being
    /// `<lambda| d_j d_k psi>` (`k >= j`), until it is `d_j psi`; one Hamiltonian action each gives
    /// the first part, computed in full so that its antisymmetry is the measured roundoff.
    fn exact_angle_hessian(&self, basis: &Basis, product: &Product<'_>) -> Result<DMatrix<f64>> {
        let ctx = &self.context;
        let d = basis.len();
        let n = product.ops.len();
        let mut states = stack::<C64>(ctx, n.max(1), d)?;
        let mut adjoints = stack::<C64>(ctx, n.max(1), d)?;
        let mut acc = ctx.allocate::<C64>(C64::accumulator_len(d))?;
        let mut row = ctx.allocate::<C64>(d)?;
        row.as_mut_slice()[basis.reference] = C64::new(1., 0.);
        for (j, (g, &angle)) in product.ops.iter().zip(&product.angles).enumerate() {
            exponential(ctx, basis, &[*g], &[angle], row.as_mut_slice(), 1, &[], acc.as_mut_slice())?;
            states.as_mut_slice()[j * d..(j + 1) * d].copy_from_slice(row.as_slice());
        }
        let mut lambda = ctx.allocate::<C64>(d)?;
        self.hamiltonian_into(basis, row.as_slice(), lambda.as_mut_slice());
        let energy = inner(row.as_slice(), lambda.as_slice()).re;
        for (l, p) in lambda.as_mut_slice().iter_mut().zip(row.as_slice()) {
            *l -= energy * p;
        }
        for k in (0..n).rev() {
            let target = &mut adjoints.as_mut_slice()[k * d..(k + 1) * d];
            act(product.ops[k], basis, lambda.as_slice(), target);
            for v in target.iter_mut() {
                *v = -*v;
            }
            if k > 0 {
                exponential(ctx, basis, &[product.ops[k]], &[-product.angles[k]], lambda.as_mut_slice(), 1, &[], acc.as_mut_slice())?;
            }
        }
        drop(lambda);
        let mut second = DMatrix::<f64>::zeros(n, n);
        for j in 0..n {
            act(product.ops[j], basis, &states.as_slice()[j * d..(j + 1) * d], row.as_mut_slice());
            second[(j, j)] = inner(&adjoints.as_slice()[j * d..(j + 1) * d], row.as_slice()).re;
            for k in j + 1..n {
                exponential(ctx, basis, &[product.ops[k]], &[product.angles[k]], row.as_mut_slice(), 1, &[], acc.as_mut_slice())?;
                second[(j, k)] = inner(&adjoints.as_slice()[k * d..(k + 1) * d], row.as_slice()).re;
                second[(k, j)] = second[(j, k)];
            }
            // `row` is now d_j psi; psi_j is no longer needed.
            states.as_mut_slice()[j * d..(j + 1) * d].copy_from_slice(row.as_slice());
        }
        drop(adjoints);
        // <d_j psi|(H - E)|d_k psi>: the n independent actions on the derivative states, one per
        // worker (`Operator::apply_each`, `parallel::parallel_each`; each the single action's bits).
        let all = states.as_slice();
        let derivatives: Vec<&[C64]> = (0..n).map(|k| &all[k * d..(k + 1) * d]).collect();
        let finish = |k: usize, row: &mut [C64]| -> Vec<f64> {
            for (r, p) in row.iter_mut().zip(derivatives[k]) {
                *r -= energy * p;
            }
            (0..n).map(|j| inner(derivatives[j], row).re).collect()
        };
        let columns = match &self.sparse {
            Some(sparse) => {
                super::parallel::parallel_each(&self.context, n, d, n.saturating_mul(d), |k, row| {
                    sparse.apply_into(derivatives[k], row);
                    finish(k, row)
                })?
            }
            None => self.hamiltonian.apply_each(basis, &derivatives, false, finish)?,
        };
        let first = DMatrix::from_fn(n, n, |j, k| columns[k][j]);
        drop(row);
        let hessian = (first + second) * 2.;
        if hessian.iter().any(|v| !v.is_finite()) {
            return Err(Error::Kernel("nonfinite parameter Hessian".into()));
        }
        Ok(hessian)
    }

    /// The parameter Hessian of the ansatz, the gradient evaluations and the Hamiltonian actions it
    /// cost: exact in product form (the angle Hessian pulled back
    /// through the links of a Hamiltonian-variational expansion), columns of central differences
    /// of the analytic gradients for the summed exponential.
    pub fn parameter_hessian(&self, ansatz: &Ansatz<'_>, options: &Options) -> Result<(DMatrix<f64>, usize, usize)> {
        let basis = self.register("the parameter Hessian")?;
        let n = ansatz.parameters.len();
        if ansatz.product_form {
            let product = self.product(ansatz.indices, ansatz.parameters)?;
            let angles = self.exact_angle_hessian(basis, &product)?;
            let mut hessian = DMatrix::<f64>::zeros(n, n);
            for (a, &(i, wi)) in product.links.iter().enumerate() {
                for (b, &(k, wk)) in product.links.iter().enumerate() {
                    hessian[(i, k)] += wi * wk * angles[(a, b)];
                }
            }
            return Ok((hessian, 0, product.ops.len() + 1));
        }
        let mut hessian = DMatrix::<f64>::zeros(n, n);
        for column in 0..n {
            let shifted = |sign: f64| -> Result<Vec<f64>> {
                let mut theta = ansatz.parameters.to_vec();
                theta[column] += sign * options.hessian_step;
                Ok(self.evaluate(ansatz.indices, &theta, false)?.gradient)
            };
            let (up, down) = (shifted(1.)?, shifted(-1.)?);
            for row in 0..n {
                hessian[(row, column)] = (up[row] - down[row]) / (2. * options.hessian_step);
            }
        }
        Ok((hessian, 2 * n, 2 * n))
    }

    /// The exact tangent `psi' = sum_k z_k d psi / d theta_k` of the ansatz state.
    fn tangent(&self, ansatz: &Ansatz<'_>, z: &[f64]) -> Result<crate::Buffer<C64>> {
        let ctx = &self.context;
        let basis = self.register("the Lagrangian tangent state")?;
        let d = basis.len();
        let mut out = ctx.allocate::<C64>(d)?;
        if ansatz.product_form {
            let product = self.product(ansatz.indices, ansatz.parameters)?;
            let rows = self.forward_tangent(basis, &product, z)?;
            out.as_mut_slice().copy_from_slice(&rows.as_slice()[d..]);
            return Ok(out);
        }
        let product = self.product(ansatz.indices, ansatz.parameters)?;
        let rows = super::evolution::state::<C64>(ctx, basis, &product.ops, &product.angles, false, true)?;
        for (j, &(i, w)) in product.links.iter().enumerate() {
            let coefficient = w * z[i];
            for (o, r) in out.as_mut_slice().iter_mut().zip(&rows.as_slice()[(j + 1) * d..(j + 2) * d]) {
                *o += r * coefficient;
            }
        }
        Ok(out)
    }

    /// The parameter gradient of the state `state`, which must be this ansatz's state, and the
    /// solution of `H z = g` with the states `psi ± s psi'`.
    pub fn lagrangian(
        &self,
        ansatz: &Ansatz<'_>,
        state: &[C64],
        options: &Options,
    ) -> Result<Residual> {
        if self.precision != Precision::Float64 {
            return Err(invalid("the Lagrangian force correction needs float64 states"));
        }
        let Ansatz {
            indices,
            parameters,
            product_form,
        } = *ansatz;
        if parameters.iter().any(|v| !v.is_finite())
            || !(options.floor >= 0.)
            || !(options.hessian_step > 0.)
        {
            return Err(invalid("the Lagrangian correction needs finite parameters, a nonnegative floor and a positive step"));
        }
        self.context.run(|| {
            let at = self.evaluate(indices, parameters, product_form)?;
            let prepared = at.state.complex128()?;
            if prepared.len() != state.len() {
                return Err(invalid("the state and the ansatz live in different registers"));
            }
            let overlap: C64 = prepared.iter().zip(state).map(|(a, b)| a.conj() * b).sum();
            if (1. - overlap.norm()).abs() > 1e-8 {
                return Err(invalid(format!(
                    "the state is not the one the operator sequence and parameters prepare (overlap {:.3e}): \
                     the Lagrangian correction needs the optimized ansatz it differentiates",
                    overlap.norm()
                )));
            }
            let gradient = at.gradient;
            let gradient_norm = norm(&gradient);
            if gradient.len() != parameters.len() || !gradient_norm.is_finite() {
                return Err(invalid("the parameter gradient is nonfinite or mis-sized"));
            }
            if gradient_norm <= options.floor {
                return Ok(Residual::Stationary { gradient_norm });
            }
            let n = parameters.len();
            if n > options.max_parameters {
                return Err(invalid(format!(
                    "the Lagrangian force correction refuses {n} parameters (limit {})",
                    options.max_parameters
                )));
            }
            let (hessian, gradient_evaluations, hamiltonian_actions) =
                self.parameter_hessian(ansatz, options)?;
            let solution = symmetric_solve::solve(&hessian, &gradient, &options.solver).map_err(|e| {
                invalid(format!(
                    "the parameter Hessian cannot be inverted for the Lagrangian force correction \
                     (|g| = {gradient_norm:.3e}): {e}"
                ))
            })?;
            let z_norm = norm(&solution.x);
            if !(z_norm > 0.) || !z_norm.is_finite() {
                return Err(invalid(format!(
                    "every direction of the parameter Hessian lies below its noise cutoff {:.3e} (|g| = {gradient_norm:.3e}): \
                     there is no curvature to build the Lagrangian force correction from",
                    solution.cutoff
                )));
            }
            let z_dot_g = solution.x.iter().zip(&gradient).map(|(z, g)| z * g).sum();
            let solve = Solve {
                parameters: n,
                exact: product_form,
                gradient_evaluations,
                hamiltonian_actions,
                noise: solution.noise,
                cutoff: solution.cutoff,
                lowest: solution.eigenvalues[0],
                highest: solution.eigenvalues[n - 1],
                smallest_kept: solution.smallest_kept,
                dropped: solution.dropped,
                dropped_gradient: solution.dropped_rhs,
                neglected_scale: solution.neglected_scale(),
            };
            let tangent = self.tangent(ansatz, &solution.x)?;
            let tangent_norm = inner_wide(tangent.as_slice(), tangent.as_slice()).re.sqrt();
            if !(tangent_norm > 0.) || !tangent_norm.is_finite() {
                return Err(invalid("the tangent of the ansatz along z vanishes: z lies in the redundant parameter directions"));
            }
            let scale = 1. / tangent_norm;
            let shifted = |sign: f64| -> Result<State> {
                let mut buffer = self.context.allocate::<C64>(state.len())?;
                for ((o, p), t) in buffer.as_mut_slice().iter_mut().zip(prepared).zip(tangent.as_slice()) {
                    *o = p + t * (sign * scale);
                }
                Ok(C64::wrap(buffer))
            };
            let (plus, minus) = (shifted(1.)?, shifted(-1.)?);
            Ok(Residual::Corrected {
                gradient_norm,
                z: ZVector {
                    z: solution.x.clone(),
                    z_norm,
                    z_dot_g,
                    solve,
                    scale,
                    plus,
                    minus,
                },
            })
        })
    }
}

/// How one energy measured on a device enters the measured gradient and Hessian, with its
/// standard error: `g[i] += c E` for every `(i, c)` of `gradient` and `H[p][q] += c E` for every
/// `(p, q, c)` of `hessian` (both triangles listed). The energies of one batch are independent
/// (distinct circuits, independent shots), so these give the covariance of `z` exactly to first
/// order: `Cov(z) = M diag(sigma_e^2) M^T`, `M = dz/dE`.
#[derive(Clone, Debug, Default)]
pub struct MeasuredEnergy {
    pub error: f64,
    pub gradient: Vec<(usize, f64)>,
    pub hessian: Vec<(usize, usize, f64)>,
}

/// A measured Hessian eigenvalue must lie this many of its own standard errors from zero.
///
/// `z = sum_i v_i (v_i . g) / lambda_i`, and the error of `z` is propagated to first order. At
/// `|lambda| = k sigma` the first-order relative error of `1 / lambda` is `1/k` and the term the
/// propagation drops, the bias of `1 / lambda` itself, is `(sigma / lambda)^2 = 1/k^2`: with
/// `k = 5` the reported error is right to about 4 percent, well inside the precision with which
/// any sampled error bar is known, and a significant eigenvalue fluctuates below the threshold
/// (or through zero) with a probability under 3e-7. Below it the correction is set by shot noise
/// and the force refuses rather than report an error bar that does not describe it.
pub const CURVATURE_SIGNIFICANCE: f64 = 5.;

/// The Z vector of a gradient and Hessian measured on a device, by the same truncated solve
/// ([`symmetric_solve`]): `solution` is `None` when `|g|` is under the floor. A measured Hessian
/// is symmetric by construction (each pair's four energies give one element), so its measured
/// noise is zero and the cutoff is the relative one. On a sampled device every eigenvalue's
/// standard error `sigma(lambda_i)^2 = sum_e (v_i^T dH/dE_e v_i)^2 sigma_e^2` is checked first: an
/// eigenvalue within [`CURVATURE_SIGNIFICANCE`] of its error raises by name with both numbers, of
/// either sign; a negative curvature beyond it (the optimization did not reach a minimum) raises
/// in the solve. `z_covariance` is `Cov(z)` from the energies' errors (`None` on an exact device).
pub struct MeasuredZ {
    pub gradient_norm: f64,
    pub solution: Option<symmetric_solve::Solution>,
    /// `sigma(lambda_i)`, in the solution's eigenvalue order (empty without a solution).
    pub curvature_errors: Vec<f64>,
    pub z_covariance: Option<DMatrix<f64>>,
    /// `|z|` (zero without a solution).
    pub z_norm: f64,
    /// On a sampled device with a solution: the error of the lowest eigenvalue, and the smallest
    /// `|lambda_i| / sigma(lambda_i)` over the eigenvalues with an error (`None` otherwise).
    pub lowest_curvature_error: Option<f64>,
    pub curvature_significance: Option<f64>,
}

/// `out += dH_e x`.
fn add_hessian_action(energy: &MeasuredEnergy, x: &[f64], out: &mut [f64]) {
    for &(p, q, c) in &energy.hessian {
        out[p] += c * x[q];
    }
}

pub fn measured_z(
    gradient: &[f64],
    hessian: &DMatrix<f64>,
    energies: &[MeasuredEnergy],
    options: &Options,
) -> Result<MeasuredZ> {
    let n = gradient.len();
    let gradient_norm = norm(gradient);
    if !gradient_norm.is_finite() || hessian.shape() != (n, n) {
        return Err(invalid("the measured gradient and Hessian must be finite and of one size"));
    }
    if energies.iter().any(|e| {
        !(e.error >= 0.)
            || e.gradient.iter().any(|&(i, c)| i >= n || !c.is_finite())
            || e.hessian.iter().any(|&(p, q, c)| p >= n || q >= n || !c.is_finite())
    }) {
        return Err(invalid("a measured energy's error and dependences must be finite and within the parameters"));
    }
    if gradient_norm <= options.floor {
        return Ok(MeasuredZ {
            gradient_norm,
            solution: None,
            curvature_errors: vec![],
            z_covariance: None,
            z_norm: 0.,
            lowest_curvature_error: None,
            curvature_significance: None,
        });
    }
    let sampled = energies.iter().any(|e| e.error > 0.);
    let mut curvature_errors = vec![0.; n];
    if sampled {
        // Checked before the solve, so a noise-swamped eigenvalue that came out negative is
        // reported as what it is, not as a negative curvature.
        let symmetric = (hessian + hessian.transpose()) * 0.5;
        let (values, vectors) = crate::math::linalg::real_eigen(symmetric)?;
        for (i, error) in curvature_errors.iter_mut().enumerate() {
            let v = vectors.column(i);
            *error = energies
                .iter()
                .filter(|e| e.error > 0.)
                .map(|e| {
                    let d: f64 = e.hessian.iter().map(|&(p, q, c)| c * v[p] * v[q]).sum();
                    (d * e.error).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            if *error > 0. && values[i].abs() < CURVATURE_SIGNIFICANCE * *error {
                return Err(invalid(format!(
                    "the measured parameter Hessian has the curvature {:.3e} Ha with a shot-noise standard \
                     error of {:.3e} Ha (eigenvalue {} of {n}), within {CURVATURE_SIGNIFICANCE} standard \
                     errors of zero: the device force's Z-vector correction along that direction would be \
                     set by shot noise (|g| = {gradient_norm:.3e}). Raise shots by at least a factor {:.1} \
                     (the error falls as 1/sqrt(shots)), or optimize closer to the minimum",
                    values[i],
                    *error,
                    i + 1,
                    (CURVATURE_SIGNIFICANCE * *error / values[i].abs().max(f64::MIN_POSITIVE)).powi(2).min(1e12),
                )));
            }
        }
    }
    let solution = symmetric_solve::solve(hessian, gradient, &options.solver).map_err(|e| {
        invalid(format!(
            "the measured parameter Hessian cannot be inverted for the device force's Z-vector correction \
             (|g| = {gradient_norm:.3e}): {e}"
        ))
    })?;
    if !(norm(&solution.x) > 0.) {
        return Err(invalid(format!(
            "every direction of the measured parameter Hessian lies below its cutoff {:.3e} (|g| = {gradient_norm:.3e})",
            solution.cutoff
        )));
    }
    let z_covariance = sampled.then(|| covariance(&solution, gradient, energies));
    let z_norm = norm(&solution.x);
    let lowest_curvature_error = sampled.then(|| curvature_errors[0]);
    let curvature_significance = sampled.then(|| {
        solution
            .eigenvalues
            .iter()
            .zip(&curvature_errors)
            .filter(|(_, e)| **e > 0.)
            .map(|(v, e)| v.abs() / e)
            .fold(f64::INFINITY, f64::min)
    });
    Ok(MeasuredZ {
        gradient_norm,
        solution: Some(solution),
        curvature_errors,
        z_covariance,
        z_norm,
        lowest_curvature_error,
        curvature_significance,
    })
}

/// `Cov(z) = sum_e sigma_e^2 dz_e dz_e^T` with `dz_e` the first-order change of the truncated
/// solve `z = H^+ g` per unit of energy `e` (the pseudo-inverse at constant rank, Golub and
/// Pereyra, SIAM J. Numer. Anal. 10, 413 (1973)): `dz = H^+ (dg - dH z) + H^+ H^+ dH P g +
/// P dH H^+ z`, `P` the projector on the dropped directions (both last terms vanish when none is
/// dropped).
fn covariance(solution: &symmetric_solve::Solution, gradient: &[f64], energies: &[MeasuredEnergy]) -> DMatrix<f64> {
    let n = gradient.len();
    let v = &solution.vectors;
    let mut pseudo = DMatrix::<f64>::zeros(n, n);
    let mut dropped = DMatrix::<f64>::zeros(n, n);
    for (i, &value) in solution.eigenvalues.iter().enumerate() {
        let column = v.column(i);
        if value.abs() <= solution.cutoff {
            dropped += &column * column.transpose();
        } else {
            pseudo += &column * column.transpose() / value;
        }
    }
    let z = &solution.x;
    let rest: Option<(Vec<f64>, Vec<f64>)> = (solution.dropped > 0).then(|| {
        let g = nalgebra::DVector::from_column_slice(gradient);
        let r = &dropped * &g;
        let y = &pseudo * nalgebra::DVector::from_column_slice(z);
        (r.iter().copied().collect(), y.iter().copied().collect())
    });
    let mut out = DMatrix::<f64>::zeros(n, n);
    let mut u = vec![0.; n];
    for energy in energies.iter().filter(|e| e.error > 0.) {
        u.iter_mut().for_each(|x| *x = 0.);
        for &(i, c) in &energy.gradient {
            u[i] += c;
        }
        let mut action = vec![0.; n];
        add_hessian_action(energy, z, &mut action);
        for (a, b) in u.iter_mut().zip(&action) {
            *a -= b;
        }
        let mut dz = &pseudo * nalgebra::DVector::from_column_slice(&u);
        if let Some((r, y)) = &rest {
            let mut on_r = vec![0.; n];
            add_hessian_action(energy, r, &mut on_r);
            let on_r = nalgebra::DVector::from_column_slice(&on_r);
            dz += &pseudo * (&pseudo * on_r);
            let mut on_y = vec![0.; n];
            add_hessian_action(energy, y, &mut on_y);
            dz += &dropped * nalgebra::DVector::from_column_slice(&on_y);
        }
        out.ger(energy.error * energy.error, &dz, &dz, 1.);
    }
    out
}
