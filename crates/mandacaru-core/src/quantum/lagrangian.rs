//! The parameter-space half of the Lagrangian (Z-vector) force correction.
//!
//! A variational state `psi(theta)` that stopped at `theta~ = theta* + delta` has a parameter
//! gradient `g = dE/dtheta` that is not zero. With `H` the parameter Hessian at fixed geometry,
//! `delta = H^-1 g` to first order, and the force of the optimized surface is the geometry
//! derivative of `E(theta~, R) - z . g(theta~, R)` with `H z = g` held fixed. This module finds `g`
//! and `z` and prepares the two states `phi± = psi(theta~ ± s z)` whose energies give
//! `z . g(R) = (E[phi+](R) - E[phi-](R)) / (2 s)` at any geometry `R`; the molecular response
//! route contracts their reduced density matrices against every displaced Hamiltonian.
use super::{solver::CachedProblem, State, C64};
use crate::{math::krylov, math::storage::Precision, Error, Result};

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
    /// Length `|h v|` of the central step of each Hessian-vector product, in the parameters.
    pub hessian_step: f64,
    /// Length `|s z|` of the shift that makes `phi±`, in the parameters.
    pub shift: f64,
    pub solver: krylov::Options,
}
impl Default for Options {
    fn default() -> Self {
        Self { floor: 1e-10, hessian_step: 1e-4, shift: 1e-3, solver: krylov::Options::default() }
    }
}

/// The solved Z vector and the two shifted states.
pub struct ZVector {
    pub z: Vec<f64>,
    pub z_norm: f64,
    /// `z . g`, the first-order energy lowering a Newton step would bring.
    pub z_dot_g: f64,
    /// Conjugate-gradient iterations, and the gradient evaluations they cost.
    pub iterations: usize,
    pub gradient_evaluations: usize,
    pub residual: f64,
    /// A zero-curvature (redundant) direction was left out: the solution is the minimum-norm one.
    pub truncated: bool,
    pub lowest_curvature: f64,
    /// `phi± = psi(theta ± scale z)`.
    pub scale: f64,
    pub plus: State,
    pub minus: State,
}

pub enum Residual {
    /// `|g|` is under the floor: the state is stationary and the correction is skipped.
    Stationary { gradient_norm: f64 },
    Corrected { gradient_norm: f64, z: ZVector },
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

impl CachedProblem {
    /// The parameter gradient of the state `state`, which must be this ansatz's state, and the
    /// solution of `H z = g` with the states `psi(theta ± s z)`.
    pub fn lagrangian(&self, ansatz: &Ansatz<'_>, state: &[C64], options: &Options) -> Result<Residual> {
        if self.precision != Precision::Float64 {
            return Err(invalid("the Lagrangian force correction differences parameter gradients and needs float64 states"));
        }
        let Ansatz { indices, parameters, product_form } = *ansatz;
        if parameters.iter().any(|v| !v.is_finite())
            || !(options.floor >= 0.)
            || !(options.hessian_step > 0.)
            || !(options.shift > 0.)
        {
            return Err(invalid("the Lagrangian correction needs finite parameters, a nonnegative floor and positive steps"));
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
            let mut evaluations = 0;
            let mut hessian_product = |v: &[f64]| -> Result<Vec<f64>> {
                let step = options.hessian_step / norm(v);
                let shifted = |sign: f64| -> Result<Vec<f64>> {
                    let theta: Vec<f64> =
                        parameters.iter().zip(v).map(|(t, d)| t + sign * step * d).collect();
                    Ok(self.evaluate(indices, &theta, product_form)?.gradient)
                };
                let (up, down) = (shifted(1.)?, shifted(-1.)?);
                evaluations += 2;
                Ok(up.iter().zip(&down).map(|(u, d)| (u - d) / (2. * step)).collect())
            };
            let solution = krylov::minimum_norm(&gradient, &mut hessian_product, &options.solver)
                .map_err(|e| {
                    invalid(format!(
                        "the parameter Hessian cannot be inverted for the Lagrangian force correction \
                         (|g| = {gradient_norm:.3e}): {e}"
                    ))
                })?;
            let z_norm = norm(&solution.x);
            if !(z_norm > 0.) || !z_norm.is_finite() {
                return Err(invalid("the Lagrangian Z vector vanished or is nonfinite although the parameter gradient is not"));
            }
            let z_dot_g = solution.x.iter().zip(&gradient).map(|(z, g)| z * g).sum();
            let scale = options.shift / z_norm;
            let shifted = |sign: f64| -> Result<State> {
                let theta: Vec<f64> =
                    parameters.iter().zip(&solution.x).map(|(t, z)| t + sign * scale * z).collect();
                self.prepare_state(indices, &theta, product_form)
            };
            let (plus, minus) = (shifted(1.)?, shifted(-1.)?);
            Ok(Residual::Corrected {
                gradient_norm,
                z: ZVector {
                    z: solution.x,
                    z_norm,
                    z_dot_g,
                    iterations: solution.iterations,
                    gradient_evaluations: evaluations,
                    residual: solution.residual,
                    truncated: solution.truncated,
                    lowest_curvature: solution.lowest_curvature,
                    scale,
                    plus,
                    minus,
                },
            })
        })
    }
}
