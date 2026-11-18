//! Outer self-consistent linear-response loop for DFT+U parameters (`LR-12`).
//!
//! Off by default. Iterates single-shot linear-response measurement (`hubbard_response_with_options`)
//! with damped updates `U_{k+1} = (1 - damping) U_k + damping U_out` until element-averaged
//! Hubbard parameters stabilize within `tolerance` (Hartree), recording `U_source = "lr-scf"`.

use super::{
    hubbard::Spec,
    hubbard_response::{detect_column_sources, hubbard_response_with_spec_and_symmetry, ResponseResult},
    invalid, Crystal, Options as ScfOptions, PeriodicResult,
};
use crate::models::{
    hubbard::Form,
    hubbard_response::{Fit, Kernel, Options as ResponseOptions},
};
use crate::Result;

/// Controls for the outer self-consistent linear-response loop (`LR-12`).
#[derive(Clone, Debug)]
pub struct Options {
    pub max_iterations: usize,
    /// Convergence tolerance on `max |U_out - U_in|` in Hartree (default ~0.01 eV).
    pub tolerance: f64,
    /// Damping factor `lambda` in `(0, 1]` for `U_next = (1 - lambda) U_in + lambda U_out`.
    pub damping: f64,
    pub kernel: Kernel,
    pub form: Form,
    pub spread_tolerance: f64,
    pub alphas: Vec<f64>,
    pub scf_options: ScfOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 0.01 / 27.211_386_245_988,
            damping: 0.5,
            kernel: Kernel::SpinSummed,
            form: Form::Dudarev,
            spread_tolerance: 0.5 / 27.211_386_245_988,
            alphas: vec![0.05 / 27.211_386_245_988, 0.10 / 27.211_386_245_988],
            scf_options: ScfOptions::default(),
        }
    }
}

/// Output of the outer self-consistent linear-response loop.
pub struct SelfConsistentResponse {
    pub iterations: usize,
    pub converged: bool,
    /// Per-element `(symbol, U_Ha, J_Ha)` at termination.
    pub elements: Vec<(String, f64, f64)>,
    /// History of `(symbol, U_Ha, J_Ha)` across outer iterations.
    pub history: Vec<Vec<(String, f64, f64)>>,
    pub final_hubbard: Spec,
    pub response: ResponseResult,
    pub state: PeriodicResult,
}

/// Runs the outer damped self-consistent linear-response loop (`LR-12`), returning updated
/// Hubbard shell parameters with `U_source = "lr-scf"`.
pub fn converge_hubbard_u(
    crystal: &Crystal,
    initial_state: &PeriodicResult,
    options: &Options,
) -> Result<SelfConsistentResponse> {
    if options.max_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || !options.damping.is_finite()
        || options.damping <= 0.
        || options.damping > 1.
        || options.alphas.is_empty()
    {
        return Err(invalid(
            "invalid LR-12 self-consistent Hubbard response options",
        ));
    }
    let mut spec = crystal
        .hubbard
        .as_ref()
        .cloned()
        .ok_or_else(|| invalid("LR-12 requires a crystal with a hubbard specification"))?;
    let n_shells = spec.shells.len();
    let mut owned_state: Option<PeriodicResult> = None;
    let fit = if options.alphas.len() >= 2 {
        Fit::Richardson
    } else {
        Fit::Central
    };
    let resp_opts = ResponseOptions::new(&options.alphas, fit)?;
    let mut history = Vec::new();
    let mut converged = false;
    let mut last_response = None;
    let sources = detect_column_sources(crystal, initial_state);

    for iteration in 1..=options.max_iterations {
        let current = owned_state.as_ref().unwrap_or(initial_state);
        let mut res = hubbard_response_with_spec_and_symmetry(
            crystal,
            current,
            Some(&spec),
            &resp_opts,
            &sources,
        )?;
        res.self_consistent = true;
        let groups = res.hubbard_u(
            options.form,
            options.kernel,
            None,
            options.spread_tolerance,
        )?;
        let iter_elems: Vec<(String, f64, f64)> = groups
            .iter()
            .map(|(g, u, j)| (g.key.clone(), *u, *j))
            .collect();
        history.push(iter_elems);

        let current_u: Vec<(f64, f64)> = spec.shells.iter().map(|s| (s.u, s.j)).collect();
        let mut target_u = vec![(0.0, 0.0); n_shells];
        for (g, u, j) in &groups {
            for &sh in &g.shells {
                target_u[sh] = (*u, *j);
            }
        }
        let max_diff = current_u
            .iter()
            .zip(&target_u)
            .map(|((u0, _), (u1, _))| (u1 - u0).abs())
            .fold(0.0_f64, f64::max);

        let damped: Vec<(f64, f64)> = current_u
            .iter()
            .zip(&target_u)
            .map(|(&(u0, _), &(u1, j1))| {
                let next_u = (1.0 - options.damping) * u0 + options.damping * u1;
                (next_u.max(0.0), j1.min(next_u.max(0.0)))
            })
            .collect();
        spec.set_interactions_with_source(&damped, "lr-scf")?;

        last_response = Some(res);
        let next_state = crystal.solve_with_hubbard(
            &options.scf_options,
            Some(current),
            None,
            Some(&spec),
        )?;
        if !next_state.converged {
            return Err(invalid(format!(
                "SCF failed to converge at LR-12 iteration {iteration}"
            )));
        }
        owned_state = Some(next_state);
        if max_diff <= options.tolerance {
            converged = true;
            break;
        }
    }

    let response = last_response.expect("max_iterations >= 1 ran at least one iteration");
    let state = owned_state.expect("at least one iteration solved");
    let elements = history.last().cloned().unwrap_or_default();
    Ok(SelfConsistentResponse {
        iterations: history.len(),
        converged,
        elements,
        history,
        final_hubbard: spec,
        response,
        state,
    })
}
