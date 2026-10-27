use super::{
    algebra::{excitation_pool, Generator},
    evolution, optimizer, Basis, Encoding, Operator, C64,
};
use crate::{product, Buffer, Context, Error, Reservation, Result};
use serde_json::Value;
use std::{fs::File, io::Read, path::Path, time::Instant};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn integer(value: &Value, name: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(&format!("{name} must be a nonnegative integer")))
}

pub struct CachedProblem {
    pub context: Context,
    pub encoding: Encoding,
    pub hamiltonian: Operator,
    pub basis: Basis,
    pub pool: Vec<Generator>,
    pub dense: Option<Buffer<C64>>,
    pub sparse: Option<super::sparse::SparseMatrix>,
}
impl CachedProblem {
    /// Read a bounded schema-v1 JSON cache; integrals and quantum SDKs are absent.
    pub fn load(
        ctx: &Context,
        path: &Path,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| invalid(&format!("cannot open Hamiltonian cache: {e}")))?;
        let bytes = usize::try_from(file.metadata().map_err(|e| invalid(&e.to_string()))?.len())
            .map_err(|_| Error::Overflow)?;
        if bytes > 64 * 1024 * 1024 {
            return Err(invalid(
                "JSON Hamiltonian cache exceeds the 64 MiB input limit",
            ));
        }
        // Conservatively cover parsed JSON nodes, source bytes and term snapshots.
        let _json = ctx.reserve(product(&[
            bytes.checked_add(1).ok_or(Error::Overflow)?,
            32,
        ])?)?;
        let mut text = Vec::new();
        text.try_reserve_exact(bytes + 1)
            .map_err(|_| Error::Allocation)?;
        file.take(bytes as u64 + 1)
            .read_to_end(&mut text)
            .map_err(|e| invalid(&e.to_string()))?;
        if text.len() > bytes {
            return Err(invalid("Hamiltonian cache changed while reading"));
        }
        let payload: Value = serde_json::from_slice(&text)
            .map_err(|e| invalid(&format!("invalid JSON Hamiltonian cache: {e}")))?;
        if payload["format"].as_str() != Some("mandacaru-qubit-hamiltonian")
            || payload["version"].as_u64() != Some(1)
        {
            return Err(invalid(
                "expected a mandacaru-qubit-hamiltonian schema-v1 JSON cache",
            ));
        }
        if let Some(unit) = payload.get("metadata").and_then(|m| m.get("energy_unit")) {
            if unit.as_str() != Some("Ha") {
                return Err(invalid("cached Hamiltonian coefficients must be in Ha"));
            }
        }
        let n = integer(&payload["num_qubits"], "num_qubits")?;
        let m = integer(&payload["n_spatial_orbitals"], "n_spatial_orbitals")?;
        let particles = payload["num_particles"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| invalid("cache requires two spin-resolved particle counts"))?;
        let encoding = Encoding::new(
            payload["mapping"]
                .as_str()
                .ok_or_else(|| invalid("cache requires a mapping"))?,
            product(&[m, 2])?,
            [
                integer(&particles[0], "n_alpha")?,
                integer(&particles[1], "n_beta")?,
            ],
        )?;
        if n != encoding.n_qubits() {
            return Err(invalid(
                "cache qubit count disagrees with its mapping and orbital count",
            ));
        }
        let rows = payload["terms"]
            .as_array()
            .ok_or_else(|| invalid("cache requires Pauli terms"))?;
        let mut terms = Vec::new();
        for row in rows {
            let row = row
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or_else(|| invalid("cache terms must be [label, real, imaginary]"))?;
            let label = row[0]
                .as_str()
                .ok_or_else(|| invalid("cache Pauli label must be a string"))?
                .to_owned();
            let re = row[1]
                .as_f64()
                .ok_or_else(|| invalid("cache coefficient must be numeric"))?;
            let im = row[2]
                .as_f64()
                .ok_or_else(|| invalid("cache coefficient must be numeric"))?;
            terms.push((label, C64::new(re, im)));
        }
        let hamiltonian = Operator::new(ctx, n, &terms)?;
        Self::new(ctx, encoding, hamiltonian, sector, dense, singles)
    }
    pub fn new(
        ctx: &Context,
        encoding: Encoding,
        hamiltonian: Operator,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<Self> {
        if hamiltonian.n != encoding.n_qubits() || !hamiltonian.is_hermitian() {
            return Err(invalid(
                "Hamiltonian must be Hermitian and match its encoding",
            ));
        }
        let basis = Basis::new(ctx, &encoding, sector)?;
        basis.check_invariant(&hamiltonian)?;
        let pool = excitation_pool(ctx, &encoding, singles)?;
        for g in &pool {
            basis.check_invariant(&g.operator)?;
        }
        let dense = if dense {
            Some(hamiltonian.dense(ctx, &basis)?)
        } else {
            None
        };
        Ok(Self {
            context: ctx.clone(),
            encoding,
            hamiltonian,
            basis,
            pool,
            dense,
            sparse: None,
        })
    }
    pub fn evaluate(
        &self,
        indices: &[usize],
        parameters: &[f64],
        trotter: bool,
    ) -> Result<evolution::Evaluation> {
        let ops: Result<Vec<_>> = indices
            .iter()
            .map(|&i| {
                self.pool
                    .get(i)
                    .map(|g| &g.operator)
                    .ok_or_else(|| invalid("generator index exceeds pool"))
            })
            .collect();
        evolution::evaluate(
            &self.context,
            &self.basis,
            &self.hamiltonian,
            self.dense.as_ref().map(|d| d.as_slice()),
            self.sparse.as_ref(),
            &ops?,
            parameters,
            trotter,
        )
    }
    pub fn solve(&self, options: &SolveOptions, initial: Option<&[f64]>) -> Result<SolveResult> {
        self.context.run(|| self.solve_inner(options, initial))
    }
    fn solve_inner(&self, options: &SolveOptions, initial: Option<&[f64]>) -> Result<SolveResult> {
        options.validate()?;
        let start = Instant::now();
        let ctx = &self.context;
        let max_parameters = if options.adapt {
            options.max_growth
        } else {
            self.pool.len()
        };
        // Hold output/history reservations for the entire lifetime of the result.
        let _memory = ctx.reserve(
            product(&[
                options.optimizer.max_evaluations,
                if options.adapt {
                    options.max_growth.max(1)
                } else {
                    1
                },
                8,
            ])?
            .checked_add(product(&[max_parameters, 256])?)
            .ok_or(Error::Overflow)?,
        )?;
        let reference = self.evaluate(&[], &[], true)?.energy;
        let mut indices = Vec::new();
        let mut parameters = Vec::new();
        let mut iterations = Vec::new();
        let mut history = Vec::new();
        let mut steps = 0;
        let mut success = false;
        let mut message = "growth budget exhausted".to_owned();
        let mut final_gradient = 0.;
        if !options.adapt {
            indices = (0..self.pool.len()).collect();
            parameters = initial
                .map(|x| x.to_vec())
                .unwrap_or_else(|| vec![0.; indices.len()]);
            if parameters.len() != indices.len() {
                return Err(invalid(
                    "initial parameter count differs from the UCCSD pool",
                ));
            }
            let outcome = optimizer::minimize(ctx, &parameters, &options.optimizer, |t| {
                let v = self.evaluate(&indices, t, options.trotter)?;
                Ok((v.energy, v.gradient))
            })?;
            parameters = outcome.parameters;
            history = outcome.history;
            steps = outcome.steps;
            success = outcome.success;
            message = outcome.message;
            final_gradient = self
                .evaluate(&indices, &parameters, options.trotter)?
                .gradient
                .iter()
                .fold(0f64, |a, g| a.max(g.abs()));
        } else {
            if initial.is_some_and(|p| !p.is_empty()) {
                return Err(invalid("basic ADAPT starts with an empty ansatz"));
            }
            let mut delta = None;
            for growth in 0..=options.max_growth {
                let evaluation = self.evaluate(&indices, &parameters, true)?;
                let energy = evaluation.energy;
                let hpsi = self
                    .hamiltonian
                    .apply(ctx, &self.basis, evaluation.state.as_slice())?;
                let mut action = ctx.allocate::<C64>(self.basis.len())?;
                let mut selected = 0;
                final_gradient = 0.;
                for (i, g) in self.pool.iter().enumerate() {
                    g.operator.apply_into(
                        &self.basis,
                        evaluation.state.as_slice(),
                        action.as_mut_slice(),
                    );
                    let magnitude =
                        (2. * evolution::inner(hpsi.as_slice(), action.as_slice()).re).abs();
                    if magnitude > final_gradient {
                        final_gradient = magnitude;
                        selected = i;
                    }
                }
                let gradient_ok = options
                    .gradient_tolerance
                    .is_none_or(|t| final_gradient < t);
                let energy_ok = options
                    .energy_tolerance
                    .is_none_or(|t| delta.is_some_and(|d: f64| d.abs() < t));
                if final_gradient < 1e-12 || (gradient_ok && energy_ok) {
                    success = true;
                    message = "pool gradients converged".into();
                    break;
                }
                if growth == options.max_growth {
                    break;
                }
                indices.push(selected);
                parameters.push(0.);
                let outcome = optimizer::minimize(ctx, &parameters, &options.optimizer, |t| {
                    let v = self.evaluate(&indices, t, true)?;
                    Ok((v.energy, v.gradient))
                })?;
                delta = Some(outcome.energy - energy);
                parameters = outcome.parameters;
                steps += outcome.steps;
                iterations.push(Iteration {
                    pool_index: selected,
                    energy: outcome.energy,
                    max_gradient: final_gradient,
                    optimizer_steps: outcome.steps,
                    num_evaluations: outcome.history.len(),
                    success: outcome.success,
                    message: outcome.message.clone(),
                });
                history.extend(outcome.history);
                if !outcome.success {
                    message = format!("inner optimization failed: {}", outcome.message);
                    // Recompute the final screening below, after the updated state.
                    break;
                }
            }
        }
        let evaluation = self.evaluate(
            &indices,
            &parameters,
            if options.adapt { true } else { options.trotter },
        )?;
        let energy = evaluation.energy;
        if options.adapt {
            let hpsi = self
                .hamiltonian
                .apply(ctx, &self.basis, evaluation.state.as_slice())?;
            let mut action = ctx.allocate(self.basis.len())?;
            final_gradient = 0.;
            for g in &self.pool {
                g.operator.apply_into(
                    &self.basis,
                    evaluation.state.as_slice(),
                    action.as_mut_slice(),
                );
                final_gradient = final_gradient
                    .max((2. * evolution::inner(hpsi.as_slice(), action.as_slice()).re).abs());
            }
        }
        let labels = indices
            .iter()
            .map(|&i| self.pool[i].label.clone())
            .collect();
        Ok(SolveResult {
            energy,
            reference,
            parameters,
            state: evaluation.state,
            indices,
            labels,
            iterations,
            history,
            optimizer_steps: steps,
            success,
            message,
            final_gradient,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            _memory,
        })
    }
}
#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub adapt: bool,
    pub trotter: bool,
    pub max_growth: usize,
    pub gradient_tolerance: Option<f64>,
    pub energy_tolerance: Option<f64>,
    pub optimizer: optimizer::Options,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            adapt: false,
            trotter: false,
            max_growth: 50,
            gradient_tolerance: Some(1e-7),
            energy_tolerance: None,
            optimizer: optimizer::Options::default(),
        }
    }
}
impl SolveOptions {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        if self
            .gradient_tolerance
            .into_iter()
            .chain(self.energy_tolerance)
            .any(|t| !t.is_finite() || t <= 0.)
            || (self.adapt && self.gradient_tolerance.is_none() && self.energy_tolerance.is_none())
        {
            return Err(invalid(
                "ADAPT needs positive finite convergence thresholds and at least one criterion",
            ));
        }
        Ok(())
    }
}
pub struct Iteration {
    pub pool_index: usize,
    pub energy: f64,
    pub max_gradient: f64,
    pub optimizer_steps: usize,
    pub num_evaluations: usize,
    pub success: bool,
    pub message: String,
}
pub struct SolveResult {
    pub energy: f64,
    pub reference: f64,
    pub parameters: Vec<f64>,
    pub state: Buffer<C64>,
    pub indices: Vec<usize>,
    pub labels: Vec<String>,
    pub iterations: Vec<Iteration>,
    pub history: Vec<f64>,
    pub optimizer_steps: usize,
    pub success: bool,
    pub message: String,
    pub final_gradient: f64,
    pub elapsed_seconds: f64,
    _memory: Reservation,
}
