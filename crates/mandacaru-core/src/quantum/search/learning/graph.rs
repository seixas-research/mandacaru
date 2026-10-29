//! Hamiltonian factor-graph and excitation descriptors for learned proposals.
use crate::{
    product,
    quantum::{CachedProblem, Encoding, C64},
    Context, Error, Reservation, Result,
};
use nalgebra::DMatrix;
use serde_json::Value;
pub type Matrix = DMatrix<f64>;
pub struct Graph {
    pub qubit: Matrix,
    pub term: Matrix,
    pub incidence: [Matrix; 3],
    pub weighted: [Matrix; 3],
    pub pool: [Matrix; 3],
    pub descriptors: Matrix,
    pub term_weights: Vec<f64>,
    pub support: Vec<f64>,
    pub reach: Vec<f64>,
    pub global: Matrix,
    pub reference_energy: f64,
    _memory: Reservation,
}
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn terms(value: &Value, n: usize) -> Result<Vec<(String, C64)>> {
    value
        .as_array()
        .ok_or_else(|| invalid("model terms must be an array"))?
        .iter()
        .map(|value| {
            let row = value
                .as_array()
                .filter(|v| v.len() == 3)
                .ok_or_else(|| invalid("model terms need label, real and imaginary coefficient"))?;
            let label = row[0]
                .as_str()
                .filter(|v| v.len() == n && v.bytes().all(|b| b"IXYZ".contains(&b)))
                .ok_or_else(|| invalid("model Pauli labels disagree with the register"))?;
            let real = row[1]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid("model coefficients must be finite"))?;
            let imag = row[2]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid("model coefficients must be finite"))?;
            Ok((label.to_owned(), C64::new(real, imag)))
        })
        .collect()
}
impl Graph {
    pub fn from_metadata(context: &Context, value: &Value) -> Result<Self> {
        if value["format"] != "mandacaru-search-problem" || value["version"] != 1 {
            return Err(invalid("unsupported learned-proposal problem format"));
        }
        let n = value["num_qubits"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("model problem needs a qubit count"))?;
        let m = value["spatial_orbitals"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("model problem needs an orbital count"))?;
        let particles = value["num_particles"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| invalid("model problem needs two spin counts"))?;
        let a = particles[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("invalid model alpha count"))?;
        let b = particles[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("invalid model beta count"))?;
        let encoding = Encoding::new(
            value["mapping"]
                .as_str()
                .ok_or_else(|| invalid("model problem needs an encoding"))?,
            m.checked_mul(2).ok_or(Error::Overflow)?,
            [a, b],
        )?;
        if encoding.n_qubits() != n {
            return Err(invalid(
                "model problem encoding disagrees with its register",
            ));
        }
        let h = terms(&value["hamiltonian"], n)?;
        let pool = value["pool"]
            .as_array()
            .ok_or_else(|| invalid("model problem needs an excitation pool"))?
            .iter()
            .map(|v| terms(v, n))
            .collect::<Result<Vec<_>>>()?;
        Self::new(context, n, encoding.reference(), &h, &pool)
    }
    pub fn from_problem(problem: &CachedProblem) -> Result<Self> {
        let _temporary = problem.context.reserve(product(&[
            problem.hamiltonian.terms.len()
                + problem
                    .pool
                    .iter()
                    .map(|g| g.operator.terms.len())
                    .sum::<usize>(),
            problem.hamiltonian.n + 64,
        ])?)?;
        let pool = problem
            .pool
            .iter()
            .map(|g| g.operator.labels())
            .collect::<Vec<_>>();
        Self::new(
            &problem.context,
            problem.hamiltonian.n,
            problem.encoding.reference(),
            &problem.hamiltonian.labels(),
            &pool,
        )
    }
    pub fn new(
        context: &Context,
        n: usize,
        reference: u64,
        h: &[(String, C64)],
        pool: &[Vec<(String, C64)>],
    ) -> Result<Self> {
        if n == 0 || n > 62 || reference >= (1u64 << n) || h.iter().any(|(_, c)| c.im.abs() > 1e-12)
        {
            return Err(invalid(
                "factor graphs need a positive register and a Hermitian Hamiltonian",
            ));
        }
        let t = h.len();
        let p = pool.len();
        let sizes = [
            product(&[t, n, 6])?,
            product(&[p, n, 3])?,
            product(&[t, 8])?,
            product(&[n, 6])?,
            product(&[p, 4])?,
            3,
        ];
        let cells = sizes
            .into_iter()
            .try_fold(0usize, |a, b| a.checked_add(b).ok_or(Error::Overflow))?;
        let memory = context.reserve(
            product(&[cells, 8])?
                .checked_add(product(&[n + t + p, 128])?)
                .ok_or(Error::Overflow)?,
        )?;
        let mut incidence = std::array::from_fn(|_| Matrix::zeros(t, n));
        let mut weighted = std::array::from_fn(|_| Matrix::zeros(t, n));
        let mut pool_edges = std::array::from_fn(|_| Matrix::zeros(p, n));
        let mut descriptors = Matrix::zeros(p, 4);
        let mut term = Matrix::zeros(t, 6);
        let mut qubit = Matrix::zeros(n, 4);
        let scale = h.iter().map(|(_, c)| c.norm()).fold(0., f64::max);
        let scale = if scale > 0. { scale } else { 1. };
        let mut reference_energy = 0.;
        let mut weights = vec![0.; t];
        let mut support = vec![0.; t];
        let mut reach = vec![0.; n];
        let mut degree = vec![0.; n];
        let letter = |b: u8| match b {
            b'X' => Some(0),
            b'Y' => Some(1),
            b'Z' => Some(2),
            _ => None,
        };
        for (a, (label, coefficient)) in h.iter().enumerate() {
            if label.len() != n
                || label.bytes().any(|b| !b"IXYZ".contains(&b))
                || !coefficient.re.is_finite()
                || !coefficient.im.is_finite()
            {
                return Err(invalid("invalid factor graph Hamiltonian term"));
            }
            let magnitude = coefficient.norm() / scale;
            weights[a] = magnitude;
            let counts: Vec<_> = (0..3)
                .map(|s| label.bytes().filter(|&b| letter(b) == Some(s)).count())
                .collect();
            let size = counts.iter().sum::<usize>();
            if counts[0] + counts[1] == 0 {
                let odd = label
                    .bytes()
                    .enumerate()
                    .filter(|(i, b)| *b == b'Z' && reference & (1 << (n - 1 - i)) != 0)
                    .count()
                    % 2
                    != 0;
                reference_energy += coefficient.re * if odd { -1. } else { 1. };
            }
            term[(a, 0)] = magnitude;
            term[(a, 1)] = (magnitude + 1e-6).ln();
            term[(a, 2)] = size as f64 / n as f64;
            for s in 0..3 {
                term[(a, 3 + s)] = counts[s] as f64 / n as f64;
            }
            for (i, b) in label.bytes().enumerate() {
                if let Some(s) = letter(b) {
                    incidence[s][(a, i)] = 1.;
                    weighted[s][(a, i)] = magnitude;
                    degree[i] += 1.;
                    reach[i] += magnitude;
                    if size == 1 && s == 2 {
                        qubit[(i, 2)] += coefficient.re / scale;
                    }
                }
            }
            support[a] = 1. / (size as f64 + 1e-12);
        }
        for (i, row) in reach.iter_mut().enumerate() {
            *row = 1. / (*row + 1e-12);
            qubit[(i, 0)] = if reference & (1 << (n - 1 - i)) != 0 {
                1.
            } else {
                0.
            };
            qubit[(i, 1)] = i as f64 / n.saturating_sub(1).max(1) as f64;
            qubit[(i, 3)] = f64::ln_1p(degree[i]) / f64::ln_1p(t.max(1) as f64);
        }
        for (mu, terms) in pool.iter().enumerate() {
            let total = terms.iter().map(|(_, c)| c.norm()).sum::<f64>();
            if total == 0. {
                continue;
            }
            for (label, coefficient) in terms {
                if label.len() != n
                    || label.bytes().any(|b| !b"IXYZ".contains(&b))
                    || !coefficient.re.is_finite()
                    || !coefficient.im.is_finite()
                {
                    return Err(invalid("invalid factor graph excitation term"));
                }
                let counts: Vec<_> = (0..3)
                    .map(|s| label.bytes().filter(|&b| letter(b) == Some(s)).count())
                    .collect();
                let size = counts.iter().sum::<usize>();
                let denominator = size.max(1) as f64;
                let weight = coefficient.norm() / total;
                descriptors[(mu, 0)] += weight * size as f64 / n as f64;
                for s in 0..3 {
                    descriptors[(mu, s + 1)] += weight * counts[s] as f64 / denominator;
                }
                for (i, b) in label.bytes().enumerate() {
                    if let Some(s) = letter(b) {
                        pool_edges[s][(mu, i)] += weight / denominator;
                    }
                }
            }
        }
        Ok(Self {
            qubit,
            term,
            incidence,
            weighted,
            pool: pool_edges,
            descriptors,
            term_weights: weights,
            support,
            reach,
            global: Matrix::from_row_slice(
                1,
                3,
                &[scale.ln(), (n as f64).ln(), (t as f64).ln_1p()],
            ),
            reference_energy,
            _memory: memory,
        })
    }
}
pub fn compress(value: f64) -> f64 {
    value.signum() * (value.abs() / 1e-5).ln_1p()
}
pub struct Candidates {
    pub occupancy: Matrix,
    pub positional: Matrix,
    pub operators: Vec<usize>,
    pub descriptors: Matrix,
    _memory: Reservation,
}
impl Candidates {
    pub fn new(
        context: &Context,
        graph: &Graph,
        sequence: &[usize],
        operators: &[usize],
        gradients: &[f64],
        energy_above_reference: f64,
    ) -> Result<Self> {
        let pool = graph.descriptors.nrows();
        let rows = operators.len();
        if gradients.len() != pool
            || sequence.iter().chain(operators).any(|&i| i >= pool)
            || gradients.iter().any(|g| !g.is_finite())
            || !energy_above_reference.is_finite()
        {
            return Err(invalid(
                "candidate descriptors must match the excitation pool",
            ));
        }
        let memory = context.reserve(
            pool.checked_mul(16)
                .and_then(|v| v.checked_add(rows.checked_mul(96)?))
                .ok_or(Error::Overflow)?,
        )?;
        let mut occupancy = Matrix::zeros(1, pool);
        let mut positional = Matrix::zeros(1, pool);
        let mut descriptors = Matrix::zeros(rows, 9);
        let maximum = gradients.iter().map(|g| g.abs()).fold(0., f64::max);
        let length = sequence.len();
        let sum_positions = (length as f64) * (length + 1) as f64 / 2.;
        for (position, &op) in sequence.iter().enumerate() {
            occupancy[(0, op)] += 1. / length.max(1) as f64;
            positional[(0, op)] += (position + 1) as f64 / sum_positions.max(1.);
        }
        for r in 0..rows {
            let mu = operators[r];
            for column in 0..4 {
                descriptors[(r, column)] = graph.descriptors[(mu, column)];
            }
            descriptors[(r, 4)] = if maximum > 0. {
                gradients[mu].abs() / maximum
            } else {
                0.
            };
            descriptors[(r, 5)] = compress(gradients[mu].abs());
            descriptors[(r, 6)] = compress(energy_above_reference);
            descriptors[(r, 7)] = sequence.iter().filter(|&&i| i == mu).count() as f64;
            descriptors[(r, 8)] = length as f64;
        }
        Ok(Self {
            occupancy,
            positional,
            operators: operators.to_vec(),
            descriptors,
            _memory: memory,
        })
    }
}
