//! Reversible bounded architecture edits, including duplicate action masses.
//!
//! Three moves: insert, delete and replace (no swap). An insertion draws its operator from the
//! proposal softmax and its position uniformly. A deletion or a replacement draws its POSITION by
//! importance: operator `i`'s importance is the |gradient| it would have if re-inserted at its own
//! position into the current sequence (its angle set to zero, the others relaxed), and a less
//! important operator is more likely to be removed or replaced ([`removal`]). The importances are a
//! function of the evaluated state, so the reverse move's probability is computed exactly and the
//! Metropolis-Hastings test stays exact.
use crate::{math::random::Rng, product, Context, Error, Reservation, Result};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Insert,
    Delete,
    Replace,
}
impl Move {
    pub fn name(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Delete => "delete",
            Self::Replace => "replace",
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Insert => 0,
            Self::Delete => 1,
            Self::Replace => 2,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub kind: Move,
    pub position: usize,
    pub operator: usize,
}
impl Action {
    pub fn apply(self, source: &[usize]) -> Vec<usize> {
        let mut result = source.to_vec();
        match self.kind {
            Move::Insert => result.insert(self.position, self.operator),
            Move::Delete => {
                result.remove(self.position);
            }
            Move::Replace => result[self.position] = self.operator,
        }
        result
    }
    pub fn transfer(self, parameters: &[f64], inherit: bool) -> Vec<f64> {
        let mut result = parameters.to_vec();
        match self.kind {
            Move::Insert => result.insert(self.position, 0.),
            Move::Delete => {
                result.remove(self.position);
            }
            Move::Replace => {
                if !inherit {
                    result[self.position] = 0.;
                }
            }
        }
        result
    }
    fn matches(self, source: &[usize], target: &[usize]) -> bool {
        match self.kind {
            Move::Insert => {
                target.len() == source.len() + 1
                    && target[self.position] == self.operator
                    && source[..self.position] == target[..self.position]
                    && source[self.position..] == target[self.position + 1..]
            }
            Move::Delete => {
                source.len() == target.len() + 1
                    && source[..self.position] == target[..self.position]
                    && source[self.position + 1..] == target[self.position..]
            }
            Move::Replace => {
                source.len() == target.len()
                    && target[self.position] == self.operator
                    && source
                        .iter()
                        .zip(target)
                        .enumerate()
                        .all(|(i, (a, b))| i == self.position || a == b)
            }
        }
    }
}
pub struct Actions {
    pub items: Vec<(Action, f64)>,
    _memory: Reservation,
}
pub struct Kernel {
    pub pool: usize,
    pub max_length: usize,
    pub weights: [f64; 3],
}
impl Kernel {
    pub fn new(pool: usize, max_length: usize, weights: [f64; 3]) -> Result<Self> {
        if pool == 0
            || max_length == 0
            || weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || !weights.iter().sum::<f64>().is_finite()
            || weights[0] <= 0.
            || weights[1] <= 0.
        {
            return Err(invalid("architecture search needs a nonempty pool, positive length bound, and reversible insertion/deletion weights"));
        }
        Ok(Self {
            pool,
            max_length,
            weights,
        })
    }
    pub fn actions(
        &self,
        context: &Context,
        source: &[usize],
        probabilities: &[f64],
        removal: &[f64],
    ) -> Result<Actions> {
        let length = source.len();
        if removal.len() != length
            || removal.iter().any(|p| !p.is_finite() || *p <= 0.)
            || (length > 0 && (removal.iter().sum::<f64>() - 1.).abs() > 1e-9)
        {
            return Err(invalid("removal weights must be positive, normalized and one per operator"));
        }
        if length > self.max_length
            || source.iter().any(|i| *i >= self.pool)
            || probabilities.len() != self.pool
            || probabilities.iter().any(|p| !p.is_finite() || *p <= 0.)
            || (probabilities.iter().sum::<f64>() - 1.).abs() > 1e-9
        {
            return Err(invalid(
                "architecture indices and positive normalized proposal weights must match the pool",
            ));
        }
        let valid = [length < self.max_length, length > 0, length > 0 && self.pool > 1];
        let normalization = self
            .weights
            .iter()
            .zip(valid)
            .filter(|(_, v)| *v)
            .map(|(w, _)| w)
            .sum::<f64>();
        let mass = |kind: Move| {
            if valid[kind.index()] {
                self.weights[kind.index()] / normalization
            } else {
                0.
            }
        };
        let capacity = product(&[length.checked_add(1).ok_or(Error::Overflow)?, self.pool, 2])?
            .checked_add(length)
            .ok_or(Error::Overflow)?;
        let memory = context.reserve(product(&[capacity, 64])?)?;
        let mut items = Vec::with_capacity(capacity);
        if mass(Move::Insert) > 0. {
            for position in 0..=length {
                for (operator, &p) in probabilities.iter().enumerate() {
                    items.push((
                        Action {
                            kind: Move::Insert,
                            position,
                            operator,
                        },
                        mass(Move::Insert) * p / (length + 1) as f64,
                    ));
                }
            }
        }
        if mass(Move::Delete) > 0. {
            for position in 0..length {
                items.push((
                    Action {
                        kind: Move::Delete,
                        position,
                        operator: 0,
                    },
                    mass(Move::Delete) * removal[position],
                ));
            }
        }
        if mass(Move::Replace) > 0. {
            for (position, &current) in source.iter().enumerate() {
                let remaining = probabilities
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != current)
                    .map(|(_, p)| p)
                    .sum::<f64>();
                for (operator, &p) in probabilities
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != current)
                {
                    items.push((
                        Action {
                            kind: Move::Replace,
                            position,
                            operator,
                        },
                        mass(Move::Replace) * removal[position] * p / remaining,
                    ));
                }
            }
        }
        Ok(Actions {
            items,
            _memory: memory,
        })
    }
    pub fn draw(
        &self,
        context: &Context,
        source: &[usize],
        probabilities: &[f64],
        removal: &[f64],
        random: &mut Rng,
    ) -> Result<Action> {
        let actions = self.actions(context, source, probabilities, removal)?;
        let mut draw = random.uniform();
        for (action, p) in &actions.items {
            draw -= p;
            if draw < 0. {
                return Ok(*action);
            }
        }
        actions
            .items
            .last()
            .map(|(a, _)| *a)
            .ok_or_else(|| invalid("no reversible architecture edit is available"))
    }
    pub fn log_q(
        &self,
        context: &Context,
        target: &[usize],
        source: &[usize],
        probabilities: &[f64],
        removal: &[f64],
    ) -> Result<f64> {
        Ok(self
            .actions(context, source, probabilities, removal)?
            .items
            .iter()
            .filter(|(a, _)| a.matches(source, target))
            .map(|(_, p)| p)
            .sum::<f64>()
            .ln())
    }
}
/// The position weights of a deletion or a replacement: `exp(-(|g_i| / max|g|) / T)` over the
/// operators' importances `g` (uniform when every importance is zero), so the least important
/// operator is the most likely to go.
pub fn removal(importances: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if !temperature.is_finite() || temperature <= 0. || importances.iter().any(|g| !g.is_finite()) {
        return Err(invalid("removal weights need finite importances and a positive temperature"));
    }
    if importances.is_empty() {
        return Ok(vec![]);
    }
    let scale = importances.iter().map(|g| g.abs()).fold(0., f64::max);
    if scale == 0. {
        return Ok(vec![1. / importances.len() as f64; importances.len()]);
    }
    let mut weights: Vec<_> = importances.iter().map(|g| (-(g.abs() / scale) / temperature).exp()).collect();
    if weights.contains(&0.) {
        return Err(invalid("proposal temperature loses representable reverse support"));
    }
    let total = weights.iter().sum::<f64>();
    for w in &mut weights {
        *w /= total;
    }
    Ok(weights)
}
pub fn softmax(gradients: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if !temperature.is_finite() || temperature <= 0. || gradients.iter().any(|g| !g.is_finite()) {
        return Err(invalid(
            "gradient softmax needs finite gradients and positive temperature",
        ));
    }
    if gradients.is_empty() {
        return Ok(vec![]);
    }
    let scale = gradients.iter().map(|g| g.abs()).fold(0., f64::max);
    if scale == 0. {
        return Ok(vec![1. / gradients.len() as f64; gradients.len()]);
    }
    let mut weights: Vec<_> = gradients
        .iter()
        .map(|g| ((g.abs() / scale - 1.) / temperature).exp())
        .collect();
    if weights.contains(&0.) {
        return Err(invalid(
            "proposal temperature loses representable reverse support",
        ));
    }
    let total = weights.iter().sum::<f64>();
    for w in &mut weights {
        *w /= total;
    }
    Ok(weights)
}
pub fn log_acceptance(
    temperature: f64,
    current: f64,
    proposed: f64,
    forward: f64,
    reverse: f64,
) -> f64 {
    if !proposed.is_finite() || reverse == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if temperature == 0. {
        return if proposed < current {
            f64::INFINITY
        } else if proposed > current {
            f64::NEG_INFINITY
        } else {
            reverse - forward
        };
    }
    -(proposed - current) / temperature + reverse - forward
}
