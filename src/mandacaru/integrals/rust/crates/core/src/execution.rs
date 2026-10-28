//! Provider-independent QWC grouping, count reductions, gate streams and quotas.
use crate::{product, Context, Error, Reservation, Result};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn labels_width(labels: &[String]) -> Result<usize> {
    let n = labels.first().map(String::len).unwrap_or(0);
    if labels
        .iter()
        .any(|s| s.len() != n || s.bytes().any(|c| !b"IXYZ".contains(&c)))
    {
        return Err(invalid(
            "Pauli labels must have one width and contain only IXYZ",
        ));
    }
    Ok(n)
}
pub struct Groups {
    pub groups: Vec<(String, Vec<usize>)>,
    _memory: Reservation,
}
pub fn qwc(ctx: &Context, labels: &[String], weights: Option<&[f64]>) -> Result<Groups> {
    let n = labels_width(labels)?;
    if weights.is_some_and(|w| w.len() != labels.len() || w.iter().any(|x| !x.is_finite())) {
        return Err(invalid("finite grouping weights must match labels"));
    }
    let memory = ctx.reserve(product(&[
        labels.len(),
        n.checked_add(128).ok_or(Error::Overflow)?,
        4,
    ])?)?;
    let mut order: Vec<_> = (0..labels.len()).collect();
    order.sort_by(|&i, &j| {
        weights
            .map(|w| w[j].partial_cmp(&w[i]).unwrap())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| labels[i].cmp(&labels[j]))
    });
    let mut bases: Vec<(Vec<u8>, Vec<usize>)> = Vec::new();
    for i in order {
        let label = labels[i].as_bytes();
        if let Some((basis, members)) = bases.iter_mut().find(|(b, _)| {
            b.iter()
                .zip(label)
                .all(|(&a, &b)| a == b || a == b'I' || b == b'I')
        }) {
            for (a, &b) in basis.iter_mut().zip(label) {
                if *a == b'I' {
                    *a = b;
                }
            }
            members.push(i);
        } else {
            bases.push((label.to_vec(), vec![i]));
        }
    }
    Ok(Groups {
        groups: bases
            .into_iter()
            .map(|(b, m)| (String::from_utf8(b).unwrap(), m))
            .collect(),
        _memory: memory,
    })
}
pub fn expectations(
    ctx: &Context,
    labels: &[String],
    counts: &[(String, f64)],
) -> Result<Vec<f64>> {
    let n = labels_width(labels)?;
    if counts.iter().any(|(bits, c)| {
        bits.len() != n || bits.bytes().any(|b| b != b'0' && b != b'1') || !c.is_finite() || *c < 0.
    }) {
        return Err(invalid(
            "counts require matching binary strings and finite nonnegative frequencies",
        ));
    }
    let _memory = ctx.reserve(product(&[
        labels
            .len()
            .checked_add(counts.len())
            .ok_or(Error::Overflow)?,
        n.checked_add(32).ok_or(Error::Overflow)?,
    ])?)?;
    let total: f64 = counts.iter().map(|(_, c)| c).sum();
    if !total.is_finite() {
        return Err(invalid("total count overflows"));
    }
    Ok(labels
        .iter()
        .map(|label| {
            if label.bytes().all(|c| c == b'I') {
                return 1.;
            }
            if total == 0. {
                return 0.;
            }
            counts
                .iter()
                .map(|(bits, c)| {
                    let parity = label
                        .bytes()
                        .zip(bits.bytes())
                        .filter(|(p, b)| *p != b'I' && *b == b'1')
                        .count()
                        % 2;
                    c / total * if parity == 0 { 1. } else { -1. }
                })
                .sum()
        })
        .collect())
}
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: &'static str,
    pub first: usize,
    pub second: Option<usize>,
    pub angle: Option<f64>,
}
pub struct Circuit {
    pub gates: Vec<Gate>,
    _memory: Reservation,
}
/// Commuting imaginary Pauli generators. Global identity phases are omitted,
/// matching the SDK circuit contract; checkpoint reconstruction retains them.
pub fn compile(
    ctx: &Context,
    n: usize,
    occupied: &[usize],
    generators: &[Vec<(String, f64, f64)>],
    parameters: &[f64],
) -> Result<Circuit> {
    if n > 62
        || occupied.iter().any(|q| *q >= n)
        || generators.len() != parameters.len()
        || parameters.iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "finite circuit angles and reference must match the register and generators",
        ));
    }
    let terms = generators.iter().try_fold(0usize, |sum, g| {
        sum.checked_add(g.len()).ok_or(Error::Overflow)
    })?;
    let capacity = product(&[
        terms,
        n.checked_mul(6)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::Overflow)?,
    ])?
    .checked_add(occupied.len())
    .ok_or(Error::Overflow)?;
    let memory = ctx.reserve(product(&[capacity, std::mem::size_of::<Gate>()])?)?;
    let _workspace = ctx.reserve(product(&[
        terms,
        n.checked_add(128).ok_or(Error::Overflow)?,
        4,
    ])?)?;
    let mut gates = Vec::new();
    gates
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    for &q in occupied {
        gates.push(Gate {
            name: "x",
            first: q,
            second: None,
            angle: None,
        });
    }
    for (generator, theta) in generators.iter().zip(parameters) {
        let mut terms = std::collections::BTreeMap::<String, (f64, f64)>::new();
        for (label, re, im) in generator {
            if label.len() != n
                || label.bytes().any(|c| !b"IXYZ".contains(&c))
                || !re.is_finite()
                || !im.is_finite()
            {
                return Err(invalid("circuit generator contains invalid Pauli terms"));
            }
            let entry = terms.entry(label.clone()).or_default();
            entry.0 += re;
            entry.1 += im;
        }
        let active: Vec<_> = terms
            .into_iter()
            .filter(|(_, c)| c.0.hypot(c.1) > 1e-12)
            .collect();
        for (label, (re, im)) in &active {
            if !re.is_finite() || !im.is_finite() || re.abs() > 1e-8 * im.abs().max(1.) {
                return Err(invalid("circuit generators must be anti-Hermitian"));
            }
            for (other, (_, other_im)) in &active {
                if im.abs() <= 1e-12 || other_im.abs() <= 1e-12 {
                    continue;
                }
                let parity = label
                    .bytes()
                    .zip(other.bytes())
                    .filter(|(a, b)| a != b && *a != b'I' && *b != b'I')
                    .count()
                    % 2;
                if parity != 0 {
                    return Err(invalid("generator terms anticommute; exact exponential is not a product of Pauli rotations"));
                }
            }
        }
        for (label, (_, coefficient)) in active {
            if coefficient.abs() <= 1e-12 {
                continue;
            }
            let angle = theta * coefficient;
            if !angle.is_finite() || !(-2. * angle).is_finite() {
                return Err(invalid("circuit rotation angle overflows"));
            }
            if angle == 0. {
                continue;
            }
            let active: Vec<_> = label
                .bytes()
                .enumerate()
                .filter(|(_, c)| *c != b'I')
                .collect();
            if active.is_empty() {
                continue;
            }
            let mut push = |name, first, second, angle| {
                gates.push(Gate {
                    name,
                    first,
                    second,
                    angle,
                })
            };
            for &(q, c) in &active {
                if c == b'Y' {
                    push("sdg", q, None, None);
                }
                if c == b'Y' || c == b'X' {
                    push("h", q, None, None);
                }
            }
            for pair in active.windows(2) {
                push("cx", pair[0].0, Some(pair[1].0), None);
            }
            push("rz", active.last().unwrap().0, None, Some(-2. * angle));
            for pair in active.windows(2).rev() {
                push("cx", pair[0].0, Some(pair[1].0), None);
            }
            for &(q, c) in &active {
                if c == b'X' || c == b'Y' {
                    push("h", q, None, None);
                }
                if c == b'Y' {
                    push("s", q, None, None);
                }
            }
        }
    }
    Ok(Circuit {
        gates,
        _memory: memory,
    })
}
#[derive(Clone, Debug)]
pub struct Usage {
    pub attempted_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub total_shots: u64,
}
pub struct RunLedger {
    pub usage: Usage,
    context: Context,
    job_limit: Option<u64>,
    shot_limit: Option<u64>,
    history: Vec<(u64, crate::Buffer<u8>, Reservation)>,
}
impl RunLedger {
    pub fn new(context: &Context, job_limit: Option<u64>, shot_limit: Option<u64>) -> Self {
        Self {
            usage: Usage {
                attempted_jobs: 0,
                completed_jobs: 0,
                failed_jobs: 0,
                total_shots: 0,
            },
            context: context.clone(),
            job_limit,
            shot_limit,
            history: Vec::new(),
        }
    }
    pub fn reserve(&mut self, jobs: u64, shots: u64) -> Result<Vec<u64>> {
        let next = self
            .usage
            .attempted_jobs
            .checked_add(jobs)
            .ok_or(Error::Overflow)?;
        let total = self
            .usage
            .total_shots
            .checked_add(shots)
            .ok_or(Error::Overflow)?;
        if jobs == 0
            || self.job_limit.is_some_and(|v| next > v)
            || self.shot_limit.is_some_and(|v| total > v)
        {
            return Err(invalid(
                "submission would exceed run budget or contains no jobs",
            ));
        }
        let count = usize::try_from(jobs).map_err(|_| Error::Overflow)?;
        let memory = self.context.reserve(128)?;
        let _tickets = self.context.reserve(product(&[count, 8])?)?;
        let history = self.context.allocate::<u8>(count)?;
        let tickets: Vec<_> = (self.usage.attempted_jobs..next).collect();
        self.history
            .push((self.usage.attempted_jobs, history, memory));
        self.usage.attempted_jobs = next;
        self.usage.total_shots = total;
        Ok(tickets)
    }
    pub fn finish(&mut self, ticket: u64, success: bool) -> Result<()> {
        let entry = self
            .history
            .iter_mut()
            .rev()
            .find_map(|(start, states, _)| {
                ticket
                    .checked_sub(*start)
                    .and_then(|i| usize::try_from(i).ok())
                    .and_then(|i| states.as_mut_slice().get_mut(i))
            })
            .ok_or_else(|| invalid("unknown or already completed job ticket"))?;
        if *entry != 0 {
            return Err(invalid("unknown or already completed job ticket"));
        }
        *entry = if success { 1 } else { 2 };
        if success {
            self.usage.completed_jobs += 1;
        } else {
            self.usage.failed_jobs += 1;
        }
        Ok(())
    }
}
