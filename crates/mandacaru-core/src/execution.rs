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
/// What one measurement of `bases` qubit-wise-commuting bases costs: `(jobs, shots)`.
///
/// `max_bases_per_job` splits the bases over jobs (`None`: one job carries them all);
/// `job_per_basis` is a provider that submits one task per basis. Each basis is measured at every
/// one of `noise_factors` zero-noise-extrapolation factors with `shots`, or, when
/// `randomizations > 1`, with `randomizations x shots_per_randomization` (twirled instances share
/// the requested shots). The provider's own option schema decides those multipliers; this is the
/// arithmetic the run ledger is checked with before the first job is queued.
pub fn measurement_cost(
    bases: u64,
    max_bases_per_job: Option<u64>,
    job_per_basis: bool,
    shots: u64,
    noise_factors: u64,
    randomizations: u64,
    shots_per_randomization: u64,
) -> Result<(u64, u64)> {
    if max_bases_per_job == Some(0) || noise_factors == 0 {
        return Err(invalid("a job carries at least one basis and one noise factor"));
    }
    let jobs = if job_per_basis {
        bases
    } else {
        match max_bases_per_job {
            Some(limit) => bases.div_ceil(limit),
            None => u64::from(bases > 0),
        }
    };
    let per_instance = if randomizations > 1 {
        randomizations.checked_mul(shots_per_randomization).ok_or(Error::Overflow)?
    } else {
        shots
    };
    let total = bases.checked_mul(noise_factors).and_then(|v| v.checked_mul(per_instance))
        .ok_or(Error::Overflow)?;
    Ok((jobs, total))
}

/// `(values, standard errors)` of Pauli strings read off bit-string counts: the values of
/// [`expectations`], and for each the standard error `sqrt(max(1 - v^2, 0) / N)` of a +-1 outcome
/// averaged over the `N` counted shots (zero for an identity string or no shots).
pub fn count_statistics(
    ctx: &Context,
    labels: &[String],
    counts: &[(String, f64)],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let values = expectations(ctx, labels, counts)?;
    let total: f64 = counts.iter().map(|(_, c)| c).sum();
    let errors = labels
        .iter()
        .zip(&values)
        .map(|(label, v)| {
            if total <= 0. || label.bytes().all(|c| c == b'I') {
                0.
            } else {
                ((1. - v * v).max(0.) / total).sqrt()
            }
        })
        .collect();
    Ok((values, errors))
}

/// A Hamiltonian's measurement plan: its non-identity strings in qubit-wise-commuting groups
/// (`(basis, [(label, re, im)])`, largest coefficient first, as [`qwc`] groups them) and the
/// identity coefficient, which needs no measurement.
pub struct HamiltonianGroups {
    pub groups: Vec<(String, Vec<(String, f64, f64)>)>,
    pub identity: f64,
}

/// Group the strings of `terms` whose magnitude exceeds `atol`. Every all-`I` string adds the real
/// part of its coefficient to the identity; with `drop_identity` false that sum is measured as a
/// string of its own (in a group) instead, so the constant counts exactly once either way.
pub fn qwc_hamiltonian(
    ctx: &Context,
    terms: &[(String, f64, f64)],
    drop_identity: bool,
    atol: f64,
) -> Result<HamiltonianGroups> {
    if !atol.is_finite() || atol < 0. {
        return Err(invalid("the drop tolerance must be finite and nonnegative"));
    }
    let labels: Vec<String> = terms.iter().map(|(l, _, _)| l.clone()).collect();
    let n = labels_width(&labels)?;
    if terms.iter().any(|(_, re, im)| !re.is_finite() || !im.is_finite()) {
        return Err(invalid("Pauli coefficients must be finite"));
    }
    let mut identity = 0.;
    let mut kept: Vec<(String, f64, f64)> = Vec::new();
    for (label, re, im) in terms {
        if re.hypot(*im) <= atol {
            continue;
        }
        if label.bytes().all(|c| c == b'I') {
            identity += re;
        } else {
            kept.push((label.clone(), *re, *im));
        }
    }
    if !drop_identity && identity != 0. {
        kept.push(("I".repeat(n), identity, 0.));
        identity = 0.;
    }
    let names: Vec<String> = kept.iter().map(|(l, _, _)| l.clone()).collect();
    let weights: Vec<f64> = kept.iter().map(|(_, re, im)| re.hypot(*im)).collect();
    let grouped = qwc(ctx, &names, Some(&weights))?;
    Ok(HamiltonianGroups {
        groups: grouped
            .groups
            .iter()
            .map(|(basis, members)| (basis.clone(), members.iter().map(|&i| kept[i].clone()).collect()))
            .collect(),
        identity,
    })
}

/// The coefficient one-norm bound on the standard error of a shot-based energy at `shots` per
/// group: `sum |c_j| / sqrt(shots)` over the non-identity strings above `atol` (each `<P_j>` has
/// variance at most one). `shots` below one counts as one.
pub fn shot_noise(terms: &[(String, f64, f64)], shots: u64, atol: f64) -> Result<f64> {
    if terms.iter().any(|(_, re, im)| !re.is_finite() || !im.is_finite()) {
        return Err(invalid("Pauli coefficients must be finite"));
    }
    let norm: f64 = terms
        .iter()
        .filter(|(label, re, im)| re.hypot(*im) > atol && !label.bytes().all(|c| c == b'I'))
        .map(|(_, re, im)| re.hypot(*im))
        .sum();
    Ok(norm / (shots.max(1) as f64).sqrt())
}

/// `(randomizations, shots per randomization)` of twirled circuit instances sharing `shots`, by
/// the Runtime estimator's rule: both unset, `max(64, ceil(shots / 32))` shots per randomization
/// and `ceil(shots / that)` randomizations; one given, the other `ceil(shots / given)`. Each is
/// at least one.
pub fn twirled_instances(
    shots: u64,
    randomizations: Option<u64>,
    shots_per_randomization: Option<u64>,
) -> Result<(u64, u64)> {
    if randomizations == Some(0) || shots_per_randomization == Some(0) {
        return Err(invalid("twirling needs at least one randomization and one shot per randomization"));
    }
    let (count, per) = match (randomizations, shots_per_randomization) {
        (None, None) => {
            let per = shots.div_ceil(32).max(64);
            (shots.div_ceil(per), per)
        }
        (None, Some(per)) => (shots.div_ceil(per), per),
        (Some(count), None) => (count, shots.div_ceil(count)),
        (Some(count), Some(per)) => (count, per),
    };
    Ok((count.max(1), per.max(1)))
}

/// The positions in `labels` of each submission when their qubit-wise-commuting bases ([`qwc`])
/// are split `max_bases` to a submission, whole bases only. No limit (`None` or zero), or one
/// the bases fit in, gives one submission of every label in its order; no labels give none.
pub fn basis_chunks(
    ctx: &Context,
    labels: &[String],
    weights: Option<&[f64]>,
    max_bases: Option<u64>,
) -> Result<Vec<Vec<usize>>> {
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    let whole = || vec![(0..labels.len()).collect::<Vec<_>>()];
    let Some(limit) = max_bases.filter(|&v| v > 0) else {
        labels_width(labels)?;
        return Ok(whole());
    };
    let partition = qwc(ctx, labels, weights)?;
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    if partition.groups.len() <= limit {
        return Ok(whole());
    }
    Ok(partition
        .groups
        .chunks(limit)
        .map(|bases| bases.iter().flat_map(|(_, members)| members.iter().copied()).collect::<Vec<_>>())
        .filter(|chunk| !chunk.is_empty())
        .collect())
}

/// `(strings, bases)` of one batch of measurement requests: the Pauli strings of every request
/// and the qubit-wise-commuting bases each request's strings need, summed over the requests (each
/// request is its own circuit).
pub fn batch_size(ctx: &Context, requests: &[Vec<String>]) -> Result<(u64, u64)> {
    let mut strings = 0u64;
    let mut bases = 0u64;
    for labels in requests {
        strings = strings.checked_add(labels.len() as u64).ok_or(Error::Overflow)?;
        let groups = if labels.is_empty() { 0 } else { qwc(ctx, labels, None)?.groups.len() };
        bases = bases.checked_add(groups as u64).ok_or(Error::Overflow)?;
    }
    Ok((strings, bases))
}

/// The run-budget check of `times` submissions of `per = (jobs, shots)` each after `used`:
/// `(after, exceeded)`, `after` the totals once they are queued and `exceeded` every limit they
/// pass as `(name, total, limit)` with the names `"jobs"` and `"total_shots"`. Nothing is
/// recorded; a caller queues them only when `exceeded` is empty.
#[allow(clippy::type_complexity)]
pub fn budget_check(
    used: (u64, u64),
    per: (u64, u64),
    times: u64,
    limits: (Option<u64>, Option<u64>),
) -> Result<((u64, u64), Vec<(&'static str, u64, u64)>)> {
    let add = |base: u64, each: u64| {
        each.checked_mul(times).and_then(|v| v.checked_add(base)).ok_or(Error::Overflow)
    };
    let after = (add(used.0, per.0)?, add(used.1, per.1)?);
    let exceeded = [("jobs", after.0, limits.0), ("total_shots", after.1, limits.1)]
        .into_iter()
        .filter_map(|(name, total, limit)| limit.filter(|&l| total > l).map(|l| (name, total, l)))
        .collect();
    Ok((after, exceeded))
}

/// The standard error an Estimator targets at `shots` per circuit, `1 / sqrt(shots)`, and zero
/// (exact) without shots.
pub fn estimator_precision(shots: u64) -> f64 {
    if shots == 0 { 0. } else { 1. / (shots as f64).sqrt() }
}

/// The memory, in GB, a matrix-product-state simulator holds for one job of `bases` measurement
/// bases: measured 2026-11-09 at about 0.67 MB a basis (a 13,000-basis job of a 50-qubit register
/// took 8.7 GB, against 0.35 GB at 100 bases).
pub fn mps_job_gigabytes(bases: u64) -> f64 {
    0.67e-3 * bases as f64
}

/// `<psi| P |psi>` of every Pauli string on a state of the whole `n`-qubit register (qubit zero
/// the most significant bit of an amplitude's index): the exact values a state-vector simulator
/// returns in place of shots.
pub fn state_expectations(
    ctx: &Context,
    labels: &[String],
    state: &[crate::quantum::C64],
) -> Result<Vec<f64>> {
    use crate::quantum::{Basis, Operator, C64};
    let n = labels_width(labels)?;
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    let basis = Basis::register(ctx, n)?;
    if state.len() != basis.len() {
        return Err(invalid("the state must cover the whole register of the strings"));
    }
    labels
        .iter()
        .map(|label| {
            let operator = Operator::new(ctx, n, &[(label.clone(), C64::new(1., 0.))])?;
            let image = operator.apply(ctx, &basis, state)?;
            Ok(state.iter().zip(image.as_slice()).map(|(a, b)| (a.conj() * b).re).sum())
        })
        .collect()
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
