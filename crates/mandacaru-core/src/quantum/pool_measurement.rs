//! The Pauli strings a device is asked to read, and the pool's commutators `[H, A_k]` formed in
//! bounded batches.
//!
//! A measured ADAPT-VQE step reads `<[H, A_k]>` for every generator off one circuit. The
//! commutators of a 50-qubit pool hold millions of terms, but a step needs only their distinct
//! Pauli strings (to ask the device once) and, afterwards, the sum `sum_t c_t <P_t>` of each
//! commutator. Neither needs more than one batch of commutators alive, so the pool is walked
//! twice, forming each commutator again instead of keeping it: memory is the distinct strings
//! (16 bytes each) plus one batch, whatever the pool's size.
use super::{algebra::Term, CachedProblem, Operator, C64};
use crate::{Context, Reservation, Result};

/// Accounted bytes of one commutator batch, unless the context's memory limit is smaller.
const BATCH_BYTES: usize = 8 << 20;
/// What an [`Operator`] reserves from the context per term it was built from.
const TERM_BYTES: usize = 128;
/// What a label string and the vector that holds it cost while a request is in flight.
const LABEL_BYTES: usize = 128;
/// A distinct Pauli string, as its `x` and `z` masks.
const KEY_BYTES: usize = 16;

/// The accounted size of one commutator batch in `ctx`: a fixed 8 MiB, and never more than an
/// eighth of the context's memory limit (a batch holds at least one commutator).
pub fn batch_bytes(ctx: &Context) -> usize {
    BATCH_BYTES.min(ctx.memory_limit() / 8).max(TERM_BYTES)
}

/// The distinct non-identity Pauli strings of some operators, sorted by their masks. The
/// request to a device lists them in this order, and [`PauliSet::position`] finds a term's
/// answer in it.
pub struct PauliSet {
    n: usize,
    keys: Vec<(u64, u64)>,
    _memory: Reservation,
}
impl PauliSet {
    pub fn new(ctx: &Context, n: usize) -> Result<Self> {
        Ok(Self {
            n,
            keys: Vec::new(),
            _memory: ctx.reserve(0)?,
        })
    }
    /// Add the strings of `operators`; the set stays sorted and free of repeats, and its
    /// accounted size (the strings, plus the room to merge a batch in) is reserved from `ctx`.
    pub fn insert<'a>(
        &mut self,
        ctx: &Context,
        operators: impl IntoIterator<Item = &'a Operator>,
    ) -> Result<()> {
        let operators: Vec<_> = operators.into_iter().collect();
        let incoming: usize = operators.iter().map(|o| o.terms.len()).sum();
        let _merge = ctx.reserve((self.keys.len() + incoming) * KEY_BYTES)?;
        for operator in operators {
            self.keys.extend(
                operator
                    .terms
                    .iter()
                    .filter(|t| (t.x, t.z) != (0, 0))
                    .map(|t| (t.x, t.z)),
            );
        }
        self.keys.sort_unstable();
        self.keys.dedup();
        self.keys.shrink_to_fit();
        self._memory = ctx.reserve(self.keys.len() * KEY_BYTES)?;
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// The strings, in the set's order.
    pub fn labels(&self) -> Vec<String> {
        self.keys
            .iter()
            .map(|&(x, z)| {
                Term {
                    x,
                    z,
                    coefficient: C64::default(),
                }
                .label(self.n)
            })
            .collect()
    }
    /// Where a term's string sits in [`PauliSet::labels`]; `None` for the identity and for a
    /// string the set does not hold.
    pub fn position(&self, term: &Term) -> Option<usize> {
        self.keys.binary_search(&(term.x, term.z)).ok()
    }
    /// The accounted bytes a request carrying these strings holds while it is in flight.
    pub fn request_bytes(&self) -> usize {
        self.keys.len() * LABEL_BYTES
    }
}

/// What walking the pool once learns: the distinct strings of every `[H, A_k]`, and how many
/// of the commutators are not zero.
pub struct PoolPaulis {
    pub set: PauliSet,
    pub measured: usize,
}

impl CachedProblem {
    /// Form each `[H, A_k]` in pool order and hand them to `visit` in batches whose accounted
    /// size is about `bound` bytes (a batch holds at least one commutator). A batch is dropped
    /// before the next is formed.
    pub fn for_each_pool_commutator_batch(
        &self,
        bound: usize,
        mut visit: impl FnMut(&[Operator]) -> Result<()>,
    ) -> Result<()> {
        let mut batch: Vec<Operator> = Vec::new();
        let mut held = 0;
        for generator in &self.pool {
            let operator = self
                .hamiltonian
                .commutator(&self.context, &generator.operator)?;
            held += operator.terms.len().max(1) * TERM_BYTES;
            batch.push(operator);
            if held >= bound {
                visit(&batch)?;
                batch.clear();
                held = 0;
            }
        }
        if !batch.is_empty() {
            visit(&batch)?;
        }
        Ok(())
    }
    /// The distinct Pauli strings of every pool commutator, in batches of about `bound` bytes.
    pub fn pool_paulis(&self, bound: usize) -> Result<PoolPaulis> {
        let mut set = PauliSet::new(&self.context, self.hamiltonian.n)?;
        let mut measured = 0;
        self.for_each_pool_commutator_batch(bound, |batch| {
            measured += batch.iter().filter(|o| !o.terms.is_empty()).count();
            set.insert(&self.context, batch)
        })?;
        Ok(PoolPaulis { set, measured })
    }
}
