use super::{amplitude::Amplitude, parallel::parallel_rows, C64};
use crate::{product, Buffer, Context, Error, Reservation, Result};
use std::collections::BTreeMap;

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn phase(n: i32) -> C64 {
    match n.rem_euclid(4) {
        0 => C64::new(1., 0.),
        1 => C64::new(0., 1.),
        2 => C64::new(-1., 0.),
        _ => C64::new(0., -1.),
    }
}
fn bit(n: usize, j: usize) -> u64 {
    1 << (n - 1 - j)
}
fn width(n: usize) -> Result<()> {
    if n > 62 {
        Err(invalid("quantum registers support at most 62 modes"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub x: u64,
    pub z: u64,
    pub coefficient: C64,
}
impl Term {
    /// The Pauli string on `n` qubits, qubit zero first (`I`, `X`, `Y`, `Z`).
    pub fn label(&self, n: usize) -> String {
        (0..n)
            .map(
                |j| match (self.x & bit(n, j) != 0, self.z & bit(n, j) != 0) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                },
            )
            .collect()
    }
    pub(super) fn amplitude(&self, ket: u64) -> C64 {
        self.coefficient
            * phase((self.x & self.z).count_ones() as i32)
            * if (ket & self.z).count_ones() % 2 == 0 {
                1.
            } else {
                -1.
            }
    }
}

/// The rows of one partial sum of a large threaded overlap: fixed, so the sum's bits do not
/// depend on the thread count.
const OVERLAP_BLOCK: usize = 4096;

/// A canonical, deterministic Pauli sum with calculation-owned storage.
#[derive(Debug)]
pub struct Operator {
    pub n: usize,
    pub terms: Vec<Term>,
    /// Optional physical decomposition retained across calculation ownership.
    pub decomposition: Vec<Operator>,
    pub(super) rotation_frequency: Option<f64>,
    _memory: Reservation,
}
impl Operator {
    /// Copy an operator into another calculation's checked allocation budget.
    pub fn copy_into(&self, ctx: &Context) -> Result<Self> {
        let _temporary = ctx.reserve(product(&[self.terms.len(), 128])?)?;
        let mut out = Self::from_terms(ctx, self.n, self.terms.clone())?;
        for group in &self.decomposition {
            out.decomposition.push(group.copy_into(ctx)?);
        }
        Ok(out)
    }
    pub(crate) fn from_terms(ctx: &Context, n: usize, terms: Vec<Term>) -> Result<Self> {
        Self::from_masks(ctx, n, terms)
    }
    pub fn new(ctx: &Context, n: usize, terms: &[(String, C64)]) -> Result<Self> {
        width(n)?;
        let _temporary = ctx.reserve(product(&[terms.len(), 128])?)?;
        let mut masks = Vec::new();
        for (label, c) in terms {
            if label.len() != n || !c.re.is_finite() || !c.im.is_finite() {
                return Err(invalid(
                    "Pauli labels must match the register and coefficients must be finite",
                ));
            }
            let (mut x, mut z) = (0, 0);
            for (j, ch) in label.bytes().enumerate() {
                match ch {
                    b'I' => {}
                    b'X' => x |= bit(n, j),
                    b'Z' => z |= bit(n, j),
                    b'Y' => {
                        x |= bit(n, j);
                        z |= bit(n, j)
                    }
                    _ => return Err(invalid("invalid Pauli letter")),
                }
            }
            masks.push(Term {
                x,
                z,
                coefficient: *c,
            });
        }
        Self::from_masks(ctx, n, masks)
    }
    fn from_masks(ctx: &Context, n: usize, terms: Vec<Term>) -> Result<Self> {
        width(n)?;
        // Includes the temporary ordered map and its nodes, not just final terms.
        let memory = ctx.reserve(product(&[terms.len(), 128])?)?;
        let mut combined = BTreeMap::new();
        for t in terms {
            if !t.coefficient.re.is_finite() || !t.coefficient.im.is_finite() {
                return Err(invalid("nonfinite operator coefficient"));
            }
            if t.x >= (1u64 << n) || t.z >= (1u64 << n) {
                return Err(invalid("Pauli mask exceeds register"));
            }
            let value = combined.entry((t.x, t.z)).or_insert(C64::default());
            *value += t.coefficient;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(invalid("operator coefficient overflow"));
            }
        }
        let terms: Vec<_> = combined
            .into_iter()
            .filter(|(_, c)| c.norm() > 1e-14)
            .map(|((x, z), coefficient)| Term { x, z, coefficient })
            .collect();
        if terms
            .iter()
            .any(|t| !t.coefficient.re.is_finite() || !t.coefficient.im.is_finite())
        {
            return Err(invalid("nonfinite operator coefficient"));
        }
        let rotation_frequency = rotation_frequency(ctx, &terms);
        Ok(Self {
            n,
            terms,
            decomposition: Vec::new(),
            rotation_frequency,
            _memory: memory,
        })
    }
    pub fn labels(&self) -> Vec<(String, C64)> {
        self.terms
            .iter()
            .map(|t| (t.label(self.n), t.coefficient))
            .collect()
    }
    pub fn is_hermitian(&self) -> bool {
        self.terms.iter().all(|t| t.coefficient.im.abs() < 1e-12)
    }
    pub fn is_antihermitian(&self) -> bool {
        self.terms.iter().all(|t| t.coefficient.re.abs() < 1e-12)
    }
    pub fn norm_bound(&self) -> f64 {
        self.terms.iter().map(|t| t.coefficient.norm()).sum()
    }
    pub fn dagger(&self, ctx: &Context) -> Result<Self> {
        let _temporary = ctx.reserve(product(&[self.terms.len(), 32])?)?;
        Self::from_masks(
            ctx,
            self.n,
            self.terms
                .iter()
                .map(|t| Term {
                    coefficient: t.coefficient.conj(),
                    ..*t
                })
                .collect(),
        )
    }
    pub fn add(&self, ctx: &Context, other: &Self, scale: C64) -> Result<Self> {
        if !scale.re.is_finite() || !scale.im.is_finite() {
            return Err(invalid("operator scale must be finite"));
        }
        if self.n != other.n {
            return Err(invalid("operator register mismatch"));
        }
        let _temporary = ctx.reserve(product(&[self.terms.len() + other.terms.len(), 32])?)?;
        Self::from_masks(
            ctx,
            self.n,
            self.terms
                .iter()
                .copied()
                .chain(other.terms.iter().map(|t| Term {
                    coefficient: t.coefficient * scale,
                    ..*t
                }))
                .collect(),
        )
    }
    pub fn compose(&self, ctx: &Context, other: &Self) -> Result<Self> {
        if self.n != other.n {
            return Err(invalid("operator register mismatch"));
        }
        let _temporary = ctx.reserve(product(&[self.terms.len(), other.terms.len(), 32])?)?;
        let mut terms = Vec::new();
        for a in &self.terms {
            for b in &other.terms {
                let (x, z) = (a.x ^ b.x, a.z ^ b.z);
                let exponent = (a.x & a.z).count_ones() as i32 + (b.x & b.z).count_ones() as i32
                    - (x & z).count_ones() as i32
                    + 2 * (a.z & b.x).count_ones() as i32;
                terms.push(Term {
                    x,
                    z,
                    coefficient: a.coefficient * b.coefficient * phase(exponent),
                });
            }
        }
        Self::from_masks(ctx, self.n, terms)
    }
    /// `[self, other]` from the pairs of strings that anticommute, each contributing
    /// `2 sigma_a sigma_b`: the commuting pairs, which cancel in `AB - BA`, are never formed.
    pub fn commutator(&self, ctx: &Context, other: &Self) -> Result<Self> {
        if self.n != other.n {
            return Err(invalid("operator register mismatch"));
        }
        let _temporary = ctx.reserve(product(&[self.terms.len(), other.terms.len(), 32])?)?;
        let mut terms = Vec::new();
        for a in &self.terms {
            for b in &other.terms {
                if ((a.x & b.z).count_ones() + (a.z & b.x).count_ones()) % 2 == 0 {
                    continue;
                }
                let (x, z) = (a.x ^ b.x, a.z ^ b.z);
                let exponent = (a.x & a.z).count_ones() as i32 + (b.x & b.z).count_ones() as i32
                    - (x & z).count_ones() as i32
                    + 2 * (a.z & b.x).count_ones() as i32;
                terms.push(Term {
                    x,
                    z,
                    coefficient: 2. * a.coefficient * b.coefficient * phase(exponent),
                });
            }
        }
        Self::from_masks(ctx, self.n, terms)
    }
    /// `(A + A^dagger) / 2`. Every Pauli string is Hermitian, so this keeps the real part of each
    /// coefficient and drops the imaginary one; no matrix is formed.
    pub fn hermitian_part(&self, ctx: &Context) -> Result<Self> {
        let _temporary = ctx.reserve(product(&[self.terms.len(), 32])?)?;
        Self::from_masks(
            ctx,
            self.n,
            self.terms
                .iter()
                .filter(|t| t.coefficient.re != 0.)
                .map(|t| Term {
                    coefficient: C64::new(t.coefficient.re, 0.),
                    ..*t
                })
                .collect(),
        )
    }
    /// Whether `[self, other]` vanishes as a Pauli sum: an algebraic test, `O(terms^2)`, with no
    /// state vector and no matrix.
    pub fn commutes_with(&self, ctx: &Context, other: &Self) -> Result<bool> {
        Ok(self
            .commutator(ctx, other)?
            .terms
            .iter()
            .all(|t| t.coefficient.norm() <= 1e-11))
    }
    /// `<ket| self |ket>` for the computational basis state `ket` (qubit zero the most
    /// significant bit): only the terms without an `X` or `Y` contribute, each `+-` its
    /// coefficient. `O(terms)`, no state vector.
    pub fn expectation_at(&self, ket: u64) -> C64 {
        self.terms
            .iter()
            .filter(|t| t.x == 0)
            .map(|t| t.amplitude(ket))
            .sum()
    }
    /// Apply without constructing a matrix. Projection is allowed only after
    /// `Basis::check_invariant` has validated a solver's Hamiltonian/generators.
    /// The input is a stored state of either precision; the output row is
    /// float64, so a many-term sum accumulates without float32 rounding.
    pub fn apply_into<A: Amplitude>(&self, basis: &Basis, input: &[A], output: &mut [C64]) {
        output.fill(C64::default());
        Self::apply_terms(&self.terms, basis, input, C64::new(1., 0.), output)
    }
    /// `output += scale * H input`, accumulated in float64.
    pub fn add_into<A: Amplitude>(
        &self,
        basis: &Basis,
        input: &[A],
        scale: C64,
        output: &mut [C64],
    ) {
        Self::apply_terms(&self.terms, basis, input, scale, output)
    }
    pub(super) fn identity(&self) -> C64 {
        self.terms
            .first()
            .filter(|term| term.x == 0 && term.z == 0)
            .map_or(C64::default(), |term| term.coefficient)
    }
    pub(super) fn centered_diagonal(&self, ket: u64) -> C64 {
        self.terms
            .iter()
            .take_while(|t| t.x == 0)
            .filter(|t| t.z != 0)
            .map(|t| t.amplitude(ket))
            .sum()
    }
    fn centered_terms(&self) -> &[Term] {
        if self
            .terms
            .first()
            .is_some_and(|term| term.x == 0 && term.z == 0)
        {
            &self.terms[1..]
        } else {
            &self.terms
        }
    }
    pub(super) fn apply_centered_into<A: Amplitude>(
        &self,
        basis: &Basis,
        input: &[A],
        output: &mut [C64],
    ) {
        output.fill(C64::default());
        Self::apply_terms(
            self.centered_terms(),
            basis,
            input,
            C64::new(1., 0.),
            output,
        )
    }
    /// `output += scale * (H - identity) input`, accumulated in float64.
    pub(super) fn add_centered_into<A: Amplitude>(
        &self,
        basis: &Basis,
        input: &[A],
        scale: C64,
        output: &mut [C64],
    ) {
        Self::apply_terms(self.centered_terms(), basis, input, scale, output)
    }
    /// The single action kernel: adds `scale * (terms input)` into a float64
    /// row. Float64 rows are the stored rows; float32 callers narrow once.
    ///
    /// A small action, or one on a sparse input (fewer than one row in eight
    /// nonzero, as near the reference determinant), scatters each nonzero
    /// column on the calling thread. A large action on a dense input gathers
    /// across the context's workers ([`Self::gather_terms`]). The choice
    /// depends on the data, never on the thread count. (A cost rule that
    /// gathered whenever eight workers would beat the scatter measured slower,
    /// 2026-11-10: a 30-iteration H2O ADAPT growth took 125.8 s on 8 threads
    /// and 329.3 s on 1, against 115.1 s and 162.0 s with this rule.)
    fn apply_terms<A: Amplitude>(
        terms: &[Term],
        basis: &Basis,
        input: &[A],
        scale: C64,
        output: &mut [C64],
    ) {
        let rows = output.len();
        let work = rows.saturating_mul(terms.len());
        if work >= super::parallel::PARALLEL_WORK {
            let nonzero = input.iter().filter(|a| **a != A::default()).count();
            if nonzero.saturating_mul(8) >= rows {
                return Self::gather_terms(terms, basis, input, scale, output);
            }
        }
        Self::scatter_terms(terms, basis, input, scale, output)
    }
    /// The common flip mask of every string, when there is one (an excitation generator: its
    /// strings all flip the same modes).
    pub(super) fn single_mask(&self) -> Option<u64> {
        let first = self.terms.first()?.x;
        (first != 0 && self.terms.iter().all(|t| t.x == first)).then_some(first)
    }
    /// Each string's `(z, coefficient * i^|x & z|)`: `Term::amplitude(ket)` is that value times
    /// the parity sign of `ket & z`, so a sum over strings formed from these is bit-identical to
    /// one formed from `amplitude`, without recomputing the phase for every row.
    fn phased(&self) -> Vec<(u64, C64)> {
        self.terms
            .iter()
            .map(|t| (t.z, t.coefficient * phase((t.x & t.z).count_ones() as i32)))
            .collect()
    }
    /// `sum_t amplitude_t(ket)` from [`Self::phased`].
    fn summed(phased: &[(u64, C64)], ket: u64) -> C64 {
        phased
            .iter()
            .map(|&(z, c)| c * if (ket & z).count_ones() % 2 == 0 { 1. } else { -1. })
            .sum()
    }
    /// The basis state of row `i`.
    fn ket(basis: &Basis, i: usize) -> u64 {
        if basis.full_register {
            i as u64
        } else {
            basis.indices.as_slice()[i]
        }
    }
    /// The new value of row `r` (basis state `ket_r`) after the rotation of [`Self::rotate_pairs`],
    /// from the old values at `r` and at its partner `ket_r ^ x`: the one formula every rotation
    /// kernel (in place, gathered, fused with the adjoint overlap) shares, so all of them give the
    /// same bits. `to_r` and `to_c` are [`Self::summed`] at the partner's and at this row's state.
    #[inline]
    fn rotated<A: Amplitude>(
        to_r: C64,
        to_c: C64,
        linear: f64,
        quadratic: f64,
        old_r: A,
        old_c: A,
    ) -> A {
        let one = C64::new(1., 0.);
        let zero = C64::default();
        let first = |coefficient: C64, source: A| {
            if source == A::default() {
                A::default()
            } else {
                A::narrow(zero + coefficient * (one * source.widen()))
            }
        };
        let second = |coefficient: C64, source: A| {
            if source == A::default() {
                zero
            } else {
                zero + coefficient * (one * source.widen())
            }
        };
        // G sends column c to row r with its strings at ket_c, and r to c at ket_r.
        let (first_r, first_c) = (first(to_r, old_c), first(to_c, old_r));
        A::narrow(old_r.widen() + (linear * first_r.widen() + quadratic * second(to_r, first_c)))
    }
    /// `row += linear G row + quadratic G (G row)` for a generator whose strings share the flip
    /// mask `x` ([`Self::single_mask`]): `G` maps each basis state to at most one partner, so the
    /// new value of a row depends on the old values of that row and its partner `c = r ^ x` alone.
    /// Every value is formed as the two separate actions and the update form it (the same
    /// products, zero skips and roundings into the stored precision), so the row is bit-identical
    /// to them. A large row is gathered into `scratch` in row blocks across the context's threads
    /// and copied back; a small one is updated pair by pair in place. Both give the same bits.
    pub(super) fn rotate_pairs<A: Amplitude>(
        &self,
        basis: &Basis,
        x: u64,
        linear: f64,
        quadratic: f64,
        row: &mut [A],
        scratch: &mut [A],
    ) {
        let phased = self.phased();
        // The new value at `r` from the old values at `r` and its partner `c`.
        let updated = |ket_r: u64, old_r: A, old_c: A| {
            let (to_r, to_c) = (Self::summed(&phased, ket_r ^ x), Self::summed(&phased, ket_r));
            Self::rotated(to_r, to_c, linear, quadratic, old_r, old_c)
        };
        let rows = row.len();
        let work = rows.saturating_mul(self.terms.len()).saturating_mul(2);
        let threads = super::parallel::threads_for(&basis.context, work, rows);
        if threads > 1 && scratch.len() >= rows {
            self.rotate_gather(basis, x, linear, quadratic, row, &mut scratch[..rows]);
            row.copy_from_slice(&scratch[..rows]);
            return;
        }
        for r in 0..rows {
            let ket_r = Self::ket(basis, r);
            let Some(c) = basis.index(ket_r ^ x) else {
                continue;
            };
            if c < r {
                continue;
            }
            let (old_r, old_c) = (row[r], row[c]);
            row[r] = updated(ket_r, old_r, old_c);
            row[c] = updated(ket_r ^ x, old_c, old_r);
        }
    }
    /// Whether the lockstep kernels ([`Self::rotate_gather`], [`Self::adjoint_step`]) run on this
    /// problem: a large action (the same threshold as the threaded overlap, whose block sums
    /// they reproduce) with more than one worker. Otherwise the in-place kernels run; the two
    /// families give the same bits.
    pub(super) fn lockstep(&self, basis: &Basis) -> bool {
        let rows = basis.len();
        let work = rows.saturating_mul(self.terms.len());
        work >= super::parallel::PARALLEL_WORK
            && super::parallel::threads_for(&basis.context, work, rows.div_ceil(OVERLAP_BLOCK)) > 1
    }
    /// The rotation of [`Self::rotate_pairs`] gathered from `source` into `out` in row blocks
    /// across the context's threads: the caller swaps the two buffers instead of copying back.
    pub(super) fn rotate_gather<A: Amplitude>(
        &self,
        basis: &Basis,
        x: u64,
        linear: f64,
        quadratic: f64,
        source: &[A],
        out: &mut [A],
    ) {
        let phased = self.phased();
        let rows = source.len();
        let work = rows.saturating_mul(self.terms.len()).saturating_mul(2);
        parallel_rows(&basis.context, rows, work, &mut out[..rows], |start, block| {
            for (offset, value) in block.iter_mut().enumerate() {
                let r = start + offset;
                let ket_r = Self::ket(basis, r);
                *value = match basis.index(ket_r ^ x) {
                    Some(c) => {
                        let (to_r, to_c) = (Self::summed(&phased, ket_r ^ x), Self::summed(&phased, ket_r));
                        Self::rotated(to_r, to_c, linear, quadratic, source[r], source[c])
                    }
                    None => source[r],
                };
            }
        });
    }
    /// One step of the adjoint sweep in one pass over the rows: `sum_r conj(lambda_r) (G psi)_r`
    /// ([`Self::overlap_single`], the same fixed blocks) and both rows rotated by
    /// [`Self::rotate_pairs`]' rule into `psi_out` and `lambda_out`. Each row's values come from
    /// the old rows alone, and each partner pair's strings are summed once for the three results.
    /// One parallel region replaces the overlap's and the two rotations'.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn adjoint_step<A: Amplitude>(
        &self,
        basis: &Basis,
        x: u64,
        linear: f64,
        quadratic: f64,
        psi: &[A],
        lambda: &[A],
        psi_out: &mut [A],
        lambda_out: &mut [A],
    ) -> C64 {
        let one = C64::new(1., 0.);
        let phased = self.phased();
        let rows = psi.len();
        let mut partial = vec![C64::default(); rows.div_ceil(OVERLAP_BLOCK)];
        super::parallel::parallel_row_pairs(
            &basis.context,
            OVERLAP_BLOCK,
            &mut psi_out[..rows],
            &mut lambda_out[..rows],
            &mut partial,
            |first, psi_block, lambda_block, sum| {
                for (offset, (p, l)) in psi_block.iter_mut().zip(lambda_block.iter_mut()).enumerate() {
                    let r = first + offset;
                    let ket_r = Self::ket(basis, r);
                    let Some(c) = basis.index(ket_r ^ x) else {
                        (*p, *l) = (psi[r], lambda[r]);
                        continue;
                    };
                    let (to_r, to_c) = (Self::summed(&phased, ket_r ^ x), Self::summed(&phased, ket_r));
                    if psi[c] != A::default() {
                        *sum += lambda[r].widen().conj() * (C64::default() + to_r * (one * psi[c].widen()));
                    }
                    *p = Self::rotated(to_r, to_c, linear, quadratic, psi[r], psi[c]);
                    *l = Self::rotated(to_r, to_c, linear, quadratic, lambda[r], lambda[c]);
                }
            },
        );
        partial.into_iter().fold(C64::default(), |sum, block| sum + block)
    }
    /// `sum_r conj(bra_r) (G ket)_r` for a single-mask generator: the action and the overlap in
    /// one pass, without the action's row. A small one folds in row order, equal to forming the
    /// row and folding the overlap; a large one sums fixed blocks of [`OVERLAP_BLOCK`] rows across
    /// the context's threads and adds the block sums in order, so its bits do not depend on the
    /// thread count either.
    pub(super) fn overlap_single<A: Amplitude>(&self, basis: &Basis, x: u64, bra: &[A], ket: &[A]) -> C64 {
        let one = C64::new(1., 0.);
        let phased = self.phased();
        let term = |r: usize, b: &A| {
            let ket_r = Self::ket(basis, r);
            let c = basis.index(ket_r ^ x)?;
            let source = ket[c];
            if source == A::default() {
                return None;
            }
            let to_r = Self::summed(&phased, ket_r ^ x);
            Some(b.widen().conj() * (C64::default() + to_r * (one * source.widen())))
        };
        let rows = bra.len();
        let work = rows.saturating_mul(self.terms.len());
        if work < super::parallel::PARALLEL_WORK {
            let mut sum = C64::default();
            for (r, b) in bra.iter().enumerate() {
                if let Some(value) = term(r, b) {
                    sum += value;
                }
            }
            return sum;
        }
        let blocks = rows.div_ceil(OVERLAP_BLOCK);
        let mut partial = vec![C64::default(); blocks];
        let threads = super::parallel::threads_for(&basis.context, work, blocks);
        let mut lanes = vec![(); threads];
        super::parallel::parallel_tasks(&basis.context, &mut partial, &mut lanes, |first, sums, _| {
            for (k, sum) in sums.iter_mut().enumerate() {
                let start = (first + k) * OVERLAP_BLOCK;
                let end = (start + OVERLAP_BLOCK).min(rows);
                for r in start..end {
                    if let Some(value) = term(r, &bra[r]) {
                        *sum += value;
                    }
                }
            }
        });
        partial.into_iter().fold(C64::default(), |sum, block| sum + block)
    }
    /// Scatter every nonzero column through the strings, on one thread.
    fn scatter_terms<A: Amplitude>(
        terms: &[Term],
        basis: &Basis,
        input: &[A],
        scale: C64,
        output: &mut [C64],
    ) {
        if let Some(first) = terms.first() {
            if terms.iter().all(|term| term.x == first.x) {
                // Excitation strings share their flip mask. Sum their exact
                // Pauli coefficient before multiplying a state amplitude:
                // forbidden occupations cancel without roundoff leakage.
                for (column, (&ket, &amplitude)) in
                    basis.indices.as_slice().iter().zip(input).enumerate()
                {
                    if amplitude == A::default() {
                        continue;
                    }
                    if let Some(row) = basis.index(ket ^ first.x) {
                        let coefficient: C64 = terms.iter().map(|term| term.amplitude(ket)).sum();
                        output[row] += coefficient * (scale * input[column].widen());
                    }
                }
                return;
            }
        }
        if basis.full_register {
            for (ket, &amplitude) in input.iter().enumerate() {
                if amplitude == A::default() {
                    continue;
                }
                let weighted = scale * amplitude.widen();
                for term in terms {
                    output[(ket as u64 ^ term.x) as usize] += term.amplitude(ket as u64) * weighted;
                }
            }
            return;
        }
        for (col, &ket) in basis.indices.as_slice().iter().enumerate() {
            if input[col] == A::default() {
                continue;
            }
            let weighted = scale * input[col].widen();
            for t in terms {
                if let Some(row) = basis.index(ket ^ t.x) {
                    output[row] += t.amplitude(ket) * weighted;
                }
            }
        }
    }
    /// Gather: each output row reads, for every flip mask `x`, the one source
    /// amplitude at `ket ^ x`, with the exact sum of that mask's Pauli
    /// coefficients (forbidden occupations cancel without roundoff leakage).
    /// The terms are stored in `(x, z)` order, so a mask's strings are
    /// contiguous. Rows are independent, so the rows split into contiguous
    /// blocks across the context's threads, and every row's arithmetic is the
    /// same whatever the thread count.
    fn gather_terms<A: Amplitude>(
        terms: &[Term],
        basis: &Basis,
        input: &[A],
        scale: C64,
        output: &mut [C64],
    ) {
        // Each mask's strings as `(z, coefficient * i^|x & z|)`, formed once per action.
        let groups: Vec<(u64, Vec<(u64, C64)>)> = terms
            .chunk_by(|a, b| a.x == b.x)
            .map(|group| {
                let phased = group
                    .iter()
                    .map(|t| (t.z, t.coefficient * phase((t.x & t.z).count_ones() as i32)))
                    .collect();
                (group[0].x, phased)
            })
            .collect();
        let rows = output.len();
        let block = |start: usize, out: &mut [C64]| {
            for (offset, value) in out.iter_mut().enumerate() {
                let ket = if basis.full_register {
                    (start + offset) as u64
                } else {
                    basis.indices.as_slice()[start + offset]
                };
                let mut sum = C64::default();
                for (x, phased) in &groups {
                    let source = ket ^ x;
                    let Some(column) = basis.index(source) else {
                        continue;
                    };
                    let amplitude = input[column];
                    if amplitude == A::default() {
                        continue;
                    }
                    sum += Self::summed(phased, source) * amplitude.widen();
                }
                *value += scale * sum;
            }
        };
        parallel_rows(&basis.context, rows, rows.saturating_mul(terms.len()), output, block);
    }
    /// `finish(k, H inputs[k])` for every input (`centered`: without the identity term), in
    /// order. The applications are independent, so they spread over the context's workers, each
    /// application running serially on one worker into that worker's row, which `finish` may
    /// modify. Every result is the one [`Self::apply_into`] gives on its own, on any thread count:
    /// this is the route for many Hamiltonian actions on sparse states, whose single action is a
    /// serial scatter that no row split can reproduce bit for bit.
    pub fn apply_each<A: Amplitude, T: Send>(
        &self,
        basis: &Basis,
        inputs: &[&[A]],
        centered: bool,
        finish: impl Fn(usize, &mut [C64]) -> T + Sync,
    ) -> Result<Vec<T>> {
        if self.n != basis.n || inputs.iter().any(|input| input.len() != basis.len()) {
            return Err(invalid("states must match the operator basis"));
        }
        let rows = basis.len();
        let work = inputs.len().saturating_mul(rows).saturating_mul(self.terms.len());
        super::parallel::parallel_each(&basis.context, inputs.len(), rows, work, |k, out| {
            if centered {
                self.apply_centered_into(basis, inputs[k], out);
            } else {
                self.apply_into(basis, inputs[k], out);
            }
            finish(k, out)
        })
    }
    pub fn apply(&self, ctx: &Context, basis: &Basis, input: &[C64]) -> Result<Buffer<C64>> {
        if self.n != basis.n
            || input.len() != basis.len()
            || input.iter().any(|c| !c.re.is_finite() || !c.im.is_finite())
        {
            return Err(invalid("state must be finite and match the operator basis"));
        }
        let mut out = ctx.allocate(basis.len())?;
        self.apply_into(basis, input, out.as_mut_slice());
        Ok(out)
    }
    /// The matrix on `basis` in compressed-row storage: `(row offsets, column indices, values)`
    /// with the columns of each row ascending and the cancelled entries dropped. Strings that
    /// share a flip mask `x` send column `c` to the one row `c ^ x`, so each mask contributes at
    /// most one entry per row and no two masks reach the same entry: nothing is summed twice and
    /// the storage is the result plus one row of scratch.
    pub fn csr(&self, ctx: &Context, basis: &Basis) -> Result<Csr> {
        if self.n != basis.n {
            return Err(invalid("operator basis mismatch"));
        }
        let d = basis.len();
        let _scratch = ctx.reserve(product(&[self.terms.len(), 128])?)?;
        let mut flips = BTreeMap::<u64, Vec<(u64, C64)>>::new();
        for t in &self.terms {
            flips.entry(t.x).or_default().push((t.z, t.amplitude(0)));
        }
        let mut capacity = 0usize;
        for &ket in basis.indices.as_slice() {
            for &flip in flips.keys() {
                if basis.index(ket ^ flip).is_some() {
                    capacity = capacity.checked_add(1).ok_or(Error::Overflow)?;
                }
            }
        }
        let mut offsets = ctx.allocate(d.checked_add(1).ok_or(Error::Overflow)?)?;
        let mut columns = ctx.allocate(capacity)?;
        let mut values = ctx.allocate(capacity)?;
        let mut row_entries: Vec<(usize, C64)> = Vec::with_capacity(flips.len());
        let mut cursor = 0;
        for (row, &ket) in basis.indices.as_slice().iter().enumerate() {
            row_entries.clear();
            for (&flip, group) in &flips {
                let source = ket ^ flip;
                if let Some(column) = basis.index(source) {
                    let value: C64 = group
                        .iter()
                        .map(|&(z, coefficient)| {
                            if (z & source).count_ones() % 2 == 0 {
                                coefficient
                            } else {
                                -coefficient
                            }
                        })
                        .sum();
                    if value.norm() > 1e-14 {
                        row_entries.push((column, value));
                    }
                }
            }
            row_entries.sort_unstable_by_key(|&(column, _)| column);
            for &(column, value) in &row_entries {
                columns.as_mut_slice()[cursor] = column;
                values.as_mut_slice()[cursor] = value;
                cursor += 1;
            }
            offsets.as_mut_slice()[row + 1] = cursor;
        }
        Ok(Csr {
            offsets,
            columns,
            values,
            nnz: cursor,
        })
    }
    pub fn dense(&self, ctx: &Context, basis: &Basis) -> Result<Buffer<C64>> {
        if self.n != basis.n {
            return Err(invalid("operator register mismatch"));
        }
        let d = basis.len();
        let mut out = ctx.allocate(product(&[d, d])?)?;
        for (col, &ket) in basis.indices.as_slice().iter().enumerate() {
            for t in &self.terms {
                if let Some(row) = basis.index(ket ^ t.x) {
                    out.as_mut_slice()[row * d + col] += t.amplitude(ket);
                }
            }
        }
        Ok(out)
    }
}

/// A matrix in compressed-row storage; the buffers can be longer than `nnz` entries.
pub struct Csr {
    pub offsets: Buffer<usize>,
    pub columns: Buffer<usize>,
    pub values: Buffer<C64>,
    pub nnz: usize,
}

/// Certify A³ = -omega² A in the complete Pauli algebra before using the
/// exact excitation-rotation polynomial. Other generators keep the general
/// exponential action; no spectral assumption is made from a pool's name.
fn rotation_frequency(ctx: &Context, terms: &[Term]) -> Option<f64> {
    if terms.is_empty() || terms.len() > 16 || terms.iter().any(|t| t.coefficient.re != 0.) {
        return None;
    }
    let _memory = ctx.reserve(terms.len().pow(3) * 128).ok()?;
    let multiply = |a: &Term, b: &Term| {
        let (x, z) = (a.x ^ b.x, a.z ^ b.z);
        let power = (a.x & a.z).count_ones() as i32 + (b.x & b.z).count_ones() as i32
            - (x & z).count_ones() as i32
            + 2 * (a.z & b.x).count_ones() as i32;
        ((x, z), a.coefficient * b.coefficient * phase(power))
    };
    let mut square = BTreeMap::<(u64, u64), C64>::new();
    for a in terms {
        for b in terms {
            let (key, value) = multiply(a, b);
            *square.entry(key).or_default() += value;
        }
    }
    let norm = terms.iter().map(|t| t.coefficient.norm_sqr()).sum::<f64>();
    let frequency_squared = square.values().map(C64::norm_sqr).sum::<f64>() / norm;
    if !frequency_squared.is_finite() || frequency_squared <= 0. {
        return None;
    }
    let mut residual: BTreeMap<_, _> = terms
        .iter()
        .map(|t| ((t.x, t.z), t.coefficient * frequency_squared))
        .collect();
    for ((x, z), coefficient) in square {
        let a = Term { x, z, coefficient };
        for b in terms {
            let (key, value) = multiply(&a, b);
            *residual.entry(key).or_default() += value;
        }
    }
    let error = residual.values().map(C64::norm_sqr).sum::<f64>().sqrt();
    (error <= 1e-13 * frequency_squared * norm.sqrt()).then(|| frequency_squared.sqrt())
}

/// Invertible binary occupation encoding; parity reduction fixes two symmetries.
#[derive(Debug)]
pub struct Encoding {
    pub mapping: String,
    pub modes: usize,
    pub particles: [usize; 2],
    rows: Vec<u64>,
    inverse: Vec<u64>,
}
impl Encoding {
    pub fn new(mapping: &str, modes: usize, particles: [usize; 2]) -> Result<Self> {
        width(modes)?;
        if modes == 0 || modes % 2 != 0 || particles.iter().any(|&p| p > modes / 2) {
            return Err(invalid(
                "spin-blocked encodings need a positive even mode count and valid particle counts",
            ));
        }
        Self::build(mapping, modes, particles)
    }
    /// An encoding of any positive mode count with no particle sector: Jordan-Wigner, parity or
    /// Bravyi-Kitaev as a plain binary change of occupation basis. The reduced parity encoding
    /// fixes two qubits by the particle numbers and is refused here.
    pub fn plain(mapping: &str, modes: usize) -> Result<Self> {
        width(modes)?;
        if modes == 0 {
            return Err(invalid("an encoding needs at least one mode"));
        }
        if mapping == "parity_reduced" {
            return Err(invalid("parity_reduced requires num_particles"));
        }
        Self::build(mapping, modes, [0, 0])
    }
    fn build(mapping: &str, modes: usize, particles: [usize; 2]) -> Result<Self> {
        let mut rows = vec![0; modes];
        for (i, row) in rows.iter_mut().enumerate() {
            match mapping {
                "jordan_wigner" => *row = bit(modes, i),
                "parity" | "parity_reduced" => {
                    for j in 0..=i {
                        *row |= bit(modes, j);
                    }
                }
                "bravyi_kitaev" => {
                    let start = (i + 1) - ((i + 1) & (i + 1).wrapping_neg());
                    for j in start..=i {
                        *row |= bit(modes, j);
                    }
                }
                _ => return Err(invalid("unsupported fermion mapping")),
            }
        }
        let mut left = rows.clone();
        let mut inverse: Vec<_> = (0..modes).map(|i| bit(modes, i)).collect();
        for j in 0..modes {
            let pivot = (j..modes)
                .find(|&i| left[i] & bit(modes, j) != 0)
                .ok_or_else(|| invalid("singular encoding"))?;
            left.swap(j, pivot);
            inverse.swap(j, pivot);
            for i in 0..modes {
                if i != j && left[i] & bit(modes, j) != 0 {
                    left[i] ^= left[j];
                    inverse[i] ^= inverse[j];
                }
            }
        }
        Ok(Self {
            mapping: mapping.into(),
            modes,
            particles,
            rows,
            inverse,
        })
    }
    pub fn n_qubits(&self) -> usize {
        self.modes
            - if self.mapping == "parity_reduced" {
                2
            } else {
                0
            }
    }
    fn transform(&self, bits: u64, rows: &[u64]) -> u64 {
        rows.iter().enumerate().fold(0, |out, (i, row)| {
            out | (((bits & row).count_ones() as u64 % 2) * bit(self.modes, i))
        })
    }
    fn reduce_bits(&self, bits: u64) -> u64 {
        if self.mapping != "parity_reduced" {
            return bits;
        }
        (0..self.modes)
            .filter(|&j| j != self.modes / 2 - 1 && j != self.modes - 1)
            .fold(0, |out, j| {
                (out << 1) | u64::from(bits & bit(self.modes, j) != 0)
            })
    }
    pub fn encode(&self, occupations: u64) -> u64 {
        self.reduce_bits(self.transform(occupations, &self.rows))
    }
    pub fn decode(&self, encoded: u64) -> u64 {
        let mut full = 0;
        let mut cursor = self.n_qubits();
        for j in 0..self.modes {
            let occupied = if self.mapping == "parity_reduced" && j == self.modes / 2 - 1 {
                self.particles[0] % 2 != 0
            } else if self.mapping == "parity_reduced" && j == self.modes - 1 {
                (self.particles[0] + self.particles[1]) % 2 != 0
            } else {
                cursor -= 1;
                encoded & (1 << cursor) != 0
            };
            if occupied {
                full |= bit(self.modes, j);
            }
        }
        self.transform(full, &self.inverse)
    }
    /// The alpha and beta particle-number operators on this encoding's register. A Hamiltonian
    /// or a pool generator that commutes with both preserves every particle sector, which is the
    /// algebraic form of `Basis::check_invariant` and needs no basis.
    pub fn number_operators(&self, ctx: &Context) -> Result<[Operator; 2]> {
        let m = self.modes / 2;
        let spin = |modes: std::ops::Range<usize>| -> Result<Operator> {
            let mut total = Operator::from_masks(ctx, self.n_qubits(), vec![])?;
            for j in modes {
                total = total.add(
                    ctx,
                    &self.monomial(ctx, &[(j, true), (j, false)])?,
                    C64::new(1., 0.),
                )?;
            }
            Ok(total)
        };
        Ok([spin(0..m)?, spin(m..2 * m)?])
    }
    pub fn reference(&self) -> u64 {
        let m = self.modes / 2;
        self.encode(
            (0..self.particles[0])
                .chain(m..m + self.particles[1])
                .fold(0, |out, j| out | bit(self.modes, j)),
        )
    }
    pub fn map_operator(&self, ctx: &Context, op: &Operator) -> Result<Operator> {
        if op.n != self.modes {
            return Err(invalid("encoding mode mismatch"));
        }
        let _temporary = ctx.reserve(product(&[op.terms.len(), 32])?)?;
        let mut terms = Vec::new();
        for t in &op.terms {
            let x = self.transform(t.x, &self.rows);
            let mut z = 0;
            for j in 0..self.modes {
                if t.z & bit(self.modes, j) != 0 {
                    z ^= self.inverse[j];
                }
            }
            let mut c = t.coefficient
                * phase((t.x & t.z).count_ones() as i32 - (x & z).count_ones() as i32);
            if self.mapping == "parity_reduced" {
                for (j, p) in [
                    (self.modes / 2 - 1, self.particles[0]),
                    (self.modes - 1, self.particles[0] + self.particles[1]),
                ] {
                    if x & bit(self.modes, j) != 0 {
                        return Err(invalid("operator changes a tapered particle parity"));
                    }
                    if z & bit(self.modes, j) != 0 && p % 2 != 0 {
                        c = -c;
                    }
                }
            }
            terms.push(Term {
                x: self.reduce_bits(x),
                z: self.reduce_bits(z),
                coefficient: c,
            });
        }
        Operator::from_masks(ctx, self.n_qubits(), terms)
    }
    /// `sum_t c_t prod a^(dagger)` of the fermionic terms, each product multiplied in the
    /// Jordan-Wigner algebra and mapped once, merged in one pass. A reduced encoding maps each
    /// product on its own, so every term has to preserve both tapered parities.
    pub fn map_fermion(
        &self,
        ctx: &Context,
        terms: &[(Vec<(usize, bool)>, C64)],
    ) -> Result<Operator> {
        let _temporary = ctx.reserve(product(&[terms.len(), 640])?)?;
        let mut mapped = Vec::new();
        for (ladders, coefficient) in terms {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(invalid("nonfinite fermionic coefficient"));
            }
            mapped.extend(self.monomial(ctx, ladders)?.terms.iter().map(|t| Term {
                coefficient: t.coefficient * coefficient,
                ..*t
            }));
        }
        Operator::from_masks(ctx, self.n_qubits(), mapped)
    }
    pub fn ladder(&self, ctx: &Context, j: usize, dagger: bool) -> Result<Operator> {
        if j >= self.modes {
            return Err(invalid("fermionic mode exceeds register"));
        }
        // JW a†=(X-iY)/2 with a Z on each preceding mode.
        let x = bit(self.modes, j);
        let z = (0..j).fold(0, |v, k| v | bit(self.modes, k));
        let jw = Operator::from_masks(
            ctx,
            self.modes,
            vec![
                Term {
                    x,
                    z,
                    coefficient: C64::new(0.5, 0.),
                },
                Term {
                    x,
                    z: z | x,
                    coefficient: C64::new(0., if dagger { -0.5 } else { 0.5 }),
                },
            ],
        )?;
        self.map_operator(ctx, &jw)
    }
    /// Ordered ladder product, mapped only after multiplication so a reduced
    /// encoding can represent number-conserving products of untaperable ladders.
    pub fn monomial(&self, ctx: &Context, ladders: &[(usize, bool)]) -> Result<Operator> {
        let jw = Self::build("jordan_wigner", self.modes, self.particles)?;
        let mut op = Operator::from_masks(
            ctx,
            self.modes,
            vec![Term {
                x: 0,
                z: 0,
                coefficient: C64::new(1., 0.),
            }],
        )?;
        for &(j, dagger) in ladders {
            op = op.compose(ctx, &jw.ladder(ctx, j, dagger)?)?;
        }
        self.map_operator(ctx, &op)
    }
}

#[derive(Debug)]
pub struct Basis {
    context: Context,
    pub n: usize,
    pub indices: Buffer<u64>,
    pub reference: usize,
    full_register: bool,
    lookup: Option<Buffer<u32>>,
}
pub(crate) fn choose(n: usize, k: usize) -> Result<usize> {
    let k = k.min(n - k);
    let mut count = 1u128;
    for i in 0..k {
        count = count.checked_mul((n - i) as u128).ok_or(Error::Overflow)? / (i + 1) as u128;
    }
    usize::try_from(count).map_err(|_| Error::Overflow)
}
fn combinations(n: usize, k: usize) -> Vec<u64> {
    fn visit(n: usize, k: usize, start: usize, bits: u64, out: &mut Vec<u64>) {
        if k == 0 {
            out.push(bits);
            return;
        }
        for j in start..=n - k {
            visit(n, k - 1, j + 1, bits | bit(n, j), out);
        }
    }
    let mut out = Vec::new();
    visit(n, k, 0, 0, &mut out);
    out
}
impl Basis {
    /// Embed a compact sector state into the user-visible qubit register.
    pub fn expand_state<A: Amplitude>(&self, ctx: &Context, state: &[A]) -> Result<Buffer<A>> {
        if state.len() != self.len() || !super::amplitude::finite(state) {
            return Err(invalid(
                "state must be finite and match the simulation basis",
            ));
        }
        let size = 1usize.checked_shl(self.n as u32).ok_or(Error::Overflow)?;
        let mut out = ctx.allocate::<A>(size)?;
        for (&index, &value) in self.indices.as_slice().iter().zip(state) {
            out.as_mut_slice()[index as usize] = value;
        }
        Ok(out)
    }
    /// A register state may enter a sector calculation only without leakage.
    pub fn compress_state<A: Amplitude>(&self, ctx: &Context, state: &[A]) -> Result<Buffer<A>> {
        let size = 1usize.checked_shl(self.n as u32).ok_or(Error::Overflow)?;
        if state.len() != size || !super::amplitude::finite(state) {
            return Err(invalid("state must be finite and match the qubit register"));
        }
        let mut out = ctx.allocate::<A>(self.len())?;
        let mut position = 0;
        let mut leakage = 0.;
        for (i, &value) in state.iter().enumerate() {
            if self.indices.as_slice().get(position) == Some(&(i as u64)) {
                out.as_mut_slice()[position] = value;
                position += 1;
            } else {
                leakage += value.widen().norm_sqr();
            }
        }
        let norm = super::amplitude::norm_squared(state);
        if !norm.is_finite() || leakage > 1e-12 * norm.max(1.) {
            return Err(invalid("state leaves the configured particle sector"));
        }
        Ok(out)
    }
    pub fn register(ctx: &Context, n: usize) -> Result<Self> {
        Self::register_at(ctx, n, 0)
    }
    /// The whole `n`-qubit register with the computational state `reference` (occupation bits,
    /// qubit zero the most significant) as the state an ansatz starts from.
    pub fn register_at(ctx: &Context, n: usize, reference: u64) -> Result<Self> {
        width(n)?;
        let d = 1usize.checked_shl(n as u32).ok_or(Error::Overflow)?;
        if reference >= d as u64 {
            return Err(invalid("the reference determinant exceeds the register"));
        }
        let mut indices = ctx.allocate(d)?;
        for (k, v) in indices.as_mut_slice().iter_mut().enumerate() {
            *v = k as u64;
        }
        Ok(Self {
            context: ctx.clone(),
            n,
            indices,
            reference: reference as usize,
            full_register: true,
            lookup: None,
        })
    }
    pub fn new(ctx: &Context, encoding: &Encoding, sector: bool) -> Result<Self> {
        let n = encoding.n_qubits();
        let m = encoding.modes / 2;
        let d = if sector {
            product(&[
                choose(m, encoding.particles[0])?,
                choose(m, encoding.particles[1])?,
            ])?
        } else {
            1usize.checked_shl(n as u32).ok_or(Error::Overflow)?
        };
        let mut indices = ctx.allocate(d)?;
        if sector {
            let _temporary = ctx.reserve(product(&[
                choose(m, encoding.particles[0])?
                    .checked_add(choose(m, encoding.particles[1])?)
                    .ok_or(Error::Overflow)?,
                16,
            ])?)?;
            let alpha = combinations(m, encoding.particles[0]);
            let beta = combinations(m, encoding.particles[1]);
            let mut k = 0;
            for a in alpha {
                for &b in &beta {
                    indices.as_mut_slice()[k] = encoding.encode((a << m) | b);
                    k += 1;
                }
            }
            indices.as_mut_slice().sort_unstable();
        } else {
            for (k, v) in indices.as_mut_slice().iter_mut().enumerate() {
                *v = k as u64;
            }
        }
        let reference = indices
            .as_slice()
            .binary_search(&encoding.reference())
            .map_err(|_| invalid("reference absent from basis"))?;
        let register = 1usize.checked_shl(n as u32).unwrap_or(usize::MAX);
        let lookup = if sector
            && d >= 256
            && register <= d.saturating_mul(32)
            && register <= (ctx.memory_limit() / 32).saturating_div(4)
            && register <= (ctx.memory_limit() - ctx.memory_used()).saturating_div(16)
            && d < u32::MAX as usize
        {
            let mut lookup = ctx.allocate::<u32>(register)?;
            lookup.as_mut_slice().fill(u32::MAX);
            for (row, &ket) in indices.as_slice().iter().enumerate() {
                lookup.as_mut_slice()[ket as usize] = row as u32;
            }
            Some(lookup)
        } else {
            None
        };
        Ok(Self {
            context: ctx.clone(),
            n,
            indices,
            reference,
            full_register: !sector,
            lookup,
        })
    }
    #[inline]
    pub fn index(&self, ket: u64) -> Option<usize> {
        if self.full_register {
            return (ket < self.len() as u64).then_some(ket as usize);
        }
        if let Some(lookup) = &self.lookup {
            return lookup
                .as_slice()
                .get(usize::try_from(ket).ok()?)
                .copied()
                .filter(|&row| row != u32::MAX)
                .map(|row| row as usize);
        }
        self.indices.as_slice().binary_search(&ket).ok()
    }
    pub fn len(&self) -> usize {
        self.indices.as_slice().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn check_invariant(&self, op: &Operator) -> Result<()> {
        if op.n != self.n {
            return Err(invalid("operator basis mismatch"));
        }
        if self.full_register {
            return Ok(());
        }
        // Terms with the same flip can cancel outside the sector. Test their
        // combined amplitude, never each Pauli string separately.
        let _memory = self.context.reserve(product(&[op.terms.len(), 128])?)?;
        let mut flips = std::collections::BTreeMap::<u64, Vec<&Term>>::new();
        for term in &op.terms {
            flips.entry(term.x).or_default().push(term);
        }
        for &ket in self.indices.as_slice() {
            for (&flip, terms) in &flips {
                if self.index(ket ^ flip).is_none() {
                    let amplitude: C64 = terms.iter().map(|term| term.amplitude(ket)).sum();
                    if amplitude.norm() > 1e-11 {
                        return Err(invalid(
                            "operator does not preserve the requested particle sector",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Generator {
    pub label: String,
    pub operator: Operator,
}
pub fn excitation_pool(
    ctx: &Context,
    encoding: &Encoding,
    singles: bool,
) -> Result<Vec<Generator>> {
    let m = encoding.modes / 2;
    let [na, nb] = encoding.particles;
    let occ: Vec<_> = (0..na).chain(m..m + nb).collect();
    let virt: Vec<_> = (na..m).chain(m + nb..2 * m).collect();
    let mut pool = Vec::new();
    let mut append = |label: String, ladders: &[(usize, bool)]| -> Result<()> {
        let t = encoding.monomial(ctx, ladders)?;
        let operator = t.add(ctx, &t.dagger(ctx)?, C64::new(-1., 0.))?;
        pool.push(Generator { label, operator });
        Ok(())
    };
    if singles {
        for &i in &occ {
            for &a in &virt {
                if i / m == a / m {
                    append(format!("S({i}->{a})"), &[(a, true), (i, false)])?;
                }
            }
        }
    }
    for (ix, &i) in occ.iter().enumerate() {
        for &j in &occ[ix + 1..] {
            for (ax, &a) in virt.iter().enumerate() {
                for &b in &virt[ax + 1..] {
                    if i / m + j / m == a / m + b / m {
                        append(
                            format!("D({i},{j}->{a},{b})"),
                            &[(a, true), (b, true), (j, false), (i, false)],
                        )?;
                    }
                }
            }
        }
    }
    Ok(pool)
}
