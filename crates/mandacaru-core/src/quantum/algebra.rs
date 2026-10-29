use super::C64;
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
            .map(|t| {
                (
                    (0..self.n)
                        .map(
                            |j| match (t.x & bit(self.n, j) != 0, t.z & bit(self.n, j) != 0) {
                                (false, false) => 'I',
                                (true, false) => 'X',
                                (false, true) => 'Z',
                                (true, true) => 'Y',
                            },
                        )
                        .collect(),
                    t.coefficient,
                )
            })
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
    pub fn commutator(&self, ctx: &Context, other: &Self) -> Result<Self> {
        self.compose(ctx, other)?
            .add(ctx, &other.compose(ctx, self)?, C64::new(-1., 0.))
    }
    /// Apply without constructing a matrix. Projection is allowed only after
    /// `Basis::check_invariant` has validated a solver's Hamiltonian/generators.
    pub fn apply_into(&self, basis: &Basis, input: &[C64], output: &mut [C64]) {
        Self::apply_terms(&self.terms, basis, input, output)
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
    pub(super) fn apply_centered_into(&self, basis: &Basis, input: &[C64], output: &mut [C64]) {
        let terms = if self
            .terms
            .first()
            .is_some_and(|term| term.x == 0 && term.z == 0)
        {
            &self.terms[1..]
        } else {
            &self.terms
        };
        Self::apply_terms(terms, basis, input, output)
    }
    fn apply_terms(terms: &[Term], basis: &Basis, input: &[C64], output: &mut [C64]) {
        output.fill(C64::default());
        if let Some(first) = terms.first() {
            if terms.iter().all(|term| term.x == first.x) {
                // Excitation strings share their flip mask. Sum their exact
                // Pauli coefficient before multiplying a state amplitude:
                // forbidden occupations cancel without roundoff leakage.
                for (column, (&ket, &amplitude)) in
                    basis.indices.as_slice().iter().zip(input).enumerate()
                {
                    if amplitude == C64::default() {
                        continue;
                    }
                    if let Some(row) = basis.index(ket ^ first.x) {
                        let coefficient: C64 = terms.iter().map(|term| term.amplitude(ket)).sum();
                        output[row] += coefficient * input[column];
                    }
                }
                return;
            }
        }
        if basis.full_register {
            for (ket, &amplitude) in input.iter().enumerate() {
                if amplitude == C64::default() {
                    continue;
                }
                for term in terms {
                    output[(ket as u64 ^ term.x) as usize] +=
                        term.amplitude(ket as u64) * amplitude;
                }
            }
            return;
        }
        for (col, &ket) in basis.indices.as_slice().iter().enumerate() {
            if input[col] == C64::default() {
                continue;
            }
            for t in terms {
                if let Some(row) = basis.index(ket ^ t.x) {
                    output[row] += t.amplitude(ket) * input[col];
                }
            }
        }
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
    pub(super) fn apply_centered(
        &self,
        ctx: &Context,
        basis: &Basis,
        input: &[C64],
    ) -> Result<Buffer<C64>> {
        if self.n != basis.n
            || input.len() != basis.len()
            || input.iter().any(|c| !c.re.is_finite() || !c.im.is_finite())
        {
            return Err(invalid("state must be finite and match the operator basis"));
        }
        let mut out = ctx.allocate(basis.len())?;
        self.apply_centered_into(basis, input, out.as_mut_slice());
        Ok(out)
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
        let jw = Self::new("jordan_wigner", self.modes, self.particles)?;
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
    pub fn expand_state(&self, ctx: &Context, state: &[C64]) -> Result<Buffer<C64>> {
        if state.len() != self.len() || state.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(invalid(
                "state must be finite and match the simulation basis",
            ));
        }
        let size = 1usize.checked_shl(self.n as u32).ok_or(Error::Overflow)?;
        let mut out = ctx.allocate::<C64>(size)?;
        for (&index, &value) in self.indices.as_slice().iter().zip(state) {
            out.as_mut_slice()[index as usize] = value;
        }
        Ok(out)
    }
    /// A register state may enter a sector calculation only without leakage.
    pub fn compress_state(&self, ctx: &Context, state: &[C64]) -> Result<Buffer<C64>> {
        let size = 1usize.checked_shl(self.n as u32).ok_or(Error::Overflow)?;
        if state.len() != size || state.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(invalid("state must be finite and match the qubit register"));
        }
        let mut out = ctx.allocate::<C64>(self.len())?;
        let mut position = 0;
        let mut leakage = 0.;
        for (i, &value) in state.iter().enumerate() {
            if self.indices.as_slice().get(position) == Some(&(i as u64)) {
                out.as_mut_slice()[position] = value;
                position += 1;
            } else {
                leakage += value.norm_sqr();
            }
        }
        let norm = state.iter().map(|z| z.norm_sqr()).sum::<f64>();
        if !norm.is_finite() || leakage > 1e-12 * norm.max(1.) {
            return Err(invalid("state leaves the configured particle sector"));
        }
        Ok(out)
    }
    pub fn register(ctx: &Context, n: usize) -> Result<Self> {
        width(n)?;
        let d = 1usize.checked_shl(n as u32).ok_or(Error::Overflow)?;
        let mut indices = ctx.allocate(d)?;
        for (k, v) in indices.as_mut_slice().iter_mut().enumerate() {
            *v = k as u64;
        }
        Ok(Self {
            context: ctx.clone(),
            n,
            indices,
            reference: 0,
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
