//! Z2 symmetry tapering: one qubit removed per conserved parity of a qubit Hamiltonian.
//!
//! The parity mapping's two-qubit reduction removes the two particle-number parities, known in
//! advance. A molecular Hamiltonian usually commutes with more Z2 operators (the spatial symmetries
//! of the molecule), and each is a qubit that carries no information. They are read off the
//! Hamiltonian itself, so nothing about the point group is supplied: a symmetry the orbitals break
//! is not found, and one the user did not think of is.
//!
//! * The Z-type strings commuting with every term are the kernel, over GF(2), of the matrix of the
//!   terms' X supports: a Z-type `tau` commutes with a term exactly when `x . tau = 0 (mod 2)`.
//! * The generators are brought to reduced row echelon form, so each has a pivot qubit no other
//!   generator acts on (its anchor `q_j`). Then `U_j = (tau_j + X_{q_j}) / sqrt 2` is a Hermitian
//!   Clifford that sends `tau_j` to `X_{q_j}` and commutes with every other `tau_i`.
//! * After conjugating by every `U_j`, the Hamiltonian commutes with each `X_{q_j}`. Each anchor is
//!   replaced by its eigenvalue, the sign of `tau_j` on the reference determinant (the sector that
//!   holds the reference), and deleted.
//!
//! Tapering an operator that does not commute with the symmetries would replace it by its
//! projection, and `exp(PAP) != P exp(A) P`: an ansatz built from projected generators explores
//! the wrong manifold. [`Taper::operator`] therefore refuses a leaking operator;
//! [`Taper::observable`] projects one, which is exact for an expectation value in the sector.
use super::{
    algebra::{Basis, Operator, Term},
    amplitude::Amplitude,
    C64,
};
use crate::{Buffer, Context, Error, Result};

/// A tapered coefficient below this is treated as cancelled.
pub const TAPER_TOLERANCE: f64 = 1e-12;

fn bit(n: usize, j: usize) -> u64 {
    1 << (n - 1 - j)
}

/// The Z2 symmetries of one Hamiltonian and the sector of its reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Taper {
    /// The register width before tapering.
    pub n: usize,
    /// The Z-type generators found (masks in the register's bit order), the kernel basis.
    pub found: Vec<u64>,
    /// Their reduced row echelon form, one per removed qubit.
    pub generators: Vec<u64>,
    /// The anchor qubit of each reduced generator (its pivot).
    pub anchors: Vec<usize>,
    /// The eigenvalue of each reduced generator on the reference: `+1` or `-1`.
    pub signs: Vec<i8>,
}

/// Gauss-Jordan elimination over GF(2) of `rows` (masks over `n` qubits, column `j` the qubit
/// `j`): the reduced rows with their pivot columns, zero rows dropped.
fn reduced_echelon(n: usize, mut rows: Vec<u64>) -> (Vec<u64>, Vec<usize>) {
    let mut pivots = Vec::new();
    let mut row = 0;
    for col in 0..n {
        if row == rows.len() {
            break;
        }
        let Some(pick) = (row..rows.len()).find(|&r| rows[r] & bit(n, col) != 0) else {
            continue;
        };
        rows.swap(row, pick);
        for other in 0..rows.len() {
            if other != row && rows[other] & bit(n, col) != 0 {
                rows[other] ^= rows[row];
            }
        }
        pivots.push(col);
        row += 1;
    }
    rows.truncate(pivots.len());
    (rows, pivots)
}

/// A basis of the GF(2) null space of `rows` (each a mask over `n` columns), one vector per free
/// column in increasing order.
fn kernel(n: usize, rows: Vec<u64>) -> Vec<u64> {
    let (reduced, pivots) = reduced_echelon(n, rows);
    (0..n)
        .filter(|c| !pivots.contains(c))
        .map(|free| {
            let mut v = bit(n, free);
            for (r, &p) in pivots.iter().enumerate() {
                if reduced[r] & bit(n, free) != 0 {
                    v |= bit(n, p);
                }
            }
            v
        })
        .collect()
}

impl Taper {
    /// The symmetries of `hamiltonian` and the sector of the reference determinant `reference`
    /// (its occupation bits in the encoding the Hamiltonian is written in), or `None` when there is
    /// no Z2 symmetry beyond the identity.
    pub fn find(hamiltonian: &Operator, reference: u64) -> Option<Self> {
        let n = hamiltonian.n;
        let found: Vec<u64> = kernel(n, hamiltonian.terms.iter().map(|t| t.x).collect())
            .into_iter()
            .filter(|&v| v != 0)
            .collect();
        if found.is_empty() {
            return None;
        }
        let (generators, anchors) = reduced_echelon(n, found.clone());
        let signs = generators
            .iter()
            .map(|&tau| if (reference & tau).count_ones() % 2 == 0 { 1 } else { -1 })
            .collect();
        Some(Self { n, found, generators, anchors, signs })
    }

    /// The register width after tapering.
    pub fn width(&self) -> usize {
        self.n - self.anchors.len()
    }

    /// Whether the term with X support `x` fails to commute with a generator.
    pub fn leaks(&self, x: u64) -> bool {
        self.generators.iter().any(|&tau| (x & tau).count_ones() % 2 == 1)
    }

    /// `operator` reduced to the tapered register. Refuses an operator with a term that does not
    /// commute with the symmetries, which tapering would silently replace by its projection.
    pub fn operator(&self, ctx: &Context, operator: &Operator) -> Result<Operator> {
        if let Some(term) = operator.terms.iter().find(|t| self.leaks(t.x)) {
            let label = Operator::from_terms(ctx, operator.n, vec![*term])?.labels().remove(0).0;
            return Err(Error::Invalid(format!(
                "a term of this operator ({label}) does not commute with the Z2 symmetries: \
                 tapering would replace it by its projection, and an ansatz built from projected \
                 generators explores the wrong manifold"
            )));
        }
        self.reduce(ctx, operator)
    }

    /// An observable projected into the sector (its symmetry-changing terms have zero
    /// expectation there) and then tapered.
    pub fn observable(&self, ctx: &Context, operator: &Operator) -> Result<Operator> {
        let kept: Vec<Term> = operator.terms.iter().filter(|t| !self.leaks(t.x)).copied().collect();
        self.reduce(ctx, &Operator::from_terms(ctx, operator.n, kept)?)
    }

    /// The occupation bits of a determinant on the tapered register: the anchors deleted. Exact for
    /// a determinant of the sector (the Clifford leaves the other qubits as they are and puts each
    /// anchor in an X eigenstate).
    pub fn bits(&self, bits: u64) -> u64 {
        self.compress(bits)
    }

    fn compress(&self, mask: u64) -> u64 {
        let width = self.width();
        let mut out = 0;
        let mut at = 0;
        for j in 0..self.n {
            if self.anchors.contains(&j) {
                continue;
            }
            if mask & bit(self.n, j) != 0 {
                out |= bit(width, at);
            }
            at += 1;
        }
        out
    }

    /// `U_j = (tau_j + X_{q_j}) / sqrt 2`, the Hermitian Clifford sending the `j`-th generator to
    /// the `X` on its anchor.
    fn clifford(&self, ctx: &Context, j: usize) -> Result<Operator> {
        let root = std::f64::consts::FRAC_1_SQRT_2;
        Operator::from_terms(
            ctx,
            self.n,
            vec![
                Term { x: 0, z: self.generators[j], coefficient: C64::new(root, 0.) },
                Term { x: bit(self.n, self.anchors[j]), z: 0, coefficient: C64::new(root, 0.) },
            ],
        )
    }

    /// The generators as `I`/`Z` labels over the untapered register, one per removed qubit.
    pub fn labels(&self) -> Vec<String> {
        self.generators
            .iter()
            .map(|&tau| (0..self.n).map(|j| if tau & bit(self.n, j) != 0 { 'Z' } else { 'I' }).collect())
            .collect()
    }

    /// Whether the determinant `bits` (occupation bits in the untapered encoding) lies in the
    /// symmetry sector of the reference: every generator has the sign the reference fixed.
    /// Determinants of other sectors alias onto the same tapered register state.
    pub fn in_sector(&self, bits: u64) -> bool {
        self.generators
            .iter()
            .zip(&self.signs)
            .all(|(&tau, &sign)| ((bits & tau).count_ones() % 2 == 0) == (sign == 1))
    }

    /// The untapered-register state a tapered `state` stands for (qubit zero the most significant
    /// bit): each anchor is put in the `X` eigenstate of its sector sign and the Cliffords `U_j`
    /// are applied in reverse order, so every expectation value of a symmetric operator on the
    /// result equals the tapered one. The result has `2^n` amplitudes (`n` the untapered width),
    /// charged to the `Context` before it is allocated: this is the state-vector route, and the
    /// allocation is what a register too wide for one refuses.
    pub fn lift<A: Amplitude>(&self, ctx: &Context, state: &[A]) -> Result<Buffer<C64>> {
        let width = self.width();
        let tapered = 1usize.checked_shl(width as u32).ok_or(Error::Overflow)?;
        if state.len() != tapered {
            return Err(Error::Invalid(format!(
                "a tapered state has {tapered} amplitudes, got {}",
                state.len()
            )));
        }
        let full_size = 1usize.checked_shl(self.n as u32).ok_or(Error::Overflow)?;
        let mut full = ctx.allocate::<C64>(full_size)?;
        let kept: Vec<usize> = (0..self.n).filter(|j| !self.anchors.contains(j)).collect();
        let root = std::f64::consts::FRAC_1_SQRT_2.powi(self.anchors.len() as i32);
        for (t, amplitude) in state.iter().enumerate() {
            let amplitude = amplitude.widen();
            if amplitude == C64::default() {
                continue;
            }
            let base = kept
                .iter()
                .enumerate()
                .filter(|&(position, _)| t as u64 & bit(width, position) != 0)
                .fold(0u64, |out, (_, &q)| out | bit(self.n, q));
            for pattern in 0..1usize << self.anchors.len() {
                let (mut index, mut factor) = (base, root);
                for (i, (&q, &sign)) in self.anchors.iter().zip(&self.signs).enumerate() {
                    if pattern >> i & 1 == 1 {
                        index |= bit(self.n, q);
                        factor *= f64::from(sign);
                    }
                }
                full.as_mut_slice()[index as usize] = amplitude * factor;
            }
        }
        let basis = Basis::register(ctx, self.n)?;
        for j in (0..self.anchors.len()).rev() {
            full = self.clifford(ctx, j)?.apply(ctx, &basis, full.as_slice())?;
        }
        Ok(full)
    }

    fn reduce(&self, ctx: &Context, operator: &Operator) -> Result<Operator> {
        if operator.n != self.n {
            return Err(Error::Invalid("the operator is not on the tapered Hamiltonian's register".into()));
        }
        let n = self.n;
        let mut rotated = operator.copy_into(ctx)?;
        for j in 0..self.anchors.len() {
            let u = self.clifford(ctx, j)?;
            rotated = u.compose(ctx, &rotated)?.compose(ctx, &u)?;
        }
        let mut terms = Vec::with_capacity(rotated.terms.len());
        for t in rotated.terms.iter().filter(|t| t.coefficient.norm() > TAPER_TOLERANCE) {
            let mut coefficient = t.coefficient;
            for (&q, &sign) in self.anchors.iter().zip(&self.signs) {
                let (x, z) = (t.x & bit(n, q) != 0, t.z & bit(n, q) != 0);
                match (x, z) {
                    (false, false) => {}
                    (true, false) => coefficient *= f64::from(sign),
                    _ => {
                        return Err(Error::Invalid(format!(
                            "qubit {q} carries Z or Y after the rotation: the operator does not \
                             commute with X_{q} and cannot be tapered on it"
                        )))
                    }
                }
            }
            terms.push(Term { x: self.compress(t.x), z: self.compress(t.z), coefficient });
        }
        let out = Operator::from_terms(ctx, self.width(), terms)?;
        let kept: Vec<Term> = out.terms.iter().filter(|t| t.coefficient.norm() > TAPER_TOLERANCE).copied().collect();
        Operator::from_terms(ctx, self.width(), kept)
    }
}
