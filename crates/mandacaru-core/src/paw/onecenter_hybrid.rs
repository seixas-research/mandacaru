//! The one-center terms of a screened hybrid and of a Hartree-Fock Hamiltonian,
//! evaluated in projector space on the one-center tensors of
//! [`super::onecenter`] and the frozen one-body pieces of
//! [`super::onecenter_terms`].
use super::invalid;
use crate::{math::linalg::real_eigen, product, Buffer, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Eigenvalues of a sphere's pair-pair matrix below this fraction of the largest are
/// dropped from its factors (round-off: a real basis has the antisymmetric pairs as
/// an exact null space).
pub const ONE_CENTER_RANK_FLOOR: f64 = 1e-10;

/// The channel block of each `(l, m, index)` projector pair with equal `(l, m)`:
/// `out[i, j] = block[l][index_i, index_j]`.
pub fn expand_blocks(
    layout: &[(usize, i32, usize)],
    blocks: &[(usize, &DMatrix<f64>)],
) -> Result<DMatrix<f64>> {
    let n = layout.len();
    let mut out = DMatrix::zeros(n, n);
    for (i, a) in layout.iter().enumerate() {
        for (j, b) in layout.iter().enumerate() {
            if (a.0, a.1) == (b.0, b.1) {
                let block = blocks
                    .iter()
                    .find(|(l, _)| *l == a.0)
                    .ok_or_else(|| invalid(format!("no block for the channel l = {}", a.0)))?
                    .1;
                if a.2 >= block.nrows() || b.2 >= block.ncols() {
                    return Err(invalid("a projector index exceeds its channel block"));
                }
                out[(i, j)] = block[(a.2, b.2)];
            }
        }
    }
    Ok(out)
}

/// One sphere of a screened hybrid: its columns of the projections, the exchange
/// tensor `Delta W^SR[i, j, k, l]` (`n^4`) and the frozen exchange potential `V`.
pub struct HybridAtom {
    pub members: Vec<usize>,
    pub tensor: Vec<Complex64>,
    pub v: DMatrix<f64>,
}

/// The terms of the hybrid at one projected density matrix.
pub struct Terms {
    pub energy: f64,
    pub exact_exchange: f64,
    pub semilocal: f64,
    /// Row-major operators of `dim x dim` each: one per spin channel evaluated. Each
    /// buffer keeps its bytes charged to the [`Context`] until it is dropped or moved
    /// out with [`Buffer::into_parts`].
    pub operators: Vec<Buffer<Complex64>>,
    pub dim: usize,
}

pub struct Hybrid {
    atoms: Vec<HybridAtom>,
    projectors: usize,
    fraction: f64,
    constant: f64,
}

/// `Delta K_il = sum_jk D_jk W[i, j, k, l]` for the block of `members` of `density`,
/// written into `out` (`n x n`); `block` (`n x n`) is the gathered density block.
fn exchange(
    atom: &HybridAtom,
    density: &[Complex64],
    stride: usize,
    block: &mut [Complex64],
    out: &mut [Complex64],
) {
    let n = atom.members.len();
    for (jk, slot) in block.iter_mut().enumerate() {
        *slot = density[atom.members[jk / n] * stride + atom.members[jk % n]];
    }
    out.fill(Complex64::new(0., 0.));
    for i in 0..n {
        for (jk, d) in block.iter().enumerate() {
            if d.norm() == 0. {
                continue;
            }
            for l in 0..n {
                out[i * n + l] += atom.tensor[(i * n * n + jk) * n + l] * *d;
            }
        }
    }
}

impl Hybrid {
    /// `frozen` is the sum over spheres of the frozen constants `X0 - V.D0`; the
    /// per-species constant of the hybrid is `-fraction * frozen`.
    pub fn new(
        atoms: Vec<HybridAtom>,
        projectors: usize,
        fraction: f64,
        frozen: f64,
    ) -> Result<Self> {
        for atom in &atoms {
            let n = atom.members.len();
            if atom.tensor.len() != n * n * n * n
                || atom.v.nrows() != n
                || atom.v.ncols() != n
                || atom.members.iter().any(|m| *m >= projectors)
            {
                return Err(invalid(
                    "a hybrid sphere's arrays do not match its projectors",
                ));
            }
        }
        Ok(Self {
            atoms,
            projectors,
            fraction,
            constant: -fraction * frozen,
        })
    }
    pub fn constant(&self) -> f64 {
        self.constant
    }
    pub fn projectors(&self) -> usize {
        self.projectors
    }

    /// Complex elements of the scratch a call needs: the gathered density block and
    /// the exchange block of the largest sphere.
    fn scratch(&self) -> Result<usize> {
        let n = self
            .atoms
            .iter()
            .map(|a| a.members.len())
            .max()
            .unwrap_or(0);
        product(&[2, n, n])
    }

    /// One density (closed shell, both spins) or two (spin channels), each `P x P`.
    ///
    /// Memory, reserved before it is allocated: the operators (`spins P^2` complex
    /// numbers, which the result keeps) and the scratch of the exchange blocks
    /// (`2 n_max^2`, released on return), 16 bytes each.
    pub fn evaluate(&self, ctx: &Context, spins: &[&[Complex64]]) -> Result<Terms> {
        let p = self.projectors;
        if spins.is_empty() || spins.len() > 2 || spins.iter().any(|d| d.len() != p * p) {
            return Err(invalid("one or two P x P density matrices are needed"));
        }
        let a = self.fraction;
        let scale = if spins.len() == 1 { 0.25 } else { 0.5 };
        let mut operators = (0..spins.len())
            .map(|_| ctx.allocate::<Complex64>(product(&[p, p])?))
            .collect::<Result<Vec<_>>>()?;
        let mut scratch = ctx.allocate::<Complex64>(self.scratch()?)?;
        let (mut exact, mut linear) = (0., 0.);
        for atom in &self.atoms {
            let n = atom.members.len();
            for i in 0..n {
                for j in 0..n {
                    let total = spins
                        .iter()
                        .map(|d| d[atom.members[j] * p + atom.members[i]])
                        .sum::<Complex64>();
                    linear += atom.v[(i, j)] * total.re;
                }
            }
            for (s, density) in spins.iter().enumerate() {
                let (block, k) = scratch.as_mut_slice().split_at_mut(n * n);
                let k = &mut k[..n * n];
                exchange(atom, density, p, block, k);
                for i in 0..n {
                    for j in 0..n {
                        exact -= scale
                            * a
                            * (density[atom.members[j] * p + atom.members[i]] * k[i * n + j]).re;
                        operators[s].as_mut_slice()[atom.members[i] * p + atom.members[j]] +=
                            k[i * n + j] * (-2. * scale * a)
                                - Complex64::new(a * atom.v[(i, j)], 0.);
                    }
                }
            }
        }
        let semilocal = self.constant - a * linear;
        Ok(Terms {
            energy: exact + semilocal,
            exact_exchange: exact,
            semilocal,
            operators,
            dim: p,
        })
    }

    /// The four spin blocks `[[uu, ud], [du, dd]]` of a two-component density
    /// matrix; one `2P x 2P` operator, alpha block first. Memory as for
    /// [`Self::evaluate`], with one operator of `4 P^2` complex numbers.
    pub fn evaluate_spinor(&self, ctx: &Context, blocks: [[&[Complex64]; 2]; 2]) -> Result<Terms> {
        let p = self.projectors;
        if blocks.iter().flatten().any(|d| d.len() != p * p) {
            return Err(invalid("the spinor blocks must each be P x P"));
        }
        let a = self.fraction;
        let dim = 2 * p;
        let mut operator = ctx.allocate::<Complex64>(product(&[dim, dim])?)?;
        let mut scratch = ctx.allocate::<Complex64>(self.scratch()?)?;
        let (mut exact, mut linear) = (0., 0.);
        for atom in &self.atoms {
            let n = atom.members.len();
            for i in 0..n {
                for j in 0..n {
                    let index = atom.members[j] * p + atom.members[i];
                    linear += atom.v[(i, j)] * (blocks[0][0][index] + blocks[1][1][index]).re;
                }
            }
            for s in 0..2 {
                for t in 0..2 {
                    let (block, k) = scratch.as_mut_slice().split_at_mut(n * n);
                    let k = &mut k[..n * n];
                    exchange(atom, blocks[s][t], p, block, k);
                    for i in 0..n {
                        for j in 0..n {
                            exact -= 0.5
                                * a
                                * (blocks[t][s][atom.members[j] * p + atom.members[i]]
                                    * k[i * n + j])
                                    .re;
                            let row = s * p + atom.members[i];
                            let col = t * p + atom.members[j];
                            let slot = &mut operator.as_mut_slice()[row * dim + col];
                            *slot += k[i * n + j] * (-a);
                            if s == t {
                                *slot -= Complex64::new(a * atom.v[(i, j)], 0.);
                            }
                        }
                    }
                }
            }
        }
        let semilocal = self.constant - a * linear;
        Ok(Terms {
            energy: exact + semilocal,
            exact_exchange: exact,
            semilocal,
            operators: vec![operator],
            dim,
        })
    }
}

/// The exact one-center terms of one sphere of a molecular Hamiltonian.
pub struct ExactAtom {
    pub v: DMatrix<f64>,
    pub k: DMatrix<f64>,
    pub d0: DMatrix<f64>,
    /// `-V - Delta W . D0 + K`: the one-body block added to the nonlocal coupling.
    pub coupling: DMatrix<f64>,
    /// Weights of the low-rank factors of `Delta W[(ab), (cd)] = sum_k lambda_k U^k_ab U^k_cd`.
    pub values: Vec<f64>,
    pub factors: Vec<DMatrix<f64>>,
}

/// `tensor` is the real `n^4` exchange tensor with `omega = 0`; `v`, `k`, `d0` the
/// layout-expanded one-body pieces.
pub fn exact_atom(
    n: usize,
    tensor: &[f64],
    v: DMatrix<f64>,
    k: DMatrix<f64>,
    d0: DMatrix<f64>,
) -> Result<ExactAtom> {
    if tensor.len() != n * n * n * n {
        return Err(invalid(
            "the one-center tensor does not match the projectors",
        ));
    }
    let mut hartree = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            hartree[(i, j)] = (0..n * n)
                .map(|kl| tensor[(i * n + j) * n * n + kl] * d0[(kl / n, kl % n)])
                .sum();
        }
    }
    let hartree = (&hartree + hartree.transpose()) * 0.5;
    let coupling = -&v - hartree + &k;
    let pair = DMatrix::from_fn(n * n, n * n, |ab, cd| {
        0.5 * (tensor[ab * n * n + cd] + tensor[cd * n * n + ab])
    });
    let (values, vectors) = real_eigen(pair)?;
    let largest = values.iter().fold(0_f64, |m, v| m.max(v.abs())).max(1e-300);
    let floor = ONE_CENTER_RANK_FLOOR * largest;
    let (mut kept, mut factors) = (Vec::new(), Vec::new());
    for (index, value) in values.iter().enumerate() {
        if value.abs() > floor {
            kept.push(*value);
            factors.push(DMatrix::from_fn(n, n, |a, b| {
                0.5 * (vectors[(a * n + b, index)] + vectors[(b * n + a, index)])
            }));
        }
    }
    Ok(ExactAtom {
        v,
        k,
        d0,
        coupling,
        values: kept,
        factors,
    })
}

/// A sphere's projection columns and low-rank factors.
pub struct Factors<'a> {
    pub members: &'a [usize],
    pub factors: &'a [DMatrix<f64>],
}

/// The stacked channels, real or complex; the storage stays charged to the
/// [`Context`] while the buffer lives.
pub enum ChannelData {
    Real(Buffer<f64>),
    Complex(Buffer<Complex64>),
}

/// `count` square `rows x rows` channels, stacked in one buffer
/// (`data[(k * rows + r) * rows + s]`).
pub struct ChannelStack {
    pub count: usize,
    pub rows: usize,
    pub data: ChannelData,
}

impl ChannelStack {
    fn new(ctx: &Context, real: bool, count: usize, rows: usize) -> Result<Self> {
        let len = product(&[count, rows, rows])?;
        let data = if real {
            ChannelData::Real(ctx.allocate::<f64>(len)?)
        } else {
            ChannelData::Complex(ctx.allocate::<Complex64>(len)?)
        };
        Ok(Self { count, rows, data })
    }

    /// Store `square` (`rows x rows`) as channel `k`; a real stack keeps the real part.
    fn put(&mut self, k: usize, square: &[Complex64]) {
        let size = self.rows * self.rows;
        match &mut self.data {
            ChannelData::Real(b) => {
                for (slot, v) in b.as_mut_slice()[k * size..(k + 1) * size]
                    .iter_mut()
                    .zip(square)
                {
                    *slot = v.re;
                }
            }
            ChannelData::Complex(b) => {
                b.as_mut_slice()[k * size..(k + 1) * size].copy_from_slice(square);
            }
        }
    }
}

/// Rows `x` columns `members` of `c` (`rows x width`, row major), into `out`.
fn columns(c: &[Complex64], rows: usize, width: usize, members: &[usize], out: &mut [Complex64]) {
    for row in 0..rows {
        for (k, m) in members.iter().enumerate() {
            out[row * members.len() + k] = c[row * width + m];
        }
    }
}

/// `X U Y^T` for `X` (`rows x n`) and `Y` (`rows x n`), `U` real `n x n`, into `out`
/// (`rows x rows`); `left` (`rows x n`) is scratch.
fn sandwich(
    x: &[Complex64],
    y: &[Complex64],
    u: &DMatrix<f64>,
    rows: usize,
    left: &mut [Complex64],
    out: &mut [Complex64],
) {
    let n = u.nrows();
    for r in 0..rows {
        for b in 0..n {
            left[r * n + b] = (0..n).map(|a| x[r * n + a] * u[(a, b)]).sum();
        }
    }
    for r in 0..rows {
        for s in 0..rows {
            out[r * rows + s] = (0..n).map(|b| left[r * n + b] * y[s * n + b]).sum();
        }
    }
}

/// The shapes of the projections and factors, checked before any index is taken.
fn check_channels(c: &[Complex64], rows: usize, width: usize, atoms: &[Factors<'_>]) -> Result<()> {
    if c.len() != product(&[rows, width])? {
        return Err(invalid("the projections must be rows x width"));
    }
    for atom in atoms {
        let n = atom.members.len();
        if atom.members.iter().any(|m| *m >= width) {
            return Err(invalid("a member column exceeds the projections"));
        }
        if atom
            .factors
            .iter()
            .any(|u| u.nrows() != n || u.ncols() != n)
        {
            return Err(invalid(
                "a factor must be n x n for the n members of its sphere",
            ));
        }
    }
    Ok(())
}

/// Complex elements of the scratch for `parts` gathered column blocks (`rows x n_max`),
/// one `left` block and the `rows x rows` square.
fn channel_scratch(rows: usize, atoms: &[Factors<'_>], parts: usize) -> Result<usize> {
    let n = atoms.iter().map(|a| a.members.len()).max().unwrap_or(0);
    product(&[rows, n, parts + 1])?
        .checked_add(product(&[rows, rows])?)
        .ok_or(crate::Error::Overflow)
}

/// `C_A U^k C_A^T` for every factor of every sphere (`C` is `M x P`, row major),
/// stacked in one buffer.
///
/// Memory, reserved before it is allocated: the stack (`K M^2` numbers, 8 or 16 bytes,
/// which the result keeps) and the scratch (`M (2 n_max + M)` complex numbers, released
/// on return).
pub fn exact_channels(
    ctx: &Context,
    c: &[Complex64],
    rows: usize,
    width: usize,
    atoms: &[Factors<'_>],
    real: bool,
) -> Result<ChannelStack> {
    check_channels(c, rows, width, atoms)?;
    let count = atoms.iter().map(|a| a.factors.len()).sum();
    let mut out = ChannelStack::new(ctx, real, count, rows)?;
    let mut work = ctx.allocate::<Complex64>(channel_scratch(rows, atoms, 1)?)?;
    let mut k = 0;
    for atom in atoms {
        let n = atom.members.len();
        let (ca, rest) = work.as_mut_slice().split_at_mut(rows * n);
        let (left, rest) = rest.split_at_mut(rows * n);
        let square = &mut rest[..rows * rows];
        columns(c, rows, width, atom.members, ca);
        for u in atom.factors {
            sandwich(ca, ca, u, rows, left, square);
            out.put(k, square);
            k += 1;
        }
    }
    Ok(out)
}

/// `dC_A U^k C_A^T + C_A U^k dC_A^T` for every factor (zeros where `dC` does not
/// touch the sphere). Memory as for [`exact_channels`], with one more gathered block
/// in the scratch.
pub fn exact_channel_derivatives(
    ctx: &Context,
    c: &[Complex64],
    dc: &[Complex64],
    rows: usize,
    width: usize,
    atoms: &[Factors<'_>],
    real: bool,
) -> Result<ChannelStack> {
    check_channels(c, rows, width, atoms)?;
    if dc.len() != c.len() {
        return Err(invalid("dC must have the shape of C"));
    }
    let count = atoms.iter().map(|a| a.factors.len()).sum();
    let mut out = ChannelStack::new(ctx, real, count, rows)?;
    let mut work = ctx.allocate::<Complex64>(channel_scratch(rows, atoms, 2)?)?;
    let mut k = 0;
    for atom in atoms {
        let n = atom.members.len();
        let (ca, rest) = work.as_mut_slice().split_at_mut(rows * n);
        let (dca, rest) = rest.split_at_mut(rows * n);
        let (left, rest) = rest.split_at_mut(rows * n);
        let square = &mut rest[..rows * rows];
        columns(c, rows, width, atom.members, ca);
        columns(dc, rows, width, atom.members, dca);
        let moving = dca.iter().any(|v| v.norm() != 0.);
        for u in atom.factors {
            if moving {
                sandwich(dca, ca, u, rows, left, square);
                // square + square^T, in place.
                for r in 0..rows {
                    for s in r..rows {
                        let sum = square[r * rows + s] + square[s * rows + r];
                        square[r * rows + s] = sum;
                        square[s * rows + r] = sum;
                    }
                }
                out.put(k, square);
            }
            k += 1;
        }
    }
    Ok(out)
}
