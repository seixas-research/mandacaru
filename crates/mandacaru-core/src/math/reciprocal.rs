//! Symmetry-averaged reciprocal aliases of a discrete periodic grid.
use super::{fft::frequency, invalid, SpatialGrid};
use crate::{finite, Buffer, Context, Reservation, Result};
use nalgebra::{Matrix3, Vector3};
use std::{
    collections::HashSet,
    f64::consts::PI,
    ops::{Add, Mul, Range, Sub},
};

/// `x` (FFT index units) moved by whole periods `n` into `[-n/2, n/2)`; a face
/// within 1e-9 reads as its lower side, so a zone-boundary image is read alike.
fn boxed(x: f64, n: f64) -> f64 {
    x - n * ((x + 0.5 * n + 1e-9) / n).floor()
}

fn key(w: &Matrix3<f64>) -> [i64; 9] {
    std::array::from_fn(|k| w[(k / 3, k % 3)].round() as i64)
}
fn allowed(
    w: &Matrix3<f64>,
    cell: &Matrix3<f64>,
    inverse: &Matrix3<f64>,
    shape: [usize; 3],
) -> bool {
    if (w.determinant().abs() - 1.).abs() > 1e-9 {
        return false;
    }
    let r = cell * w * inverse;
    if (r.transpose() * r - Matrix3::identity()).norm() > 2e-2 {
        return false;
    }
    let v = w.try_inverse().unwrap().transpose();
    for i in 0..3 {
        for j in 0..3 {
            let nodes = w[(i, j)] * shape[i] as f64 / shape[j] as f64;
            let aliases = v[(i, j)] * shape[j] as f64 / shape[i] as f64;
            if (nodes - nodes.round()).abs() > 1e-9 || (aliases - aliases.round()).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}
struct Lattice {
    reciprocal: Matrix3<f64>,
    group: Vec<Matrix3<f64>>,
    operations: Vec<Matrix3<f64>>,
    _storage: Reservation,
}
fn lattice(ctx: &Context, grid: &SpatialGrid) -> Result<Lattice> {
    let shape = grid.index.shape();
    let cell = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    let inverse = cell
        .try_inverse()
        .ok_or_else(|| invalid("singular reciprocal cell"))?;
    let reciprocal = inverse.transpose() * (2. * PI);
    // Cubic grids have at most 48 operations; a generous bound also covers
    // mildly strained/skew cells while preventing an unbounded closure.
    let _storage = ctx.reserve(256 * 512)?;
    let mut group = Vec::new();
    let mut known = HashSet::new();
    for mut code in 0..3usize.pow(9) {
        let mut entries = [0.; 9];
        for value in &mut entries {
            *value = (code % 3) as f64 - 1.;
            code /= 3;
        }
        let w = Matrix3::from_row_slice(&entries);
        if allowed(&w, &cell, &inverse, shape) {
            known.insert(key(&w));
            group.push(w);
        }
    }
    loop {
        let n = group.len();
        for i in 0..n {
            for j in 0..n {
                let w = group[i] * group[j];
                if allowed(&w, &cell, &inverse, shape) && known.insert(key(&w)) {
                    if group.len() == 256 {
                        return Err(invalid(
                            "reciprocal symmetry closure exceeds 256 operations",
                        ));
                    }
                    group.push(w);
                }
            }
        }
        if group.len() == n {
            break;
        }
    }
    let operations: Vec<_> = group
        .iter()
        .map(|w| w.try_inverse().unwrap().transpose())
        .collect();
    Ok(Lattice {
        reciprocal,
        group,
        operations,
        _storage,
    })
}
pub fn norms(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    let shape = grid.index.shape();
    let Lattice {
        reciprocal,
        group,
        operations,
        _storage,
    } = lattice(ctx, grid)?;
    let signed_permutations = operations
        .iter()
        .all(|v| (0..3).all(|i| (0..3).map(|j| v[(i, j)].abs()).sum::<f64>() == 1.));
    let mut out = ctx.allocate::<f64>(grid.index.points())?;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let indices = [i, j, k];
                let m = Vector3::from_fn(|a, _| frequency(indices[a], shape[a]));
                let norm = if signed_permutations {
                    (reciprocal * m).norm()
                } else {
                    let mut sum = 0.;
                    for (w, v) in group.iter().zip(&operations) {
                        let image = v * m;
                        let wrapped = Vector3::from_fn(|a, _| {
                            let n = shape[a] as f64;
                            (image[a] + (shape[a] / 2) as f64).rem_euclid(n) - (shape[a] / 2) as f64
                        });
                        // v = w^-T, so v^-1 = w^T, exactly for integer operations.
                        sum += (reciprocal * (w.transpose() * wrapped)).norm();
                    }
                    sum / group.len() as f64
                };
                out.as_mut_slice()[(i * shape[1] + j) * shape[2] + k] = norm;
            }
        }
    }
    Ok(out)
}

/// Alias-averaged Bloch derivative symbols and kinetic-energy multipliers.
/// Cartesian k is in inverse Bohr. The mean square is retained separately
/// from the square of the mean vector, including on Nyquist faces.
pub struct BlochSymbols {
    pub vectors: Buffer<f64>,
    pub squares: Buffer<f64>,
}
pub fn bloch(ctx: &Context, grid: &SpatialGrid, k: [f64; 3]) -> Result<BlochSymbols> {
    finite(&k, "Bloch wave vector")?;
    let shape = grid.index.shape();
    let n = grid.index.points();
    let lattice = lattice(ctx, grid)?;
    let kappa = lattice
        .reciprocal
        .try_inverse()
        .ok_or_else(|| invalid("singular reciprocal cell"))?
        * Vector3::from(k);
    let mut vectors = ctx.allocate::<f64>(crate::product(&[3, n])?)?;
    let mut squares = ctx.allocate::<f64>(n)?;
    for at in 0..n {
        let index = [
            at / (shape[1] * shape[2]),
            at / shape[2] % shape[1],
            at % shape[2],
        ];
        let q = Vector3::from_fn(|a, _| frequency(index[a], shape[a]) + kappa[a]);
        let mut mean = Vector3::zeros();
        let mut mean_square = 0.;
        for (w, v) in lattice.group.iter().zip(&lattice.operations) {
            let image = v * q;
            let wrapped = Vector3::from_fn(|a, _| boxed(image[a], shape[a] as f64));
            let wave = lattice.reciprocal * (w.transpose() * wrapped);
            mean += wave;
            mean_square += wave.norm_squared();
        }
        mean /= lattice.operations.len() as f64;
        squares.as_mut_slice()[at] = mean_square / lattice.operations.len() as f64;
        for a in 0..3 {
            vectors.as_mut_slice()[a * n + at] = mean[a];
        }
    }
    Ok(BlochSymbols { vectors, squares })
}

/// Reciprocal vectors of the grid's cell as the columns of a matrix (Bohr^-1).
pub fn reciprocal_cell(grid: &SpatialGrid) -> Result<Matrix3<f64>> {
    let shape = grid.index.shape();
    let cell = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
    Ok(cell
        .try_inverse()
        .ok_or_else(|| invalid("singular reciprocal cell"))?
        .transpose()
        * (2. * PI))
}

/// The FFT-box wave vector `B m` of the flat grid index `at` (C order).
pub fn box_vector(reciprocal: &Matrix3<f64>, shape: [usize; 3], at: usize) -> [f64; 3] {
    let index = [
        at / (shape[1] * shape[2]),
        at / shape[2] % shape[1],
        at % shape[2],
    ];
    (reciprocal * Vector3::from_fn(|a, _| frequency(index[a], shape[a]))).into()
}

/// The frequencies whose box wave vector an operation of the grid's lattice
/// does not carry onto the box vector of the image, with their other aliases.
/// Entries are sorted by `index`; each is one distinct (frequency, alias) pair
/// and `weights` is the fraction of the operations that pick it.
#[derive(Debug)]
pub struct AliasTable {
    pub index: Vec<usize>,
    pub vectors: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    _storage: Reservation,
}
impl AliasTable {
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
    /// Consecutive entry ranges of at most `most` entries, each ending where one
    /// frequency's entries end (a longer run stands alone), so `combine` can
    /// work one range at a time.
    pub fn blocks(&self, most: usize) -> Vec<Range<usize>> {
        let (mut out, mut start) = (Vec::new(), 0);
        while start < self.len() {
            let mut end = (start + most.max(1)).min(self.len());
            while end < self.len() && self.index[end] == self.index[end - 1] {
                end += 1;
            }
            out.push(start..end);
            start = end;
        }
        out
    }
    /// Mean over the aliases: `row[i] += sum w (f(alias) - f(box_i))` for the
    /// entries `range` (a range of `blocks`), `alias_values[k]` being the
    /// function at `vectors[range.start + k]`; `row` holds the box values.
    pub fn combine<T>(&self, row: &mut [T], range: Range<usize>, alias_values: &[T])
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f64, Output = T>,
    {
        let (mut last, mut base) = (usize::MAX, None);
        for (k, entry) in range.enumerate() {
            let at = self.index[entry];
            if at != last {
                last = at;
                base = Some(row[at]);
            }
            let base = base.expect("set on the first entry");
            row[at] = row[at] + (alias_values[k] - base) * self.weights[entry];
        }
    }
}

pub fn alias_table(ctx: &Context, grid: &SpatialGrid) -> Result<AliasTable> {
    let shape = grid.index.shape();
    let lattice = lattice(ctx, grid)?;
    let count = lattice.group.len() as f64;
    let moved = |at: usize, visit: &mut dyn FnMut(usize, [i64; 3])| {
        let index = [
            at / (shape[1] * shape[2]),
            at / shape[2] % shape[1],
            at % shape[2],
        ];
        let m = Vector3::from_fn(|a, _| frequency(index[a], shape[a]));
        for (w, v) in lattice.group.iter().zip(&lattice.operations) {
            let image = v * m;
            let wrap = Vector3::from_fn(|a, _| boxed(image[a], shape[a] as f64) - image[a]);
            if wrap.iter().any(|x| *x != 0.) {
                let alias = m + w.transpose() * wrap;
                visit(at, [0, 1, 2].map(|a| alias[a].round() as i64));
            }
        }
    };
    let points = grid.index.points();
    let mut raw = 0usize;
    for at in 0..points {
        moved(at, &mut |_, _| raw += 1);
    }
    let _scratch = ctx.reserve(crate::product(&[raw, 32])?)?;
    let mut entries: Vec<(usize, [i64; 3])> = Vec::new();
    entries
        .try_reserve_exact(raw)
        .map_err(|_| crate::Error::Allocation)?;
    for at in 0..points {
        moved(at, &mut |i, m| entries.push((i, m)));
    }
    entries.sort_unstable();
    let mut runs: Vec<(usize, [i64; 3], usize)> = Vec::new();
    for entry in entries {
        match runs.last_mut() {
            Some(last) if (last.0, last.1) == entry => last.2 += 1,
            _ => runs.push((entry.0, entry.1, 1)),
        }
    }
    let storage = ctx.reserve(crate::product(&[runs.len(), 40])?)?;
    let reciprocal = lattice.reciprocal;
    Ok(AliasTable {
        index: runs.iter().map(|r| r.0).collect(),
        vectors: runs
            .iter()
            .map(|r| (reciprocal * Vector3::from_fn(|a, _| r.1[a] as f64)).into())
            .collect(),
        weights: runs.iter().map(|r| r.2 as f64 / count).collect(),
        _storage: storage,
    })
}

/// A function of the wave vector on the grid's reciprocal set (C order), each
/// frequency's value the mean over the aliases the lattice operations pick.
/// A frequency every operation keeps inside the box is the plain `f(B m)`.
pub fn alias_mean<T>(
    ctx: &Context,
    grid: &SpatialGrid,
    function: impl Fn([f64; 3]) -> T,
) -> Result<Buffer<T>>
where
    T: Default + Copy + Add<Output = T> + Sub<Output = T> + Mul<f64, Output = T>,
{
    let table = alias_table(ctx, grid)?;
    let b = reciprocal_cell(grid)?;
    let shape = grid.index.shape();
    let mut out = ctx.allocate::<T>(grid.index.points())?;
    for (at, value) in out.as_mut_slice().iter_mut().enumerate() {
        *value = function(box_vector(&b, shape, at));
    }
    let alias: Vec<T> = table.vectors.iter().map(|v| function(*v)).collect();
    table.combine(out.as_mut_slice(), 0..table.len(), &alias);
    Ok(out)
}

/// The box wave vectors averaged over the grid's lattice operations, `(3, n)`
/// component-major: the Nyquist planes of a signed-permutation lattice are
/// zeroed (the inversion averages `+-n/2`), a skewed one takes
/// `B (m + mean_V V^-1 (box(V m) - V m))`. Commutes with every operation.
pub fn symmetric_wavevectors(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    let shape = grid.index.shape();
    let n = grid.index.points();
    let lattice = lattice(ctx, grid)?;
    let permutations = lattice
        .operations
        .iter()
        .all(|v| (0..3).all(|i| (0..3).map(|j| v[(i, j)].abs()).sum::<f64>() == 1.));
    let mut out = ctx.allocate::<f64>(crate::product(&[3, n])?)?;
    for at in 0..n {
        let index = [
            at / (shape[1] * shape[2]),
            at / shape[2] % shape[1],
            at % shape[2],
        ];
        let mut m = Vector3::from_fn(|a, _| frequency(index[a], shape[a]));
        if permutations {
            for a in 0..3 {
                if 2. * m[a].abs() == shape[a] as f64 {
                    m[a] = 0.;
                }
            }
        } else {
            let mut shift = Vector3::zeros();
            for (w, v) in lattice.group.iter().zip(&lattice.operations) {
                let image = v * m;
                let wrap = Vector3::from_fn(|a, _| boxed(image[a], shape[a] as f64) - image[a]);
                shift += w.transpose() * wrap;
            }
            m += shift / lattice.group.len() as f64;
        }
        let wave = lattice.reciprocal * m;
        for a in 0..3 {
            out.as_mut_slice()[a * n + at] = wave[a];
        }
    }
    Ok(out)
}
