//! Derivative samples and the center derivative of the projections of the periodic setup: the
//! row-center blocks `G` that the force engine ([`crate::periodic::gradient`]) contracts.
//!
//! A matrix term of the basis is a bilinear form of Bloch sums `chi_{mu k}` sampled on the grid
//! (the overlap, the kinetic operator, the local potential) or on a quadrature sphere (the
//! short-range local potential, the projections). Moving the center of the ROW function `mu`
//! rigidly, with the grid or the sphere held fixed, changes `chi_mu(r)` into
//! `chi_mu(r - delta)`, so its derivative is MINUS the Cartesian derivative that the Bloch kernel
//! returns (`derivative = axis`). Every function here returns the center derivative, the sign
//! included: `G^a[mu, nu] = -<d_a chi_mu | O | chi_nu>`. The derivative is that of the
//! DISCRETIZED form (the sampled gradient, not a spectral one), which is what the energy at the
//! frozen state depends on. The engine turns `G` into the force of atom `X` by the ownership
//! identity `dM_X = P_X G + G^+ P_X`.
//!
//! Memory: one k-point at a time. The grid terms are streamed ([`cross_streamed`]: blocks of mesh
//! points sized from the free budget, [`kinetic_setup`]: the kinetic ket, the one array of the
//! size of the mesh and the basis); [`KSamples`] (the values and the three derivatives of every
//! function on the whole mesh, `4 M N` complex numbers) remains for the meta-GGA's kinetic-density
//! operator and the oracles of the tests.
use super::{
    basis_gradient::{short_range_cross, subset, ShortRange},
    images::Region,
    shells::{DenseRows, Sampling, Shells},
    spheres::{projection_slopes, projections, Projector},
};
use crate::{
    math::{fft::Fft3, gemm::complex_product, reciprocal, spline::Cubic, SpatialGrid},
    paw::fields::place_by,
    periodic::invalid,
    product, Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DMatrixView, Matrix3, Vector3};
use num_complex::Complex64;
use std::ops::Range;

pub type Matrix = DMatrix<Complex64>;

/// `a^H a` without forming an owned adjoint of the grid-sized input.
fn hermitian_gram(a: &Matrix) -> Result<Matrix> {
    crate::math::gemm::complex_gram(a.as_slice(), a.nrows(), a.ncols(), 1.)
}

/// The grid nodes of the home cell as the Bloch kernels read them, and the ball that bounds
/// them.
pub struct Mesh {
    points: [Vec<f64>; 3],
    region: Region,
    /// The volume element and the cell volume.
    pub dv: f64,
    pub volume: f64,
    _held: Reservation,
}

impl Mesh {
    pub fn new(ctx: &Context, grid: &SpatialGrid) -> Result<Self> {
        let n = grid.index.points();
        let held = ctx.reserve(product(&[n, 3, 8])?)?;
        let mut points = [vec![0.; n], vec![0.; n], vec![0.; n]];
        for at in 0..n {
            let p = grid.point(at);
            for axis in 0..3 {
                points[axis][at] = p[axis];
            }
        }
        // The ball about the box of nodes, grown to hold the farthest of the eight corners.
        let shape = grid.index.shape();
        let last = Vector3::from_fn(|a, _| (shape[a] - 1) as f64);
        let origin = Vector3::from(grid.origin);
        let center = origin + grid.step * (last * 0.5);
        let mut radius = 0f64;
        for bits in 0..8 {
            let corner = Vector3::from_fn(|a, _| if bits & (1 << a) == 0 { 0. } else { last[a] });
            radius = radius.max((origin + grid.step * corner - center).norm());
        }
        let dv = grid.step.determinant().abs();
        Ok(Self {
            points,
            region: Region {
                center: center.into(),
                radius: radius + 1e-6,
            },
            dv,
            volume: dv * n as f64,
            _held: held,
        })
    }
    pub fn len(&self) -> usize {
        self.points[0].len()
    }
    pub fn is_empty(&self) -> bool {
        self.points[0].is_empty()
    }
    pub fn points(&self) -> [&[f64]; 3] {
        [&self.points[0], &self.points[1], &self.points[2]]
    }
    /// The ball that holds every node.
    pub fn region(&self) -> Region {
        self.region
    }
}

/// Bytes of the values and the three derivatives of `functions` Bloch sums on `points` nodes
/// (the memory one [`KSamples`] reserves, and a caller's sizing rule for the mesh).
pub fn samples_bytes(functions: usize, points: usize) -> Result<usize> {
    product(&[4, functions, points, 16])
}

/// The Bloch sums of every function at one k-point on the mesh, and their Cartesian derivatives
/// `d chi / d x_axis`; `(function, point)` row-major.
pub struct KSamples {
    pub k: [f64; 3],
    functions: usize,
    points: usize,
    psi: Buffer<Complex64>,
    slope: [Buffer<Complex64>; 3],
}

impl KSamples {
    pub fn new(
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        mesh: &Mesh,
        k: [f64; 3],
    ) -> Result<Self> {
        let (m, n) = (shells.functions(), mesh.len());
        let sampling = Sampling {
            lattice,
            region: mesh.region,
            kpoints: &[k],
        };
        let mut psi = ctx.allocate::<Complex64>(product(&[m, n])?)?;
        shells.values(
            ctx,
            &sampling,
            mesh.points(),
            &mut DenseRows::new(psi.as_mut_slice(), m, n),
        )?;
        let mut slope = Vec::with_capacity(3);
        for axis in 0..3 {
            let mut rows = ctx.allocate::<Complex64>(product(&[m, n])?)?;
            shells.derivatives(
                ctx,
                &sampling,
                mesh.points(),
                axis,
                &mut DenseRows::new(rows.as_mut_slice(), m, n),
            )?;
            slope.push(rows);
        }
        let slope: [Buffer<Complex64>; 3] = slope.try_into().ok().expect("three axes");
        Ok(Self {
            k,
            functions: m,
            points: n,
            psi,
            slope,
        })
    }
    pub fn functions(&self) -> usize {
        self.functions
    }
    /// `chi_{mu k}(x_j)`, `(function, point)`.
    pub fn values(&self) -> &[Complex64] {
        self.psi.as_slice()
    }
    /// `d chi_{mu k} / d x_axis (x_j)` (the Cartesian derivative, not the center derivative).
    pub fn slope(&self, axis: usize) -> &[Complex64] {
        self.slope[axis].as_slice()
    }
}

/// The `(points, count)` column-major view of a `(count, points)` row-major array.
fn columns(rows: &[Complex64], points: usize, count: usize) -> DMatrixView<'_, Complex64> {
    DMatrixView::from_slice(rows, points, count)
}

/// `-sum_x conj(bra[mu, x]) w_x ket[nu, x]` for the three derivative bras; `weighted` is the ket
/// already multiplied by the weights.
fn minus_cross(samples: &KSamples, weighted: &Matrix) -> Result<[Matrix; 3]> {
    let mut out = [
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
    ];
    for (axis, to) in out.iter_mut().enumerate() {
        let bra = columns(samples.slope(axis), samples.points, samples.functions);
        *to = -complex_product(bra, true, weighted.as_view())?;
    }
    Ok(out)
}

/// Points per block of a streamed pass over the mesh: `copies` arrays of `functions x block`
/// complex values must fit in a quarter of the free budget (at most 256 MiB), never fewer than
/// 512 points nor more than the mesh. A smaller budget gives smaller blocks, never an error.
pub fn block_len(ctx: &Context, functions: usize, points: usize, copies: usize) -> usize {
    let free = ctx.memory_limit().saturating_sub(ctx.memory_used());
    let target = (free / 4).min(256 << 20);
    (target / (copies.max(1) * functions.max(1) * 16))
        .max(512)
        .min(points.max(1))
}

/// The Bloch sums `(function, point)` of the points `range` of the mesh, with (`slopes`) their
/// three Cartesian derivatives.
pub fn block_samples(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    mesh: &Mesh,
    k: [f64; 3],
    range: &Range<usize>,
    slopes: bool,
) -> Result<(Buffer<Complex64>, Vec<Buffer<Complex64>>)> {
    let (m, nb) = (shells.functions(), range.len());
    let all = mesh.points();
    let points = [
        &all[0][range.clone()],
        &all[1][range.clone()],
        &all[2][range.clone()],
    ];
    let sampling = Sampling {
        lattice,
        region: mesh.region,
        kpoints: &[k],
    };
    let mut values = ctx.allocate::<Complex64>(product(&[m, nb])?)?;
    shells.values(
        ctx,
        &sampling,
        points,
        &mut DenseRows::new(values.as_mut_slice(), m, nb),
    )?;
    let mut derivatives = Vec::new();
    if slopes {
        for axis in 0..3 {
            let mut rows = ctx.allocate::<Complex64>(product(&[m, nb])?)?;
            shells.derivatives(
                ctx,
                &sampling,
                points,
                axis,
                &mut DenseRows::new(rows.as_mut_slice(), m, nb),
            )?;
            derivatives.push(rows);
        }
    }
    Ok((values, derivatives))
}

/// The ket a streamed cross term contracts the derivative bras with.
pub enum Ket<'a> {
    /// `dV w(x) chi_nu(x)`: the overlap (`None`) or a local potential.
    Weighted(Option<&'a [f64]>),
    /// Columns `(point, function)` already formed on the whole mesh (the kinetic ket).
    Columns(&'a Matrix),
}

/// `G^a = -sum_x conj(d_a chi_mu(x)) ket_nu(x)` for every ket of `kets`, three `m x m` matrices
/// each, from blocks of mesh points: the values and slopes of one block at a time are formed once
/// for all the kets, so the memory is `O(functions x block)` and not `O(functions x mesh)`.
pub fn cross_streamed(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    mesh: &Mesh,
    k: [f64; 3],
    kets: &[Ket<'_>],
) -> Result<Vec<[Matrix; 3]>> {
    let block = block_len(ctx, shells.functions(), mesh.len(), 6 + kets.len());
    cross_streamed_blocks(ctx, shells, lattice, mesh, k, kets, block)
}

/// [`cross_streamed`] with blocks of `block` points (the tests' way to force many blocks).
#[doc(hidden)]
pub fn cross_streamed_blocks(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    mesh: &Mesh,
    k: [f64; 3],
    kets: &[Ket<'_>],
    block: usize,
) -> Result<Vec<[Matrix; 3]>> {
    let (m, n) = (shells.functions(), mesh.len());
    let block = block.clamp(1, n.max(1));
    for ket in kets {
        match ket {
            Ket::Weighted(Some(v)) if v.len() != n => {
                return Err(invalid("the local potential needs one value per mesh node"))
            }
            Ket::Columns(c) if c.shape() != (n, m) => {
                return Err(invalid("a ket needs one column per function on the mesh"))
            }
            _ => {}
        }
    }
    let mut out: Vec<[Matrix; 3]> = kets
        .iter()
        .map(|_| std::array::from_fn(|_| Matrix::zeros(m, m)))
        .collect();
    let mut start = 0;
    while start < n {
        let range = start..(start + block).min(n);
        let nb = range.len();
        let (values, slopes) = block_samples(ctx, shells, lattice, mesh, k, &range, true)?;
        let _held = ctx.reserve(product(&[m, nb, 16, 2])?)?;
        for (ket, to) in kets.iter().zip(out.iter_mut()) {
            let weighted;
            let view = match ket {
                Ket::Weighted(potential) => {
                    weighted = Matrix::from_fn(nb, m, |x, nu| {
                        values.as_slice()[nu * nb + x]
                            * (mesh.dv * potential.map_or(1., |v| v[range.start + x]))
                    });
                    weighted.as_view()
                }
                Ket::Columns(c) => c.view((range.start, 0), (nb, m)),
            };
            for (axis, slope) in slopes.iter().enumerate() {
                to[axis] -= complex_product(columns(slope.as_slice(), nb, m), true, view)?;
            }
        }
        start = range.end;
    }
    Ok(out)
}

/// What the kinetic term needs of the basis on the reciprocal mesh at `k`.
pub struct KineticSetup {
    /// `K_nu(x) = dV e^{ik.x} sum_G e^{iG.x} w(G) u_nu(G)`, `(point, function)`: the kinetic ket in
    /// real space, so that `<d chi_mu | T | chi_nu> = sum_x conj(d chi_mu(x)) K_nu(x)`.
    pub ket: Matrix,
    /// `T[mu, nu] = sum_G conj(u_mu) w u_nu` and, per Voigt pair, `S[mu, nu] = sum_G conj(u_mu)
    /// Q_ab(G) u_nu / Omega` (`Q` the alias mean of `Q_a Q_b`): the symbol terms of the stress.
    pub moments: Option<(Matrix, Vec<Matrix>)>,
}

/// The kinetic ket of every function at `k`, one shell at a time (`u = dV FFT[chi e^{-ik.x}]` is
/// held for the whole basis only as the `n x m` array the ket then replaces in place: one
/// `n x m` array in all). With `moments`, `T` and the six `S` are accumulated from `u` before it
/// is transformed.
pub fn kinetic_setup(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    moments: bool,
) -> Result<KineticSetup> {
    let (m, n) = (shells.functions(), mesh.len());
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    let weight = kinetic_weights(ctx, grid, mesh, k)?;
    let _held = ctx.reserve(product(&[n, m, 16])?)?;
    let mut u = Matrix::zeros(n, m);
    let sampling = Sampling {
        lattice,
        region: mesh.region,
        kpoints: &[k],
    };
    for shell in shells.shells() {
        let part = subset(shells, &shell.rows)?;
        let nm = shell.rows.len();
        let mut rows = ctx.allocate::<Complex64>(product(&[nm, n])?)?;
        part.values(
            ctx,
            &sampling,
            mesh.points(),
            &mut DenseRows::new(rows.as_mut_slice(), nm, n),
        )?;
        let (periodic, _held_part) = periodic_parts(ctx, &mut fft, mesh, k, rows.as_slice(), nm)?;
        for (i, &row) in shell.rows.iter().enumerate() {
            u.column_mut(row).copy_from(&periodic.column(i));
        }
    }
    let moments = if moments {
        let outer = reciprocal::bloch_outer(ctx, grid, k)?;
        let chunk = block_len(ctx, m, n, 6);
        let mut t = Matrix::zeros(m, m);
        let mut symbols = vec![Matrix::zeros(m, m); 6];
        let mut start = 0;
        while start < n {
            let stop = (start + chunk).min(n);
            let bra = u.view((start, 0), (stop - start, m));
            let scaled = |factor: &dyn Fn(usize) -> f64| {
                Matrix::from_fn(stop - start, m, |x, j| {
                    u[(start + x, j)] * factor(start + x)
                })
            };
            t += complex_product(bra, true, scaled(&|g| weight[g]).as_view())?;
            for (pair, symbol) in symbols.iter_mut().enumerate() {
                let row = &outer.as_slice()[pair * n..(pair + 1) * n];
                *symbol += complex_product(bra, true, scaled(&|g| row[g] / mesh.volume).as_view())?;
            }
            start = stop;
        }
        Some((t, symbols))
    } else {
        None
    };
    // The ket in place: `w u`, the backward transform (the library's is divided by `N`: undone
    // here), the phase and `dV`.
    let [x, y, z] = mesh.points();
    let phase: Vec<Complex64> = (0..n)
        .map(|j| {
            let angle = k[0] * x[j] + k[1] * y[j] + k[2] * z[j];
            Complex64::new(angle.cos(), angle.sin()) * (mesh.dv * n as f64)
        })
        .collect();
    for column in u.as_mut_slice().chunks_mut(n) {
        for (v, s) in column.iter_mut().zip(&weight) {
            *v *= *s;
        }
        fft.transform(column, true)?;
        for (v, p) in column.iter_mut().zip(&phase) {
            *v *= *p;
        }
    }
    Ok(KineticSetup { ket: u, moments })
}

/// The grid overlap, `G^a = -dV sum_x conj(d_a chi_mu) chi_nu`, or with `potential` the local
/// potential `G^a = -dV sum_x conj(d_a chi_mu) v(x) chi_nu` (P3 for the potential of the fields).
pub fn grid_cross(
    ctx: &Context,
    mesh: &Mesh,
    samples: &KSamples,
    potential: Option<&[f64]>,
) -> Result<[Matrix; 3]> {
    let (m, n) = (samples.functions, samples.points);
    if potential.is_some_and(|v| v.len() != n) {
        return Err(invalid("the local potential needs one value per mesh node"));
    }
    let _held = ctx.reserve(product(&[m, n, 16])?)?;
    let ket = samples.values();
    let weighted = Matrix::from_fn(n, m, |x, nu| {
        ket[nu * n + x] * (mesh.dv * potential.map_or(1., |v| v[x]))
    });
    minus_cross(samples, &weighted)
}

/// The periodic part `dV * FFT(e^{-ik.r} f)` of every row of `rows` (`count` functions on the
/// mesh), column-major `(points, count)`, with the reservation that charges it to the `Context`
/// for as long as the caller keeps it.
pub(crate) fn periodic_parts(
    ctx: &Context,
    fft: &mut Fft3,
    mesh: &Mesh,
    k: [f64; 3],
    rows: &[Complex64],
    count: usize,
) -> Result<(Matrix, Reservation)> {
    let n = mesh.len();
    if rows.len() != product(&[n, count])? {
        return Err(invalid(
            "orbital samples do not match the grid and function count",
        ));
    }
    let held = ctx.reserve(product(&[n, count, 16])?)?;
    let [x, y, z] = mesh.points();
    let phase: Vec<Complex64> = (0..n)
        .map(|j| {
            let angle = -(k[0] * x[j] + k[1] * y[j] + k[2] * z[j]);
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    let mut out = vec![Complex64::new(0., 0.); n * count];
    for (to, from) in out.chunks_mut(n).zip(rows.chunks(n)) {
        for ((t, f), p) in to.iter_mut().zip(from).zip(&phase) {
            *t = f * p;
        }
        fft.transform(to, false)?;
        for t in to.iter_mut() {
            *t *= mesh.dv;
        }
    }
    Ok((Matrix::from_vec(n, count, out), held))
}

/// The multiplier of the kinetic operator on the periodic part at `k`:
/// `|G + k|^2 / 2` averaged over the aliases the grid's lattice operations allow, over the cell
/// volume (`reciprocal::bloch`, which also pins the equivariance of the star of `k`).
pub(crate) fn kinetic_weights(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    k: [f64; 3],
) -> Result<Vec<f64>> {
    let symbols = reciprocal::bloch(ctx, grid, k)?;
    Ok(symbols
        .squares
        .as_slice()
        .iter()
        .map(|s| 0.5 * s / mesh.volume)
        .collect())
}

/// The kinetic matrix `T[mu, nu] = sum_G conj(u_mu(G)) w(G) u_nu(G)` between two sets of
/// functions on the mesh at `k` (`bra`: `nb` rows, `ket`: `nc` rows, `(function, point)`). With
/// the basis on both sides it is the kinetic matrix of the crystal.
pub fn kinetic_form(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    k: [f64; 3],
    bra: (&[Complex64], usize),
    ket: (&[Complex64], usize),
) -> Result<Matrix> {
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    let w = kinetic_weights(ctx, grid, mesh, k)?;
    let (a, _held_bra) = periodic_parts(ctx, &mut fft, mesh, k, bra.0, bra.1)?;
    let (mut b, _held_ket) = periodic_parts(ctx, &mut fft, mesh, k, ket.0, ket.1)?;
    for mut column in b.column_iter_mut() {
        for (v, s) in column.iter_mut().zip(&w) {
            *v *= *s;
        }
    }
    Ok(a.adjoint() * b)
}

/// Kinetic matrix with the same orbitals on both sides. Weighting the single
/// transform by sqrt(w) gives the same bilinear form while avoiding a second
/// full orbital-grid transform and its workspace.
pub fn kinetic_self(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    k: [f64; 3],
    samples: &[Complex64],
    functions: usize,
) -> Result<Matrix> {
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    let mut weights = kinetic_weights(ctx, grid, mesh, k)?;
    for weight in &mut weights {
        *weight = weight.sqrt();
    }
    let (mut transformed, _held) = periodic_parts(ctx, &mut fft, mesh, k, samples, functions)?;
    for mut column in transformed.column_iter_mut() {
        for (value, weight) in column.iter_mut().zip(&weights) {
            *value *= *weight;
        }
    }
    hermitian_gram(&transformed)
}

/// Same-basis kinetic matrix without materializing the mesh coordinate arrays.
pub fn kinetic_grid(
    ctx: &Context,
    grid: &SpatialGrid,
    k: [f64; 3],
    samples: &[Complex64],
    functions: usize,
) -> Result<Matrix> {
    kinetic_grid_typed(ctx, grid, k, samples, functions, |x| x)
}

/// Float32 sample variant of [`kinetic_grid`].
pub fn kinetic_grid_f32(
    ctx: &Context,
    grid: &SpatialGrid,
    k: [f64; 3],
    samples: &[num_complex::Complex32],
    functions: usize,
) -> Result<Matrix> {
    kinetic_grid_typed(ctx, grid, k, samples, functions, |x| {
        Complex64::new(x.re as f64, x.im as f64)
    })
}

fn kinetic_grid_typed<T: Copy>(
    ctx: &Context,
    grid: &SpatialGrid,
    k: [f64; 3],
    samples: &[T],
    functions: usize,
    convert: impl Fn(T) -> Complex64,
) -> Result<Matrix> {
    let n = grid.index.points();
    if samples.len() != product(&[n, functions])? {
        return Err(invalid(
            "orbital samples do not match the grid and function count",
        ));
    }
    let dv = grid.step.determinant().abs();
    let symbols = reciprocal::bloch(ctx, grid, k)?;
    let _weights_held = ctx.reserve(product(&[n, 8])?)?;
    let weights: Vec<f64> = symbols
        .squares
        .as_slice()
        .iter()
        .map(|s| (0.5 * s / (dv * n as f64)).sqrt() * dv)
        .collect();
    drop(symbols);
    let mut phase = ctx.allocate::<Complex64>(n)?;
    for (j, value) in phase.as_mut_slice().iter_mut().enumerate() {
        let r = grid.point(j);
        let angle = -(k[0] * r[0] + k[1] * r[1] + k[2] * r[2]);
        *value = Complex64::new(angle.cos(), angle.sin());
    }
    let _held = ctx.reserve(product(&[n, functions, 16])?)?;
    let mut transformed = vec![Complex64::new(0., 0.); n * functions];
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    for (to, from) in transformed.chunks_mut(n).zip(samples.chunks(n)) {
        for ((target, source), factor) in to.iter_mut().zip(from).zip(phase.as_slice()) {
            *target = convert(*source) * factor;
        }
        fft.transform(to, false)?;
        for (target, factor) in to.iter_mut().zip(&weights) {
            *target *= *factor;
        }
    }
    let transformed = Matrix::from_vec(n, functions, transformed);
    hermitian_gram(&transformed)
}

/// P2: `G^a = -sum_G conj(FFT[e^{-ik.r} d_a chi_mu]) w(G) FFT[e^{-ik.r} chi_nu]`, the derivative
/// of [`kinetic_form`] at the basis, with the SAMPLED derivative (the energy at the frozen
/// state is a function of the samples, not of the band-limited function).
pub fn kinetic_cross(
    ctx: &Context,
    grid: &SpatialGrid,
    mesh: &Mesh,
    samples: &KSamples,
) -> Result<[Matrix; 3]> {
    let (m, k) = (samples.functions, samples.k);
    let mut fft = Fft3::new(ctx, grid.index.shape())?;
    let w = kinetic_weights(ctx, grid, mesh, k)?;
    let (mut ket, _held_ket) = periodic_parts(ctx, &mut fft, mesh, k, samples.values(), m)?;
    for mut column in ket.column_iter_mut() {
        for (v, s) in column.iter_mut().zip(&w) {
            *v *= *s;
        }
    }
    let mut out = [
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
    ];
    for (axis, to) in out.iter_mut().enumerate() {
        let (bra, _held_bra) = periodic_parts(ctx, &mut fft, mesh, k, samples.slope(axis), m)?;
        *to = -complex_product(bra.as_view(), true, ket.as_view())?;
    }
    Ok(out)
}

/// P6, the meta-GGA operator: the row-center derivative of
/// `T = sum_axis (d_axis chi)^+ diag(dv v_tau / 2) (d_axis chi)`, the matrix
/// [`crate::periodic::fields::kinetic_operator`] adds to `H`, with the SCF's spectral gradient
/// ([`crate::periodic::fields::bloch_gradients`]) and the SAMPLED center derivative of the bra
/// (as [`kinetic_cross`]): `G^a = -sum_axis (d_axis slope_a)^+ diag(dv v_tau / 2) d_axis chi`.
pub fn tau_cross(
    ctx: &Context,
    grid: &SpatialGrid,
    samples: &KSamples,
    potential: &[f64],
) -> Result<[Matrix; 3]> {
    use crate::periodic::fields::bloch_gradients;
    let (m, n, k) = (samples.functions, samples.points, samples.k);
    if potential.len() != n || n != grid.index.points() {
        return Err(invalid(
            "crystal gradient: the kinetic potential must match the grid",
        ));
    }
    let half_dv = 0.5 * grid.index.dv();
    // [axis][function][point], the ket weighted by dv v_tau / 2 in place.
    let mut ket = bloch_gradients(ctx, grid, k, samples.values(), m)?;
    for row in ket.as_mut_slice().chunks_mut(n) {
        for (value, v) in row.iter_mut().zip(potential) {
            *value *= half_dv * v;
        }
    }
    let mut out = [
        Matrix::zeros(m, m),
        Matrix::zeros(m, m),
        Matrix::zeros(m, m),
    ];
    for (a, to) in out.iter_mut().enumerate() {
        let bra = bloch_gradients(ctx, grid, k, samples.slope(a), m)?;
        for axis in 0..3 {
            let span = axis * m * n..(axis + 1) * m * n;
            let b = columns(&bra.as_slice()[span.clone()], n, m);
            let w = columns(&ket.as_slice()[span], n, m);
            *to -= complex_product(b, true, w)?;
        }
    }
    Ok(out)
}

/// E6: the slope `d tau_core / d c_axis` of one atom's partial-core kinetic density, the von
/// Weizsaecker form of its smooth core ([`crate::paw::fields::core_kinetic`], the table the
/// setup samples for a meta-GGA), placed like [`core_gradient`].
pub fn core_kinetic_gradient(
    ctx: &Context,
    grid: &SpatialGrid,
    radius: &[f64],
    core: &[f64],
    center: [f64; 3],
    axis: usize,
) -> Result<Buffer<f64>> {
    let density = Cubic::new(ctx, ctx.copy_slice(radius)?, ctx.copy_slice(core)?)?;
    let tau = crate::paw::fields::core_kinetic(ctx, &density)?;
    core_gradient(ctx, grid, radius, tau.values.as_slice(), center, axis)
}

/// `C_k` (`m x p`): `<chi_{mu k} | p_p>` on the projector spheres.
pub fn projection(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
) -> Result<Matrix> {
    let c = projections(ctx, shells, lattice, &[k], projectors)?;
    Ok(Matrix::from_row_slice(
        shells.functions(),
        projectors.len(),
        c.as_slice(),
    ))
}

/// `G_C^a(k) = dC/dc_mu`: the derivative of `C_k` with respect to the center of the ROW function
/// (the projector and its sphere fixed), `-<d_a chi_mu | p_p>` summed over the images with the
/// phases of `C`. Translation invariance of the sphere rule makes `-G_C[mu, p]` the derivative
/// with respect to the center of the projector `p` as well: `dC = P_X G_C - G_C Q_X`.
pub fn projection_gradient(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
) -> Result<[Matrix; 3]> {
    let (m, p) = (shells.functions(), projectors.len());
    let mut out = [
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
        Matrix::zeros(0, 0),
    ];
    for (axis, to) in out.iter_mut().enumerate() {
        let c = projection_slopes(ctx, shells, lattice, &[k], projectors, Some(axis))?;
        *to = -Matrix::from_row_slice(m, p, c.as_slice());
    }
    Ok(out)
}

/// P4, per sphere atom: `G_A^a = -<d_a chi_mu | V^sr_A | chi_nu>` at `k`, the row-center
/// derivative of atom `A`'s short-range local matrix (its sphere fixed). One entry per sphere.
pub fn short_range_gradients(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    spheres: &[ShortRange<'_>],
) -> Result<Vec<[Matrix; 3]>> {
    let m = shells.functions();
    spheres
        .iter()
        .map(|sphere| {
            let mut out = [
                Matrix::zeros(0, 0),
                Matrix::zeros(0, 0),
                Matrix::zeros(0, 0),
            ];
            for (axis, to) in out.iter_mut().enumerate() {
                let block =
                    short_range_cross(ctx, lattice, &[k], sphere, shells, Some(axis), shells)?;
                *to = -Matrix::from_row_slice(m, m, block.as_slice());
            }
            Ok(out)
        })
        .collect()
}

/// E5: `d rho_core / d R_axis` of one atom's smooth core on the mesh, `-rho'(|r - R|)
/// (r - R)_axis / |r - R|` summed over the images, from the analytic slope of the spline the
/// placement samples (zero at the nucleus, where the direction is undefined and the core is
/// even). `<v_xc | d rho_core>` is [`mesh_dot`] of this field.
pub fn core_gradient(
    ctx: &Context,
    grid: &SpatialGrid,
    radius: &[f64],
    core: &[f64],
    center: [f64; 3],
    axis: usize,
) -> Result<Buffer<f64>> {
    if axis > 2 {
        return Err(invalid("a gradient axis is 0, 1 or 2"));
    }
    let profile = Cubic::new(ctx, ctx.copy_slice(radius)?, ctx.copy_slice(core)?)?;
    place_by(ctx, grid, &profile, &[center], &|distance, offset| {
        if distance > 0. {
            -profile.value_slope(distance).1 * offset[axis] / distance
        } else {
            0.
        }
    })
}

/// `dV sum_x a_x b_x`.
pub fn mesh_dot(mesh: &Mesh, a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != mesh.len() || b.len() != mesh.len() {
        return Err(invalid("two fields on the mesh need one value per node"));
    }
    Ok(mesh.dv * a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>())
}
