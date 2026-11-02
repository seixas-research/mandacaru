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
//! Memory: one k-point at a time. [`KSamples`] holds the values and the three derivatives of
//! every function on the mesh (`4 M N` complex numbers, three times the values beyond the
//! values), charged to the `Context`; the caller drops it before the next k-point.
use super::{
    basis_gradient::{short_range_cross, ShortRange},
    images::Region,
    shells::{DenseRows, Sampling, Shells},
    spheres::{projection_slopes, projections, Projector},
};
use crate::{
    math::{fft::Fft3, reciprocal, spline::Cubic, SpatialGrid},
    paw::fields::place_by,
    periodic::invalid,
    product, Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DMatrixView, Matrix3, Vector3};
use num_complex::Complex64;

pub type Matrix = DMatrix<Complex64>;

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
fn minus_cross(samples: &KSamples, weighted: &Matrix) -> [Matrix; 3] {
    std::array::from_fn(|axis| {
        let bra = columns(samples.slope(axis), samples.points, samples.functions);
        -(bra.adjoint() * weighted)
    })
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
    Ok(minus_cross(samples, &weighted))
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
        *to = -(bra.adjoint() * &ket);
    }
    Ok(out)
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
