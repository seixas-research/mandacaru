//! Quadrature spheres around the atoms of the home cell, and the projections of the Bloch sums
//! on the projectors that live in them.
//!
//! A short-range term (a projector `p` of a cutoff `r_cut`, a short-range potential) is a
//! cell integral of a Bloch sum that unfolds to an all-space integral of the home-cell term. It
//! is evaluated on a product rule over the ball about the atom, streamed in chunks of points
//! sorted into 1-Bohr bins so that the kernels cull lattice images per block:
//!
//! ```text
//! C_{mu p}(k) = <chi_{mu k} | p_p> = sum_R e^{-i k.R} sum_x conj(phi_mu(x - R)) p_p(x) w_x
//! ```
//!
//! The sum over the points runs image by image before the phases of the k-points are applied,
//! so the work on the points does not grow with the number of k-points. Only the functions that
//! reach the sphere contribute ([`Shells::reaching`]); the others are exactly zero on it.
use super::images::{Images, Region};
use super::shells::{DenseRows, Radial, Sampling, Shells};
use crate::{
    kernels::{self, ShellImages},
    paw::local_split::{angular_rule, radial_rule},
    product, Buffer, Context, Error, Reservation, Result,
};
use nalgebra::{DMatrix, DMatrixView, Matrix3};
use num_complex::Complex64;
use std::{collections::BTreeMap, ops::Range, sync::Arc};

/// Radial, polar and azimuthal points of a projector sphere.
pub const PROJECTION_RADIAL: usize = 64;
pub const PROJECTION_POLAR: usize = 24;
pub const PROJECTION_AZIMUTHAL: usize = 48;
/// Radial, polar and azimuthal points of a short-range local sphere (two radial panels).
pub const SHORT_RANGE_RADIAL: usize = 32;
pub const SHORT_RANGE_POLAR: usize = 16;
pub const SHORT_RANGE_AZIMUTHAL: usize = 36;
/// Points per chunk of a streamed sphere, and the bytes of Bloch sums one chunk may hold. A
/// chunk of 2048 points holds an eighth of the Bloch sums a chunk of 16384 did, and the sums
/// over the chunks are the same quadrature.
pub const CHUNK_POINTS: usize = 2048;
pub const CHUNK_BYTES: usize = 1 << 29;

/// Points per chunk when `kpoints` x `functions` Bloch sums (and their conjugates) of that many
/// points would pass [`CHUNK_BYTES`]: a function of the sizes alone, never of the thread count.
pub fn chunk_size(kpoints: usize, functions: usize) -> usize {
    let fit = CHUNK_BYTES / (32 * kpoints.max(1) * functions.max(1));
    CHUNK_POINTS.min(fit).max(512)
}

/// A product rule on a ball: points sorted into 1-Bohr bins, their weights (`w r^2 x` solid
/// angle) and the radius of each point's shell (a radial potential multiplies the weights).
#[derive(Debug)]
pub struct Sphere {
    pub center: [f64; 3],
    pub radius: f64,
    points: [Vec<f64>; 3],
    weights: Vec<f64>,
    radii: Vec<f64>,
    _held: Reservation,
}

impl Sphere {
    /// Gauss-Legendre in the radius (two panels when `panel` lies inside `(0, radius)`), a
    /// Gauss-Legendre by midpoint product over the angles.
    pub fn rule(
        ctx: &Context,
        center: [f64; 3],
        radius: f64,
        radial: usize,
        polar: usize,
        azimuthal: usize,
        panel: f64,
    ) -> Result<Self> {
        if !(radius.is_finite() && radius > 0.) || center.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid(
                "a sphere needs a finite center and a positive radius".into(),
            ));
        }
        let (r, w_r) = radial_rule(radius, panel, radial)?;
        let (directions, w_ang) = angular_rule(polar, azimuthal)?;
        let n = r.len() * w_ang.len();
        let held = ctx.reserve(product(&[n, 6, 8])?)?;
        // Radial major, as the rule is generated; then the stable sort into bins.
        let at = |j: usize| (j / w_ang.len(), j % w_ang.len());
        let coordinate = |j: usize, axis: usize| {
            let (i, a) = at(j);
            center[axis] + r[i] * directions[axis][a]
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&j| {
            (
                coordinate(j, 0).floor() as i64,
                coordinate(j, 1).floor() as i64,
                coordinate(j, 2).floor() as i64,
            )
        });
        let points = [0, 1, 2].map(|axis| order.iter().map(|&j| coordinate(j, axis)).collect());
        let weights = order
            .iter()
            .map(|&j| {
                let (i, a) = at(j);
                w_r[i] * w_ang[a]
            })
            .collect();
        let radii = order.iter().map(|&j| r[at(j).0]).collect();
        Ok(Self {
            center,
            radius,
            points,
            weights,
            radii,
            _held: held,
        })
    }
    /// The rule of a projector: [`PROJECTION_RADIAL`] x [`PROJECTION_POLAR`] x
    /// [`PROJECTION_AZIMUTHAL`] points on one panel.
    pub fn projector(ctx: &Context, center: [f64; 3], r_cut: f64) -> Result<Self> {
        Self::rule(
            ctx,
            center,
            r_cut,
            PROJECTION_RADIAL,
            PROJECTION_POLAR,
            PROJECTION_AZIMUTHAL,
            0.,
        )
    }
    pub fn len(&self) -> usize {
        self.weights.len()
    }
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
    pub fn region(&self) -> Region {
        Region {
            center: self.center,
            radius: self.radius,
        }
    }
    /// Consecutive index ranges of at most `size` points.
    pub fn chunks(&self, size: usize) -> impl Iterator<Item = Range<usize>> {
        let n = self.len();
        (0..n)
            .step_by(size.max(1))
            .map(move |start| start..(start + size.max(1)).min(n))
    }
    pub fn points(&self, range: Range<usize>) -> [&[f64]; 3] {
        [0, 1, 2].map(|axis| &self.points[axis][range.clone()])
    }
    pub fn weights(&self, range: Range<usize>) -> &[f64] {
        &self.weights[range]
    }
    /// The radius of the shell each point of `range` sits on.
    pub fn radii(&self, range: Range<usize>) -> &[f64] {
        &self.radii[range]
    }
}

/// One projector function `p_{l m i}(r - c) = R_i(|r - c|) Y_lm`, complex Condon-Shortley
/// harmonics, with the atom it belongs to and the radius beyond which it vanishes.
#[derive(Clone, Debug)]
pub struct Projector {
    pub atom: usize,
    pub l: usize,
    pub m: i32,
    /// Radial index within the `(atom, l, m)` channel.
    pub index: usize,
    pub center: [f64; 3],
    pub r_cut: f64,
    pub radial: Arc<Radial>,
}

/// `p(x)` on `points` into `out`, by the Bloch kernel with the single image at the center.
pub(crate) fn evaluate(
    ctx: &Context,
    projector: &Projector,
    points: [&[f64]; 3],
    out: &mut [Complex64],
) -> Result<()> {
    let spline = projector.radial.spline(projector.l)?;
    // The function ends at the last node of its table (an endpoint-exclusive support).
    let support = f64::from_bits(projector.radial.extent().to_bits() + 1);
    out.fill(Complex64::new(0., 0.));
    kernels::bloch_shell(
        ctx,
        &ShellImages {
            xyz: points,
            center: projector.center,
            images: &[[0.; 3]],
            phases: &[Complex64::new(1., 0.)],
            support,
            spline: &spline,
            ms: &[projector.m],
            derivative: -1,
        },
        &[0],
        1,
        out,
    )
}

/// Adds `sum_x conj(chi_mu k(x)) columns[p, x]` for the shells that reach `region`, into
/// `out[(k * functions + mu) * P + p]`. `columns` is `(P, points)`, already weighted.
pub fn add_projections(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    region: &Region,
    points: [&[f64]; 3],
    columns: &[Complex64],
    out: &mut [Complex64],
) -> Result<()> {
    add_projection_slopes(
        ctx, shells, lattice, kpoints, region, points, columns, None, out,
    )
}

/// [`add_projections`] with the Bloch sums replaced by their Cartesian derivative along `axis`
/// (`conj(d chi / d x_axis)`): the projection's derivative with respect to minus the center of
/// the function, the projector and its quadrature sphere held fixed.
pub fn add_projection_slopes(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    region: &Region,
    points: [&[f64]; 3],
    columns: &[Complex64],
    axis: Option<usize>,
    out: &mut [Complex64],
) -> Result<()> {
    let derivative = axis.map_or(-1, |a| a.min(3) as i32);
    let n = points[0].len();
    let count = columns.len() / n.max(1);
    let functions = shells.functions();
    if n == 0 || columns.len() != count * n || out.len() != kpoints.len() * functions * count {
        return Err(Error::Invalid(
            "projection columns or output do not match the points".into(),
        ));
    }
    for shell in shells.shells() {
        let images = Images::around(ctx, lattice, region, shell.center, shell.radial.support())?;
        if images.is_empty() {
            continue;
        }
        let spline = shell.radial.spline(shell.l)?;
        // The kernel sums over the points image by image; the phases are applied here.
        let unit = vec![Complex64::new(1., 0.); images.len()];
        let blocks = kernels::bloch_project(
            ctx,
            &ShellImages {
                xyz: points,
                center: shell.center,
                images: images.vectors(),
                phases: &unit,
                support: shell.radial.support(),
                spline: &spline,
                ms: &shell.ms,
                derivative,
            },
            columns,
        )?;
        let phases = images.phases(ctx, kpoints, -1.)?;
        let nm = shell.ms.len();
        for k in 0..kpoints.len() {
            for (i, &row) in shell.rows.iter().enumerate() {
                let target = &mut out[(k * functions + row) * count..][..count];
                for r in 0..images.len() {
                    let phase = phases.as_slice()[k * images.len() + r];
                    let block = &blocks.as_slice()[(r * nm + i) * count..][..count];
                    for (to, from) in target.iter_mut().zip(block) {
                        *to += phase * from;
                    }
                }
            }
        }
    }
    Ok(())
}

/// `C[k, mu, p] = <chi_{mu k} | p_p>`, `(k, function, projector)` row-major.
///
/// Projectors of one atom with the same cutoff share one sphere and one set of Bloch-sum images.
pub fn projections(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    projectors: &[Projector],
) -> Result<Buffer<Complex64>> {
    projection_slopes(ctx, shells, lattice, kpoints, projectors, None)
}

/// [`projections`] with the functions replaced by their Cartesian derivative along `axis`
/// (`<d_axis chi_mu | p_p>`; the center derivative of `C` is minus this, see
/// [`super::pulay::projection_gradient`]).
pub fn projection_slopes(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    projectors: &[Projector],
    axis: Option<usize>,
) -> Result<Buffer<Complex64>> {
    let (functions, count) = (shells.functions(), projectors.len());
    if kpoints.is_empty() {
        return Err(Error::Invalid("projections need a k-point".into()));
    }
    let mut out = ctx.allocate::<Complex64>(product(&[kpoints.len(), functions, count])?)?;
    let mut spheres: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for (p, projector) in projectors.iter().enumerate() {
        if !(projector.r_cut.is_finite() && projector.r_cut > 0.) {
            return Err(Error::Invalid(
                "a projector needs a positive cutoff radius".into(),
            ));
        }
        spheres
            .entry((projector.atom, (projector.r_cut * 1e12).round() as i64))
            .or_default()
            .push(p);
    }
    for members in spheres.values() {
        let first = &projectors[members[0]];
        let region = Region {
            center: first.center,
            radius: first.r_cut,
        };
        let reached = shells.reaching(ctx, lattice, &region)?;
        if reached.rows.is_empty() {
            continue;
        }
        let sphere = Sphere::projector(ctx, first.center, first.r_cut)?;
        let near = reached.rows.len();
        let mut block =
            ctx.allocate::<Complex64>(product(&[kpoints.len(), near, members.len()])?)?;
        for range in sphere.chunks(CHUNK_POINTS) {
            let points = sphere.points(range.clone());
            let weights = sphere.weights(range.clone());
            let mut columns =
                ctx.allocate::<Complex64>(product(&[members.len(), weights.len()])?)?;
            for (row, &p) in columns
                .as_mut_slice()
                .chunks_mut(weights.len())
                .zip(members)
            {
                evaluate(ctx, &projectors[p], points, row)?;
                for (value, w) in row.iter_mut().zip(weights) {
                    *value *= *w;
                }
            }
            add_projection_slopes(
                ctx,
                &reached.shells,
                lattice,
                kpoints,
                &region,
                points,
                columns.as_slice(),
                axis,
                block.as_mut_slice(),
            )?;
        }
        for k in 0..kpoints.len() {
            for (a, &mu) in reached.rows.iter().enumerate() {
                for (b, &p) in members.iter().enumerate() {
                    out.as_mut_slice()[(k * functions + mu) * count + p] +=
                        block.as_slice()[(k * near + a) * members.len() + b];
                }
            }
        }
    }
    Ok(out)
}

/// The matrix `sum_x conj(chi_a(x)) w(x) chi_b(x)` of a weighted sphere between the functions
/// that reach it: `values` is `(k, rows, columns)` row-major, with the positions of the rows and
/// the columns in the full bases.
#[derive(Debug)]
pub struct Gram {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub values: Buffer<Complex64>,
}

/// `<bra_k | w | ket_k>` over `sphere` for every k-point; `ket = None` is the bra set again (its
/// Bloch sums are then formed once). `weights` is one value per point of the sphere, in its
/// order (quadrature weight times the radial potential), streamed in chunks of
/// [`chunk_size`] points. Only the functions that reach the sphere appear.
pub fn weighted_gram(
    ctx: &Context,
    bra: &Shells,
    ket: Option<&Shells>,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    sphere: &Sphere,
    weights: &[f64],
) -> Result<Gram> {
    weighted_cross(ctx, bra, None, ket, lattice, kpoints, sphere, weights)
}

/// [`weighted_gram`] with the bra functions replaced by their Cartesian derivative along
/// `bra_axis` (`<d_axis chi_a | w | chi_b>`, the row-center derivative of a matrix term with the
/// sphere fixed). A derivative bra needs a ket set of its own.
pub fn weighted_cross(
    ctx: &Context,
    bra: &Shells,
    bra_axis: Option<usize>,
    ket: Option<&Shells>,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    sphere: &Sphere,
    weights: &[f64],
) -> Result<Gram> {
    if weights.len() != sphere.len() || kpoints.is_empty() {
        return Err(Error::Invalid(
            "a sphere matrix needs one weight per point and a k-point".into(),
        ));
    }
    if bra_axis.is_some() && ket.is_none() {
        return Err(Error::Invalid(
            "a derivative bra needs a ket set of its own".into(),
        ));
    }
    let region = sphere.region();
    let left = bra.reaching(ctx, lattice, &region)?;
    let right = match ket {
        Some(set) => Some(set.reaching(ctx, lattice, &region)?),
        None => None,
    };
    let columns = right.as_ref().map_or(left.rows.clone(), |r| r.rows.clone());
    let (nb, nc, nk) = (left.rows.len(), columns.len(), kpoints.len());
    let mut values = ctx.allocate::<Complex64>(product(&[nk, nb, nc])?)?;
    if nb == 0 || nc == 0 {
        return Ok(Gram {
            rows: left.rows,
            columns,
            values,
        });
    }
    let sampling = Sampling {
        lattice,
        region,
        kpoints,
    };
    let size = chunk_size(nk, nb + if right.is_some() { nc } else { 0 });
    for range in sphere.chunks(size) {
        let points = sphere.points(range.clone());
        let n = range.len();
        let mut chi_b = ctx.allocate::<Complex64>(product(&[nk, nb, n])?)?;
        let mut sink = DenseRows::new(chi_b.as_mut_slice(), nb, n);
        match bra_axis {
            None => left.shells.values(ctx, &sampling, points, &mut sink)?,
            Some(axis) => left
                .shells
                .derivatives(ctx, &sampling, points, axis, &mut sink)?,
        }
        let mut chi_c = None;
        if let Some(reached) = &right {
            let mut buffer = ctx.allocate::<Complex64>(product(&[nk, nc, n])?)?;
            reached.shells.values(
                ctx,
                &sampling,
                points,
                &mut DenseRows::new(buffer.as_mut_slice(), nc, n),
            )?;
            chi_c = Some(buffer);
        }
        let ket_rows = chi_c.as_ref().unwrap_or(&chi_b);
        let w = &weights[range];
        for k in 0..nk {
            // Column-major `(points, functions)` views of the row-major `(function, point)`.
            let a = DMatrixView::from_slice(&chi_b.as_slice()[k * nb * n..(k + 1) * nb * n], n, nb);
            let source = &ket_rows.as_slice()[k * nc * n..(k + 1) * nc * n];
            let b = DMatrix::from_fn(n, nc, |x, j| source[j * n + x] * w[x]);
            let block = a.adjoint() * b;
            let target = &mut values.as_mut_slice()[k * nb * nc..(k + 1) * nb * nc];
            for i in 0..nb {
                for j in 0..nc {
                    target[i * nc + j] += block[(i, j)];
                }
            }
        }
    }
    Ok(Gram {
        rows: left.rows,
        columns,
        values,
    })
}
