//! The compensation shapes of a crystal: the plain inputs the lattice sums take,
//! the transient-memory budget, the radial shape tables `F_L(q)` and the
//! channel transforms `4 pi (-i)^L Y_LM(G) F_L(|G|) exp(-i G.R)`.
use crate::{
    math::{angular, radial::integrate},
    paw::{multipole::shape_function, pseudize_tables::linspace},
    periodic::invalid,
    Context, Reservation, Result,
};
use mandacaru_kernels::two_center::spherical_bessel;
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use std::{collections::BTreeMap, f64::consts::PI};

/// Reciprocal radius (Bohr^-1) of the dense compensation sums.
pub const COMPENSATION_CUTOFF: f64 = 15.;
/// A shape table spans at least this wave number, at this step.
pub const SHAPE_TABLE_MAX: f64 = 120.;
pub const SHAPE_TABLE_STEP: f64 = 0.01;
/// Radial points of the shape transforms and of the on-site integrals.
pub const SHAPE_POINTS: usize = 1201;

/// One atom: what the lattice sums need of its species.
#[derive(Clone, Copy, Debug)]
pub struct Atom {
    pub center: [f64; 3],
    pub charge: f64,
    pub compensation_radius: f64,
}
/// `(atom, L, M)` of one multipole channel.
pub type Channel = (usize, usize, i32);

/// The geometry of one cell: atoms, the channels in `sorted(multipole_blocks)`
/// order, the lattice vectors as COLUMNS (Bohr) and the Gaussian ion width.
#[derive(Clone, Copy)]
pub struct Sites<'a> {
    pub atoms: &'a [Atom],
    pub channels: &'a [Channel],
    pub lattice: Matrix3<f64>,
    pub sigma: f64,
}
impl Sites<'_> {
    pub fn check(&self) -> Result<()> {
        if !self.sigma.is_finite() || self.sigma <= 0. {
            return Err(invalid("the Gaussian ion width must be positive"));
        }
        if self.lattice.iter().any(|x| !x.is_finite()) || self.lattice.determinant().abs() < 1e-12 {
            return Err(invalid("the lattice must be finite and nonsingular"));
        }
        for atom in self.atoms {
            if atom.center.iter().any(|x| !x.is_finite())
                || !atom.charge.is_finite()
                || !(atom.compensation_radius.is_finite() && atom.compensation_radius > 0.)
            {
                return Err(invalid(
                    "atoms need finite centers and a positive compensation radius",
                ));
            }
        }
        // The Python crystal keeps the channels sorted; every consumer relies on it.
        if self.channels.windows(2).any(|w| w[0] >= w[1])
            || self
                .channels
                .iter()
                .any(|c| c.0 >= self.atoms.len() || c.2.unsigned_abs() as usize > c.1)
        {
            return Err(invalid(
                "channels must be distinct, sorted (atom, L, M) of known atoms",
            ));
        }
        Ok(())
    }
    pub fn volume(&self) -> f64 {
        self.lattice.determinant().abs()
    }
    /// Reciprocal vectors as columns (`2 pi A^-T`).
    pub fn reciprocal(&self) -> Result<Matrix3<f64>> {
        Ok(self
            .lattice
            .try_inverse()
            .ok_or_else(|| invalid("singular lattice"))?
            .transpose()
            * (2. * PI))
    }
}

/// The setup's transient memory: `min(limit / 16, 20 MiB)`, split into the
/// dense-sum block (8 of 20), the radial-transform chunk (8 of 20) and scratch.
#[derive(Clone, Copy, Debug)]
pub struct Transient {
    pub dense_bytes: usize,
    pub radial_bytes: usize,
}
impl Transient {
    pub fn of(ctx: &Context) -> Self {
        let total = (ctx.memory_limit() / 16).min(20 << 20);
        Self {
            dense_bytes: total * 2 / 5,
            radial_bytes: total * 2 / 5,
        }
    }
}

/// Every lattice vector `n . columns` of norm `<= radius` (`G = 0` kept unless
/// `skip_zero`), in blocks of at most `block` vectors, `n1` slowest.
pub fn visit_lattice(
    columns: &Matrix3<f64>,
    radius: f64,
    skip_zero: bool,
    block: usize,
    visit: &mut dyn FnMut(&[[f64; 3]]) -> Result<()>,
) -> Result<()> {
    let dual = columns
        .try_inverse()
        .ok_or_else(|| invalid("singular lattice"))?;
    let mut reach = [0i64; 3];
    for (a, r) in reach.iter_mut().enumerate() {
        let count = (radius * dual.row(a).norm()).ceil();
        if !count.is_finite() || count > 1e6 {
            return Err(crate::Error::Overflow);
        }
        *r = count as i64;
    }
    let mut pending: Vec<[f64; 3]> = Vec::with_capacity(block.min(1 << 16));
    for i in -reach[0]..=reach[0] {
        for j in -reach[1]..=reach[1] {
            for k in -reach[2]..=reach[2] {
                let v = columns * Vector3::new(i as f64, j as f64, k as f64);
                if v.norm() > radius + 1e-12 || (skip_zero && v.norm_squared() == 0.) {
                    continue;
                }
                pending.push(v.into());
                if pending.len() == block {
                    visit(&pending)?;
                    pending.clear();
                }
            }
        }
    }
    if !pending.is_empty() {
        visit(&pending)?;
    }
    Ok(())
}

/// `F_L(q) = int_0^inf f(r) j_L(q r) r^2 dr` by the trapezoid rule on the table
/// `r`, for every `q`, `chunk` rows of the Bessel matrix at a time. The chunk is
/// reserved from the context: a limit too small for one row raises `Budget`.
pub fn radial_transform(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
    q: &[f64],
    chunk_bytes: usize,
) -> Result<Vec<f64>> {
    let rows = (chunk_bytes / (8 * r.len())).clamp(1, q.len().max(1));
    let mut scratch = ctx.allocate::<f64>(rows * r.len())?;
    let weighted: Vec<f64> = r.iter().zip(values).map(|(r, v)| v * r * r).collect();
    let mut out = Vec::with_capacity(q.len());
    for chunk in q.chunks(rows) {
        for (row, &x) in scratch.as_mut_slice().chunks_mut(r.len()).zip(chunk) {
            for ((cell, &radius), &w) in row.iter_mut().zip(r).zip(&weighted) {
                *cell = spherical_bessel(l as i32, x * radius) * w;
            }
        }
        for row in scratch.as_slice().chunks(r.len()).take(chunk.len()) {
            out.push(integrate(r, row)?);
        }
    }
    Ok(out)
}

/// `F_L(q)` of one shape on a uniform table, linearly interpolated.
pub struct ShapeTable {
    pub q: Vec<f64>,
    pub f: Vec<f64>,
}
impl ShapeTable {
    /// `np.interp`: clamped outside the table.
    pub fn at(&self, x: f64) -> f64 {
        let n = self.q.len();
        if x <= self.q[0] {
            return self.f[0];
        }
        if x >= self.q[n - 1] {
            return self.f[n - 1];
        }
        let j = self.q.partition_point(|v| *v <= x) - 1;
        let slope = (self.f[j + 1] - self.f[j]) / (self.q[j + 1] - self.q[j]);
        slope * (x - self.q[j]) + self.f[j]
    }
    /// `d at / dx`: the slope of the linear piece [`ShapeTable::at`] reads, the right-hand one
    /// on a node and zero outside the table. Piecewise constant, discontinuous at every node:
    /// the derivative of the interpolant, not of the radial transform it tabulates.
    pub fn slope(&self, x: f64) -> f64 {
        let n = self.q.len();
        if x <= self.q[0] || x >= self.q[n - 1] {
            return 0.;
        }
        let j = self.q.partition_point(|v| *v <= x) - 1;
        (self.f[j + 1] - self.f[j]) / (self.q[j + 1] - self.q[j])
    }
}

/// The shape tables of a crystal's channels, built once per (radius, L).
pub struct Shapes {
    tables: BTreeMap<(i64, usize), ShapeTable>,
    top: f64,
    _storage: Reservation,
}
fn key(radius: f64, l: usize) -> (i64, usize) {
    ((radius * 1e12).round() as i64, l)
}
impl Shapes {
    /// Reuse precomputed shape tables, preserving their exact interpolation nodes.
    pub fn from_tables(
        ctx: &Context,
        sites: &Sites,
        source: Vec<(f64, usize, Vec<f64>, Vec<f64>)>,
    ) -> Result<Self> {
        sites.check()?;
        let mut tables = BTreeMap::new();
        let mut bytes = 0usize;
        let mut top = f64::INFINITY;
        for (radius, l, q, f) in source {
            if !(radius.is_finite() && radius > 0.)
                || q.len() < 2
                || q.len() != f.len()
                || q.iter().chain(&f).any(|v| !v.is_finite())
                || q.windows(2).any(|v| v[0] >= v[1])
                || q[0] > 0.
            {
                return Err(invalid("invalid compensation shape table"));
            }
            bytes = bytes
                .checked_add(crate::product(&[q.len(), 16])?)
                .ok_or(crate::Error::Overflow)?;
            top = top.min(*q.last().unwrap());
            if tables.insert(key(radius, l), ShapeTable { q, f }).is_some() {
                return Err(invalid("duplicate compensation shape table"));
            }
        }
        for &(atom, l, _) in sites.channels {
            if !tables.contains_key(&key(sites.atoms[atom].compensation_radius, l)) {
                return Err(invalid("missing compensation shape table"));
            }
        }
        if tables.is_empty() {
            top = SHAPE_TABLE_MAX;
        }
        let storage = ctx.reserve(bytes)?;
        Ok(Self {
            tables,
            top,
            _storage: storage,
        })
    }
    /// Tables on `[0, max(q_max, 15, 120)]` at step 0.01 for every distinct
    /// (radius, L) of the channels.
    pub fn new(ctx: &Context, sites: &Sites, q_max: f64) -> Result<Self> {
        sites.check()?;
        let top = q_max.max(COMPENSATION_CUTOFF).max(SHAPE_TABLE_MAX);
        let points = (top / SHAPE_TABLE_STEP) as usize + 1;
        let q = linspace(0., top, points);
        let mut tables = BTreeMap::new();
        let mut distinct: Vec<(f64, usize)> = Vec::new();
        for &(atom, l, _) in sites.channels {
            let radius = sites.atoms[atom].compensation_radius;
            if !distinct.iter().any(|d| key(d.0, d.1) == key(radius, l)) {
                distinct.push((radius, l));
            }
        }
        let storage = ctx.reserve(crate::product(&[distinct.len(), points, 16])?)?;
        for (radius, l) in distinct {
            let r = linspace(0., radius, SHAPE_POINTS);
            let shape = shape_function(&r, radius, l)?;
            let f = radial_transform(ctx, &r, &shape, l, &q, Transient::of(ctx).radial_bytes)?;
            tables.insert(key(radius, l), ShapeTable { q: q.clone(), f });
        }
        Ok(Self {
            tables,
            top,
            _storage: storage,
        })
    }
    /// Largest wave number the tables cover; a transform beyond it raises.
    pub fn top(&self) -> f64 {
        self.top
    }
    pub fn table(&self, radius: f64, l: usize) -> Result<&ShapeTable> {
        self.tables
            .get(&key(radius, l))
            .ok_or_else(|| invalid("no shape table for this (radius, L)"))
    }
    /// `8 int_cutoff^inf F_L(q)^2 dq` on the table, trapezoid: the on-site
    /// self-energy of a unit `g_L Y_LM` beyond the dense sum's cutoff.
    pub fn tail(&self, radius: f64, l: usize, cutoff: f64) -> Result<f64> {
        let table = self.table(radius, l)?;
        let from = table.q.partition_point(|v| *v < cutoff);
        let squares: Vec<f64> = table.f[from..].iter().map(|f| f * f).collect();
        Ok(8. * integrate(&table.q[from..], &squares)?)
    }

    /// Every channel's compensation shape at `vectors`:
    /// `out[row * stride + v] = 4 pi (-i)^L Y_LM(G^) F_L(|G|) exp(-i G.R_atom)`.
    pub fn transforms(
        &self,
        sites: &Sites,
        vectors: &[[f64; 3]],
        out: &mut [Complex64],
        stride: usize,
    ) -> Result<()> {
        self.evaluate(sites, vectors, out, stride, false)
    }

    /// The gradient of [`Shapes::transforms`] with respect to the wave vector at a FIXED atom
    /// phase: `out[(row * 3 + b) * stride + v] = exp(-i G.R) d/dG_b [4 pi (-i)^L Y_LM(G^) F_L(|G|)]`,
    /// the radial part through [`ShapeTable::slope`] (the interpolant the shapes use, not the
    /// transform it tabulates). Zero at `G = 0`, where `L > 0` shapes have a direction-dependent
    /// slope that the strain derivative `-G_a d_b` multiplies by zero.
    ///
    /// Under the strain `x -> (1 + eps) x`, `G -> (1 - eps^T) G` and `G.R` is invariant, so the
    /// strain derivative of a shape is `d shape / d eps_ab = -G_a out[b]`.
    pub fn transform_gradients(
        &self,
        sites: &Sites,
        vectors: &[[f64; 3]],
        out: &mut [Complex64],
        stride: usize,
    ) -> Result<()> {
        self.evaluate(sites, vectors, out, stride, true)
    }

    fn evaluate(
        &self,
        sites: &Sites,
        vectors: &[[f64; 3]],
        out: &mut [Complex64],
        stride: usize,
        gradient: bool,
    ) -> Result<()> {
        let channels = sites.channels;
        let mut radial = Vec::new();
        let mut radial_keys: Vec<(i64, usize)> = Vec::new();
        for &(atom, l, _) in channels {
            let k = key(sites.atoms[atom].compensation_radius, l);
            if !radial_keys.contains(&k) {
                radial_keys.push(k);
                radial.push(self.table(sites.atoms[atom].compensation_radius, l)?);
            }
        }
        let mut lm: Vec<(usize, i32)> = Vec::new();
        for &(_, l, m) in channels {
            if !lm.contains(&(l, m)) {
                lm.push((l, m));
            }
        }
        let lm_of: Vec<usize> = channels
            .iter()
            .map(|&(_, l, m)| lm.iter().position(|e| *e == (l, m)).unwrap())
            .collect();
        let shape_of: Vec<usize> = channels
            .iter()
            .map(|&(atom, l, _)| {
                let k = key(sites.atoms[atom].compensation_radius, l);
                radial_keys.iter().position(|e| *e == k).unwrap()
            })
            .collect();
        let (mut harmonic, mut shape, mut phase) = (
            vec![Complex64::default(); lm.len()],
            vec![0.; radial.len()],
            vec![Complex64::default(); sites.atoms.len()],
        );
        let (mut harmonic_slope, mut shape_slope) = (
            vec![[Complex64::default(); 3]; lm.len()],
            vec![0.; radial.len()],
        );
        for (v, g) in vectors.iter().enumerate() {
            let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if norm > self.top {
                return Err(invalid(format!(
                    "wave number {norm} is beyond the shape tables ({}); build them with a larger q_max",
                    self.top
                )));
            }
            // The zero vector reads as +x, as the spherical coordinates of the
            // oracle do; every L > 0 shape vanishes there.
            let direction = if norm > 0. {
                [g[0] / norm, g[1] / norm, g[2] / norm]
            } else {
                [1., 0., 0.]
            };
            for (value, &(l, m)) in harmonic.iter_mut().zip(&lm) {
                *value = angular::complex(l, m, direction);
            }
            for (value, table) in shape.iter_mut().zip(&radial) {
                *value = table.at(norm);
            }
            if gradient {
                for (value, &(l, m)) in harmonic_slope.iter_mut().zip(&lm) {
                    *value = angular::complex_gradient(l, m, *g);
                }
                for (value, table) in shape_slope.iter_mut().zip(&radial) {
                    *value = table.slope(norm);
                }
            }
            for (value, atom) in phase.iter_mut().zip(sites.atoms) {
                let angle =
                    -(g[0] * atom.center[0] + g[1] * atom.center[1] + g[2] * atom.center[2]);
                *value = Complex64::new(angle.cos(), angle.sin());
            }
            for (row, &(atom, l, _)) in channels.iter().enumerate() {
                let (h, s) = (harmonic[lm_of[row]], shape[shape_of[row]]);
                let unit = [
                    Complex64::new(1., 0.),
                    Complex64::new(0., -1.),
                    Complex64::new(-1., 0.),
                    Complex64::new(0., 1.),
                ][l % 4];
                if !gradient {
                    out[row * stride + v] = unit * h * phase[atom] * (4. * PI * s);
                    continue;
                }
                for b in 0..3 {
                    // d(F Y)/dG_b = F' G^_b Y + F dY/dG_b; the direction gradient carries 1/|G|.
                    let radial_part = if norm > 0. {
                        h * (shape_slope[shape_of[row]] * g[b] / norm)
                    } else {
                        Complex64::default()
                    };
                    let angular_part = harmonic_slope[lm_of[row]][b] * s;
                    out[(row * 3 + b) * stride + v] =
                        unit * (radial_part + angular_part) * phase[atom] * (4. * PI);
                }
            }
        }
        Ok(())
    }
}
