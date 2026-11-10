//! The linear tetrahedron method for the density of states on a uniform k mesh.
//!
//! Each parallelepiped of the Gamma-centered `n1 x n2 x n3` mesh is split into six tetrahedra
//! that share its shortest main diagonal in Cartesian reciprocal space (Bloechl, Jepsen and
//! Andersen, 1994). Inside a tetrahedron every band is the linear interpolation of its four
//! corner energies, and the density of states of a linear band is closed-form: piecewise
//! quadratic in `E` with kinks at the corner energies (Lehmann and Taut, 1972). No broadening
//! enters: the method converges as the mesh refines, with an error of second order in the
//! mesh step for a smooth band.
//!
//! A projected density of states interpolates each band's projection linearly as well; its
//! value at `E` is the tetrahedron's density times the projection averaged over the cross-section
//! of the tetrahedron at energy `E` (a triangle or a quadrilateral). The average of a linear
//! function over a planar polygon is its value at the polygon's centroid, and the ratio of
//! two coplanar areas survives the affine map onto a reference tetrahedron, so the average is
//! computed there. Projections that sum to one per band give a projected density that sums to
//! the total exactly.
//!
//! Units are those of the energies. Values are states per energy per cell per band channel; the
//! caller multiplies by the spin degeneracy.
use super::invalid;
use crate::{finite, product, Result};
use nalgebra::{Matrix3, Vector3};

/// A Gamma-centered k mesh split into tetrahedra.
#[derive(Clone, Debug)]
pub struct TetrahedronMesh {
    pub shape: [usize; 3],
    /// Mesh indices (first axis slowest) of the corners of every tetrahedron.
    pub tetrahedra: Vec<[usize; 4]>,
}

impl TetrahedronMesh {
    /// `reciprocal`: the reciprocal lattice vectors as columns (any common scale).
    pub fn new(shape: [usize; 3], reciprocal: &Matrix3<f64>) -> Result<Self> {
        if shape.contains(&0) {
            return Err(invalid("every mesh size must be positive"));
        }
        finite(reciprocal.as_slice(), "reciprocal lattice")?;
        if reciprocal.determinant().abs() <= 1e-14 {
            return Err(invalid("the reciprocal lattice is singular"));
        }
        let cells = product(&shape)?;
        let step = reciprocal
            * Matrix3::from_diagonal(&Vector3::new(
                1. / shape[0] as f64,
                1. / shape[1] as f64,
                1. / shape[2] as f64,
            ));
        // The four main diagonals of a subcell, by their starting corner.
        let starts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let length = |s: &[i64; 3]| {
            let d = Vector3::new(
                (1 - 2 * s[0]) as f64,
                (1 - 2 * s[1]) as f64,
                (1 - 2 * s[2]) as f64,
            );
            (step * d).norm()
        };
        let start = *starts
            .iter()
            .min_by(|a, b| length(a).total_cmp(&length(b)))
            .expect("four diagonals");
        let direction = start.map(|s| 1 - 2 * s);
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let index = |c: [i64; 3]| -> usize {
            let w: [usize; 3] = std::array::from_fn(|a| c[a].rem_euclid(shape[a] as i64) as usize);
            (w[0] * shape[1] + w[1]) * shape[2] + w[2]
        };
        let mut tetrahedra = Vec::with_capacity(cells * 6);
        for i in 0..shape[0] as i64 {
            for j in 0..shape[1] as i64 {
                for k in 0..shape[2] as i64 {
                    let origin = [i + start[0], j + start[1], k + start[2]];
                    for order in &permutations {
                        let mut corner = origin;
                        let mut tetrahedron = [index(corner); 4];
                        for (slot, &axis) in order.iter().enumerate() {
                            corner[axis] += direction[axis];
                            tetrahedron[slot + 1] = index(corner);
                        }
                        tetrahedra.push(tetrahedron);
                    }
                }
            }
        }
        Ok(Self { shape, tetrahedra })
    }
    pub fn points(&self) -> usize {
        self.shape.iter().product()
    }
    /// The mesh index of every fractional k-point; together they must be the whole mesh,
    /// each point once (any order, any reciprocal-lattice image).
    pub fn indices(&self, kpoints: &[[f64; 3]]) -> Result<Vec<usize>> {
        let n = self.points();
        if kpoints.len() != n {
            return Err(invalid(format!(
                "the tetrahedron method needs the full {}x{}x{} Gamma-centered mesh: got {} k-points for {} mesh points",
                self.shape[0], self.shape[1], self.shape[2], kpoints.len(), n
            )));
        }
        let mut seen = vec![false; n];
        let mut out = Vec::with_capacity(n);
        for k in kpoints {
            finite(k, "k-point")?;
            let mut c = [0usize; 3];
            for a in 0..3 {
                let x = k[a] * self.shape[a] as f64;
                if (x - x.round()).abs() > 1e-6 {
                    return Err(invalid(
                        "a k-point is not on the Gamma-centered mesh: the tetrahedron method needs a Gamma-centered mesh",
                    ));
                }
                c[a] = (x.round() as i64).rem_euclid(self.shape[a] as i64) as usize;
            }
            let i = (c[0] * self.shape[1] + c[1]) * self.shape[2] + c[2];
            if seen[i] {
                return Err(invalid("a mesh point appears twice"));
            }
            seen[i] = true;
            out.push(i);
        }
        Ok(out)
    }
    /// Density of states and integrated density (states per cell, per channel) on `axis`.
    /// `eigenvalues` is `(points, bands)` in mesh order.
    pub fn dos(
        &self,
        eigenvalues: &[f64],
        bands: usize,
        axis: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check(eigenvalues, bands, axis)?;
        let weight = 1. / self.tetrahedra.len() as f64;
        let mut density = vec![0.; axis.len()];
        let mut integrated = vec![0.; axis.len()];
        for tetrahedron in &self.tetrahedra {
            for band in 0..bands {
                let mut e = tetrahedron.map(|i| eigenvalues[i * bands + band]);
                e.sort_by(f64::total_cmp);
                for (x, (d, n)) in axis
                    .iter()
                    .zip(density.iter_mut().zip(integrated.iter_mut()))
                {
                    *d += weight * corner_density(&e, *x);
                    *n += weight * corner_count(&e, *x);
                }
            }
        }
        Ok((density, integrated))
    }
    /// Projected density of states on `axis`, `(axis, channels)`; `projections` is
    /// `(points, bands, channels)` in mesh order.
    pub fn projected_dos(
        &self,
        eigenvalues: &[f64],
        bands: usize,
        projections: &[f64],
        channels: usize,
        axis: &[f64],
    ) -> Result<Vec<f64>> {
        self.check(eigenvalues, bands, axis)?;
        if projections.len() != product(&[self.points(), bands, channels])? {
            return Err(invalid("projections must be (points, bands, channels)"));
        }
        finite(projections, "projections")?;
        let weight = 1. / self.tetrahedra.len() as f64;
        let mut out = vec![0.; product(&[axis.len(), channels])?];
        let mut values = vec![[0.; 4]; channels];
        for tetrahedron in &self.tetrahedra {
            for band in 0..bands {
                let mut order = [0usize, 1, 2, 3];
                let raw = tetrahedron.map(|i| eigenvalues[i * bands + band]);
                order.sort_by(|&a, &b| raw[a].total_cmp(&raw[b]));
                let e = order.map(|c| raw[c]);
                for (c, value) in values.iter_mut().enumerate() {
                    *value = order.map(|corner| {
                        projections[(tetrahedron[corner] * bands + band) * channels + c]
                    });
                }
                for (x_index, &x) in axis.iter().enumerate() {
                    let d = corner_density(&e, x);
                    if d == 0. {
                        continue;
                    }
                    let barycenter = section_barycenter(&e, x);
                    for (c, value) in values.iter().enumerate() {
                        let average: f64 = (0..4).map(|v| barycenter[v] * value[v]).sum();
                        out[x_index * channels + c] += weight * d * average;
                    }
                }
            }
        }
        Ok(out)
    }
    /// The density of states summed over spin channels, each channel times `degeneracy` (2 for
    /// an unpolarized calculation). `eigenvalues` is `(spins, kpoints, bands)` at `kpoints` in any
    /// order (they must be the whole mesh); `projections`, with its channel count, is
    /// `(spins, kpoints, bands, channels)`. Returns the density, the integrated density and, with
    /// projections, the projected density `(axis, channels)`.
    #[allow(clippy::type_complexity)]
    pub fn spin_summed(
        &self,
        kpoints: &[[f64; 3]],
        eigenvalues: &[f64],
        spins: usize,
        bands: usize,
        projections: Option<(&[f64], usize)>,
        degeneracy: f64,
        axis: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Option<Vec<f64>>)> {
        if !degeneracy.is_finite() || degeneracy <= 0. {
            return Err(invalid("degeneracy must be positive and finite"));
        }
        let order = self.indices(kpoints)?;
        let nk = order.len();
        if spins == 0 || eigenvalues.len() != product(&[spins, nk, bands])? {
            return Err(invalid("eigenvalues must be (spins, kpoints, bands)"));
        }
        if let Some((weights, channels)) = projections {
            if weights.len() != product(&[spins, nk, bands, channels])? {
                return Err(invalid("projections must be (spins, kpoints, bands, channels)"));
            }
        }
        let mut dos = vec![0.; axis.len()];
        let mut integrated = vec![0.; axis.len()];
        let mut projected: Option<Vec<f64>> = None;
        for s in 0..spins {
            let mut sorted = vec![0.; nk * bands];
            for (k, &target) in order.iter().enumerate() {
                sorted[target * bands..(target + 1) * bands]
                    .copy_from_slice(&eigenvalues[(s * nk + k) * bands..(s * nk + k + 1) * bands]);
            }
            let (d, n) = self.dos(&sorted, bands, axis)?;
            for i in 0..axis.len() {
                dos[i] += degeneracy * d[i];
                integrated[i] += degeneracy * n[i];
            }
            if let Some((weights, channels)) = projections {
                let span = bands * channels;
                let mut sorted_weights = vec![0.; nk * span];
                for (k, &target) in order.iter().enumerate() {
                    sorted_weights[target * span..(target + 1) * span]
                        .copy_from_slice(&weights[(s * nk + k) * span..(s * nk + k + 1) * span]);
                }
                let p = self.projected_dos(&sorted, bands, &sorted_weights, channels, axis)?;
                let total = projected.get_or_insert_with(|| vec![0.; p.len()]);
                for (t, v) in total.iter_mut().zip(&p) {
                    *t += degeneracy * v;
                }
            }
        }
        Ok((dos, integrated, projected))
    }
    fn check(&self, eigenvalues: &[f64], bands: usize, axis: &[f64]) -> Result<()> {
        if bands == 0 || eigenvalues.len() != product(&[self.points(), bands])? {
            return Err(invalid(
                "eigenvalues must be (points, bands) on the whole mesh",
            ));
        }
        finite(eigenvalues, "eigenvalues")?;
        finite(axis, "energy axis")
    }
}

/// Density of one linear band in a tetrahedron of unit volume fraction, sorted corners `e`.
fn corner_density(e: &[f64; 4], x: f64) -> f64 {
    let [e1, e2, e3, e4] = *e;
    if x <= e1 || x >= e4 {
        0.
    } else if x < e2 {
        3. * (x - e1).powi(2) / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    } else if x < e3 {
        (3. * (e2 - e1) + 6. * (x - e2)
            - 3. * (e3 - e1 + e4 - e2) * (x - e2).powi(2) / ((e3 - e2) * (e4 - e2)))
            / ((e3 - e1) * (e4 - e1))
    } else {
        3. * (e4 - x).powi(2) / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    }
}

/// Fraction of a tetrahedron below `x`.
fn corner_count(e: &[f64; 4], x: f64) -> f64 {
    let [e1, e2, e3, e4] = *e;
    if x <= e1 {
        0.
    } else if x >= e4 {
        1.
    } else if x < e2 {
        (x - e1).powi(3) / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    } else if x < e3 {
        let d = x - e2;
        ((e2 - e1).powi(2) + 3. * (e2 - e1) * d + 3. * d * d
            - (e3 - e1 + e4 - e2) * d.powi(3) / ((e3 - e2) * (e4 - e2)))
            / ((e3 - e1) * (e4 - e1))
    } else {
        1. - (e4 - x).powi(3) / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    }
}

/// Barycentric coordinates (on the sorted corners) of the centroid of the cross-section of the
/// tetrahedron at energy `x`, with `e1 < x < e4`.
fn section_barycenter(e: &[f64; 4], x: f64) -> [f64; 4] {
    // A point on edge (a, b) at the crossing, in barycentric coordinates.
    let cut = |a: usize, b: usize| -> [f64; 4] {
        let t = (x - e[a]) / (e[b] - e[a]);
        let mut p = [0.; 4];
        p[a] = 1. - t;
        p[b] += t;
        p
    };
    let centroid = |points: &[[f64; 4]]| -> [f64; 4] {
        let n = points.len() as f64;
        std::array::from_fn(|v| points.iter().map(|p| p[v]).sum::<f64>() / n)
    };
    if x < e[1] {
        centroid(&[cut(0, 1), cut(0, 2), cut(0, 3)])
    } else if x >= e[2] {
        centroid(&[cut(0, 3), cut(1, 3), cut(2, 3)])
    } else {
        // Quadrilateral 02-03-13-12, split along 02-13; areas in the reference tetrahedron.
        let (p, q, r, s) = (cut(0, 2), cut(0, 3), cut(1, 3), cut(1, 2));
        let embed = |b: [f64; 4]| Vector3::new(b[1], b[2], b[3]);
        let area = |a: [f64; 4], b: [f64; 4], c: [f64; 4]| {
            (embed(b) - embed(a)).cross(&(embed(c) - embed(a))).norm()
        };
        let (first, second) = (area(p, q, r), area(p, r, s));
        let total = first + second;
        if total <= 0. {
            return centroid(&[p, q, r, s]);
        }
        let c1 = centroid(&[p, q, r]);
        let c2 = centroid(&[p, r, s]);
        std::array::from_fn(|v| (first * c1[v] + second * c2[v]) / total)
    }
}
