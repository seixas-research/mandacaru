//! Band paths through the Brillouin zone, their selection on a k-mesh and the irreducible zone
//! of a mesh.
//!
//! A Bloch run of the supercell resolves the commensurate k-points of its mesh only. A band path
//! is therefore a list of vertices with a stated convention ([`convention`]), cut into straight
//! segments, and [`select`] keeps the mesh points (and their reciprocal-lattice images, which
//! carry the same Bloch operator) that lie on a segment, measured in the Cartesian reciprocal
//! metric of the periodic axes. A vertex that is no mesh point is reported as missing; the path
//! is `complete` when none is. The irreducible zone ([`zone`]) is the mesh reduced by the
//! rotations of the crystal's space group that map the mesh onto itself and by time reversal.
//!
//! Lengths are in whatever unit the cell is given; k-space distances come back in its inverse
//! and include the factor `2 pi`.
pub mod convention;
pub mod select;
pub mod zone;

use crate::periodic::invalid;
use crate::Result;
pub use convention::{convention, Convention};
use nalgebra::{DMatrix, Matrix3};
pub use select::{select, PathPoints};
pub use zone::{irreducible_zone, Zone, AUDIT_TOLERANCE};

/// A vertex of a band path: its label (empty for an unnamed point) and its fractional
/// coordinates in the reciprocal basis of the cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    pub label: String,
    pub fractional: [f64; 3],
}

/// The vertices of a path, in order, and where it jumps.
#[derive(Clone, Debug, PartialEq)]
pub struct BandPath {
    /// The lattice class the named points came from, or `"explicit"`.
    pub convention: String,
    pub vertices: Vec<Vertex>,
    /// `jump[i]`: the path jumps from vertex `i` to vertex `i + 1` (a comma in the text); no
    /// segment lies between them and no distance is added.
    pub jump: Vec<bool>,
}

impl BandPath {
    /// The path of `text`, e.g. `"GXMG"` or `"GX,MR"`, over the named points of the lattice's
    /// convention. A label is one capital letter and optional digits; a comma ends a run;
    /// `None` takes the convention's default path.
    pub fn named(lattice: &Matrix3<f64>, periodic: [bool; 3], text: Option<&str>) -> Result<Self> {
        let convention = convention(lattice, periodic)?;
        let axes: Vec<usize> = (0..3).filter(|&a| periodic[a]).collect();
        let text = text.unwrap_or(convention.default_path);
        let mut vertices = vec![];
        let mut jump = vec![];
        let mut start = true;
        let mut chars = text.chars().filter(|c| !c.is_whitespace()).peekable();
        while let Some(c) = chars.next() {
            if c == ',' {
                if start || chars.peek().is_none() {
                    return Err(invalid(format!(
                        "the band path {text:?} has an empty run around a comma"
                    )));
                }
                start = true;
                continue;
            }
            if !c.is_ascii_uppercase() {
                return Err(invalid(format!(
                    "the band path {text:?} holds {c:?}: a label is a capital letter and \
                     optional digits, a comma ends a run"
                )));
            }
            let mut label = c.to_string();
            while let Some(d) = chars.next_if(char::is_ascii_digit) {
                label.push(d);
            }
            let local = convention.point(&label)?;
            let mut fractional = [0.; 3];
            for (component, &axis) in axes.iter().enumerate() {
                fractional[axis] = local[component];
            }
            if !vertices.is_empty() {
                jump.push(start);
            }
            start = false;
            vertices.push(Vertex { label, fractional });
        }
        if vertices.len() < 2 {
            return Err(invalid(format!(
                "the band path {text:?} needs at least two vertices"
            )));
        }
        Ok(Self {
            convention: convention.name.to_string(),
            vertices,
            jump,
        })
    }

    /// A path through explicit fractional points (one run), each with an optional label.
    pub fn explicit(points: &[(String, [f64; 3])]) -> Result<Self> {
        if points.len() < 2 {
            return Err(invalid("a band path needs at least two points"));
        }
        for (_, p) in points {
            crate::finite(p, "band path point")?;
        }
        Ok(Self {
            convention: "explicit".into(),
            vertices: points
                .iter()
                .map(|(label, fractional)| Vertex {
                    label: label.clone(),
                    fractional: *fractional,
                })
                .collect(),
            jump: vec![false; points.len() - 1],
        })
    }
}

/// The reciprocal metric of the periodic axes: `|k|^2 = (2 pi)^2 f^T G^-1 f` for the fractional
/// `f` and the Gram matrix `G` of the periodic cell vectors, zero along a vacuum axis.
#[derive(Clone, Debug)]
pub struct Metric {
    inverse: Matrix3<f64>,
    periodic: [bool; 3],
}

impl Metric {
    /// `lattice` holds the cell vectors as rows.
    pub fn new(lattice: &Matrix3<f64>, periodic: [bool; 3]) -> Result<Self> {
        let axes: Vec<usize> = (0..3).filter(|&a| periodic[a]).collect();
        if axes.is_empty() {
            return Err(invalid("a band path needs a periodic axis"));
        }
        crate::finite(lattice.as_slice(), "lattice")?;
        let gram = DMatrix::from_fn(axes.len(), axes.len(), |i, j| {
            lattice.row(axes[i]).dot(&lattice.row(axes[j]))
        });
        let inverse = gram
            .try_inverse()
            .ok_or_else(|| invalid("the periodic cell vectors must be linearly independent"))?;
        let mut full = Matrix3::zeros();
        for (i, &a) in axes.iter().enumerate() {
            for (j, &b) in axes.iter().enumerate() {
                full[(a, b)] = inverse[(i, j)] * (2. * std::f64::consts::PI).powi(2);
            }
        }
        Ok(Self {
            inverse: full,
            periodic,
        })
    }
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }
    /// The inner product of two fractional vectors.
    pub fn dot(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| a[i] * self.inverse[(i, j)] * b[j])
                    .sum::<f64>()
            })
            .sum()
    }
    pub fn norm(&self, a: [f64; 3]) -> f64 {
        self.dot(a, a).max(0.).sqrt()
    }
}
