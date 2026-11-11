//! The named high-symmetry points of a lattice and the default band path through them.
//!
//! The lattice class is read from the Gram matrix of the periodic primitive vectors alone, so
//! the vacuum axes of a slab or a chain play no part, and the fractional coordinates of every
//! point refer to the periodic axes in the order the cell lists them. The conventions are the
//! standard ones of Setyawan and Curtarolo (Comput. Mater. Sci. 49, 299 (2010)) where that paper
//! defines the lattice, and the obvious extensions to one and two dimensions elsewhere:
//!
//! | lattice | points (fractional, reciprocal basis) | default path |
//! | --- | --- | --- |
//! | chain | G 0, X 1/2 | `GX` |
//! | square | G 0, X (1/2, 0), M (1/2, 1/2) | `GXMG` |
//! | rectangular | G 0, X (1/2, 0), Y (0, 1/2), S (1/2, 1/2) | `GXSYG` |
//! | hexagonal 2D, gamma 120 | G 0, M (1/2, 0), K (1/3, 1/3) | `GMKG` |
//! | hexagonal 2D, gamma 60 | G 0, M (1/2, 0), K (1/3, -1/3) | `GMKG` |
//! | cubic (a1, a2, a3 orthogonal, equal) | G, X (0, 1/2, 0), M (1/2, 1/2, 0), R (1/2, 1/2, 1/2) | `GXMGRX,MR` |
//! | fcc (a1 = a(0,1,1)/2, a2 = a(1,0,1)/2, a3 = a(1,1,0)/2) | G, K (3/8, 3/8, 3/4), L (1/2, 1/2, 1/2), U (5/8, 1/4, 5/8), W (1/2, 1/4, 3/4), X (1/2, 0, 1/2) | `GXWKGLUWLK,UX` |
//! | bcc (a1 = a(-1,1,1)/2, a2 = a(1,-1,1)/2, a3 = a(1,1,-1)/2) | G, H (1/2, -1/2, 1/2), N (0, 0, 1/2), P (1/4, 1/4, 1/4) | `GHNGPH,PN` |
//! | tetragonal (a = b, c orthogonal) | G, A (1/2, 1/2, 1/2), M (1/2, 1/2, 0), R (0, 1/2, 1/2), X (0, 1/2, 0), Z (0, 0, 1/2) | `GXMGZRAZ,XR,MA` |
//! | orthorhombic (orthogonal axes, lengths listed a < b < c, so X lies along the shortest) | G, R (1/2, 1/2, 1/2), S (1/2, 1/2, 0), T (0, 1/2, 1/2), U (1/2, 0, 1/2), X (1/2, 0, 0), Y (0, 1/2, 0), Z (0, 0, 1/2) | `GXSYGZURTZ,YT,UX,SR` |
//! | hexagonal 3D (a = b, gamma 120, c orthogonal) | G, A (0, 0, 1/2), H (1/3, 1/3, 1/2), K (1/3, 1/3, 0), L (1/2, 0, 1/2), M (1/2, 0, 0) | `GMKGALHA,LM,KH` |
//!
//! `G` is the label of the zone center. A comma ends a continuous run: the path jumps from the
//! last point before it to the first point after it. Any other lattice (oblique, centered
//! rectangular, rhombohedral, monoclinic, triclinic, base-centered or a cell listed in another
//! orientation) raises naming its lengths and angles: pass explicit points instead.
use crate::periodic::invalid;
use crate::Result;
use nalgebra::Matrix3;

/// The lattice lengths and the cosines agree to this relative tolerance.
const TOLERANCE: f64 = 1e-5;

/// A lattice class: the named points and the default path.
#[derive(Clone, Debug, PartialEq)]
pub struct Convention {
    pub name: &'static str,
    pub points: Vec<(&'static str, [f64; 3])>,
    pub default_path: &'static str,
}

impl Convention {
    /// The fractional point named `label` (an index per periodic axis, in order).
    pub fn point(&self, label: &str) -> Result<[f64; 3]> {
        self.points
            .iter()
            .find(|(name, _)| *name == label)
            .map(|(_, p)| *p)
            .ok_or_else(|| {
                let known: Vec<&str> = self.points.iter().map(|(name, _)| *name).collect();
                invalid(format!(
                    "the {} lattice has no high-symmetry point {label:?}; its points are {known:?}",
                    self.name
                ))
            })
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOLERANCE * a.abs().max(b.abs()).max(1e-300)
}
fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOLERANCE
}

fn table(
    name: &'static str,
    points: &[(&'static str, [f64; 3])],
    default_path: &'static str,
) -> Convention {
    Convention {
        name,
        points: points.to_vec(),
        default_path,
    }
}

/// The convention of the lattice spanned by the `periodic` rows of `lattice` (the cell vectors as
/// rows). The coordinates of every point are returned over the periodic axes in order, padded
/// with zeros.
pub fn convention(lattice: &Matrix3<f64>, periodic: [bool; 3]) -> Result<Convention> {
    let axes: Vec<usize> = (0..3).filter(|&a| periodic[a]).collect();
    let gram = |i: usize, j: usize| lattice.row(axes[i]).dot(&lattice.row(axes[j]));
    let length = |i: usize| gram(i, i).sqrt();
    let cosine = |i: usize, j: usize| gram(i, j) / (length(i) * length(j));
    let g = "G";
    match axes.len() {
        0 => Err(invalid("a band path needs a periodic axis")),
        1 => Ok(table(
            "chain",
            &[(g, [0., 0., 0.]), ("X", [0.5, 0., 0.])],
            "GX",
        )),
        2 => {
            let (a, b, c) = (length(0), length(1), cosine(0, 1));
            if near(c, 0.) && close(a, b) {
                Ok(table(
                    "square",
                    &[(g, [0.; 3]), ("X", [0.5, 0., 0.]), ("M", [0.5, 0.5, 0.])],
                    "GXMG",
                ))
            } else if near(c, 0.) {
                Ok(table(
                    "rectangular",
                    &[
                        (g, [0.; 3]),
                        ("X", [0.5, 0., 0.]),
                        ("Y", [0., 0.5, 0.]),
                        ("S", [0.5, 0.5, 0.]),
                    ],
                    "GXSYG",
                ))
            } else if close(a, b) && near(c, -0.5) {
                Ok(table(
                    "hexagonal-2d",
                    &[
                        (g, [0.; 3]),
                        ("M", [0.5, 0., 0.]),
                        ("K", [1. / 3., 1. / 3., 0.]),
                    ],
                    "GMKG",
                ))
            } else if close(a, b) && near(c, 0.5) {
                Ok(table(
                    "hexagonal-2d",
                    &[
                        (g, [0.; 3]),
                        ("M", [0.5, 0., 0.]),
                        ("K", [1. / 3., -1. / 3., 0.]),
                    ],
                    "GMKG",
                ))
            } else {
                Err(refusal("two-dimensional", &[a, b], &[c]))
            }
        }
        _ => {
            let l = [length(0), length(1), length(2)];
            let cos = [cosine(0, 1), cosine(0, 2), cosine(1, 2)];
            let orthogonal = cos.iter().all(|&c| near(c, 0.));
            let equal = close(l[0], l[1]) && close(l[1], l[2]);
            if orthogonal && equal {
                Ok(table(
                    "cubic",
                    &[
                        (g, [0.; 3]),
                        ("X", [0., 0.5, 0.]),
                        ("M", [0.5, 0.5, 0.]),
                        ("R", [0.5, 0.5, 0.5]),
                    ],
                    "GXMGRX,MR",
                ))
            } else if orthogonal && close(l[0], l[1]) {
                Ok(table(
                    "tetragonal",
                    &[
                        (g, [0.; 3]),
                        ("A", [0.5, 0.5, 0.5]),
                        ("M", [0.5, 0.5, 0.]),
                        ("R", [0., 0.5, 0.5]),
                        ("X", [0., 0.5, 0.]),
                        ("Z", [0., 0., 0.5]),
                    ],
                    "GXMGZRAZ,XR,MA",
                ))
            } else if orthogonal && (close(l[1], l[2]) || close(l[0], l[2])) {
                Err(invalid(
                    "a tetragonal cell must list its two equal lengths first (a = b, c unique): \
                     reorder the cell vectors, or pass explicit path points",
                ))
            } else if orthogonal && !(l[0] < l[1] && l[1] < l[2]) {
                Err(invalid(
                    "an orthorhombic cell must list its lengths in increasing order (a < b < c, so \
                     the convention's X lies along the shortest axis): reorder the cell vectors, or \
                     pass explicit path points",
                ))
            } else if orthogonal {
                Ok(table(
                    "orthorhombic",
                    &[
                        (g, [0.; 3]),
                        ("R", [0.5, 0.5, 0.5]),
                        ("S", [0.5, 0.5, 0.]),
                        ("T", [0., 0.5, 0.5]),
                        ("U", [0.5, 0., 0.5]),
                        ("X", [0.5, 0., 0.]),
                        ("Y", [0., 0.5, 0.]),
                        ("Z", [0., 0., 0.5]),
                    ],
                    "GXSYGZURTZ,YT,UX,SR",
                ))
            } else if equal && cos.iter().all(|&c| near(c, 0.5)) {
                Ok(table(
                    "fcc",
                    &[
                        (g, [0.; 3]),
                        ("K", [0.375, 0.375, 0.75]),
                        ("L", [0.5, 0.5, 0.5]),
                        ("U", [0.625, 0.25, 0.625]),
                        ("W", [0.5, 0.25, 0.75]),
                        ("X", [0.5, 0., 0.5]),
                    ],
                    "GXWKGLUWLK,UX",
                ))
            } else if equal && cos.iter().all(|&c| near(c, -1. / 3.)) {
                Ok(table(
                    "bcc",
                    &[
                        (g, [0.; 3]),
                        ("H", [0.5, -0.5, 0.5]),
                        ("N", [0., 0., 0.5]),
                        ("P", [0.25, 0.25, 0.25]),
                    ],
                    "GHNGPH,PN",
                ))
            } else if close(l[0], l[1])
                && near(cos[0], -0.5)
                && near(cos[1], 0.)
                && near(cos[2], 0.)
            {
                Ok(table(
                    "hexagonal",
                    &[
                        (g, [0.; 3]),
                        ("A", [0., 0., 0.5]),
                        ("H", [1. / 3., 1. / 3., 0.5]),
                        ("K", [1. / 3., 1. / 3., 0.]),
                        ("L", [0.5, 0., 0.5]),
                        ("M", [0.5, 0., 0.]),
                    ],
                    "GMKGALHA,LM,KH",
                ))
            } else {
                Err(refusal("three-dimensional", &l, &cos))
            }
        }
    }
}

fn refusal(kind: &str, lengths: &[f64], cosines: &[f64]) -> crate::Error {
    invalid(format!(
        "no named high-symmetry points for this {kind} lattice (lengths {lengths:?}, cosines \
         {cosines:?}): the conventions cover the chain, the square, rectangular and hexagonal \
         planes, and the cubic, fcc, bcc, tetragonal, orthorhombic and hexagonal cells in the \
         orientations of `periodic::kpath::convention`; pass the path as explicit points"
    ))
}
