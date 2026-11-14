//! A band path sampled with a given number of points, the way a band-structure plot wants it:
//! the points are shared among the straight segments in proportion to their lengths in the
//! Cartesian reciprocal metric (`2 pi` included), every run ends on its last vertex, and a jump
//! between runs (a comma) adds no length.
use super::{BandPath, Metric};
use crate::periodic::invalid;
use crate::Result;

/// A band path as `kpoints` fractional points, with the path text and the named points of the
/// lattice convention it was read from.
#[derive(Clone, Debug, PartialEq)]
pub struct Interpolated {
    pub kpoints: Vec<[f64; 3]>,
    /// The cumulative length along the path of each point.
    pub distances: Vec<f64>,
    /// The cumulative length at each vertex, in order.
    pub vertex_distances: Vec<f64>,
}

/// Python's `round` of a nonnegative float: to the nearest integer, ties to even.
fn round_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `npoints` points along `path` (at least the vertices): each segment of length `L` gets
/// `max(2, round(L (npoints - taken) / remaining))` points (the last of them is the first of the
/// next segment), so the total is `npoints` up to rounding, and the path ends on its last vertex.
pub fn interpolate(path: &BandPath, metric: &Metric, npoints: usize) -> Result<Interpolated> {
    if npoints < 2 {
        return Err(invalid("a band path needs at least two points"));
    }
    let vertices = &path.vertices;
    let lengths: Vec<f64> = (0..vertices.len() - 1)
        .map(|i| {
            if path.jump[i] {
                0.
            } else {
                let d: [f64; 3] = std::array::from_fn(|a| {
                    vertices[i + 1].fractional[a] - vertices[i].fractional[a]
                });
                metric.norm(d)
            }
        })
        .collect();
    let length: f64 = lengths.iter().sum();
    let (mut kpoints, mut x, mut vertex_distances) = (Vec::new(), Vec::new(), vec![0.]);
    let mut x0 = 0.0f64;
    for i in 0..vertices.len() - 1 {
        let (p, q, l) = (
            vertices[i].fractional,
            vertices[i + 1].fractional,
            lengths[i],
        );
        let diff = length - x0;
        let n = if diff.abs() < 1e-6 {
            0
        } else {
            round_even(l * (npoints as f64 - x.len() as f64) / diff).max(2)
        };
        if n >= 2 {
            let step = 1.0 / (n - 1) as f64;
            for k in 0..n - 1 {
                let t = k as f64 * step;
                kpoints.push(std::array::from_fn(|a| p[a] + t * (q[a] - p[a])));
                x.push(x0 + t * l);
            }
        }
        x0 += l;
        vertex_distances.push(x0);
    }
    if let Some(last) = vertices.last() {
        kpoints.push(last.fractional);
        x.push(x0);
    }
    Ok(Interpolated {
        kpoints,
        distances: x,
        vertex_distances,
    })
}
