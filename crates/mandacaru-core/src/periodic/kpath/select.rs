//! The mesh points on a band path.
use super::{BandPath, Metric};
use crate::periodic::invalid;
use crate::Result;

/// A point is on a segment when its perpendicular distance is below this fraction of the
/// segment length (in the Cartesian reciprocal metric).
const ON_SEGMENT: f64 = 1e-7;
/// Reciprocal-lattice images `m + G` searched along each periodic axis.
const IMAGES: i32 = 2;

/// The mesh points on a path.
#[derive(Clone, Debug, PartialEq)]
pub struct PathPoints {
    pub convention: String,
    /// The fractional k of every selected point: a mesh point plus a reciprocal-lattice vector.
    pub kpoints: Vec<[f64; 3]>,
    /// The index (into the mesh handed to [`select`]) of the mesh point each one is an image of.
    pub mesh_index: Vec<usize>,
    /// Distance along the path (the inverse of the cell's length unit, `2 pi` included).
    pub distances: Vec<f64>,
    /// One entry per vertex, a jump merged into `"A|B"`.
    pub labels: Vec<String>,
    pub label_distances: Vec<f64>,
    /// The labels of the vertices that are no mesh point, in path order.
    pub missing: Vec<String>,
}

impl PathPoints {
    /// Every vertex is a mesh point.
    pub fn complete(&self) -> bool {
        self.missing.is_empty()
    }
}

fn is_integer(x: f64) -> bool {
    (x - x.round()).abs() < 1e-8
}

/// The points of `mesh` (fractional) on `path`, in path order, each exactly once: where two
/// segments meet the shared vertex counts once, and a vertex visited twice (the zone center at
/// both ends of `GXMG`) counts at each of its distances.
pub fn select(path: &BandPath, metric: &Metric, mesh: &[[f64; 3]]) -> Result<PathPoints> {
    let periodic = metric.periodic();
    let flat = |p: [f64; 3]| (0..3).all(|a| periodic[a] || p[a] == 0.);
    if mesh.is_empty()
        || mesh
            .iter()
            .any(|p| !p.iter().all(|x| x.is_finite()) || !flat(*p))
    {
        return Err(invalid(
            "the k-mesh must be finite with its vacuum axes at zero",
        ));
    }
    if path.vertices.iter().any(|v| !flat(v.fractional)) {
        return Err(invalid(
            "a band path must stay on the periodic axes of the cell",
        ));
    }
    let range = |a: usize| if periodic[a] { -IMAGES..=IMAGES } else { 0..=0 };
    // Distance of every vertex along the path.
    let mut at = vec![0.; path.vertices.len()];
    let mut lengths = vec![0.; path.vertices.len().saturating_sub(1)];
    for i in 0..lengths.len() {
        if !path.jump[i] {
            let d = std::array::from_fn(|a| {
                path.vertices[i + 1].fractional[a] - path.vertices[i].fractional[a]
            });
            lengths[i] = metric.norm(d);
            if lengths[i] <= 1e-12 {
                return Err(invalid(format!(
                    "the band path repeats vertex {:?} in a row",
                    path.vertices[i].label
                )));
            }
        }
        at[i + 1] = at[i] + lengths[i];
    }
    let total = *at.last().unwrap();
    let mut found: Vec<(f64, usize, [f64; 3])> = vec![];
    for i in 0..lengths.len() {
        if path.jump[i] {
            continue;
        }
        let (p, q) = (path.vertices[i].fractional, path.vertices[i + 1].fractional);
        let d: [f64; 3] = std::array::from_fn(|a| q[a] - p[a]);
        let dd = metric.dot(d, d);
        for (m, point) in mesh.iter().enumerate() {
            for i0 in range(0) {
                for i1 in range(1) {
                    for i2 in range(2) {
                        let image = [
                            point[0] + f64::from(i0),
                            point[1] + f64::from(i1),
                            point[2] + f64::from(i2),
                        ];
                        let r: [f64; 3] = std::array::from_fn(|a| image[a] - p[a]);
                        let t = metric.dot(r, d) / dd;
                        if !(-1e-9..=1. + 1e-9).contains(&t) {
                            continue;
                        }
                        let perpendicular: [f64; 3] = std::array::from_fn(|a| r[a] - t * d[a]);
                        if metric.norm(perpendicular) > ON_SEGMENT * lengths[i] {
                            continue;
                        }
                        found.push((at[i] + t.clamp(0., 1.) * lengths[i], m, image));
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    found.dedup_by(|b, a| a.1 == b.1 && (a.0 - b.0).abs() <= 1e-9 * total.max(1.));
    if found.is_empty() {
        return Err(invalid(
            "no point of the k-mesh lies on the band path: use a denser mesh or a path through \
             mesh points",
        ));
    }
    let mut labels: Vec<String> = vec![];
    let mut label_distances: Vec<f64> = vec![];
    for (i, vertex) in path.vertices.iter().enumerate() {
        if i > 0 && path.jump[i - 1] {
            let last = labels.last_mut().unwrap();
            *last = format!("{last}|{}", vertex.label);
        } else {
            labels.push(vertex.label.clone());
            label_distances.push(at[i]);
        }
    }
    let mut missing = vec![];
    for vertex in &path.vertices {
        let on_mesh = mesh
            .iter()
            .any(|m| (0..3).all(|a| is_integer(vertex.fractional[a] - m[a])));
        if !on_mesh && !missing.contains(&vertex.label) {
            missing.push(vertex.label.clone());
        }
    }
    Ok(PathPoints {
        convention: path.convention.clone(),
        kpoints: found.iter().map(|f| f.2).collect(),
        mesh_index: found.iter().map(|f| f.1).collect(),
        distances: found.iter().map(|f| f.0).collect(),
        labels,
        label_distances,
        missing,
    })
}
