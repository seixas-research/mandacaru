//! Isosurfaces and sampled clouds of a scalar field on a structured grid: the geometry the 3D
//! viewer draws (`utils/viewer.py`), free of any plotting and of any physical model.
//!
//! * [`marching_tetrahedra`]: the triangles of `values == level`, each cell cut into the six Kuhn
//!   tetrahedra along its main diagonal (neighboring cells cut a shared face the same way, so the
//!   surface has no cracks).
//! * [`enclosing_level`]: the smallest weight `w0` such that the nodes with `w >= w0` hold a
//!   fraction of the total (the "85% of the probability" isovalue convention).
//! * [`weighted_cloud`]: points drawn with probability proportional to non-negative node weights,
//!   each placed uniformly inside its node's cell (exact sampling of the piecewise-constant field).
use super::random::Rng;
use crate::{Error, Result};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

/// The Kuhn decomposition of a cube: corner `c` of a cell is node `(c >> 2, (c >> 1) & 1, c & 1)`
/// relative to the cell's first node, and every tetrahedron runs along the diagonal 0 -> 7.
const KUHN: [[usize; 4]; 6] = [[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7], [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]];

/// One triangle: three vertices of three Cartesian coordinates.
pub type Triangle = [[f64; 3]; 3];

/// The triangles of the surface `values == level` (`values` row-major over `shape`, the Cartesian
/// coordinates of every node in `coordinates`, so a skewed grid works as well as an orthogonal
/// one). The order is fixed: the tetrahedra with one corner above the level (one triangle each),
/// then those with three (one each), then the two triangles of each two-above quad (the first of
/// every quad, then the second), each group in the order of the crossed cells and of [`KUHN`].
/// The winding is not consistent.
pub fn marching_tetrahedra(
    shape: [usize; 3],
    values: &[f64],
    coordinates: [&[f64]; 3],
    level: f64,
) -> Result<Vec<Triangle>> {
    let [nx, ny, nz] = shape;
    let n = crate::product(&shape)?;
    if values.len() != n || coordinates.iter().any(|c| c.len() != n) {
        return Err(invalid("an isosurface needs one value and three coordinates per node"));
    }
    if !level.is_finite() || values.iter().chain(coordinates.iter().copied().flatten()).any(|v| !v.is_finite()) {
        return Err(invalid("an isosurface needs a finite field, coordinates and level"));
    }
    if nx.min(ny).min(nz) < 2 {
        return Ok(Vec::new());
    }
    let offsets: [usize; 8] = std::array::from_fn(|c| (c >> 2) * ny * nz + ((c >> 1) & 1) * nz + (c & 1));
    // Each kept tetrahedron with its corners sorted above-first (stably) and its count above.
    let mut kept: Vec<([usize; 4], usize)> = Vec::new();
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            for k in 0..nz - 1 {
                let base = i * ny * nz + j * nz + k;
                let above = offsets.iter().filter(|&&o| values[base + o] > level).count();
                if above == 0 || above == 8 {
                    continue;
                }
                for tetrahedron in &KUHN {
                    let nodes = tetrahedron.map(|c| base + offsets[c]);
                    let count = nodes.iter().filter(|&&p| values[p] > level).count();
                    if count == 0 || count == 4 {
                        continue;
                    }
                    let mut sorted = [0; 4];
                    let mut at = 0;
                    for pass in [true, false] {
                        for &p in &nodes {
                            if (values[p] > level) == pass {
                                sorted[at] = p;
                                at += 1;
                            }
                        }
                    }
                    kept.push((sorted, count));
                }
            }
        }
    }
    let point = |p: usize| [coordinates[0][p], coordinates[1][p], coordinates[2][p]];
    let edge = |nodes: &[usize; 4], a: usize, b: usize| -> [f64; 3] {
        let (va, vb) = (values[nodes[a]], values[nodes[b]]);
        let t = (level - va) / (vb - va);
        let (pa, pb) = (point(nodes[a]), point(nodes[b]));
        std::array::from_fn(|x| pa[x] + t * (pb[x] - pa[x]))
    };
    let mut out = Vec::with_capacity(kept.len() * 2);
    for (nodes, _) in kept.iter().filter(|(_, c)| *c == 1) {
        out.push([edge(nodes, 0, 1), edge(nodes, 0, 2), edge(nodes, 0, 3)]);
    }
    for (nodes, _) in kept.iter().filter(|(_, c)| *c == 3) {
        out.push([edge(nodes, 3, 0), edge(nodes, 3, 1), edge(nodes, 3, 2)]);
    }
    let quads: Vec<[[f64; 3]; 4]> = kept
        .iter()
        .filter(|(_, c)| *c == 2)
        .map(|(nodes, _)| [edge(nodes, 0, 2), edge(nodes, 0, 3), edge(nodes, 1, 3), edge(nodes, 1, 2)])
        .collect();
    out.extend(quads.iter().map(|q| [q[0], q[1], q[2]]));
    out.extend(quads.iter().map(|q| [q[0], q[2], q[3]]));
    Ok(out)
}

/// The smallest weight `w0` such that the nodes with `w >= w0` hold `fraction` of the total of the
/// non-negative `weights`: the weights in decreasing order, the first whose running share reaches
/// `fraction`.
pub fn enclosing_level(weights: &[f64], fraction: f64) -> Result<f64> {
    if !(fraction > 0. && fraction < 1.) {
        return Err(invalid(&format!("enclosed must lie strictly between 0 and 1, got {fraction}")));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.) {
        return Err(invalid("the enclosing level needs finite, non-negative weights"));
    }
    let mut sorted = weights.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = sorted.iter().sum();
    if total <= 0. {
        return Err(invalid("the field is zero everywhere; there is nothing to draw"));
    }
    let mut running = 0.;
    for (i, w) in sorted.iter().enumerate() {
        running += w;
        if running / total >= fraction {
            return Ok(sorted[i]);
        }
    }
    Ok(*sorted.last().expect("a positive total has a weight"))
}

/// `count` points drawn with probability proportional to the non-negative node `weights`: the
/// picked node of each point and its position, the node's coordinates (`nodes`, one row of three
/// per node) plus a uniform offset inside the cell, `sum_a u_a step[b][a]` with `u_a` in
/// `[-1/2, 1/2)` (`step` the grid step, one axis vector per column). The stream is
/// [`Rng`]'s from `seed`, so a seed draws the same cloud on every platform.
pub fn weighted_cloud(
    weights: &[f64],
    nodes: &[[f64; 3]],
    step: &[[f64; 3]; 3],
    count: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<[f64; 3]>)> {
    if nodes.len() != weights.len() {
        return Err(invalid("a cloud needs one position per weight"));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.) {
        return Err(invalid("a cloud needs finite, non-negative weights"));
    }
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.;
    for w in weights {
        total += w;
        cumulative.push(total);
    }
    if total <= 0. {
        return Err(invalid("the field is zero everywhere; there is nothing to sample"));
    }
    let mut rng = Rng(seed);
    let mut picked = Vec::with_capacity(count);
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        let target = rng.uniform() * total;
        // The first node whose running weight exceeds the target: a zero weight is never picked.
        let node = cumulative.partition_point(|&c| c <= target).min(weights.len() - 1);
        let u: [f64; 3] = std::array::from_fn(|_| rng.uniform() - 0.5);
        let at = nodes[node];
        points.push(std::array::from_fn(|b| at[b] + (0..3).map(|a| u[a] * step[b][a]).sum::<f64>()));
        picked.push(node);
    }
    Ok((picked, points))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cloud_follows_its_weights_and_its_seed() {
        let weights = [0., 1., 3., 0., 6.];
        let nodes: Vec<[f64; 3]> = (0..5).map(|i| [i as f64, 0., 0.]).collect();
        let step = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let (picked, points) = weighted_cloud(&weights, &nodes, &step, 20000, 7).unwrap();
        let share = |i: usize| picked.iter().filter(|&&p| p == i).count() as f64 / 20000.;
        assert_eq!((share(0), share(3)), (0., 0.));
        for (i, want) in [(1, 0.1), (2, 0.3), (4, 0.6)] {
            assert!((share(i) - want).abs() < 0.015, "node {i}: {}", share(i));
        }
        assert!(points.iter().zip(&picked).all(|(p, &i)| (p[0] - i as f64).abs() <= 0.5 && p[1].abs() <= 0.5));
        assert_eq!(weighted_cloud(&weights, &nodes, &step, 50, 7).unwrap().0, picked[..50]);
    }
}
