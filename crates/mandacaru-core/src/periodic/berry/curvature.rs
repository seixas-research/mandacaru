//! Berry curvature and Chern number of a k plane (Fukui, Hatsugai and Suzuki): the lattice field
//! strength of the link determinants of a band group, non-Abelian through the determinant.
use super::{invalid, links, plane_axes, Bands, Lattice, Metal, Sampler, LINK_TOLERANCE};
use crate::{quantum::C64, Result};
use nalgebra::Vector3;

/// The Berry curvature of one k plane, per channel.
pub struct Plane {
    /// The plane is normal to `G_normal` and spanned by `G_axes[0]`, `G_axes[1]` (cyclic).
    pub normal: usize,
    pub axes: [usize; 2],
    pub size: [usize; 2],
    /// The plane's fractional coordinate along `G_normal`.
    pub offset: f64,
    /// Cartesian unit vector along `G_a x G_b`, the orientation of `curvature`.
    pub orientation: [f64; 3],
    /// `[channel][i_a * n_b + i_b]`: the flux of the plaquette at `(i_a, i_b)` (radians).
    pub flux: Vec<Vec<f64>>,
    /// `[channel][i_a * n_b + i_b]`: the curvature along `orientation` (Bohr^2).
    pub curvature: Vec<Vec<f64>>,
    /// Fractional plaquette centers, `i_a * n_b + i_b`.
    pub centers: Vec<[f64; 3]>,
    /// `[channel]`: the Chern number `(1/2 pi) sum flux`, oriented along `orientation`: it flips
    /// sign when the cell is relabeled into the other handedness (`hall::conductivity` corrects).
    pub chern: Vec<i64>,
    /// The largest `|flux|` of any channel.
    pub max_flux: f64,
    /// The half-open band range of each channel.
    pub ranges: Vec<(usize, usize)>,
    /// The smallest singular value of every link used.
    pub minimum: f64,
}

impl Plane {
    pub fn total_chern(&self) -> i64 {
        self.chern.iter().sum()
    }
}

/// Mesh size and origin of the plane normal to `normal` with `size = (n_a, n_b)` at `offset`.
pub fn plane_mesh(normal: usize, size: [usize; 2], offset: f64) -> Result<([usize; 3], [f64; 3])> {
    let [a, b] = plane_axes(normal)?;
    if size.iter().any(|n| *n < 2) {
        return Err(invalid(format!(
            "a Berry curvature plane needs at least 2 k-points along each in-plane direction; \
             got {size:?}"
        )));
    }
    if !offset.is_finite() || !(0. ..1.).contains(&offset) {
        return Err(invalid(
            "a k plane's offset along its normal is fractional, in [0, 1)",
        ));
    }
    let mut mesh = [1; 3];
    mesh[a] = size[0];
    mesh[b] = size[1];
    let mut origin = [0.; 3];
    origin[normal] = offset;
    Ok((mesh, origin))
}

/// The Berry curvature of `bands` on the plane normal to `G_normal` (`size` points along its two
/// in-plane directions) at the fractional `offset` along the normal.
pub fn berry_curvature(
    sampler: &Sampler,
    normal: usize,
    size: [usize; 2],
    offset: f64,
    bands: Bands,
) -> Result<Plane> {
    let (mesh, origin) = plane_mesh(normal, size, offset)?;
    let manifold = sampler.manifold(bands, Metal::Undefined)?;
    let lattice = sampler.lattice(mesh, origin, &manifold)?;
    plane(sampler, &lattice, normal, 0)
}

/// The plane normal to `normal` through layer `layer` (its index along the normal) of a mesh.
pub fn plane(sampler: &Sampler, lattice: &Lattice, normal: usize, layer: usize) -> Result<Plane> {
    let [a, b] = plane_axes(normal)?;
    let size = [lattice.size[a], lattice.size[b]];
    if size.iter().any(|n| *n < 2) || layer >= lattice.size[normal] {
        return Err(invalid(
            "a Berry plane needs two in-plane points per direction and a layer",
        ));
    }
    let (ga, gb) = (
        Vector3::from(sampler.reciprocal_vector(a)),
        Vector3::from(sampler.reciprocal_vector(b)),
    );
    let cross = ga.cross(&gb);
    let area = cross.norm() / (size[0] * size[1]) as f64;
    let orientation = cross / cross.norm();
    let at = |i: usize, j: usize| {
        let mut point = [0; 3];
        point[a] = i % size[0];
        point[b] = j % size[1];
        point[normal] = layer;
        lattice.index(point)
    };
    let mut centers = Vec::with_capacity(size[0] * size[1]);
    for i in 0..size[0] {
        for j in 0..size[1] {
            let mut center = lattice.fractional({
                let mut point = [0; 3];
                point[a] = i;
                point[b] = j;
                point[normal] = layer;
                point
            });
            center[a] += 0.5 / size[0] as f64;
            center[b] += 0.5 / size[1] as f64;
            centers.push(center);
        }
    }
    let channels = lattice.frames.len();
    let mut out = Plane {
        normal,
        axes: [a, b],
        size,
        offset: lattice.fractional([0; 3])[normal] + layer as f64 / lattice.size[normal] as f64,
        orientation: orientation.into(),
        flux: vec![vec![]; channels],
        curvature: vec![vec![]; channels],
        centers,
        chern: vec![0; channels],
        max_flux: 0.,
        ranges: lattice.ranges.clone(),
        minimum: f64::INFINITY,
    };
    for s in 0..channels {
        if lattice.bands(s) == 0 {
            out.flux[s] = vec![0.; size[0] * size[1]];
            out.curvature[s] = vec![0.; size[0] * size[1]];
            continue;
        }
        // The unit-modulus link determinants along a and b at every point of the plane.
        let mut along = [
            vec![C64::default(); size[0] * size[1]],
            vec![C64::default(); size[0] * size[1]],
        ];
        for i in 0..size[0] {
            for j in 0..size[1] {
                for (d, direction) in [a, b].into_iter().enumerate() {
                    let link = sampler.link(lattice, s, at(i, j), direction)?;
                    let (det, smallest) = links::link_phase(&link, LINK_TOLERANCE)?;
                    out.minimum = out.minimum.min(smallest);
                    along[d][i * size[1] + j] = det;
                }
            }
        }
        let field = links::lattice_field(&along[0], &along[1], size)?;
        out.max_flux = out.max_flux.max(field.max_flux);
        out.curvature[s] = field.flux.iter().map(|phi| phi / area).collect();
        out.flux[s] = field.flux;
        out.chern[s] = field.chern;
    }
    Ok(out)
}
