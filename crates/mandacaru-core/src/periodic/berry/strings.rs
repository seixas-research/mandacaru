//! Berry phases and Wilson loops of the closed strings of a mesh along one reciprocal vector.
use super::{
    invalid, links, transverse_axes, Bands, Lattice, Manifold, Metal, Sampler, LINK_TOLERANCE,
};
use crate::Result;
use nalgebra::Vector3;
use std::f64::consts::PI;

/// The strings along `G_direction` of a mesh, per channel. Strings are ordered in C order of the
/// two transverse indices (`super::transverse_axes`, ascending).
pub struct Strings {
    pub direction: usize,
    pub size: [usize; 3],
    /// The fractional k-point each string starts from.
    pub starts: Vec<[f64; 3]>,
    /// `[channel][string]`: the Berry phase on (-pi, pi] (0 for an empty channel).
    pub phases: Vec<Vec<f64>>,
    /// `[channel]`: the string phases averaged on the branch of their circular mean.
    pub mean: Vec<f64>,
    /// `[channel]`: the largest distance of a string phase from that mean.
    pub spread: Vec<f64>,
    /// `[channel][string][band]`: the Wannier charge centers along `a_direction` (fractional, in
    /// [0, 1), ascending).
    pub centers: Vec<Vec<Vec<f64>>>,
    /// `[channel][string][band]`: the Berry connection in the parallel-transport gauge of the
    /// string (Bohr), `2 pi x_n / |G_direction|` with the center `x_n` taken on (-1/2, 1/2].
    pub connection: Vec<Vec<Vec<f64>>>,
    /// The half-open band range of each channel.
    pub ranges: Vec<(usize, usize)>,
    /// The smallest singular value of every link used.
    pub minimum: f64,
}

/// Berry phases of `bands` along `direction` on the Gamma-anchored mesh `size`.
pub fn berry_phase(
    sampler: &Sampler,
    direction: usize,
    size: [usize; 3],
    bands: Bands,
) -> Result<Strings> {
    transverse_axes(direction)?;
    if size[direction] < 2 {
        return Err(invalid(format!(
            "a Berry string along G_{direction} needs at least 2 k-points; the mesh is {size:?}"
        )));
    }
    let manifold: Manifold = sampler.manifold(bands, Metal::Undefined)?;
    let lattice = sampler.lattice(size, [0.; 3], &manifold)?;
    strings(sampler, &lattice, direction)
}

/// The strings along `direction` of a mesh of frames.
pub fn strings(sampler: &Sampler, lattice: &Lattice, direction: usize) -> Result<Strings> {
    let [t0, t1] = transverse_axes(direction)?;
    let size = lattice.size;
    let length = Vector3::from(sampler.reciprocal_vector(direction)).norm();
    let count = size[t0] * size[t1];
    let channels = lattice.frames.len();
    let mut out = Strings {
        direction,
        size,
        starts: Vec::with_capacity(count),
        phases: vec![Vec::with_capacity(count); channels],
        mean: vec![0.; channels],
        spread: vec![0.; channels],
        centers: vec![Vec::with_capacity(count); channels],
        connection: vec![Vec::with_capacity(count); channels],
        ranges: lattice.ranges.clone(),
        minimum: f64::INFINITY,
    };
    for i in 0..size[t0] {
        for j in 0..size[t1] {
            let mut at = [0; 3];
            at[t0] = i;
            at[t1] = j;
            out.starts.push(lattice.fractional(at));
            for s in 0..channels {
                if lattice.bands(s) == 0 {
                    out.phases[s].push(0.);
                    out.centers[s].push(vec![]);
                    out.connection[s].push(vec![]);
                    continue;
                }
                let links = (0..size[direction])
                    .map(|n| {
                        let mut point = at;
                        point[direction] = n;
                        sampler.link(lattice, s, lattice.index(point), direction)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (phase, smallest) = links::string_phase(&links, LINK_TOLERANCE)?;
                let centers = links::wannier_centers(&links, LINK_TOLERANCE)?;
                out.minimum = out.minimum.min(smallest);
                out.connection[s].push(
                    centers
                        .iter()
                        .map(|x| {
                            let centered = if *x > 0.5 { x - 1. } else { *x };
                            2. * PI * centered / length
                        })
                        .collect(),
                );
                out.phases[s].push(phase);
                out.centers[s].push(centers);
            }
        }
    }
    for s in 0..channels {
        (out.mean[s], out.spread[s]) = links::circular_average(&out.phases[s])?;
    }
    Ok(out)
}
