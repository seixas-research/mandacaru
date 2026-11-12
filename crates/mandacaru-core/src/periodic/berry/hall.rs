//! The anomalous Hall conductivity from the filling-weighted plane numbers of the three k-plane
//! families: `K = sign(det G) sum_c nu_c G_c` and `sigma_ab = (e^2/h)(1/2 pi) eps_abd K_d`
//! (`sigma_xy = (e^2/h) C` with `C = (1/2 pi) int F_xy`, `F = curl A`, `A = i <u|grad u>`).
//! In atomic units `e^2/h = 1/(2 pi)`, so `sigma = eps K / (4 pi^2)` in `e^2/(hbar a0)`.
//!
//! An insulator's `nu_c` is `f C_c` (`f` electrons per band, `C_c` the integer Chern number,
//! equal on every layer). A metal's is the intrinsic Fermi-sea term `(1/2 pi) sum_n int f_n Omega_n`
//! from the occupation-weighted lattice field ([`super::weighted`]), averaged over the layers, at
//! the Fermi level re-solved on the Berry mesh; its convergence with the mesh is reported
//! (`sigma_coarse`, the stride-2 sub-mesh, and the fraction of merged plaquettes) and warned about.
use super::{curvature, invalid, plane_axes, weighted, Bands, Lattice, Manifold, Metal, Sampler};
use crate::periodic::spectrum;
use crate::{quantum::C64, units::Quantity, Error, Result};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;

/// Marks the insulator path's refusal of a metal, which [`anomalous_hall`] answers with the
/// intrinsic term instead.
const METAL: &str = "the crystal is a metal: its Hall conductivity is the intrinsic Fermi-sea term";

/// A metal's mesh estimate differing from the stride-2 one by more than this fraction of the
/// largest component is warned about.
pub const CONVERGENCE_WARNING: f64 = 0.1;
/// The smallest even mesh size along every direction with a stride-2 estimate (whose sub-mesh
/// then has at least 4 points: a 2-point plane is exactly zero).
pub const COARSE_MESH: usize = 8;
/// A larger fraction of plaquettes with merged band groups is warned about: the mesh does not
/// resolve the bands near the Fermi level.
pub const MERGED_WARNING: f64 = 0.25;

pub struct Hall {
    pub size: [usize; 3],
    /// A metal: `nu_c` is the intrinsic Fermi-sea term, not `f C_c`.
    pub metallic: bool,
    /// `[c][channel]`: an insulator's Chern number of the planes normal to `G_c` (equal on every
    /// layer); empty for a metal.
    pub chern: [Vec<i64>; 3],
    /// `[c]`: an insulator's Chern numbers summed over channels; zero for a metal.
    pub total: [i64; 3],
    /// `[c]`: `nu_c`, summed over channels, electrons per band included.
    pub numbers: [f64; 3],
    /// `[c][channel][layer]`: the plane number of every layer normal to `G_c`.
    pub layers: [Vec<Vec<f64>>; 3],
    /// `K = sign(det G) sum_c nu_c G_c` (1/Bohr).
    pub vector: [f64; 3],
    /// `sigma_ab` in `e^2/(hbar a0)`.
    pub sigma: [[f64; 3]; 3],
    /// A metal's `sigma` on the stride-2 sub-mesh of the same states (every `n_j` even and at
    /// least [`COARSE_MESH`]).
    pub sigma_coarse: Option<[[f64; 3]; 3]>,
    /// The largest plaquette (or band-group) flux of any plane.
    pub max_flux: f64,
    /// An insulator's smallest link singular value; NaN for a metal (its band groups vary).
    pub minimum: f64,
    /// A metal's band cap (bands linked per point); 0 for an insulator.
    pub cap: usize,
    /// A metal's plaquettes where a band crossing or an aliased node merged two groups, out of
    /// `plaquettes` (every layer of every plane family).
    pub merged: usize,
    pub plaquettes: usize,
    /// A metal's Fermi level on the Berry mesh (Hartree, the scale of `PeriodicResult::fermi`),
    /// solved with the converged smearing so the mesh holds the cell's electrons; the SCF level
    /// for an insulator.
    pub fermi: f64,
    /// Electrons the SCF Fermi level would put in the cell on the Berry mesh, minus the cell's:
    /// what re-solving the level corrected (0 for an insulator).
    pub scf_electron_error: f64,
    pub warnings: Vec<String>,
}

/// `(K, sigma)` of the plane numbers `numbers[c]` of the planes normal to `G_c` (the columns of
/// `reciprocal`, 1/Bohr): `K = sign(det G) sum_c nu_c G_c` and `sigma_ab = eps_abd K_d / (4 pi^2)`
/// in `e^2/(hbar a0)`, antisymmetric. Each `nu_c` is oriented along `G_a x G_b` (`(a, b, c)`
/// cyclic), which points against `a_c` in a left-handed cell: the sign of `det G` (that of the
/// lattice) restores the Cartesian orientation, so relabeling the cell leaves `sigma` unchanged.
pub fn conductivity_of(reciprocal: &Matrix3<f64>, numbers: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let handedness = reciprocal.determinant().signum();
    let vector: [f64; 3] = std::array::from_fn(|a| {
        handedness * (0..3).map(|c| numbers[c] * reciprocal[(a, c)]).sum::<f64>()
    });
    let mut sigma = [[0.; 3]; 3];
    for (a, b, d) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        sigma[a][b] = vector[d] / (4. * PI * PI);
        sigma[b][a] = -sigma[a][b];
    }
    (vector, sigma)
}

/// [`conductivity_of`] integer Chern numbers (one electron per band).
pub fn conductivity(reciprocal: &Matrix3<f64>, chern: [i64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    conductivity_of(reciprocal, chern.map(|c| c as f64))
}

/// The Hall conductivity on the mesh `size`: an insulator's quantized one from its Chern numbers,
/// a metal's intrinsic one ([`intrinsic`]).
pub fn anomalous_hall(sampler: &Sampler, size: [usize; 3]) -> Result<Hall> {
    match insulating(sampler, size) {
        Err(Error::Unsupported(message)) if message.ends_with(METAL) => intrinsic(sampler, size),
        other => other,
    }
}

fn mesh_check(size: [usize; 3]) -> Result<()> {
    if size.iter().any(|n| *n < 2) {
        return Err(invalid(format!(
            "the Hall conductivity needs at least 2 k-points along every direction; the mesh is \
             {size:?}"
        )));
    }
    Ok(())
}

/// The Hall conductivity of an insulator: every layer of every plane family gives its Chern
/// number, and the layers of one family must agree.
fn insulating(sampler: &Sampler, size: [usize; 3]) -> Result<Hall> {
    mesh_check(size)?;
    let manifold = sampler.manifold(Bands::Occupied, Metal::Unimplemented(METAL))?;
    let lattice = sampler.lattice(size, [0.; 3], &manifold)?;
    let channels = lattice.frames.len();
    let mut chern: [Vec<i64>; 3] = std::array::from_fn(|_| vec![0; channels]);
    let mut max_flux = 0f64;
    let mut minimum = f64::INFINITY;
    for c in 0..3 {
        for layer in 0..size[c] {
            let plane = curvature::plane(sampler, &lattice, c, layer)?;
            max_flux = max_flux.max(plane.max_flux);
            minimum = minimum.min(plane.minimum);
            if layer == 0 {
                chern[c] = plane.chern.clone();
            } else if plane.chern != chern[c] {
                return Err(invalid(format!(
                    "the Chern numbers of the planes normal to G_{c} differ between layers \
                     ({:?} and {:?}): the gap closes between them or the mesh is too coarse",
                    chern[c], plane.chern
                )));
            }
        }
    }
    let f = sampler.degeneracy();
    let total: [i64; 3] = std::array::from_fn(|c| chern[c].iter().sum());
    let numbers = total.map(|t| f * t as f64);
    let layers = std::array::from_fn(|c| {
        chern[c]
            .iter()
            .map(|n| vec![f * *n as f64; size[c]])
            .collect()
    });
    let (vector, sigma) = conductivity_of(&sampler.reciprocal, numbers);
    Ok(Hall {
        size,
        metallic: false,
        chern,
        total,
        numbers,
        layers,
        vector,
        sigma,
        sigma_coarse: None,
        max_flux,
        minimum,
        cap: 0,
        merged: 0,
        plaquettes: 0,
        fermi: sampler.state.fermi,
        scf_electron_error: 0.,
        warnings: vec![],
    })
}

/// The links of every point of `lattice` along the three directions over `stride` steps, per
/// channel: `[channel][direction][point]`.
fn all_links(
    sampler: &Sampler,
    lattice: &Lattice,
    stride: usize,
) -> Result<Vec<[Vec<DMatrix<C64>>; 3]>> {
    (0..lattice.frames.len())
        .map(|s| {
            let mut out: [Vec<DMatrix<C64>>; 3] = std::array::from_fn(|_| vec![]);
            for (d, links) in out.iter_mut().enumerate() {
                for p in 0..lattice.points() {
                    let at = lattice.indices(p);
                    if at.iter().any(|i| i % stride != 0) {
                        links.push(DMatrix::zeros(0, 0));
                        continue;
                    }
                    links.push(sampler.link_stride(lattice, s, p, d, stride)?);
                }
            }
            Ok(out)
        })
        .collect()
}

struct Sum {
    numbers: [f64; 3],
    layers: [Vec<Vec<f64>>; 3],
    max_flux: f64,
    merged: usize,
    plaquettes: usize,
}

/// The filling-weighted plane numbers of the sub-mesh of stride `stride`.
fn plane_numbers(
    sampler: &Sampler,
    lattice: &Lattice,
    levels: &[Vec<Vec<f64>>],
    occupations: &[Vec<Vec<f64>>],
    stride: usize,
) -> Result<Sum> {
    let links = all_links(sampler, lattice, stride)?;
    let channels = lattice.frames.len();
    let full = sampler.degeneracy();
    let mut out = Sum {
        numbers: [0.; 3],
        layers: std::array::from_fn(|_| vec![vec![]; channels]),
        max_flux: 0.,
        merged: 0,
        plaquettes: 0,
    };
    for c in 0..3 {
        let [a, b] = plane_axes(c)?;
        let (na, nb) = (lattice.size[a] / stride, lattice.size[b] / stride);
        for s in 0..channels {
            for layer in (0..lattice.size[c]).step_by(stride) {
                let mut point_levels = Vec::with_capacity(na * nb);
                let mut point_occupations = Vec::with_capacity(na * nb);
                let mut along_a = Vec::with_capacity(na * nb);
                let mut along_b = Vec::with_capacity(na * nb);
                for i in 0..na {
                    for j in 0..nb {
                        let mut at = [0; 3];
                        at[a] = i * stride;
                        at[b] = j * stride;
                        at[c] = layer;
                        let p = lattice.index(at);
                        point_levels.push(levels[s][p].clone());
                        point_occupations.push(occupations[s][p].clone());
                        along_a.push(links[s][a][p].clone());
                        along_b.push(links[s][b][p].clone());
                    }
                }
                let field = weighted::weighted_field(
                    &point_levels,
                    &point_occupations,
                    &along_a,
                    &along_b,
                    [na, nb],
                    full,
                )?;
                out.max_flux = out.max_flux.max(field.max_flux);
                out.merged += field.merged;
                out.plaquettes += field.plaquettes;
                out.layers[c][s].push(field.number);
            }
            let layers = &out.layers[c][s];
            out.numbers[c] += layers.iter().sum::<f64>() / layers.len() as f64;
        }
    }
    Ok(out)
}

/// The intrinsic Hall conductivity of a metal (or of any crystal) on the mesh `size`: the
/// occupation-weighted lattice field of the converged smearing, averaged over the layers of each
/// plane family, with the stride-2 sub-mesh value as its convergence estimate.
pub fn intrinsic(sampler: &Sampler, size: [usize; 3]) -> Result<Hall> {
    mesh_check(size)?;
    let (c, state) = (sampler.crystal, sampler.state);
    let full = sampler.degeneracy();
    let (smearing, width) = (state.options.smearing, state.options.width);
    if !(width.is_finite() && width > 0.) {
        return Err(invalid(
            "the intrinsic Hall conductivity needs a positive smearing width",
        ));
    }
    let window = width * weighted::occupation_window(|x| smearing.occupation(x));
    let count = |levels: &[f64], mu: f64| levels.iter().filter(|e| **e < mu + window).count();
    let mut cap = state
        .eigenvalues
        .iter()
        .map(|e| count(e, state.fermi))
        .max()
        .unwrap_or(0)
        .saturating_add(2)
        .clamp(1, c.bands());
    // The Fermi level of the Berry mesh: the SCF one holds the cell's electrons on the SCF mesh
    // only, and a denser mesh would fill the bands to another count.
    let (lattice, levels, mu) = loop {
        let manifold = Manifold::window(sampler.channels(), cap);
        let lattice = sampler.lattice(size, [0.; 3], &manifold)?;
        let levels: Vec<Vec<Vec<f64>>> = lattice
            .frames
            .iter()
            .map(|frames| {
                frames
                    .iter()
                    .map(|f| f.levels.as_slice().to_vec())
                    .collect()
            })
            .collect();
        let rows: Vec<Vec<f64>> = levels.iter().flatten().cloned().collect();
        let weights = vec![1. / lattice.points() as f64; rows.len()];
        let mu = spectrum::fermi(&rows, &weights, c.electrons, smearing, width, full)?;
        let needed = rows
            .iter()
            .map(|e| count(e, mu))
            .max()
            .unwrap_or(0)
            .saturating_add(2)
            .min(c.bands());
        if needed <= cap {
            break (lattice, levels, mu);
        }
        cap = needed;
    };
    let electrons_at = |mu: f64| {
        levels
            .iter()
            .flatten()
            .flatten()
            .map(|e| full * smearing.occupation((e - mu) / width))
            .sum::<f64>()
            / lattice.points() as f64
    };
    let scf_electron_error = electrons_at(state.fermi) - c.electrons;
    let occupy = |e: &f64| full * smearing.occupation((e - mu) / width);
    let occupations: Vec<Vec<Vec<f64>>> = levels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .map(|point| point.iter().map(occupy).collect())
                .collect()
        })
        .collect();
    let fine = plane_numbers(sampler, &lattice, &levels, &occupations, 1)?;
    let (vector, sigma) = conductivity_of(&sampler.reciprocal, fine.numbers);
    // The stride-2 sub-mesh needs at least 4 points per direction to be a lattice field at all
    // (a 2-point plane gives exactly zero), so the estimate starts at n = 8.
    let coarse = if size.iter().all(|n| n % 2 == 0 && *n >= COARSE_MESH) {
        let sum = plane_numbers(sampler, &lattice, &levels, &occupations, 2)?;
        Some(conductivity_of(&sampler.reciprocal, sum.numbers).1)
    } else {
        None
    };
    let mut warnings = vec![];
    let to_s_cm = Quantity::Conductivity.factor(false);
    let largest = sigma.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    match coarse {
        None => warnings.push(format!(
            "the intrinsic Hall conductivity of a metal converges slowly with the k mesh, and \
             the mesh {size:?} gives no estimate of its error: use an even kpts of at least \
             {COARSE_MESH} per direction, whose stride-2 sub-mesh is compared"
        )),
        Some(coarse) => {
            let difference = sigma
                .iter()
                .flatten()
                .zip(coarse.iter().flatten())
                .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
            // Below 1e-9 e^2/(hbar a0) (5e-5 S/cm) both are zero for the purpose.
            if difference > CONVERGENCE_WARNING * largest && difference > 1e-9 {
                warnings.push(format!(
                    "the intrinsic Hall conductivity of this metal is not converged with the k \
                     mesh: its largest component is {:.4e} S/cm on {size:?} and differs by \
                     {:.4e} S/cm on the stride-2 sub-mesh; use a denser kpts",
                    largest * to_s_cm,
                    difference * to_s_cm
                ));
            }
        }
    }
    let merged_fraction = fine.merged as f64 / fine.plaquettes.max(1) as f64;
    if merged_fraction > MERGED_WARNING {
        warnings.push(format!(
            "{:.0}% of the plaquettes of the mesh {size:?} merged band groups at a crossing or a \
             node: the mesh does not resolve the band structure near the Fermi level, and the \
             value is mostly the shared weight of those groups; use a denser kpts",
            100. * merged_fraction
        ));
    }
    Ok(Hall {
        size,
        metallic: true,
        chern: std::array::from_fn(|_| vec![]),
        total: [0; 3],
        numbers: fine.numbers,
        layers: fine.layers,
        vector,
        sigma,
        sigma_coarse: coarse,
        max_flux: fine.max_flux,
        minimum: f64::NAN,
        cap,
        merged: fine.merged,
        plaquettes: fine.plaquettes,
        fermi: mu,
        scf_electron_error,
        warnings,
    })
}
