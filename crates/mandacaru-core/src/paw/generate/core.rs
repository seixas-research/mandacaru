//! Spherical densities of a reference atom: the frozen core, its smooth counterpart, the
//! all-electron and smooth valence densities and the compensation charge the channels imply.
//! Atomic units.
use super::assemble::PawChannel;
use crate::{
    math::radial::local_derivatives,
    paw::{invalid, pseudize::snap, radial::shape},
    Result,
};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// Smooth counterpart of a spherical density: `a + b r^2 + c r^4` inside `r_cut` (moved to the
/// nearest mesh node), matched to the density in value, slope and curvature there.
pub fn pseudize_density(r: &[f64], density: &[f64], r_cut: f64) -> Result<Vec<f64>> {
    let r_cut = snap(r, r_cut);
    let d = local_derivatives(r, density, r_cut, 2, 25, 8)?;
    let matrix = Matrix3::new(
        1.,
        r_cut.powi(2),
        r_cut.powi(4),
        0.,
        2. * r_cut,
        4. * r_cut.powi(3),
        0.,
        2.,
        12. * r_cut.powi(2),
    );
    let coefficients = matrix
        .lu()
        .solve(&Vector3::new(d[0], d[1], d[2]))
        .ok_or_else(|| invalid("the smooth density matching system is singular"))?;
    let (a, b, c) = (coefficients[0], coefficients[1], coefficients[2]);
    Ok(r.iter()
        .zip(density)
        .map(|(&x, &rho)| {
            if x <= r_cut {
                a + b * x * x + c * x.powi(4)
            } else {
                rho
            }
        })
        .collect())
}

/// One occupied core subshell: its occupancy and `u = r R(r)` on the mesh.
pub struct CoreShell<'a> {
    pub occupancy: f64,
    pub orbital: &'a [f64],
}

/// The frozen core density, `sum occupancy u^2 / (4 pi r^2)`.
pub fn core_density(r: &[f64], shells: &[CoreShell<'_>]) -> Result<Vec<f64>> {
    let mut density = vec![0.; r.len()];
    for shell in shells {
        if shell.orbital.len() != r.len() {
            return Err(invalid("a core orbital must be tabulated on the mesh"));
        }
        for (i, rho) in density.iter_mut().enumerate() {
            *rho += shell.occupancy * shell.orbital[i].powi(2) / (4. * PI * r[i] * r[i]);
        }
    }
    Ok(density)
}

/// A channel with the all-electron partial waves it was built from (`count x len(r)`, flat).
pub struct ChannelWaves<'a> {
    pub channel: &'a PawChannel,
    pub ae_waves: &'a [f64],
}

/// What the occupied references of all channels add up to.
pub struct ValenceSums {
    pub ae_valence: Vec<f64>,
    pub smooth_valence: Vec<f64>,
    /// The electrons the smooth valence density is missing, `sum occupation * q_ii`.
    pub compensation_charge: f64,
    /// The band energy of the occupied references.
    pub band: f64,
}

/// Every occupied reference of every channel: one per channel, two in a semicore one.
pub fn valence_sums(n_points: usize, channels: &[ChannelWaves<'_>]) -> Result<ValenceSums> {
    let mut sums = ValenceSums {
        ae_valence: vec![0.; n_points],
        smooth_valence: vec![0.; n_points],
        compensation_charge: 0.,
        band: 0.,
    };
    for entry in channels {
        let channel = entry.channel;
        let count = channel.reference_energies.len();
        let tabulated = channel.pseudo_waves.as_slice().len();
        if entry.ae_waves.len() != count * n_points || tabulated != count * n_points {
            return Err(invalid(
                "the partial waves of a channel must be tabulated on the mesh",
            ));
        }
        for (i, &occupation) in channel.occupations.iter().enumerate() {
            let ae = &entry.ae_waves[i * n_points..(i + 1) * n_points];
            let smooth = &channel.pseudo_waves.as_slice()[i * n_points..(i + 1) * n_points];
            for k in 0..n_points {
                sums.ae_valence[k] += occupation * ae[k].powi(2) / (4. * PI);
                sums.smooth_valence[k] += occupation * smooth[k].powi(2) / (4. * PI);
            }
            sums.compensation_charge += occupation * channel.overlap_correction[(i, i)];
            sums.band += occupation * channel.reference_energies[i];
        }
    }
    Ok(sums)
}

/// The smooth valence density plus its compensation charge, shaped by the compact charge of
/// radius `radius`.
pub fn augmented_density(
    r: &[f64],
    smooth_valence: &[f64],
    compensation_charge: f64,
    radius: f64,
) -> Vec<f64> {
    r.iter()
        .zip(smooth_valence)
        .map(|(&x, &rho)| rho + compensation_charge * shape(x, radius))
        .collect()
}
