//! The grid xc energy of fixed node values, written from the public point functions: the
//! oracle of X-1/X-3/X-4 and the source of the gradients and the flux the strain provider is
//! handed. `n_s = (rho/2 + core/2).max(0)` for the two rotated spin channels of a restricted run,
//! the spectral gradient of the grid's symmetric wave vectors, `E = dv sum f`.
#![allow(dead_code)]

use mandacaru_core::{
    math::{fft::Fft3, reciprocal, SpatialGrid},
    models::xc::point_partials,
    Context,
};
use num_complex::Complex64;

pub struct Xc {
    pub energy: f64,
    /// The energy density `f` per node (before `dv`): differenced node by node, the roundoff of
    /// a finite difference follows the small differences and not the 1-25 Ha sum.
    pub nodes: Vec<f64>,
    /// The spin-averaged potential `(v_up + v_dn)/2`, zero where `rho + core < 0`.
    pub potential: Vec<f64>,
    pub gradients: Option<Vec<f64>>,
    pub flux: Option<Vec<f64>>,
}

/// `d_axis f` of a real field by the spectral derivative with the wave vectors `wave` (`(3, n)`).
fn derivative(fft: &mut Fft3, wave: &[f64], field: &[f64], axis: usize) -> Vec<f64> {
    let n = field.len();
    let mut work: Vec<Complex64> = field.iter().map(|&v| Complex64::new(v, 0.)).collect();
    fft.transform(&mut work, false).unwrap();
    for (k, v) in work.iter_mut().enumerate() {
        *v *= Complex64::new(0., wave[axis * n + k]);
    }
    fft.transform(&mut work, true).unwrap();
    work.iter().map(|v| v.re).collect()
}

/// The symmetric wave vectors of `grid` (what the crystal stores as `xc_wave`).
pub fn wave(ctx: &Context, grid: &SpatialGrid) -> Vec<f64> {
    reciprocal::symmetric_wavevectors(ctx, grid)
        .unwrap()
        .as_slice()
        .to_vec()
}

/// `E_xc`, the potential and (for `pbe`) the gradients and flux of the valence density `rho`
/// plus half the `core` in each spin channel, on `grid` with the wave vectors `wave` and the
/// volume element `dv` (the three are separate arguments so that an oracle can strain them one
/// at a time).
pub fn evaluate(
    ctx: &Context,
    grid: &SpatialGrid,
    wave: &[f64],
    dv: f64,
    functional: &str,
    relativistic: bool,
    rho: &[f64],
    core: &[f64],
) -> Xc {
    let n = rho.len();
    let pbe = functional == "pbe";
    let mut fft = Fft3::new(ctx, grid.index.shape()).unwrap();
    let mut density = vec![0.; 2 * n];
    for s in 0..2 {
        for k in 0..n {
            density[s * n + k] = (0.5 * rho[k] + 0.5 * core[k]).max(0.);
        }
    }
    let mut gradients = vec![0.; 6 * n];
    if pbe {
        for s in 0..2 {
            for a in 0..3 {
                let g = derivative(&mut fft, wave, &density[s * n..(s + 1) * n], a);
                gradients[(3 * s + a) * n..(3 * s + a + 1) * n].copy_from_slice(&g);
            }
        }
    }
    let mut flux = vec![0.; 6 * n];
    let mut potential = vec![0.; 2 * n];
    let mut energy = 0.;
    let mut nodes = vec![0.; n];
    for k in 0..n {
        let live = density[k] + density[n + k] > 1e-10;
        let mut sigma = [0.; 3];
        if live {
            for a in 0..3 {
                let (u, d) = (gradients[a * n + k], gradients[(3 + a) * n + k]);
                sigma[0] += u * u;
                sigma[1] += u * d;
                sigma[2] += d * d;
            }
        }
        let (f, df) = point_partials(
            functional,
            [
                density[k],
                density[n + k],
                sigma[0],
                sigma[1],
                sigma[2],
                0.,
                0.,
            ],
            relativistic,
        )
        .unwrap();
        energy += f;
        nodes[k] = f;
        potential[k] = df[0];
        potential[n + k] = df[1];
        if live && pbe {
            for a in 0..3 {
                let (u, d) = (gradients[a * n + k], gradients[(3 + a) * n + k]);
                flux[a * n + k] = 2. * df[2] * u + df[3] * d;
                flux[(3 + a) * n + k] = 2. * df[4] * d + df[3] * u;
            }
        }
    }
    if pbe {
        for s in 0..2 {
            for a in 0..3 {
                let div = derivative(
                    &mut fft,
                    wave,
                    &flux[(3 * s + a) * n..(3 * s + a + 1) * n],
                    a,
                );
                for k in 0..n {
                    potential[s * n + k] -= div[k];
                }
            }
        }
    }
    let vbar: Vec<f64> = (0..n)
        .map(|k| {
            if rho[k] + core[k] < 0. {
                0.
            } else {
                0.5 * (potential[k] + potential[n + k])
            }
        })
        .collect();
    Xc {
        energy: energy * dv,
        nodes,
        potential: vbar,
        gradients: pbe.then_some(gradients),
        flux: pbe.then_some(flux),
    }
}
