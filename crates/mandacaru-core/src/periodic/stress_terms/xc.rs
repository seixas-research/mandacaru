//! X-1..X-4: the strain terms of the grid exchange-correlation energy
//! `E_xc = dv sum_i f(n_up, n_dn, sigma_uu, sigma_ud, sigma_dd)`, `n_s = rho_s + rho_core/2`
//! (the clamp at zero is the one non-smooth point), at the frozen output density.
//!
//! * X-1: the `dv` of the integral, `+ delta_ab E_xc` (node values fixed).
//! * X-2: the xc potential through `rho_i` is inside the local-potential matrix (S-3), not here.
//! * X-3: the spectral gradient of a GGA. `d (partial_c n_s) / d eps_ab = -delta_cb partial_a n_s`
//!   (the wave vectors of the derivative move as `(1 - eps^T)`, the index-space shift and the
//!   zeroed Nyquist planes are pinned), so `dE = -dv sum_i sum_s flux^s_b(i) (partial_a n_s)(i)`
//!   with `flux^s_c = d f / d (partial_c n_s)` the arrays the divergence of the potential is
//!   built from. Both rotated spin channels are summed even for a restricted run
//!   (`n_up = n_dn = rho/2 + core/2`).
//! * X-4: the partial core, `dE = dv sum_i vbar_i d rho_core,i`, `vbar = (v_up + v_dn)/2` the
//!   spin-averaged xc potential (the core is spin-unpolarized: it sees the average, never `v_up`
//!   alone), `d rho_core / d eps_ab = sum rho_c'(|d|) d_a d_b / |d|` from the analytic slope of
//!   the cubic the placement samples.
//! * X-5: a meta-GGA's partial-core kinetic density (the von Weizsaecker `tau` of the same smooth
//!   core), `dE = dv sum_i vbar_tau,i d tau_core,i` with the spin-averaged `v_tau`, placed and
//!   differentiated like X-4. The kinetic density of the bands is in the matrix term S-6.
use super::Tensor;
use crate::{periodic::invalid, Result};

/// X-1, X-3, X-4, X-5, `dE/d eps_ab`.
pub struct Terms {
    pub jacobian: Tensor<f64>,
    pub gradient: Tensor<f64>,
    pub core: Tensor<f64>,
    /// X-5, zero for a functional without `tau`.
    pub core_kinetic: Tensor<f64>,
}

/// X-1: `delta_ab E_xc`.
pub fn jacobian(energy: f64) -> Tensor<f64> {
    let mut out = [[0.; 3]; 3];
    for (a, row) in out.iter_mut().enumerate() {
        row[a] = energy;
    }
    out
}

/// X-3: `-dv sum_i sum_s flux^s_b (partial_a n_s)`; `gradients` and `flux` are `(6, n)` with row
/// `3 s + c` (spin `s = 0, 1`), `None` for the local functional (no gradient, no term).
pub fn gradient(
    dv: f64,
    n: usize,
    gradients: Option<&[f64]>,
    flux: Option<&[f64]>,
) -> Result<Tensor<f64>> {
    let mut out = [[0.; 3]; 3];
    match (gradients, flux) {
        (None, None) => {}
        (Some(g), Some(f)) if g.len() == 6 * n && f.len() == 6 * n => {
            for a in 0..3 {
                for b in 0..3 {
                    let mut sum = 0.;
                    for s in 0..2 {
                        let grad = &g[(3 * s + a) * n..(3 * s + a + 1) * n];
                        let flow = &f[(3 * s + b) * n..(3 * s + b + 1) * n];
                        sum += grad.iter().zip(flow).map(|(x, y)| x * y).sum::<f64>();
                    }
                    out[a][b] = -dv * sum;
                }
            }
        }
        _ => {
            return Err(invalid(
                "the GGA strain term needs the gradients and the flux of both spin channels, \
                 six node arrays each, or neither",
            ))
        }
    }
    Ok(out)
}
