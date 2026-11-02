//! E-1..E-7: the explicit strain derivatives of the electrostatic energy at the frozen output
//! density and moments.
//!
//! `E_G = (1/Omega) sum_G K_G (|rho_G|^2/2 + Re rho_G^* (h_G + I_G))`, `rho_G = dv sum_i rho_i
//! e^{-iG.r_i}`, `h_G = sum_a q_a hat_a(G)`, with the node values `rho_i` and the moments `q`
//! fixed:
//!
//! * E-1: `dv` (inside `rho_G`) and `1/Omega` cancel in the cross term and leave a NET
//!   `delta_ab E_rr`, `E_rr = (1/2 Omega) sum K |rho_G|^2`, in the Hartree self term;
//! * E-2: the kernel, `dK_ab = 2 K G_a G_b / G^2` per alias, alias mean;
//! * E-3, E-4: the hats (form factors, atom phases fixed) and the Gaussian ions against
//!   `K rho^* / Omega`.
//!
//! The dense sums E-5/E-6/E-7 contract the provider's `dU`, `d u_ion`, `d linear`, `dE_ion` and
//! `d c_ion` with `q` exactly as the energy does (`solver::electrostatic`).
use super::{Dense, Tensor};
use crate::{
    periodic::{invalid, setup::strain_grid::VOIGT},
    quantum::C64,
    Buffer, Result,
};

/// The reciprocal-space data of one output density, on the grid's reciprocal set.
pub struct Grid<'a> {
    /// Cell volume (Bohr^3).
    pub volume: f64,
    /// `rho_G`.
    pub density: &'a [C64],
    /// `K rho_G^* / Omega`.
    pub weights: &'a [C64],
    /// `h_G + I_G`.
    pub charge: Buffer<C64>,
    /// `dK/d eps_ab`, `(6, n)` Voigt rows.
    pub kernel: Buffer<f64>,
    /// `d h_G / d eps_ab`, `(9, n)`, row `a * 3 + b`.
    pub hats: Buffer<C64>,
    /// `d I_G / d eps_ab`, `(6, n)` Voigt rows.
    pub ions: Buffer<C64>,
}

/// E-1..E-4, `dE/d eps_ab`.
pub struct Hartree {
    pub jacobian: Tensor<f64>,
    pub kernel: Tensor<f64>,
    pub hats: Tensor<f64>,
    pub ions: Tensor<f64>,
}

/// The Voigt row of the symmetric pair `(a, b)`.
pub fn voigt(a: usize, b: usize) -> usize {
    VOIGT
        .iter()
        .position(|&(p, q)| (p, q) == (a, b) || (q, p) == (a, b))
        .expect("a pair of axes is one of the six")
}

pub fn hartree(grid: &Grid<'_>) -> Result<Hartree> {
    let n = grid.density.len();
    if grid.weights.len() != n
        || grid.charge.as_slice().len() != n
        || grid.kernel.as_slice().len() != 6 * n
        || grid.hats.as_slice().len() != 9 * n
        || grid.ions.as_slice().len() != 6 * n
        || !(grid.volume > 0.)
    {
        return Err(invalid(
            "the Hartree strain terms need one value per reciprocal node",
        ));
    }
    let rho = grid.density;
    // E-1: `E_rr = (1/2 Omega) sum K |rho|^2 = 1/2 Re sum (K rho^*/Omega) rho`.
    let e_rr: f64 = 0.5
        * grid
            .weights
            .iter()
            .zip(rho)
            .map(|(w, r)| (w * r).re)
            .sum::<f64>();
    let bare: Vec<f64> = rho
        .iter()
        .zip(grid.charge.as_slice())
        .map(|(r, c)| 0.5 * r.norm_sqr() + (r.conj() * c).re)
        .collect();
    let mut out = Hartree {
        jacobian: [[0.; 3]; 3],
        kernel: [[0.; 3]; 3],
        hats: [[0.; 3]; 3],
        ions: [[0.; 3]; 3],
    };
    for a in 0..3 {
        out.jacobian[a][a] = e_rr;
        for b in 0..3 {
            let row = voigt(a, b);
            let dk = &grid.kernel.as_slice()[row * n..(row + 1) * n];
            out.kernel[a][b] = dk.iter().zip(&bare).map(|(d, e)| d * e).sum::<f64>() / grid.volume;
            let dh = &grid.hats.as_slice()[(a * 3 + b) * n..(a * 3 + b + 1) * n];
            out.hats[a][b] = grid.weights.iter().zip(dh).map(|(w, d)| (w * d).re).sum();
            let i = &grid.ions.as_slice()[row * n..(row + 1) * n];
            out.ions[a][b] = grid.weights.iter().zip(i).map(|(w, d)| (w * d).re).sum();
        }
    }
    Ok(out)
}

/// E-5 and E-6, `dE/d eps_ab`.
pub struct DenseTerms {
    /// `Re (q^+ dU q / 2 + q^+ d u_ion) + dE_ion`.
    pub dense: Tensor<f64>,
    /// `Re (q . d linear)`.
    pub linear: Tensor<f64>,
}

pub fn dense_terms(d: &Dense, q: &[C64]) -> Result<DenseTerms> {
    let c = q.len();
    let sizes = d.u.iter().flatten().all(|u| u.shape() == (c, c))
        && d.u_ion.iter().flatten().all(|v| v.len() == c)
        && d.linear.iter().flatten().all(|v| v.len() == c);
    if !sizes {
        return Err(invalid("the dense strain terms need one entry per channel"));
    }
    let mut out = DenseTerms {
        dense: [[0.; 3]; 3],
        linear: [[0.; 3]; 3],
    };
    for a in 0..3 {
        for b in 0..3 {
            let (u, ion, linear) = (&d.u[a][b], &d.u_ion[a][b], &d.linear[a][b]);
            let mut quadratic = C64::new(0., 0.);
            let mut single = C64::new(0., 0.);
            for (i, qi) in q.iter().enumerate() {
                let uq: C64 = q.iter().enumerate().map(|(j, qj)| u[(i, j)] * qj).sum();
                quadratic += 0.5 * qi.conj() * uq + qi.conj() * ion[i];
                single += qi * linear[i];
            }
            out.dense[a][b] = quadratic.re + d.e_ion[a][b];
            out.linear[a][b] = single.re;
        }
    }
    Ok(out)
}
