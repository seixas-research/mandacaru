//! S-1..S-6: the matrix terms of the stress, `sum_{s,k} deg w_k Re tr[D dh_ab - W dS_ab]` (S-6 is a
//! meta-GGA's kinetic-density operator, [`Term::KineticDensity`]).
//!
//! The force engine's algebra with the strain tangent: the overlap enters through `-W dS`, the
//! kinetic, local and short-range matrices through `D dM`, and the projections through
//! `Z = H_C C^+ D - q C^+ W` (`p x m`, `H_C = D_ion + sum_a m_a B_a (+ O_s)`):
//! `dE_C = 2 Re sum_{mu,p} dC_ab[mu,p] Z[p,mu]`. Nothing is owner-masked; the +U operator joins
//! `D_ion` here and nowhere else, so `-W dS` carries it through the bands.
use super::{spin_orbit, Strain, Tensor};
use crate::{
    periodic::gradient::{Frozen, Matrix, Term},
    quantum::C64,
    Result,
};

/// The five matrix terms, `dE/d eps_ab`.
pub struct Terms {
    pub overlap: Tensor<f64>,
    pub kinetic: Tensor<f64>,
    pub local: Tensor<f64>,
    pub short_range: Tensor<f64>,
    pub projection: Tensor<f64>,
    /// S-6, a meta-GGA's `T_tau = 1/2 <grad chi | v_tau | grad chi>` (zero otherwise).
    pub kinetic_density: Tensor<f64>,
    /// SO-1 (spinor requests; zero otherwise).
    pub spin_orbit: Tensor<f64>,
}

/// `Re sum_{ij} left[i, j] right[j, i]` (`Re tr[left right]`).
fn trace(left: &Matrix, right: &Matrix) -> f64 {
    let (n, m) = left.shape();
    let mut sum = C64::new(0., 0.);
    for j in 0..m {
        for i in 0..n {
            sum += left[(i, j)] * right[(j, i)];
        }
    }
    sum.re
}

/// `D_ion + sum_a m_a B_a (+ O)`: the projector-space operator of one spin channel.
pub fn projector_operator<S: Strain>(
    s: &S,
    fields: &S::Fields,
    hubbard: Option<&Matrix>,
) -> Matrix {
    let mut operator = s.ion_operator().clone();
    for (block, &m) in s.blocks().iter().zip(s.moments(fields)) {
        operator += block * m;
    }
    if let Some(o) = hubbard {
        operator += o;
    }
    operator
}

pub fn contract<S: Strain>(s: &S, frozen: &Frozen<S::Fields>) -> Result<Terms> {
    let l = s.layout();
    let mut out = Terms {
        overlap: [[0.; 3]; 3],
        kinetic: [[0.; 3]; 3],
        local: [[0.; 3]; 3],
        short_range: [[0.; 3]; 3],
        projection: [[0.; 3]; 3],
        kinetic_density: [[0.; 3]; 3],
        spin_orbit: [[0.; 3]; 3],
    };
    let terms = Term::of(&l.request.functional)?;
    for k in 0..l.weights.len() {
        let c = s.projection(k)?;
        let dc = s.projection_strain(k)?;
        for spin in 0..l.spins {
            let bands = &frozen.state.bands[spin][k];
            let (density, weighted) = (bands.density(), bands.weighted(l.reference));
            let w = l.degeneracy * l.weights[k];
            for &term in &terms {
                let dm = s.matrix_strain(spin, k, term, &frozen.fields)?;
                // P1 enters the energy through -W dS, every other term through D dH.
                let (m, sign, to) = match term {
                    Term::Overlap => (&weighted, -1., &mut out.overlap),
                    Term::Kinetic => (&density, 1., &mut out.kinetic),
                    Term::Local => (&density, 1., &mut out.local),
                    Term::ShortRange => (&density, 1., &mut out.short_range),
                    Term::KineticDensity => (&density, 1., &mut out.kinetic_density),
                };
                for a in 0..3 {
                    for b in 0..3 {
                        to[a][b] += sign * w * trace(m, &dm[a][b]);
                    }
                }
            }
            let hubbard = frozen.hubbard.as_ref().map(|o| &o[spin]);
            let operator = projector_operator(s, &frozen.fields, hubbard);
            let cd = c.adjoint() * &density;
            let z = &operator * &cd - s.augmentation() * c.adjoint() * &weighted;
            for a in 0..3 {
                for b in 0..3 {
                    // `Re tr[dC Z]` with `dC` (m x p) and `Z` (p x m).
                    out.projection[a][b] += 2. * w * trace(&dc[a][b], &z);
                }
            }
        }
        if l.request.spinor {
            // The spin-orbit matrix moves with the strain only through its own projections: no
            // `dS`, no hat, no `W`. `D` is the spinor density in the doubled frame.
            for bands in frozen.state.bands.iter().map(|spin| &spin[k]) {
                let w = l.degeneracy * l.weights[k];
                spin_orbit::add_k(s, &mut out.spin_orbit, w, k, &bands.density())?;
            }
        }
    }
    Ok(out)
}
