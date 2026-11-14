//! S-1..S-6: the matrix terms of the stress, `sum_{s,k} deg w_k Re tr[D dh_ab - W dS_ab]` (S-6 is a
//! meta-GGA's kinetic-density operator, [`Term::KineticDensity`]).
//!
//! The force engine's algebra with the strain tangent: the overlap enters through `-W dS`, the
//! kinetic, local and short-range matrices through `D dM`, and the projections through
//! `Z = H_C C^+ D - q C^+ W` (`p x m`, `H_C = D_ion + sum_a m_a B_a (+ O_s)`):
//! `dE_C = 2 Re sum_{mu,p} dC_ab[mu,p] Z[p,mu]`. Nothing is owner-masked; the +U operator joins
//! `D_ion` here and nowhere else, so `-W dS` carries it through the bands.
use super::{spin_orbit, SpinMatrices, Strain, Tensor};
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

/// What one k-point contributes: every number is already weighted by `deg w_k` and signed, one per
/// strain direction, produced on a k-point lane.
struct KData {
    /// Per spin channel, per term in [`Term::of`] order, per direction.
    terms: Vec<Vec<Vec<f64>>>,
    /// Per spin channel, per direction.
    projection: Vec<Vec<f64>>,
    /// The spin-orbit share of each spin channel's bands, per direction (a spinor; else empty).
    spin_orbit: Vec<Vec<f64>>,
}

pub fn contract<S: Strain>(s: &S, frozen: &Frozen<S::Fields>) -> Result<Terms> {
    let l = s.layout();
    let directions = s.directions();
    let n = directions.len();
    let terms = Term::of(&l.request.functional)?;
    let mut sums: Vec<Vec<f64>> = vec![vec![0.; n]; 7];
    // What one point holds: C and the n tangents dC (m x p), and what the provider works with.
    let per_lane = crate::periodic::gradient::matrix_bytes(2 + n, l.m, l.p)?
        .checked_add(s.lane_bytes(terms.len())?)
        .ok_or(crate::Error::Overflow)?;
    let produce = |k: usize| -> Result<KData> {
        let w = l.degeneracy * l.weights[k];
        let c = s.projection(k)?;
        let dc = s.projection_strain(k, directions.as_slice())?;
        let matrices: Vec<SpinMatrices> = (0..l.spins)
            .map(|spin| {
                let bands = &frozen.state.bands[spin][k];
                SpinMatrices {
                    density: bands.density(),
                    weighted: bands.weighted(l.reference),
                }
            })
            .collect();
        let values =
            s.matrix_strain(k, &frozen.fields, &matrices, &terms, directions.as_slice())?;
        let mut weighted = Vec::with_capacity(l.spins);
        let mut projection = Vec::with_capacity(l.spins);
        for (spin, per_term) in values.into_iter().enumerate() {
            weighted.push(
                terms
                    .iter()
                    .zip(per_term)
                    .map(|(&term, per_direction)| {
                        // P1 enters the energy through -W dS, every other term through D dH.
                        let sign = if term == Term::Overlap { -1. } else { 1. };
                        per_direction.into_iter().map(|x| sign * w * x).collect()
                    })
                    .collect(),
            );
            let hubbard = frozen.hubbard.as_ref().map(|o| &o[spin]);
            let operator = projector_operator(s, &frozen.fields, hubbard);
            let cd = c.adjoint() * &matrices[spin].density;
            let z = &operator * &cd - s.augmentation() * c.adjoint() * &matrices[spin].weighted;
            // `Re tr[dC Z]` with `dC` (m x p) and `Z` (p x m).
            projection.push(dc.iter().map(|dc| 2. * w * trace(dc, &z)).collect());
        }
        // The spin-orbit matrix moves with the strain only through its own projections: no
        // `dS`, no hat, no `W`. `D` is the spinor density in the doubled frame.
        let spin_orbit = if l.request.spinor {
            matrices
                .iter()
                .map(|m| spin_orbit::share_k(s, w, k, &m.density))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(KData {
            terms: weighted,
            projection,
            spin_orbit,
        })
    };
    // The contraction of each point in k order, as the serial loop ran it.
    let consume = |_: usize, data: KData| -> Result<()> {
        let add =
            |to: &mut Vec<f64>, from: &[f64]| to.iter_mut().zip(from).for_each(|(t, f)| *t += f);
        for (per_term, projection) in data.terms.iter().zip(&data.projection) {
            for (&term, values) in terms.iter().zip(per_term) {
                let slot = match term {
                    Term::Overlap => 0,
                    Term::Kinetic => 1,
                    Term::Local => 2,
                    Term::ShortRange => 3,
                    Term::KineticDensity => 5,
                };
                add(&mut sums[slot], values);
            }
            add(&mut sums[4], projection);
        }
        for share in &data.spin_orbit {
            add(&mut sums[6], share);
        }
        Ok(())
    };
    crate::periodic::gradient::kpoint_windows(
        s.context(),
        l.weights.len(),
        per_lane,
        produce,
        consume,
    )?;
    let [overlap, kinetic, local, short_range, projection, kinetic_density, spin_orbit] =
        std::array::from_fn(|i| directions.assemble(&sums[i]));
    Ok(Terms {
        overlap,
        kinetic,
        local,
        short_range,
        projection,
        kinetic_density,
        spin_orbit,
    })
}
