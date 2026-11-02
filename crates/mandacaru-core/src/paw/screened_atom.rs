//! The one-center terms of a screened hybrid for one species, independent of any
//! molecule or crystal: `Delta W` under `erfc(omega R)/R`, built from the bare
//! tensor minus its `erf` part, and the reference atom's frozen short-range
//! semilocal exchange.
//!
//! The input is a species (its radial tables as a [`SpeciesView`], with the
//! reference density matrix of every channel), its projector layout and the
//! screening `omega`; nothing here knows how the projectors are projected on a
//! basis, so the molecular and the periodic engines call it alike.
use super::{
    invalid,
    onecenter::{Basis as ProjectorBasis, Channel as WaveChannel},
    onecenter_terms::{frozen_exchange_terms, Occupied},
    pair_tables::{check, pair_tensor, Projector, SpeciesView, Tables, POINTS},
    screened_polarization::Polarization,
};
use crate::{
    math::{linalg::real_eigen, spline::simpson},
    product, Context, Result,
};
use nalgebra::DMatrix;

/// The long-range part `erf(omega R)/R` of the Coulomb kernel, contracted once
/// with every pair table: `ae[l][pair][i] = w_i (K_l (w rho_pair))_i`, so that the
/// long-range form of two pair densities is a dot product.
pub struct LongRange {
    pub ae: Vec<Vec<Vec<f64>>>,
    pub compensated: Vec<Vec<Vec<f64>>>,
}
impl LongRange {
    fn new(ctx: &Context, t: &Tables, omega: f64) -> Result<Self> {
        let levels: Vec<usize> = (0..=t.lmax).collect();
        let kernels = crate::paw::onecenter_radial::long_range_radial_kernel(
            ctx, &t.r, &t.r, &levels, omega, None,
        )?;
        let weights = (0..POINTS)
            .map(|i| {
                let mut unit = vec![0.; POINTS];
                unit[i] = 1.;
                simpson(&t.r, &unit).map(|w| w * t.r[i] * t.r[i])
            })
            .collect::<Result<Vec<_>>>()?;
        let contract = |kernel: &[f64], pair: &[f64]| -> Vec<f64> {
            let weighted: Vec<f64> = pair.iter().zip(&weights).map(|(v, w)| v * w).collect();
            (0..POINTS)
                .map(|i| {
                    weights[i]
                        * kernel[i * POINTS..(i + 1) * POINTS]
                            .iter()
                            .zip(&weighted)
                            .map(|(k, v)| k * v)
                            .sum::<f64>()
                })
                .collect()
        };
        Ok(Self {
            ae: kernels
                .iter()
                .map(|kernel| {
                    t.ae_pair
                        .iter()
                        .map(|pair| contract(kernel, pair))
                        .collect()
                })
                .collect(),
            compensated: kernels
                .iter()
                .zip(&t.compensated)
                .map(|(kernel, pairs)| pairs.iter().map(|pair| contract(kernel, pair)).collect())
                .collect(),
        })
    }
    /// The long-range form of the pair densities `ab` and `cd` at level `l`, from
    /// their raw tables and the contracted ones, averaged over the two orders (the
    /// kernel is symmetric; this removes the round-off).
    pub fn form(raw: [&[f64]; 2], contracted: [&[f64]; 2]) -> f64 {
        let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(a, b)| a * b).sum() };
        0.5 * (dot(raw[0], contracted[1]) + dot(raw[1], contracted[0]))
    }
}
/// The one-center terms of a screened hybrid for one species, geometry independent.
pub struct ScreenedAtom {
    /// Low-rank factors of `Delta W^SR[(ab), (cd)]` in the projector layout.
    pub factors: Vec<DMatrix<f64>>,
    pub weights: Vec<f64>,
    /// The frozen short-range semilocal exchange potential `V` of the reference
    /// atom, `p x p` in the projector layout.
    pub v: DMatrix<f64>,
    /// `X0 - V . D0`, summed over the channels.
    pub constant: f64,
    /// The spin-scaled remainder of the spheres' semilocal short-range exchange,
    /// which an open shell adds to the unpolarized linearization above.
    pub polarization: Polarization,
}
/// `Delta W` under `erfc(omega R)/R`, the reference atom's frozen short-range
/// semilocal exchange and its low-rank factors, for `terms`' projector layout.
pub fn screened(
    ctx: &Context,
    species: &SpeciesView<'_>,
    projectors: &[Projector],
    omega: f64,
) -> Result<ScreenedAtom> {
    if !omega.is_finite() || omega <= 0. {
        return Err(invalid("the screening of a hybrid must be positive"));
    }
    check(species)?;
    let p = projectors.len();
    let pp = product(&[p, p])?;
    let lmax = projectors.iter().map(|p| p.l).max().unwrap_or(0) * 2;
    let _memory = ctx.reserve(
        product(&[pp, pp, 8, 6])?
            .checked_add(product(&[pp, POINTS, 8, 3 * (lmax + 1) + 3])?)
            .and_then(|v| v.checked_add(product(&[POINTS, POINTS, lmax + 1, 8]).ok()?))
            .ok_or(crate::Error::Overflow)?,
    )?;
    let tables = Tables::new(species, projectors)?;
    let long_range = LongRange::new(ctx, &tables, omega)?;
    let w = pair_tensor(&tables, Some(&long_range))?;
    let (values, vectors) = real_eigen(w)?;
    let floor = values.iter().map(|v| v.abs()).fold(0., f64::max) * 1e-10;
    let keep: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > floor)
        .map(|(i, _)| i)
        .collect();
    let polarization = Polarization::new(species, projectors, omega)?;
    let (constant, blocks) = frozen_exchange(ctx, species, omega)?;
    let mut v = DMatrix::zeros(p, p);
    for (a, pa) in projectors.iter().enumerate() {
        for (b, pb) in projectors.iter().enumerate() {
            if pa.channel == pb.channel && pa.m == pb.m {
                v[(a, b)] = blocks[pa.channel][(pa.radial, pb.radial)];
            }
        }
    }
    Ok(ScreenedAtom {
        factors: keep
            .iter()
            .map(|&k| {
                DMatrix::from_fn(p, p, |a, b| {
                    0.5 * (vectors[(a * p + b, k)] + vectors[(b * p + a, k)])
                })
            })
            .collect(),
        weights: keep.iter().map(|&k| values[k]).collect(),
        v,
        constant,
        polarization,
    })
}
/// `(X0 - V . D0, [V_channel])` of the reference atom in the projector basis of each
/// channel: the short-range semilocal exchange of the all-electron valence density
/// minus the smooth one, linearized at the reference occupations.
///
/// The reference density matrix of a channel need not be diagonal in the projector
/// basis, so its waves are rotated to the eigenvectors of that matrix, where it is
/// the occupation list [`frozen_exchange_terms`] takes, and `V` is rotated back.
fn frozen_exchange(
    ctx: &Context,
    species: &SpeciesView<'_>,
    omega: f64,
) -> Result<(f64, Vec<DMatrix<f64>>)> {
    let grid = species.knots;
    let n = grid.len();
    let mut levels: Vec<usize> = species.channels.iter().map(|c| c.l).collect();
    levels.sort_unstable();
    if levels.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid(
            "the screened exchange of a PAW species with two channels of one angular momentum is not implemented",
        ));
    }
    let mut rotated = Vec::new();
    for ch in &species.channels {
        let refs = ch.ae.len();
        let eigen = nalgebra::SymmetricEigen::new(ch.reference.clone());
        let mix = |waves: &[&[f64]]| -> Vec<f64> {
            let mut out = vec![0.; refs * n];
            for k in 0..refs {
                for (i, wave) in waves.iter().enumerate() {
                    let weight = eigen.eigenvectors[(i, k)];
                    for (slot, value) in out[k * n..(k + 1) * n].iter_mut().zip(wave.iter()) {
                        *slot += weight * value;
                    }
                }
            }
            out
        };
        if eigen.eigenvalues.iter().any(|o| *o < -1e-10) {
            return Err(invalid("a PAW reference density matrix must be positive"));
        }
        let occupations: Vec<f64> = eigen
            .eigenvalues
            .iter()
            .map(|o| o.max(0.) * (2 * ch.l + 1) as f64)
            .collect();
        let identity: Vec<f64> = DMatrix::<f64>::identity(refs, refs)
            .transpose()
            .as_slice()
            .to_vec();
        rotated.push((
            mix(&ch.ae),
            mix(&ch.ps),
            occupations,
            identity,
            eigen.eigenvectors,
        ));
    }
    let occupied: Vec<Occupied<'_>> = species
        .channels
        .iter()
        .zip(&rotated)
        .map(|(ch, (ae, ps, occupations, identity, _))| Occupied {
            channel: WaveChannel {
                l: ch.l,
                waves: ch.ae.len(),
                ae,
                ps,
                vanderbilt: identity,
                r_cut: ch.radius,
            },
            occupations,
            projector_radius: ch.radius,
        })
        .collect();
    let (constant, blocks) =
        frozen_exchange_terms(ctx, grid, &occupied, omega, ProjectorBasis::Dual)?;
    let matrices = species
        .channels
        .iter()
        .zip(&rotated)
        .map(|(ch, (.., vectors))| {
            let block = blocks
                .iter()
                .find(|b| b.l == ch.l)
                .ok_or_else(|| invalid("a channel has no frozen exchange block"))?;
            Ok(vectors * &block.matrix * vectors.transpose())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((constant, matrices))
}
