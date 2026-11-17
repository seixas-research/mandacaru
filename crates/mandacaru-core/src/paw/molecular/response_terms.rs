//! The one-center pieces of a PAW-LCAO basis that linear response reads: the position
//! operator, the gradient (through `paw::momentum`) and the count of the one-center Coulomb
//! factors.
//!
//! Inside a sphere the all-electron pair density differs from the smooth one by
//! `sum_ij <chi|p_i> (phi_i phi_j - phi~_i phi~_j) <p_j|chi>`. Its integral against `r - O` is
//! `(R_A - O) q_ij + d_ij`, with `q_ij` the overlap correction the orthonormal frame is built
//! with and `d_ij` the L = 1 multipole blocks of the compensation charges (real harmonics
//! `r Y_1m = sqrt(3 / 4 pi) {y, z, x}` for `m = -1, 0, 1`), so `d = sqrt(4 pi / 3) Delta^(1)`.
//! These are exactly the moments the compensation charges give the smooth density, so the
//! position integrals of the basis carry the all-electron dipole of every pair density.
use super::Setup;
use crate::{
    paw::momentum::{one_center_gradient, SphereWaves},
    Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// One atom's blocks in its projector basis: `q` and the Cartesian `d_x, d_y, d_z`.
pub(crate) struct PositionBlocks {
    pub overlap: DMatrix<f64>,
    pub dipole: [DMatrix<f64>; 3],
}

impl Setup {
    /// How many of the Hartree-Fock pair factors (`MolecularCorrections::pairs`, in order) are the
    /// one-center `DeltaW` tensors of the atoms; the compensation-charge factors follow them.
    pub(crate) fn one_center_factor_count(&self) -> usize {
        self.atoms
            .iter()
            .map(|&s| self.species_terms[s].factors.len())
            .sum()
    }
    /// The one-center position blocks of every atom, in the order of the geometry.
    pub(crate) fn position_blocks(&self) -> Vec<PositionBlocks> {
        let scale = (4. * PI / 3.).sqrt();
        self.atoms
            .iter()
            .map(|&s| {
                let terms = &self.species_terms[s];
                let p = terms.overlap.nrows();
                let mut dipole = [
                    DMatrix::zeros(p, p),
                    DMatrix::zeros(p, p),
                    DMatrix::zeros(p, p),
                ];
                for multipole in terms.multipoles.iter().filter(|m| m.l == 1) {
                    let axis = match multipole.m {
                        -1 => 1,
                        0 => 2,
                        _ => 0,
                    };
                    dipole[axis] += &multipole.block * scale;
                }
                PositionBlocks {
                    overlap: terms.overlap.clone(),
                    dipole,
                }
            })
            .collect()
    }

    /// The one-center part of the gradient operator of every atom, in its projector basis
    /// (`paw::momentum::one_center_gradient`, the routine the crystal's momentum shares).
    pub(crate) fn gradient_blocks(&self, ctx: &crate::Context) -> Result<Vec<[DMatrix<f64>; 3]>> {
        let mut out = Vec::with_capacity(self.atoms.len());
        for &s in &self.atoms {
            let species = &self.species[s];
            let channels: Vec<_> = species
                .channels
                .iter()
                .map(|c| SphereWaves {
                    radius: c.radius,
                    ae: &c.ae,
                    ps: &c.ps,
                })
                .collect();
            out.push(one_center_gradient(
                ctx,
                species.ionic.knots.as_slice(),
                &channels,
                &self.species_terms[s].projectors,
            )?);
        }
        Ok(out)
    }
}

/// One sphere's radial tables for the second change of its one-center exchange-correlation
/// energy ([`XcSpheres`]), on the knots inside the sphere.
pub struct XcSphere {
    /// The knots (Bohr) and the trapezoid weights times `r^2`.
    pub r: Vec<f64>,
    weights: Vec<f64>,
    /// The reference densities (all-electron, smooth) and their radial slopes.
    pub density: [Vec<f64>; 2],
    pub slope: [Vec<f64>; 2],
    /// Per projector of the atom's layout: `(l, m)` and the all-electron and smooth partial waves
    /// `R(r)` with their slopes, in the projector basis of the projections.
    pub harmonics: Vec<(usize, i32)>,
    pub waves: [Vec<Vec<f64>>; 2],
    pub wave_slopes: [Vec<Vec<f64>>; 2],
    /// The linearized one-center potential the energy uses (`Channel::xc`, every `m`), in the layout.
    pub linearized: DMatrix<f64>,
    /// Whether the species' exchange takes the relativistic factor.
    pub relativistic: bool,
    /// The point kernel at the unpolarized reference, per knot: first and second partials in
    /// `(up, down, uu, ud, dd)`, all-electron and smooth.
    first: [Vec<[f64; 5]>; 2],
    second: [Vec<[[f64; 5]; 5]>; 2],
    /// `(L, M, G)`, `G_ab = int Y_a Y_b Y_LM dOmega` over the layout.
    gaunts: Vec<(usize, i32, DMatrix<f64>)>,
}

/// The spheres of a molecule for the one-center exchange-correlation part of the excitation
/// diagnostic (`Setup::xc_spheres`), one per species.
pub struct XcSpheres {
    pub spheres: Vec<XcSphere>,
    atoms: Vec<usize>,
}

/// Below this density (per Bohr^3) a knot does not enter the kernel: the point functional is
/// undefined at zero and such a knot carries no response.
const XC_FLOOR: f64 = 1e-10;

impl Setup {
    /// The [`XcSpheres`] of the semilocal `functional` (`"lda"` or `"pbe"`): per species, the
    /// point kernel of `functional` at its reference atom's spherical, unpolarized densities
    /// (the all-electron one with its core, the smooth one with the smooth core) on the knots
    /// inside its sphere, the partial waves and the real Gaunt blocks of their products.
    pub fn xc_spheres(&self, functional: &str) -> Result<XcSpheres> {
        use crate::math::radial::gradient;
        if !["lda", "pbe"].contains(&functional) {
            return Err(crate::Error::Unsupported(format!(
                "the one-center exchange-correlation kernel is built for lda and pbe, not {functional}"
            )));
        }
        let rule = super::angular_rule(24, 48)?;
        let mut spheres = Vec::with_capacity(self.species.len());
        for (species, terms) in self.species.iter().zip(&self.species_terms) {
            let reference = &species.xc_reference;
            let n = reference.inside;
            let knots = &species.ionic.knots.as_slice()[..n];
            let r = knots.to_vec();
            let mut weights = vec![0.; n];
            for k in 0..n.saturating_sub(1) {
                let half = 0.5 * (r[k + 1] - r[k]);
                weights[k] += half * r[k] * r[k];
                weights[k + 1] += half * r[k + 1] * r[k + 1];
            }
            let density = [
                reference.all_electron[..n].to_vec(),
                reference.smooth[..n].to_vec(),
            ];
            let slope = [gradient(&r, &density[0])?, gradient(&r, &density[1])?];
            let layout = &terms.projectors;
            let harmonics = layout.iter().map(|p| (p.l, p.m)).collect::<Vec<_>>();
            let wave = |side: usize, a: usize| -> Vec<f64> {
                let channel = &species.channels[layout[a].channel];
                let waves = if side == 0 { &channel.ae } else { &channel.ps };
                waves[layout[a].radial][..n].to_vec()
            };
            let waves: [Vec<Vec<f64>>; 2] =
                std::array::from_fn(|side| (0..layout.len()).map(|a| wave(side, a)).collect());
            let wave_slopes = [
                waves[0]
                    .iter()
                    .map(|w| gradient(&r, w))
                    .collect::<Result<Vec<_>>>()?,
                waves[1]
                    .iter()
                    .map(|w| gradient(&r, w))
                    .collect::<Result<Vec<_>>>()?,
            ];
            let linearized = DMatrix::from_fn(layout.len(), layout.len(), |a, b| {
                let (pa, pb) = (&layout[a], &layout[b]);
                if pa.channel == pb.channel && pa.m == pb.m {
                    species.channels[pa.channel].xc[(pa.radial, pb.radial)]
                } else {
                    0.
                }
            });
            let mut first = [vec![[0.; 5]; n], vec![[0.; 5]; n]];
            let mut second = [vec![[[0.; 5]; 5]; n], vec![[[0.; 5]; 5]; n]];
            for side in 0..2 {
                for k in 0..n {
                    let rho = density[side][k];
                    if rho < XC_FLOOR {
                        continue;
                    }
                    let contraction = 0.25 * slope[side][k] * slope[side][k];
                    let (_, g, h) = crate::models::xc::point_hessian(
                        functional,
                        [0.5 * rho, 0.5 * rho, contraction, contraction, contraction],
                        species.relativistic_exchange,
                    )?;
                    first[side][k] = g;
                    second[side][k] = h;
                }
            }
            let lmax = 2 * harmonics.iter().map(|h| h.0).max().unwrap_or(0);
            let mut gaunts = Vec::new();
            for big_l in 0..=lmax {
                for big_m in -(big_l as i32)..=big_l as i32 {
                    let g = DMatrix::from_fn(layout.len(), layout.len(), |a, b| {
                        rule.iter()
                            .map(|(u, w)| {
                                w * super::harmonic(harmonics[a].0, harmonics[a].1, *u)
                                    * super::harmonic(harmonics[b].0, harmonics[b].1, *u)
                                    * super::harmonic(big_l, big_m, *u)
                            })
                            .sum()
                    });
                    if g.amax() > 1e-12 {
                        gaunts.push((big_l, big_m, g));
                    }
                }
            }
            spheres.push(XcSphere {
                r,
                weights,
                density,
                slope,
                harmonics,
                waves,
                wave_slopes,
                linearized,
                relativistic: species.relativistic_exchange,
                first,
                second,
                gaunts,
            });
        }
        Ok(XcSpheres {
            spheres,
            atoms: self.atoms.clone(),
        })
    }
}

impl XcSpheres {
    /// The sphere of `atom` (geometry order).
    pub fn sphere(&self, atom: usize) -> Result<&XcSphere> {
        self.atoms
            .get(atom)
            .map(|&s| &self.spheres[s])
            .ok_or_else(|| super::invalid("atom out of range"))
    }
    /// `d^2/de^2 (E[n^1] - E[n~^1])` at `e = 0` of `atom`'s sphere, Hartree, with the spin
    /// densities moved from the reference's `n/2` each by the changes of the projected density
    /// matrices `up` and `down` (the atom's projector layout): the second-order one-center
    /// exchange-correlation term that the Kohn-Sham energy, linearized at the reference atom,
    /// leaves out. With `dn_s = sum_LM a_s,LM(r) Y_LM`, a spherical reference makes every
    /// angular integral diagonal in `LM`:
    ///
    /// ```text
    /// sum_LM int r^2 dr [ q^T H q + 2 sum_{st} e_{st} (a_s' a_t' + L(L+1) a_s a_t / r^2) ],
    /// q = (a_up, a_dn, 2 n_up' a_up', n_up' a_dn' + n_dn' a_up', 2 n_dn' a_dn'),
    /// ```
    ///
    /// with `H` and `e_{st}` the point kernel's second and contraction partials (`st` = uu, ud, dd
    /// with weights 1, 1, 1 for the terms `a_up a_up`, `a_up a_dn`, `a_dn a_dn`).
    pub fn second_change(
        &self,
        atom: usize,
        up: &DMatrix<f64>,
        down: &DMatrix<f64>,
    ) -> Result<f64> {
        let sphere = self.sphere(atom)?;
        let p = sphere.harmonics.len();
        if up.shape() != (p, p) || down.shape() != (p, p) {
            return Err(super::invalid(
                "the sphere's density changes must span its projectors",
            ));
        }
        let n = sphere.r.len();
        let mut total = 0.;
        for (side, sign) in [(0, 1.), (1, -1.)] {
            let (waves, slopes) = (&sphere.waves[side], &sphere.wave_slopes[side]);
            for (big_l, _, g) in &sphere.gaunts {
                let lambda = (big_l * (big_l + 1)) as f64;
                let coefficient = |d: &DMatrix<f64>| -> (Vec<f64>, Vec<f64>) {
                    let mut value = vec![0.; n];
                    let mut slope = vec![0.; n];
                    for a in 0..p {
                        for b in 0..p {
                            let c = d[(a, b)] * g[(a, b)];
                            if c == 0. {
                                continue;
                            }
                            for k in 0..n {
                                value[k] += c * waves[a][k] * waves[b][k];
                                slope[k] +=
                                    c * (slopes[a][k] * waves[b][k] + waves[a][k] * slopes[b][k]);
                            }
                        }
                    }
                    (value, slope)
                };
                let ((a, da), (b, db)) = (coefficient(up), coefficient(down));
                for k in 0..n {
                    let (f, h) = (&sphere.first[side][k], &sphere.second[side][k]);
                    let half = 0.5 * sphere.slope[side][k];
                    let q = [
                        a[k],
                        b[k],
                        2. * half * da[k],
                        half * (da[k] + db[k]),
                        2. * half * db[k],
                    ];
                    let quadratic: f64 = (0..5)
                        .map(|i| (0..5).map(|j| q[i] * h[i][j] * q[j]).sum::<f64>())
                        .sum();
                    let angular = lambda / (sphere.r[k] * sphere.r[k]).max(f64::MIN_POSITIVE);
                    let gradient = 2.
                        * (f[2] * (da[k] * da[k] + angular * a[k] * a[k])
                            + f[3] * (da[k] * db[k] + angular * a[k] * b[k])
                            + f[4] * (db[k] * db[k] + angular * b[k] * b[k]));
                    total += sign * sphere.weights[k] * (quadratic + gradient);
                }
            }
        }
        crate::finite(&[total], "one-center exchange-correlation change")?;
        Ok(total)
    }
}
