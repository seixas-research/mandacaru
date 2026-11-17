//! The macroscopic dielectric function of an insulating crystal at `q -> 0` in the frequency
//! domain (`TDDFT_PLAN.md`, section 4, stage T5a): independent particles and the random-phase
//! approximation with local fields, the first step of the Bethe-Salpeter route.
//!
//! On the whole Gamma-anchored mesh (closed under `-k`, uniform weights `w_k = 1 / N_k`) the
//! Kohn-Sham polarizability at the complex frequency `z = omega + i eta` is
//!
//! ```text
//! chi0_{GG'}(z) = (g / Omega) sum_{s,k} w_k sum_{v,c} [ rho_vc(G) rho_vc(G')^* / (z - D)
//!                                                     - rho_cv(G) rho_cv(G')^* / (z + D) ]
//! ```
//!
//! with `D = e_c - e_v + scissor`, `g` the band degeneracy and `rho_ab(G) = <a| e^{-iG.r} |b>`
//! the pair densities of the screening code (`screening::transitions`, the complete PAW
//! augmentation), at `q = 0`. The `G = 0` row is the long-wavelength limit along each Cartesian
//! axis `a`: `rho_vc(q e_a, 0) / q -> P^a_vc / (e_c - e_v)` and `rho_cv -> -P^a_cv / (e_c - e_v)`
//! (k . p), with `P` the static-basis momentum of [`super::momentum`] (T6b0), the matrix the
//! real-time velocity gauge reads. The dipole keeps the Kohn-Sham gap, so a scissor moves the
//! poles only (the length-form dipole is invariant under it).
//!
//! The macroscopic tensor is `eps^{ab}(z) = delta_ab - 4 pi chibar^{ab}_00(z)`, with `chibar` the
//! response to the total macroscopic field: `chibar = chi0 + chi0 vbar chibar`, `vbar` the
//! Coulomb interaction `4 pi / |G|^2` without its `G = 0` term (the local fields), on the vectors
//! `0 < |G| <= cutoff`. Independent particles keep the head alone. For a field along `e`,
//! `e . eps . e = 1 / eps^-1_00(q -> 0 along e)` (Adler, Wiser): the quantity the cRPA head and
//! the real-time run with `kernel="hartree"` compute.
//!
//! The first moment: for large `z`, `eps^{ab} -> delta_ab - 4 pi N^{ab} / (Omega z^2)` with
//! `N^{ab} = g sum w_k 2 Re[P^a_vc P^b_cv] D / (e_c - e_v)^2`, the diamagnetic band sum of
//! [`super::momentum`] when the scissor is zero, local fields or not (the wings carry no `1 / z`
//! term on a mesh closed under `-k`). It is reported as `sum_rule`.
//!
//! Every transition is computed alone and every frequency on its own lane, each a serial
//! computation, so the result depends neither on the thread count nor on the k-point lanes.
//!
//! A spinor crystal (T8) has `g = 1`, its pair densities are charge densities summed over the two
//! components and its `P` is `p`'s, diagonal in the spin, plus the velocity `i [h^SO, r]` of the
//! spin-orbit term when the crystal has it ([`super::momentum`]); without the term a
//! Kramers-unpolarized spinor crystal gives the collinear tensor.
//!
//! Refused: metals and fractional occupations (the intraband term), hybrids,
//! DFT+U and meta-GGAs (the velocity of a nonlocal exchange potential, of the +U operator and of
//! the `v_tau` kinetic term is not in `P`), a mesh not closed under `-k`,
//! `kernel="adiabatic"`. `kernel="bse"` is the Bethe-Salpeter equation of [`super::bse`], on
//! the same columns, and `kernel="long-range"` the local fields with the model kernel
//! `-alpha / |q + G|^2` (T5c; [`macroscopic`]).
use super::{
    super::{
        berry::{Bands, Metal},
        bloch_mesh::{MeshIndex, Sampler},
        invalid,
        response::setup::Setup,
        screening::{
            medium::within,
            transitions::{Augmentation, Engine, Remainders},
        },
        setup::pulay::Mesh,
    },
    momentum::{band_momenta, check_inversion_closed, one_center_momentum},
};
use crate::{
    math::{gemm::complex_product, storage::Precision},
    product,
    quantum::C64,
    Error, Result,
};
use nalgebra::{DMatrix, DMatrixView};
use std::f64::consts::PI;

/// Transitions one product of the frequency loop multiplies at a time.
const CHUNK: usize = 2048;
/// The largest local-field vector when none is given (Bohr^-1): the screening code's default.
pub const LOCAL_FIELD_CUTOFF: f64 = 3.;

/// The response the dielectric function includes. The names are those of the molecular
/// excitations and of the real-time run: `"hartree"` is the random-phase approximation with
/// local fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Independent particles: the head of `chi0` alone.
    Independent,
    /// The Coulomb kernel with local fields (RPA).
    Hartree,
    /// The Bethe-Salpeter equation (Tamm-Dancoff, singlets): local fields and the statically
    /// screened electron-hole attraction ([`super::bse`]).
    Bse,
    /// The local fields and the long-range model kernel `f_xc(q + G) = -alpha / |q + G|^2`
    /// (diagonal in `G`; T5c), a comparison for the Bethe-Salpeter equation.
    LongRange,
}

impl Kernel {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "independent" => Ok(Self::Independent),
            "hartree" => Ok(Self::Hartree),
            "adiabatic" => Err(Error::Unsupported(
                "kernel='adiabatic' (the exchange-correlation kernel in the crystal's Dyson \
                 equation) is not built; get_real_time propagates with it"
                    .into(),
            )),
            "bse" => Ok(Self::Bse),
            "long-range" => Ok(Self::LongRange),
            other => Err(invalid(format!(
                "unknown kernel {other:?}; expected 'independent', 'hartree' (the RPA with \
                 local fields), 'bse' (the Bethe-Salpeter equation) or 'long-range' (the \
                 long-range kernel)"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Hartree => "hartree",
            Self::Bse => "bse",
            Self::LongRange => "long-range",
        }
    }
}

/// What to compute, in atomic units.
#[derive(Clone, Debug)]
pub struct Request {
    /// Real frequencies `omega` (Hartree).
    pub frequencies: Vec<f64>,
    /// `eta` (Hartree, at least zero): the function is evaluated at `omega + i eta`.
    pub damping: f64,
    pub kernel: Kernel,
    /// The largest `|G|` of the local fields (Bohr^-1).
    pub cutoff: f64,
    /// `(valence, conduction)` bands of each channel next to the gap; `None`: every band.
    pub window: Option<(usize, usize)>,
    /// Added to every transition energy (Hartree).
    pub scissor: f64,
    /// `alpha` of the long-range kernel; `None` fits it to the computed `eps_inf`:
    /// `alpha = 4.615 / eps_inf - 0.213` (Botti et al. 2004), `eps_inf` the isotropic average of
    /// the static tensor with local fields.
    pub alpha: Option<f64>,
}

impl Request {
    pub fn new(frequencies: Vec<f64>, damping: f64, kernel: Kernel) -> Self {
        Self {
            frequencies,
            damping,
            kernel,
            cutoff: LOCAL_FIELD_CUTOFF,
            window: None,
            scissor: 0.,
            alpha: None,
        }
    }
}

/// The macroscopic dielectric tensor of a crystal (module documentation).
#[derive(Clone, Debug)]
pub struct Dielectric {
    pub frequencies: Vec<f64>,
    pub damping: f64,
    pub kernel: Kernel,
    /// `eps^{ab}(omega + i eta)`, one row-major `3 x 3` tensor per frequency.
    pub tensor: Vec<[[C64; 3]; 3]>,
    /// `eps^{ab}(0)` without damping: the static (clamped-ion, electronic) dielectric tensor.
    pub static_tensor: [[f64; 3]; 3],
    /// The whole mesh.
    pub mesh: [usize; 3],
    /// The bands each channel contributes: `(valence, conduction)`.
    pub window: Vec<(usize, usize)>,
    /// Valence-to-conduction transitions over the mesh and the channels.
    pub transitions: usize,
    /// The local-field vectors (`G != 0`); zero for independent particles.
    pub vectors: usize,
    pub cutoff: f64,
    pub scissor: f64,
    /// The smallest transition energy `D` of the mesh (Hartree): the continuum's onset.
    pub onset: f64,
    /// `N^{ab}` of the first moment (module documentation).
    pub sum_rule: [[f64; 3]; 3],
    /// The valence electrons of the cell.
    pub electrons: f64,
    /// The cell volume (Bohr^3).
    pub volume: f64,
    /// The largest anti-Hermitian residual of `P` over the mesh.
    pub residual: f64,
    /// `alpha` of the long-range kernel (given or fitted); `None` for the other kernels.
    pub alpha: Option<f64>,
}

/// The frequencies of a request that names none: 0 to 20 eV in steps of 0.05 eV (Hartree).
pub fn default_frequencies() -> Vec<f64> {
    (0..=400)
        .map(|i| crate::units::from_user(0.05 * i as f64, crate::units::Quantity::Energy, false))
        .collect()
}

impl Dielectric {
    /// `e . eps . e` at each frequency along `direction` (normalized); `None`: the isotropic
    /// average, the trace over three.
    pub fn along(&self, direction: Option<[f64; 3]>) -> Result<Vec<C64>> {
        let pick = |t: &[[C64; 3]; 3], e: [f64; 3]| -> C64 {
            (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| t[a][b] * (e[a] * e[b]))
                .sum()
        };
        match direction {
            Some(d) => {
                let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !norm.is_finite() || norm == 0. {
                    return Err(invalid("the direction must be a finite nonzero vector"));
                }
                let e = d.map(|v| v / norm);
                Ok(self.tensor.iter().map(|t| pick(t, e)).collect())
            }
            None => Ok(self
                .tensor
                .iter()
                .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.)
                .collect()),
        }
    }
}

/// The loss function `-Im 1 / eps` of each value.
pub fn loss(values: &[C64]) -> Vec<f64> {
    values.iter().map(|e| -(C64::new(1., 0.) / e).im).collect()
}

/// The absorption coefficient `2 omega Im sqrt(eps) / c` (Bohr^-1; `omega` in Hartree, `c` the
/// speed of light in atomic units) at each frequency: the decay of the intensity of a plane wave
/// in the medium.
pub fn absorption(frequencies: &[f64], values: &[C64]) -> Vec<f64> {
    frequencies
        .iter()
        .zip(values)
        .map(|(w, e)| 2. * w * e.sqrt().im / crate::models::relativity::SPEED_OF_LIGHT)
        .collect()
}

/// The pair densities of every valence-to-conduction transition of the mesh, stored once for
/// all frequencies: column `2 t` is `rho_vc` and column `2 t + 1` is `rho_cv` of transition `t`,
/// each with the three long-wavelength head rows first (`P^a / (e_c - e_v)`, Bohr) and then the
/// local-field vectors. The exchange term of the Bethe-Salpeter equation reads the same columns.
pub struct Pairs {
    /// Head rows plus local-field vectors.
    pub rows: usize,
    values: Vec<C64>,
    /// `D` of each transition (Hartree, scissor included).
    pub energies: Vec<f64>,
    /// The local-field vectors `G` (Cartesian, Bohr^-1), in the order of the rows after the head.
    pub vectors: Vec<[f64; 3]>,
    /// `sqrt(4 pi) / |G|` of each local-field vector.
    coulomb: Vec<f64>,
    /// `g w_k / Omega`.
    scale: f64,
    pub mesh: [usize; 3],
    pub window: Vec<(usize, usize)>,
    pub onset: f64,
    pub sum_rule: [[f64; 3]; 3],
    pub electrons: f64,
    pub volume: f64,
    pub residual: f64,
    _stored: crate::Reservation,
}

fn check_crystal(sampler: &Sampler<'_>) -> Result<()> {
    let crystal = sampler.crystal;
    if crate::models::functional(&crystal.functional)?
        .screened_exchange
        .is_some()
    {
        return Err(Error::Unsupported(
            "the dielectric function of a hybrid crystal is not offered: the velocity of the \
             nonlocal exchange potential is not in the momentum matrix"
                .into(),
        ));
    }
    if crystal.hubbard.as_ref().is_some_and(|h| h.is_active())
        || crate::models::functional(&crystal.functional)?.requires_tau
    {
        return Err(Error::Unsupported(
            "the dielectric function of a DFT+U or meta-GGA crystal is not offered: the velocity \
             of its +U operator or of its v_tau kinetic term is not in the momentum matrix (and \
             its static response is not the band sum); get_real_time propagates both"
                .into(),
        ));
    }
    if crystal.precision != Precision::Float64 {
        return Err(invalid(
            "the dielectric function runs in float64 only (precision='float32' is refused)",
        ));
    }
    Ok(())
}

/// The dielectric tensor of the insulator `sampler` holds on `mesh` (whole, closed under `-k`).
pub fn dielectric(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    request: &Request,
) -> Result<Dielectric> {
    if request.kernel == Kernel::Bse {
        return super::bse::dielectric(sampler, mesh, request);
    }
    if request.kernel != Kernel::LongRange && request.alpha.is_some() {
        return Err(invalid(
            "alpha is the long-range kernel's: pass kernel='long-range'",
        ));
    }
    let pairs = pairs(sampler, mesh, request)?;
    let ctx = &sampler.crystal.context;
    let alpha = match request.kernel {
        Kernel::LongRange => Some(match request.alpha {
            Some(alpha) => alpha,
            None => {
                let t = pairs.tensor(C64::new(0., 0.))?;
                4.615 / ((t[0][0].re + t[1][1].re + t[2][2].re) / 3.) - 0.213
            }
        }),
        _ => None,
    };
    if let Some(alpha) = alpha {
        if !(alpha.is_finite() && alpha >= 0. && alpha < 4. * PI) {
            return Err(invalid(format!(
                "the long-range kernel needs 0 <= alpha < 4 pi (the body's v + f_xc stays \
                 repulsive); got {alpha}"
            )));
        }
    }
    let tensor = ctx.map_independent(request.frequencies.len() + 1, |j| {
        let z = match request.frequencies.get(j) {
            Some(omega) => C64::new(*omega, request.damping),
            None => C64::new(0., 0.),
        };
        macroscopic(
            &pairs.polarizability(z)?,
            &pairs.coulomb,
            alpha.unwrap_or(0.),
        )
    })?;
    let (tensor, last) = tensor.split_at(request.frequencies.len());
    let static_tensor = last[0].map(|row| row.map(|v| v.re));
    Ok(Dielectric {
        frequencies: request.frequencies.clone(),
        damping: request.damping,
        kernel: request.kernel,
        tensor: tensor.to_vec(),
        static_tensor,
        mesh: pairs.mesh,
        window: pairs.window.clone(),
        transitions: pairs.energies.len(),
        vectors: pairs.vectors.len(),
        cutoff: request.cutoff,
        scissor: request.scissor,
        onset: pairs.onset,
        sum_rule: pairs.sum_rule,
        electrons: pairs.electrons,
        volume: pairs.volume,
        residual: pairs.residual,
        alpha,
    })
}

/// The share of the independent-particle `eps(0) - 1` (its trace) that the transitions the
/// default window ([`default_window`]) leaves out may carry.
pub const STATIC_TOLERANCE: f64 = 1e-5;

/// The default `(valence, conduction)` window (one for every channel, as `Request::window`;
/// `TDDFT_PLAN.md`, section 4.5, and DECISIONS.md 2026-11-16, fifteenth session): the smallest
/// window that holds every transition of `mesh` at or below `highest` (Hartree, the top of a
/// requested energy range: `e_c - e_v + scissor <= highest`) and whose omitted transitions carry
/// at most `tolerance` ([`STATIC_TOLERANCE`] by default) of the independent-particle
/// `eps(0) - 1`, the trace of
/// `sum_t |P_cv|^2 / D_t^3` over every channel and mesh point. A range ending below most
/// transitions would otherwise drop their share of `eps_1` (on the H2 chain DZP, `eps_xx(0)`
/// 1.000 against 1.0716 with every band). For the range, a valence band belongs to the window
/// when its transition to the lowest conduction band reaches `highest` at some mesh point, a
/// conduction band when the transition from the highest valence band does; for the static
/// limit, conduction bands are added (every valence band held) until the omitted share is
/// within the tolerance, then valence bands are dropped from the bottom while it stays there.
/// The window keeps at least one band of each kind, the larger count of the two criteria and of
/// the channels, within every channel's bands.
pub fn default_window(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    highest: f64,
    scissor: f64,
    tolerance: f64,
) -> Result<(usize, usize)> {
    if !(highest.is_finite() && scissor.is_finite()) {
        return Err(invalid("the energy range and the scissor must be finite"));
    }
    if !(tolerance.is_finite() && tolerance >= 0.) {
        return Err(invalid(
            "the static tolerance must be finite and at least zero",
        ));
    }
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined)?;
    let bands = crystal.bands();
    let mut window: Vec<(usize, usize)> = vec![(1, 1); sampler.channels()];
    for &(lo, hi) in &manifold.ranges {
        if lo != 0 || hi == 0 || hi == bands {
            return Err(invalid(
                "the dielectric function needs occupied and empty bands in every spin channel",
            ));
        }
    }
    let valence = manifold.ranges.iter().map(|r| r.1).min().unwrap_or(1);
    let conduction = manifold
        .ranges
        .iter()
        .map(|r| bands - r.1)
        .min()
        .unwrap_or(1);
    // `share[(v, c)]`: the transitions' `|P|^2 / D^3`, by valence band counted down from the
    // highest (`v = 0` is the top) and conduction band counted up from the lowest.
    let mut share = DMatrix::<f64>::zeros(bands, bands);
    let one_center = one_center_momentum(ctx, sampler.setup)?;
    let grid_mesh = Mesh::new(ctx, &sampler.setup.grid)?;
    let all: Vec<Vec<usize>> = (0..sampler.channels())
        .map(|_| (0..bands).collect())
        .collect();
    let mut point = vec![0usize; sampler.channels()];
    sampler.visit(mesh, &all, |s, frame| {
        let hi = manifold.ranges[s].1;
        let levels = frame.energies.as_slice();
        let (top, bottom) = (levels[hi - 1], levels[hi]);
        let (v_range, c_range) = &mut window[s];
        while *v_range < hi && bottom - levels[hi - 1 - *v_range] + scissor <= highest {
            *v_range += 1;
        }
        while *c_range < bands - hi && levels[hi + *c_range] - top + scissor <= highest {
            *c_range += 1;
        }
        let fractional = mesh.point(point[s]);
        point[s] += 1;
        let k: [f64; 3] = (sampler.reciprocal * nalgebra::Vector3::from(fractional)).into();
        let momenta = band_momenta(ctx, sampler.setup, &grid_mesh, &frame, k, &one_center)?;
        for v in 0..hi {
            for c in hi..bands {
                let gap = levels[c] - levels[v] + scissor;
                if gap <= 0. {
                    return Err(invalid("an empty band lies below an occupied one"));
                }
                let weight: f64 = momenta.matrices.iter().map(|m| m[(c, v)].norm_sqr()).sum();
                share[(hi - 1 - v, c - hi)] += weight / gap.powi(3);
            }
        }
        Ok(())
    })?;
    let whole: f64 = share.iter().sum();
    let held = |v: usize, c: usize| -> f64 { share.view((0, 0), (v, c)).iter().sum() };
    let converged = |v: usize, c: usize| whole - held(v, c) <= tolerance * whole;
    let deepest = manifold.ranges.iter().map(|r| r.1).max().unwrap_or(1);
    let widest = manifold
        .ranges
        .iter()
        .map(|r| bands - r.1)
        .max()
        .unwrap_or(1);
    let mut c_static = 1;
    while c_static < widest && !converged(deepest, c_static) {
        c_static += 1;
    }
    let mut v_static = deepest;
    while v_static > 1 && converged(v_static - 1, c_static) {
        v_static -= 1;
    }
    let range = (
        window.iter().map(|w| w.0).max().unwrap_or(1),
        window.iter().map(|w| w.1).max().unwrap_or(1),
    );
    Ok((
        range.0.max(v_static).min(valence),
        range.1.max(c_static).min(conduction),
    ))
}

/// The window `(v, c)`, `v <= valence` and `c <= conduction`, of the most transitions per mesh
/// point for which `fits(v, c)` (the larger conduction count on a tie); `None` if not even
/// `(1, 1)` fits.
pub fn largest_window(
    valence: usize,
    conduction: usize,
    fits: impl Fn(usize, usize) -> bool,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for v in 1..=valence {
        // The fitting conduction counts of one `v` are a prefix: bisect for its end.
        let (mut lo, mut hi) = (0usize, conduction);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if fits(v, mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 {
            break;
        }
        if best.is_none_or(|(bv, bc)| v * lo > bv * bc || (v * lo == bv * bc && lo > bc)) {
            best = Some((v, lo));
        }
    }
    best
}

/// [`Pairs`] of the insulator `sampler` holds on `mesh` for `request` (its kernel decides whether
/// the local-field rows are computed; its frequencies are not read).
pub fn pairs(sampler: &Sampler<'_>, mesh: &MeshIndex, request: &Request) -> Result<Pairs> {
    columns(sampler, mesh, request, None)
}

/// [`pairs`] from the states the mesh is sampled into, or from `held` (`held[s][k]`, every band
/// of channel `s` at mesh point `k`): the Bethe-Salpeter equation reads its exchange columns and
/// dipoles from the frames its direct term uses, so both see one gauge of every state.
pub(super) fn columns(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    request: &Request,
    held: Option<&[Vec<crate::periodic::wannier::sampling::Frame>]>,
) -> Result<Pairs> {
    check_crystal(sampler)?;
    check_inversion_closed(mesh)?;
    if mesh.origin.iter().any(|x| *x != 0.) {
        return Err(invalid(
            "the dielectric function needs a Gamma-anchored mesh",
        ));
    }
    if request.frequencies.is_empty() || request.frequencies.iter().any(|w| !w.is_finite()) {
        return Err(invalid("the dielectric function needs finite frequencies"));
    }
    if !(request.damping.is_finite() && request.damping >= 0.) {
        return Err(invalid("the damping must be finite and at least zero"));
    }
    if !request.scissor.is_finite() {
        return Err(invalid("the scissor must be finite"));
    }
    let rpa = request.kernel != Kernel::Independent;
    if rpa && !(request.cutoff.is_finite() && request.cutoff > 0.) {
        return Err(invalid(
            "the local-field cutoff must be finite and positive",
        ));
    }
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined)?;
    let bands = crystal.bands();
    let mut window = Vec::with_capacity(sampler.channels());
    let mut selections = Vec::with_capacity(sampler.channels());
    for &(lo, hi) in &manifold.ranges {
        if lo != 0 || hi == 0 || hi == bands {
            return Err(invalid(
                "the dielectric function needs occupied and empty bands in every spin channel",
            ));
        }
        let (valence, conduction) = request.window.unwrap_or((hi, bands - hi));
        if valence == 0 || valence > hi || conduction == 0 || conduction > bands - hi {
            return Err(invalid(format!(
                "bands=(valence, conduction) must hold 1 to {hi} valence and 1 to {} conduction \
                 bands; got ({valence}, {conduction})",
                bands - hi
            )));
        }
        window.push((valence, conduction));
        selections.push((hi - valence..hi + conduction).collect::<Vec<usize>>());
    }
    // The local-field vectors: every `0 < |G| <= cutoff` (the set is closed under `G -> -G`).
    let body: Vec<[i64; 3]> = if rpa {
        within(&sampler.lattice, [0.; 3], request.cutoff)
            .into_iter()
            .filter(|g| *g != [0, 0, 0])
            .collect()
    } else {
        vec![]
    };
    // The columns are reserved first, before any grid-sized work: a window above the budget is
    // refused at once, by name, with the largest that fits.
    let rows = 3 + body.len();
    let per_point: usize = window.iter().map(|(v, c)| v * c).sum();
    let total = product(&[per_point, mesh.points()])?;
    let stored = ctx.reserve(product(&[total, 2, rows, 16])?).map_err(|_| {
        let valence = manifold.ranges.iter().map(|r| r.1).min().unwrap_or(1);
        let conduction = manifold
            .ranges
            .iter()
            .map(|r| bands - r.1)
            .min()
            .unwrap_or(1);
        let largest = largest_window(valence, conduction, |v, c| {
            product(&[v, c, window.len(), mesh.points(), 2, rows, 16])
                .ok()
                .and_then(|bytes| ctx.reserve(bytes).ok())
                .is_some()
        });
        Error::Unsupported(format!(
            "the dielectric function of {total} transitions (bands={:?} x {} k-points, {rows} \
             head and local-field rows) does not fit the memory budget: {}; a smaller window \
             drops its higher transitions from the static limit",
            window[0],
            mesh.points(),
            match largest {
                Some((v, c)) => format!("the largest window that fits is bands=({v}, {c})"),
                None => "not even bands=(1, 1) fits".into(),
            }
        ))
    })?;
    let paw = if rpa {
        let setup = Setup::new(crystal)?;
        let remainders = Remainders::new(ctx, &setup)?;
        Some((setup, remainders))
    } else {
        None
    };
    let engine = if rpa {
        Some(Engine::new(ctx, &crystal.grid, [0.; 3], &body)?)
    } else {
        None
    };
    let augmentation = match (&paw, &engine) {
        (Some((setup, remainders)), Some(engine)) => Some(Augmentation::new(
            crystal,
            setup,
            remainders,
            &engine.vectors,
        )?),
        _ => None,
    };
    // `-G` of each local-field vector, for `rho_cv(G) = conj(rho_vc(-G))` of the smooth part.
    let negative: Vec<usize> = body
        .iter()
        .map(|g| {
            body.iter()
                .position(|h| (0..3).all(|a| h[a] == -g[a]))
                .ok_or_else(|| invalid("the local-field vectors are not closed under G -> -G"))
        })
        .collect::<Result<_>>()?;
    let one_center = one_center_momentum(ctx, sampler.setup)?;
    let grid_mesh = Mesh::new(ctx, &sampler.setup.grid)?;
    let points = crystal.grid.index.points();
    let volume = sampler.lattice.determinant().abs();
    let degeneracy = sampler.degeneracy();
    let mut values: Vec<C64> = Vec::with_capacity(total * 2 * rows);
    let mut transition_energies: Vec<f64> = Vec::with_capacity(total);
    let mut residual = 0f64;
    let mut onset = f64::INFINITY;
    let mut sum_rule = [[0f64; 3]; 3];
    let weight = 1. / mesh.points() as f64;
    let mut point = vec![0usize; sampler.channels()];
    let mut scratch = engine.as_ref().map(|e| e.scratch(ctx)).transpose()?;
    let mut add = |s: usize,
                   frame: &crate::periodic::wannier::sampling::Frame,
                   offset: usize|
     -> Result<()> {
        let fractional = mesh.point(point[s]);
        point[s] += 1;
        let k: [f64; 3] = (sampler.reciprocal * nalgebra::Vector3::from(fractional)).into();
        let momenta = band_momenta(ctx, sampler.setup, &grid_mesh, frame, k, &one_center)?;
        residual = residual.max(momenta.residual);
        let (valence, conduction) = window[s];
        let energies = frame.energies.as_slice();
        // A spinor frame's samples and projection rows are `(component, band)`: its bands are
        // read whole and each pair density sums the two components.
        let components = frame.components;
        let held_bands = energies.len();
        let (first, span) = if components == 1 {
            (offset, valence + conduction)
        } else {
            (0, components * held_bands)
        };
        let row = |c: usize, a: usize| {
            if components == 1 {
                a
            } else {
                c * held_bands + offset + a
            }
        };
        let samples = if rpa {
            Some(
                frame
                    .samples
                    .read_range(ctx, first * points, product(&[span, points])?)?,
            )
        } else {
            None
        };
        let projections: Vec<Vec<Vec<C64>>> = if rpa {
            (0..valence + conduction)
                .map(|a| {
                    (0..components)
                        .map(|c| {
                            (0..crystal.p)
                                .map(|i| frame.projections[(c * held_bands + offset + a, i)])
                                .collect()
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![]
        };
        let mut forward = vec![C64::default(); body.len()];
        let at = |a: usize| offset + a;
        for v in 0..valence {
            for c in valence..valence + conduction {
                let gap = energies[at(c)] - energies[at(v)];
                if gap <= 0. {
                    return Err(invalid("an empty band lies below an occupied one"));
                }
                let shifted = gap + request.scissor;
                if shifted <= 0. {
                    return Err(invalid(
                        "the scissor closes the gap: a transition energy is not positive",
                    ));
                }
                onset = onset.min(shifted);
                let (fv, fc) = (at(v), at(c));
                let head: [C64; 3] = std::array::from_fn(|a| momenta.matrices[a][(fv, fc)] / gap);
                for (a, row) in sum_rule.iter_mut().enumerate() {
                    for (b, to) in row.iter_mut().enumerate() {
                        *to += degeneracy
                            * weight
                            * 2.
                            * (momenta.matrices[a][(fv, fc)] * momenta.matrices[b][(fc, fv)]).re
                            * shifted
                            / (gap * gap);
                    }
                }
                values.extend_from_slice(&head);
                if let (Some(engine), Some(scratch), Some(samples), Some(augmentation)) =
                    (&engine, scratch.as_mut(), &samples, &augmentation)
                {
                    let (pv, pc) = (&projections[v], &projections[c]);
                    let grid = |a: usize| -> Vec<&[C64]> {
                        (0..components)
                            .map(|s| &samples[row(s, a) * points..(row(s, a) + 1) * points])
                            .collect()
                    };
                    engine.smooth_components(scratch, &grid(v), &grid(c), &mut forward)?;
                    // The smooth part of `rho_cv(G)` is `conj(rho_vc(-G))` exactly (the transform
                    // of a conjugate); the augmentation is added for each order on its own.
                    let mut backward: Vec<C64> =
                        negative.iter().map(|&j| forward[j].conj()).collect();
                    for (a, b) in pv.iter().zip(pc) {
                        augmentation.add(a, b, &mut forward);
                    }
                    for (a, b) in pv.iter().zip(pc) {
                        augmentation.add(b, a, &mut backward);
                    }
                    values.extend_from_slice(&forward);
                    values.extend(head.iter().map(|h| -h.conj()));
                    values.extend_from_slice(&backward);
                } else {
                    values.extend(head.iter().map(|h| -h.conj()));
                }
                transition_energies.push(shifted);
            }
        }
        Ok(())
    };
    match held {
        None => sampler.visit(mesh, &selections, |s, frame| add(s, &frame, 0))?,
        Some(frames) => {
            if frames.len() != sampler.channels() || frames.iter().any(|f| f.len() != mesh.points())
            {
                return Err(invalid("the held frames do not cover the mesh"));
            }
            for k in 0..mesh.points() {
                for (s, channel) in frames.iter().enumerate() {
                    let frame = &channel[k];
                    let first = selections[s][0];
                    if frame.energies.as_slice().len() < selections[s].last().map_or(0, |b| b + 1) {
                        return Err(invalid("a held frame lacks the window's bands"));
                    }
                    add(s, frame, first)?;
                }
            }
        }
    }
    if transition_energies.len() != total {
        return Err(invalid("the mesh made the wrong number of transitions"));
    }
    let vectors = engine.map(|e| e.vectors).unwrap_or_default();
    let coulomb = vectors
        .iter()
        .map(|v| (4. * PI / v.iter().map(|x| x * x).sum::<f64>()).sqrt())
        .collect();
    Ok(Pairs {
        rows,
        values,
        energies: transition_energies,
        vectors,
        coulomb,
        scale: degeneracy * weight / volume,
        mesh: mesh.size,
        window,
        onset,
        sum_rule,
        electrons: crystal.electrons,
        volume,
        residual,
        _stored: stored,
    })
}

impl Pairs {
    /// Column `2 t + 1` (`rho_cv`) of transition `t` with `backward`, else column `2 t`
    /// (`rho_vc`): the three head rows, then the local-field vectors.
    pub fn column(&self, t: usize, backward: bool) -> &[C64] {
        let first = (2 * t + usize::from(backward)) * self.rows;
        &self.values[first..first + self.rows]
    }
    /// `sqrt(4 pi) / |G|` of each local-field vector.
    pub fn coulomb(&self) -> &[f64] {
        &self.coulomb
    }
    /// `g w_k / Omega`, the weight of one transition in `chi0`.
    pub fn scale(&self) -> f64 {
        self.scale
    }
    /// `chi0(z)` (rows and columns: the three head rows, then the local-field vectors; Bohr^-3
    /// Hartree^-1, the head rows divided by `q`): `sum_t rho_t w_t(z) rho_t^dagger`, in chunks of
    /// transitions in a fixed order.
    pub fn polarizability(&self, z: C64) -> Result<DMatrix<C64>> {
        polarizability(self, z, self.scale)
    }
    /// `eps^{ab}(z)` (module documentation): the head alone without local-field rows.
    pub fn tensor(&self, z: C64) -> Result<[[C64; 3]; 3]> {
        macroscopic(&self.polarizability(z)?, &self.coulomb, 0.)
    }
}

/// `chi0(z)` on the stored rows: `sum_t rho_t w_t(z) rho_t^dagger`, in chunks of transitions in
/// a fixed order.
fn polarizability(columns: &Pairs, z: C64, scale: f64) -> Result<DMatrix<C64>> {
    let n = columns.rows;
    let count = columns.energies.len();
    let mut chi = DMatrix::<C64>::zeros(n, n);
    for first in (0..count).step_by(CHUNK) {
        let last = (first + CHUNK).min(count);
        let width = 2 * (last - first);
        let left = DMatrixView::from_slice(&columns.values[first * 2 * n..last * 2 * n], n, width);
        let right = DMatrix::from_fn(width, n, |i, g| {
            let t = first + i / 2;
            let energy = columns.energies[t];
            let w = if i % 2 == 0 {
                C64::new(scale, 0.) / (z - energy)
            } else {
                C64::new(-scale, 0.) / (z + energy)
            };
            w * left[(g, i)].conj()
        });
        chi += complex_product(left, false, right.as_view())?;
    }
    if chi.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(invalid(
            "the polarizability is not finite: a frequency sits on an undamped pole",
        ));
    }
    Ok(chi)
}

/// `eps^{ab} = delta_ab - 4 pi chibar^{ab}_00` from `chi0` (head rows first, then the
/// local-field vectors with `sqrt(4 pi) / |G|` in `coulomb`).
///
/// With the long-range kernel (`alpha > 0`) the body's interaction is `v + f_xc =
/// (4 pi - alpha) / |G|^2` and the head's is `f_xc` alone (`v` has no macroscopic term), `-alpha`
/// on the head rows (divided by `q`): `chi~ = (1 + alpha chibar)^-1 chibar`, isotropic in the head
/// rows, so exact along the principal axes of a cubic crystal.
fn macroscopic(chi: &DMatrix<C64>, coulomb: &[f64], alpha: f64) -> Result<[[C64; 3]; 3]> {
    let nb = coulomb.len();
    let shrink = (1. - alpha / (4. * PI)).sqrt();
    let coulomb: Vec<f64> = coulomb.iter().map(|c| c * shrink).collect();
    let mut bar = chi.view((0, 0), (3, 3)).into_owned();
    if nb > 0 {
        // (1 - v^1/2 chi_body v^1/2) X = v^1/2 chi_{G, b}; chibar = chi_00 + chi_{a, G} v^1/2 X.
        let system = DMatrix::from_fn(nb, nb, |g, h| {
            C64::new(f64::from(g == h), 0.) - coulomb[g] * chi[(3 + g, 3 + h)] * coulomb[h]
        });
        let wings = DMatrix::from_fn(nb, 3, |g, b| coulomb[g] * chi[(3 + g, b)]);
        let solved = system
            .lu()
            .solve(&wings)
            .ok_or_else(|| invalid("the local-field dielectric matrix is singular"))?;
        for a in 0..3 {
            for b in 0..3 {
                bar[(a, b)] += (0..nb)
                    .map(|g| chi[(a, 3 + g)] * coulomb[g] * solved[(g, b)])
                    .sum::<C64>();
            }
        }
    }
    if alpha != 0. {
        let system = nalgebra::Matrix3::<C64>::identity()
            + nalgebra::Matrix3::from_fn(|a, b| bar[(a, b)] * alpha);
        let solved = system
            .try_inverse()
            .ok_or_else(|| invalid("the long-range kernel's head is singular"))?
            * nalgebra::Matrix3::from_fn(|a, b| bar[(a, b)]);
        bar = DMatrix::from_fn(3, 3, |a, b| solved[(a, b)]);
    }
    let out: [[C64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| C64::new(f64::from(a == b), 0.) - bar[(a, b)] * (4. * PI))
    });
    if out
        .iter()
        .flatten()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("the dielectric tensor is not finite"));
    }
    Ok(out)
}
