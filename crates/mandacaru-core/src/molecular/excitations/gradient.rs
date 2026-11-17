//! The energy and analytic nuclear gradient of one excited state of a Kohn-Sham or Hartree-Fock
//! PAW-LCAO molecule, restricted or unrestricted (`dev_plan/TDDFT_PLAN.md`, T7):
//! `E = E_0 + omega`, with `omega` a Tamm-Dancoff or Casida root of [`Response`] or
//! [`Unrestricted`] (adiabatic LDA, PBE, r2SCAN or HSE06, with or without +U; CIS or TDHF).
//!
//! The gradient is the explicit derivative of the Lagrangian (Furche and Ahlrichs, J. Chem. Phys.
//! 117, 7433 (2002), in the spin-orbital form of a closed shell)
//!
//! ```text
//! L = E[D] + omega[C, X] + sum_ia Z_ia F_ia[C] - sum_pq W_pq (C^T S C - 1)_pq,
//! omega = sum Q_vv F_vv - sum Q_oo F_oo + E''[T, T],
//! ```
//!
//! with `U = X + Y`, `V = X - Y` (`U^T V = 1`; Tamm-Dancoff `U = V = X`), `Q_vv = (U^T U + V^T V) / 2`,
//! `Q_oo = (U U^T + V V^T) / 2` and `T` the transition density of `U`: for a density-only kernel
//! `A - B` is the orbital-energy difference, so `omega = [U (A + B) U + V (A - B) V] / 2` is
//! stationary in `U` and `V` at a Casida root, as `X A X` is at a Tamm-Dancoff one.
//!
//! at the converged state, the root's amplitudes `X`, the Z-vector `(A + B) Z = -R` and the
//! energy-weighted density `W` that make it stationary in the orbitals. Its explicit part is the
//! fixed-density gradient `G(D)` of the energy (every term the ground-state force contracts, at
//! fixed raw-AO densities) along two directions: `d/ds G(D + s P)` with the relaxed difference
//! density `P = Delta + Z` and `d^2/dt^2 G(D + t T)` with the transition density `T`. Both are
//! taken exactly: the terms of `G` polynomial in `D` (one-body, Hartree, compensation pairs) are
//! at most quadratic, so `[G(D + P) - G(D - P)] / 2` and `G(D + T) + G(D - T) - 2 G(D)` are their
//! derivatives without truncation; the exchange-correlation terms are contracted with the exact
//! kernel `dv[P]` and the third-order change `d^2 v / dt^2` ([`xc_kernel::second_change`]) passed
//! as the potentials of the same evaluations, which makes the combination exact for them too.
//!
//! A kernel that sees the antisymmetric part of the transition density (exact exchange) has an
//! `A - B` that is not the orbital-energy difference. Its share of `omega` is the second
//! derivative of the energy along the imaginary Hermitian direction `D + t i T'`, with `T'`
//! antisymmetric (`V / 2` occupied-virtual, `-V^T / 2` virtual-occupied). So
//! `omega = sum Q F + d^2 E[T] + d^2 E[i T']`. `Im F1[i T']` adds `2 Im F1_oo V - 2 V Im F1_vv`
//! to `R`, and the matching blocks to `W`. Hartree-Fock (CIS and TDHF) takes its explicit part in
//! one contraction of an effective two-body density in Coulomb forms
//! (`Ground::hartree_fock_explicit`).
//!
//! DFT+U (Dudarev and Liechtenstein) on a Kohn-Sham reference is quadratic in the shell
//! occupations, so its first-order Fock of a general change and its imaginary Fock are exact by
//! the shell map (`MolecularHubbard::shell_linear`), its third-order change is zero, the
//! symmetric contractions carry its explicit part along `P` and `T`, and its explicit part along
//! `i T'` is `paw::molecular::gradient::hubbard_imaginary_force`.
//!
//! HSE06 adds its short-range exact exchange, whose imaginary Fock is `-a K_sr[T']` on the
//! screened pair factors the response uses, and the spheres' frozen short-range semilocal exchange
//! with its spin polarization (`paw::molecular::SpinTerms`), whose first and second changes enter
//! the first-order Fock and the third-order term. The symmetric contractions carry its semilocal
//! grid part; its exchange along all three directions is one coupled exchange of the directions'
//! pseudo-orbitals (`paw::molecular::Setup::coupled_exchange`) and its spheres are
//! `paw::molecular::Setup::hybrid_sphere_derivative`.
//!
//! r2SCAN sees the orbitals' density gradient and kinetic density (`super::meta`), so its first-
//! and second-order changes carry three potential parts (local, kinetic and flux) and the
//! symmetric contractions take all three. Its kernel sees the antisymmetric transition density
//! through the gauge-invariant `tau~ = tau - |j|^2 / 2n`: the imaginary Fock is the current's
//! (exact), and `d^2 E[i T'] = -sum_s int (v_tau / n) |j_s|^2` is not quadratic, since its weight
//! depends on the ground density. So the third-order change runs along `D + t T + t i T'` at
//! once, the kinetic densities curving by `-|j|^2 / 2n` ([`xc_kernel::second_change_meta`]), and
//! the explicit derivative of the current at fixed weight is [`MolecularIntegrals::action_gradient`]
//! of the actions of `super::meta::current_actions`.
//!
//! Along a scan the root is picked by its index ([`Track::Index`], adiabatic) or by the largest
//! overlap of its transition density with the state of the previous geometry ([`Track::Overlap`],
//! [`followed_excited_state`]), which keeps the state's character where two roots cross.
//!
//! Atomic units: Hartree and Hartree per Bohr.
use super::{
    excitations, invalid, meta, response::mo_samples, Coupling, Request, Response, Spin,
    Unrestricted,
};
use crate::math::{fft::Poisson, linalg::real_eigen};
use crate::models::{
    xc,
    xc_kernel::{self, Kernel},
};
use crate::molecular::{dft, forces::Gradient, paw_forces::Frozen, MeanField, MolecularIntegrals};
use crate::paw::molecular::gradient::Occupied;
use crate::{quantum::C64, Buffer, Result};
use nalgebra::DMatrix;

/// Two roots closer than this (Hartree) form a multiplet without a single gradient.
const DEGENERATE: f64 = 1e-6;

/// How the root is picked at each new geometry of a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    /// The requested index among the roots of the spin symmetry (adiabatic): where two roots
    /// cross, the state followed changes character.
    Index,
    /// The root whose transition density overlaps most with the state followed at the previous
    /// geometry ([`followed_excited_state`]): the state keeps its character through a crossing
    /// and its index may change.
    Overlap,
}
impl Track {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "index" => Ok(Self::Index),
            "overlap" => Ok(Self::Overlap),
            _ => Err(invalid(format!(
                "unknown track {name:?}; expected 'index' or 'overlap'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Overlap => "overlap",
        }
    }
}

/// The state a scan follows into its next geometry: its root and its symmetrized transition
/// densities (`X + Y`, one per spin channel of the reference) in the nonorthogonal atomic basis,
/// whose functions move with their atoms, so the matrices of two neighboring geometries are
/// compared element by element.
#[derive(Clone, Debug)]
pub struct Followed {
    pub root: usize,
    pub transition: Vec<DMatrix<f64>>,
}

/// `|<A, B>| / (|A| |B|)` with `<A, B> = sum_s tr(A_s S B_s S)` for symmetric `A_s`, `B_s` (the
/// spin channels) and the overlap `S`: the overlap of two transition densities in the metric of
/// the current geometry's basis.
fn transition_overlap(s: &DMatrix<f64>, a: &[DMatrix<f64>], b: &[DMatrix<f64>]) -> f64 {
    let inner = |x: &[DMatrix<f64>], y: &[DMatrix<f64>]| {
        x.iter()
            .zip(y)
            .map(|(x, y)| (x * s).component_mul(&(s * y)).sum())
            .sum::<f64>()
    };
    let norm = (inner(a, a) * inner(b, b)).sqrt();
    if norm > 0. {
        inner(a, b).abs() / norm
    } else {
        0.
    }
}

/// One excited state: its energy, excitation energy and gradient.
pub struct ExcitedState {
    /// The root (0-based, ascending) and its spin symmetry.
    pub root: usize,
    pub spin: Spin,
    /// `omega`, Hartree.
    pub excitation: f64,
    /// `E_0 + omega` with the ion-ion repulsion, Hartree (the excited state's total energy).
    pub energy: f64,
    /// `dE / dR`, Hartree per Bohr, three per atom.
    pub gradient: Gradient,
    /// Its parts (three per atom each): the ground-state gradient, the relaxed difference
    /// density's, the transition density's and the excitation's overlap term.
    pub parts: Vec<(&'static str, Vec<f64>)>,
    /// The relaxed difference density `Delta + Z` of both spins in the orthonormal frame: the
    /// change of the one-particle density from the reference to the excited state.
    pub relaxed: DMatrix<f64>,
    /// The transition density `T` of each spin and the excitation's energy-weighted density
    /// `W` (both spins), in the orthonormal frame: the directions of the explicit derivative.
    pub transition: [DMatrix<f64>; 2],
    pub energy_weighted: DMatrix<f64>,
    /// The distance to the nearest other root of the solve, Hartree.
    pub separation: f64,
    /// The Z-vector's final residual norm (relative to its right-hand side) and iterations.
    pub z_residual: f64,
    pub z_iterations: usize,
    /// The root's residual norm from the excitation solver.
    pub root_residual: f64,
    /// The state as the next geometry of a scan follows it.
    pub followed: Followed,
    /// The overlap with the previous geometry's state that picked this root
    /// ([`followed_excited_state`]), between 0 and 1; `None` for a root picked by index.
    pub overlap: Option<f64>,
}

/// One spin's canonical orbitals of the reference: a closed shell has one for both spins.
struct Channel {
    o: usize,
    energies: Vec<f64>,
    /// Canonical orbitals in the orthonormal frame and in the raw atomic basis (columns).
    frame: DMatrix<f64>,
    raw: DMatrix<f64>,
    /// The canonical orbitals on the grid, `[orbital][point]` (Kohn-Sham only).
    samples: Option<Buffer<f64>>,
    /// Their grid gradients, `[axis][orbital][point]` (a meta-GGA only).
    gradients: Option<Buffer<f64>>,
    /// The PAW compensation pair factors `C^T B C` (their weights are `Ground::weights`).
    pairs: Vec<DMatrix<f64>>,
    /// Hartree-Fock: the Coulomb pair factors `C^T L^P C` of the response's integrals (their
    /// weights are `Ground::factor_weights`).
    factors: Vec<DMatrix<f64>>,
}

/// The energy model whose excited state is differentiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    /// Kohn-Sham with a semilocal kernel (`"lda"`, `"pbe"`, the semilocal part of `"hse06"`,
    /// whose screened exact exchange is [`Hybrid`], or the meta-GGA `"r2scan"`).
    Semilocal(&'static str),
    /// Hartree-Fock (CIS and TDHF): exact exchange, the one part of its kernel that sees the
    /// imaginary (antisymmetric) transition density.
    HartreeFock,
    /// A spinor Kohn-Sham reference with LDA or PBE ([`super::spinor::gradient`]).
    Spinor,
}

/// A Kohn-Sham reference's semilocal kernel and ground-state XC potentials.
struct Semilocal {
    /// The densities the XC kernel sees (valence plus half the smooth core per spin).
    kernel_density: Buffer<f64>,
    /// The density-only kernel (LDA, PBE, HSE06's semilocal part); `None` for a meta-GGA.
    kernel: Option<Kernel>,
    /// A meta-GGA's ground fields, kernel and current-density weights.
    meta: Option<meta::MetaGround>,
    name: &'static str,
    relativistic: bool,
    /// The ground-state XC potentials of both spins: `[spin][point]`, or for a meta-GGA
    /// `[local 2n | tau 2n | flux 6n]` (the packing of every potential change here).
    potentials: Buffer<f64>,
}

/// A screened hybrid's exact exchange and spheres, in the canonical orbitals of the reference.
struct Hybrid {
    omega: f64,
    fraction: f64,
    /// The screened pair factors `C^T L^P C` of each channel (their weights are `weights`): the
    /// short-range exchange with its compensation and one-center pieces, as the response has it.
    factors: Vec<Vec<DMatrix<f64>>>,
    weights: Vec<f64>,
    /// The spheres' spin terms, their polarization kernels at the reference and its spin
    /// densities (orthonormal frame).
    spheres: Option<(
        crate::paw::molecular::SpinTerms,
        crate::paw::molecular::SphereKernels,
        [DMatrix<f64>; 2],
    )>,
}

/// The ground-state pieces the Lagrangian reuses, in the canonical orbitals of the reference.
struct Ground<'a> {
    integrals: &'a MolecularIntegrals,
    m: usize,
    /// One channel (closed shell) or alpha and beta (unrestricted).
    channels: Vec<Channel>,
    /// Kohn-Sham only.
    semilocal: Option<Semilocal>,
    /// HSE06 only.
    hybrid: Option<Hybrid>,
    /// The weights of the PAW compensation pair factors (Kohn-Sham).
    weights: Vec<f64>,
    /// The weights of the Coulomb pair factors (Hartree-Fock).
    factor_weights: Vec<f64>,
}

/// The XC potentials of `terms` packed as [`Semilocal::potentials`]: `[spin][point]`, or with
/// `meta` the local, kinetic and flux parts in turn.
fn packed(ctx: &crate::Context, terms: &xc::Xc, meta: bool) -> Result<Buffer<f64>> {
    if !meta {
        return ctx.copy_slice(terms.potentials.as_slice());
    }
    let (Some(tau), Some(flux)) = (&terms.tau_potentials, &terms.gradient_potentials) else {
        return Err(invalid(
            "a meta-GGA evaluation returned no kinetic or flux potential",
        ));
    };
    let parts = [terms.potentials.as_slice(), tau.as_slice(), flux.as_slice()];
    ctx.copy_slice(&parts.concat())
}

/// Why `reference` has no excited-state gradient here, if it has none.
fn check(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
) -> Result<Model> {
    let unsupported = |what: &str| Err(crate::Error::Unsupported(what.into()));
    if integrals.paw_setup.is_none() {
        return unsupported(
            "excited-state forces need a PAW-LCAO basis (the analytic force contraction of the ground state)",
        );
    }
    if request.coupling != Coupling::Adiabatic {
        return Err(invalid("excited-state forces follow the adiabatic kernel of the functional (kernel='adiabatic')"));
    }
    let Some(name) = reference.functional.as_deref() else {
        if integrals
            .corrections
            .as_ref()
            .is_some_and(|c| c.hubbard.is_some())
        {
            return unsupported("Hartree-Fock excited-state forces with hubbard_u are not implemented (the ground-state Hartree-Fock gradient has no +U term)");
        }
        if reference.method == "ghf" {
            return Err(invalid(
                "two-component Hartree-Fock (method='ghf') has no response yet",
            ));
        }
        return Ok(Model::HartreeFock);
    };
    let definition = crate::models::functional(name)?;
    if reference.method == "ghf" {
        return match definition.semilocal {
            "lda" | "pbe"
                if definition.nonlocal_b.is_none() && definition.screened_exchange.is_none() =>
            {
                Ok(Model::Spinor)
            }
            _ => unsupported(
                "excited-state forces of a spinor reference are implemented for xc='lda' and xc='pbe'",
            ),
        };
    }
    match definition.semilocal {
        "lda" | "pbe" | "hse06" | "r2scan" if definition.nonlocal_b.is_none() => {
            Ok(Model::Semilocal(definition.semilocal))
        }
        _ => unsupported(
            "excited-state forces are implemented for Hartree-Fock, xc='lda', xc='pbe', xc='hse06' and xc='r2scan'",
        ),
    }
}

/// The ground-state force contraction at given spin densities: `valence` (`[spin][point]`, as
/// they are, unfloored) and the symmetric matrices `(x, raw, spin, energy_weighted)` (in the
/// basis whose functions are the columns of `raw` over the raw atomic basis; a density, or with
/// `energy_weighted` an energy-weighted density of the overlap term), with the XC potentials `v`
/// (`[spin][point]`, a meta-GGA's kinetic and flux parts after them) in the exchange-correlation
/// terms. Both spins are always given: the flag gives +U its two occupations (it reaches nothing
/// else without a hybrid's exchange).
pub(super) fn frozen_contraction(
    integrals: &MolecularIntegrals,
    setup: &crate::paw::molecular::Setup,
    valence: Buffer<f64>,
    matrices: &[(&DMatrix<f64>, &DMatrix<f64>, usize, bool)],
    v: &[f64],
) -> Result<Gradient> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let mut poisson = Poisson::isolated(ctx, &integrals.grid)?;
    let mut source = ctx.allocate::<C64>(n)?;
    let mut solved = ctx.allocate::<C64>(n)?;
    for k in 0..n {
        source.as_mut_slice()[k] = C64::new(valence.as_slice()[k] + valence.as_slice()[n + k], 0.);
    }
    poisson.solve_into(source.as_slice(), solved.as_mut_slice())?;
    let mut hartree = ctx.allocate::<f64>(n)?;
    for (to, from) in hartree.as_mut_slice().iter_mut().zip(solved.as_slice()) {
        *to = from.re;
    }
    let mut orbitals = Vec::new();
    for &(matrix, raw, spin, energy_role) in matrices {
        let (values, vectors) = real_eigen((matrix + matrix.transpose()) * 0.5)?;
        let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
        for (j, &value) in values.iter().enumerate() {
            if value.abs() <= 1e-14 * largest {
                continue;
            }
            orbitals.push(Occupied {
                spin,
                weight: if energy_role { 0. } else { value },
                energy: 0.,
                w_weight: if energy_role { value } else { 0. },
                ao: (raw * vectors.column(j)).iter().copied().collect(),
            });
        }
    }
    let frozen = Frozen {
        functional: None,
        unrestricted: true,
        hf: false,
        valence,
        hartree,
        xc: Some(xc::Xc {
            energy: 0.,
            potentials: ctx.copy_slice(&v[..2 * n])?,
            tau_potentials: (v.len() > 2 * n)
                .then(|| ctx.copy_slice(&v[2 * n..4 * n]))
                .transpose()?,
            gradient_potentials: (v.len() > 2 * n)
                .then(|| ctx.copy_slice(&v[4 * n..]))
                .transpose()?,
        }),
        orbitals,
        stationarity: 0.,
        exact_exchange: false,
    };
    integrals.contract_frozen(frozen, setup)
}

impl<'a> Ground<'a> {
    /// The pieces of a closed shell, in the canonical orbitals of its singlet response.
    fn closed(
        integrals: &'a MolecularIntegrals,
        reference: &MeanField,
        model: Model,
    ) -> Result<Self> {
        let singlet = Response::new(integrals, reference, Spin::Singlet, Coupling::Adiabatic)?;
        let channel = (
            singlet.occupied(),
            singlet.orbital_energies().to_vec(),
            singlet.orbitals().clone(),
        );
        Self::new(integrals, reference, vec![channel], model)
    }
    /// `channels`: the occupied count, orbital energies and orthonormal-frame orbitals of one
    /// spin (closed shell) or of alpha and beta.
    fn new(
        integrals: &'a MolecularIntegrals,
        reference: &MeanField,
        channels: Vec<(usize, Vec<f64>, DMatrix<f64>)>,
        model: Model,
    ) -> Result<Self> {
        let ctx = &integrals.context;
        let n = integrals.grid.index.points();
        let paw = integrals.corrections.as_ref().and_then(|c| c.dft.as_ref());
        let mut screened = None;
        let semilocal = match model {
            Model::Semilocal(name) => {
                let functional = reference.functional.as_deref().unwrap_or(name);
                let screening = integrals.screening(functional)?;
                // A meta-GGA's Fock matrix needs the orbital gradients, as in the SCF.
                let orbital_gradients = if crate::models::functional(functional)?.requires_tau {
                    Some(dft::orbital_gradients(integrals)?)
                } else {
                    None
                };
                let (_, fields, _, _) = dft::response_with_shared(
                    integrals,
                    &reference.densities,
                    functional,
                    orbital_gradients.as_ref(),
                    screening.as_ref(),
                )?;
                screened = screening.zip(crate::models::functional(functional)?.screened_exchange);
                let mut kernel_density = ctx.copy_slice(fields.valence.as_slice())?;
                if let Some(paw) = paw {
                    for s in 0..2 {
                        for k in 0..n {
                            kernel_density.as_mut_slice()[s * n + k] +=
                                0.5 * paw.core.as_slice()[k];
                        }
                    }
                }
                // The SCF's potential takes the relativistic exchange factor for LDA and PBE only.
                let relativistic =
                    paw.is_some_and(|p| p.relativistic_exchange) && matches!(name, "lda" | "pbe");
                let (kernel, meta) = if name == "r2scan" {
                    let ground = meta::MetaGround::new(
                        integrals,
                        functional,
                        &reference.densities,
                        fields.valence.as_slice(),
                    )?;
                    (None, Some(ground))
                } else {
                    let kernel = Kernel::new(
                        ctx,
                        &integrals.grid,
                        kernel_density.as_slice(),
                        name,
                        relativistic,
                    )?;
                    (Some(kernel), None)
                };
                Some(Semilocal {
                    potentials: packed(ctx, &fields.terms, meta.is_some())?,
                    kernel_density,
                    kernel,
                    meta,
                    name,
                    relativistic,
                })
            }
            Model::HartreeFock => None,
            Model::Spinor => {
                return Err(invalid("a spinor reference has no collinear ground pieces"))
            }
        };
        // Hartree-Fock: the pair factors of the integrals the response's Hessian uses.
        let mut factor_weights = vec![];
        let mut factors = vec![];
        if model == Model::HartreeFock {
            let eri = integrals.eri()?;
            super::response::each_factor(&eri, integrals.m(), |w, b| {
                factor_weights.push(w);
                factors.push(b.clone());
            })?;
        }
        let channels = channels
            .into_iter()
            .map(|(o, energies, frame)| {
                let ks = semilocal.is_some();
                let meta = semilocal.as_ref().is_some_and(|s| s.meta.is_some());
                let pairs = match paw {
                    Some(paw) if ks => paw
                        .pairs
                        .factors
                        .iter()
                        .map(|b| frame.transpose() * b * &frame)
                        .collect(),
                    _ => vec![],
                };
                Ok(Channel {
                    o,
                    energies,
                    raw: &integrals.lowdin * &frame,
                    samples: if ks {
                        Some(mo_samples(integrals, &frame)?)
                    } else {
                        None
                    },
                    gradients: if meta {
                        Some(meta::gradient_samples(integrals, &frame)?)
                    } else {
                        None
                    },
                    factors: factors
                        .iter()
                        .map(|b| frame.transpose() * b * &frame)
                        .collect(),
                    frame,
                    pairs,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let hybrid = match screened {
            Some((screening, (omega, _))) => {
                let mut weights = vec![];
                let mut raw = vec![];
                super::response::each_factor(&screening.eri, integrals.m(), |w, b| {
                    weights.push(w);
                    raw.push(b.clone());
                })?;
                let factors = channels
                    .iter()
                    .map(|c| {
                        raw.iter()
                            .map(|b| c.frame.transpose() * b * &c.frame)
                            .collect()
                    })
                    .collect();
                let spheres = match screening.spin {
                    Some(terms) => {
                        let ground = [0, 1].map(|s| {
                            let c = &channels[s.min(channels.len() - 1)];
                            c.frame.columns(0, c.o) * c.frame.columns(0, c.o).transpose()
                        });
                        let kernels = terms.polarization_kernel(ctx, &ground[0], &ground[1])?;
                        Some((terms, kernels, ground))
                    }
                    None => None,
                };
                Some(Hybrid {
                    omega,
                    fraction: screening.fraction,
                    factors,
                    weights,
                    spheres,
                })
            }
            None => None,
        };
        Ok(Self {
            integrals,
            m: integrals.m(),
            hybrid,
            channels,
            weights: match paw {
                Some(paw) if semilocal.is_some() => paw.pairs.weights.clone(),
                _ => vec![],
            },
            semilocal,
            factor_weights,
        })
    }
    /// The channel of spin `s`.
    fn channel(&self, s: usize) -> &Channel {
        &self.channels[s.min(self.channels.len() - 1)]
    }
    fn points(&self) -> usize {
        self.integrals.grid.index.points()
    }
    fn semilocal(&self) -> Result<&Semilocal> {
        self.semilocal
            .as_ref()
            .ok_or_else(|| invalid("a Hartree-Fock reference has no semilocal kernel"))
    }
    /// `sum_P w_P [c_P L^P_s - L^P_s V_s L^P_s]` of the spin changes `v` (canonical-orbital
    /// matrices, general): Hartree-Fock's first-order Fock `J - K` (with `coulomb`) or its
    /// exchange `-K` alone. Exact: the Fock matrix is linear in the density.
    fn exchange_fock(&self, v: [&DMatrix<f64>; 2], coulomb: bool) -> [DMatrix<f64>; 2] {
        let m = self.m;
        let mut out = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
        for (p, &w) in self.factor_weights.iter().enumerate() {
            let l = [0, 1].map(|s| &self.channel(s).factors[p]);
            let charge = if coulomb {
                l[0].dot(v[0]) + l[1].dot(v[1])
            } else {
                0.
            };
            for s in 0..2 {
                out[s] += l[s] * (w * charge) - (l[s] * v[s] * l[s]) * w;
            }
        }
        out
    }
    /// The +U first-order Fock of the spin changes `v` (canonical-orbital matrices, general):
    /// `q_s V'_s q_s^T` with `q_s = C_s^T Q` and `V'_s` the potential change of the occupation
    /// change `q_s^T v_s q_s`, real (`imaginary = false`) or the imaginary part along `i v`
    /// (`v` antisymmetric). Exact: the energy is quadratic in the occupations. `None` without
    /// +U.
    fn hubbard_fock(
        &self,
        v: [&DMatrix<f64>; 2],
        imaginary: bool,
    ) -> Result<Option<[DMatrix<f64>; 2]>> {
        let Some(hubbard) = self
            .integrals
            .corrections
            .as_ref()
            .and_then(|c| c.hubbard.as_ref())
        else {
            return Ok(None);
        };
        let m = self.m;
        let mut out = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
        for shell in &hubbard.shells {
            let q = [0, 1].map(|s| self.channel(s).frame.transpose() * &shell.q);
            let changes: Vec<DMatrix<C64>> = (0..2)
                .map(|s| {
                    (q[s].transpose() * v[s] * &q[s]).map(|x| {
                        if imaginary {
                            C64::new(0., x)
                        } else {
                            C64::new(x, 0.)
                        }
                    })
                })
                .collect();
            let potentials = hubbard.shell_linear(&self.integrals.context, shell, &changes)?;
            for s in 0..2 {
                let part = potentials[s].map(|z| if imaginary { z.im } else { z.re });
                out[s] += &q[s] * part * q[s].transpose();
            }
        }
        Ok(Some(out))
    }
    /// `Im F1_s[i V]` of the antisymmetric spin changes `v` (canonical orbitals): the
    /// first-order Fock of the imaginary Hermitian direction `D + i V`, the antisymmetric
    /// (`A - B`) channel of the kernel: Hartree-Fock's exchange, or the +U map, a hybrid's
    /// screened exchange and a meta-GGA's current-density term of a Kohn-Sham reference. `None`
    /// for a density-only kernel, whose density, Hartree and XC potentials do not see an
    /// imaginary change.
    fn imaginary_fock(&self, v: [&DMatrix<f64>; 2]) -> Result<Option<[DMatrix<f64>; 2]>> {
        if self.semilocal.is_none() {
            return Ok(Some(self.exchange_fock(v, false)));
        }
        let parts = [
            self.hubbard_fock(v, true)?,
            self.screened_fock(v)?,
            self.current_fock(v)?,
        ];
        Ok(parts
            .into_iter()
            .flatten()
            .reduce(|a, b| [&a[0] + &b[0], &a[1] + &b[1]]))
    }
    /// A meta-GGA's paramagnetic current `j_s` (`[axis][point]`) of the antisymmetric spin change
    /// `v` (canonical orbitals) of spin `s`: `D + i v` carries it.
    fn current(&self, s: usize, v: &DMatrix<f64>) -> Result<Buffer<f64>> {
        let (n, m) = (self.points(), self.m);
        let mut out = self.integrals.context.allocate::<f64>(3 * n)?;
        meta::current(
            self.samples(s)?,
            self.gradients(s)?,
            n,
            m,
            v,
            out.as_mut_slice(),
        )?;
        Ok(out)
    }
    /// `Im F1_s[i V]` of a meta-GGA's gauge-invariant kinetic density `tau~ = tau - |j|^2 / 2n`
    /// (`molecular::dft`'s complex path): `sum_r (v_tau / 2 n) j[V] . (phi_p grad phi_q - phi_q
    /// grad phi_p) dv`, exact (the current is linear in `Im D`, its weight the ground state's).
    /// `None` without a meta-GGA.
    fn current_fock(&self, v: [&DMatrix<f64>; 2]) -> Result<Option<[DMatrix<f64>; 2]>> {
        let Some(ground) = self.semilocal.as_ref().and_then(|s| s.meta.as_ref()) else {
            return Ok(None);
        };
        let (n, m, dv) = (self.points(), self.m, self.integrals.grid.index.dv());
        let mut out = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
        for (s, f) in out.iter_mut().enumerate() {
            let mut field = self.current(s, v[s])?;
            let w = &ground.weights.as_slice()[s * n..(s + 1) * n];
            for (k, value) in field.as_mut_slice().iter_mut().enumerate() {
                *value *= 0.5 * w[k % n];
            }
            *f = meta::current_fock(
                self.samples(s)?,
                self.gradients(s)?,
                n,
                m,
                field.as_slice(),
                dv,
            )?;
        }
        Ok(Some(out))
    }
    /// The explicit derivative of a meta-GGA's `d^2 E / dt^2 = -sum_s int (v_tau / n) |j_s|^2`
    /// along `D + t i V` (`v` antisymmetric, canonical orbitals) through its current at fixed
    /// `v_tau / n`: `int u . dj`, `u = -2 (v_tau / n) j`. The weight's own derivative is the
    /// curvature `tau2` of [`Self::second`], contracted with the symmetric directions. `None`
    /// without a meta-GGA.
    fn current_force(&self, v: [&DMatrix<f64>; 2]) -> Result<Option<Vec<f64>>> {
        let Some(ground) = self.semilocal.as_ref().and_then(|s| s.meta.as_ref()) else {
            return Ok(None);
        };
        let (n, m) = (self.points(), self.m);
        let mut total = vec![0.; 3 * self.integrals.geometry.numbers().len()];
        for (s, v) in v.iter().enumerate() {
            let mut u = self.current(s, v)?;
            let w = &ground.weights.as_slice()[s * n..(s + 1) * n];
            for (k, value) in u.as_mut_slice().iter_mut().enumerate() {
                *value *= -2. * w[k % n];
            }
            let actions = meta::current_actions(
                self.integrals,
                self.samples(s)?,
                self.gradients(s)?,
                m,
                v,
                u.as_slice(),
            )?;
            let raw = &self.channel(s).raw;
            let orbitals: Vec<Vec<f64>> = (0..m)
                .map(|p| raw.column(p).iter().copied().collect())
                .collect();
            let force = self
                .integrals
                .action_gradient(&orbitals, actions.as_slice())?;
            for (t, f) in total.iter_mut().zip(force) {
                *t += f;
            }
        }
        Ok(Some(total))
    }
    /// A screened hybrid's exchange Fock `-a sum_P w_P L^P_s V_s L^P_s` of the spin changes `v`
    /// (canonical orbitals, general): its first-order Fock of a real change and the imaginary
    /// part of its Fock along `i v` (`v` antisymmetric) alike, the exchange being linear in the
    /// density. `None` without a hybrid.
    fn screened_fock(&self, v: [&DMatrix<f64>; 2]) -> Result<Option<[DMatrix<f64>; 2]>> {
        let Some(hybrid) = &self.hybrid else {
            return Ok(None);
        };
        let m = self.m;
        let mut out = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
        for (s, f) in out.iter_mut().enumerate() {
            let factors = &hybrid.factors[s.min(hybrid.factors.len() - 1)];
            for (l, &w) in factors.iter().zip(&hybrid.weights) {
                *f -= (l * v[s] * l) * (w * hybrid.fraction);
            }
        }
        Ok(Some(out))
    }
    /// The second derivative of a screened hybrid's spheres' Fock matrices along the symmetric
    /// spin changes `t` (canonical orbitals): `-a` times twice the polarization's second change
    /// (its exchange is quadratic, its frozen one-body term linear). `None` without spheres.
    fn hybrid_second(&self, t: [&DMatrix<f64>; 2]) -> Result<Option<[DMatrix<f64>; 2]>> {
        let Some((hybrid, (terms, _, ground))) = self
            .hybrid
            .as_ref()
            .and_then(|h| h.spheres.as_ref().map(|s| (h, s)))
        else {
            return Ok(None);
        };
        let frame = [0, 1].map(|s| {
            let c = &self.channel(s).frame;
            c * t[s] * c.transpose()
        });
        let second =
            terms.polarization_second_change(&ground[0], &ground[1], &frame[0], &frame[1])?;
        Ok(Some([0, 1].map(|s| {
            let c = &self.channel(s).frame;
            c.transpose() * &second[s] * c * (-2. * hybrid.fraction)
        })))
    }
    /// `n(r) = sum_pq V_pq phi_p(r) phi_q(r)` of a symmetric orbital matrix of spin `s`.
    fn density(&self, s: usize, v: &DMatrix<f64>, out: &mut [f64]) -> Result<()> {
        let (n, m) = (self.points(), self.m);
        let phi = self.samples(s)?;
        for (k, value) in out.iter_mut().enumerate().take(n) {
            let mut sum = 0.;
            for p in 0..m {
                let a = phi[p * n + k];
                if a == 0. {
                    continue;
                }
                let mut row = 0.;
                for q in 0..m {
                    row += v[(p, q)] * phi[q * n + k];
                }
                sum += a * row;
            }
            *value = sum;
        }
        Ok(())
    }
    /// The canonical orbitals of spin `s` on the grid (Kohn-Sham).
    fn samples(&self, s: usize) -> Result<&[f64]> {
        Ok(self
            .channel(s)
            .samples
            .as_ref()
            .ok_or_else(|| invalid("a Hartree-Fock reference has no grid orbitals"))?
            .as_slice())
    }
    /// The grid gradients of the canonical orbitals of spin `s` (a meta-GGA).
    fn gradients(&self, s: usize) -> Result<&[f64]> {
        Ok(self
            .channel(s)
            .gradients
            .as_ref()
            .ok_or_else(|| invalid("only a meta-GGA reference has grid orbital gradients"))?
            .as_slice())
    }
    /// The matrix of spin `s` of XC potential parts packed as [`Semilocal::potentials`]
    /// (`[spin][point]`, or a meta-GGA's `[local | tau | flux]`), with `extra` added to the
    /// local part.
    fn xc_elements(
        &self,
        s: usize,
        potentials: &[f64],
        extra: Option<&[f64]>,
    ) -> Result<DMatrix<f64>> {
        let n = self.points();
        let mut local = potentials[s * n..(s + 1) * n].to_vec();
        if let Some(extra) = extra {
            for (l, e) in local.iter_mut().zip(extra) {
                *l += e;
            }
        }
        if potentials.len() == 2 * n {
            return self.elements(s, &local);
        }
        meta::general_elements(
            self.samples(s)?,
            self.gradients(s)?,
            n,
            self.m,
            &local,
            &potentials[(2 + s) * n..(3 + s) * n],
            &potentials[(4 + 3 * s) * n..(7 + 3 * s) * n],
            self.integrals.grid.index.dv(),
        )
    }
    /// A meta-GGA's changes `(dn, dg, dtau)` of the symmetric spin changes `v`
    /// (`[spin][point]`, `[spin][axis][point]`, `[spin][point]`).
    fn meta_changes(&self, v: [&DMatrix<f64>; 2]) -> Result<[Buffer<f64>; 3]> {
        let (ctx, n, m) = (&self.integrals.context, self.points(), self.m);
        let mut density = ctx.allocate::<f64>(2 * n)?;
        let mut gradient = ctx.allocate::<f64>(6 * n)?;
        let mut tau = ctx.allocate::<f64>(2 * n)?;
        for (s, v) in v.iter().enumerate() {
            meta::general_changes(
                self.samples(s)?,
                self.gradients(s)?,
                n,
                m,
                v,
                &mut density.as_mut_slice()[s * n..(s + 1) * n],
                &mut gradient.as_mut_slice()[3 * s * n..3 * (s + 1) * n],
                &mut tau.as_mut_slice()[s * n..(s + 1) * n],
            )?;
        }
        Ok([density, gradient, tau])
    }
    /// `dv sum_r phi_p(r) w(r) phi_q(r)` of a local potential in the orbitals of spin `s`.
    fn elements(&self, s: usize, w: &[f64]) -> Result<DMatrix<f64>> {
        let (n, m) = (self.points(), self.m);
        let phi = self.samples(s)?;
        let dv = self.integrals.grid.index.dv();
        Ok(DMatrix::from_fn(m, m, |p, q| {
            let (a, b) = (&phi[p * n..(p + 1) * n], &phi[q * n..(q + 1) * n]);
            dv * (0..n).map(|k| a[k] * w[k] * b[k]).sum::<f64>()
        }))
    }
    /// The first-order Fock change `F1_s[V] = d/ds F_s[D + s V]` of the spin densities `V` (orbital
    /// matrices), in the canonical orbitals: Hartree of the total change, the compensation pairs and
    /// the XC kernel and the +U map. Also its XC potential change (`[spin][point]`).
    /// Hartree-Fock: `J - K` of its pair factors, and no potential change.
    fn fock1(&self, v: [&DMatrix<f64>; 2]) -> Result<([DMatrix<f64>; 2], Option<Buffer<f64>>)> {
        let Some(semilocal) = self.semilocal.as_ref() else {
            return Ok((self.exchange_fock(v, true), None));
        };
        let ctx = &self.integrals.context;
        let n = self.points();
        let mut spins = ctx.allocate::<f64>(2 * n)?;
        self.density(0, v[0], &mut spins.as_mut_slice()[..n])?;
        self.density(1, v[1], &mut spins.as_mut_slice()[n..])?;
        let xc_change = match (&semilocal.kernel, &semilocal.meta) {
            (Some(kernel), _) => {
                let mut xc_change = ctx.allocate::<f64>(2 * n)?;
                let mut scratch = kernel.scratch(ctx)?;
                kernel.apply(
                    &self.integrals.grid,
                    spins.as_slice(),
                    xc_change.as_mut_slice(),
                    &mut scratch,
                )?;
                xc_change
            }
            (None, Some(ground)) => {
                let [_, gradient, tau] = self.meta_changes(v)?;
                let mut xc_change = ctx.allocate::<f64>(10 * n)?;
                let (local, rest) = xc_change.as_mut_slice().split_at_mut(2 * n);
                let (kinetic, flux) = rest.split_at_mut(2 * n);
                ground.kernel.apply(
                    spins.as_slice(),
                    gradient.as_slice(),
                    tau.as_slice(),
                    xc_kernel::MetaChange {
                        local,
                        tau: kinetic,
                        flux,
                    },
                )?;
                xc_change
            }
            (None, None) => return Err(invalid("a Kohn-Sham reference has no XC kernel")),
        };
        let mut poisson = Poisson::isolated(ctx, &self.integrals.grid)?;
        let mut source = ctx.allocate::<C64>(n)?;
        let mut hartree = ctx.allocate::<C64>(n)?;
        for k in 0..n {
            source.as_mut_slice()[k] = C64::new(spins.as_slice()[k] + spins.as_slice()[n + k], 0.);
        }
        poisson.solve_into(source.as_slice(), hartree.as_mut_slice())?;
        // The compensation pairs see the total density matrix: `w_k tr[B_k (D_up + D_dn)]`.
        let charges: Vec<f64> = self
            .weights
            .iter()
            .enumerate()
            .map(|(k, w)| {
                w * (self.channel(0).pairs[k].dot(v[0]) + self.channel(1).pairs[k].dot(v[1]))
            })
            .collect();
        let hartree: Vec<f64> = hartree.as_slice().iter().map(|z| z.re).collect();
        let mut out = [DMatrix::zeros(0, 0), DMatrix::zeros(0, 0)];
        for (s, f) in out.iter_mut().enumerate() {
            *f = self.xc_elements(s, xc_change.as_slice(), Some(&hartree))?;
            for (b, c) in self.channel(s).pairs.iter().zip(&charges) {
                *f += b * *c;
            }
        }
        if let Some(hubbard) = self.hubbard_fock(v, false)? {
            for (f, u) in out.iter_mut().zip(hubbard) {
                *f += u;
            }
        }
        if let Some(screened) = self.screened_fock(v)? {
            for (f, x) in out.iter_mut().zip(screened) {
                *f += x;
            }
        }
        if let Some(hybrid) = &self.hybrid {
            if let Some((_, kernels, _)) = &hybrid.spheres {
                let frame = [0, 1].map(|s| {
                    let c = &self.channel(s).frame;
                    c * v[s] * c.transpose()
                });
                let focks = kernels.apply(&frame[0], &frame[1])?;
                for (s, f) in out.iter_mut().enumerate() {
                    let c = &self.channel(s).frame;
                    *f -= c.transpose() * &focks[s] * c * hybrid.fraction;
                }
            }
        }
        Ok((out, Some(xc_change)))
    }
    /// `d^2 v_s / dt^2` along the spin densities `t` (`[spin][point]`): twice
    /// [`xc_kernel::second_change`]. A meta-GGA's three parts (packed as
    /// [`Semilocal::potentials`]) along `D + t T + t i T'` with `imaginary` the antisymmetric
    /// `T'` of each spin: twice [`xc_kernel::second_change_meta`], whose kinetic densities curve
    /// by `tau2 = -|j[T']|^2 / 2n` (`tau~ = tau - |j|^2 / 2n`) and whose local part adds the
    /// `1/n` of `tau~`, `-v_tau tau2 / n`. The mixed terms of the two directions are of third
    /// order, so this is the second change along `T` plus that along `i T'`.
    fn second(&self, t: [&DMatrix<f64>; 2], imaginary: [&DMatrix<f64>; 2]) -> Result<Buffer<f64>> {
        let semilocal = self.semilocal()?;
        let ctx = &self.integrals.context;
        let n = self.points();
        if let Some(ground) = &semilocal.meta {
            let [density, gradient, tau] = self.meta_changes(t)?;
            let (rho, weights) = (ground.rho.as_slice(), ground.weights.as_slice());
            let mut tau2 = ctx.allocate::<f64>(2 * n)?;
            for (s, v) in imaginary.iter().enumerate() {
                let j = self.current(s, v)?;
                let j = j.as_slice();
                for k in 0..n {
                    let density = rho[s * n + k];
                    if density > 0. {
                        let j2: f64 = (0..3).map(|a| j[a * n + k].powi(2)).sum();
                        tau2.as_mut_slice()[s * n + k] = -0.5 * j2 / density;
                    }
                }
            }
            let mut out = xc_kernel::second_change_meta(
                ctx,
                rho,
                ground.tau.as_slice(),
                ground.gradient.as_slice(),
                density.as_slice(),
                gradient.as_slice(),
                tau.as_slice(),
                tau2.as_slice(),
            )?;
            for k in 0..2 * n {
                out.as_mut_slice()[k] -= weights[k] * tau2.as_slice()[k];
            }
            for v in out.as_mut_slice() {
                *v *= 2.;
            }
            return Ok(out);
        }
        let mut spins = ctx.allocate::<f64>(2 * n)?;
        self.density(0, t[0], &mut spins.as_mut_slice()[..n])?;
        self.density(1, t[1], &mut spins.as_mut_slice()[n..])?;
        let mut out = xc_kernel::second_change(
            ctx,
            &self.integrals.grid,
            semilocal.kernel_density.as_slice(),
            semilocal.name,
            semilocal.relativistic,
            spins.as_slice(),
        )?;
        for v in out.as_mut_slice() {
            *v *= 2.;
        }
        Ok(out)
    }
    /// The fixed-density gradient of the spin densities `d` (orbital matrices of each spin), with
    /// the energy-weighted densities `w` (orbital matrices of each spin) in the overlap term and
    /// the XC potentials `v` (`[spin][point]`) in the exchange-correlation terms: the ground-state
    /// force contraction at a given density, three per atom.
    fn fixed(
        &self,
        d: [&DMatrix<f64>; 2],
        w: Option<[&DMatrix<f64>; 2]>,
        v: &[f64],
    ) -> Result<Vec<f64>> {
        Ok(self.fixed_gradient(d, w, v)?.total.as_slice().to_vec())
    }
    fn fixed_gradient(
        &self,
        d: [&DMatrix<f64>; 2],
        w: Option<[&DMatrix<f64>; 2]>,
        v: &[f64],
    ) -> Result<Gradient> {
        let integrals = self.integrals;
        let ctx = &integrals.context;
        let n = self.points();
        let setup = integrals
            .paw_setup
            .as_ref()
            .ok_or_else(|| invalid("excited-state forces need a PAW setup"))?;
        // The densities of `d` as they are: `dft::grid_densities` floors a density at zero, and the
        // directions `D +- P`, `D +- T` reach negative values where the virtual orbitals dominate.
        let mut valence = ctx.allocate::<f64>(2 * n)?;
        self.density(0, d[0], &mut valence.as_mut_slice()[..n])?;
        self.density(1, d[1], &mut valence.as_mut_slice()[n..])?;
        let mut matrices = vec![
            (d[0], &self.channel(0).raw, 0, false),
            (d[1], &self.channel(1).raw, 1, false),
        ];
        if let Some(w) = w {
            matrices.push((w[0], &self.channel(0).raw, 0, true));
            matrices.push((w[1], &self.channel(1).raw, 1, true));
        }
        frozen_contraction(integrals, setup, valence, &matrices, v)
    }
    /// The orbitals (raw atomic basis) and couplings of a hybrid's exchange along the directions
    /// ([`MolecularIntegrals::coupled_exchange_gradient`]):
    /// `-(a/2) sum_s [2 K(D0_s, P_s) + 2 K(T_s, T_s) + 2 X(i T'_s)]`, the first derivative along
    /// the relaxed density `p`, the second along the transition density `t` and along the
    /// imaginary one `i T'` of the amplitudes `v` (`T'_ia = v_ia / 2`), with
    /// `K(x, y) = int x(r, r') y(r', r) v_sr` and `X(h) = K(h, h^+)`. Each density is
    /// `sum_k lambda_k phi_k phi_k^+`: the occupied orbitals, the eigenvectors of `p` and `t`,
    /// and `(a_k +- i b_k) / sqrt 2` (weights `-+ sigma_k`) of the singular pairs of `v / 2`. A
    /// closed shell's one channel stands for both spins.
    fn coupled_directions(
        &self,
        p: &[DMatrix<f64>; 2],
        t: &[DMatrix<f64>; 2],
        v: &[DMatrix<f64>; 2],
    ) -> Result<(Vec<Vec<C64>>, DMatrix<f64>)> {
        let m = self.m;
        let spins = if self.channels.len() == 1 { 2. } else { 1. };
        let mut orbitals: Vec<Vec<C64>> = vec![];
        // (channel, density, weight); density 0 the occupied, 1 relaxed, 2 symmetric, 3 imaginary.
        let mut tags: Vec<(usize, usize, f64)> = vec![];
        for (s, channel) in self.channels.iter().enumerate() {
            let (raw, o) = (&channel.raw, channel.o);
            let real = |x: nalgebra::DVector<f64>| x.iter().map(|&z| C64::new(z, 0.)).collect();
            for i in 0..o {
                orbitals.push(real(raw.column(i).into_owned()));
                tags.push((s, 0, 1.));
            }
            for (density, x) in [(1, &p[s]), (2, &t[s])] {
                let (values, vectors) = real_eigen((x + x.transpose()) * 0.5)?;
                let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
                for (j, &value) in values.iter().enumerate() {
                    if value.abs() > 1e-14 * largest {
                        orbitals.push(real(raw * vectors.column(j)));
                        tags.push((s, density, value));
                    }
                }
            }
            let svd = (&v[s] * 0.5).svd(true, true);
            let (Some(left), Some(right)) = (svd.u.as_ref(), svd.v_t.as_ref()) else {
                return Err(invalid("the amplitudes have no singular vectors"));
            };
            let largest = svd.singular_values.iter().fold(0f64, |a, v| a.max(*v));
            for (k, &sigma) in svd.singular_values.iter().enumerate() {
                if sigma <= 1e-14 * largest {
                    continue;
                }
                let a = raw.columns(0, o) * left.column(k);
                let b = raw.columns(o, m - o) * right.row(k).transpose();
                for sign in [1., -1.] {
                    orbitals.push(
                        a.iter()
                            .zip(b.iter())
                            .map(|(&x, &y)| C64::new(x, sign * y) * std::f64::consts::FRAC_1_SQRT_2)
                            .collect(),
                    );
                    tags.push((s, 3, -sign * sigma));
                }
            }
        }
        let couplings = DMatrix::from_fn(orbitals.len(), orbitals.len(), |k, l| {
            let ((sk, dk, wk), (sl, dl, wl)) = (tags[k], tags[l]);
            if sk != sl {
                return 0.;
            }
            spins
                * match (dk, dl) {
                    (0, 1) => wl,
                    (1, 0) => wk,
                    (2, 2) | (3, 3) => 2. * wk * wl,
                    _ => 0.,
                }
        });
        Ok((orbitals, couplings))
    }
    /// Hartree-Fock's explicit derivative of the whole Lagrangian at fixed raw-AO matrices, ion-ion
    /// repulsion and ground state included, through the two-body route of
    /// [`MolecularIntegrals::fixed_density_gradient`]: the one-body density `D + P`, the
    /// energy-weighted density `W_0 + W` and the effective two-body density
    /// `C[D, D] + C[D, P] + C[P, D] + 2 C[T, T] + 2 C[iT', iT']` of the bilinear
    /// `C[x, y] = J(x, y) - sum_s K(x_s, y_s)` (`E_2 = C[D, D] / 2`, `K(x, y) = tr(x L y L)` per
    /// factor), with `T` the symmetric and `iT'` the imaginary transition density of each spin
    /// (canonical-orbital matrices). `d^2 E_2 / dt^2` along `D + t iT'` is the exchange of
    /// `A - B`: `C[iT', iT'] = -C[T', T']`, whose Coulomb part vanishes.
    fn hartree_fock_explicit(
        &self,
        d: &[DMatrix<f64>; 2],
        p: &[DMatrix<f64>; 2],
        t: &[DMatrix<f64>; 2],
        imaginary: &[DMatrix<f64>; 2],
        w: &[DMatrix<f64>; 2],
    ) -> Result<Gradient> {
        use crate::molecular::variational_gradient::{two_body_forms, FixedDensities};
        let m = self.m;
        // The `m^4` two-body density, its pair matrix and the eigenvectors of the forms.
        let _memory = self
            .integrals
            .context
            .reserve(crate::product(&[m, m, m, m, 8, 3])?)?;
        let ao = |x: &[DMatrix<f64>; 2]| {
            [0, 1].map(|s| {
                let c = &self.channel(s).raw;
                c * &x[s] * c.transpose()
            })
        };
        let (d, p, t, imaginary) = (ao(d), ao(p), ao(t), ao(imaginary));
        // Gamma_pqrs = C_(pr),(qs), the convention of `two_body_forms`.
        let at = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
        let mut two = vec![0.; m.pow(4)];
        let mut add = |x: &[DMatrix<f64>; 2], y: &[DMatrix<f64>; 2], c: f64, coulomb: bool| {
            if coulomb {
                let (x, y) = (&x[0] + &x[1], &y[0] + &y[1]);
                for p in 0..m {
                    for q in 0..m {
                        for r in 0..m {
                            for s in 0..m {
                                two[at(p, r, q, s)] += c * x[(p, q)] * y[(r, s)];
                            }
                        }
                    }
                }
            }
            for spin in 0..2 {
                let (x, y) = (&x[spin], &y[spin]);
                for p in 0..m {
                    for q in 0..m {
                        for r in 0..m {
                            for s in 0..m {
                                two[at(p, q, s, r)] -= c * x[(p, q)] * y[(r, s)];
                            }
                        }
                    }
                }
            }
        };
        add(&d, &d, 1., true);
        add(&d, &p, 1., true);
        add(&p, &d, 1., true);
        add(&t, &t, 2., true);
        add(&imaginary, &imaginary, -2., false);
        let ground_weighted = [0, 1].map(|s| {
            let channel = self.channel(s);
            DMatrix::from_fn(m, m, |a, b| {
                if a == b && a < channel.o {
                    channel.energies[a]
                } else {
                    0.
                }
            })
        });
        let (ground_weighted, w) = (ao(&ground_weighted), ao(w));
        let densities = FixedDensities {
            one: &d[0] + &d[1] + &p[0] + &p[1],
            weighted: &ground_weighted[0] + &ground_weighted[1] + &w[0] + &w[1],
            frame: DMatrix::identity(m, m),
            two: two_body_forms(&two, m)?,
            cluster_frame: None,
        };
        self.integrals.fixed_density_gradient(&densities)
    }
}

/// The orbital Hessian `A + B` the Z-vector solves with: the closed shell's singlet, or the
/// unrestricted reference's spin-conserving one.
pub(super) trait Hessian {
    fn gaps(&self) -> &[f64];
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
}
impl Hessian for Response<'_> {
    fn gaps(&self) -> &[f64] {
        self.delta()
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.sum(block)
    }
}
impl Hessian for Unrestricted<'_> {
    fn gaps(&self) -> &[f64] {
        self.delta()
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.sum(block)
    }
}

/// `A z = b` by conjugate gradients on the singlet `A + B` of `response`, preconditioned by the
/// orbital-energy differences: the solution, the final relative residual and the iterations.
pub(super) fn z_vector(
    response: &dyn Hessian,
    b: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, f64, usize)> {
    let dimension = b.len();
    let delta = response.gaps();
    let norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut x = vec![0.; dimension];
    if norm == 0. {
        return Ok((x, 0., 0));
    }
    let apply = |v: &[f64]| -> Result<Vec<f64>> {
        Ok(response
            .apply(&DMatrix::from_column_slice(dimension, 1, v))?
            .column(0)
            .iter()
            .copied()
            .collect())
    };
    let mut r = b.to_vec();
    let mut z: Vec<f64> = r.iter().zip(delta).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut residual = 1.;
    for iteration in 1..=max_iterations {
        let q = apply(&p)?;
        let pq: f64 = p.iter().zip(&q).map(|(a, b)| a * b).sum();
        if pq <= 0. {
            return Err(invalid("the singlet orbital Hessian of the reference is not positive: the ground state is unstable"));
        }
        let alpha = rz / pq;
        for i in 0..dimension {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        residual = r.iter().map(|v| v * v).sum::<f64>().sqrt() / norm;
        if residual < tolerance {
            return Ok((x, residual, iteration));
        }
        z = r.iter().zip(delta).map(|(r, d)| r / d).collect();
        let next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        for i in 0..dimension {
            p[i] = z[i] + (next / rz) * p[i];
        }
        rz = next;
    }
    Err(crate::Error::Kernel(format!(
        "the Z-vector did not converge in {max_iterations} iterations (relative residual {residual:.2e})"
    )))
}

/// The energy and analytic gradient of the excited state `root` (0-based) of `request`'s spin
/// symmetry (Tamm-Dancoff or Casida, adiabatic LDA or PBE, a closed-shell PAW-LCAO molecule).
pub fn excited_state(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
    root: usize,
) -> Result<ExcitedState> {
    pick(integrals, reference, request, root, None)?.gradient(integrals, reference)
}

/// [`excited_state`] for the next geometry of a scan ([`Track::Overlap`]): of the lowest
/// `previous.root + 3` roots, the one whose transition density overlaps most with
/// `previous.transition` (in the overlap metric of this geometry's basis).
pub fn followed_excited_state(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
    previous: &Followed,
) -> Result<ExcitedState> {
    pick(integrals, reference, request, previous.root, Some(previous))?
        .gradient(integrals, reference)
}

/// One excited state picked for its energy: the excitation solve and the choice of the root.
/// [`Picked::gradient`] adds the gradient (the Z-vector and the explicit derivative), several
/// times the solve's cost (6.7 times on water, PBE DZP, grid 0.2 A), so an energy needs no more.
pub struct Picked {
    found: crate::molecular::excitations::Excitations,
    model: Model,
    /// The roots degenerate with `root` (within [`DEGENERATE`]), `root` included: one root, or
    /// a spinor group whose state-averaged energy and gradient this state carries.
    group: Vec<usize>,
    pub root: usize,
    pub spin: Spin,
    /// `omega` and `E_0 + omega` with the ion-ion repulsion, Hartree.
    pub excitation: f64,
    pub energy: f64,
    /// The distance to the nearest other root of the solve, Hartree.
    pub separation: f64,
    pub root_residual: f64,
    /// The state as the next geometry of a scan follows it.
    pub followed: Followed,
    /// The overlap with the previous geometry's state that picked this root, between 0 and 1;
    /// `None` for a root picked by index.
    pub overlap: Option<f64>,
}

/// The root `root` of `request` ([`excited_state`]) or, given `previous`, the one that follows it
/// ([`followed_excited_state`]), without its gradient.
pub fn pick(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
    root: usize,
    previous: Option<&Followed>,
) -> Result<Picked> {
    let model = check(integrals, reference, request)?;
    let mut solve = request.clone();
    solve.num_states = solve
        .num_states
        .max(root + if previous.is_some() { 3 } else { 2 });
    let found = excitations(integrals, reference, &solve)?;
    let spin = found.spin;
    let count = found.energies.len();
    // A spinor state is followed by its complex half transition density (real and imaginary
    // parts), with a phase-invariant overlap.
    let spinor = found.spinor.as_ref();
    let densities = |j: usize| -> Result<Vec<DMatrix<f64>>> {
        if let Some(spinors) = spinor {
            return Ok(super::spinor::gradient::followed_transition(
                spinors,
                found.channels[0].occupied,
                j,
            ));
        }
        (0..found.channels.len())
            .map(|c| found.transition_density(j, c))
            .collect()
    };
    let (root, overlap) = match previous {
        None => (root, None),
        Some(previous) => {
            let m = integrals.m();
            let (parts, size) = match spinor {
                Some(_) => (2, 2 * m),
                None => (found.channels.len(), m),
            };
            if previous.transition.len() != parts
                || previous
                    .transition
                    .iter()
                    .any(|t| t.shape() != (size, size))
            {
                return Err(invalid(
                    "the state followed from the previous geometry has a different basis",
                ));
            }
            let s = DMatrix::from_row_slice(m, m, integrals.overlap.as_slice());
            let mut best = (0, -1.);
            for j in 0..count {
                let current = densities(j)?;
                let value = match spinor {
                    Some(_) => super::spinor::gradient::transition_overlap(
                        &s,
                        &previous.transition,
                        &current,
                    ),
                    None => transition_overlap(&s, &previous.transition, &current),
                };
                if value > best.1 {
                    best = (j, value);
                }
            }
            (best.0, Some(best.1))
        }
    };
    if root >= count {
        return Err(invalid(format!(
            "root {root} is beyond the {count} roots of this basis"
        )));
    }
    let followed = Followed {
        root,
        transition: densities(root)?,
    };
    // A spinor group of degenerate roots (an |Omega| >= 1 pair) has no gradient of one member,
    // but the mean of its energies is smooth and its gradient is the mean of the members'
    // (invariant under turns within the group): the state-averaged energy and gradient.
    let group: Vec<usize> = (0..count)
        .filter(|&j| (found.energies[j] - found.energies[root]).abs() < DEGENERATE)
        .collect();
    let omega = group.iter().map(|&j| found.energies[j]).sum::<f64>() / group.len() as f64;
    let separation = (0..count)
        .filter(|j| !group.contains(j))
        .map(|j| (found.energies[j] - omega).abs())
        .fold(f64::INFINITY, f64::min);
    if group.len() > 1 && model != Model::Spinor {
        return Err(crate::Error::Unsupported(format!(
            "root {root} is degenerate with another root (within {DEGENERATE:.0e} Ha): a multiplet has no single gradient"
        )));
    }
    Ok(Picked {
        root_residual: group.iter().map(|&j| found.residuals[j]).fold(0., f64::max),
        found,
        model,
        group,
        root,
        spin,
        excitation: omega,
        energy: reference.free_energy() + integrals.nuclear + omega,
        separation,
        followed,
        overlap,
    })
}

impl Picked {
    /// The excited state with its analytic gradient (`integrals` and `reference` those it was
    /// picked from).
    pub fn gradient(
        &self,
        integrals: &MolecularIntegrals,
        reference: &MeanField,
    ) -> Result<ExcitedState> {
        let (found, model, root, spin) = (&self.found, self.model, self.root, self.spin);
        let (omega, separation) = (self.excitation, self.separation);
        if model == Model::Spinor {
            let mut state = super::spinor::gradient::state_gradient(
                integrals,
                reference,
                found,
                self.group[0],
            )?
            .0;
            for &j in &self.group[1..] {
                let other =
                    super::spinor::gradient::state_gradient(integrals, reference, found, j)?.0;
                let total = state.gradient.total.as_mut_slice();
                total
                    .iter_mut()
                    .zip(other.gradient.total.as_slice())
                    .for_each(|(a, b)| *a += b);
                for ((_, a), (_, b)) in state.parts.iter_mut().zip(&other.parts) {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                }
                state.relaxed += &other.relaxed;
                for (a, b) in state.transition.iter_mut().zip(&other.transition) {
                    *a += b;
                }
                state.energy_weighted += &other.energy_weighted;
                state.z_residual = state.z_residual.max(other.z_residual);
                state.z_iterations += other.z_iterations;
            }
            if self.group.len() > 1 {
                let share = 1. / self.group.len() as f64;
                state
                    .gradient
                    .total
                    .as_mut_slice()
                    .iter_mut()
                    .for_each(|a| *a *= share);
                state
                    .parts
                    .iter_mut()
                    .for_each(|(_, a)| a.iter_mut().for_each(|x| *x *= share));
                state.relaxed *= share;
                state.transition.iter_mut().for_each(|a| *a *= share);
                state.energy_weighted *= share;
            }
            return Ok(ExcitedState {
                root,
                spin,
                excitation: omega,
                energy: self.energy,
                gradient: state.gradient,
                parts: state.parts,
                relaxed: state.relaxed,
                transition: state.transition,
                energy_weighted: state.energy_weighted,
                separation,
                z_residual: state.z_residual,
                z_iterations: state.z_iterations,
                root_residual: self.root_residual,
                followed: self.followed.clone(),
                overlap: self.overlap,
            });
        }
        integrals.context.run(|| {
            // A closed shell's Z-vector solves with the singlet Hessian whatever the root's
            // spin; an unrestricted reference's with its spin-conserving one.
            let unrestricted = spin == Spin::Unrestricted;
            let singlet;
            let open;
            let (ground, hessian): (Ground, &dyn Hessian) = if unrestricted {
                open = Unrestricted::new(integrals, reference, Coupling::Adiabatic)?;
                let channels = (0..2)
                    .map(|s| {
                        (
                            found.channels[s].occupied,
                            open.orbital_energies(s).to_vec(),
                            open.orbitals(s).clone(),
                        )
                    })
                    .collect();
                (Ground::new(integrals, reference, channels, model)?, &open)
            } else {
                singlet = Response::new(integrals, reference, Spin::Singlet, Coupling::Adiabatic)?;
                let channel = (
                    singlet.occupied(),
                    singlet.orbital_energies().to_vec(),
                    singlet.orbitals().clone(),
                );
                (
                    Ground::new(integrals, reference, vec![channel], model)?,
                    &singlet,
                )
            };
            for (s, channel) in ground.channels.iter().enumerate() {
                let c = &found.channels[s];
                if c.occupied + c.virtual_orbitals != ground.m
                    || (&c.atomic_orbitals - &channel.raw).amax() > 1e-10
                {
                    return Err(invalid(
                        "the excitation and the gradient see different canonical orbitals",
                    ));
                }
            }
            let m = ground.m;
            let sizes = [0, 1].map(|s| (ground.channel(s).o, m - ground.channel(s).o));
            // Spin-orbital amplitudes of each spin: U = X + Y (the kernel's), V = X - Y. A closed
            // shell's spin-adapted root is 1/sqrt(2) of each spin, with the triplet's sign.
            let (us, vs): ([DMatrix<f64>; 2], [DMatrix<f64>; 2]) = if unrestricted {
                (
                    [found.plus[root][0].clone(), found.plus[root][1].clone()],
                    [found.minus[root][0].clone(), found.minus[root][1].clone()],
                )
            } else {
                let x = &found.plus[root][0] * std::f64::consts::FRAC_1_SQRT_2;
                let y = &found.minus[root][0] * std::f64::consts::FRAC_1_SQRT_2;
                let sign = if spin == Spin::Triplet { -1. } else { 1. };
                ([x.clone(), &x * sign], [y.clone(), &y * sign])
            };
            let xs = &us;
            let qvv =
                [0, 1].map(|s| (us[s].transpose() * &us[s] + vs[s].transpose() * &vs[s]) * 0.5);
            let qoo =
                [0, 1].map(|s| (&us[s] * us[s].transpose() + &vs[s] * vs[s].transpose()) * 0.5);
            // The transition and unrelaxed difference densities of each spin (orbital matrices).
            let ov_symmetric = |s: usize, a: &DMatrix<f64>| {
                let (o, v) = sizes[s];
                let mut t = DMatrix::zeros(m, m);
                t.view_mut((0, o), (o, v)).copy_from(&(a * 0.5));
                t.view_mut((o, 0), (v, o)).copy_from(&(a.transpose() * 0.5));
                t
            };
            let t = [0, 1].map(|s| ov_symmetric(s, &xs[s]));
            let difference = [0, 1].map(|s| {
                let (o, v) = sizes[s];
                let mut d = DMatrix::zeros(m, m);
                d.view_mut((o, o), (v, v)).copy_from(&qvv[s]);
                d.view_mut((0, 0), (o, o)).copy_from(&(-&qoo[s]));
                d
            });
            let (kt, dv_t) = ground.fock1([&t[0], &t[1]])?;
            let (kd, dv_d) = ground.fock1([&difference[0], &difference[1]])?;
            // The imaginary transition density `i T'` of `V` (`T'` antisymmetric) and
            // `Im F1[i T']`, the antisymmetric (A - B) channel of the kernel: `omega` is the
            // second derivative of the energy along `D + t T` plus that along `D + t i T'`.
            let imaginary = [0, 1].map(|s| {
                let (o, v) = sizes[s];
                let mut t = DMatrix::zeros(m, m);
                t.view_mut((0, o), (o, v)).copy_from(&(&vs[s] * 0.5));
                t.view_mut((o, 0), (v, o))
                    .copy_from(&(vs[s].transpose() * -0.5));
                t
            });
            let ki = ground.imaginary_fock([&imaginary[0], &imaginary[1]])?;
            // The third-order XC change (none for Hartree-Fock, whose energy is quadratic): along
            // `T`, and for a meta-GGA along `i T'` too.
            let second = match ground.semilocal {
                Some(_) => Some(ground.second([&t[0], &t[1]], [&imaginary[0], &imaginary[1]])?),
                None => None,
            };
            let mut g = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
            if let Some(second) = &second {
                for (s, g) in g.iter_mut().enumerate() {
                    *g = ground.xc_elements(s, second.as_slice(), None)?;
                }
            }
            if let Some(spheres) = ground.hybrid_second([&t[0], &t[1]])? {
                for (g, x) in g.iter_mut().zip(spheres) {
                    *g += x;
                }
            }
            let block = |a: &DMatrix<f64>, r: usize, c: usize, nr: usize, nc: usize| {
                a.view((r, c), (nr, nc)).into_owned()
            };
            // The orbital gradient R of each spin (a closed shell's down spin equals its up spin
            // for a singlet and a triplet alike, and its Hessian acts on the up block).
            let r = [0, 1].map(|s| {
                let (o, v) = sizes[s];
                let mut r = (block(&kd[s], 0, o, o, v) * 2.)
                    + (&xs[s] * block(&kt[s], o, o, v, v)) * 2.
                    - (block(&kt[s], 0, 0, o, o) * &xs[s]) * 2.
                    + block(&g[s], 0, o, o, v) * 2.;
                if let Some(ki) = &ki {
                    r += (block(&ki[s], 0, 0, o, o) * &vs[s]) * 2.
                        - (&vs[s] * block(&ki[s], o, o, v, v)) * 2.;
                }
                r
            });
            let spins: &[usize] = if unrestricted { &[0, 1] } else { &[0] };
            let rhs: Vec<f64> = spins
                .iter()
                .flat_map(|&s| {
                    let (o, v) = sizes[s];
                    let r = &r[s];
                    (0..o).flat_map(move |i| (0..v).map(move |a| -r[(i, a)]))
                })
                .collect();
            let (z, z_residual, z_iterations) = z_vector(hessian, &rhs, 1e-11, 500)?;
            let z = {
                let (oa, va) = sizes[0];
                let alpha = DMatrix::from_row_slice(oa, va, &z[..oa * va]);
                let beta = if unrestricted {
                    let (ob, vb) = sizes[1];
                    DMatrix::from_row_slice(ob, vb, &z[oa * va..oa * va + ob * vb])
                } else {
                    alpha.clone()
                };
                [alpha, beta]
            };
            let zt = [0, 1].map(|s| ov_symmetric(s, &z[s]));
            let (kz, dv_z) = ground.fock1([&zt[0], &zt[1]])?;
            // The energy-weighted density of the excitation of each spin, in its canonical orbitals.
            let w = [0, 1].map(|s| {
                let (o, v) = sizes[s];
                let eps = &ground.channel(s).energies;
                let xs = &xs[s];
                let mt = block(&kt[s], 0, o, o, v);
                let mut oo = block(&kd[s], 0, 0, o, o)
                    + block(&kz[s], 0, 0, o, o)
                    + block(&g[s], 0, 0, o, o)
                    + (xs * mt.transpose() + &mt * xs.transpose()) * 0.5;
                for i in 0..o {
                    for j in 0..o {
                        oo[(i, j)] -= 0.5 * (eps[i] + eps[j]) * qoo[s][(i, j)];
                    }
                }
                let mut vv = (xs.transpose() * &mt + mt.transpose() * xs) * 0.5;
                for a in 0..v {
                    for b in 0..v {
                        vv[(a, b)] += 0.5 * (eps[o + a] + eps[o + b]) * qvv[s][(a, b)];
                    }
                }
                let mut ov = block(&kt[s], 0, 0, o, o) * xs;
                for i in 0..o {
                    for a in 0..v {
                        ov[(i, a)] += 0.5 * eps[i] * z[s][(i, a)];
                    }
                }
                if let Some(ki) = &ki {
                    let (vs, mi) = (&vs[s], block(&ki[s], 0, o, o, v));
                    oo += (&mi * vs.transpose() + vs * mi.transpose()) * 0.5;
                    vv += (mi.transpose() * vs + vs.transpose() * &mi) * 0.5;
                    ov -= block(&ki[s], 0, 0, o, o) * vs;
                }
                let mut ws = DMatrix::zeros(m, m);
                ws.view_mut((0, 0), (o, o)).copy_from(&oo);
                ws.view_mut((o, o), (v, v)).copy_from(&vv);
                ws.view_mut((0, o), (o, v)).copy_from(&ov);
                ws.view_mut((o, 0), (v, o)).copy_from(&ov.transpose());
                ws
            });
            // The explicit derivative, exact in both directions (module documentation).
            let d0 = [0, 1].map(|s| {
                let mut d = DMatrix::zeros(m, m);
                d.view_mut((0, 0), (sizes[s].0, sizes[s].0))
                    .fill_with_identity();
                d
            });
            let p = [0, 1].map(|s| &difference[s] + &zt[s]);
            let gradient = integrals.gradient(reference)?;
            let size = gradient.total.as_slice().len();
            // The state from the ground gradient and the parts of the excitation's.
            let finish = |mut gradient: Gradient, parts: Vec<(&'static str, Vec<f64>)>| {
                let total: Vec<f64> = (0..size)
                    .map(|k| parts.iter().map(|(_, v)| v[k]).sum())
                    .collect();
                if total.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("nonfinite excited-state gradient"));
                }
                gradient.total.as_mut_slice().copy_from_slice(&total);
                gradient.terms.clear();
                // Orbital matrices of each spin to the orthonormal frame.
                let framed = |s: usize, a: &DMatrix<f64>| {
                    let frame = &ground.channel(s).frame;
                    frame * a * frame.transpose()
                };
                Ok(ExcitedState {
                    root,
                    spin,
                    excitation: omega,
                    energy: self.energy,
                    gradient,
                    parts,
                    relaxed: framed(0, &p[0]) + framed(1, &p[1]),
                    transition: [0, 1].map(|s| framed(s, &t[s])),
                    energy_weighted: framed(0, &w[0]) + framed(1, &w[1]),
                    separation,
                    z_residual,
                    z_iterations,
                    root_residual: self.root_residual,
                    followed: self.followed.clone(),
                    overlap: self.overlap,
                })
            };
            let (Some(semilocal), Some(second), Some(dv_t), Some(dv_d), Some(dv_z)) =
                (ground.semilocal.as_ref(), second, dv_t, dv_d, dv_z)
            else {
                // Hartree-Fock: the whole Lagrangian in one fixed-density contraction.
                let total = ground.hartree_fock_explicit(&d0, &p, &t, &imaginary, &w)?;
                let total = total.total.as_slice();
                let parts = vec![
                    ("ground", gradient.total.as_slice().to_vec()),
                    (
                        "excitation",
                        (0..size)
                            .map(|k| total[k] - gradient.total.as_slice()[k])
                            .collect(),
                    ),
                ];
                return finish(gradient, parts);
            };
            let v0 = semilocal.potentials.as_slice();
            let sec = second.as_slice();
            let (dt, dp) = (dv_t.as_slice(), (dv_d.as_slice(), dv_z.as_slice()));
            let along = |sign: f64| -> Vec<f64> {
                (0..v0.len())
                    .map(|k| v0[k] + sign * (dp.0[k] + dp.1[k]))
                    .collect()
            };
            let across = |sign: f64| -> Vec<f64> {
                (0..v0.len())
                    .map(|k| v0[k] + sign * dt[k] + 0.5 * sec[k])
                    .collect()
            };
            let shifted = |a: &[DMatrix<f64>; 2], sign: f64| [0, 1].map(|s| &d0[s] + &a[s] * sign);
            let (pp, pm) = (shifted(&p, 1.), shifted(&p, -1.));
            let (tp, tm) = (shifted(&t, 1.), shifted(&t, -1.));
            let plus = ground.fixed([&pp[0], &pp[1]], None, &along(1.))?;
            let minus = ground.fixed([&pm[0], &pm[1]], None, &along(-1.))?;
            let up = ground.fixed([&tp[0], &tp[1]], None, &across(1.))?;
            let down = ground.fixed([&tm[0], &tm[1]], None, &across(-1.))?;
            let middle = ground.fixed([&d0[0], &d0[1]], None, v0)?;
            // The overlap term: the same contraction with and without `W`.
            let weighted = ground.fixed([&d0[0], &d0[1]], Some([&w[0], &w[1]]), v0)?;
            let mut parts: Vec<(&'static str, Vec<f64>)> = vec![
                ("ground", gradient.total.as_slice().to_vec()),
                (
                    "relaxed",
                    (0..size).map(|k| 0.5 * (plus[k] - minus[k])).collect(),
                ),
                (
                    "transition",
                    (0..size)
                        .map(|k| up[k] + down[k] - 2. * middle[k])
                        .collect(),
                ),
                (
                    "overlap",
                    (0..size).map(|k| weighted[k] - middle[k]).collect(),
                ),
            ];
            // The +U share of `d^2 E[i T']`, which the symmetric contractions cannot carry.
            let setup = integrals
                .paw_setup
                .as_ref()
                .ok_or_else(|| invalid("excited-state forces need a PAW setup"))?;
            let raw = [0, 1].map(|s| {
                let c = &ground.channel(s).raw;
                c * &imaginary[s] * c.transpose()
            });
            if let Some(force) = crate::paw::molecular::gradient::hubbard_imaginary_force(
                setup,
                integrals,
                [&raw[0], &raw[1]],
            )? {
                parts.push(("hubbard_imaginary", force));
            }
            // A meta-GGA's share through its current, at fixed `v_tau / n`.
            if let Some(force) = ground.current_force([&imaginary[0], &imaginary[1]])? {
                parts.push(("current", force));
            }
            if let Some(hybrid) = &ground.hybrid {
                let (orbitals, couplings) = ground.coupled_directions(&p, &t, &vs)?;
                let ao = |x: &[DMatrix<f64>; 2]| {
                    [0, 1].map(|s| {
                        let c = &ground.channel(s).raw;
                        c * &x[s] * c.transpose()
                    })
                };
                let (d0, p, t) = (ao(&d0), ao(&p), ao(&t));
                parts.push((
                    "hybrid_spheres",
                    setup.hybrid_sphere_derivative(
                        integrals,
                        hybrid.omega,
                        hybrid.fraction,
                        [&d0[0], &d0[1]],
                        [&p[0], &p[1]],
                        [&t[0], &t[1]],
                    )?,
                ));
                parts.push((
                    "hybrid_exchange",
                    integrals.coupled_exchange_gradient(
                        setup,
                        &orbitals,
                        &couplings,
                        hybrid.omega,
                        hybrid.fraction,
                    )?,
                ));
            }
            finish(gradient, parts)
        })
    }
}

/// Test seams: the pieces of the Lagrangian against which the gates check the engine's own
/// numbers.
#[doc(hidden)]
pub mod seams {
    use super::*;

    /// The gradient of spinor state `state` of `found` (the spinor excitations of `reference`)
    /// without the degeneracy check, and the identities the gates read (`omega` rebuilt, the
    /// Lagrangian's asymmetry, the time-odd share of `P`, the ground contraction's miss).
    pub fn spinor_state(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        found: &crate::molecular::excitations::Excitations,
        state: usize,
    ) -> Result<(
        Vec<f64>,
        Vec<(&'static str, Vec<f64>)>,
        super::super::spinor::gradient::Diagnostics,
    )> {
        let (g, diagnostics) =
            super::super::spinor::gradient::state_gradient(integrals, reference, found, state)?;
        Ok((g.gradient.total.as_slice().to_vec(), g.parts, diagnostics))
    }

    /// The fixed-density gradient of the reference's own densities, energy-weighted density and
    /// XC potentials: it equals the ground-state gradient.
    pub fn ground_fixed(integrals: &MolecularIntegrals, reference: &MeanField) -> Result<Vec<f64>> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals.context.run(|| {
            let ground = Ground::closed(integrals, reference, model)?;
            let (o, m) = (ground.channel(0).o, ground.m);
            let mut d0 = DMatrix::zeros(m, m);
            d0.view_mut((0, 0), (o, o)).fill_with_identity();
            let w = DMatrix::from_fn(m, m, |p, q| {
                if p == q && p < o {
                    ground.channel(0).energies[p]
                } else {
                    0.
                }
            });
            ground.fixed(
                [&d0, &d0],
                Some([&w, &w]),
                ground.semilocal()?.potentials.as_slice(),
            )
        })
    }
    /// The fixed-density contraction at the spin densities `d` and energy-weighted density `w`
    /// (canonical-orbital matrices), with the XC potentials of `d` itself; and the canonical
    /// orbitals in the raw atomic basis.
    pub fn fixed_at(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        d: [&DMatrix<f64>; 2],
        w: Option<&DMatrix<f64>>,
    ) -> Result<(Vec<f64>, DMatrix<f64>)> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals.context.run(|| {
            let ground = Ground::closed(integrals, reference, model)?;
            let frame = d.map(|x| {
                (&ground.channel(0).frame * x * ground.channel(0).frame.transpose())
                    .map(|z| C64::new(z, 0.))
            });
            let name = reference
                .functional
                .as_deref()
                .unwrap_or(ground.semilocal()?.name);
            let (_, fields, _) = dft::response_with_energy(integrals, &frame, name)?;
            let zero = DMatrix::zeros(ground.m, ground.m);
            let meta = ground.semilocal()?.meta.is_some();
            let potentials = packed(&integrals.context, &fields.terms, meta)?;
            Ok((
                ground.fixed(d, w.map(|w| [w, &zero]), potentials.as_slice())?,
                ground.channel(0).raw.clone(),
            ))
        })
    }
    /// The fixed-density contraction at the spin densities `d` (canonical-orbital matrices) with
    /// the given XC potentials `v` (`[spin][point]`), and the ground-state potentials.
    pub fn fixed_with(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        d: [&DMatrix<f64>; 2],
        v: Option<&[f64]>,
    ) -> Result<(Vec<(&'static str, Vec<f64>)>, Vec<f64>)> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals.context.run(|| {
            let ground = Ground::closed(integrals, reference, model)?;
            let v0 = ground.semilocal()?.potentials.as_slice().to_vec();
            let g = ground.fixed_gradient(d, None, v.unwrap_or(&v0))?;
            Ok((
                g.terms
                    .iter()
                    .map(|(n, b)| (*n, b.as_slice().to_vec()))
                    .collect(),
                v0,
            ))
        })
    }
    /// `F1_s[V]` (canonical orbitals) of the symmetric spin changes `v` and the second change
    /// `d^2 v_s / dt^2` along them (empty for Hartree-Fock; a meta-GGA's three parts, packed
    /// `[local 2n | tau 2n | flux 6n]`), with the canonical orbitals in the orthonormal frame.
    pub fn first_and_second(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        v: [&DMatrix<f64>; 2],
    ) -> Result<([DMatrix<f64>; 2], Vec<f64>, DMatrix<f64>)> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals.context.run(|| {
            let ground = Ground::closed(integrals, reference, model)?;
            let (f, _) = ground.fock1(v)?;
            let zero = DMatrix::zeros(ground.m, ground.m);
            let second = match ground.semilocal {
                Some(_) => ground.second(v, [&zero, &zero])?.as_slice().to_vec(),
                None => vec![],
            };
            Ok((f, second, ground.channel(0).frame.clone()))
        })
    }
    /// The +U first-order Fock (canonical orbitals) of the spin changes `v`, real or along
    /// `i v` (`v` antisymmetric), with the canonical orbitals in the orthonormal frame; `None`
    /// without +U.
    pub fn hubbard_fock(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        v: [&DMatrix<f64>; 2],
        imaginary: bool,
    ) -> Result<(Option<[DMatrix<f64>; 2]>, DMatrix<f64>)> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals.context.run(|| {
            let ground = Ground::closed(integrals, reference, model)?;
            Ok((
                ground.hubbard_fock(v, imaginary)?,
                ground.channel(0).frame.clone(),
            ))
        })
    }
    /// A screened hybrid's coupled exchange of `orbitals` (complex coefficients over the raw
    /// atomic basis) with the real symmetric `couplings`: its energy
    /// `-(a/2) sum_kl c_kl (rho_kl | rho_lk)` from the screened pair factors the response uses
    /// (`sum_P w_P |phi_k^+ L^P phi_l|^2`), and its explicit gradient at fixed coefficients
    /// ([`MolecularIntegrals::coupled_exchange_gradient`]).
    pub fn coupled_exchange(
        integrals: &MolecularIntegrals,
        functional: &str,
        orbitals: &[Vec<C64>],
        couplings: &DMatrix<f64>,
    ) -> Result<(f64, Vec<f64>)> {
        integrals.context.run(|| {
            let (Some(screening), Some((omega, fraction))) = (
                integrals.screening(functional)?,
                crate::models::functional(functional)?.screened_exchange,
            ) else {
                return Err(invalid("the coupled exchange needs a screened hybrid"));
            };
            let setup = integrals
                .paw_setup
                .as_ref()
                .ok_or_else(|| invalid("the coupled exchange needs a PAW setup"))?;
            let m = integrals.m();
            let lu = integrals.lowdin.clone().lu();
            let solve = |part: Vec<f64>| {
                lu.solve(&nalgebra::DVector::from_vec(part))
                    .ok_or_else(|| invalid("a singular orthonormalization"))
            };
            let frame = orbitals
                .iter()
                .map(|c| {
                    let re = solve(c.iter().map(|z| z.re).collect())?;
                    let im = solve(c.iter().map(|z| z.im).collect())?;
                    Ok(re.zip_map(&im, C64::new))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut energy = 0.;
            super::super::response::each_factor(&screening.eri, m, |w, l| {
                let l = l.map(|v| C64::new(v, 0.));
                let projected: Vec<_> = frame.iter().map(|x| &l * x).collect();
                for k in 0..frame.len() {
                    for j in 0..frame.len() {
                        let c = couplings[(k, j)];
                        if c != 0. {
                            energy -=
                                0.5 * fraction * c * w * frame[k].dotc(&projected[j]).norm_sqr();
                        }
                    }
                }
            })?;
            let gradient =
                integrals.coupled_exchange_gradient(setup, orbitals, couplings, omega, fraction)?;
            Ok((energy, gradient))
        })
    }
    /// The Fock matrices (orthonormal frame) of `functional` at the densities `d`, which may be
    /// complex, from the builder of the real-time path that reads `Im D` (a meta-GGA's current,
    /// +U's complex occupations).
    pub fn complex_fock(
        integrals: &MolecularIntegrals,
        functional: &str,
        d: &[DMatrix<C64>],
    ) -> Result<Vec<DMatrix<C64>>> {
        integrals.context.run(|| {
            let ctx = &integrals.context;
            let n = integrals.grid.index.points();
            let mut poisson = Poisson::isolated(ctx, &integrals.grid)?;
            let gradients = if crate::models::functional(functional)?.requires_tau {
                Some(dft::orbital_gradients(integrals)?)
            } else {
                None
            };
            let screening = integrals.screening(functional)?;
            let (mut total, mut hartree) = (ctx.allocate::<C64>(n)?, ctx.allocate::<C64>(n)?);
            let (fock, _, _) = dft::fock_and_energy(
                integrals,
                &integrals.kohn_sham_core(),
                d,
                functional,
                &mut poisson,
                gradients.as_ref(),
                screening.as_ref(),
                (total.as_mut_slice(), hartree.as_mut_slice()),
                None,
            )?;
            Ok(fock)
        })
    }
    /// `Im F1_s[i V]` (canonical orbitals) of the antisymmetric spin changes `v`: the
    /// antisymmetric channel of the kernel, `None` for a density-only one.
    pub fn imaginary_fock(
        integrals: &MolecularIntegrals,
        reference: &MeanField,
        v: [&DMatrix<f64>; 2],
    ) -> Result<Option<[DMatrix<f64>; 2]>> {
        let model = check(integrals, reference, &Request::new(1))?;
        integrals
            .context
            .run(|| Ground::closed(integrals, reference, model)?.imaginary_fock(v))
    }
}
