//! The Bethe-Salpeter equation of an insulating crystal on its Kohn-Sham states, in the
//! Tamm-Dancoff approximation (`TDDFT_PLAN.md`, section 4.2, stage T5b): bound excitons and the
//! excitonic `epsilon(omega)` at `q -> 0`.
//!
//! The transitions `t = (k, v, c)` of a window of valence and conduction bands on the whole
//! Gamma-anchored mesh (closed under `-k`, `N_k` points, cell volume `Omega`) span
//!
//! ```text
//! H_{t t'} = D_t delta_{t t'} + K^x_{t t'} - K^d_{t t'},     D_t = e_ck - e_vk + scissor,
//! K^x_{t t'} = (g / (N_k Omega)) sum_{0 < |G| <= cutoff} rho_t(G)^* v(G) rho_t'(G)   (singlets),
//! K^d_{t t'} = (1 / (N_k Omega)) sum_{G G'} M_vv'(p + G')^* W_{G'G}(p) M_cc'(p + G),
//! ```
//!
//! with `g = 2` the band degeneracy of a closed shell, `rho_t(G) = <vk| e^{-iG.r} |ck>` the
//! exchange columns of the dielectric function ([`super::dielectric::Pairs`]), `p = k' - k` the
//! mesh vector of the pair, `M_ab(Q) = <a k| e^{-iQ.r} |b k'>` the pair densities of the
//! screening code with the complete PAW augmentation, and `W(p)` the statically screened
//! interaction of the whole mesh in the full RPA ([`crate::periodic::screening::interaction`]).
//! Triplets drop `K^x`. With the direct term written this way `H` is Hermitian by the symmetry
//! of `W` alone; the assembled matrix's anti-Hermitian residual is reported.
//!
//! **Spinor crystals** (T8, with or without spin-orbit coupling): the transitions are spinor
//! pairs, `g = 1`, every pair density is the charge density `psi_a^dagger psi_b`, summed over the
//! two components, and the dipoles carry the spin-orbit velocity ([`super::momentum`]). There is
//! no triplet apart: `spin='singlet'` is the whole operator, and on a Kramers-unpolarized crystal
//! without the term its roots are the collinear singlets once and the triplets three times.
//!
//! **The `q = 0` head.** The `G = G' = 0` element of `W(0)` is the mean of
//! `4 pi / (q . eps_M . q)` over the mesh's mini-zone ([`head_average`]), with `eps_M` the static
//! macroscopic tensor of the same states with local fields (T5a, every band, the screening
//! cutoff); the wings of `W(0)` are dropped and its body is the inverse of the body alone. At
//! the other mesh vectors (and on every diagonal `G`) `W_GG(p) = eps^-1_GG(p) <4 pi / |q + G|^2>`,
//! the mean over the mini-zone cell around `p + G` ([`interaction::cell_average`]): the mesh sum
//! integrates the Coulomb singularity cell by cell (gate E3).
//!
//! **The spectrum.** With `h^a_t = P^a_vc / (e_c - e_v)` the k . p dipoles of T5a and
//! `o^a_l = sum_t h^a_t A^l_t` for each exciton `(E_l, A^l)`,
//!
//! ```text
//! eps^{ab}(z) = delta_ab - 4 pi s sum_l [ o^a_l o^b_l^* / (z - E_l) - o^a_l^* o^b_l / (z + E_l) ],
//! ```
//!
//! `s = g / (N_k Omega)`: with `K^d = 0` it is T5a's local-field tensor in the Tamm-Dancoff
//! approximation, and with `K^x = 0` too its independent-particle tensor.
//!
//! **The coupling form** (`Request::coupling`, `approximation="casida"`): the resonant
//! transitions `(v c)` couple to the antiresonant ones `(c v)` of the same mesh point through
//!
//! ```text
//! B_{vck, v'c'k'} = s sum_G rho_vc(G)^* v(G) rho_c'v'(G)
//!                 - (1 / (N_k Omega)) sum_{G G'} M_{vk, c'k'}(G')^* W_{G'G}(p) M_{ck, v'k'}(G),
//! A'_{tt'}        = D_t delta + s sum_G rho_cv(G)^* v(G) rho_c'v'(G) - conj(K^d_{tt'})
//! ```
//!
//! (the exchange from both orders of T5a's columns, the direct term from the screening code's
//! pair densities across `k' = k + p`). `M = [[A, B], [B^dagger, A']]` is Hermitian and the
//! operator is `diag(1, -1) M`; with `M` positive definite (`M = L L^dagger`) its eigenvalues are
//! those of the Hermitian `L^dagger diag(1, -1) L`, `n` positive and `n` negative, with
//! eigenvectors `Z = diag(1, -1) L u / sqrt|E|` (`Z^dagger diag(1, -1) Z = sign E`). Every
//! eigenpair enters the tensor, `chi^{ab}(z) = s sum_l sign(E_l) o^a_l o^b_l^* / (z - E_l)` with
//! `o_l = sum_t (h_t Z_tl + h~_t Z_{n+t,l})` and `h~` the head rows of `rho_cv`. Dense only, up
//! to [`COUPLING_LIMIT`] transitions. With the exchange alone it is T5a's Dyson equation (gate
//! E1, against [`coupled_tensor`], the Woodbury form of the same equation). Below [`DENSE_LIMIT`]
//! transitions `H` is diagonalized whole; above it the lowest excitons come from Davidson on the
//! real form `[[Re H, -Im H], [Im H, Re H]]` (each root twice, then a complex Rayleigh-Ritz step),
//! and the spectrum from the Lanczos recursion of [`crate::math::lanczos`] from each dipole
//! vector (Haydock), the off-diagonal elements by polarization (the tensor of a crystal with time
//! reversal is symmetric).
//!
//! **Spin-polarized crystals**: the transitions are the spin-conserving ones of both channels
//! (`g = 1`), ordered mesh point, channel, valence, conduction. The exchange `K^x` couples every
//! pair of them (the Hartree kernel of the charge), the direct term keeps the channel, and `W`
//! screens with the transitions of both. There is no triplet apart (`spin='singlet'` is the whole
//! operator); on an unpolarized crystal solved spin-polarized its roots are the singlets and the
//! triplets, each once. Each channel takes its own default window (every band).
//!
//! Refused here besides T5a's refusals: `spin='triplet'` on a spinor or spin-polarized crystal, a
//! window wider than the memory budget for the explicit `H`.
use super::{
    super::{
        bloch_mesh::{MeshIndex, Sampler},
        invalid,
        response::setup::Setup,
        screening::{
            bloch::Bloch,
            interaction::{self, Block},
            medium::{Medium, Paw},
            transitions::{components_of, Augmentation, Engine, Remainders, Scratch},
        },
        spectrum::Smearing,
        wannier::sampling::Frame,
    },
    dielectric::{columns, largest_window, Dielectric, Kernel, Pairs, Request as Optical},
};
use crate::{
    math::{
        davidson::{self, Symmetric},
        gemm::complex_product,
        lanczos,
        linalg::eigen,
        storage::ComplexRead,
    },
    product,
    quantum::C64,
    Error, Result,
};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;

/// The largest number of transitions whose `H` is diagonalized whole.
pub const DENSE_LIMIT: usize = 3000;
/// The largest number of transitions of the coupling form (dense, `2n x 2n`).
pub const COUPLING_LIMIT: usize = 1500;
/// Lanczos vectors of one Haydock spectrum when none are given.
pub const KRYLOV: usize = 300;
/// Gauss-Legendre points per direction of each cell of the head average.
const HEAD_ORDER: usize = 8;

/// The spin of a closed shell's excitons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    /// Exchange and direct terms.
    Singlet,
    /// The direct term alone.
    Triplet,
}

impl Spin {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "singlet" => Ok(Self::Singlet),
            "triplet" => Ok(Self::Triplet),
            other => Err(invalid(format!(
                "unknown spin {other:?}; expected 'singlet' or 'triplet'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Singlet => "singlet",
            Self::Triplet => "triplet",
        }
    }
}

/// What the Bethe-Salpeter operator is built from, in atomic units.
#[derive(Clone, Debug)]
pub struct Request {
    /// `Bse` (exchange and direct), `Hartree` (exchange only: T5a's local fields in the
    /// Tamm-Dancoff approximation) or `Independent` (neither).
    pub kernel: Kernel,
    pub spin: Spin,
    /// `(valence, conduction)` bands next to the gap; `None`: every band.
    pub window: Option<(usize, usize)>,
    /// Added to every transition energy (Hartree).
    pub scissor: f64,
    /// The largest `|q + G|` of the exchange and the screened interaction (Bohr^-1).
    pub cutoff: f64,
    /// `false`: the Tamm-Dancoff operator; `true`: the coupling form (module documentation).
    pub coupling: bool,
}

impl Request {
    pub fn new(kernel: Kernel, spin: Spin) -> Self {
        Self {
            kernel,
            spin,
            window: None,
            scissor: 0.,
            cutoff: super::dielectric::LOCAL_FIELD_CUTOFF,
            coupling: false,
        }
    }
}

/// One transition of the operator: mesh point, valence and conduction band (indices into the
/// crystal's bands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub k: usize,
    /// The spin channel (0 unless the crystal is spin-polarized).
    pub channel: usize,
    pub valence: usize,
    pub conduction: usize,
}

/// The Bethe-Salpeter operator of a crystal (module documentation).
pub struct Operator {
    /// `H` (Hartree), Hermitized.
    pub hamiltonian: DMatrix<C64>,
    /// `D_t` (Hartree, scissor included).
    pub energies: Vec<f64>,
    pub transitions: Vec<Transition>,
    /// `h^a_t = P^a_vc / (e_c - e_v)` (Bohr).
    pub dipoles: Vec<[C64; 3]>,
    /// `s = g / (N_k Omega)`.
    pub scale: f64,
    pub kernel: Kernel,
    pub spin: Spin,
    /// The `G = G' = 0` element of `W(0)` (Hartree Bohr^3); zero without the direct term.
    pub head: f64,
    /// `eps_M(0)` that screens the head; zero without the direct term.
    pub static_tensor: [[f64; 3]; 3],
    /// `max |H - H^dagger| / max |H|` before the Hermitization.
    pub hermiticity: f64,
    /// The exchange vectors (`G != 0`).
    pub vectors: usize,
    pub cutoff: f64,
    pub scissor: f64,
    pub mesh: [usize; 3],
    pub window: (usize, usize),
    /// The smallest `D_t`: the onset of the interband continuum on this mesh.
    pub onset: f64,
    pub electrons: f64,
    pub volume: f64,
    /// The largest anti-Hermitian residual of `P` over the mesh.
    pub residual: f64,
    /// The blocks of the coupling form, when requested.
    pub coupling: Option<Coupling>,
    _memory: crate::Reservation,
}

/// The blocks of the coupling form beside `hamiltonian` (`A`; module documentation).
pub struct Coupling {
    /// `B`, resonant rows by antiresonant columns.
    pub resonant: DMatrix<C64>,
    /// `A'`, Hermitized.
    pub antiresonant: DMatrix<C64>,
    /// `h~^a_t`, the head rows of `rho_cv` (Bohr).
    pub dipoles: Vec<[C64; 3]>,
    /// `max |B - B^T| / max |B|` (zero by time reversal and the symmetry of `W`).
    pub symmetry: f64,
}

/// Every eigenpair of the coupling form: `E` ascending (`n` negative, then `n` positive) and the
/// columns `Z` (`2n x 2n`, `Z^dagger diag(1, -1) Z = sign E`).
#[derive(Clone, Debug)]
pub struct Branches {
    pub energies: Vec<f64>,
    pub vectors: DMatrix<C64>,
}

/// The lowest excitons of an [`Operator`].
#[derive(Clone, Debug)]
pub struct Excitons {
    /// `E_l` (Hartree), ascending.
    pub energies: Vec<f64>,
    /// `A^l` as columns, each normalized.
    pub vectors: DMatrix<C64>,
    /// The residual norm of each root (zero for a dense solve).
    pub residuals: Vec<f64>,
    pub converged: bool,
    /// `"dense"`, `"davidson"` or `"coupling"`.
    pub solver: &'static str,
    /// The antiresonant amplitudes `Y` of the coupling form (columns beside `vectors`, the `X`,
    /// with `X^dagger X - Y^dagger Y = 1`); `None` in the Tamm-Dancoff approximation.
    pub antiresonant: Option<DMatrix<C64>>,
    /// Every eigenpair of the coupling form, which its tensor sums.
    pub branches: Option<Branches>,
}

/// Summary observables extracted from a converged [`Excitons`] solution.
pub struct ExcitonSummary {
    pub binding: Vec<f64>,
    pub areas: Vec<f64>,
    pub participation: Vec<f64>,
    pub kpoint_weights: Vec<f64>,
    pub dominant: Vec<(usize, usize, usize, f64)>,
    pub dominant_channels: Vec<usize>,
}

fn check(sampler: &Sampler<'_>, request: &Request) -> Result<()> {
    if sampler.channels() == 2 && request.spin == Spin::Triplet {
        return Err(invalid(
            "a spin-polarized crystal's excitons are the spin-conserving ones of both channels, \
             coupled by the exchange: none is a triplet apart, and its operator is spin='singlet'",
        ));
    }
    if sampler.crystal.spinor && request.spin == Spin::Triplet {
        return Err(invalid(
            "a spinor crystal's excitons mix the spins, so none is a triplet apart: its operator \
             (exchange and direct terms on every spinor transition) is spin='singlet'",
        ));
    }
    if request.kernel == Kernel::LongRange {
        return Err(invalid(
            "the long-range kernel is a model of the dielectric function, not an exciton kernel: \
             crystal excitons take kernel='bse', 'hartree' or 'independent'",
        ));
    }
    if !request.scissor.is_finite() {
        return Err(invalid("the scissor must be finite"));
    }
    if !(request.cutoff.is_finite() && request.cutoff > 0.) {
        return Err(invalid("the cutoff must be finite and positive"));
    }
    Ok(())
}

/// The Fermi level and width that make the static weights of an insulator's frames integral:
/// the middle of the gap over every channel (`frames[s]`, `occupied[s]` bands) and a 80th of it.
fn midgap(frames: &[Vec<Frame>], occupied: &[usize]) -> Result<(f64, f64)> {
    let edge =
        |s: usize, f: &Frame, above: bool| f.energies.as_slice()[occupied[s] - usize::from(!above)];
    let top = (0..frames.len())
        .flat_map(|s| frames[s].iter().map(move |f| (s, f)))
        .map(|(s, f)| edge(s, f, false))
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = (0..frames.len())
        .flat_map(|s| frames[s].iter().map(move |f| (s, f)))
        .map(|(s, f)| edge(s, f, true))
        .fold(f64::INFINITY, f64::min);
    if !(bottom > top) {
        return Err(invalid(
            "the Bethe-Salpeter equation needs an insulator: the conduction bands overlap the \
             valence bands on this mesh",
        ));
    }
    Ok((0.5 * (top + bottom), (bottom - top) / 80.))
}

/// The Bethe-Salpeter operator of the insulator `sampler` holds on `mesh` for `request`.
pub fn operator(sampler: &Sampler<'_>, mesh: &MeshIndex, request: &Request) -> Result<Operator> {
    check(sampler, request)?;
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let bands = crystal.bands();
    let all: Vec<usize> = (0..bands).collect();
    let frames = sampler.frames(mesh, &vec![all; sampler.channels()], |_, _| Ok(()))?;
    let mut optical = Optical::new(vec![0.], 0., request.kernel);
    if request.kernel == Kernel::Bse {
        optical.kernel = Kernel::Hartree;
    }
    optical.window = request.window;
    optical.scissor = request.scissor;
    optical.cutoff = request.cutoff;
    let pairs = columns(sampler, mesh, &optical, Some(&frames))?;
    let (valence, conduction) = pairs.window[0];
    let windows = pairs.window.clone();
    let n = pairs.energies.len();
    let nk = mesh.points();
    if request.coupling && n > COUPLING_LIMIT {
        return Err(Error::Unsupported(format!(
            "the coupling form of {n} transitions ({valence} valence x {conduction} conduction \
             bands x {nk} k-points) is above its dense limit of {COUPLING_LIMIT}: pass a smaller \
             bands=(valence, conduction) or approximation='tda'"
        )));
    }
    let bytes = |n: usize| -> Result<usize> {
        let tda = product(&[n, n, 16, 3])?;
        if request.coupling {
            tda.checked_add(product(&[2 * n, 2 * n, 16, 6])?)
                .ok_or(Error::Overflow)
        } else {
            Ok(tda)
        }
    };
    let memory = ctx.reserve(bytes(n)?).map_err(|_| {
        let per_point: Vec<usize> = pairs.window.iter().map(|w| w.0 * w.1).collect();
        let largest = largest_window(valence, conduction, |v, c| {
            let count = per_point.iter().map(|_| v * c).sum::<usize>() * nk;
            bytes(count)
                .ok()
                .and_then(|b| ctx.reserve(b).ok())
                .is_some()
        });
        Error::Unsupported(format!(
            "the Bethe-Salpeter operator of {n} transitions ({valence} valence x {conduction} \
             conduction bands x {nk} k-points) does not fit the memory budget: {}",
            match largest {
                Some((v, c)) => format!("the largest window that fits is bands=({v}, {c})"),
                None => "not even bands=(1, 1) fits".into(),
            }
        ))
    })?;
    // Each channel's occupied bands: its window ends at the last of them. The transitions are in
    // the order of the columns: mesh point, channel, valence, conduction.
    let occupied = occupied_bands(sampler)?;
    let per_point: usize = windows.iter().map(|(v, c)| v * c).sum();
    let offset = |k: usize, s: usize| -> usize {
        k * per_point + windows[..s].iter().map(|(v, c)| v * c).sum::<usize>()
    };
    let mut transitions: Vec<Transition> = Vec::with_capacity(n);
    for k in 0..nk {
        for (s, &(nv, nc)) in windows.iter().enumerate() {
            for v in 0..nv {
                for c in 0..nc {
                    transitions.push(Transition {
                        k,
                        channel: s,
                        valence: occupied[s] - nv + v,
                        conduction: occupied[s] + c,
                    });
                }
            }
        }
    }
    let dipoles: Vec<[C64; 3]> = (0..n)
        .map(|t| {
            let column = pairs.column(t, false);
            [column[0], column[1], column[2]]
        })
        .collect();
    let mut hamiltonian = DMatrix::<C64>::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        pairs.energies.iter().map(|e| C64::new(*e, 0.)),
    ));
    let exchanged = request.kernel != Kernel::Independent && request.spin == Spin::Singlet;
    // The coupling blocks before the direct term: `B` and `A'` of the exchange and +U.
    let mut coupled = if request.coupling {
        let zero = || DMatrix::<C64>::zeros(n, n);
        let (mut resonant, mut antiresonant) = if exchanged {
            (
                exchange(&pairs, false, true)?,
                exchange(&pairs, true, true)? - exchange(&pairs, false, false)?.map(|v| v.conj()),
            )
        } else {
            (zero(), zero())
        };
        if request.kernel != Kernel::Independent {
            if let Some(hu_rb) = hubbard_block(&pairs, false, true, request.spin)? {
                resonant += hu_rb;
            }
            if let (Some(hu_bb), Some(hu_ff)) = (
                hubbard_block(&pairs, true, true, request.spin)?,
                hubbard_block(&pairs, false, false, request.spin)?,
            ) {
                antiresonant += hu_bb - hu_ff.map(|v| v.conj());
            }
        }
        Some((resonant, antiresonant))
    } else {
        None
    };
    if exchanged {
        hamiltonian += exchange(&pairs, false, false)?;
    }
    if request.kernel != Kernel::Independent {
        if let Some(hu) = hubbard_block(&pairs, false, false, request.spin)? {
            hamiltonian += hu;
        }
    }
    let mut head = 0.;
    let mut static_tensor = [[0.; 3]; 3];
    if request.kernel == Kernel::Bse {
        let setup = Setup::new(crystal)?;
        let remainders = Remainders::new(ctx, &setup)?;
        let blocks;
        (blocks, head, static_tensor) = screening(
            sampler,
            mesh,
            &frames,
            &occupied,
            &setup,
            &remainders,
            request.cutoff,
        )?;
        let scale = 1. / (nk as f64 * pairs.volume);
        // The direct term keeps the spin: one block of each channel's transitions per pair.
        for (s, &(nv, nc)) in windows.iter().enumerate() {
            let window = Window {
                lowest: occupied[s] - nv,
                valence: nv,
                occupied: occupied[s],
                conduction: nc,
            };
            let direct = direct(
                sampler,
                &frames[s],
                mesh,
                &blocks,
                &window,
                &paw_parts(&setup, &remainders),
                request.coupling,
            )?;
            for (k, row_block) in direct.into_iter().enumerate() {
                for (kp, block, across) in row_block {
                    let (r0, c0) = (offset(k, s), offset(kp, s));
                    for i in 0..nv * nc {
                        for j in 0..nv * nc {
                            hamiltonian[(r0 + i, c0 + j)] -= block[(i, j)] * scale;
                        }
                    }
                    if let (Some((resonant, _)), Some(across)) = (coupled.as_mut(), across) {
                        for i in 0..nv * nc {
                            for j in 0..nv * nc {
                                resonant[(r0 + i, c0 + j)] -= across[(i, j)] * scale;
                            }
                        }
                    }
                }
            }
        }
    }
    let largest = hamiltonian.iter().map(|v| v.norm()).fold(0f64, f64::max);
    let hermiticity = (&hamiltonian - hamiltonian.adjoint())
        .iter()
        .map(|v| v.norm())
        .fold(0f64, f64::max)
        / largest.max(f64::MIN_POSITIVE);
    // `A' = conj(A) + K^x_cv - conj(K^x_vc)`: the direct term's `conj(K^d)` with the exchange of
    // the antiresonant order (the unhermitized `A`, so `A'` reads the same direct blocks).
    let coupling = coupled
        .map(|(resonant, correction)| -> Result<Coupling> {
            let antiresonant = hamiltonian.map(|v| v.conj()) + correction;
            let largest = resonant.iter().map(|v| v.norm()).fold(0f64, f64::max);
            let symmetry = (&resonant - resonant.transpose())
                .iter()
                .map(|v| v.norm())
                .fold(0f64, f64::max)
                / largest.max(f64::MIN_POSITIVE);
            Ok(Coupling {
                antiresonant: (&antiresonant + antiresonant.adjoint()) * C64::new(0.5, 0.),
                resonant,
                dipoles: (0..n)
                    .map(|t| {
                        let column = pairs.column(t, true);
                        [column[0], column[1], column[2]]
                    })
                    .collect(),
                symmetry,
            })
        })
        .transpose()?;
    let hamiltonian = (&hamiltonian + hamiltonian.adjoint()) * C64::new(0.5, 0.);
    let finite = |m: &DMatrix<C64>| m.iter().all(|v| v.re.is_finite() && v.im.is_finite());
    if !finite(&hamiltonian)
        || coupling
            .as_ref()
            .is_some_and(|c| !(finite(&c.resonant) && finite(&c.antiresonant)))
    {
        return Err(invalid("the Bethe-Salpeter operator is not finite"));
    }
    Ok(Operator {
        hamiltonian,
        energies: pairs.energies.clone(),
        transitions,
        dipoles,
        scale: pairs.scale(),
        kernel: request.kernel,
        spin: request.spin,
        head,
        static_tensor,
        hermiticity,
        vectors: pairs.vectors.len(),
        cutoff: request.cutoff,
        scissor: request.scissor,
        mesh: pairs.mesh,
        window: (valence, conduction),
        onset: pairs.onset,
        electrons: pairs.electrons,
        volume: pairs.volume,
        residual: pairs.residual,
        coupling,
        _memory: memory,
    })
}

/// The statically screened interaction of the direct term (module documentation): `W(p)` of
/// every mesh vector, the `q = 0` head (returned beside it) in place, and the static
/// macroscopic tensor with local fields of every band that screens the head.
fn screening(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    frames: &[Vec<Frame>],
    occupied: &[usize],
    setup: &Setup,
    remainders: &Remainders,
    cutoff: f64,
) -> Result<(Vec<Block>, f64, [[f64; 3]; 3])> {
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    // eps_M(0) of every band with local fields: the screening of the head.
    let mut optical = Optical::new(vec![0.], 0., Kernel::Hartree);
    optical.cutoff = cutoff;
    let macroscopic = columns(sampler, mesh, &optical, Some(frames))?;
    let static_tensor = macroscopic
        .tensor(C64::new(0., 0.))?
        .map(|row| row.map(|v| v.re));
    let eps = Matrix3::from_fn(|a, b| static_tensor[a][b]);
    let head = interaction::head_average(&sampler.reciprocal, mesh, &eps, HEAD_ORDER)?;
    let (fermi, width) = midgap(frames, occupied)?;
    let medium = Medium {
        ctx,
        grid: &crystal.grid,
        volume: macroscopic.volume,
        fermi,
        width,
        degeneracy: sampler.degeneracy(),
        smearing: Smearing::FermiDirac,
        constrained: false,
        paw: Some(Paw {
            crystal,
            setup,
            remainders,
        }),
        workers: None,
    };
    let points = crystal.grid.index.points();
    let states: Vec<Vec<Bloch<'_>>> = frames
        .iter()
        .map(|channel| {
            channel
                .iter()
                .map(|frame| Bloch {
                    fractional: frame.fractional,
                    energies: frame.energies.as_slice(),
                    target: vec![0.; frame.energies.as_slice().len()],
                    samples: &frame.samples,
                    projections: Some(&frame.projections),
                    points,
                    components: frame.components,
                })
                .collect()
        })
        .collect();
    let mut blocks = interaction::screened(
        &medium,
        &states,
        mesh,
        &sampler.lattice,
        &sampler.reciprocal,
        cutoff,
    )?;
    let origin = blocks[0]
        .g
        .iter()
        .position(|g| *g == [0, 0, 0])
        .ok_or_else(|| invalid("the screened vectors of q = 0 lack G = 0"))?;
    blocks[0].screened[(origin, origin)] = C64::new(head, 0.);
    Ok((blocks, head, static_tensor))
}

/// Each channel's occupied bands.
fn occupied_bands(sampler: &Sampler<'_>) -> Result<Vec<usize>> {
    Ok(sampler
        .manifold(
            super::super::berry::Bands::Occupied,
            super::super::berry::Metal::Undefined,
        )?
        .ranges
        .iter()
        .map(|r| r.1)
        .collect())
}

/// The pieces of the direct term, for an oracle that reads none of the operator's bookkeeping:
/// `W(p)` of every mesh vector (the `q = 0` head in place) and the pair densities of every pair
/// of bands across `k' = k + p`.
pub struct DirectParts {
    /// The screened interaction, one block per mesh vector `p` (`Block::steps`).
    pub blocks: Vec<Block>,
    /// `densities[s][t]`, channel `s` and block `t`: for each mesh point `k`, `(k, k', M)` with
    /// `M` the vectors of the block by `a * bands + b`, `M_{ak, bk'}(p + G)` with the complete
    /// augmentation (a spinor's summed over its components).
    pub densities: Vec<Vec<Vec<(usize, usize, DMatrix<C64>)>>>,
    /// The bands of every frame (every band of the crystal).
    pub bands: usize,
    /// Each channel's occupied bands.
    pub occupied: Vec<usize>,
}

/// The [`DirectParts`] of the insulator `sampler` on `mesh` with the screening `cutoff`, the
/// `W` and pair densities [`operator`] builds its direct term from. Every band pair of every
/// mesh pair is kept: a check for small cells, not a route to results.
pub fn direct_parts(sampler: &Sampler<'_>, mesh: &MeshIndex, cutoff: f64) -> Result<DirectParts> {
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let bands = crystal.bands();
    let all: Vec<usize> = (0..bands).collect();
    let frames = sampler.frames(mesh, &vec![all; sampler.channels()], |_, _| Ok(()))?;
    let occupied = occupied_bands(sampler)?;
    let setup = Setup::new(crystal)?;
    let remainders = Remainders::new(ctx, &setup)?;
    let (blocks, _, _) = screening(
        sampler,
        mesh,
        &frames,
        &occupied,
        &setup,
        &remainders,
        cutoff,
    )?;
    let mut densities = Vec::with_capacity(frames.len());
    for channel in &frames {
        let mut per_block = Vec::with_capacity(blocks.len());
        for block in &blocks {
            let engine = Engine::new(ctx, &crystal.grid, block.q, &block.g)?;
            let augmentation = Augmentation::new(crystal, &setup, &remainders, &engine.vectors)?;
            let mut scratch = engine.scratch(ctx)?;
            let mut row = vec![C64::default(); block.g.len()];
            let mut per_k = Vec::with_capacity(mesh.points());
            for k in 0..mesh.points() {
                let kp = mesh.shift(k, block.steps);
                let (left, right) = (&channel[k], &channel[kp]);
                let (l, r) = (values(ctx, left)?, values(ctx, right)?);
                let mut m = DMatrix::<C64>::zeros(block.g.len(), bands * bands);
                for a in 0..bands {
                    for b in 0..bands {
                        let (left, right) = ((left, &l[..]), (right, &r[..]));
                        pair_density(
                            &engine,
                            &augmentation,
                            &mut scratch,
                            left,
                            right,
                            a,
                            b,
                            &mut row,
                        )?;
                        m.column_mut(a * bands + b).copy_from_slice(&row);
                    }
                }
                per_k.push((k, kp, m));
            }
            per_block.push(per_k);
        }
        densities.push(per_block);
    }
    Ok(DirectParts {
        blocks,
        densities,
        bands,
        occupied,
    })
}

/// `K^x` between the orders `backward` of the rows and of the columns of every transition of
/// `pairs`: `s (V R_rows)^dagger (V R_columns)`, `R` the local-field rows of each `rho_vc`
/// (`false`) or `rho_cv` (`true`) column, `V` the square root of `v(G)`.
fn exchange(pairs: &Pairs, rows: bool, columns: bool) -> Result<DMatrix<C64>> {
    let coulomb = pairs.coulomb();
    let nb = coulomb.len();
    let n = pairs.energies.len();
    let weighted = |backward: bool| {
        DMatrix::from_fn(nb, n, |g, i| pairs.column(i, backward)[3 + g] * coulomb[g])
    };
    let (left, right) = (weighted(rows), weighted(columns));
    let product = complex_product(left.as_view(), true, right.as_view())?;
    Ok(product * C64::new(pairs.scale(), 0.))
}

fn hubbard_block(
    pairs: &Pairs,
    rows: bool,
    columns: bool,
    spin: Spin,
) -> Result<Option<DMatrix<C64>>> {
    let Some(hk) = pairs.hubbard() else {
        return Ok(None);
    };
    let kernel_mat = match spin {
        Spin::Triplet => hk.triplet.as_ref().ok_or_else(|| {
            invalid("Hubbard triplet optical kernel requires a restricted reference")
        })?,
        Spin::Singlet => &hk.singlet,
    };
    let nb = pairs.coulomb().len();
    let mu = hk.size;
    let n = pairs.energies.len();
    let extract = |backward: bool| {
        DMatrix::from_fn(mu, n, |u, i| pairs.column(i, backward)[3 + nb + u])
    };
    let (left, right) = (extract(rows), extract(columns));
    let mid = complex_product(kernel_mat.as_view(), false, right.as_view())?;
    let prod = complex_product(left.as_view(), true, mid.as_view())?;
    let scale = pairs.scale() * pairs.volume / pairs.degeneracy();
    Ok(Some(prod * C64::new(scale, 0.)))
}

/// The band window of the direct term (indices into a frame's bands).
struct Window {
    lowest: usize,
    valence: usize,
    occupied: usize,
    conduction: usize,
}

/// What the augmentation of the direct term's pair densities reads.
struct PawParts<'a> {
    setup: &'a Setup,
    remainders: &'a Remainders,
}

fn paw_parts<'a>(setup: &'a Setup, remainders: &'a Remainders) -> PawParts<'a> {
    PawParts { setup, remainders }
}

/// The unscaled direct blocks `sum M_v^* W M_c` (rows `(v, c)`, columns `(v', c')`), for every
/// pair `(k, k')` grouped by `k`: one task per mesh vector `p`, every `k` with `k' = k + p`.
/// With `coupling`, beside each its block of `B`: `sum M_{vc'}^* W M_{cv'}` on the same rows and
/// columns.
#[allow(clippy::type_complexity)]
fn direct(
    sampler: &Sampler<'_>,
    frames: &[Frame],
    mesh: &MeshIndex,
    blocks: &[Block],
    window: &Window,
    paw: &PawParts<'_>,
    coupling: bool,
) -> Result<Vec<Vec<(usize, DMatrix<C64>, Option<DMatrix<C64>>)>>> {
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let nk = mesh.points();
    let (nv, nc) = (window.valence, window.conduction);
    #[allow(clippy::type_complexity)]
    let one = |t: usize| -> Result<Vec<(usize, usize, DMatrix<C64>, Option<DMatrix<C64>>)>> {
        let block = &blocks[t];
        let engine = Engine::new(ctx, &crystal.grid, block.q, &block.g)?;
        let augmentation = Augmentation::new(crystal, paw.setup, paw.remainders, &engine.vectors)?;
        let mut scratch = engine.scratch(ctx)?;
        let ng = block.g.len();
        let mut out = Vec::with_capacity(nk);
        for k in 0..nk {
            let kp = mesh.shift(k, block.steps);
            let (left, right) = (&frames[k], &frames[kp]);
            let (l, r) = (values(ctx, left)?, values(ctx, right)?);
            let density = |scratch: &mut _, a: usize, b: usize, out: &mut [C64]| -> Result<()> {
                pair_density(
                    &engine,
                    &augmentation,
                    scratch,
                    (left, &l[..]),
                    (right, &r[..]),
                    a,
                    b,
                    out,
                )
            };
            // Columns (c c') and (v v'), each the pair density over the vectors.
            let mut conduction = DMatrix::<C64>::zeros(ng, nc * nc);
            let mut valence = DMatrix::<C64>::zeros(ng, nv * nv);
            let mut row = vec![C64::default(); ng];
            for c in 0..nc {
                for cp in 0..nc {
                    density(
                        &mut scratch,
                        window.occupied + c,
                        window.occupied + cp,
                        &mut row,
                    )?;
                    conduction.column_mut(c * nc + cp).copy_from_slice(&row);
                }
            }
            for v in 0..nv {
                for vp in 0..nv {
                    density(
                        &mut scratch,
                        window.lowest + v,
                        window.lowest + vp,
                        &mut row,
                    )?;
                    valence.column_mut(v * nv + vp).copy_from_slice(&row);
                }
            }
            // Columns (v c') and (c v') of the coupling block: valence at k with conduction at
            // k', and conduction at k with valence at k'.
            let across = if coupling {
                let mut forward = DMatrix::<C64>::zeros(ng, nv * nc);
                let mut backward = DMatrix::<C64>::zeros(ng, nc * nv);
                for v in 0..nv {
                    for c in 0..nc {
                        density(
                            &mut scratch,
                            window.lowest + v,
                            window.occupied + c,
                            &mut row,
                        )?;
                        forward.column_mut(v * nc + c).copy_from_slice(&row);
                        density(
                            &mut scratch,
                            window.occupied + c,
                            window.lowest + v,
                            &mut row,
                        )?;
                        backward.column_mut(c * nv + v).copy_from_slice(&row);
                    }
                }
                let pairs = forward.adjoint() * (&block.screened * &backward);
                Some(DMatrix::from_fn(nv * nc, nv * nc, |i, j| {
                    let (v, c) = (i / nc, i % nc);
                    let (vp, cp) = (j / nc, j % nc);
                    pairs[(v * nc + cp, c * nv + vp)]
                }))
            } else {
                None
            };
            out.push((
                k,
                kp,
                contract(&valence, &block.screened, &conduction, nv, nc)?,
                across,
            ));
        }
        Ok(out)
    };
    let solved = ctx.map_independent(blocks.len(), one)?;
    let mut grouped: Vec<Vec<(usize, DMatrix<C64>, Option<DMatrix<C64>>)>> =
        (0..nk).map(|_| vec![]).collect();
    for list in solved {
        for (k, kp, kernel, across) in list {
            grouped[k].push((kp, kernel, across));
        }
    }
    Ok(grouped)
}

/// Every grid value of `frame` (`[component][band][point]`).
fn values<'a>(ctx: &crate::Context, frame: &'a Frame) -> Result<ComplexRead<'a>> {
    frame.samples.read_range(ctx, 0, frame.samples.len())
}

/// The pair density `M_{ak, bk'}(q + G)` of band `a` of `left` and band `b` of `right` (their
/// grid values `l` and `r`, a frame's layout) on the engine's vectors, with the complete PAW
/// augmentation. A spinor frame's rows are `(component, band)`; its density sums the components.
#[allow(clippy::too_many_arguments)]
fn pair_density(
    engine: &Engine,
    augmentation: &Augmentation<'_>,
    scratch: &mut Scratch,
    (left, l): (&Frame, &[C64]),
    (right, r): (&Frame, &[C64]),
    a: usize,
    b: usize,
    out: &mut [C64],
) -> Result<()> {
    let components = left.components;
    let (lb, rb) = (
        left.energies.as_slice().len(),
        right.energies.as_slice().len(),
    );
    let points = l.len() / (components * lb);
    engine.smooth_components(
        scratch,
        &components_of(l, lb, points, components, a),
        &components_of(r, rb, points, components, b),
        out,
    )?;
    let projection = |frame: &Frame, band: usize, c: usize| -> Vec<C64> {
        let row = c * frame.energies.as_slice().len() + band;
        (0..frame.projections.ncols())
            .map(|i| frame.projections[(row, i)])
            .collect()
    };
    for c in 0..components {
        augmentation.add(&projection(left, a, c), &projection(right, b, c), out);
    }
    Ok(())
}

/// The unscaled direct block of one pair `(k, k')`: `sum_{G G'} M_vv'(G')^* W_{G'G} M_cc'(G)` with
/// rows `(v, c)` and columns `(v', c')`, from the pair densities `valence` (vectors x `(v, v')`)
/// and `conduction` (vectors x `(c, c')`) and `W` on the same vectors.
pub fn contract(
    valence: &DMatrix<C64>,
    screened: &DMatrix<C64>,
    conduction: &DMatrix<C64>,
    nv: usize,
    nc: usize,
) -> Result<DMatrix<C64>> {
    if valence.shape() != (screened.nrows(), nv * nv)
        || conduction.shape() != (screened.ncols(), nc * nc)
    {
        return Err(invalid("the direct term's pair densities do not match W"));
    }
    // Serial products (nalgebra's loops, not the platform multiply), so a block's bits depend
    // neither on the lane that computes it nor on the thread count (gate E7).
    let screened = screened * conduction;
    let pairs = valence.adjoint() * screened;
    Ok(DMatrix::from_fn(nv * nc, nv * nc, |i, j| {
        let (v, c) = (i / nc, i % nc);
        let (vp, cp) = (j / nc, j % nc);
        pairs[(v * nv + vp, c * nc + cp)]
    }))
}

/// Every eigenpair of the Hermitian `h`, ascending.
pub fn solve_dense(h: &DMatrix<C64>) -> Result<Excitons> {
    let (energies, vectors) = eigen(h.clone())?;
    Ok(Excitons {
        residuals: vec![0.; energies.len()],
        energies,
        vectors,
        converged: true,
        solver: "dense",
        antiresonant: None,
        branches: None,
    })
}

/// The `roots` lowest eigenpairs of the Hermitian `h` by Davidson on its real form (each root
/// twice), then a complex Rayleigh-Ritz step on the span of the real roots.
pub fn solve_davidson(
    h: &DMatrix<C64>,
    roots: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Excitons> {
    let n = h.nrows();
    if roots == 0 || roots > n {
        return Err(invalid(format!(
            "num_states must be between 1 and the {n} transitions of the window"
        )));
    }
    let mut options = davidson::Options::new((2 * roots).min(2 * n));
    options.tolerance = tolerance;
    options.max_iterations = max_iterations;
    let real = davidson::lowest(&RealForm(h), &options)?;
    let mut basis: Vec<nalgebra::DVector<C64>> = vec![];
    for j in 0..real.vectors.ncols() {
        let mut v = nalgebra::DVector::from_fn(n, |i, _| {
            C64::new(real.vectors[(i, j)], real.vectors[(n + i, j)])
        });
        for _ in 0..2 {
            for b in &basis {
                let overlap = b.dotc(&v);
                v -= b * overlap;
            }
        }
        let norm = v.norm();
        if norm > 1e-6 {
            basis.push(v / C64::new(norm, 0.));
        }
    }
    if basis.len() < roots {
        return Err(invalid(
            "the Davidson roots of the real form do not span the requested excitons",
        ));
    }
    let span = DMatrix::from_columns(&basis);
    let image = complex_product(h.as_view(), false, span.as_view())?;
    let projected = complex_product(span.as_view(), true, image.as_view())?;
    let (values, small) = eigen((&projected + projected.adjoint()) * C64::new(0.5, 0.))?;
    let vectors = &span * small.columns(0, roots);
    let residuals = (0..roots)
        .map(|l| {
            let x = vectors.column(l);
            (h * x - x * C64::new(values[l], 0.)).norm()
        })
        .collect::<Vec<f64>>();
    Ok(Excitons {
        energies: values[..roots].to_vec(),
        converged: real.converged && residuals.iter().all(|r| *r <= 10. * tolerance),
        vectors,
        residuals,
        solver: "davidson",
        antiresonant: None,
        branches: None,
    })
}

/// The real form `[[Re H, -Im H], [Im H, Re H]]` of a Hermitian matrix.
struct RealForm<'a>(&'a DMatrix<C64>);

impl Symmetric for RealForm<'_> {
    fn dimension(&self) -> usize {
        2 * self.0.nrows()
    }
    fn diagonal(&self) -> Vec<f64> {
        let d: Vec<f64> = (0..self.0.nrows()).map(|i| self.0[(i, i)].re).collect();
        d.iter().chain(&d).copied().collect()
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n = self.0.nrows();
        let complex = DMatrix::from_fn(n, block.ncols(), |i, j| {
            C64::new(block[(i, j)], block[(n + i, j)])
        });
        let image = complex_product(self.0.as_view(), false, complex.as_view())?;
        Ok(DMatrix::from_fn(2 * n, block.ncols(), |i, j| {
            if i < n {
                image[(i, j)].re
            } else {
                image[(i - n, j)].im
            }
        }))
    }
}

impl Operator {
    /// Every exciton, by a dense diagonalization.
    pub fn dense(&self) -> Result<Excitons> {
        solve_dense(&self.hamiltonian)
    }

    /// The `roots` lowest excitons: dense up to [`DENSE_LIMIT`] transitions, else
    /// [`solve_davidson`] with residuals below `tolerance` in at most `max_iterations` steps.
    pub fn lowest(&self, roots: usize, tolerance: f64, max_iterations: usize) -> Result<Excitons> {
        let n = self.energies.len();
        if roots == 0 || roots > n {
            return Err(invalid(format!(
                "num_states must be between 1 and the {n} transitions of the window"
            )));
        }
        if self.coupling.is_some() {
            let mut all = self.coupled()?;
            all.energies.truncate(roots);
            all.vectors = all.vectors.columns(0, roots).into_owned();
            all.antiresonant = all.antiresonant.map(|y| y.columns(0, roots).into_owned());
            all.residuals.truncate(roots);
            return Ok(all);
        }
        if n > DENSE_LIMIT {
            return solve_davidson(&self.hamiltonian, roots, tolerance, max_iterations);
        }
        let all = self.dense()?;
        Ok(Excitons {
            energies: all.energies[..roots].to_vec(),
            vectors: all.vectors.columns(0, roots).into_owned(),
            residuals: vec![0.; roots],
            converged: true,
            solver: "dense",
            antiresonant: None,
            branches: None,
        })
    }

    /// Every exciton of the coupling form (module documentation): the `n` positive roots with
    /// their `X` and `Y`, and every eigenpair for the tensor. Refuses an operator built without
    /// the coupling blocks, and an `M` that is not positive definite (an unstable reference).
    pub fn coupled(&self) -> Result<Excitons> {
        let Some(coupling) = &self.coupling else {
            return Err(invalid(
                "the operator was built in the Tamm-Dancoff approximation (Request::coupling)",
            ));
        };
        let n = self.energies.len();
        let (a, b, a2) = (
            &self.hamiltonian,
            &coupling.resonant,
            &coupling.antiresonant,
        );
        let m = DMatrix::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
            (true, true) => a[(i, j)],
            (true, false) => b[(i, j - n)],
            (false, true) => b[(j, i - n)].conj(),
            (false, false) => a2[(i - n, j - n)],
        });
        // A complex Cholesky takes the square root of a negative pivot without failing, so the
        // pivots are checked: real and positive for a positive definite `M`.
        let unstable = || {
            Error::Unsupported(
                "the coupling form needs [[A, B], [B^dagger, A']] positive definite, and it is \
                 not: the reference is unstable against the coupling (approximation='tda' keeps \
                 the resonant block)"
                    .into(),
            )
        };
        let lower = m.cholesky().ok_or_else(unstable)?.l();
        if (0..2 * n).any(|i| {
            let pivot = lower[(i, i)];
            !(pivot.re > 0. && pivot.im.abs() <= 1e-12 * pivot.re)
        }) {
            return Err(unstable());
        }
        let mut signed = lower.clone();
        for i in n..2 * n {
            signed.row_mut(i).neg_mut();
        }
        let w = lower.adjoint() * &signed;
        let (energies, u) = eigen((&w + w.adjoint()) * C64::new(0.5, 0.))?;
        if energies[n - 1] >= 0. || energies[n] <= 0. {
            return Err(invalid(
                "the coupling form's roots do not split into n negative and n positive ones",
            ));
        }
        let mut vectors = &signed * u;
        for (l, e) in energies.iter().enumerate() {
            vectors.column_mut(l).unscale_mut(e.abs().sqrt());
        }
        Ok(Excitons {
            energies: energies[n..].to_vec(),
            vectors: vectors.view((0, n), (n, n)).into_owned(),
            residuals: vec![0.; n],
            converged: true,
            solver: "coupling",
            antiresonant: Some(vectors.view((n, n), (n, n)).into_owned()),
            branches: Some(Branches { energies, vectors }),
        })
    }

    /// `o^a_l` of each exciton (Bohr): `sum_t h^a_t X^l_t`, plus `sum_t h~^a_t Y^l_t` in the
    /// coupling form.
    pub fn dipoles_of(&self, excitons: &Excitons) -> Vec<[C64; 3]> {
        self.amplitude_dipoles(&excitons.vectors, excitons.antiresonant.as_ref())
    }

    /// Summary observables of `excitons` (binding energies, line areas in `Im eps`, inverse
    /// participation ratios, k-point weights, and dominant transitions).
    pub fn analyze(&self, excitons: &Excitons) -> ExcitonSummary {
        let roots = excitons.energies.len();
        let nk = self.mesh.iter().product::<usize>();
        let binding = excitons.energies.iter().map(|e| self.onset - e).collect();
        let areas = self
            .dipoles_of(excitons)
            .iter()
            .flat_map(|o| o.map(|v| 4. * std::f64::consts::PI.powi(2) * self.scale * v.norm_sqr()))
            .collect();
        let mut participation = Vec::with_capacity(roots);
        let mut kpoint_weights = vec![0.; roots * nk];
        let mut dominant = Vec::with_capacity(roots);
        let mut dominant_channels = Vec::with_capacity(roots);
        for l in 0..roots {
            let a = excitons.vectors.column(l).normalize();
            let ipr: f64 = a.iter().map(|x| x.norm_sqr().powi(2)).sum();
            participation.push(1. / ipr);
            for (t, transition) in self.transitions.iter().enumerate() {
                kpoint_weights[l * nk + transition.k] += a[t].norm_sqr();
            }
            let (t, weight) = a
                .iter()
                .enumerate()
                .map(|(t, x)| (t, x.norm_sqr()))
                .fold((0, -1.), |best, x| if x.1 > best.1 { x } else { best });
            let transition = self.transitions[t];
            dominant.push((
                transition.k,
                transition.valence,
                transition.conduction,
                weight,
            ));
            dominant_channels.push(transition.channel);
        }
        ExcitonSummary {
            binding,
            areas,
            participation,
            kpoint_weights,
            dominant,
            dominant_channels,
        }
    }

    fn amplitude_dipoles(&self, x: &DMatrix<C64>, y: Option<&DMatrix<C64>>) -> Vec<[C64; 3]> {
        (0..x.ncols())
            .map(|l| {
                std::array::from_fn(|a| {
                    let resonant: C64 = self
                        .dipoles
                        .iter()
                        .enumerate()
                        .map(|(t, h)| h[a] * x[(t, l)])
                        .sum();
                    let antiresonant: C64 = match (y, &self.coupling) {
                        (Some(y), Some(c)) => c
                            .dipoles
                            .iter()
                            .enumerate()
                            .map(|(t, h)| h[a] * y[(t, l)])
                            .sum(),
                        _ => C64::new(0., 0.),
                    };
                    resonant + antiresonant
                })
            })
            .collect()
    }

    /// `eps^{ab}(z)` from excitons (module documentation): exact with every exciton, and in the
    /// coupling form from every eigenpair of [`Operator::coupled`].
    pub fn tensor(&self, excitons: &Excitons, z: C64) -> [[C64; 3]; 3] {
        if let Some(branches) = &excitons.branches {
            let n = self.energies.len();
            let x = branches.vectors.rows(0, n).into_owned();
            let y = branches.vectors.rows(n, n).into_owned();
            let o = self.amplitude_dipoles(&x, Some(&y));
            return std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    let chi: C64 = branches
                        .energies
                        .iter()
                        .zip(&o)
                        .map(|(e, o)| o[a] * o[b].conj() * e.signum() / (z - e))
                        .sum::<C64>()
                        * self.scale;
                    C64::new(f64::from(a == b), 0.) - chi * (4. * PI)
                })
            });
        }
        let o = self.dipoles_of(excitons);
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let chi: C64 = excitons
                    .energies
                    .iter()
                    .zip(&o)
                    .map(|(e, o)| o[a] * o[b].conj() / (z - e) - o[a].conj() * o[b] / (z + e))
                    .sum::<C64>()
                    * self.scale;
                C64::new(f64::from(a == b), 0.) - chi * (4. * PI)
            })
        })
    }

    /// The Haydock spectra of the operator: the Lanczos poles from `conj(e . h)` for the three
    /// axes and the three pairs of axes (`krylov` vectors each).
    pub fn haydock(&self, ctx: &crate::Context, krylov: usize) -> Result<Haydock> {
        let start = |e: [f64; 3]| -> Vec<C64> {
            self.dipoles
                .iter()
                .map(|h| (h[0] * e[0] + h[1] * e[1] + h[2] * e[2]).conj())
                .collect()
        };
        let lines = [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 0.],
            [1., 0., 1.],
            [0., 1., 1.],
        ];
        let h = &self.hamiltonian;
        let poles = lines
            .iter()
            .map(|e| {
                lanczos::poles(ctx, &start(*e), krylov, |v, out| {
                    let x = nalgebra::DVectorView::from_slice(v, v.len());
                    let y = h * x;
                    out.copy_from_slice(y.as_slice());
                    Ok(())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Haydock {
            poles,
            scale: self.scale,
        })
    }
}

/// The Lanczos poles of the six dipole lines of an [`Operator`] (axes, then the pairs `xy`, `xz`,
/// `yz`).
pub struct Haydock {
    pub poles: Vec<lanczos::Ritz>,
    scale: f64,
}

impl Haydock {
    /// `eps^{ab}(z)`: `chi^{ee} = s [R(z) + R(-z)]` per line and the off-diagonal elements by
    /// polarization.
    pub fn tensor(&self, z: C64) -> [[C64; 3]; 3] {
        let chi: Vec<C64> = self
            .poles
            .iter()
            .map(|p| (p.resolvent(z) + p.resolvent(-z)) * self.scale)
            .collect();
        let pair = |a: usize, b: usize| -> C64 {
            let line = match (a.min(b), a.max(b)) {
                (0, 1) => 3,
                (0, 2) => 4,
                _ => 5,
            };
            (chi[line] - chi[a] - chi[b]) * 0.5
        };
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let value = if a == b { chi[a] } else { pair(a, b) };
                C64::new(f64::from(a == b), 0.) - value * (4. * PI)
            })
        })
    }
}

/// The full (coupling) form of the exchange-only equation, dense: `eps^{ab}(z)` from
/// `R (Lambda0(z) - K)^-1 R^dagger s` over both orders of every transition of `pairs`, with
/// `Lambda0 = diag(z - D, -(z + D))` and `K = s R^dagger v R` on the local-field rows. By the
/// Woodbury identity this is the Dyson equation of `pairs.tensor` (gate E1): the oracle of the
/// coupling form's exchange, one linear solve per frequency, not a route to results.
pub fn coupled_tensor(pairs: &Pairs, z: C64) -> Result<[[C64; 3]; 3]> {
    let n = pairs.energies.len();
    let coulomb = pairs.coulomb();
    let nb = coulomb.len();
    let weighted = DMatrix::from_fn(nb, 2 * n, |g, i| {
        pairs.column(i / 2, i % 2 == 1)[3 + g] * coulomb[g]
    });
    let mut kernel = complex_product(weighted.as_view(), true, weighted.as_view())?
        * C64::new(pairs.scale(), 0.);
    if let Some(hk) = pairs.hubbard() {
        let u_cols = DMatrix::from_fn(hk.size, 2 * n, |u, i| {
            pairs.column(i / 2, i % 2 == 1)[3 + nb + u]
        });
        let mid = complex_product(hk.singlet.as_view(), false, u_cols.as_view())?;
        let hu = complex_product(u_cols.as_view(), true, mid.as_view())?
            * C64::new(pairs.scale() * pairs.volume / pairs.degeneracy(), 0.);
        kernel += hu;
    }
    let system = DMatrix::from_fn(2 * n, 2 * n, |i, j| {
        let diagonal = if i == j {
            let d = pairs.energies[i / 2];
            if i % 2 == 0 {
                z - d
            } else {
                -(z + d)
            }
        } else {
            C64::new(0., 0.)
        };
        diagonal - kernel[(i, j)]
    });
    let heads = DMatrix::from_fn(2 * n, 3, |i, b| pairs.column(i / 2, i % 2 == 1)[b].conj());
    let solved = system
        .lu()
        .solve(&heads)
        .ok_or_else(|| invalid("the coupled exchange system is singular"))?;
    Ok(std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let chi: C64 = (0..2 * n)
                .map(|i| pairs.column(i / 2, i % 2 == 1)[a] * solved[(i, b)])
                .sum::<C64>()
                * pairs.scale();
            C64::new(f64::from(a == b), 0.) - chi * (4. * PI)
        })
    }))
}

/// The dielectric tensor of [`Kernel::Bse`] on `optical`'s frequencies: the singlet operator of
/// the window, its excitons (dense) or its Haydock spectra.
pub fn dielectric(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    optical: &Optical,
) -> Result<Dielectric> {
    if optical.alpha.is_some() {
        return Err(invalid(
            "alpha is the long-range kernel's: pass kernel='long-range'",
        ));
    }
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.window = optical.window;
    request.scissor = optical.scissor;
    request.cutoff = optical.cutoff;
    let op = operator(sampler, mesh, &request)?;
    let ctx = &sampler.crystal.context;
    let n = op.energies.len();
    let evaluate: Box<dyn Fn(C64) -> [[C64; 3]; 3] + Sync> = if n <= DENSE_LIMIT {
        let all = op.dense()?;
        let op = &op;
        Box::new(move |z| op.tensor(&all, z))
    } else {
        let haydock = op.haydock(ctx, KRYLOV)?;
        Box::new(move |z| haydock.tensor(z))
    };
    let tensor = ctx.map_independent(optical.frequencies.len() + 1, |j| {
        Ok(match optical.frequencies.get(j) {
            Some(omega) => evaluate(C64::new(*omega, optical.damping)),
            None => evaluate(C64::new(0., 0.)),
        })
    })?;
    if tensor
        .iter()
        .flatten()
        .flatten()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid("the dielectric tensor is not finite"));
    }
    let (tensor, last) = tensor.split_at(optical.frequencies.len());
    Ok(Dielectric {
        frequencies: optical.frequencies.clone(),
        damping: optical.damping,
        kernel: Kernel::Bse,
        tensor: tensor.to_vec(),
        static_tensor: last[0].map(|row| row.map(|v| v.re)),
        mesh: op.mesh,
        window: vec![op.window],
        transitions: n,
        vectors: op.vectors,
        cutoff: op.cutoff,
        scissor: op.scissor,
        onset: op.onset,
        sum_rule: first_moment(&op),
        electrons: op.electrons,
        volume: op.volume,
        residual: op.residual,
        alpha: None,
    })
}

/// `N^{ab}` of the first moment of the excitonic spectrum: `eps -> delta - 4 pi N / (Omega z^2)`
/// with `N^{ab} = 2 Omega s Re[(h^b)^dagger H h^a]` (the commutator of `H` with the dipoles).
pub fn first_moment(op: &Operator) -> [[f64; 3]; 3] {
    let n = op.energies.len();
    let column =
        |a: usize| nalgebra::DVector::from_iterator(n, op.dipoles.iter().map(|h| h[a].conj()));
    let images: Vec<_> = (0..3).map(|a| &op.hamiltonian * column(a)).collect();
    std::array::from_fn(|a| {
        std::array::from_fn(|b| 2. * op.volume * op.scale * column(b).dotc(&images[a]).re)
    })
}
