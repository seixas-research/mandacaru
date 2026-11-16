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
//! approximation, and with `K^x = 0` too its independent-particle tensor. The full (coupling)
//! form with the exchange alone, [`coupled_tensor`], is T5a's Dyson equation by the Woodbury
//! identity (gate E1), a dense check and not a route to results. Below [`DENSE_LIMIT`]
//! transitions `H` is diagonalized whole; above it the lowest excitons come from Davidson on the
//! real form `[[Re H, -Im H], [Im H, Re H]]` (each root twice, then a complex Rayleigh-Ritz step),
//! and the spectrum from the Lanczos recursion of [`crate::math::lanczos`] from each dipole
//! vector (Haydock), the off-diagonal elements by polarization (the tensor of a crystal with time
//! reversal is symmetric).
//!
//! Refused here besides T5a's refusals: spin-polarized crystals (the two-channel exchange is not
//! built), a window wider than the memory budget for the explicit `H`.
use super::{
    super::{
        bloch_mesh::{MeshIndex, Sampler},
        invalid,
        response::setup::Setup,
        screening::{
            bloch::Bloch,
            interaction::{self, Block},
            medium::{Medium, Paw},
            transitions::{Augmentation, Engine, Remainders},
        },
        spectrum::Smearing,
        wannier::sampling::Frame,
    },
    dielectric::{columns, Dielectric, Kernel, Pairs, Request as Optical},
};
use crate::{
    math::{
        davidson::{self, Symmetric},
        gemm::complex_product,
        lanczos,
        linalg::eigen,
    },
    product,
    quantum::C64,
    Error, Result,
};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;

/// The largest number of transitions whose `H` is diagonalized whole.
pub const DENSE_LIMIT: usize = 3000;
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
}

impl Request {
    pub fn new(kernel: Kernel, spin: Spin) -> Self {
        Self {
            kernel,
            spin,
            window: None,
            scissor: 0.,
            cutoff: super::dielectric::LOCAL_FIELD_CUTOFF,
        }
    }
}

/// One transition of the operator: mesh point, valence and conduction band (indices into the
/// crystal's bands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub k: usize,
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
    _memory: crate::Reservation,
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
    /// `"dense"` or `"davidson"`.
    pub solver: &'static str,
}

fn check(sampler: &Sampler<'_>, request: &Request) -> Result<()> {
    if sampler.channels() != 1 {
        return Err(Error::Unsupported(
            "the Bethe-Salpeter equation of a spin-polarized crystal is not built (its \
             exchange couples the two spin channels)"
                .into(),
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
/// the middle of the gap and a 80th of it.
fn midgap(frames: &[Frame], occupied: usize) -> Result<(f64, f64)> {
    let top = frames
        .iter()
        .map(|f| f.energies.as_slice()[occupied - 1])
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = frames
        .iter()
        .map(|f| f.energies.as_slice()[occupied])
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
    let frames = sampler.frames(mesh, &[all], |_, _| Ok(()))?;
    let mut optical = Optical::new(vec![0.], 0., request.kernel);
    if request.kernel == Kernel::Bse {
        optical.kernel = Kernel::Hartree;
    }
    optical.window = request.window;
    optical.scissor = request.scissor;
    optical.cutoff = request.cutoff;
    let pairs = columns(sampler, mesh, &optical, Some(&frames))?;
    let (valence, conduction) = pairs.window[0];
    let n = pairs.energies.len();
    let nk = mesh.points();
    let memory = ctx.reserve(product(&[n, n, 16, 3])?).map_err(|_| {
        Error::Unsupported(format!(
            "the Bethe-Salpeter operator of {n} transitions ({valence} valence x {conduction} \
             conduction bands x {nk} k-points) does not fit the memory budget: pass a smaller \
             bands=(valence, conduction)"
        ))
    })?;
    // The channel's occupied bands: the window ends at the last of them.
    let occupied = sampler
        .manifold(
            super::super::berry::Bands::Occupied,
            super::super::berry::Metal::Undefined,
        )?
        .ranges[0]
        .1;
    let lowest = occupied - valence;
    let transitions: Vec<Transition> = (0..nk)
        .flat_map(|k| {
            (0..valence).flat_map(move |v| {
                (0..conduction).map(move |c| Transition {
                    k,
                    valence: lowest + v,
                    conduction: occupied + c,
                })
            })
        })
        .collect();
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
    if request.kernel != Kernel::Independent && request.spin == Spin::Singlet {
        hamiltonian += exchange(&pairs, &(0..n).collect::<Vec<_>>())?;
    }
    let mut head = 0.;
    let mut static_tensor = [[0.; 3]; 3];
    if request.kernel == Kernel::Bse {
        // eps_M(0) of every band with local fields: the screening of the head.
        let mut screening = Optical::new(vec![0.], 0., Kernel::Hartree);
        screening.cutoff = request.cutoff;
        let macroscopic = columns(sampler, mesh, &screening, Some(&frames))?;
        static_tensor = macroscopic
            .tensor(C64::new(0., 0.))?
            .map(|row| row.map(|v| v.re));
        let eps = Matrix3::from_fn(|a, b| static_tensor[a][b]);
        head = interaction::head_average(&sampler.reciprocal, mesh, &eps, HEAD_ORDER)?;
        let setup = Setup::new(crystal)?;
        let remainders = Remainders::new(ctx, &setup)?;
        let (fermi, width) = midgap(&frames[0], occupied)?;
        let medium = Medium {
            ctx,
            grid: &crystal.grid,
            volume: pairs.volume,
            fermi,
            width,
            degeneracy: sampler.degeneracy(),
            smearing: Smearing::FermiDirac,
            constrained: false,
            paw: Some(Paw {
                crystal,
                setup: &setup,
                remainders: &remainders,
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
            request.cutoff,
        )?;
        let origin = blocks[0]
            .g
            .iter()
            .position(|g| *g == [0, 0, 0])
            .ok_or_else(|| invalid("the screened vectors of q = 0 lack G = 0"))?;
        blocks[0].screened[(origin, origin)] = C64::new(head, 0.);
        let window = Window {
            lowest,
            valence,
            occupied,
            conduction,
        };
        let scale = 1. / (nk as f64 * pairs.volume);
        let direct = direct(
            sampler,
            &frames[0],
            mesh,
            &blocks,
            &window,
            &paw_parts(&setup, &remainders),
        )?;
        for (k, row_block) in direct.into_iter().enumerate() {
            for (kp, block) in row_block {
                let (r0, c0) = (k * valence * conduction, kp * valence * conduction);
                for i in 0..valence * conduction {
                    for j in 0..valence * conduction {
                        hamiltonian[(r0 + i, c0 + j)] -= block[(i, j)] * scale;
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
    let hamiltonian = (&hamiltonian + hamiltonian.adjoint()) * C64::new(0.5, 0.);
    if hamiltonian
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
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
        _memory: memory,
    })
}

/// `K^x` on the transitions `rows` x `rows` of `pairs`: `s (V R)^dagger (V R)`, `R` the
/// local-field rows of each `rho_vc` column, `V` the square root of `v(G)`.
fn exchange(pairs: &Pairs, rows: &[usize]) -> Result<DMatrix<C64>> {
    let coulomb = pairs.coulomb();
    let nb = coulomb.len();
    let weighted = DMatrix::from_fn(nb, rows.len(), |g, i| {
        pairs.column(rows[i], false)[3 + g] * coulomb[g]
    });
    let product = complex_product(weighted.as_view(), true, weighted.as_view())?;
    Ok(product * C64::new(pairs.scale(), 0.))
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
#[allow(clippy::type_complexity)]
fn direct(
    sampler: &Sampler<'_>,
    frames: &[Frame],
    mesh: &MeshIndex,
    blocks: &[Block],
    window: &Window,
    paw: &PawParts<'_>,
) -> Result<Vec<Vec<(usize, DMatrix<C64>)>>> {
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let nk = mesh.points();
    let points = crystal.grid.index.points();
    let (nv, nc) = (window.valence, window.conduction);
    let p = crystal.p;
    let projection = |frame: &Frame, a: usize| -> Vec<C64> {
        (0..p).map(|i| frame.projections[(a, i)]).collect()
    };
    let one = |t: usize| -> Result<Vec<(usize, usize, DMatrix<C64>)>> {
        let block = &blocks[t];
        let engine = Engine::new(ctx, &crystal.grid, block.q, &block.g)?;
        let augmentation = Augmentation::new(crystal, paw.setup, paw.remainders, &engine.vectors)?;
        let mut scratch = engine.scratch(ctx)?;
        let ng = block.g.len();
        let mut out = Vec::with_capacity(nk);
        for k in 0..nk {
            let kp = mesh.shift(k, block.steps);
            let (left, right) = (&frames[k], &frames[kp]);
            let l = left.samples.read_range(
                ctx,
                0,
                product(&[left.energies.as_slice().len(), points])?,
            )?;
            let r = right.samples.read_range(
                ctx,
                0,
                product(&[right.energies.as_slice().len(), points])?,
            )?;
            let density = |scratch: &mut _, a: usize, b: usize, out: &mut [C64]| -> Result<()> {
                engine.smooth(
                    scratch,
                    &l[a * points..(a + 1) * points],
                    &r[b * points..(b + 1) * points],
                    out,
                )?;
                augmentation.add(&projection(left, a), &projection(right, b), out);
                Ok(())
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
            out.push((
                k,
                kp,
                contract(&valence, &block.screened, &conduction, nv, nc)?,
            ));
        }
        Ok(out)
    };
    let solved = ctx.map_independent(blocks.len(), one)?;
    let mut grouped: Vec<Vec<(usize, DMatrix<C64>)>> = (0..nk).map(|_| vec![]).collect();
    for list in solved {
        for (k, kp, kernel) in list {
            grouped[k].push((kp, kernel));
        }
    }
    Ok(grouped)
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
        })
    }

    /// `o^a_l = sum_t h^a_t A^l_t` of each exciton (Bohr).
    pub fn dipoles_of(&self, excitons: &Excitons) -> Vec<[C64; 3]> {
        (0..excitons.energies.len())
            .map(|l| {
                std::array::from_fn(|a| {
                    self.dipoles
                        .iter()
                        .enumerate()
                        .map(|(t, h)| h[a] * excitons.vectors[(t, l)])
                        .sum()
                })
            })
            .collect()
    }

    /// `eps^{ab}(z)` from excitons (module documentation): exact with every exciton.
    pub fn tensor(&self, excitons: &Excitons, z: C64) -> [[C64; 3]; 3] {
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
/// Woodbury identity this is the Dyson equation of `pairs.tensor` (gate E1); it is a check, not
/// a route to results.
pub fn coupled_tensor(pairs: &Pairs, z: C64) -> Result<[[C64; 3]; 3]> {
    let n = pairs.energies.len();
    let coulomb = pairs.coulomb();
    let nb = coulomb.len();
    let weighted = DMatrix::from_fn(nb, 2 * n, |g, i| {
        pairs.column(i / 2, i % 2 == 1)[3 + g] * coulomb[g]
    });
    let kernel = complex_product(weighted.as_view(), true, weighted.as_view())?
        * C64::new(pairs.scale(), 0.);
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
