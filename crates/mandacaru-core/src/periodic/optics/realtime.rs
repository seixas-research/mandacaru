//! Real-time propagation of a crystal in the velocity gauge (`TDDFT_PLAN.md`, section 13.5, T6b).
//!
//! A spatially uniform vector potential `A(t) = A_0 s(t) e` (`A = A_phys / c`, a momentum in
//! Bohr^-1) enters the fixed-basis Hamiltonian of every k-point (formulation F1, owner decision
//! Q20a): `H_o(k; A) = L^dagger H(k)[D] L + A . P_o(k) + (A^2 / 2) 1`, with `P_o = L^dagger P L` the
//! momentum of the Bloch basis functions ([`super::momentum::basis_momenta`]: the exact
//! derivative of the SCF's own kinetic matrix plus the PAW one-center momentum) and `L` the
//! Loewdin frame of the SCF. The occupied orbitals `C_k(t)` of every spin channel and k-point are
//! propagated by `math::propagator` (ETRS by default, owner decision Q18b) and the macroscopic
//! current density is `j(t) = -(1 / Omega) sum_{s,k} w_k tr[D_o (P_o + A 1)]` (electron charge
//! `-1`; `D_o` carries the band degeneracy), `-(1 / Omega) dE / dA` at fixed `D`.
//!
//! **The mesh is rebuilt whole.** A uniform `A` breaks time reversal and every operation that
//! moves `e`; the SCF's symmetrization would project the response out without an error
//! (section 13.5.4). The propagation therefore builds its own setup with `symmetry = false` on
//! the whole mesh (closed under `-k`, uniform weights) from the SCF's recipe, and starts each
//! k-point from the converged potential of the SCF; no SCF code runs with a symmetrizer.
//!
//! **The diamagnetic balance** (owner decision Q21a). In a finite basis the static paramagnetic
//! response `N_dia` (the band sum of [`super::momentum::transitions`]) does not cancel the
//! diamagnetic `N`, and a kick leaves a constant current `-A_0 (N - N_dia) / Omega`. The run
//! reports `N`, `N_dia` and the deficit, and `corrected` is the current along `e` with it
//! removed, `j + A (N - N_dia) / Omega`. The band sum is exact for the static part of a
//! density-only kernel only; DFT+U (through its shell occupations) and a meta-GGA (through the
//! current term of `tau~`) respond to `Im D`, so with the adiabatic kernel their `N_dia e` is
//! measured first by self-consistency at a static `+-A_0` (R9b; vera's oracle, cross-checked by
//! two adiabatic ramps) and the correction uses that.
//!
//! The kernels: the functional's own (`Adiabatic`), the exchange-correlation potential frozen at
//! the ground state (`Hartree`: TD-Hartree, RPA with local fields), or the whole Hamiltonian frozen
//! (`Independent`).
//!
//! **A meta-GGA reads the gauge-invariant `tau~ = tau - |j|^2 / (2 n)`** (owner decision Q19a,
//! section 13.6.1; T6c-c): `j` is the paramagnetic current of the propagated Bloch orbitals,
//! from their Bloch gradients, and the Fock matrix gains `v_tau |j|^2 / (2 n^2)` in the local
//! potential and the imaginary-antisymmetric `F^j`, so it stays the derivative of the energy.
//!
//! Refused: spinor crystals, hybrids (Q24), rVV10 (Q28), DFT+U or a meta-GGA with
//! `formulation="moved"` or `kernel="hartree"`, metals and fractional occupations
//! (Q25), a charged cell, float32, a mesh not closed under `-k`, and a snapshot without a setup
//! recipe.
use super::{
    super::{
        bloch_mesh::{MeshIndex, Sampler},
        fields::{bloch_gradients, kinetic_density},
        invalid,
        setup::{parse::create_setup, pulay::Mesh, PeriodicSetup},
        solver::{
            collinear_xc_energy, density, diagonalize_with_exchange, energy, exchange_correlation,
            hamiltonian, local_potential,
        },
        wannier::sampling::Frame,
        Crystal, KPoint, PeriodicResult,
    },
    momentum::{basis_momenta, check_inversion_closed, one_center_momentum, transitions},
};
use crate::{
    math::{
        fft::Fft3,
        linalg::eigen,
        propagator::{Channel, Evaluation, FockMap, Options, Propagator, Scheme},
        storage::{MatrixSource, Precision, StoredMatrix},
    },
    product,
    quantum::C64,
    Buffer, Error, Level, Result,
};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// What responds to the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// The functional's own adiabatic Hamiltonian, rebuilt from the propagated density.
    Adiabatic,
    /// `H` frozen at the ground state: independent particles (the oracle of R8 and R10).
    Independent,
    /// The exchange-correlation potential frozen at the ground state, the Hartree potential
    /// rebuilt: TD-Hartree, the random-phase approximation with local fields (R13).
    Hartree,
}
impl Kernel {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "adiabatic" => Ok(Self::Adiabatic),
            "independent" => Ok(Self::Independent),
            "hartree" => Ok(Self::Hartree),
            _ => Err(invalid(format!(
                "unknown kernel {name:?}; expected 'adiabatic', 'hartree' or 'independent'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Adiabatic => "adiabatic",
            Self::Independent => "independent",
            Self::Hartree => "hartree",
        }
    }
}

/// How the vector potential enters the static LCAO basis (TDDFT_PLAN.md 13.5.2, Q20).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formulation {
    /// F1: the basis stays at `k` and `A . P + A^2 / 2` enters the Hamiltonian; the velocity
    /// form, any `A(t)`.
    Fixed,
    /// F2: the gauge transform `exp(i A . r)` moves the occupied bands into the Bloch basis at
    /// `k + A`, where they evolve field-free; the current is `-(1/Omega) sum w tr[D P(k + A)]`. A
    /// constant `A` only (a kick); the length form of the intra-atomic position.
    Moved,
}
impl Formulation {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "fixed" => Ok(Self::Fixed),
            "moved" => Ok(Self::Moved),
            _ => Err(invalid(format!(
                "unknown formulation {name:?}; expected 'fixed' or 'moved'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Moved => "moved",
        }
    }
}

/// The time profile `s(t)` of `A(t) = A_0 s(t) e`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Field {
    /// `s = 1` for `t >= 0`: the electric field `-A_0 delta(t) e` (the `t = 0` sample is `0+`).
    Kick,
    /// `s = sin^2(pi t / 2 T)` up to `T = duration`, then 1: the adiabatic switch-on of a static
    /// `A` (R9b) and a smooth pulse for the power balance (R2).
    Ramp { duration: f64 },
}
impl Field {
    pub fn profile(self, time: f64) -> f64 {
        match self {
            Self::Kick => 1.,
            Self::Ramp { duration } => {
                if time >= duration {
                    1.
                } else if time <= 0. {
                    0.
                } else {
                    (std::f64::consts::FRAC_PI_2 * time / duration)
                        .sin()
                        .powi(2)
                }
            }
        }
    }
    /// `ds / dt`.
    pub fn rate(self, time: f64) -> f64 {
        match self {
            Self::Kick => 0.,
            Self::Ramp { duration } => {
                if time <= 0. || time >= duration {
                    0.
                } else {
                    let x = std::f64::consts::PI * time / duration;
                    std::f64::consts::FRAC_PI_2 / duration * x.sin()
                }
            }
        }
    }
}

/// A crystal run, in atomic units.
#[derive(Clone, Debug)]
pub struct Request {
    pub direction: [f64; 3],
    /// `A_0`, Bohr^-1.
    pub strength: f64,
    pub field: Field,
    pub kernel: Kernel,
    pub formulation: Formulation,
    pub scheme: Scheme,
    pub time_step: f64,
    pub steps: usize,
    /// The ETRS corrector tolerance at the default kick, scaled with the kick as for a molecule
    /// (owner decision Q18c).
    pub corrector_tolerance: f64,
    pub max_corrector_iterations: usize,
    /// The largest energy deviation after a kick, in kick energies; `None` checks nothing. A
    /// ramp changes the energy by design and is never checked.
    pub max_energy_drift: Option<f64>,
    /// The Gamma-anchored propagation mesh; `None` takes the SCF's mesh, whole.
    pub mesh: Option<[usize; 3]>,
    /// The static response `N_dia e` the correction uses when known (a run along the same
    /// direction measured it); `None` reads the band sum, or measures it by self-consistency at a
    /// static field for a kernel with an `Im D` channel.
    pub statics: Option<[f64; 3]>,
}
impl Request {
    /// The molecular defaults (Q18b): ETRS, 1e-6 at the default kick, 8 iterations, a drift of
    /// at most 0.05 kick energies; a kick on the SCF's mesh with the adiabatic kernel.
    pub fn new(direction: [f64; 3], strength: f64, time_step: f64, steps: usize) -> Self {
        Self {
            direction,
            strength,
            field: Field::Kick,
            kernel: Kernel::Adiabatic,
            formulation: Formulation::Fixed,
            scheme: Scheme::Etrs,
            time_step,
            steps,
            corrector_tolerance: 1e-6,
            max_corrector_iterations: 8,
            max_energy_drift: Some(0.05),
            mesh: None,
            statics: None,
        }
    }
    /// [`Request::new`] from user units (`atomic`: already atomic): `strength` in 1/Angstrom,
    /// `time_step` and `duration` in femtoseconds, as the molecular request
    /// (`molecular::realtime::Request::from_user_units`, which owns the step count rule).
    pub fn from_user_units(
        direction: [f64; 3],
        strength: f64,
        time_step: f64,
        duration: f64,
        atomic: bool,
    ) -> Result<Self> {
        let user = crate::molecular::realtime::Request::from_user_units(
            direction, strength, time_step, duration, atomic,
        )?;
        Ok(Self::new(
            direction,
            user.strength,
            user.time_step,
            user.steps,
        ))
    }
    /// The tolerance the run applies: `corrector_tolerance x abs(strength) / default_strength()`.
    pub fn effective_corrector_tolerance(&self) -> f64 {
        self.corrector_tolerance * self.strength.abs()
            / crate::molecular::realtime::default_strength()
    }
}

/// Refuses the molecular options a crystal does not take: a crystal is kicked by a uniform
/// vector potential (`kick="density"`) in the velocity gauge (`gauge` `None` or `"velocity"`);
/// the position operator of a crystal is not periodic.
pub fn check_options(kick: &str, gauge: Option<&str>) -> Result<()> {
    if kick != "density" {
        return Err(invalid(format!(
            "a crystal takes kick='density' only (a vector-potential kick), not {kick:?}"
        )));
    }
    if let Some(gauge) = gauge.filter(|g| *g != "velocity") {
        return Err(invalid(format!(
            "a crystal runs in the velocity gauge: gauge must be None or 'velocity', not \
             {gauge:?} (the position operator of a crystal is not periodic)"
        )));
    }
    Ok(())
}

/// The trajectory of a crystal run and its record.
#[derive(Clone, Debug)]
pub struct CrystalRealTime {
    pub times: Vec<f64>,
    /// `j(t)`, the macroscopic current density (Cartesian, e / (Bohr^2 a.u. of time)).
    pub current: Vec<[f64; 3]>,
    /// `e . j(t)`.
    pub along: Vec<f64>,
    /// `e . j(t) + A(t) (N - N_dia) / Omega`: the current without the basis's static deficit.
    pub corrected: Vec<f64>,
    /// `j(t) + A(t) (N e - N_dia e) / Omega` along the three axes (F2: `j(t)` without the moved
    /// basis's static current): `e .` of it is `corrected`; a kick along each axis gives the
    /// tensor of [`propagate_axes`].
    pub corrected_current: Vec<[f64; 3]>,
    /// `A(t) . e`, Bohr^-1.
    pub vector_potential: Vec<f64>,
    pub energy: Vec<f64>,
    pub norm_error: Vec<f64>,
    pub ground_energy: f64,
    /// `E(0+) - E_ground` (a kick; `N A_0^2 / 2` for a ground state without current).
    pub kick_energy: f64,
    pub energy_drift: f64,
    /// `W = e_max - e_min` of the matrices at `t = 0`, the largest over the run, and `dt W`.
    pub width: f64,
    pub largest_width: f64,
    pub step_width: f64,
    pub worst_norm_error: f64,
    pub corrector_tolerance: f64,
    /// The share of the bare band sum `N_dia` carried by transitions above `pi / dt`.
    pub aliased_weight: f64,
    pub builds: usize,
    pub corrector_iterations: usize,
    /// The valence electrons of the cell, the band sum along `e` and their difference.
    pub electrons: f64,
    pub diamagnetic: f64,
    pub deficit: f64,
    /// The column `N_dia e` of the static paramagnetic response, and whether it was measured by
    /// self-consistency at a static field ([`Prepared::static_response`]; DFT+U and meta-GGAs,
    /// whose kernels respond to `Im D`; owner decision Q21a) instead of read from the band sum
    /// `band_sum` (a density-only kernel).
    pub diamagnetic_vector: [f64; 3],
    pub self_consistent: bool,
    pub band_sum: f64,
    /// The largest anti-Hermitian residual of `P` before Hermitization.
    pub momentum_residual: f64,
    /// The cell volume (Bohr^3), the propagation mesh and its k count.
    pub volume: f64,
    pub mesh: [usize; 3],
    pub kpoints: usize,
    /// The largest change of the `t = 0` density over the SCF's (R15), per grid point.
    pub initial_density_change: f64,
    pub formulation: Formulation,
    /// F2: the static current density of the moved basis's ground state along `e` (removed from
    /// `corrected`) and the transfer's departure from orthonormality; F1: zero.
    pub static_current: f64,
    pub transfer_defect: f64,
}

fn unit(direction: [f64; 3]) -> Result<[f64; 3]> {
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0. {
        return Err(invalid("the direction must be a finite nonzero vector"));
    }
    Ok(direction.map(|v| v / norm))
}

/// The refusals of the module documentation that a crystal decides by itself.
fn check_crystal(crystal: &Crystal, setup: &PeriodicSetup) -> Result<()> {
    let unsupported = |what: &str| Err(Error::Unsupported(what.into()));
    if crystal.spinor {
        return unsupported(
            "real-time propagation of a spinor crystal is not implemented (the noncollinear \
             kernel, T8)",
        );
    }
    let definition = crate::models::functional(&crystal.functional)?;
    if definition.screened_exchange.is_some() {
        return unsupported(
            "real-time propagation of a hybrid crystal is not offered (one hybrid iteration per \
             step; owner decision Q24)",
        );
    }
    if definition.nonlocal_b.is_some() {
        return unsupported("real-time propagation with rVV10 is not offered in v1 (Q28)");
    }
    if setup.charge.is_some_and(|q| q != 0.) {
        return unsupported("real-time propagation of a charged crystal is not offered");
    }
    if crystal.precision != Precision::Float64 {
        return Err(invalid(
            "real-time propagation runs in float64 only (precision='float32' is refused)",
        ));
    }
    Ok(())
}

/// The occupied bands of each spin channel of an insulating SCF: every occupation is 0 or the
/// band degeneracy (to 1e-6), with the same count at every k-point.
fn occupied_bands(crystal: &Crystal, state: &PeriodicResult) -> Result<Vec<usize>> {
    let degeneracy = crystal.degeneracy();
    let nk = crystal.points.len();
    let mut out = vec![];
    for s in 0..crystal.spins {
        let mut count = None;
        for k in 0..nk {
            let occupations = &state.occupations[s * nk + k];
            let mut here = 0;
            for f in occupations {
                if (f - degeneracy).abs() < 1e-6 {
                    here += 1;
                } else if f.abs() > 1e-6 {
                    return Err(Error::Unsupported(
                        "real-time propagation of a metal or of fractional occupations is not \
                         offered (owner decision Q25)"
                            .into(),
                    ));
                }
            }
            if count.is_some_and(|c| c != here) {
                return Err(Error::Unsupported(
                    "real-time propagation of a metal is not offered: the occupied band count \
                     changes across the mesh (owner decision Q25)"
                        .into(),
                ));
            }
            count = Some(here);
        }
        out.push(count.unwrap_or(0));
    }
    Ok(out)
}

/// The Gamma-anchored mesh the SCF's k-points were made from: the recipe's Monkhorst-Pack mesh
/// (Gamma-centered only), or an explicit whole list on uniform weights.
pub fn scf_mesh(setup: &PeriodicSetup) -> Result<[usize; 3]> {
    let recipe = setup.recipe.as_ref().ok_or_else(|| {
        Error::Unsupported(
            "real-time propagation needs the crystal's setup recipe; a snapshot without one \
             cannot rebuild its mesh"
                .into(),
        )
    })?;
    if let Some((size, gamma)) = recipe.mesh {
        if !gamma && size.iter().any(|n| n % 2 == 0) {
            return Err(invalid(
                "the SCF used a shifted Monkhorst-Pack mesh; pass kpts for the propagation mesh \
                 (Gamma-anchored, closed under k -> -k)",
            ));
        }
        return Ok(size);
    }
    let points = &setup.kpoints_frac;
    let size: [usize; 3] = std::array::from_fn(|a| {
        let mut values: Vec<f64> = points.iter().map(|k| k[a] - k[a].floor()).collect();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        values.len()
    });
    let whole = size.iter().product::<usize>() == points.len()
        && setup
            .weights
            .iter()
            .all(|w| (w - 1. / points.len() as f64).abs() < 1e-12);
    if !whole {
        return Err(invalid(
            "the SCF's explicit k-points are not a whole uniform mesh; pass kpts for the \
             propagation mesh",
        ));
    }
    Ok(size)
}

/// The whole-mesh setup and crystal of the propagation (module documentation).
fn whole_mesh(
    ctx: &crate::Context,
    setup: &PeriodicSetup,
    size: [usize; 3],
) -> Result<(PeriodicSetup, Crystal)> {
    let mut recipe = setup
        .recipe
        .clone()
        .ok_or_else(|| Error::Unsupported("a crystal without a setup recipe".into()))?;
    let mesh = MeshIndex::new(size, [0.; 3])?;
    check_inversion_closed(&mesh)?;
    recipe.mesh = None;
    recipe.symmetry = false;
    recipe.kpoints_frac = (0..mesh.points()).map(|i| mesh.point(i)).collect();
    recipe.weights = vec![1. / mesh.points() as f64; mesh.points()];
    let whole = create_setup(ctx, recipe, setup.datasets.clone())?;
    let crystal = whole.build(ctx)?;
    Ok((whole, crystal))
}

/// The setup and crystal on the explicit fractional `points` (uniform weights, no symmetry): F2's
/// mesh moved by `A`, which is not closed under `-k` by design.
fn whole_mesh_at(
    ctx: &crate::Context,
    setup: &PeriodicSetup,
    points: Vec<[f64; 3]>,
) -> Result<(PeriodicSetup, Crystal)> {
    let mut recipe = setup
        .recipe
        .clone()
        .ok_or_else(|| Error::Unsupported("a crystal without a setup recipe".into()))?;
    recipe.mesh = None;
    recipe.symmetry = false;
    recipe.weights = vec![1. / points.len() as f64; points.len()];
    recipe.kpoints_frac = points;
    let whole = create_setup(ctx, recipe, setup.datasets.clone())?;
    let crystal = whole.build(ctx)?;
    Ok((whole, crystal))
}

/// The non-orthogonal-basis density matrices `L D_o L^dagger` of every channel.
fn basis_densities(c: &Crystal, densities: &[Matrix]) -> Result<Vec<StoredMatrix>> {
    let nk = c.points.len();
    densities
        .iter()
        .enumerate()
        .map(|(idx, d)| {
            let lowdin = &c.points[idx % nk].lowdin;
            StoredMatrix::new(&c.context, lowdin * d * lowdin.adjoint(), false)
        })
        .collect()
}

/// The orbital samples `Phi` (`n x m`, column `a` the Bloch function `phi_a`) of a k-point and
/// their Bloch gradients `G_axis` (same layout), from [`super::super::fields::bloch_gradients`].
fn samples_and_gradients(c: &Crystal, point: &KPoint) -> Result<(Matrix, [Matrix; 3])> {
    let (m, n) = (c.m, c.grid.index.points());
    let mut rows = c.context.allocate::<C64>(product(&[m, n])?)?;
    let mut scratch = c.context.allocate::<C64>(n)?;
    for a in 0..m {
        let row = point.psi.slice(a * n, scratch.as_mut_slice())?;
        rows.as_mut_slice()[a * n..(a + 1) * n].copy_from_slice(row);
    }
    let gradients = bloch_gradients(&c.context, &c.grid, point.k, rows.as_slice(), m)?;
    let samples = Matrix::from_column_slice(n, m, rows.as_slice());
    let gradients = std::array::from_fn(|axis| {
        Matrix::from_column_slice(
            n,
            m,
            &gradients.as_slice()[axis * m * n..(axis + 1) * m * n],
        )
    });
    Ok((samples, gradients))
}

/// The orbital samples and Bloch gradients of every k-point for a meta-GGA run. They depend on
/// the static basis only, so they are made once and kept for the run when they fit in a quarter
/// of the free budget, and made at each use otherwise (the same numbers either way): the current
/// `j` and the operator `F^j` of every build then share one set instead of each transforming the
/// basis again.
struct Gradients {
    kept: Option<Vec<(Matrix, [Matrix; 3])>>,
    _memory: Option<crate::Reservation>,
}

impl Gradients {
    fn new(c: &Crystal, meta: bool) -> Result<Self> {
        let ctx = &c.context;
        let bytes = product(&[c.points.len(), 4, c.m, c.grid.index.points(), 16])?;
        let free = ctx.memory_limit().saturating_sub(ctx.memory_used());
        if !meta || bytes > free / 4 {
            return Ok(Self {
                kept: None,
                _memory: None,
            });
        }
        let memory = ctx.reserve(bytes)?;
        let kept = c
            .points
            .iter()
            .map(|point| samples_and_gradients(c, point))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kept: Some(kept),
            _memory: Some(memory),
        })
    }

    /// `apply(samples, gradients)` of k-point `k`.
    fn with<T>(
        &self,
        c: &Crystal,
        k: usize,
        apply: impl FnOnce(&Matrix, &[Matrix; 3]) -> Result<T>,
    ) -> Result<T> {
        match &self.kept {
            Some(kept) => apply(&kept[k].0, &kept[k].1),
            None => {
                let (samples, gradients) = samples_and_gradients(c, &c.points[k])?;
                apply(&samples, &gradients)
            }
        }
    }
}

/// The gauge-invariant kinetic density of complex Bloch orbitals (TDDFT_PLAN.md 13.6.1): the
/// paramagnetic current `j_s = Im sum_ab D_ba conj(phi_a) grad phi_b` (summed over the mesh with
/// its weights; the partial core carries none), the per-spin density `n_s` with the core, and the
/// `tau` the functional reads, `tau_s - |j_s|^2 / (2 n_s)` per spin (one block of the total for an
/// unpolarized crystal).
struct GaugeInvariant {
    tau: Buffer<f64>,
    /// `[spin][axis][point]`, per spin channel (half the total current when unpolarized).
    current: Vec<f64>,
    /// `n_s` with half the partial core, `[spin][point]`.
    density: Vec<f64>,
}

fn gauge_invariant_kinetic(
    c: &Crystal,
    matrices: &[StoredMatrix],
    rho: &[f64],
    shared: &Gradients,
) -> Result<GaugeInvariant> {
    let (n, nk) = (c.grid.index.points(), c.points.len());
    let mut tau = kinetic_density(c, matrices)?;
    let share = if c.spins == 1 { 0.5 } else { 1. };
    let mut current = vec![0f64; c.spins * 3 * n];
    for (k, point) in c.points.iter().enumerate() {
        shared.with(c, k, |samples, gradients| {
            for s in 0..c.spins {
                let d = matrices[s * nk + k].matrix(&c.context)?;
                // T_rb = sum_a conj(Phi_ra) D_ba, then X_r = sum_b T_rb G_rb.
                let t = samples.conjugate() * d.transpose();
                for (axis, g) in gradients.iter().enumerate() {
                    let out = &mut current[(s * 3 + axis) * n..(s * 3 + axis + 1) * n];
                    for (r, to) in out.iter_mut().enumerate() {
                        let x: C64 = (0..c.m).map(|b| t[(r, b)] * g[(r, b)]).sum();
                        *to += share * point.weight * x.im;
                    }
                }
            }
            Ok(())
        })?;
    }
    let mut density = vec![0f64; c.spins * n];
    for s in 0..c.spins {
        for r in 0..n {
            let n_s = share * rho[s * n + r] + 0.5 * c.core.as_slice()[r];
            density[s * n + r] = n_s;
            if n_s > 0. {
                let j2: f64 = (0..3)
                    .map(|axis| current[(s * 3 + axis) * n + r].powi(2))
                    .sum();
                // The functional reads per-spin tau as share * tau: the total loses both spins'.
                tau.as_mut_slice()[s * n + r] -= 0.5 * j2 / n_s / share;
            }
        }
    }
    Ok(GaugeInvariant {
        tau,
        current,
        density,
    })
}

impl GaugeInvariant {
    /// Adds `v_tau,s |j_s|^2 / (2 n_s^2)` to the local potential `[spin][point]` and returns the
    /// weights `(v_tau,s / n_s) j_s` of [`current_operator`], `[spin][axis * n + point]`.
    fn current_terms(&self, c: &Crystal, v_tau: &[f64], local: &mut [f64]) -> Vec<Vec<f64>> {
        let n = c.grid.index.points();
        (0..c.spins)
            .map(|s| {
                let mut weights = vec![0f64; 3 * n];
                for r in 0..n {
                    let n_s = self.density[s * n + r];
                    if n_s <= 0. {
                        continue;
                    }
                    let v = v_tau[s * n + r];
                    let mut j2 = 0.;
                    for axis in 0..3 {
                        let j = self.current[(s * 3 + axis) * n + r];
                        j2 += j * j;
                        weights[axis * n + r] = v / n_s * j;
                    }
                    local[s * n + r] += 0.5 * v * j2 / (n_s * n_s);
                }
                weights
            })
            .collect()
    }
}

/// `F^j_ab = (i/2) sum_axis int w_axis (conj(phi_a) d phi_b - conj(d phi_a) phi_b)` in the Bloch
/// basis of `point`: `(i/2) (Y - Y^dagger)` with `Y = Phi^dagger diag(w dv) G`.
fn current_operator(c: &Crystal, k: usize, weights: &[f64], shared: &Gradients) -> Result<Matrix> {
    let n = c.grid.index.points();
    let dv = c.grid.index.dv();
    shared.with(c, k, |samples, gradients| {
        let mut out = Matrix::zeros(c.m, c.m);
        for (axis, g) in gradients.iter().enumerate() {
            let mut scaled = g.clone();
            for r in 0..n {
                let w = C64::new(weights[axis * n + r] * dv, 0.);
                for b in 0..c.m {
                    scaled[(r, b)] *= w;
                }
            }
            let y = samples.adjoint() * scaled;
            out += (&y - y.adjoint()) * C64::new(0., 0.5);
        }
        Ok(out)
    })
}

/// The Fock map of the propagation: every `(spin, k)` channel's `H_o(k; A(t))`.
struct CrystalMap<'a> {
    crystal: &'a Crystal,
    request: &'a Request,
    direction: [f64; 3],
    /// `P_o(k)` along the three axes, per k-point.
    momenta: &'a [[Matrix; 3]],
    /// The ground-state `H_o` of each channel, for the independent kernel.
    frozen: Option<Vec<Matrix>>,
    /// The ground state's exchange-correlation potential (`[spin][point]`), energy and spin
    /// densities, for the Hartree kernel: `E_xc` is then linear in the density about them.
    frozen_xc: Option<(Buffer<f64>, f64, Buffer<f64>)>,
    /// Whether `A` enters the Hamiltonian (F1); F2 propagates field-free in the moved basis.
    field_on: bool,
    /// A meta-GGA: the kinetic density is the gauge-invariant `tau~` of the propagated state.
    meta: bool,
    fft: Fft3,
    workspaces: Vec<Option<crate::math::projection::ComplexProjectorWorkspace>>,
    /// A meta-GGA's orbital samples and Bloch gradients, shared by `j` and `F^j`.
    gradients: Gradients,
}

impl<'a> CrystalMap<'a> {
    fn new(
        crystal: &'a Crystal,
        momenta: &'a [[Matrix; 3]],
        request: &'a Request,
        direction: [f64; 3],
    ) -> Result<Self> {
        let ctx = &crystal.context;
        let (m, n) = (crystal.m, crystal.grid.index.points());
        let tasks = product(&[crystal.spins, crystal.points.len()])?;
        let lanes = super::super::solver::kpoint_lanes(crystal, tasks)?;
        let meta = crate::models::functional(&crystal.functional)?.requires_tau;
        Ok(Self {
            crystal,
            request,
            direction,
            momenta,
            frozen: None,
            frozen_xc: None,
            field_on: request.formulation == Formulation::Fixed,
            meta,
            fft: Fft3::new(ctx, crystal.grid.index.shape())?,
            workspaces: (0..lanes)
                .map(|_| crate::math::projection::ComplexProjectorWorkspace::new(ctx, m, n).ok())
                .collect(),
            gradients: Gradients::new(crystal, meta)?,
        })
    }

    fn field(&self, time: f64) -> [f64; 3] {
        if !self.field_on {
            return [0.; 3];
        }
        let a = self.request.strength * self.request.field.profile(time);
        self.direction.map(|e| a * e)
    }

    /// `H_o[D]` of every channel (no field) and the energy, from the orthonormal-frame densities.
    fn bare(&mut self, densities: &[Matrix]) -> Result<(Vec<Matrix>, f64)> {
        let c = self.crystal;
        let ctx = &c.context;
        let nk = c.points.len();
        let n = c.grid.index.points();
        let matrices = basis_densities(c, densities)?;
        let (rho, q) = density(c, &matrices)?;
        let meta = if self.meta {
            Some(gauge_invariant_kinetic(
                c,
                &matrices,
                rho.as_slice(),
                &self.gradients,
            )?)
        } else {
            None
        };
        let mut potential = local_potential(
            c,
            &mut self.fft,
            rho.as_slice(),
            q.as_slice(),
            meta.as_ref().map(|m| m.tau.as_slice()),
        )?;
        // The current's share of the Fock matrix (TDDFT_PLAN.md 13.6.1, vera fixes 2 and 3):
        // d tau~ / d n_s adds v_tau |j_s|^2 / (2 n_s^2) to the local potential, and the weights
        // (v_tau,s / n_s) j_s give F^j below.
        let current_weights = match (meta.as_ref(), potential.kinetic.as_ref()) {
            (Some(meta), Some(v_tau)) => {
                Some(meta.current_terms(c, v_tau.as_slice(), potential.local.as_mut_slice()))
            }
            _ => None,
        };
        // TD-Hartree: the exchange-correlation potential of this density is replaced by the
        // ground state's (the moment potentials are electrostatic only).
        if let Some((v0, _, _)) = &self.frozen_xc {
            let now = exchange_correlation(c, &mut self.fft, rho.as_slice(), None)?;
            for ((v, x), x0) in potential
                .local
                .as_mut_slice()
                .iter_mut()
                .zip(now.potential.as_slice())
                .zip(v0.as_slice())
            {
                *v += x0 - x;
            }
        }
        let (v, w) = (potential.local.as_slice(), potential.moments.as_slice());
        let kinetic = potential.kinetic.as_ref().map(|t| t.as_slice());
        // DFT+U on the complex Hermitian shell occupations of this density (their Im part is
        // what a current-carrying shell responds with), never symmetrized on the whole mesh.
        let hubbard = match c.hubbard.as_ref().filter(|h| h.is_active()) {
            Some(spec) => {
                let n = spec.occupations(c, &matrices)?;
                Some((spec.operators(c, &n)?, spec.energy(c, &n)?))
            }
            None => None,
        };
        let operators = hubbard.as_ref().map(|(o, _)| o);
        let gradients = &self.gradients;
        let focks = ctx.map_tasks(
            Level::Kpts,
            &mut self.workspaces,
            densities.len(),
            |ws, idx| {
                let (s, k) = (idx / nk, idx % nk);
                let point = &c.points[k];
                let h = hamiltonian(
                    c,
                    point,
                    &v[s * n..(s + 1) * n],
                    w,
                    kinetic.map(|t| &t[s * n..(s + 1) * n]),
                    operators.map(|o| &o[s]),
                    None,
                    ws.as_mut(),
                )?;
                let h = match &current_weights {
                    Some(weights) => h + current_operator(c, k, &weights[s], gradients)?,
                    None => h,
                };
                let h = point.lowdin.adjoint() * h * &point.lowdin;
                Ok((&h + h.adjoint()) * C64::new(0.5, 0.))
            },
        )?;
        let mut terms = energy(
            c,
            &mut self.fft,
            &matrices,
            rho.as_slice(),
            q.as_slice(),
            meta.as_ref().map(|m| m.tau.as_slice()),
        )?;
        if let Some((v0, e0, rho0)) = &self.frozen_xc {
            let dv = c.grid.index.dv();
            terms[2] = e0
                + dv * v0
                    .as_slice()
                    .iter()
                    .zip(rho.as_slice())
                    .zip(rho0.as_slice())
                    .map(|((v, r), r0)| v * (r - r0))
                    .sum::<f64>();
        }
        let hubbard_energy = hubbard.map_or(0., |(_, e)| e);
        Ok((focks, terms.iter().sum::<f64>() + hubbard_energy))
    }

    /// `sum_{s,k} w_k tr[D_o P_o]` along the three axes (the paramagnetic momentum).
    fn momentum(&self, densities: &[Matrix]) -> [f64; 3] {
        let nk = self.crystal.points.len();
        let mut out = [0.; 3];
        for (idx, d) in densities.iter().enumerate() {
            let k = idx % nk;
            let weight = self.crystal.points[k].weight;
            for (axis, value) in out.iter_mut().enumerate() {
                *value += weight * (d * &self.momenta[k][axis]).trace().re;
            }
        }
        out
    }
}

impl FockMap for CrystalMap<'_> {
    fn evaluate(&mut self, time: f64, densities: &[Matrix]) -> Result<Evaluation> {
        let nk = self.crystal.points.len();
        let (mut matrices, mut energy) = match &self.frozen {
            Some(frozen) => {
                let energy = densities
                    .iter()
                    .zip(frozen)
                    .enumerate()
                    .map(|(idx, (d, h))| self.crystal.points[idx % nk].weight * (d * h).trace().re)
                    .sum();
                (frozen.clone(), energy)
            }
            None => self.bare(densities)?,
        };
        let a = self.field(time);
        let squared: f64 = a.iter().map(|v| v * v).sum();
        let paramagnetic = self.momentum(densities);
        energy += (0..3).map(|axis| a[axis] * paramagnetic[axis]).sum::<f64>()
            + 0.5 * squared * self.crystal.electrons;
        for (idx, h) in matrices.iter_mut().enumerate() {
            let p = &self.momenta[idx % nk];
            for axis in 0..3 {
                if a[axis] != 0. {
                    *h += &p[axis] * C64::new(a[axis], 0.);
                }
            }
            for i in 0..h.nrows() {
                h[(i, i)] += C64::new(0.5 * squared, 0.);
            }
        }
        Ok(Evaluation { matrices, energy })
    }
}

/// The state a run starts from, on the whole mesh (module documentation): everything the
/// propagation reads, exposed so the frozen-Hamiltonian identity (R8) can be checked against an
/// exact spectral evolution.
pub struct Prepared {
    pub setup: PeriodicSetup,
    pub crystal: Crystal,
    pub mesh: [usize; 3],
    pub direction: [f64; 3],
    /// `P_o(k)` per k-point along the three axes, Hermitian, in the Loewdin frame.
    pub momenta: Vec<[Matrix; 3]>,
    /// The occupied orbitals of every `(spin, k)` channel (index `s * nk + k`), the lowest bands of
    /// `ground[idx]`.
    pub channels: Vec<Channel>,
    /// `H_o[D_0]` of every channel without the field, and the energy of `D_0`.
    pub ground: Vec<Matrix>,
    pub ground_energy: f64,
    /// `W` of the ground-state matrices.
    pub width: f64,
    /// The largest change of the occupied projector from the SCF potential's bands to the start's.
    pub stationarity: f64,
    /// The largest difference of the start's density from the SCF's, per grid point (R15).
    pub initial_density_change: f64,
    pub momentum_residual: f64,
    pub formulation: Formulation,
    /// F2: the paramagnetic momentum `sum w tr[D_g P(k + A)]` of the moved basis's own ground
    /// state (a static current the basis carries, removed from `corrected`), and the largest
    /// departure of the transferred occupied overlap from the identity before it was
    /// orthonormalized (the basis's incompleteness at first order in `A`). F1: zero.
    pub offset: [f64; 3],
    pub transfer_defect: f64,
}

/// [`Prepared`] for `request` (its mesh, direction and functional checks).
pub fn prepare(sampler: &Sampler<'_>, request: &Request) -> Result<Prepared> {
    check_crystal(sampler.crystal, sampler.setup)?;
    let direction = unit(request.direction)?;
    if !(request.strength.is_finite() && request.strength != 0.) {
        return Err(invalid("the field strength must be finite and nonzero"));
    }
    if !(request.time_step.is_finite() && request.time_step > 0.) || request.steps == 0 {
        return Err(invalid(
            "real-time propagation needs a positive time step and at least one step",
        ));
    }
    if let Field::Ramp { duration } = request.field {
        if !(duration.is_finite() && duration > 0.) {
            return Err(invalid("a ramp needs a positive finite duration"));
        }
    }
    let occupied = occupied_bands(sampler.crystal, sampler.state)?;
    let size = match request.mesh {
        Some(size) => size,
        None => scf_mesh(sampler.setup)?,
    };
    if sampler
        .crystal
        .hubbard
        .as_ref()
        .is_some_and(|h| h.is_active())
    {
        if request.formulation == Formulation::Moved {
            return Err(Error::Unsupported(
                "formulation='moved' with DFT+U is not offered: its mesh at k + A is not closed \
                 under -k, which the shell occupations' completion needs"
                    .into(),
            ));
        }
        if request.kernel == Kernel::Hartree {
            return Err(Error::Unsupported(
                "kernel='hartree' with DFT+U is not offered (the +U term is not frozen with the \
                 exchange-correlation potential); use kernel='adiabatic' or 'independent'"
                    .into(),
            ));
        }
    }
    let meta = crate::models::functional(&sampler.crystal.functional)?.requires_tau;
    if meta && request.formulation == Formulation::Moved {
        return Err(Error::Unsupported(
            "formulation='moved' with a meta-GGA is not offered: its static current is taken from \
             the bands, which the current term of tau~ moves (the ramp measures it; F1 only)"
                .into(),
        ));
    }
    if request.kernel == Kernel::Hartree && meta {
        return Err(Error::Unsupported(
            "kernel='hartree' with a meta-GGA is not offered (its kinetic potential is not frozen \
             with the exchange-correlation potential); use kernel='adiabatic' or 'independent'"
                .into(),
        ));
    }
    if request.formulation == Formulation::Moved && request.field != Field::Kick {
        return Err(invalid(
            "formulation='moved' takes a kick (a constant A); a time-dependent A needs \
             formulation='fixed'",
        ));
    }
    sampler.crystal.context.run(|| {
        let fixed = prepare_on(sampler, request, direction, &occupied, size)?;
        match request.formulation {
            Formulation::Fixed => Ok(fixed),
            Formulation::Moved => prepare_moved(sampler, request, &occupied, fixed),
        }
    })
}

/// F2 from the F1 start `fixed` (its mesh, its ground energy): the whole mesh moved by `A`, the
/// occupied bands transferred by `exp(i A . r)` (`<psi_n(k + A)| exp(i A . r) |psi_v(k)>` from the
/// periodic-part overlaps of `Sampler::overlap`) and orthonormalized, the momentum at `k + A`.
fn prepare_moved(
    sampler: &Sampler<'_>,
    request: &Request,
    occupied: &[usize],
    fixed: Prepared,
) -> Result<Prepared> {
    let ctx = &sampler.crystal.context;
    let state = sampler.state;
    let a: [f64; 3] = fixed.direction.map(|e| request.strength * e);
    let shift = sampler
        .reciprocal
        .try_inverse()
        .ok_or_else(|| invalid("singular reciprocal lattice"))?
        * nalgebra::Vector3::from(a);
    let points: Vec<[f64; 3]> = fixed
        .crystal
        .points
        .iter()
        .map(|p| std::array::from_fn(|i| p.fractional[i] + shift[i]))
        .collect();
    let (setup, c) = whole_mesh_at(ctx, sampler.setup, points)?;
    let (k_crystal, nk) = (&fixed.crystal, fixed.crystal.points.len());
    let degeneracy = c.degeneracy();
    let bands: Vec<usize> = (0..c.bands()).collect();
    let one_center = one_center_momentum(ctx, &setup)?;
    let grid_mesh = Mesh::new(ctx, &setup.grid)?;
    let mut momentum_residual = fixed.momentum_residual;
    let mut momenta = Vec::with_capacity(nk);
    for point in &c.points {
        let basis = basis_momenta(ctx, &setup, &grid_mesh, point, &one_center)?;
        momentum_residual = momentum_residual.max(basis.residual);
        momenta.push(basis.matrices.map(|p| {
            let p = point.lowdin.adjoint() * p * &point.lowdin;
            (&p + p.adjoint()) * C64::new(0.5, 0.)
        }));
    }
    drop(grid_mesh);
    let mut channels = Vec::with_capacity(fixed.channels.len());
    let mut offset = [0.; 3];
    let mut transfer_defect = 0f64;
    for s in 0..c.spins {
        let o = occupied[s];
        for k in 0..nk {
            let here = Frame::from_point(k_crystal, state, &k_crystal.points[k], s, &bands)?;
            let point = &c.points[k];
            let there = Frame::from_point(&c, state, point, s, &bands)?;
            // <psi_m(k)| exp(-i A . r) |psi_n(k + A)>, so the transferred band v has the amplitude
            // conj(M_vn) on psi_n(k + A).
            let overlap = sampler.overlap(&here, &there, a)?;
            let frame = point.lowdin.adjoint() * &point.overlap * &there.coefficients;
            let amplitudes = Matrix::from_fn(bands.len(), o, |n, v| overlap[(v, n)].conj());
            let moved = &frame * amplitudes;
            let gram = moved.adjoint() * &moved;
            let (values, vectors) = eigen(gram.clone())?;
            for value in &values {
                transfer_defect = transfer_defect.max((1. - value).abs());
            }
            if values[0] <= 0. {
                return Err(Error::Kernel(
                    "the occupied bands lost their rank in the moved basis".into(),
                ));
            }
            let inverse_root = &vectors
                * Matrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    o,
                    values.iter().map(|v| C64::new(1. / v.sqrt(), 0.)),
                ))
                * vectors.adjoint();
            channels.push(Channel {
                orbitals: moved * inverse_root,
                occupations: vec![degeneracy; o],
            });
            // The moved basis's own ground state at k + A and its momentum.
            let ground = frame.columns(0, o);
            let density = ground * ground.adjoint() * C64::new(degeneracy, 0.);
            for (axis, value) in offset.iter_mut().enumerate() {
                *value += point.weight * (&density * &momenta[k][axis]).trace().re;
            }
        }
    }
    let (ground, width) = {
        let mut map = CrystalMap::new(&c, &momenta, request, fixed.direction)?;
        let densities: Vec<Matrix> = channels.iter().map(Channel::density).collect();
        let (ground, _) = map.bare(&densities)?;
        let mut width = 0f64;
        for h in &ground {
            let (levels, _) = eigen(h.clone())?;
            width = width.max(levels[levels.len() - 1] - levels[0]);
        }
        (ground, width)
    };
    Ok(Prepared {
        setup,
        crystal: c,
        mesh: fixed.mesh,
        direction: fixed.direction,
        momenta,
        channels,
        ground,
        ground_energy: fixed.ground_energy,
        width,
        stationarity: fixed.stationarity,
        initial_density_change: fixed.initial_density_change,
        momentum_residual,
        formulation: Formulation::Moved,
        offset,
        transfer_defect,
    })
}

fn prepare_on(
    sampler: &Sampler<'_>,
    request: &Request,
    direction: [f64; 3],
    occupied: &[usize],
    size: [usize; 3],
) -> Result<Prepared> {
    let (scf, state) = (sampler.crystal, sampler.state);
    let ctx = &scf.context;
    let (setup, c) = whole_mesh(ctx, sampler.setup, size)?;
    let nk = c.points.len();
    let n = c.grid.index.points();
    let m = c.m;
    let tasks = product(&[c.spins, nk])?;
    let degeneracy = c.degeneracy();
    if occupied.iter().any(|&o| o == 0 || o >= m) {
        return Err(invalid(
            "every spin channel of the crystal needs occupied and empty bands",
        ));
    }
    // The momentum of the Bloch basis, in the Loewdin frame of each k-point.
    let one_center = one_center_momentum(ctx, &setup)?;
    let grid_mesh = Mesh::new(ctx, &setup.grid)?;
    let mut momentum_residual = 0f64;
    let mut momenta = Vec::with_capacity(nk);
    for point in &c.points {
        let basis = basis_momenta(ctx, &setup, &grid_mesh, point, &one_center)?;
        momentum_residual = momentum_residual.max(basis.residual);
        momenta.push(basis.matrices.map(|p| {
            let p = point.lowdin.adjoint() * p * &point.lowdin;
            (&p + p.adjoint()) * C64::new(0.5, 0.)
        }));
    }
    drop(grid_mesh);
    // The SCF's converged potential at every point of the whole mesh.
    let potential = state.potential.as_slice();
    let mut densities = Vec::with_capacity(tasks);
    for s in 0..c.spins {
        for point in &c.points {
            let (_, _, y) = diagonalize_with_exchange(
                &c,
                point,
                &potential[s * n..(s + 1) * n],
                state.moment_potential.as_slice(),
                state
                    .kinetic_potential
                    .as_ref()
                    .map(|v| &v.as_slice()[s * n..(s + 1) * n]),
                state.hubbard.as_ref().map(|h| &h.operators[s]),
                None,
            )?;
            let occ = y.columns(0, occupied[s]);
            densities.push(occ * occ.adjoint() * C64::new(degeneracy, 0.));
        }
    }
    let (channels, ground, ground_energy, width, stationarity) = {
        let mut map = CrystalMap::new(&c, &momenta, request, direction)?;
        // The start is the lowest bands of H_o[D_0]: a state that is not its own fixed point
        // would move without a field.
        let (focks, _) = map.bare(&densities)?;
        let mut channels = Vec::with_capacity(tasks);
        let mut width = 0f64;
        let mut stationarity = 0f64;
        for (idx, fock) in focks.iter().enumerate() {
            let o = occupied[idx / nk];
            let (levels, vectors) = eigen(fock.clone())?;
            if levels[o] - levels[o - 1] <= 1e-6 {
                return Err(Error::Unsupported(format!(
                    "the crystal has no gap at k-point {} of the propagation mesh (a metal; \
                     owner decision Q25)",
                    idx % nk
                )));
            }
            width = width.max(levels[m - 1] - levels[0]);
            let occ = vectors.columns(0, o).into_owned();
            let start = &occ * occ.adjoint() * C64::new(degeneracy, 0.);
            stationarity = stationarity.max(
                (&start - &densities[idx])
                    .iter()
                    .map(|v| v.norm())
                    .fold(0., f64::max)
                    / degeneracy,
            );
            channels.push(Channel {
                orbitals: occ,
                occupations: vec![degeneracy; o],
            });
        }
        let densities: Vec<Matrix> = channels.iter().map(Channel::density).collect();
        let (ground, ground_energy) = map.bare(&densities)?;
        (channels, ground, ground_energy, width, stationarity)
    };
    if stationarity > 1e-5 {
        return Err(invalid(format!(
            "the crystal's SCF is not stationary on the propagation mesh: the occupied bands of \
             its own Hamiltonian differ from the SCF potential's by {stationarity:.2e} (above \
             1e-5); converge the SCF tighter"
        )));
    }
    let densities: Vec<Matrix> = channels.iter().map(Channel::density).collect();
    let (rho, _) = density(&c, &basis_densities(&c, &densities)?)?;
    let initial_density_change = rho
        .as_slice()
        .iter()
        .zip(state.density.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    Ok(Prepared {
        setup,
        crystal: c,
        mesh: size,
        direction,
        momenta,
        channels,
        ground,
        ground_energy,
        width,
        stationarity,
        initial_density_change,
        momentum_residual,
        formulation: Formulation::Fixed,
        offset: [0.; 3],
        transfer_defect: 0.,
    })
}

/// Propagates the crystal `sampler` holds after the field of `request` (module documentation).
pub fn propagate(sampler: &Sampler<'_>, request: &Request) -> Result<CrystalRealTime> {
    let prepared = prepare(sampler, request)?;
    sampler
        .crystal
        .context
        .run(|| prepared.propagate(sampler, request))
}

/// The three runs of `request` with the field along x, y and z (`direction="all"`): column `b`
/// of the tensor is the run along axis `b`, `corrected_current[t][a]` of it the element `ab`.
/// F1 shares one start (its state does not depend on the direction); F2 moves the mesh by each
/// field.
pub fn propagate_axes(sampler: &Sampler<'_>, request: &Request) -> Result<Vec<CrystalRealTime>> {
    let along = |axis: usize| {
        let mut one = request.clone();
        one.direction = std::array::from_fn(|i| if i == axis { 1. } else { 0. });
        one
    };
    if request.formulation == Formulation::Moved {
        return (0..3)
            .map(|axis| propagate(sampler, &along(axis)))
            .collect();
    }
    let mut prepared = prepare(sampler, &along(0))?;
    let mut runs = Vec::with_capacity(3);
    for axis in 0..3 {
        let one = along(axis);
        prepared.direction = one.direction;
        runs.push(
            sampler
                .crystal
                .context
                .run(|| prepared.propagate(sampler, &one))?,
        );
    }
    Ok(runs)
}

/// The adiabatic ramp of [`Prepared::ramp_response`], the cross-check of the static response
/// (R9b): `A` rises over `RAMP` a.u. (sin^2), is held for `SETTLE` a.u., and the current is
/// averaged over the next `AVERAGE` a.u. (the R9b gate's window: 0.23 % on LiH; a 20 a.u. window
/// left 1.1 %).
const RAMP: f64 = 100.;
const SETTLE: f64 = 20.;
const AVERAGE: f64 = 80.;
/// The largest density-matrix change at which the static self-consistency stops.
pub const STATIC_TOLERANCE: f64 = 1e-12;
/// The iterations the static self-consistency may take at each sign of the field.
pub const STATIC_ITERATIONS: usize = 300;

/// The static paramagnetic response of [`Prepared::static_response`].
#[derive(Clone, Copy, Debug)]
pub struct StaticResponse {
    /// `N_dia e` (Cartesian).
    pub vector: [f64; 3],
    /// The iterations the slower of the two signs took.
    pub iterations: usize,
}

impl Prepared {
    /// The run of `request` from this start; `sampler` gives the band sum on the same mesh. A
    /// DFT+U or meta-GGA crystal with the adiabatic kernel first measures its static response
    /// `N_dia e` by self-consistency at `+-A_0` ([`Self::static_response`]; module
    /// documentation; owner decision Q21a).
    pub fn propagate(&self, sampler: &Sampler<'_>, request: &Request) -> Result<CrystalRealTime> {
        // The kernels with an Im D channel: DFT+U's shell occupations and the meta-GGA current
        // term of tau~ (TDDFT_PLAN.md 13.5.3: the band sum is exact for a density-only kernel).
        let channel = self.crystal.hubbard.as_ref().is_some_and(|h| h.is_active())
            || crate::models::functional(&self.crystal.functional)?.requires_tau;
        let ramp = channel
            && request.kernel == Kernel::Adiabatic
            && self.formulation == Formulation::Fixed;
        if let Some(given) = request.statics {
            return self.trace(sampler, request, Some(given));
        }
        if !ramp {
            return self.trace(sampler, request, None);
        }
        let measured = self.static_response(request)?.vector;
        let mut run = self.trace(sampler, request, Some(measured))?;
        run.self_consistent = true;
        Ok(run)
    }

    /// The static paramagnetic response `N_dia e` along `request`'s direction by self-consistency
    /// at a fixed field (vera's oracle, R9b): at `A = +-A_0 e` the occupied bands of
    /// `H_o[D] + A . P_o` of every channel are iterated to a fixed point, and
    /// `N_dia e = -(t(+A_0) - t(-A_0)) / (2 A_0)` with `t_axis = sum_{s,k} w_k tr[D P_o,axis]` (the
    /// static current is `-A_0 (N e - N_dia e) / Omega`). Exact for any kernel at the order the
    /// symmetric pair leaves (`A_0^2`), and the state the adiabatic ramp tends to. The iteration
    /// takes the new density whole; when the change grows three times in a row the density is
    /// mixed half and half from then on. It raises when the change is still above
    /// [`STATIC_TOLERANCE`] after [`STATIC_ITERATIONS`].
    pub fn static_response(&self, request: &Request) -> Result<StaticResponse> {
        if self.formulation == Formulation::Moved {
            return Err(invalid(
                "formulation='moved' takes a kick (a constant A); a static field needs \
                 formulation='fixed'",
            ));
        }
        let mut statics = request.clone();
        statics.kernel = Kernel::Adiabatic;
        let mut map = CrystalMap::new(&self.crystal, &self.momenta, &statics, self.direction)?;
        let nk = self.crystal.points.len();
        let lowest = |focks: &[Matrix], a: f64| -> Result<Vec<Matrix>> {
            focks
                .iter()
                .enumerate()
                .map(|(idx, f)| {
                    let mut h = f.clone();
                    for axis in 0..3 {
                        h += &self.momenta[idx % nk][axis] * C64::new(a * self.direction[axis], 0.);
                    }
                    let (_, vectors) = eigen((&h + h.adjoint()) * C64::new(0.5, 0.))?;
                    let channel = &self.channels[idx];
                    Ok(Channel {
                        orbitals: vectors.columns(0, channel.orbitals.ncols()).into_owned(),
                        occupations: channel.occupations.clone(),
                    }
                    .density())
                })
                .collect()
        };
        let trace = |densities: &[Matrix]| -> [f64; 3] {
            std::array::from_fn(|axis| {
                densities
                    .iter()
                    .enumerate()
                    .map(|(idx, d)| {
                        self.crystal.points[idx % nk].weight
                            * (d * &self.momenta[idx % nk][axis]).trace().re
                    })
                    .sum()
            })
        };
        let mut traces = [[0f64; 3]; 2];
        let mut iterations = 0usize;
        for (side, sign) in [1f64, -1.].into_iter().enumerate() {
            let a = sign * request.strength;
            let mut d = lowest(&self.ground, a)?;
            let (mut previous, mut rising, mut mixing) = (f64::INFINITY, 0usize, 1f64);
            let mut converged = false;
            for iteration in 1..=STATIC_ITERATIONS {
                let (focks, _) = map.bare(&d)?;
                let next = lowest(&focks, a)?;
                let change = next
                    .iter()
                    .zip(&d)
                    .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).norm()))
                    .fold(0f64, f64::max);
                rising = if change > previous { rising + 1 } else { 0 };
                if rising >= 3 {
                    mixing = 0.5;
                }
                previous = change;
                d = next
                    .iter()
                    .zip(&d)
                    .map(|(x, y)| x * C64::new(mixing, 0.) + y * C64::new(1. - mixing, 0.))
                    .collect();
                iterations = iterations.max(iteration);
                if change < STATIC_TOLERANCE {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err(Error::Kernel(format!(
                    "the static response at a fixed field did not converge in \
                     {STATIC_ITERATIONS} iterations (last density change {previous:.3e})"
                )));
            }
            traces[side] = trace(&d);
        }
        Ok(StaticResponse {
            vector: std::array::from_fn(|axis| {
                -(traces[0][axis] - traces[1][axis]) / (2. * request.strength)
            }),
            iterations,
        })
    }

    /// The static paramagnetic response `N_dia e` measured by two adiabatic ramps of `request`'s
    /// strength to `+-A_0` along its direction (R9b), the cross-check of
    /// [`Self::static_response`]: `A` rises over `RAMP` a.u. (sin^2) and is held, and the odd
    /// current averaged over `AVERAGE` a.u. after `SETTLE` more is `-A_0 (N e - N_dia e) / Omega`.
    /// Exact for any kernel in the adiabatic limit, up to the ramp's own error (2e-4 relative on
    /// LiH r2SCAN, 0.23 % on LiH LDA, where the band sum is exact).
    pub fn ramp_response(&self, sampler: &Sampler<'_>, request: &Request) -> Result<[f64; 3]> {
        if self.formulation == Formulation::Moved {
            return Err(invalid(
                "formulation='moved' takes a kick (a constant A); a ramp needs formulation='fixed'",
            ));
        }
        let mut up = request.clone();
        up.field = Field::Ramp { duration: RAMP };
        up.steps = ((RAMP + SETTLE + AVERAGE) / request.time_step).ceil() as usize;
        let mut down = up.clone();
        down.strength = -up.strength;
        let (plus, minus) = (
            self.trace(sampler, &up, None)?,
            self.trace(sampler, &down, None)?,
        );
        let mut late = [0f64; 3];
        let mut count = 0usize;
        for ((t, a), b) in plus.times.iter().zip(&plus.current).zip(&minus.current) {
            if *t >= RAMP + SETTLE {
                count += 1;
                for axis in 0..3 {
                    late[axis] += 0.5 * (a[axis] - b[axis]);
                }
            }
        }
        Ok(std::array::from_fn(|axis| {
            plus.electrons * self.direction[axis]
                + plus.volume * late[axis] / (count as f64 * request.strength)
        }))
    }

    /// The run of `request` with the static response `N_dia e` given (`None`: the band sum).
    fn trace(
        &self,
        sampler: &Sampler<'_>,
        request: &Request,
        measured: Option<[f64; 3]>,
    ) -> Result<CrystalRealTime> {
        let c = &self.crystal;
        let ctx = &c.context;
        let direction = self.direction;
        let size = self.mesh;
        let nk = c.points.len();
        let ground_energy = self.ground_energy;
        let mut momentum_residual = self.momentum_residual;
        let mut map = CrystalMap::new(c, &self.momenta, request, direction)?;
        match request.kernel {
            Kernel::Independent => map.frozen = Some(self.ground.clone()),
            Kernel::Hartree => {
                let densities: Vec<Matrix> = self.channels.iter().map(Channel::density).collect();
                let (rho0, _) = density(c, &basis_densities(c, &densities)?)?;
                let xc0 = exchange_correlation(c, &mut map.fft, rho0.as_slice(), None)?;
                let e0 = collinear_xc_energy(c, &mut map.fft, rho0.as_slice(), c.spins)?;
                map.frozen_xc = Some((xc0.potential, e0, rho0));
            }
            Kernel::Adiabatic => {}
        }
        // The field adds A . P_o to the width the stability bound reads.
        let largest_momentum = self
            .momenta
            .iter()
            .flat_map(|p| p.iter())
            .map(|p| eigen(p.clone()).map(|(e, _)| e[e.len() - 1] - e[0]))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0., f64::max);
        let width = self.width + request.strength.abs() * largest_momentum;
        let limit = request.scheme.step_width_bound();
        if request.time_step * width > limit {
            return Err(invalid(format!(
            "time_step = {} gives dt W = {:.3} with the basis width W = {width:.3} Hartree, above \
             the stability bound {limit} of propagator='{}'; lower time_step",
            request.time_step,
            request.time_step * width,
            request.scheme.name()
        )));
        }
        // The band sum along e on the same whole mesh, from the SCF's own frames.
        let lines = transitions(sampler, &MeshIndex::new(size, [0.; 3])?, direction)?;
        momentum_residual = momentum_residual.max(lines.residual);
        let band_sum = lines.diamagnetic(sampler.channels());
        let nyquist = std::f64::consts::PI / request.time_step;
        let aliased: f64 = lines
            .gaps
            .iter()
            .zip(&lines.weights)
            .filter(|(gap, _)| **gap > nyquist)
            .map(|(gap, w)| 2. * lines.degeneracy * w / gap)
            .sum();
        let aliased_weight = if band_sum.total > 0. {
            aliased / band_sum.total
        } else {
            0.
        };
        let volume = lines.volume;
        let electrons = c.electrons;
        let statics = measured.unwrap_or(band_sum.vector);
        let diamagnetic: f64 = (0..3).map(|axis| statics[axis] * direction[axis]).sum();
        let deficit = electrons - diamagnetic;
        let offset_along: f64 = (0..3).map(|axis| self.offset[axis] * direction[axis]).sum();
        let initial_density_change = self.initial_density_change;
        let channels = self.channels.clone();

        let mut options = Options::new(request.scheme, request.time_step);
        options.corrector_tolerance = request.effective_corrector_tolerance();
        options.max_corrector_iterations = request.max_corrector_iterations;
        let mut propagator = Propagator::new(ctx, map, channels, 0., options)?;
        let observe = |propagator: &Propagator<CrystalMap<'_>>, time: f64| {
            let map = propagator.map();
            let a = map.field(time);
            let paramagnetic = map.momentum(&propagator.densities());
            let j: [f64; 3] =
                std::array::from_fn(|axis| -(paramagnetic[axis] + electrons * a[axis]) / volume);
            let along: f64 = (0..3).map(|axis| j[axis] * direction[axis]).sum();
            let potential: f64 = (0..3).map(|axis| a[axis] * direction[axis]).sum();
            let corrected = match self.formulation {
                Formulation::Fixed => along + potential * deficit / volume,
                Formulation::Moved => along + offset_along / volume,
            };
            let vector: [f64; 3] = std::array::from_fn(|axis| match self.formulation {
                Formulation::Fixed => {
                    j[axis] + potential * (electrons * direction[axis] - statics[axis]) / volume
                }
                Formulation::Moved => j[axis] + self.offset[axis] / volume,
            });
            let potential = match self.formulation {
                Formulation::Fixed => potential,
                Formulation::Moved => request.strength * request.field.profile(time),
            };
            (j, along, corrected, vector, potential)
        };
        let initial = propagator.energy();
        let kick_energy = initial - ground_energy;
        let worst = |p: &Propagator<CrystalMap<'_>>| {
            p.channels()
                .iter()
                .map(Channel::norm_error)
                .fold(0., f64::max)
        };
        let (j, along, corrected, vector, potential) = observe(&propagator, 0.);
        let mut out = CrystalRealTime {
            times: vec![0.],
            current: vec![j],
            along: vec![along],
            corrected: vec![corrected],
            corrected_current: vec![vector],
            vector_potential: vec![potential],
            energy: vec![initial],
            norm_error: vec![worst(&propagator)],
            ground_energy,
            kick_energy,
            energy_drift: 0.,
            width,
            largest_width: 0.,
            step_width: request.time_step * width,
            worst_norm_error: 0.,
            corrector_tolerance: request.effective_corrector_tolerance(),
            aliased_weight,
            builds: 0,
            corrector_iterations: 0,
            electrons,
            diamagnetic,
            deficit,
            diamagnetic_vector: statics,
            self_consistent: false,
            band_sum: band_sum.total,
            momentum_residual,
            volume,
            mesh: size,
            kpoints: nk,
            initial_density_change,
            formulation: self.formulation,
            static_current: -offset_along / volume,
            transfer_defect: self.transfer_defect,
        };
        let checked = request.field == Field::Kick;
        for _ in 0..request.steps {
            let step = propagator.step()?;
            out.largest_width = out.largest_width.max(step.width);
            out.corrector_iterations = out.corrector_iterations.max(step.corrector_iterations);
            let (j, along, corrected, vector, potential) = observe(&propagator, step.time);
            out.times.push(step.time);
            out.current.push(j);
            out.along.push(along);
            out.corrected.push(corrected);
            out.corrected_current.push(vector);
            out.vector_potential.push(potential);
            out.energy.push(step.energy);
            out.norm_error.push(step.norm_error);
            if checked && kick_energy > 0. {
                out.energy_drift = out
                    .energy_drift
                    .max((step.energy - initial).abs() / kick_energy);
                if let Some(limit) = request.max_energy_drift {
                    if out.energy_drift > limit {
                        return Err(Error::Kernel(format!(
                            "the energy drifted by {:.3e} kick energies by t = {} (above \
                         max_energy_drift = {limit}); lower time_step",
                            out.energy_drift, step.time
                        )));
                    }
                }
            }
        }
        out.worst_norm_error = out.norm_error.iter().fold(0f64, |t, e| t.max(*e));
        // Two bare builds before the propagator's own.
        out.builds = propagator.builds() + 2;
        Ok(out)
    }
}
