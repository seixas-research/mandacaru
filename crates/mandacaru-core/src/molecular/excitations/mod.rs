//! Linear-response excitations of a molecule: TDDFT (adiabatic LDA, PBE and HSE06) and, on a
//! Hartree-Fock reference, TDHF and CIS; closed shells spin-adapted (singlets and triplets),
//! unrestricted (open-shell) references in the spin-conserving spin-orbital form.
//!
//! The Tamm-Dancoff approximation solves `A X = omega X` ([`math::davidson::lowest`]); the full
//! (Casida) form solves `(A - B)(A + B)(X + Y) = omega^2 (X + Y)`
//! ([`math::davidson::lowest_paired`]). Both run on one operator, [`Response`], whose kernel is
//! the exact second derivative of the reference's energy (Casida, in *Recent Advances in Density
//! Functional Methods*, World Scientific (1995); Hirata and Head-Gordon, Chem. Phys. Lett. 314,
//! 291 (1999)). Singlets and triplets of a closed shell are separate spin-adapted solves; an
//! unrestricted reference runs [`Unrestricted`], whose vector holds both spins' transitions.
//!
//! Atomic units throughout: energies in Hartree, dipoles in e Bohr.
//!
//! [`math::davidson::lowest`]: crate::math::davidson::lowest
//! [`math::davidson::lowest_paired`]: crate::math::davidson::lowest_paired
mod properties;
mod response;
mod unrestricted;

pub use response::Response;
pub use unrestricted::Unrestricted;

use super::{invalid, MeanField, MolecularIntegrals};
use crate::{
    math::davidson::{self, Options, Paired, Symmetric},
    Result,
};
use nalgebra::DMatrix;

/// Roots closer than this (Hartree) belong to one multiplet: a request that would cut one is
/// extended to the whole multiplet, whose summed strengths do not depend on the frame the
/// solver returns inside it.
pub const DEGENERATE_ROOTS: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approximation {
    TammDancoff,
    Casida,
}
impl Approximation {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "tda" => Ok(Self::TammDancoff),
            "casida" => Ok(Self::Casida),
            _ => Err(invalid(format!(
                "unknown approximation {name:?}; expected 'tda' or 'casida'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::TammDancoff => "tda",
            Self::Casida => "casida",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Singlet,
    Triplet,
    /// Both spins of an unrestricted reference, spin-conserving (not a user choice: what an
    /// unrestricted reference gives).
    Unrestricted,
}
impl Spin {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "singlet" => Ok(Self::Singlet),
            "triplet" => Ok(Self::Triplet),
            _ => Err(invalid(format!(
                "unknown spin {name:?}; expected 'singlet' or 'triplet'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Singlet => "singlet",
            Self::Triplet => "triplet",
            Self::Unrestricted => "unrestricted",
        }
    }
}
/// Which part of the reference's kernel acts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coupling {
    /// The reference's whole kernel: Hartree and adiabatic XC (Kohn-Sham), Coulomb and exchange
    /// (Hartree-Fock).
    Adiabatic,
    /// The Coulomb (Hartree) kernel only: the random-phase approximation.
    Hartree,
    /// No kernel: the bare orbital-energy differences.
    Independent,
}
impl Coupling {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "adiabatic" => Ok(Self::Adiabatic),
            "hartree" => Ok(Self::Hartree),
            "independent" => Ok(Self::Independent),
            _ => Err(invalid(format!(
                "unknown kernel {name:?}; expected 'adiabatic', 'hartree' or 'independent'"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Adiabatic => "adiabatic",
            Self::Hartree => "hartree",
            Self::Independent => "independent",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub num_states: usize,
    pub approximation: Approximation,
    /// `Singlet` or `Triplet` for a closed shell (`None` is `Singlet`); `None` for an
    /// unrestricted reference, which raises for a spin symmetry.
    pub spin: Option<Spin>,
    pub coupling: Coupling,
    /// The largest residual norm of a root.
    pub tolerance: f64,
    pub max_iterations: usize,
}
impl Request {
    pub fn new(num_states: usize) -> Self {
        Self {
            num_states,
            approximation: Approximation::TammDancoff,
            spin: None,
            coupling: Coupling::Adiabatic,
            tolerance: 1e-5,
            max_iterations: 200,
        }
    }
}

/// One dominant orbital transition of a root: its spin channel (0 for a closed shell; 0 alpha and
/// 1 beta for an unrestricted reference), occupied and empty orbital indices (in the channel's
/// ascending order) and its weight (`X^2` for TDA, `X^2 - Y^2` for Casida).
#[derive(Clone, Debug)]
pub struct Transition {
    pub channel: usize,
    pub occupied: usize,
    pub virtual_orbital: usize,
    pub weight: f64,
}

/// One spin channel of the reference: its occupied and empty orbital counts, orbital energies
/// (Hartree, ascending) and orbitals in the nonorthogonal atomic basis (columns).
pub struct Channel {
    pub occupied: usize,
    pub virtual_orbitals: usize,
    pub orbital_energies: Vec<f64>,
    pub atomic_orbitals: DMatrix<f64>,
}

/// The excitations of one request.
pub struct Excitations {
    pub approximation: Approximation,
    pub spin: Spin,
    pub coupling: Coupling,
    /// The functional of a Kohn-Sham reference; `None` for Hartree-Fock.
    pub functional: Option<String>,
    /// `omega_n`, Hartree, ascending.
    pub energies: Vec<f64>,
    /// `f_n = 2/3 omega_n abs(mu_n)^2` (zero for triplets).
    pub oscillator_strengths: Vec<f64>,
    /// `mu_n = <0| r |n>`, e Bohr (zero for triplets; the sign of each root is a convention).
    pub transition_dipoles: Vec<[f64; 3]>,
    /// The velocity form `p_n = <0| grad |n>` (Bohr^-1; zero for triplets), from `X - Y`.
    pub transition_velocities: Vec<[f64; 3]>,
    /// `f^V_n = 2/3 abs(p_n)^2 / omega_n`. It equals the length form for exact states of a local
    /// Hamiltonian; in a finite basis, or with exact exchange, the two differ.
    pub velocity_oscillator_strengths: Vec<f64>,
    /// `X + Y` of each root and channel (occupied x virtual), `sum (X + Y) . (X - Y) = 1`
    /// (TDA: `X`, normalized).
    pub plus: Vec<Vec<DMatrix<f64>>>,
    /// `X - Y` of each root and channel (TDA: `X`).
    pub minus: Vec<Vec<DMatrix<f64>>>,
    pub transitions: Vec<Vec<Transition>>,
    /// The natural transition orbital weights of each root (`sigma_k^2 / sum sigma^2`).
    pub nto_weights: Vec<Vec<f64>>,
    /// The first-order shift of each root from the one-center Hartree kernel that the
    /// linearized PAW one-center energy omits (PAW-LCAO Kohn-Sham only), Hartree. The one-center
    /// exchange-correlation part is not estimated.
    pub one_center_hartree: Option<Vec<f64>>,
    /// The reference's spin channels: one for a closed shell, alpha and beta otherwise.
    pub channels: Vec<Channel>,
    /// Unrestricted references: `<S^2>` of each root as the singles combination of the
    /// reference's determinants with the root's `X` amplitudes (normalized), and of the reference.
    pub spin_squared: Option<Vec<f64>>,
    pub reference_spin_squared: Option<f64>,
    /// Roots asked for, and how many more the multiplet rule added.
    pub requested: usize,
    pub added: usize,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub applications: usize,
}
impl Excitations {
    fn amplitude(&self, root: usize, channel: usize) -> Result<(&DMatrix<f64>, &Channel)> {
        let plus = self
            .plus
            .get(root)
            .ok_or_else(|| invalid("excitation root out of range"))?;
        let c = self
            .channels
            .get(channel)
            .ok_or_else(|| invalid("spin channel out of range"))?;
        Ok((&plus[channel], c))
    }
    /// The transition density matrix of `root` in `channel` in the nonorthogonal atomic basis,
    /// `T = C_o (X + Y) C_v^T`, symmetrized (the spatial factor of that spin's change).
    pub fn transition_density(&self, root: usize, channel: usize) -> Result<DMatrix<f64>> {
        let (plus, c) = self.amplitude(root, channel)?;
        let o = c.occupied;
        let a = &c.atomic_orbitals;
        let t = a.columns(0, o) * plus * a.columns(o, c.virtual_orbitals).transpose();
        Ok((&t + t.transpose()) * 0.5)
    }
    /// The natural transition orbitals of `root` in `channel`: weights (within the channel) and
    /// the occupied and empty orbital pairs (columns in the nonorthogonal atomic basis), in
    /// descending weight.
    pub fn natural_transition_orbitals(
        &self,
        root: usize,
        channel: usize,
    ) -> Result<(Vec<f64>, DMatrix<f64>, DMatrix<f64>)> {
        let (plus, c) = self.amplitude(root, channel)?;
        properties::natural_transition_orbitals(plus, &c.atomic_orbitals, c.occupied)
    }
}

/// The Fock matrices of the energy `reference` minimizes, at the density matrices `densities`
/// in the orthonormal frame: one per-spin matrix of a restricted state, or both spin matrices.
/// This is the builder the response kernel differentiates (the oracle of its finite-difference
/// test): the Kohn-Sham Fock of `dft::response` for a functional, `h + J[D_total] - K[D_spin]`
/// with the SCF's pair factors for Hartree-Fock. Complex densities keep their imaginary part in
/// the exchange term.
pub fn reference_fock(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    densities: &[DMatrix<crate::quantum::C64>],
) -> Result<Vec<DMatrix<crate::quantum::C64>>> {
    use crate::quantum::C64;
    let m = integrals.m();
    if densities.is_empty() || densities.len() > 2 || densities.iter().any(|d| d.shape() != (m, m))
    {
        return Err(invalid(
            "reference_fock needs one or two m x m density matrices",
        ));
    }
    integrals
        .context
        .run(|| match reference.functional.as_deref() {
            Some(name) => Ok(super::dft::response(integrals, densities, name)?.0),
            None => {
                let eri = integrals.eri()?;
                let h = DMatrix::from_fn(m, m, |a, b| {
                    C64::new(integrals.one.as_slice()[a * m + b], 0.)
                });
                let parts: Vec<_> = densities
                    .iter()
                    .map(|d| super::scf::jk(&eri, m, d))
                    .collect();
                if densities.len() == 1 {
                    let (j, k) = &parts[0];
                    return Ok(vec![&h + j * C64::new(2., 0.) - k]);
                }
                let total = &parts[0].0 + &parts[1].0;
                Ok(parts.iter().map(|(_, k)| &h + &total - k).collect())
            }
        })
}

/// A linear-response operator over its occupied-to-empty transitions: the restricted
/// [`Response`] or the [`Unrestricted`] one.
pub(super) trait Operator {
    fn dimension(&self) -> usize;
    /// `e_a - e_i` of every transition (the preconditioner).
    fn delta(&self) -> &[f64];
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
    fn resonant(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>>;
    fn spin(&self) -> Spin;
}

struct Tda<'r, O>(&'r O);
impl<O: Operator> Symmetric for Tda<'_, O> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.0.delta().to_vec()
    }
    fn apply(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.0.resonant(block)
    }
}
struct Full<'r, O>(&'r O);
impl<O: Operator> Paired for Full<'_, O> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.0.delta().to_vec()
    }
    fn sum(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.0.sum(block)
    }
    fn difference(&self, block: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.0.difference(block)
    }
}

/// The roots of one solve: energies, `X + Y`, `X - Y` (columns), residuals, counters.
struct Solved {
    values: Vec<f64>,
    plus: DMatrix<f64>,
    minus: DMatrix<f64>,
    residuals: Vec<f64>,
    iterations: usize,
    applications: usize,
}

fn solve<O: Operator>(response: &O, request: &Request, roots: usize) -> Result<Solved> {
    let mut options = Options::new(roots);
    options.tolerance = request.tolerance;
    options.max_iterations = request.max_iterations;
    let unconverged = |iterations: usize, residual: f64| {
        crate::Error::Kernel(format!(
            "the excitation solver did not converge in {iterations} iterations (largest residual \
             {residual:.2e} above the tolerance {:.1e}); raise max_iterations or the tolerance",
            request.tolerance
        ))
    };
    match request.approximation {
        Approximation::TammDancoff => {
            let roots = davidson::lowest(&Tda(response), &options)?;
            if !roots.converged {
                let worst = roots.residuals.iter().copied().fold(0., f64::max);
                return Err(unconverged(roots.iterations, worst));
            }
            // A non-positive root of A is a lower state of this symmetry than the reference: the
            // restricted reference is unstable (a stretched bond's triplet, for example).
            if let Some((root, omega)) = roots.values.iter().enumerate().find(|(_, w)| **w <= 0.) {
                return Err(invalid(format!(
                    "the {} Tamm-Dancoff problem has a non-positive root (root {root}, omega = \
                     {omega:.6e} Ha): the reference is unstable in this symmetry; {}",
                    response.spin().name(),
                    remedy(response.spin())
                )));
            }
            Ok(Solved {
                values: roots.values,
                plus: roots.vectors.clone(),
                minus: roots.vectors,
                residuals: roots.residuals,
                iterations: roots.iterations,
                applications: roots.applications,
            })
        }
        Approximation::Casida => {
            let roots =
                davidson::lowest_paired(&Full(response), &options).map_err(|e| match e {
                    crate::Error::Invalid(message) => invalid(format!(
                    "the {} Casida problem has no real root: {message}; the reference is unstable \
                     in this symmetry; {}",
                    response.spin().name(),
                    remedy(response.spin())
                )),
                    other => other,
                })?;
            if !roots.converged {
                let worst = roots.residuals.iter().copied().fold(0., f64::max);
                return Err(unconverged(roots.iterations, worst));
            }
            Ok(Solved {
                values: roots.values,
                plus: roots.plus,
                minus: roots.minus,
                residuals: roots.residuals,
                iterations: roots.iterations,
                applications: roots.applications,
            })
        }
    }
}

/// What an unstable reference of `spin` should be replaced with.
fn remedy(spin: Spin) -> &'static str {
    match spin {
        Spin::Unrestricted => {
            "this unrestricted state is not the lowest of its symmetry: converge another \
             (for example from other initial magnetic moments)"
        }
        _ => "use a spin-unrestricted reference",
    }
}

/// The lowest roots of `operator` for `request`, completed to whole multiplets: the solve and
/// the number of roots kept.
fn roots<O: Operator>(operator: &O, request: &Request) -> Result<(Solved, usize)> {
    let n = operator.dimension();
    if request.num_states > n {
        return Err(invalid(format!(
            "num_states={} exceeds the {n} occupied-to-empty transitions of this basis",
            request.num_states
        )));
    }
    let mut extra = 3;
    loop {
        let roots = (request.num_states + extra).min(n);
        let solved = solve(operator, request, roots)?;
        let mut count = request.num_states;
        while count < roots && solved.values[count] - solved.values[count - 1] < DEGENERATE_ROOTS {
            count += 1;
        }
        if count < roots || roots == n {
            return Ok((solved, count));
        }
        extra += 4;
    }
}

/// The lowest `request.num_states` excitations of `reference` (completed to whole multiplets):
/// spin-adapted for a closed-shell (restricted) reference, spin-conserving for an unrestricted
/// one.
pub fn excitations(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    request: &Request,
) -> Result<Excitations> {
    if request.num_states == 0 {
        return Err(invalid("num_states must be at least 1"));
    }
    if !request.tolerance.is_finite() || request.tolerance <= 0. || request.max_iterations == 0 {
        return Err(invalid(
            "the excitation tolerance must be positive and finite and max_iterations at least 1",
        ));
    }
    integrals.context.run(|| {
        if reference.method == "uhf" {
            if request.spin.is_some() {
                return Err(invalid(
                    "an unrestricted reference has spin-conserving excitations of both spins; \
                     spin='singlet' or 'triplet' is the spin adaptation of a closed shell (leave \
                     spin unset)",
                ));
            }
            let operator = Unrestricted::new(integrals, reference, request.coupling)?;
            let (solved, count) = roots(&operator, request)?;
            return properties::assemble_unrestricted(
                integrals, reference, &operator, request, solved, count,
            );
        }
        let spin = request.spin.unwrap_or(Spin::Singlet);
        let response = Response::new(integrals, reference, spin, request.coupling)?;
        let (solved, count) = roots(&response, request)?;
        properties::assemble(integrals, reference, &response, request, solved, count)
    })
}
