//! The analytic stress of a plain-DFT PAW-LCAO crystal (AS-4/AS-5): the strain `eps` as one more
//! perturbation of the force engine's chain rule.
//!
//! # Derivation
//!
//! Mermin free energy with the Lowdin carry `C = S(eps)^{-1/2} y` (`y` and the occupations
//! fixed), at the converged, output-density, re-diagonalized state of [`gradient::Frozen`]:
//!
//! `dF/d eps_ab = sum_{s,k} deg w_k Re tr[D dh_ab - W dS_ab] + explicit_ab`,
//! `D = C f C^+`, `W = C f (eps - reference) C^+` (unshifted eigenvalues).
//!
//! `dh_ab` is the strain derivative of the one-electron operator with the potentials `v_i`, the
//! channel potentials `m_a` and the moments `q` held at their output values; `explicit_ab`
//! differentiates the energy functional at fixed `D`, `rho_i`, `q`. The matrix terms are the
//! force engine's algebra with a strain tangent instead of an atom displacement (rows are NOT
//! owner-masked: every function, projector and node moves); the derivatives are of the
//! DISCRETIZED functional at the pins of `ANALYTIC_STRESS_PLAN` section 2 (node counts, ion
//! width, filter cutoff, lattice operations of the alias averages, integer triples of every
//! cutoff list, sphere rules).
//!
//! Convention: `x -> (1 + eps) x` on the lattice, the atoms and the grid nodes; `G -> (1 - eps^T) G`,
//! `d f(G) / d eps_ab = -G_a d_b f`, `d chi / d eps_ab = sum_T e^{ik.T} (d_a phi)(y) y_b`; `eps`
//! has nine independent entries. [`Stress::sigma`] is `(g_ab + g_ba) / (2 Omega)` with
//! `g_ab = dF/d eps_ab`, Hartree/Bohr^3 (`units.py` owns the eV/A^3 factor); the antisymmetric
//! part is reported as a diagnostic.
//!
//! # Terms (a [`Breakdown`] entry each, `g_ab` per entry)
//!
//! * [`pulay`]: S-1 overlap (`-W dS`), S-2 kinetic, S-3 local potential, S-4 short-range spheres,
//!   S-5 projections (through the engine's `Z`).
//! * [`electrostatic`]: E-1 Hartree Jacobian, E-2 kernel, E-3 hats, E-4 ions, E-5 dense sums,
//!   E-6 short-range ion compensation, E-7 ion constants.
//! * [`xc`]: X-1 Jacobian, X-3 GGA spectral gradient, X-4 partial core (X-2, the xc potential
//!   through `rho_i`, is inside S-3).
//!
//! [`stress`] raises `Unsupported` for hybrids, spin-orbit, meta-GGA, rVV10, float32 and
//! `spins != 1` until their packages; a missing term would give a silently wrong stress, so a
//! provider that cannot supply a term raises, it never returns zero.
pub mod electrostatic;
pub mod pulay;
pub mod xc;

use super::{
    gradient::{Frozen, Layout, Matrix, Term},
    invalid,
};
use crate::{models, quantum::C64, Error, Result};

/// `[a][b]`: the derivative with respect to the strain entry `eps_ab`.
pub type Tensor<T> = [[T; 3]; 3];

/// The dense channel arrays differentiated (E-5, E-6): `U` (`channels x channels`), `u_ion`,
/// `linear` (per channel), and `E_ion`.
pub struct Dense {
    pub u: Tensor<Matrix>,
    pub u_ion: Tensor<Vec<C64>>,
    pub linear: Tensor<Vec<C64>>,
    pub e_ion: Tensor<f64>,
}

/// A geometry provider for the strain: everything the stress needs that is not the converged
/// state. Every derivative method returns the derivative of the named quantity with respect to
/// `eps_ab` (energies, never a stress) at the pins of the module documentation.
pub trait Strain {
    /// The potentials, moments and output-density data (opaque to the engine).
    type Fields;

    fn layout(&self) -> &Layout;
    /// The channel blocks `B_a` (`p x p`, strain independent).
    fn blocks(&self) -> &[Matrix];
    /// `q` with `S = S_grid + C q C^+`.
    fn augmentation(&self) -> &Matrix;
    /// `D_ion`.
    fn ion_operator(&self) -> &Matrix;
    /// The cell volume (Bohr^3).
    fn volume(&self) -> f64;
    /// `C_k` (`m x p`).
    fn projection(&self, k: usize) -> Result<Matrix>;
    /// `dC_ab` (`m x p`) with the quadrature spheres rigid about their atoms.
    fn projection_strain(&self, k: usize) -> Result<Tensor<Matrix>>;
    /// `d M_ab` of one matrix term for `spin` at k-point `k` (`m x m`) with the potentials of
    /// `fields` held fixed. [`Term::Overlap`] and [`Term::Kinetic`] carry the `delta_ab` of their
    /// `dv`; [`Term::Local`] does not (the node values `rho_i` stay fixed: the `dv` Jacobian of the
    /// charge is [`electrostatic`]'s E-1), [`Term::ShortRange`] is the sum over the spheres.
    fn matrix_strain(
        &self,
        spin: usize,
        k: usize,
        term: Term,
        fields: &Self::Fields,
    ) -> Result<Tensor<Matrix>>;
    /// The channel potentials `m_a`.
    fn moments<'a>(&self, fields: &'a Self::Fields) -> &'a [C64];
    /// E-1..E-4 at the output density of `fields` and the moments `q`.
    fn hartree(&self, fields: &Self::Fields, q: &[C64]) -> Result<electrostatic::Hartree>;
    /// E-5 and E-6.
    fn dense(&self) -> Result<Dense>;
    /// E-7.
    fn constants(&self) -> Result<Tensor<f64>>;
    /// X-1, X-3, X-4.
    fn xc(&self, fields: &Self::Fields) -> Result<xc::Terms>;
}

/// `dF/d eps_ab` term by term, so a caller can report, gate or damage one term.
#[derive(Clone, Debug, PartialEq)]
pub struct Breakdown {
    pub overlap: Tensor<f64>,
    pub kinetic: Tensor<f64>,
    pub local: Tensor<f64>,
    pub short_range: Tensor<f64>,
    pub projection: Tensor<f64>,
    pub hartree: Tensor<f64>,
    pub kernel: Tensor<f64>,
    pub hats: Tensor<f64>,
    pub ions: Tensor<f64>,
    pub dense: Tensor<f64>,
    pub linear: Tensor<f64>,
    pub constants: Tensor<f64>,
    pub xc_jacobian: Tensor<f64>,
    pub xc_gradient: Tensor<f64>,
    pub xc_core: Tensor<f64>,
}

impl Breakdown {
    pub fn terms(&self) -> [(&'static str, &Tensor<f64>); 15] {
        [
            ("overlap", &self.overlap),
            ("kinetic", &self.kinetic),
            ("local", &self.local),
            ("short_range", &self.short_range),
            ("projection", &self.projection),
            ("hartree", &self.hartree),
            ("kernel", &self.kernel),
            ("hats", &self.hats),
            ("ions", &self.ions),
            ("dense", &self.dense),
            ("linear", &self.linear),
            ("constants", &self.constants),
            ("xc_jacobian", &self.xc_jacobian),
            ("xc_gradient", &self.xc_gradient),
            ("xc_core", &self.xc_core),
        ]
    }

    /// `g_ab = dF/d eps_ab`, the sum of every term.
    pub fn gradient(&self) -> Tensor<f64> {
        let mut total = [[0.; 3]; 3];
        for (_, term) in self.terms() {
            for a in 0..3 {
                for b in 0..3 {
                    total[a][b] += term[a][b];
                }
            }
        }
        total
    }
}

/// The stress of a converged crystal.
#[derive(Clone, Debug)]
pub struct Stress {
    pub breakdown: Breakdown,
    /// `g_ab = dF/d eps_ab` (Hartree), unsymmetrized.
    pub gradient: Tensor<f64>,
    /// The cell volume (Bohr^3).
    pub volume: f64,
    /// `sigma_ab = (g_ab + g_ba) / (2 Omega)` (Hartree/Bohr^3).
    pub sigma: Tensor<f64>,
    /// `(g_ab - g_ba) / 2` (Hartree): the torque diagnostic. It vanishes up to the grid floor
    /// for a non-magnetic state with complete (atom, l, m) shells.
    pub antisymmetric: Tensor<f64>,
}

impl Stress {
    fn new(breakdown: Breakdown, volume: f64) -> Self {
        let gradient = breakdown.gradient();
        let sigma = std::array::from_fn(|a| {
            std::array::from_fn(|b| 0.5 * (gradient[a][b] + gradient[b][a]) / volume)
        });
        let antisymmetric = std::array::from_fn(|a| {
            std::array::from_fn(|b| 0.5 * (gradient[a][b] - gradient[b][a]))
        });
        Self {
            breakdown,
            gradient,
            volume,
            sigma,
            antisymmetric,
        }
    }

    /// `-tr(sigma) / 3`.
    pub fn pressure(&self) -> f64 {
        -(self.sigma[0][0] + self.sigma[1][1] + self.sigma[2][2]) / 3.
    }
}

/// Raises for everything this module does not differentiate yet.
pub fn check_request(layout: &Layout) -> Result<()> {
    let refuse = |what: &str| {
        Err(Error::Unsupported(format!(
            "the analytic crystal stress is not available for {what}"
        )))
    };
    let request = &layout.request;
    let definition = models::functional(&request.functional)?;
    if definition.screened_exchange.is_some() {
        return refuse("a hybrid functional (the hybrid strain terms are AS-9)");
    }
    if request.spinor {
        return refuse("a spinor crystal (the spin-orbit strain terms are AS-8)");
    }
    if request.single_precision {
        return refuse("float32 data");
    }
    if definition.requires_tau {
        return refuse(&format!("the meta-GGA {}", definition.name));
    }
    if definition.nonlocal_b.is_some() {
        return refuse(&format!("the rVV10 correction of {}", definition.name));
    }
    if layout.spins != 1 {
        return refuse(
            "a spin-polarized crystal (the partial core is differentiated against one xc potential)",
        );
    }
    Ok(())
}

fn validate<S: Strain>(s: &S, frozen: &Frozen<S::Fields>) -> Result<()> {
    let l = s.layout();
    let fine = l.degeneracy > 0.0
        && l.degeneracy.is_finite()
        && l.reference.is_finite()
        && l.weights.iter().all(|w| w.is_finite())
        && l.row_owner.len() == l.m
        && l.projector_owner.len() == l.p
        && s.volume() > 0.
        && s.augmentation().shape() == (l.p, l.p)
        && s.ion_operator().shape() == (l.p, l.p)
        && s.blocks().iter().all(|b| b.shape() == (l.p, l.p))
        && frozen.q.len() == s.blocks().len();
    if !fine {
        return Err(invalid("crystal stress: inconsistent layout"));
    }
    let bands = &frozen.state.bands;
    if bands.len() != l.spins || bands.iter().any(|b| b.len() != l.weights.len()) {
        return Err(invalid(
            "crystal stress: one set of bands per spin and k-point",
        ));
    }
    let consistent = bands.iter().flatten().all(|b| {
        let nb = b.coefficients.ncols();
        b.coefficients.nrows() == l.m
            && b.eigenvalues.len() == nb
            && b.occupations.len() == nb
            && b.eigenvalues.iter().all(|x| x.is_finite())
            && b.occupations.iter().all(|f| f.is_finite() && *f >= 0.)
    });
    if !consistent {
        return Err(invalid("crystal stress: inconsistent bands"));
    }
    if let Some(o) = &frozen.hubbard {
        if o.len() != l.spins || o.iter().any(|m| m.shape() != (l.p, l.p)) {
            return Err(invalid("crystal stress: one p x p +U operator per spin"));
        }
    }
    Ok(())
}

/// The stress of a consistent frozen state (no re-diagonalization): refuses what is not
/// differentiated, contracts every term.
pub fn stress<S: Strain>(s: &S, frozen: &Frozen<S::Fields>) -> Result<Stress> {
    check_request(s.layout())?;
    validate(s, frozen)?;
    Ok(Stress::new(contract(s, frozen)?, s.volume()))
}

/// The term-by-term contraction on a consistent frozen state.
pub fn contract<S: Strain>(s: &S, frozen: &Frozen<S::Fields>) -> Result<Breakdown> {
    validate(s, frozen)?;
    let matrix = pulay::contract(s, frozen)?;
    let hartree = s.hartree(&frozen.fields, &frozen.q)?;
    let dense = electrostatic::dense_terms(&s.dense()?, &frozen.q)?;
    let constants = s.constants()?;
    let xc = s.xc(&frozen.fields)?;
    Ok(Breakdown {
        overlap: matrix.overlap,
        kinetic: matrix.kinetic,
        local: matrix.local,
        short_range: matrix.short_range,
        projection: matrix.projection,
        hartree: hartree.jacobian,
        kernel: hartree.kernel,
        hats: hartree.hats,
        ions: hartree.ions,
        dense: dense.dense,
        linear: dense.linear,
        constants,
        xc_jacobian: xc.jacobian,
        xc_gradient: xc.gradient,
        xc_core: xc.core,
    })
}
