//! The plain-DFT force engine (CF-3): the contraction algebra of the crystal force, generic over
//! [`Derivatives`], the one trait a geometry provider implements (the Rust setup, later; a toy
//! with analytic `C(X)`, `S(X)` today).
//!
//! # Derivation
//!
//! Mermin free energy, orthonormality carried by the Lowdin frame (`C = lowdin y`, `y` fixed):
//!
//! `dF/dtau_X = sum_{s,k} deg w_k Re tr[D dH_X|frozen - W dS_X] + explicit_X`,
//! `D = C f C^+`, `W = C f (eps - reference) C^+` with the UNSHIFTED eigenvalues.
//!
//! * *Matrix terms* (basis function `mu` on atom `A` moves rigidly): the overlap (P1), the
//!   kinetic operator (P2), the local potential (P3) and the short-range spheres (P4) each
//!   enter as `dM_X = P_X G + G^+ P_X` with `G` the derivative with respect to the center of the
//!   ROW function and `P_X = diag(own_X rows)`, so `tr[D dM_X] = 2 Re sum_{mu in X} (G D)_{mumu}`.
//! * *Projections* (P5): `C_{mu p} = <chi_mu|p_p>` depends on the pair only, so
//!   `dC_X = P_X G_C - G_C Q_X` (`Q_X = diag(own_X projector columns)`), entering `C q C^+` in `S`
//!   and `C (D_ion + sum_a m_a B_a) C^+` in `H`. The +U operator `O_s` is added to `D_ion` at
//!   exactly these two sites (the [`Hubbard`] hook); nothing else changes.
//! * *Explicit terms*: E1 structure factors of the hats, E2 `ion(G)` (potential term and dense
//!   `E_ion`), E3 dense `U`, `U_ion`, `linear`, E4 the ion constants, E5 the partial core
//!   against the xc potential.
//!
//! Two traps of the solver are the engine's to close ([`state::freeze`]): the potentials and the
//! +U operator are those of the OUTPUT density, and `W` takes eigenvalues with the reference
//! removed.
//!
//! # Hybrid functionals (HX-W)
//!
//! The exact exchange `E_x[D; R] = -(a/2) sum_k w_k tr(D_k K_k[D])` is quadratic in the density
//! matrix, so at fixed `D` its derivative is the explicit terms of [`super::hybrid_gradient`]
//! (the phase of the compensation shapes, the basis functions of the grid pair, bra and ket, and
//! the response `Z` of the compensation moments and of the one-center terms to the projections).
//! The provider returns them through [`Derivatives::exchange_gradient`]; [`contract`] adds them as
//! four terms of the [`Breakdown`], the response `Z` chained to the atoms with the same ownership
//! identity as the projection term (its weights are inside `Z`, so it never joins the engine's
//! own `z`). The Hamiltonian the bands are diagonalized with is `H_semilocal - a K[D_out]`: `K` is
//! Hamiltonian-like, so `W` takes the eigenvalues of THAT `H` and the matrix terms above are the
//! semilocal ones. A provider that does not implement the hook raises; it never returns a zero.
//!
//! # Spin polarization (F1)
//!
//! A collinear unrestricted crystal (`Layout::spins = 2`, degeneracy 1) is the same contraction
//! with one density matrix `D_s` and one energy-weighted `W_s` per spin channel, a common Fermi
//! level and the Mermin occupations of the whole mesh. What is spin-resolved: the local-potential
//! term (`D_s` against the potential `v_ee + v_xc,s` of ITS spin: [`Derivatives::cross`] takes the
//! spin), the projection terms (`H_C C^+ D_s - q C^+ W_s`, summed over spins). What is spin-summed:
//! the compensation moments `q` and the hats, the ions, the dense lattice sums (the electrostatic
//! energy sees the total charge). The partial core is split in halves between the spin densities
//! (`rho_s + core/2` in the functional), so `dE_xc/d core = (v_up + v_dn) / 2`: the provider keeps
//! that ONE spin-averaged array, not two, because nothing else reads the spin resolution of `v_xc`
//! in the explicit terms (a core differentiated against `v_up` alone is wrong by the half
//! difference of the two potentials, and no sum rule catches it). For a hybrid the exact exchange
//! acts within a spin, `K_s[D_s]`, and the spheres' spin-scaling remainder enters through the
//! one-center operators of each spin, which depend on both projected densities.
//!
//! # Conventions
//!
//! Atomic units; derivatives and the [`Breakdown`] are `dE/dtau` (the force is the negative);
//! every contraction is per spin channel times `Layout::degeneracy`. The collinear
//! scalar-relativistic LDA/PBE/HSE06 case (restricted or collinear spin-polarized) and the spinor
//! (spin-orbit) LDA/PBE/HSE06 case with float64 data are supported (a spinor hybrid only for the
//! time-reversal-symmetric crystal [`Crystal`](crate::periodic::Crystal) can build); [`gradient`]
//! raises `Unsupported` for meta-GGA, rVV10 and float32 requests. Derivative data are requested one k-point at a time and dropped after use.
//!
//! # The spinor frame (the contract of a spinor provider)
//!
//! A spinor request is presented in the DOUBLED frame ([`spinor`]): `Layout::m` is `2M` (spin
//! major, the `alpha` block first), `row_owner` repeats the owners per spin, the bands carry `2M`
//! rows, and the scalar operators and projections are `I2 x (.)`, so the scalar terms contract
//! `D_alpha,alpha + D_beta,beta` and `W_alpha,alpha + W_beta,beta` with no change of the
//! algebra. The local potential is the Pauli sum `sum_a sigma_a x G^a` of the four channel
//! derivatives (one channel for a time-reversal crystal). The spin-orbit matrix is not of that
//! form: its projections stay in the scalar frame (`M x P_so`), reached through
//! [`Derivatives::spin_orbit_projection`], and [`SpinOrbit::add_k`] contracts them with the
//! full `2M x 2M` density of the frozen state, once per k-point, in [`pulay::contract`]
//! (`Breakdown::spin_orbit_rows` / `spin_orbit_columns`). A provider without spin-orbit data
//! raises `Unsupported`, never a zero term. [`gradient`] differentiates the LDA/PBE/HSE06
//! spinor; DFT+U, a meta-GGA and float32 raise with their cause. The exact-exchange response of
//! a spinor is in the scalar frame of one spin block ([`exchange`]).
pub mod core;
pub mod electrostatic;
pub mod exchange;
pub mod hellmann_feynman;
pub mod pulay;
pub mod scf;
pub mod spin_orbit;
pub mod spinor;
pub mod state;

use super::{hybrid_gradient, invalid};
use crate::{models, quantum::C64, Error, Result};
use nalgebra::DMatrix;

pub use spin_orbit::SpinOrbit;
pub use state::{freeze, Bands, Frozen, State};

pub type Matrix = DMatrix<C64>;

/// What the caller asked for, as far as the engine can refuse it.
#[derive(Clone, Debug)]
pub struct Request {
    pub functional: String,
    pub spinor: bool,
    pub single_precision: bool,
}

/// Sizes, ownership and conventions of the crystal behind a [`Derivatives`].
#[derive(Clone, Debug)]
pub struct Layout {
    pub atoms: usize,
    /// Basis functions and projectors.
    pub m: usize,
    pub p: usize,
    pub spins: usize,
    /// 2 for a restricted run (one spin channel standing for both), 1 otherwise.
    pub degeneracy: f64,
    /// The k-point weights.
    pub weights: Vec<f64>,
    /// The atom each basis function (projector) belongs to.
    pub row_owner: Vec<usize>,
    pub projector_owner: Vec<usize>,
    /// The constant the solver adds to the eigenvalues it returns.
    pub reference: f64,
    pub request: Request,
}

/// The matrix terms whose derivative is a row-center cross matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// P1: the grid overlap (the sphere part enters through `C q C^+`).
    Overlap,
    /// P2: the alias-averaged kinetic operator.
    Kinetic,
    /// P3: the grid local potential of the output density.
    Local,
    /// P4: the short-range spheres.
    ShortRange,
}

impl Term {
    pub const ALL: [Term; 4] = [Term::Overlap, Term::Kinetic, Term::Local, Term::ShortRange];
}

/// The derivative of the dense channel arrays with respect to one coordinate of one atom.
pub struct Dense {
    /// `dU`, `channels x channels`.
    pub u: Matrix,
    pub uion: Vec<C64>,
    pub linear: Vec<C64>,
}

/// A geometry provider: everything the force needs that is not the converged state.
///
/// Derivative methods return the derivative of the named quantity with respect to
/// `tau_atom^axis` (the energy derivative, never a force).
pub trait Derivatives {
    /// The potentials and moments of the output density (opaque to the engine).
    type Fields;

    fn layout(&self) -> &Layout;
    /// The channel blocks `B_a` (`p x p`, position independent).
    fn blocks(&self) -> &[Matrix];
    /// `q` with `S = S_grid + C q C^+`.
    fn augmentation(&self) -> &Matrix;
    /// `D_ion`, the position-independent part of the projector operator of `H`.
    fn ion_operator(&self) -> &Matrix;
    /// `C_k` (`m x p`).
    fn projection(&self, k: usize) -> Result<Matrix>;
    /// `G_C^a(k)`: the derivative of `C_k` with respect to the center of the ROW function.
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]>;
    /// The row-center derivative of one matrix term for `spin` at k-point `k`, per axis (`m x m`).
    fn cross(
        &self,
        spin: usize,
        k: usize,
        term: Term,
        fields: &Self::Fields,
    ) -> Result<[Matrix; 3]>;
    /// P4's third center. The short-range matrix of the sphere about atom `A` moves with `A`
    /// itself; by translation invariance of the sphere rule that equals moving EVERY function
    /// the opposite way with the sphere fixed. A provider whose short-range term has such
    /// spheres returns, per sphere atom `A` (one entry per atom, in atom order), the row-center
    /// derivative `G_A` of THAT sphere's matrix over all rows; the engine charges
    /// `-2 Re sum_mu (G_A D)_{mumu}` to `A`, while the rows' own share stays in
    /// [`Term::ShortRange`] (the sum over the spheres). A term made of pair functions alone
    /// (the toy) has no third center and returns none.
    fn sphere_cross(
        &self,
        spin: usize,
        k: usize,
        fields: &Self::Fields,
    ) -> Result<Vec<[Matrix; 3]>>;

    /// Potentials and channel potentials `m_a` built from the density of `state` (and its `q`).
    fn fields(&self, state: &State, q: &[C64]) -> Result<Self::Fields>;
    fn moments<'a>(&self, fields: &'a Self::Fields) -> &'a [C64];
    /// Diagonalizes `H_k` for `spin` with these fields, plus the +U operator when there is one.
    /// The eigenvalues are returned as the solver returns them (shifted by the reference).
    fn bands(
        &self,
        spin: usize,
        k: usize,
        fields: &Self::Fields,
        hubbard: Option<&Matrix>,
    ) -> Result<Bands>;

    /// E1: `<K rho | d hat_a>` per channel (`rho` the output density).
    fn hat_overlap(&self, fields: &Self::Fields, atom: usize, axis: usize) -> Result<Vec<C64>>;
    /// E2: `<K rho | d ion>`.
    fn ion_overlap(&self, fields: &Self::Fields, atom: usize, axis: usize) -> Result<f64>;
    /// E2: the derivative of the dense `E_ion`.
    fn ion_energy(&self, atom: usize, axis: usize) -> Result<f64>;
    /// E3.
    fn dense(&self, atom: usize, axis: usize) -> Result<Dense>;
    /// E4: the derivative of the ion constants.
    fn constant(&self, atom: usize, axis: usize) -> Result<f64>;
    /// E5: `<v_xc | d rho_core>`.
    fn core_overlap(&self, fields: &Self::Fields, atom: usize, axis: usize) -> Result<f64>;

    /// The spin-orbit operator `B` of a spinor crystal with the owners of the basis functions and of
    /// the union projectors ([`SpinOrbit`]). A provider without spin-orbit data raises
    /// `Unsupported` (never an empty operator).
    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        Err(no_spin_orbit())
    }
    /// `C^SO_k` (`m x p_so`): the projections of the Bloch sums on the spin-orbit union
    /// projectors, in the scalar frame (`m` basis functions, not `2m`).
    fn spin_orbit_projection(&self, _k: usize) -> Result<Matrix> {
        Err(no_spin_orbit())
    }
    /// `G^SO_k`: the derivative of `C^SO_k` with respect to the center of the ROW function.
    fn spin_orbit_projection_gradient(&self, _k: usize) -> Result<[Matrix; 3]> {
        Err(no_spin_orbit())
    }

    /// The explicit terms of the exact-exchange derivative at the FIXED occupied states of
    /// `state` ([`hybrid_gradient::explicit`] with the basis functions moving): required, with
    /// [`Gradient::functions`](hybrid_gradient::Gradient::functions) present, for a hybrid
    /// request. A provider without the screened exchange raises `Unsupported` (never an empty
    /// term).
    fn exchange_gradient(&self, _state: &State) -> Result<hybrid_gradient::Gradient> {
        Err(Error::Unsupported(
            "this geometry provider carries no screened exact exchange".into(),
        ))
    }
}

fn no_spin_orbit() -> Error {
    Error::Unsupported("this geometry provider carries no spin-orbit projectors".into())
}

/// The +U hook: the operator `O_s = W V W^T` (`p x p`, one per spin channel, the complete
/// `V = V_Hub - V_dc`) of the density of `state`.
pub trait Hubbard {
    fn operators(&self, state: &State) -> Result<Vec<Matrix>>;
}

/// `dE/dtau` per atom, term by term, so a caller can report, gate or damage one term.
#[derive(Clone, Debug, PartialEq)]
pub struct Breakdown {
    pub overlap: Vec<[f64; 3]>,
    pub kinetic: Vec<[f64; 3]>,
    pub local: Vec<[f64; 3]>,
    pub short_range: Vec<[f64; 3]>,
    /// P5, the basis functions moving.
    pub projection_rows: Vec<[f64; 3]>,
    /// P5, the projectors moving.
    pub projection_columns: Vec<[f64; 3]>,
    pub hats: Vec<[f64; 3]>,
    pub ion: Vec<[f64; 3]>,
    pub dense: Vec<[f64; 3]>,
    pub constants: Vec<[f64; 3]>,
    pub core: Vec<[f64; 3]>,
    /// The spin-orbit union-projector term, the basis functions moving (spinor requests only).
    pub spin_orbit_rows: Vec<[f64; 3]>,
    /// The same, the union projectors moving.
    pub spin_orbit_columns: Vec<[f64; 3]>,
    /// Hybrids only: the phase of the compensation shapes of the exact exchange.
    pub exchange_phase: Vec<[f64; 3]>,
    /// The basis functions of the exchange pair density moving, bra and ket slots together.
    pub exchange_functions: Vec<[f64; 3]>,
    /// The projection response of the exact exchange, the basis functions moving.
    pub exchange_rows: Vec<[f64; 3]>,
    /// The same, the projectors moving.
    pub exchange_columns: Vec<[f64; 3]>,
}

impl Breakdown {
    pub fn terms(&self) -> [(&'static str, &Vec<[f64; 3]>); 17] {
        [
            ("overlap", &self.overlap),
            ("kinetic", &self.kinetic),
            ("local", &self.local),
            ("short_range", &self.short_range),
            ("projection_rows", &self.projection_rows),
            ("projection_columns", &self.projection_columns),
            ("hats", &self.hats),
            ("ion", &self.ion),
            ("dense", &self.dense),
            ("constants", &self.constants),
            ("core", &self.core),
            ("spin_orbit_rows", &self.spin_orbit_rows),
            ("spin_orbit_columns", &self.spin_orbit_columns),
            ("exchange_phase", &self.exchange_phase),
            ("exchange_functions", &self.exchange_functions),
            ("exchange_rows", &self.exchange_rows),
            ("exchange_columns", &self.exchange_columns),
        ]
    }

    /// `dE/dtau` per atom.
    pub fn gradient(&self) -> Vec<[f64; 3]> {
        let mut total = vec![[0.0; 3]; self.overlap.len()];
        for (_, term) in self.terms() {
            for (sum, x) in total.iter_mut().zip(term) {
                for a in 0..3 {
                    sum[a] += x[a];
                }
            }
        }
        total
    }

    /// The force `-dE/dtau` per atom.
    pub fn forces(&self) -> Vec<[f64; 3]> {
        self.gradient().iter().map(|g| g.map(|x| -x)).collect()
    }
}

/// Whether the functional of `request` carries screened exact exchange.
fn is_hybrid(request: &Request) -> Result<bool> {
    Ok(models::functional(&request.functional)?
        .screened_exchange
        .is_some())
}

/// Raises for everything the engine does not differentiate.
pub fn check_request(request: &Request) -> Result<()> {
    let refuse = |what: &str| {
        Err(Error::Unsupported(format!(
            "crystal forces are not available for {what}"
        )))
    };
    let definition = models::functional(&request.functional)?;
    if request.single_precision {
        return refuse("float32 data");
    }
    if definition.requires_tau {
        return refuse(&format!("the meta-GGA {}", definition.name));
    }
    if definition.nonlocal_b.is_some() {
        return refuse(&format!("the rVV10 correction of {}", definition.name));
    }
    Ok(())
}

pub(super) fn validate<D: Derivatives>(d: &D) -> Result<()> {
    let l = d.layout();
    let fine = l.degeneracy > 0.0
        && l.degeneracy.is_finite()
        && l.reference.is_finite()
        && l.weights.iter().all(|w| w.is_finite())
        && l.row_owner.len() == l.m
        && l.projector_owner.len() == l.p
        && l.row_owner
            .iter()
            .chain(&l.projector_owner)
            .all(|&x| x < l.atoms)
        && d.augmentation().shape() == (l.p, l.p)
        && d.ion_operator().shape() == (l.p, l.p)
        && d.blocks().iter().all(|b| b.shape() == (l.p, l.p));
    if !fine {
        return Err(invalid("crystal gradient: inconsistent layout"));
    }
    Ok(())
}

/// The force engine: the output-density state, the contraction, the explicit terms.
pub fn gradient<D: Derivatives>(
    d: &D,
    input: &State,
    hubbard: Option<&dyn Hubbard>,
) -> Result<Breakdown> {
    check_request(&d.layout().request)?;
    validate(d)?;
    contract(d, &freeze(d, input, hubbard)?)
}

/// The contraction algebra on a consistent frozen state (no re-diagonalization).
pub fn contract<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Breakdown> {
    validate(d)?;
    frozen.state.check(d)?;
    let matrix = pulay::contract(d, frozen)?;
    let explicit = hellmann_feynman::explicit(d, frozen)?;
    let exchange = if is_hybrid(&d.layout().request)? {
        exchange::contract(d, &frozen.state)?
    } else {
        exchange::Terms::new(d.layout().atoms)
    };
    Ok(Breakdown {
        overlap: matrix.overlap,
        kinetic: matrix.kinetic,
        local: matrix.local,
        short_range: matrix.short_range,
        projection_rows: matrix.rows,
        projection_columns: matrix.columns,
        hats: explicit.hats,
        ion: explicit.ion,
        dense: explicit.dense,
        constants: explicit.constants,
        core: explicit.core,
        spin_orbit_rows: matrix.spin_orbit.rows,
        spin_orbit_columns: matrix.spin_orbit.columns,
        exchange_phase: exchange.phase,
        exchange_functions: exchange.functions,
        exchange_rows: exchange.rows,
        exchange_columns: exchange.columns,
    })
}
