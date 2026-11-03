//! The explicit atom-position derivative of the periodic screened exact exchange (HX-1).
//!
//! The force of a hybrid crystal is the derivative of the exchange energy at FIXED occupied
//! states (the AO coefficients `c_n(q)`, so the density matrices `D`), plus the orthonormality
//! terms the force engine already owns. At fixed `D` the energy that
//! [`crate::periodic::screened_exchange`] builds is
//!
//! ```text
//! E_x = -(a/2) mult sum_{k,q} w_k sum_{m,n} sum_Q v(Q) |rho_mn(Q)|^2 + E_oc[P]
//! rho_mn(Q) = FFT[conj(u_m(k)) u_n(q)](Q) + (1/Omega) sum_a Q^a_mn hat_a(Q),  Q^a_mn = a_m^+ B_a a_n,  a_n = C(q)^+ c_n
//! ```
//!
//! and only these terms depend on the position of atom `X` (whose basis functions, projectors,
//! spheres and compensation charges move rigidly with it):
//!
//! * the basis functions in the grid pair, bra and ket slot ([`Gradient::functions`], from the
//!   center derivatives of the Bloch sums, a [`Motion`]);
//! * the phase of the compensation shapes, `d hat_a / d R = -i Q hat_a` ([`Gradient::phase`]);
//! * the projections `C(k)`, through the compensation moments and through the projected density
//!   `P` of the one-center terms. Their response is a `p x m` matrix per (spin, k), `Z`, in the
//!   form the force engine contracts with `dC/dR`: `dE = 2 Re sum_k tr[dC_k Z_k]`
//!   ([`Gradient::compensation`], [`Gradient::sphere`], [`projection_forces`]).
//!
//! Not position terms: the screened kernel, `q_c`, the Gygi-Baldereschi head and the `Q = 0`
//! limit, the occupation factors, the grid origin phase. The Lowdin/overlap carry enters through
//! the hybrid Hamiltonian the engine diagonalizes (`K` is Hamiltonian-like), not here.
//!
//! The force engine calls [`explicit`] through `Derivatives::exchange_gradient`
//! (`setup::provider::hybrid` implements it with the basis functions moving) and chains the
//! response with [`projection_forces`] in `gradient::exchange`.
//!
//! The pair assembly is [`crate::periodic::screened_exchange`]'s own (`weighted_pairs`), called
//! with tangent inputs, so a change of the energy is a change of this derivative.
//!
//! A spinor crystal (time-reversal symmetric: [`Crystal`] only builds a spinor hybrid in that
//! case) is the same pass with the rows doubled (`2m`, `D_spin = D`, occupation 1; SH-F1). The
//! motion's owners and the responses are always in the SCALAR frame of one spin block: `owner`
//! names an atom for each of the `m` functions (a bra row `(sigma, mu)` goes to `owner[mu]`) and
//! every `Z` is `p x m`, summed over the two components because `dC/dR` is spin diagonal.
mod chain;
mod pass;
mod sphere;

use super::{
    screened_augmentation::Augmentation,
    screened_exchange::{ExxDivTreatment, Screening},
    Crystal,
};
use crate::{math::storage::MatrixSource, quantum::C64, Buffer, Result};
use nalgebra::DMatrix;

pub use chain::{projection_forces, ProjectionForces};

type Matrix = DMatrix<C64>;

/// How the basis functions of the grid pair move with their atoms.
pub trait Motion {
    /// The atom each basis function belongs to.
    fn owner(&self) -> &[usize];
    /// `d chi_mu(k) / d c_mu^axis` at mesh point `k`: the derivative of the Bloch sums (the
    /// samples `Crystal` stores, phase included) with respect to the center of the function,
    /// i.e. minus the Cartesian derivative. Three arrays of `m x N`, function-major.
    fn center_derivative(&self, c: &Crystal, k: usize) -> Result<[Buffer<C64>; 3]>;
}

/// `dE_x/dR` at fixed occupied states, term by term (atomic units; derivatives, not forces).
pub struct Gradient {
    /// The grid-plus-compensation exchange energy these terms differentiate: equal to the
    /// exchange energy of the augmented build minus the one-center energy.
    pub energy: f64,
    /// The one-center energy `E_oc[P]` (its derivative is [`Gradient::sphere`]).
    pub one_center_energy: f64,
    /// Per atom: the phase of its compensation shapes.
    pub phase: Vec<[f64; 3]>,
    /// Per atom: its basis functions moving in the grid pair; `None` without a [`Motion`].
    pub functions: Option<Functions>,
    /// `Z` per (spin, k), `spin * nk + k`, `p x m`: the response of the compensation moments to
    /// the projections.
    pub compensation: Vec<Matrix>,
    /// The same for the one-center terms (`-a w_k O C^+ D`).
    pub sphere: Vec<Matrix>,
    /// The terms above split by the spin component of the pair rows that produced them (one
    /// entry, or two for a spinor): `phase`, `functions` and `compensation` are their sums. The
    /// split is how a test builds a wrong variant (a component dropped or mislabeled) from the
    /// production pieces.
    pub components: Vec<Component>,
}

/// What the pair rows of ONE spin component contribute (a row `(sigma, mu)` of a spinor belongs to
/// component `sigma`; a scalar crystal has the single component 0).
pub struct Component {
    pub phase: Vec<[f64; 3]>,
    /// Per atom: the bra functions of the rows of this component moving.
    pub bra: Vec<[f64; 3]>,
    /// Per atom: the ket functions (the state's own component) moving.
    pub ket: Vec<[f64; 3]>,
    /// `Z` per (spin, k), `p x m`: this component's part of the response of the compensation moments.
    pub compensation: Vec<Matrix>,
}

/// The basis functions of the grid pair `conj(u_m(k)) u_n(q)` moving: the bra slot (the target
/// point's functions) and the ket slot (the occupied state's own functions), per atom.
pub struct Functions {
    pub bra: Vec<[f64; 3]>,
    pub ket: Vec<[f64; 3]>,
}

impl Functions {
    pub fn total(&self) -> Vec<[f64; 3]> {
        self.bra
            .iter()
            .zip(&self.ket)
            .map(|(b, k)| std::array::from_fn(|a| b[a] + k[a]))
            .collect()
    }
}

impl Gradient {
    /// `Z` of the compensation charges and the spheres together.
    pub fn response(&self) -> Vec<Matrix> {
        self.compensation
            .iter()
            .zip(&self.sphere)
            .map(|(a, b)| a + b)
            .collect()
    }
}

/// The explicit terms of the exchange derivative for the density matrices `matrices` (the
/// layout of `Crystal::hybrid_exchange`), with the compensation charges and the one-center
/// model of `augmentation`.
pub fn explicit<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    augmentation: &Augmentation,
    motion: Option<&dyn Motion>,
) -> Result<Gradient> {
    if let Some(motion) = motion {
        if motion.owner().len() != c.m
            || motion
                .owner()
                .iter()
                .any(|&a| a >= augmentation.sites().atoms.len())
        {
            return Err(super::invalid(
                "the motion's owners must name an atom for each of the m basis functions",
            ));
        }
    }
    let grid = pass::run(c, matrices, screening, exxdiv, augmentation, motion)?;
    let (sphere, one_center_energy) = sphere::response(c, matrices, screening, augmentation)?;
    let atoms = augmentation.sites().atoms.len();
    let add = |pick: &dyn Fn(&Component) -> &Vec<[f64; 3]>| -> Vec<[f64; 3]> {
        (0..atoms)
            .map(|atom| {
                std::array::from_fn(|a| grid.components.iter().map(|p| pick(p)[atom][a]).sum())
            })
            .collect()
    };
    let functions = motion.map(|_| Functions {
        bra: add(&|p| &p.bra),
        ket: add(&|p| &p.ket),
    });
    let phase = add(&|p| &p.phase);
    let compensation = (0..c.spins * c.points.len())
        .map(|at| {
            grid.components
                .iter()
                .fold(Matrix::zeros(c.p, c.m), |z, p| z + &p.compensation[at])
        })
        .collect();
    Ok(Gradient {
        energy: grid.energy,
        one_center_energy,
        phase,
        functions,
        compensation,
        sphere,
        components: grid.components,
    })
}
