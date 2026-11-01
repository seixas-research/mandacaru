//! The +U increment of the crystal force and of the strain derivative (HU-10a), with no engine.
//!
//! # Derivation
//!
//! The +U energy depends on the state only through the shell occupations `n_I = W^T R_I W`,
//! with `R^s = sum_k w_k C_k^dagger D^s_k C_k` restricted to the projectors of the shell `I`.
//! `D^s_k` is the density matrix of the (non-orthogonal) basis, `D = sum_i f_i c_i c_i^dagger`
//! with `c^dagger S c = 1`; `C_{mu p}(k) = <chi_mu,k | p_p,k>` is the basis-projector overlap;
//! `W` is the radial weight of the shell, a constant of the dataset (`dW/dX = 0`, also under
//! strain). With `V_I = dE_U/dn_I` (the complete `V = V_Hub - V_dc`), `dE_U = sum_I tr[V_I dn_I]
//! = sum tr[O dR]` with the `p x p` operator `O = W V W^T` scattered over the shell projectors.
//!
//! The total energy is stationary in the orbitals under `c^dagger S c = 1`. With the Fock matrix
//! `H = H0 + C O C^dagger` and the energy-weighted density `W_e = sum f_i e_i c_i c_i^dagger`
//! (`e_i` the eigenvalues of that full `H`), the derivative along any coordinate `x` is
//!
//! `dE/dx = tr[D dH0/dx] - tr[W_e dS/dx] + sum_s Re 2 tr[D_s dC O_s C^dagger]`.
//!
//! *Automatic* (already in the engine once `O` is in the Fock matrix): the term `-tr[W_e dS/dx]`
//! carries +U because the eigenvalues are those of `H0 + C O C^dagger`; no `dV/dn` term appears
//! (stationarity); no separate `E_dc` derivative appears, because `E_dc` depends on a position
//! only through `n` and `O` is built from the complete `V` including `V_dc` (a force that builds
//! `O` from `V_Hub` alone is wrong). *New with +U* (this file): the last term, the explicit
//! dependence of `R` at fixed `D` through `C(x)`, and nothing else. Two traps stay with the
//! caller: `O` and `D` must be those of the converged output density (`O(n_out)`, not the mixer's
//! input), and `W_e` takes the unshifted eigenvalues.
//!
//! `C_{mu p}(k) = sum_R e^{-ik.R} B_R[mu, p]`, with `B_R` the overlap of the basis function `mu`
//! (centered at `c_mu + R`) with the projector `p` (centered at `c_p`): it depends on the
//! geometry only through `d_R = c_mu + R - c_p`. Writing `G^a_R = dB_R/dd^a` (the derivative with
//! respect to the center of `mu`, one real `m x p` block per image and axis):
//!
//! * *Atom X moving* (`dd/dtau_X = [own(mu) = X] - [own(p) = X]`, atoms rigid with their
//!   functions and projectors): `dC^X = diag(own_X rows) G(k) - G(k) diag(own_X columns)` with
//!   `G(k) = sum_R e^{-ik.R} G_R`. The contribution to the force is `-dE/dtau_X`. Every entry of
//!   `G` enters once with each sign, so the forces sum to zero.
//! * *Strain* (cell, atoms and grid scale together, functions and spheres rigid, fractional `k`
//!   fixed): `dd^b/d eps_ab -> d_R^b` and `dC/d eps_ab = sum_R e^{-ik.R} G^a_R d_R^b`. The
//!   phases `k.R` do not change under the scaling, the weights `w_k` are fractional, `E_U` has no
//!   volume factor (it is an energy per cell, and a stress divides by the cell volume outside):
//!   those terms vanish, as do `dW`, a kernel derivative and a double-counting derivative.
//!
//! # Conventions
//!
//! Atomic units; every quantity is per spin channel (no hidden restricted factor: a restricted
//! caller passes the weights and the potential so that `sum_s tr[O_s dR_s]` is `dE_U`). The
//! phase is `e^{-ik.R}` with Cartesian `k` and `R`. `gradient` returns the force `-dE_U/dtau`
//! per atom and the energy derivative `dE_U/d eps_ab` (not divided by the volume).
//! Memory: `O(m p)` per `k` beyond the inputs; the caller owns the `Context` reservation.
use super::invalid;
use crate::{quantum::C64, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// A Hubbard shell: its projectors and the radial weights `W` (`projectors x orbitals`).
pub struct Shell {
    pub projectors: Vec<usize>,
    pub weights: DMatrix<f64>,
}

/// Where the basis functions, projectors and image cells sit (Bohr, Cartesian).
pub struct Geometry<'a> {
    pub atoms: usize,
    /// Cartesian lattice vectors `R` of the images summed in `C(k)`.
    pub images: &'a [[f64; 3]],
    pub basis_centers: &'a [[f64; 3]],
    pub projector_centers: &'a [[f64; 3]],
    /// The atom each basis function (projector) belongs to; the atom owns its rows (columns).
    pub basis_owner: &'a [usize],
    pub projector_owner: &'a [usize],
}

/// One spin channel: `V` per shell (`orbitals x orbitals`) and `D_k` per k-point.
pub struct Channel<'a> {
    pub potentials: &'a [Matrix],
    pub density: &'a [Matrix],
}

pub struct Input<'a> {
    pub geometry: Geometry<'a>,
    pub shells: &'a [Shell],
    pub weights: &'a [f64],
    pub kpoints: &'a [[f64; 3]],
    /// `C_k`, `m x p`.
    pub projections: &'a [Matrix],
    /// `gradients[image][axis]`: the unweighted real blocks `G^a_R`, `m x p`.
    pub gradients: &'a [[DMatrix<f64>; 3]],
    pub channels: &'a [Channel<'a>],
}

/// `forces[X] = -dE_U/dtau_X`; `strain[a][b] = dE_U/d eps_ab`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub forces: Vec<[f64; 3]>,
    pub strain: [[f64; 3]; 3],
}

/// The vector `d_R = c_mu + R - c_p` of the pair (`mu`, `p`) in the image `image`.
pub fn displacement(g: &Geometry, image: usize, mu: usize, p: usize) -> [f64; 3] {
    std::array::from_fn(|b| g.basis_centers[mu][b] + g.images[image][b] - g.projector_centers[p][b])
}

/// `O = W V W^T` scattered into the `p x p` projector space (`tr[V dn] = tr[O dR]`).
pub fn operator(projectors: usize, shells: &[Shell], potentials: &[Matrix]) -> Result<Matrix> {
    if shells.len() != potentials.len() {
        return Err(invalid("hubbard gradient: one potential per shell"));
    }
    let mut o = Matrix::zeros(projectors, projectors);
    let mut seen = vec![false; projectors];
    for (shell, v) in shells.iter().zip(potentials) {
        let (np, nm) = (shell.projectors.len(), shell.weights.ncols());
        if shell.weights.nrows() != np || v.nrows() != nm || v.ncols() != nm {
            return Err(invalid(
                "hubbard gradient: shell weights or potential shape",
            ));
        }
        for &q in &shell.projectors {
            if q >= projectors || std::mem::replace(&mut seen[q], true) {
                return Err(invalid(
                    "hubbard gradient: projector out of range or in two shells",
                ));
            }
        }
        for i in 0..np {
            for j in 0..np {
                let mut sum = C64::new(0.0, 0.0);
                for a in 0..nm {
                    for b in 0..nm {
                        sum += v[(a, b)] * (shell.weights[(i, a)] * shell.weights[(j, b)]);
                    }
                }
                o[(shell.projectors[i], shell.projectors[j])] += sum;
            }
        }
    }
    Ok(o)
}

fn validate(input: &Input) -> Result<(usize, usize)> {
    let g = &input.geometry;
    let (m, p) = (g.basis_centers.len(), g.projector_centers.len());
    let nk = input.kpoints.len();
    let counts = [
        (g.basis_owner.len() == m, "basis_owner"),
        (g.projector_owner.len() == p, "projector_owner"),
        (input.weights.len() == nk, "weights"),
        (input.projections.len() == nk, "projections"),
        (input.gradients.len() == g.images.len(), "gradients"),
        (
            !input.channels.is_empty() && nk > 0,
            "channels and k-points",
        ),
        (!input.shells.is_empty(), "shells"),
    ];
    if let Some((_, name)) = counts.iter().find(|(ok, _)| !ok) {
        return Err(invalid(format!(
            "hubbard gradient: inconsistent length of {name}"
        )));
    }
    if g.basis_owner
        .iter()
        .chain(g.projector_owner)
        .any(|&x| x >= g.atoms)
    {
        return Err(invalid("hubbard gradient: owner atom out of range"));
    }
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    let flat = |v: &[[f64; 3]]| v.iter().flatten().all(|x| x.is_finite());
    if !finite(input.weights)
        || !flat(input.kpoints)
        || !flat(g.images)
        || !flat(g.basis_centers)
        || !flat(g.projector_centers)
    {
        return Err(invalid("hubbard gradient: nonfinite geometry or weights"));
    }
    if input.projections.iter().any(|c| c.shape() != (m, p))
        || input
            .gradients
            .iter()
            .flatten()
            .any(|b| b.shape() != (m, p))
    {
        return Err(invalid(
            "hubbard gradient: projection and gradient blocks are m x p",
        ));
    }
    for channel in input.channels {
        if channel.density.len() != nk || channel.density.iter().any(|d| d.shape() != (m, m)) {
            return Err(invalid("hubbard gradient: one m x m density per k-point"));
        }
    }
    Ok((m, p))
}

/// The +U force and strain derivative with the exact displacement `d_R`.
pub fn gradient(input: &Input) -> Result<Gradient> {
    gradient_with(input, displacement)
}

/// [`gradient`] with the displacement map of the strain term given (the one seam: a test
/// replaces it to show that a wrong `d_R` is caught).
pub fn gradient_with(
    input: &Input,
    displacement: fn(&Geometry, usize, usize, usize) -> [f64; 3],
) -> Result<Gradient> {
    let (m, p) = validate(input)?;
    let g = &input.geometry;
    let mut grad = vec![[0.0; 3]; g.atoms];
    let mut strain = [[0.0; 3]; 3];
    let operators = input
        .channels
        .iter()
        .map(|c| operator(p, input.shells, c.potentials))
        .collect::<Result<Vec<_>>>()?;
    for (k, (&w, kpoint)) in input.weights.iter().zip(input.kpoints).enumerate() {
        let c_adjoint = input.projections[k].adjoint();
        for (channel, o) in input.channels.iter().zip(&operators) {
            // Y = O C^dagger D (p x m): the explicit term is 2 Re sum dC[mu,p] Y[p,mu].
            let y = &(o * &c_adjoint) * &channel.density[k];
            for (image, blocks) in input.gradients.iter().enumerate() {
                let arg = -(0..3).map(|b| kpoint[b] * g.images[image][b]).sum::<f64>();
                let phase = C64::new(arg.cos(), arg.sin());
                for mu in 0..m {
                    for q in 0..p {
                        let weight = 2.0 * w * (phase * y[(q, mu)]).re;
                        let d = displacement(g, image, mu, q);
                        for (a, block) in blocks.iter().enumerate() {
                            let term = weight * block[(mu, q)];
                            grad[g.basis_owner[mu]][a] += term;
                            grad[g.projector_owner[q]][a] -= term;
                            for (b, &db) in d.iter().enumerate() {
                                strain[a][b] += term * db;
                            }
                        }
                    }
                }
            }
        }
    }
    let forces = grad.iter().map(|v| v.map(|x| -x)).collect();
    Ok(Gradient { forces, strain })
}
