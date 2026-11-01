//! What the toy test compares: the plain-DFT part of the analytic derivative
//! (`tr[D dH0] - tr[W dS]`), the layer's input assembled from a converged state, and exact
//! (Richardson) finite differences of the converged energy.
use super::{
    functional::Functional,
    scf::{solve, State, SIGNS},
    shell_weights, Kind, Matrix, Toy,
};
use mandacaru_core::periodic::hubbard_gradient::{
    gradient, gradient_with, Channel, Geometry, Gradient, Input, Shell,
};
use mandacaru_core::{quantum::C64, Result};
use nalgebra::DMatrix;

/// `dE/dtau` per atom and `dE/d eps`, plain part. `bare_weighted` builds `W` as `D H0 D`
/// instead of from the eigenvalues of the full `H` (a negative control).
pub fn plain(toy: &Toy, state: &State, bare_weighted: bool) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
    let mut grad = vec![[0.0; 3]; toy.atoms()];
    let mut strain = [[0.0; 3]; 3];
    let trace = |a: &Matrix, b: &Matrix| (a * b).trace().re;
    for (k, (&kpoint, &weight)) in toy.kpoints().iter().zip(&toy.weights).enumerate() {
        for (s, &sign) in SIGNS.iter().enumerate() {
            let d = &state.density[s][k];
            let w = if bare_weighted {
                let h0 = toy.bloch(Kind::Hopping, kpoint) + toy.onsite(sign);
                d * h0 * d
            } else {
                state.weighted[s][k].clone()
            };
            let term = |dh: &Matrix, ds: &Matrix| weight * (trace(d, dh) - trace(&w, ds));
            for (x, row) in grad.iter_mut().enumerate() {
                for (a, g) in row.iter_mut().enumerate() {
                    *g += term(
                        &toy.bloch_atom(Kind::Hopping, kpoint, x, a),
                        &toy.bloch_atom(Kind::Overlap, kpoint, x, a),
                    );
                }
            }
            for (a, row) in strain.iter_mut().enumerate() {
                for (b, g) in row.iter_mut().enumerate() {
                    *g += term(
                        &toy.bloch_strain(Kind::Hopping, kpoint, a, b),
                        &toy.bloch_strain(Kind::Overlap, kpoint, a, b),
                    );
                }
            }
        }
    }
    (grad, strain)
}

/// Everything the layer is given, owned, so a test can change one input before the call.
pub struct Packed {
    pub atoms: usize,
    pub images: Vec<[f64; 3]>,
    pub basis_centers: Vec<[f64; 3]>,
    pub projector_centers: Vec<[f64; 3]>,
    pub basis_owner: Vec<usize>,
    pub projector_owner: Vec<usize>,
    pub kpoints: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub shells: Vec<Shell>,
    pub projections: Vec<Matrix>,
    pub gradients: Vec<[DMatrix<f64>; 3]>,
    pub potentials: Vec<Vec<Matrix>>,
    pub density: Vec<Vec<Matrix>>,
}

impl Packed {
    pub fn new(toy: &Toy, state: &State) -> Self {
        let kpoints = toy.kpoints();
        Packed {
            atoms: toy.atoms(),
            images: toy.images(),
            basis_centers: toy.basis_centers(),
            projector_centers: toy.projector_centers(),
            basis_owner: toy.basis_owner(),
            projector_owner: toy.projector_owner(),
            weights: toy.weights.clone(),
            shells: (0..toy.shell_atoms.len())
                .map(|s| Shell {
                    projectors: (s * super::NP..(s + 1) * super::NP).collect(),
                    weights: shell_weights(),
                })
                .collect(),
            projections: kpoints
                .iter()
                .map(|&k| toy.bloch(Kind::Projection, k))
                .collect(),
            gradients: toy.blocks(Kind::Projection).gradients,
            potentials: state.potentials.clone(),
            density: state.density.clone(),
            kpoints,
        }
    }

    pub fn run(
        &self,
        displacement: Option<fn(&Geometry, usize, usize, usize) -> [f64; 3]>,
    ) -> Result<Gradient> {
        let channels: Vec<_> = (0..2)
            .map(|s| Channel {
                potentials: &self.potentials[s],
                density: &self.density[s],
            })
            .collect();
        let input = Input {
            geometry: Geometry {
                atoms: self.atoms,
                images: &self.images,
                basis_centers: &self.basis_centers,
                projector_centers: &self.projector_centers,
                basis_owner: &self.basis_owner,
                projector_owner: &self.projector_owner,
            },
            shells: &self.shells,
            weights: &self.weights,
            kpoints: &self.kpoints,
            projections: &self.projections,
            gradients: &self.gradients,
            channels: &channels,
        };
        match displacement {
            Some(map) => gradient_with(&input, map),
            None => gradient(&input),
        }
    }
}

/// Central difference with two Richardson levels (error `O(h^6)`) of `f` at zero.
pub fn richardson(f: impl Fn(f64) -> f64, h: f64) -> f64 {
    let central = |h: f64| (f(h) - f(-h)) / (2.0 * h);
    let level = |h: f64| (4.0 * central(0.5 * h) - central(h)) / 3.0;
    (16.0 * level(0.5 * h) - level(h)) / 15.0
}

pub const POSITION_STEP: f64 = 0.05;
pub const STRAIN_STEP: f64 = 0.02;

/// `dE/dtau` per atom from finite differences of the converged energy.
pub fn finite_forces(toy: &Toy, functional: &Functional, base: &State) -> Vec<[f64; 3]> {
    (0..toy.atoms())
        .map(|x| {
            std::array::from_fn(|a| {
                richardson(
                    |t| solve(&toy.displaced(x, a, t), functional, Some(&base.n)).energy,
                    POSITION_STEP,
                )
            })
        })
        .collect()
}

/// `dE/d eps_ab` from finite differences of the converged energy.
pub fn finite_strain(toy: &Toy, functional: &Functional, base: &State) -> [[f64; 3]; 3] {
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            richardson(
                |t| solve(&toy.deformed(a, b, t), functional, Some(&base.n)).energy,
                STRAIN_STEP,
            )
        })
    })
}

pub fn c(x: f64) -> C64 {
    C64::new(x, 0.0)
}
