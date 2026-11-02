//! What the toy test compares against: exact (two-level Richardson) finite differences of the
//! converged energy, and the rigid rotation of a model.
use super::{
    scf::{observe, solve, Solution, TOLERANCE},
    shells, Model, Prepared,
};
use crate::hubbard_gradient_support::{
    reference::{richardson, POSITION_STEP},
    Kind, Matrix,
};
use mandacaru_core::periodic::{
    gradient::State,
    hubbard_gradient::{gradient as layer, operator, Channel, Geometry as LayerGeometry, Input},
};

/// `dE/dtau` per atom from reconverged energies, each started from the converged state.
pub fn finite_gradient(model: &Model, base: &Solution) -> Vec<[f64; 3]> {
    (0..model.toy.atoms())
        .map(|x| {
            std::array::from_fn(|a| {
                richardson(
                    |t| {
                        let moved = Model {
                            toy: model.toy.displaced(x, a, t),
                            ..model.clone()
                        };
                        solve(&Prepared::new(&moved), Some(&base.output), TOLERANCE).energy
                    },
                    POSITION_STEP,
                )
            })
        })
        .collect()
}

/// The model with every atom and the lattice vector rotated by `rotation`.
pub fn rotated(model: &Model, rotation: [[f64; 3]; 3]) -> Model {
    let turn = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| (0..3).map(|b| rotation[a][b] * v[b]).sum())
    };
    let mut toy = model.toy.clone();
    toy.tau = toy.tau.iter().map(|&t| turn(t)).collect();
    toy.a1 = turn(toy.a1);
    Model {
        toy,
        ..model.clone()
    }
}

pub fn turn(rotation: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| (0..3).map(|b| rotation[a][b] * v[b]).sum())
}

/// The largest component of `a - b`.
pub fn deviation(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

pub fn largest(a: &[[f64; 3]]) -> f64 {
    a.iter().flatten().fold(0.0, |m, x| m.max(x.abs()))
}

/// `dE_U/dtau` per atom from the HU-10a layer, on the densities of `state` and the potentials
/// `V` of its occupations (restricted: one channel with the weights doubled).
pub fn hubbard_layer(prep: &Prepared, state: &State) -> Vec<[f64; 3]> {
    let (model, toy) = (&prep.model, &prep.model.toy);
    let n = observe(prep, state).n;
    let down = n.get(1).unwrap_or(&n[0]);
    let evaluations: Vec<_> = (0..n[0].len())
        .map(|i| model.correction.evaluate([&n[0][i], &down[i]]))
        .collect();
    let potentials: Vec<Vec<Matrix>> = (0..model.spins())
        .map(|s| evaluations.iter().map(|e| e.potential[s].clone()).collect())
        .collect();
    let densities: Vec<Vec<Matrix>> = (0..model.spins())
        .map(|s| state.bands[s].iter().map(|b| b.density()).collect())
        .collect();
    let channels: Vec<_> = (0..model.spins())
        .map(|s| Channel {
            potentials: &potentials[s],
            density: &densities[s],
        })
        .collect();
    let gradients = toy.blocks(Kind::Projection).gradients;
    let (images, kpoints) = (toy.images(), toy.kpoints());
    let (centers, projectors) = (toy.basis_centers(), toy.projector_centers());
    let (rows, columns) = (toy.basis_owner(), toy.projector_owner());
    let weights: Vec<f64> = toy.weights.iter().map(|w| w * model.degeneracy).collect();
    let shell_list = shells(toy);
    let input = Input {
        geometry: LayerGeometry {
            atoms: toy.atoms(),
            images: &images,
            basis_centers: &centers,
            projector_centers: &projectors,
            basis_owner: &rows,
            projector_owner: &columns,
        },
        shells: &shell_list,
        weights: &weights,
        kpoints: &kpoints,
        projections: &prep.projection,
        gradients: &gradients,
        channels: &channels,
    };
    let forces = layer(&input).unwrap().forces;
    forces.iter().map(|f| f.map(|x| -x)).collect()
}

/// `O_s` built from `V_Hub` alone: the complete `V` with the double-counting shift added back.
pub fn hubbard_without_double_counting(prep: &Prepared, state: &State) -> Vec<Matrix> {
    let (model, toy) = (&prep.model, &prep.model.toy);
    let n = observe(prep, state).n;
    let down = n.get(1).unwrap_or(&n[0]);
    let evaluations: Vec<_> = (0..n[0].len())
        .map(|i| model.correction.evaluate([&n[0][i], &down[i]]))
        .collect();
    (0..model.spins())
        .map(|s| {
            let v: Vec<Matrix> = evaluations
                .iter()
                .map(|e| &e.potential[s] + &e.double_counting[s])
                .collect();
            operator(toy.projectors(), &shells(toy), &v).unwrap()
        })
        .collect()
}
