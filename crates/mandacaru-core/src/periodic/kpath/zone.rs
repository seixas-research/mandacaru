//! The irreducible zone of a k-mesh.
//!
//! The group is the set of rotation parts `W` of the crystal's space group (found by
//! [`space_group::find`], which keeps only operations that map every atom onto an atom of its
//! own species) that send the mesh onto itself, with time reversal (`k -> -k`) added: the
//! Hamiltonians of a Bloch run are real. A translation moves no k-point, so only the distinct
//! rotations reach the reduction. A mesh the lattice symmetry does not fit (a 2 x 1 mesh of a
//! square lattice) keeps the subgroup that does, which is the group the supercell itself has.
use crate::periodic::{
    invalid,
    setup::symmetry::MeshIntegers,
    space_group::{find_with_context, Cell},
    spectrum::{self, Mesh},
};
use crate::{Context, Result};
use nalgebra::Matrix3;
use std::collections::HashSet;

/// The relative `L1` difference between the spectral functions of two k-points of one star above
/// which the state is not symmetric under the point group.
pub const AUDIT_TOLERANCE: f64 = 1e-6;

/// A mesh and its reduction.
pub struct Zone {
    /// Operations of the space group, before any filter.
    pub space_group_operations: usize,
    /// Distinct rotations among them.
    pub distinct_rotations: usize,
    /// The rotations (row-major `W`, `k' = k W`) that preserve the mesh.
    pub rotations: Vec<[i32; 9]>,
    /// The full mesh, one point per star, the weight of each star and the star of every point.
    pub mesh: Mesh,
}

impl Zone {
    /// The index into `mesh.points` of the star that holds the fractional point `k`
    /// (a mesh point modulo a reciprocal-lattice vector).
    pub fn star_of(&self, k: [f64; 3]) -> Result<usize> {
        let same =
            |a: &[f64; 3]| (0..3).all(|i| ((a[i] - k[i]) - (a[i] - k[i]).round()).abs() < 1e-8);
        let full = self
            .mesh
            .full
            .iter()
            .position(same)
            .ok_or_else(|| invalid(format!("k = {k:?} is not a point of the mesh")))?;
        Ok(self.mesh.mapping[full])
    }
}

/// The space group of the cell (`lattice` rows, fractional positions, species `numbers`, Cartesian
/// tolerance `symprec` in the lattice's unit) and the reduction of the `size` mesh by it.
pub fn irreducible_zone(
    ctx: &Context,
    lattice: &Matrix3<f64>,
    fractional: &[[f64; 3]],
    numbers: &[u32],
    size: [usize; 3],
    gamma: bool,
    symprec: f64,
) -> Result<Zone> {
    if !symprec.is_finite() || symprec <= 0. {
        return Err(invalid(
            "the symmetry tolerance must be positive and finite",
        ));
    }
    let cell = Cell {
        lattice: *lattice,
        fractional,
        labels: numbers,
    };
    let ops = find_with_context(ctx, &cell, symprec)?;
    let points = MeshIntegers::new(size, gamma)?;
    let mut seen = HashSet::new();
    let mut distinct = 0;
    let mut rotations = vec![];
    for (s, w) in ops.rotations.iter().enumerate() {
        if !seen.insert(ops.op_factors[s].0) {
            continue;
        }
        distinct += 1;
        if points.preserved_by(w) {
            rotations.push(std::array::from_fn(|i| w[(i / 3, i % 3)]));
        }
    }
    let mesh = spectrum::mesh(ctx, size, gamma, &rotations, true)?;
    Ok(Zone {
        space_group_operations: ops.len(),
        distinct_rotations: distinct,
        rotations,
        mesh,
    })
}
