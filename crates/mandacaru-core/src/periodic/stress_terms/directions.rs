//! The strain directions the stress is contracted along.
//!
//! The strain derivative `g_ab = dF/d eps_ab` is linear in the strain, and the stress
//! `sigma = (g + g^T) / (2 Omega)` of a crystal is the symmetric part, averaged over the point group
//! of its mesh. For an orthonormal basis `E_1 .. E_n` (Frobenius) of the symmetric tensors that
//! every operation of the group leaves invariant, the projection of `g` on that space is
//! `sum_i <g, E_i> E_i`, and it is the symmetrized group average of `g` (`<avg g, E> = <g, E>` for an
//! invariant `E`). So the terms are contracted along `n` directions, one combined tangent
//! `sum_ab E_ab d/d eps_ab` each, and `sigma` is rebuilt from the `n` numbers:
//!
//! | family | n |
//! |---|---|
//! | cubic | 1 |
//! | hexagonal, trigonal, tetragonal | 2 |
//! | orthorhombic | 3 |
//! | monoclinic | 4 |
//! | triclinic, or no operations | 6 |
//!
//! The nine unit tensors of [`Directions::all_entries`] reproduce every entry of `g`, the
//! unsymmetrized ones included (the finite-difference tests gate them), and average over the group
//! afterwards as the nine-tangent contraction did.
use super::Tensor;
use crate::{
    periodic::{invalid, point_group::PointGroup},
    Result,
};

/// A strain direction: `E_ab` multiplies `d / d eps_ab`.
pub type Direction = [[f64; 3]; 3];

/// The directions of a contraction and how the `n` numbers become a tensor.
#[derive(Clone, Debug)]
pub struct Directions {
    basis: Vec<Direction>,
    /// The operations averaged over after the assembly of [`Directions::all_entries`]; the
    /// invariant basis is already the average.
    average: Option<PointGroup>,
    all_entries: bool,
}

const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

fn dot(a: &Direction, b: &Direction) -> f64 {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| a[i][j] * b[i][j])
        .sum()
}

impl Directions {
    /// The orthonormal basis of the symmetric tensors invariant under `group` (`None`: the six
    /// symmetric unit tensors). Raises for a group whose invariants are not those of a crystal
    /// family.
    pub fn invariant(group: Option<&PointGroup>) -> Result<Self> {
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        let mut basis: Vec<Direction> = Vec::new();
        for &(a, b) in &VOIGT {
            let mut e = [[0.; 3]; 3];
            if a == b {
                e[a][a] = 1.;
            } else {
                e[a][b] = scale;
                e[b][a] = scale;
            }
            if let Some(group) = group {
                group.tensor(&mut e);
            }
            for earlier in &basis {
                let overlap = dot(&e, earlier);
                for i in 0..3 {
                    for j in 0..3 {
                        e[i][j] -= overlap * earlier[i][j];
                    }
                }
            }
            let norm = dot(&e, &e).sqrt();
            if norm > 1e-8 {
                for row in &mut e {
                    for x in row {
                        *x /= norm;
                    }
                }
                basis.push(e);
            }
        }
        if !matches!(basis.len(), 1 | 2 | 3 | 4 | 6) {
            return Err(invalid(format!(
                "the point group leaves {} independent symmetric strain tensors invariant, which \
                 is not the invariant space of any crystal family",
                basis.len()
            )));
        }
        Ok(Self {
            basis,
            average: None,
            all_entries: false,
        })
    }

    /// The nine unit tensors `e_ab`, every entry of `g_ab` unsymmetrized; `group` is averaged over
    /// afterwards. For the finite-difference tests and the torque diagnostic: never the
    /// production contraction.
    pub fn all_entries(group: Option<PointGroup>) -> Self {
        let basis = (0..9)
            .map(|index| {
                let mut e = [[0.; 3]; 3];
                e[index / 3][index % 3] = 1.;
                e
            })
            .collect();
        Self {
            basis,
            average: group,
            all_entries: true,
        }
    }

    pub fn as_slice(&self) -> &[Direction] {
        &self.basis
    }

    pub fn len(&self) -> usize {
        self.basis.len()
    }

    pub fn is_empty(&self) -> bool {
        self.basis.is_empty()
    }

    /// Whether this is the nine-entry contraction (which also reports the antisymmetric part).
    pub fn is_all_entries(&self) -> bool {
        self.all_entries
    }

    /// `sum_i values_i E_i`, averaged over the group of [`Directions::all_entries`].
    pub fn assemble(&self, values: &[f64]) -> Tensor<f64> {
        debug_assert_eq!(values.len(), self.basis.len());
        let mut out = [[0.; 3]; 3];
        for (value, e) in values.iter().zip(&self.basis) {
            for a in 0..3 {
                for b in 0..3 {
                    out[a][b] += value * e[a][b];
                }
            }
        }
        if let Some(group) = &self.average {
            group.tensor(&mut out);
        }
        out
    }

    /// The part of a full tensor `g_ab` this contraction reports: `<g, E_i>` rebuilt.
    pub fn project(&self, g: &Tensor<f64>) -> Tensor<f64> {
        let values: Vec<f64> = self.basis.iter().map(|e| dot(g, e)).collect();
        self.assemble(&values)
    }

    /// `sum_ab E_ab t_ab` of a full tensor.
    pub fn along(&self, g: &Tensor<f64>) -> Vec<f64> {
        self.basis.iter().map(|e| dot(g, e)).collect()
    }
}
