//! The space-group operations a symmetry-reduced k mesh was built with, and the symmetrization
//! of the force and of the stress they require (CF-6).
//!
//! A mesh reduced by the operations `{(W_s, t_s)}` (`x -> W x + t` in fractional coordinates of the
//! cell) holds one k-point per star, weighted by the star's size. The SCF symmetrizes the density
//! and the channel moments with the same operations, so every FIELD the engines read is already
//! symmetric; what is not is the k-sum of the matrix terms, which visited one point of every
//! star. The full-mesh sum is the group average of the reduced one:
//!
//! `F_A = (1/n) sum_s R_s^T F^red_{s(A)}`,  `g = (1/n) sum_s R_s g^red R_s^T`,
//!
//! with `R_s = L W_s L^{-1}` the Cartesian rotation (`L` the lattice, columns), `s(A)` the atom
//! the operation sends `A` to, and `g_ab = dF/d eps_ab` the strain derivative (a rank-2 tensor).
//! Every term of the engines' breakdowns is a k-sum of covariant contributions or an explicit
//! term of the symmetric fields, so each is averaged on its own and the totals stay sums of terms.
//!
//! The operations come from the snapshot's `setup` block, written by the setup that reduced the
//! mesh. They are refused unless they reproduce what the snapshot already carries: an integer
//! `W` and an orthogonal `R`, atom maps that are permutations sending every atom onto an atom,
//! and, for every operation, the grid-node permutation the SCF symmetrizes the density with.
use super::invalid;
use crate::{product, Context, Error, Result};
use nalgebra::{Matrix3, Vector3};
use serde_json::Value;

/// Atoms are matched to the images of their positions to this distance (Bohr).
const ATOM_TOLERANCE: f64 = 1e-3;
/// `W N_i / N_j` is an integer matrix to this tolerance, the node shift `t N` to the next one.
const SCALE_TOLERANCE: f64 = 1e-9;
const SHIFT_TOLERANCE: f64 = 1e-6;
const ORTHOGONALITY: f64 = 1e-8;
/// The numbers of one operation held at a time while it is read (a rotation, a translation).
const SCRATCH_NUMBERS: usize = 16;

/// The kept operations of a reduced mesh: the Cartesian rotations and the atom each atom goes to.
#[derive(Clone, Debug, PartialEq)]
pub struct PointGroup {
    rotations: Vec<Matrix3<f64>>,
    atom_maps: Vec<Vec<usize>>,
}

/// The heap a group of `n` operations on `atoms` atoms takes: a `3 x 3` rotation, an atom map and
/// the vector headers of each.
pub fn bytes(n: usize, atoms: usize) -> Result<usize> {
    let per_operation = product(&[atoms, std::mem::size_of::<usize>()])?
        .checked_add(std::mem::size_of::<Matrix3<f64>>() + std::mem::size_of::<Vec<usize>>())
        .ok_or(Error::Overflow)?;
    product(&[n, per_operation])
}

fn count(value: &Value, what: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid(format!("the symmetry operations: {what} must be an index")))
}

fn numbers(value: &Value, n: usize, what: &str) -> Result<Vec<f64>> {
    let list = value
        .as_array()
        .filter(|a| a.len() == n)
        .ok_or_else(|| invalid(format!("the symmetry operations: {what} needs {n} numbers")))?;
    let mut out = Vec::with_capacity(n);
    for x in list {
        out.push(
            x.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                invalid(format!("the symmetry operations: {what} must be finite"))
            })?,
        );
    }
    Ok(out)
}

impl PointGroup {
    /// Builds and checks the group of a snapshot.
    ///
    /// `operations` is the `operations` object of the `setup` block (`rotations`: `n x 9` integers,
    /// row-major `W`; `translations`: `n x 3`; `atom_maps`: `n` lists of one atom per atom).
    /// `lattice` holds the cell vectors as columns (Bohr), `centers` the atoms (Bohr),
    /// `permutations` the grid-node permutation of every operation the snapshot symmetrizes with.
    ///
    /// The group is reserved from `ctx` before anything is allocated (its own bytes, [`bytes`], and
    /// the scratch of the check) and the reservation ends when it returns: a holder of the group
    /// reserves [`bytes`] for as long as it keeps it. Every count is read from the lengths of the
    /// arrays the file holds, and equals the number of node permutations the snapshot carries.
    pub fn parse(
        ctx: &Context,
        operations: &Value,
        lattice: &Matrix3<f64>,
        centers: &[[f64; 3]],
        shape: [usize; 3],
        permutations: &[Vec<usize>],
    ) -> Result<Self> {
        let rotations = operations["rotations"]
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("the symmetry operations need their rotations"))?;
        let n = rotations.len();
        let translations = operations["translations"]
            .as_array()
            .filter(|a| a.len() == n)
            .ok_or_else(|| invalid("the symmetry operations need one translation each"))?;
        let maps = operations["atom_maps"]
            .as_array()
            .filter(|a| a.len() == n)
            .ok_or_else(|| invalid("the symmetry operations need one atom map each"))?;
        if permutations.len() != n {
            return Err(invalid(format!(
                "the setup block lists {n} symmetry operations, the snapshot symmetrizes with {}",
                permutations.len()
            )));
        }
        let inverse = lattice
            .try_inverse()
            .ok_or_else(|| invalid("the lattice must be nonsingular"))?;
        // Reserved before the first allocation: the group, then the fractional centers and the
        // visited flags of an atom map (per atom `3 f64` and one byte) and the few numbers an
        // operation is read into.
        let scratch = product(&[centers.len(), 3 * std::mem::size_of::<f64>() + 1])?
            .checked_add(SCRATCH_NUMBERS * std::mem::size_of::<f64>())
            .ok_or(Error::Overflow)?;
        let _memory = ctx.reserve(
            bytes(n, centers.len())?
                .checked_add(scratch)
                .ok_or(Error::Overflow)?,
        )?;
        let mut fractional: Vec<Vector3<f64>> = Vec::with_capacity(centers.len());
        fractional.extend(centers.iter().map(|c| inverse * Vector3::from(*c)));
        let nodes = product(&shape)?;
        let size = shape.map(|v| v as i64);
        let (mut cartesian, mut atom_maps) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for s in 0..n {
            let w = numbers(&rotations[s], 9, "a rotation")?;
            let w = Matrix3::from_row_slice(&w);
            if w.iter().any(|x| (x - x.round()).abs() > SCALE_TOLERANCE) {
                return Err(invalid("a symmetry rotation is not an integer matrix"));
            }
            let t = numbers(&translations[s], 3, "a translation")?;
            let t = Vector3::new(t[0], t[1], t[2]);
            let r = lattice * w * inverse;
            if (r * r.transpose() - Matrix3::identity()).norm() > ORTHOGONALITY {
                return Err(invalid(
                    "a symmetry rotation is not orthogonal in the cell's Cartesian frame",
                ));
            }
            let entries = maps[s]
                .as_array()
                .filter(|a| a.len() == centers.len())
                .ok_or_else(|| invalid("an atom map needs one entry per atom"))?;
            let mut map: Vec<usize> = Vec::with_capacity(centers.len());
            for v in entries {
                map.push(count(v, "an atom map entry")?);
            }
            let mut seen = vec![false; centers.len()];
            for (a, &b) in map.iter().enumerate() {
                if b >= centers.len() || std::mem::replace(&mut seen[b], true) {
                    return Err(invalid("an atom map is not a permutation of the atoms"));
                }
                let image = w * fractional[a] + t - fractional[b];
                let image = image.map(|x| x - x.round());
                if (lattice * image).norm() > ATOM_TOLERANCE {
                    return Err(invalid(format!(
                        "operation {s} does not send atom {a} onto atom {b}"
                    )));
                }
            }
            // The permutation of the grid nodes the SCF averages the density with.
            let mut scale = [[0i64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let value = w[(i, j)] * shape[i] as f64 / shape[j] as f64;
                    if (value - value.round()).abs() > SCALE_TOLERANCE {
                        return Err(invalid(
                            "a symmetry rotation does not map the grid onto itself",
                        ));
                    }
                    scale[i][j] = value.round() as i64;
                }
            }
            let mut shift = [0i64; 3];
            for i in 0..3 {
                let value = t[i] * shape[i] as f64;
                if (value - value.round()).abs() > SHIFT_TOLERANCE {
                    return Err(invalid(
                        "a symmetry translation does not map the grid onto itself",
                    ));
                }
                shift[i] = value.round() as i64;
            }
            if permutations[s].len() != nodes {
                return Err(invalid("a symmetry permutation has one entry per node"));
            }
            let mut flat = 0;
            for a in 0..size[0] {
                for b in 0..size[1] {
                    for c in 0..size[2] {
                        let node = [a, b, c];
                        let image: [i64; 3] = std::array::from_fn(|i| {
                            let sum: i64 = (0..3).map(|j| scale[i][j] * node[j]).sum();
                            (sum + shift[i]).rem_euclid(size[i])
                        });
                        let at = ((image[0] * size[1] + image[1]) * size[2] + image[2]) as usize;
                        if permutations[s][flat] != at {
                            return Err(invalid(format!(
                                "the node permutation of operation {s} is not the one its \
                                 rotation and translation generate"
                            )));
                        }
                        flat += 1;
                    }
                }
            }
            cartesian.push(r);
            atom_maps.push(map);
        }
        Ok(Self {
            rotations: cartesian,
            atom_maps,
        })
    }

    /// The number of operations.
    pub fn len(&self) -> usize {
        self.rotations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rotations.is_empty()
    }

    /// A group from Cartesian rotations and atom maps, unchecked: for the controls of the
    /// symmetrizer (a wrong atom map must be detectable).
    pub fn from_parts(rotations: Vec<Matrix3<f64>>, atom_maps: Vec<Vec<usize>>) -> Self {
        Self {
            rotations,
            atom_maps,
        }
    }

    pub fn rotations(&self) -> &[Matrix3<f64>] {
        &self.rotations
    }

    pub fn atom_maps(&self) -> &[Vec<usize>] {
        &self.atom_maps
    }

    /// `(1/n) sum_s R_s^T f_{s(A)}` for a per-atom vector (a force, or `dE/dtau`).
    pub fn vectors(&self, per_atom: &mut [[f64; 3]]) {
        let source = per_atom.to_vec();
        let weight = 1. / self.len() as f64;
        for (a, out) in per_atom.iter_mut().enumerate() {
            let mut sum = Vector3::zeros();
            for (r, map) in self.rotations.iter().zip(&self.atom_maps) {
                sum += r.transpose() * Vector3::from(source[map[a]]);
            }
            *out = (sum * weight).into();
        }
    }

    /// `(1/n) sum_s R_s g R_s^T` for a rank-2 tensor `g[a][b]` (a strain derivative).
    pub fn tensor(&self, g: &mut [[f64; 3]; 3]) {
        let source = Matrix3::from_fn(|a, b| g[a][b]);
        let mut sum = Matrix3::zeros();
        for r in &self.rotations {
            sum += r * source * r.transpose();
        }
        sum /= self.len() as f64;
        for a in 0..3 {
            for b in 0..3 {
                g[a][b] = sum[(a, b)];
            }
        }
    }
}
