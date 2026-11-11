//! DFT+U on a reduced k mesh (HU-4): the shell occupations completed over the kept operations
//! and time reversal.
//!
//! A mesh reduced by the crystal's operations (and by time reversal) holds one k-point per
//! star, so the k-sum `R = sum_k w_k C^dagger D_k C` visits one point of every star. The
//! density and the channel moments are completed by the SCF itself (`solver::sym_field`,
//! `Crystal::qsym`); the shell occupations `n = W^T R W` must be completed the same way, or the
//! operator `V = dE_U/dn` would break the symmetry the wedge relies on.
//!
//! The projectors are `p_{lm}(x) = R(|x - c|) Y_lm(x - c)` with the complex Condon-Shortley
//! harmonics, so `n_{m m'} = <p_m| rho |p_m'>` is in the basis the rotation tables of
//! `setup::symmetry::rotation_tables` act on: `Y_lm(R u) = sum_m' Y_lm'(u) T[m', m]`. For an
//! operation `g` (`x -> R x + t`) sending atom `a` to atom `g(a)`, `p^{g(a)}_m(g x) = sum_m'
//! p^a_m'(x) T[m', m]`, so `n^{g(a)} = T^dagger n^a T` for a symmetric `rho`. Completing a
//! partial sum is the group average, written with the image atom's block like `qsym`:
//!
//! `n^a <- (1/|G|) sum_g T_g n^{g(a)} T_g^dagger`.
//!
//! Time reversal (the scalar Hamiltonian is real, `rho(-k) = K rho(k) K`) acts as
//! `n_{m m'} <- (-1)^{m+m'} conj(n_{-m,-m'})`, since `K Y_lm = (-1)^m Y_l,-m`. It is applied
//! whenever the k-points are not closed under `k -> -k` (a half mesh), exactly when the density
//! is completed the same way; on a mesh that holds both `k` and `-k` the occupations already
//! have the property and nothing is touched, so those runs are bit-identical to before HU-4.
//!
//! The averaging runs inside [`Spec::occupations`](super::hubbard::Spec::occupations), the seam
//! every consumer shares: the SCF, the converged report, the energy and the operators of the
//! force and the stress, the frozen occupations and the linear response.
//!
//! The linear response shifts one shell. The shift `P_J = sum_m |p_Jm><p_Jm|` is invariant under
//! every operation that fixes the atom of `J` and is moved by one that does not, so
//! [`require_fixed_atoms`] lets the response run on a reduced mesh only when no kept operation
//! moves a +U atom.
use super::{
    hubbard::Spec, invalid, point_group::PointGroup, setup::symmetry::rotation_tables, Crystal,
};
use crate::{quantum::C64, Result};
use nalgebra::DMatrix;
use std::collections::HashMap;

type Matrix = DMatrix<C64>;
/// Rotation tables are tabulated up to the largest shell (`f`).
const MAX_L: usize = 3;
/// The rotation of an operation is orthogonal to this (Cartesian entries).
const ORTHOGONALITY: f64 = 1e-6;
/// Two k-points are the same, and their weights equal, to this (fractional coordinates).
const KPOINT_TOLERANCE: f64 = 1e-8;
/// An explicit seed must be invariant under the completion to this.
const SEED_TOLERANCE: f64 = 1e-8;

/// What the occupations of a crystal are completed with.
pub struct Averager {
    time_reversal: bool,
    operations: Vec<Operation>,
}

/// One kept operation: the shell it sends each shell to, and `T^l` for `l = 0..=MAX_L`, held
/// once per operation (a table with 4096 shells holds no more rotation matrices than one shell).
struct Operation {
    images: Vec<usize>,
    tables: Vec<Matrix>,
}

/// Whether the k-points hold, for each `k`, its time-reversal partner `-k` (modulo a reciprocal
/// vector) with the same weight: the mesh a plain sum over it already completes.
fn closed_under_time_reversal(c: &Crystal) -> bool {
    // Coordinates are taken modulo 1; a value a rounding error below 1 is the point at 0.
    let size = (1. / KPOINT_TOLERANCE).round() as i64;
    let canonical = |f: [f64; 3]| -> [i64; 3] {
        f.map(|x| (((x - x.floor()) / KPOINT_TOLERANCE).round() as i64).rem_euclid(size))
    };
    let weights: HashMap<[i64; 3], f64> = c
        .points
        .iter()
        .map(|p| (canonical(p.fractional), p.weight))
        .collect();
    c.points.iter().all(|p| {
        weights
            .get(&canonical(p.fractional.map(|x| -x)))
            .is_some_and(|w| (w - p.weight).abs() < 1e-9)
    })
}

impl Averager {
    /// The averager of `spec` for the operations `group` (`None`: no operation beyond the
    /// identity) and time reversal `time_reversal`. Raises, naming the cause, when the shell
    /// table is not closed under an operation.
    pub fn new(spec: &Spec, group: Option<&PointGroup>, time_reversal: bool) -> Result<Self> {
        let mut operations = Vec::new();
        if let Some(group) = group {
            let mut by_atom: HashMap<(usize, usize), usize> = HashMap::new();
            for (index, shell) in spec.shells.iter().enumerate() {
                by_atom.insert((shell.atom, shell.l), index);
            }
            for (rotation, map) in group.rotations().iter().zip(group.atom_maps()) {
                let tables = rotation_tables(rotation, MAX_L, ORTHOGONALITY)?;
                let mut images = Vec::with_capacity(spec.shells.len());
                for shell in &spec.shells {
                    let atom = *map.get(shell.atom).ok_or_else(|| {
                        invalid("DFT+U symmetrization: an atom map is shorter than the atom list")
                    })?;
                    let image = *by_atom.get(&(atom, shell.l)).ok_or_else(|| {
                        invalid(format!(
                            "DFT+U on a symmetry-reduced mesh needs a Hubbard shell l = {} on every atom \
                             equivalent to atom {}, and atom {atom} has none: use symmetry=False",
                            shell.l, shell.atom
                        ))
                    })?;
                    let other = &spec.shells[image];
                    if other.u != shell.u
                        || other.j != shell.j
                        || other.weights != shell.weights
                        || other.projectors.len() != shell.projectors.len()
                    {
                        return Err(invalid(format!(
                            "DFT+U on a symmetry-reduced mesh needs equivalent atoms {} and {atom} to carry \
                             the same Hubbard shell (U, J and radial weights): use symmetry=False",
                            shell.atom
                        )));
                    }
                    images.push(image);
                }
                operations.push(Operation { images, tables });
            }
        }
        Ok(Self {
            time_reversal,
            operations,
        })
    }

    /// The averager the crystal's mesh needs, `None` when its occupations are already complete.
    /// A reduced mesh needs the operations of the snapshot's `setup` block; a crystal without
    /// them raises.
    pub fn for_crystal(spec: &Spec, c: &Crystal) -> Result<Option<Self>> {
        let reduced = !c.permutations.is_empty();
        let group = if reduced {
            Some(
                c.description
                    .as_ref()
                    .and_then(|d| d.point_group.as_ref())
                    .ok_or_else(|| {
                        invalid(format!(
                            "DFT+U on a symmetry-reduced mesh needs the symmetry operations of the \
                             snapshot's setup block, which this crystal does not carry{}: use symmetry=False",
                            c.description_unavailable
                                .as_ref()
                                .map_or(String::new(), |why| format!(" ({why})"))
                        ))
                    })?,
            )
        } else {
            None
        };
        let time_reversal = !c.spinor && (reduced || !closed_under_time_reversal(c));
        if group.is_none() && !time_reversal {
            return Ok(None);
        }
        Self::new(spec, group, time_reversal).map(Some)
    }

    /// Completes `n` (`[spin * shells + shell]`, each `(2l+1)^2` row-major) in place.
    pub fn apply(&self, spec: &Spec, n: &mut [Vec<C64>]) {
        let shells = spec.shells.len();
        let blocks: Vec<Matrix> = n
            .iter()
            .enumerate()
            .map(|(index, flat)| {
                let p = 2 * spec.shells[index % shells].l + 1;
                let plain = Matrix::from_fn(p, p, |a, b| flat[a * p + b]);
                if !self.time_reversal {
                    return plain;
                }
                // `(-1)^(m+m') = (-1)^(a+b)`: the row index is `m + l` and `2l` is even.
                let reversed = Matrix::from_fn(p, p, |a, b| {
                    let sign = if (a + b) % 2 == 0 { 1. } else { -1. };
                    flat[(p - 1 - a) * p + (p - 1 - b)].conj() * sign
                });
                (plain + reversed) * C64::new(0.5, 0.)
            })
            .collect();
        for (index, out) in n.iter_mut().enumerate() {
            let (spin, shell) = (index / shells, index % shells);
            let p = 2 * spec.shells[shell].l + 1;
            let matrix = if self.operations.is_empty() {
                blocks[index].clone()
            } else {
                let mut sum = Matrix::zeros(p, p);
                for operation in &self.operations {
                    let table = &operation.tables[spec.shells[shell].l];
                    let image = operation.images[shell];
                    sum += table * &blocks[spin * shells + image] * table.adjoint();
                }
                sum * C64::new(1. / self.operations.len() as f64, 0.)
            };
            for (a, to) in out.iter_mut().enumerate() {
                *to = matrix[(a / p, a % p)];
            }
        }
    }
}

/// The first seed `n` of the SCF, which must already be invariant under the completion: a seed
/// that is not (an explicit non-symmetric matrix on a reduced mesh) raises instead of being
/// changed silently.
pub fn check_seed(spec: &Spec, c: &Crystal, seed: &[Vec<C64>]) -> Result<()> {
    let Some(averager) = Averager::for_crystal(spec, c)? else {
        return Ok(());
    };
    let mut completed = seed.to_vec();
    averager.apply(spec, &mut completed);
    let worst = seed
        .iter()
        .zip(&completed)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).norm()))
        .fold(0., f64::max);
    if worst > SEED_TOLERANCE {
        return Err(invalid(format!(
            "the explicit Hubbard seed breaks the symmetry of the reduced k mesh (it changes by {worst:.1e} \
             under the kept operations and time reversal): seed every equivalent shell alike, or use symmetry=False"
        )));
    }
    Ok(())
}

/// Raises when a kept operation moves an atom of a Hubbard shell. A localized potential shift
/// (the linear response) on such an atom breaks the symmetry the reduced mesh and the density
/// completion assume.
pub fn require_fixed_atoms(spec: &Spec, c: &Crystal) -> Result<()> {
    if c.permutations.is_empty() {
        return Ok(());
    }
    let group = c
        .description
        .as_ref()
        .and_then(|d| d.point_group.as_ref())
        .ok_or_else(|| {
            invalid("the Hubbard response on a symmetry-reduced mesh needs the symmetry operations of the setup block: use symmetry=False")
        })?;
    for shell in &spec.shells {
        if let Some(moved) = group
            .atom_maps()
            .iter()
            .find_map(|map| map.get(shell.atom).copied().filter(|&b| b != shell.atom))
        {
            return Err(invalid(format!(
                "the Hubbard response on a symmetry-reduced mesh needs every +U atom fixed by every kept \
                 operation, because the localized shift of one atom breaks the symmetry; atom {} is sent to \
                 atom {moved}: use symmetry=False",
                shell.atom
            )));
        }
    }
    Ok(())
}
