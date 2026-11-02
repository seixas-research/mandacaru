//! The space-group operations the grid and the k-mesh carry. The search
//! (`space_group::find`) runs on the species; here an operation is kept when it
//! maps every atom onto an atom of the same `(species, moment)` label, maps grid
//! nodes onto grid nodes, and, for the k-reduction, maps the mesh onto itself.
//! Kept operations stay in the factored form of `Operations` (one atom map per
//! distinct rotation and per pure translation), so nothing here is an
//! `n_ops x n_atoms` or `n_ops x n_nodes` table. Atomic units.
use super::shapes::Channel;
use crate::{
    paw::multipole::{angular_grid, directions},
    periodic::{
        invalid,
        space_group::{find, Cell, Operations},
        spectrum::{self, Mesh},
    },
    Buffer, Context, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};

/// The cell and the grid the operations must respect.
pub struct Geometry<'a> {
    /// Lattice vectors as COLUMNS (Bohr), as `shapes::Sites`.
    pub lattice: Matrix3<f64>,
    pub fractional: &'a [[f64; 3]],
    /// What the space-group search distinguishes atoms by (the atomic number).
    pub species: &'a [u32],
    /// The finer label an operation must also preserve (species and initial moment).
    pub labels: &'a [u32],
    pub grid: [usize; 3],
    /// Cartesian tolerance (Bohr) of the search and of the grid-shift rule.
    pub symprec: f64,
    /// Harmonic rotations are tabulated for `L <= max_l`.
    pub max_l: usize,
}

/// How an operation moves grid nodes: `n'_i = sum_j matrix[i][j] n_j + shift[i]` (mod `N_i`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMap {
    pub matrix: [[i64; 3]; 3],
    pub shift: [i64; 3],
}

pub struct Symmetry {
    ops: Operations,
    lattice: Matrix3<f64>,
    grid: [usize; 3],
    /// Operations of the space group before any filter.
    pub n_space_group: usize,
    /// Indices into `ops` of the kept operations, ascending.
    kept: Vec<usize>,
    shifts: Vec<[i64; 3]>,
    /// Grid matrix of each distinct rotation, `None` when the grid does not carry it.
    scales: Vec<Option<[[i64; 3]; 3]>>,
    /// `T^L` of each distinct rotation that has a kept operation, `L = 0..=max_l`.
    harmonics: Vec<Option<Vec<DMatrix<Complex64>>>>,
}

impl Symmetry {
    /// Search the space group of `geometry` and filter it.
    pub fn new(geometry: &Geometry) -> Result<Self> {
        check(geometry)?;
        let cell = Cell {
            lattice: geometry.lattice.transpose(),
            fractional: geometry.fractional,
            labels: geometry.species,
        };
        Self::from_operations(find(&cell, geometry.symprec)?, geometry)
    }

    /// Filter already found operations (found for the same cell).
    pub fn from_operations(ops: Operations, geometry: &Geometry) -> Result<Self> {
        check(geometry)?;
        let n = geometry.fractional.len();
        let grid = geometry.grid.map(|v| v as i64);
        let mut rep_rotation = vec![Matrix3::identity(); ops.representative_maps.len()];
        for (s, &(r, _)) in ops.op_factors.iter().enumerate() {
            rep_rotation[r] = ops.rotations[s];
        }
        let scales: Vec<_> = rep_rotation.iter().map(|w| grid_matrix(w, grid)).collect();
        let (mut kept, mut shifts) = (vec![], vec![]);
        let mut image = vec![0; n];
        'ops: for s in 0..ops.len() {
            let Some(_) = scales[ops.op_factors[s].0] else {
                continue;
            };
            let Some(shift) = grid_shift(&ops.translations[s], geometry) else {
                continue;
            };
            ops.atom_map(s, &mut image);
            for (a, &b) in image.iter().enumerate() {
                if geometry.labels[b] != geometry.labels[a] {
                    continue 'ops;
                }
            }
            kept.push(s);
            shifts.push(shift);
        }
        let mut harmonics: Vec<Option<Vec<DMatrix<Complex64>>>> = vec![None; scales.len()];
        let shortest = (0..3)
            .map(|c| geometry.lattice.column(c).norm())
            .fold(f64::INFINITY, f64::min);
        for &s in &kept {
            let r = ops.op_factors[s].0;
            if harmonics[r].is_none() {
                let cartesian = geometry.lattice
                    * rep_rotation[r].map(f64::from)
                    * geometry
                        .lattice
                        .try_inverse()
                        .ok_or_else(|| invalid("the lattice must be nonsingular"))?;
                let bound = 10. * geometry.symprec / shortest + 1e-12;
                harmonics[r] = Some(rotation_tables(&cartesian, geometry.max_l, bound)?);
            }
        }
        Ok(Self {
            n_space_group: ops.len(),
            lattice: geometry.lattice,
            grid: geometry.grid,
            ops,
            kept,
            shifts,
            scales,
            harmonics,
        })
    }

    /// Number of kept operations.
    pub fn len(&self) -> usize {
        self.kept.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }
    /// Nothing beyond the identity survives: no field is symmetrized.
    pub fn is_trivial(&self) -> bool {
        self.kept.len() <= 1
    }
    pub fn rotation(&self, k: usize) -> &Matrix3<i32> {
        &self.ops.rotations[self.kept[k]]
    }
    pub fn translation(&self, k: usize) -> [f64; 3] {
        self.ops.translations[self.kept[k]]
    }
    /// The atom at which operation `k` sends each atom.
    pub fn atom_map(&self, k: usize, out: &mut [usize]) {
        self.ops.atom_map(self.kept[k], out)
    }
    /// `T^L` of operation `k`: `Y_LM(R u) = sum_M' Y_LM'(u) T[(M' + L, M + L)]`.
    pub fn harmonic(&self, k: usize, l: usize) -> &DMatrix<Complex64> {
        &self.harmonics[self.ops.op_factors[self.kept[k]].0]
            .as_ref()
            .expect("a kept operation has its rotation table")[l]
    }
    pub fn node_map(&self, k: usize) -> NodeMap {
        let scale = self.scales[self.ops.op_factors[self.kept[k]].0].expect("kept");
        NodeMap {
            matrix: scale,
            shift: self.shifts[k],
        }
    }
    /// Flat (C order) node that operation `k` sends each node to.
    pub fn permutation(&self, k: usize, out: &mut [usize]) {
        let map = self.node_map(k);
        let [n0, n1, n2] = self.grid;
        assert_eq!(out.len(), n0 * n1 * n2);
        let size = self.grid.map(|v| v as i64);
        let mut flat = 0;
        for a in 0..n0 as i64 {
            for b in 0..n1 as i64 {
                for c in 0..n2 as i64 {
                    let n = [a, b, c];
                    let image: [i64; 3] = std::array::from_fn(|i| {
                        let sum: i64 = (0..3).map(|j| map.matrix[i][j] * n[j]).sum();
                        (sum + map.shift[i]).rem_euclid(size[i])
                    });
                    out[flat] = ((image[0] * size[1] + image[1]) * size[2] + image[2]) as usize;
                    flat += 1;
                }
            }
        }
    }
    /// The distinct rotations of the kept operations, row-major `W`, in
    /// ascending order of the first operation that has them.
    pub fn distinct_rotations(&self) -> Vec<[i32; 9]> {
        let mut seen = HashSet::new();
        let mut out = vec![];
        for &s in &self.kept {
            if seen.insert(self.ops.op_factors[s].0) {
                let w = &self.ops.rotations[s];
                out.push(std::array::from_fn(|i| w[(i / 3, i % 3)]));
            }
        }
        out
    }

    /// Drop the operations whose rotation does not map the Monkhorst-Pack mesh
    /// onto itself (the mesh of `spectrum::mesh`, `k' = k W`).
    pub fn keeping_mesh(mut self, size: [usize; 3], gamma: bool) -> Result<Self> {
        let points = MeshIntegers::new(size, gamma)?;
        let mut verdict: HashMap<usize, bool> = HashMap::new();
        let mut index = 0;
        let (mut kept, mut shifts) = (vec![], vec![]);
        for &s in &self.kept {
            let r = self.ops.op_factors[s].0;
            let keep = *verdict
                .entry(r)
                .or_insert_with(|| points.preserved_by(&self.ops.rotations[s]));
            if keep {
                kept.push(s);
                shifts.push(self.shifts[index]);
            }
            index += 1;
        }
        self.kept = kept;
        self.shifts = shifts;
        Ok(self)
    }

    /// The irreducible mesh under the kept operations and time reversal: only the
    /// distinct rotations reach the reduction, a pure translation moves no k.
    /// Call [`Self::keeping_mesh`] first; a rotation that breaks the mesh raises.
    pub fn mesh(&self, ctx: &Context, size: [usize; 3], gamma: bool) -> Result<Mesh> {
        spectrum::mesh(ctx, size, gamma, &self.distinct_rotations(), true)
    }

    /// `1/n_ops sum_s T^L(s)` over the channels: the average that sends the
    /// multipoles `q[(atom, L, M)]` to their symmetrized values, row `a` of the
    /// returned `(n_ch, n_ch)` matrix. `channels` is sorted; the identity when trivial.
    pub fn qsym(&self, ctx: &Context, channels: &[Channel]) -> Result<Buffer<Complex64>> {
        let count = channels.len();
        if channels.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("channels must be sorted and distinct"));
        }
        let mut out =
            ctx.allocate::<Complex64>(count.checked_mul(count).ok_or(crate::Error::Overflow)?)?;
        let rows = out.as_mut_slice();
        if self.is_trivial() {
            for a in 0..count {
                rows[a * count + a] = Complex64::new(1., 0.);
            }
            return Ok(out);
        }
        let mut slots: HashMap<(usize, usize), Vec<Option<usize>>> = HashMap::new();
        for (i, &(atom, l, m)) in channels.iter().enumerate() {
            if m.unsigned_abs() as usize > l || l >= self.harmonic_degrees() {
                return Err(invalid(
                    "a channel has |M| > L or L above the rotation tables",
                ));
            }
            slots
                .entry((atom, l))
                .or_insert_with(|| vec![None; 2 * l + 1])[(m + l as i32) as usize] = Some(i);
        }
        let weight = 1. / self.len() as f64;
        let mut image = vec![0; self.ops.representative_maps.first().map_or(0, Vec::len)];
        for k in 0..self.len() {
            self.atom_map(k, &mut image);
            for (a, &(atom, l, m)) in channels.iter().enumerate() {
                let Some(source) = image.get(atom).and_then(|&b| slots.get(&(b, l))) else {
                    continue;
                };
                let table = self.harmonic(k, l);
                for (other, slot) in source.iter().enumerate() {
                    if let Some(j) = slot {
                        rows[a * count + j] += table[((m + l as i32) as usize, other)] * weight;
                    }
                }
            }
        }
        Ok(out)
    }

    fn harmonic_degrees(&self) -> usize {
        self.harmonics
            .iter()
            .flatten()
            .next()
            .map_or(usize::MAX, Vec::len)
    }

    /// The lattice vectors (columns, Bohr) the operations were found for.
    pub fn lattice(&self) -> &Matrix3<f64> {
        &self.lattice
    }
}

fn check(geometry: &Geometry) -> Result<()> {
    let n = geometry.fractional.len();
    if n == 0 || geometry.species.len() != n || geometry.labels.len() != n {
        return Err(invalid(
            "atoms, species and labels must have the same length",
        ));
    }
    if geometry.grid.contains(&0) || !(geometry.symprec.is_finite() && geometry.symprec > 0.) {
        return Err(invalid(
            "the grid and the symmetry tolerance must be positive",
        ));
    }
    Ok(())
}

/// `W_ij N_i / N_j` as integers when all are integral (the grid carries the rotation).
fn grid_matrix(w: &Matrix3<i32>, grid: [i64; 3]) -> Option<[[i64; 3]; 3]> {
    let mut out = [[0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let numerator = i64::from(w[(i, j)]) * grid[i];
            if numerator % grid[j] != 0 {
                return None;
            }
            *value = numerator / grid[j];
        }
    }
    Some(out)
}

/// `round(N t)` when the translation is a grid shift to within `symprec`: the
/// Cartesian length of the fractional miss `(N t - round(N t)) / N`.
fn grid_shift(t: &[f64; 3], geometry: &Geometry) -> Option<[i64; 3]> {
    let mut shift = [0; 3];
    let mut miss = Vector3::zeros();
    for i in 0..3 {
        let node = t[i] * geometry.grid[i] as f64;
        shift[i] = node.round() as i64;
        miss[i] = (node - node.round()) / geometry.grid[i] as f64;
    }
    ((geometry.lattice * miss).norm() <= geometry.symprec).then_some(shift)
}

/// Integers `q` with `k = q / (2 N)` of the mesh points of `spectrum::mesh`.
struct MeshIntegers {
    size: [i64; 3],
    points: HashSet<[i64; 3]>,
}
impl MeshIntegers {
    fn new(size: [usize; 3], gamma: bool) -> Result<Self> {
        if size.contains(&0) {
            return Err(invalid("k-mesh dimensions must be positive"));
        }
        let n = size.map(|v| v as i64);
        let q = |a: usize, i: i64| {
            (2 * i + 1 - n[a] + i64::from(gamma && n[a] % 2 == 0)).rem_euclid(2 * n[a])
        };
        let mut points = HashSet::new();
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    points.insert([q(0, i), q(1, j), q(2, k)]);
                }
            }
        }
        Ok(Self { size: n, points })
    }
    /// Whether `k -> k W` sends every mesh point to a mesh point (modulo a reciprocal vector).
    fn preserved_by(&self, w: &Matrix3<i32>) -> bool {
        let n = self.size;
        let common = n.iter().fold(1i128, |l, &v| lcm(l, v as i128));
        self.points.iter().all(|q| {
            let mut image = [0i64; 3];
            for a in 0..3 {
                let total: i128 = (0..3)
                    .map(|b| {
                        i128::from(q[b])
                            * i128::from(w[(b, a)])
                            * i128::from(n[a])
                            * (common / n[b] as i128)
                    })
                    .sum();
                if total % common != 0 {
                    return false;
                }
                image[a] = ((total / common) as i64).rem_euclid(2 * n[a]);
            }
            self.points.contains(&image)
        })
    }
}
fn lcm(a: i128, b: i128) -> i128 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// `T^L` for `L = 0..=max_l` of the Cartesian operation `r` (orthogonal to within
/// `bound`; the nearest orthogonal matrix is used), by the exact product
/// quadrature `T_{M'M} = int conj(Y_LM'(u)) Y_LM(R u) dOmega`.
fn rotation_tables(r: &Matrix3<f64>, max_l: usize, bound: f64) -> Result<Vec<DMatrix<Complex64>>> {
    let svd = r.svd(true, true);
    let (u, vt) = (
        svd.u.ok_or_else(|| invalid("SVD failed"))?,
        svd.v_t.ok_or_else(|| invalid("SVD failed"))?,
    );
    let q = u * vt;
    if (r - q).amax() > bound {
        return Err(invalid(
            "the lattice does not carry the operation as a rotation",
        ));
    }
    let mut tables = Vec::with_capacity(max_l + 1);
    for l in 0..=max_l {
        let (theta, phi, weight) = angular_grid(l + 2, 2 * l + 2)?;
        let points = directions(&theta, &phi);
        let width = 2 * l + 1;
        let order = |m: usize| m as i32 - l as i32;
        let harmonics = |direction: [f64; 3]| -> Vec<Complex64> {
            (0..width)
                .map(|m| crate::math::angular::complex(l, order(m), direction))
                .collect()
        };
        let mut table = DMatrix::<Complex64>::zeros(width, width);
        for (p, w) in points.iter().zip(&weight) {
            let moved = q * Vector3::from(*p);
            let (y, yr) = (harmonics(*p), harmonics([moved[0], moved[1], moved[2]]));
            for (mp, y) in y.iter().enumerate() {
                for (m, yr) in yr.iter().enumerate() {
                    table[(mp, m)] += y.conj() * yr * *w;
                }
            }
        }
        let defect = (table.adjoint() * &table - DMatrix::<Complex64>::identity(width, width))
            .iter()
            .map(|v| v.norm())
            .fold(0., f64::max);
        if defect > 1e-10 {
            return Err(invalid(format!(
                "the L = {l} rotation is not unitary ({defect:.1e})"
            )));
        }
        tables.push(table);
    }
    Ok(tables)
}
