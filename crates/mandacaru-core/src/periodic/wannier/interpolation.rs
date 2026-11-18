//! Model-free Fourier/Wannier interpolation of `H(k)`, `dH/dk`, position matrices `A(k)`, and
//! Wang-Yates-Souza-Vanderbilt (PRB 74, 195118 (2006)) Berry curvature and anomalous Hall
//! conductivity (`dev_plan/WANNIER_PLAN.md` sections W3-W4).
//!
//! Operates solely on replica vectors and R-space operator blocks (`no crystal`, no setup).
use super::{invalid, Matrix};
use crate::{
    math::linalg::eigen,
    periodic::{
        berry::hall::{self, Hall, COARSE_MESH, CONVERGENCE_WARNING},
        bloch_mesh::MeshIndex,
        spectrum::{self, Smearing},
    },
    product,
    quantum::C64,
    units::Quantity,
    Buffer, Context, Result,
};
use nalgebra::{Matrix3, Vector3};
use std::{collections::BTreeMap, f64::consts::PI};

/// Energy denominator floor (Hartree) below which band pairs are treated as degenerate in `D_a`.
pub const DEGENERATE_GAP: f64 = 1e-8;
/// Curvature magnitude threshold (`|Omega_axial|`, Bohr^2) triggering adaptive sub-mesh refinement.
pub const ADAPTIVE_CURVATURE_THRESHOLD: f64 = 25.0;
/// Adaptive refinement sub-grid per triggered microcell (`3 x 3 x 3 = 27` points).
const ADAPTIVE_SUBGRID: usize = 3;

/// Shared Wigner-Seitz minimal-image replica table for a supercell and Wannier center set.
pub struct ReplicaTable {
    pub cell: Matrix3<f64>,
    pub size: [usize; 3],
    pub functions: usize,
    pub vectors: Vec<[i32; 3]>,
    /// For each `bvk_r in 0..N` and pair `(i, j)`: list of `(replica_idx, weight)` targets.
    assignments: Vec<Vec<(usize, f64)>>,
}

impl ReplicaTable {
    /// Choose all shortest Born-von Karman images for every `(r_bvk, i, j)` triple, splitting
    /// tied distances equally so the replica list `vectors` is closed under inversion `R -> -R`.
    pub fn build(cell: Matrix3<f64>, size: [usize; 3], centers: &[[f64; 3]]) -> Result<Self> {
        let functions = centers.len();
        let n = product(&size)?;
        if size.contains(&0)
            || functions == 0
            || centers
                .iter()
                .flatten()
                .chain(cell.iter())
                .any(|v| !v.is_finite())
        {
            return Err(invalid("invalid Wannier replica geometry"));
        }
        let supercell = cell * Matrix3::from_diagonal(&Vector3::from(size.map(|v| v as f64)));
        let inverse = supercell
            .try_inverse()
            .ok_or_else(|| invalid("singular Wannier hopping cell"))?;
        let mut unique: BTreeMap<[i32; 3], usize> = BTreeMap::new();
        let mut raw_assignments = Vec::with_capacity(n * functions * functions);
        for r in 0..n {
            let vector = [
                r / (size[1] * size[2]),
                (r / size[2]) % size[1],
                r % size[2],
            ];
            for i in 0..functions {
                for j in 0..functions {
                    let displacement = cell * Vector3::from(vector.map(|v| v as f64))
                        + Vector3::from(centers[j])
                        - Vector3::from(centers[i]);
                    let fractional = inverse * displacement;
                    let first = fractional.map(|v| -v.round());
                    let mut best = (displacement + supercell * first).norm_squared();
                    let radius = best.sqrt() + 1e-8;
                    let bounds: [(i32, i32); 3] = std::array::from_fn(|a| {
                        let extent = inverse.row(a).norm() * radius;
                        (
                            (-fractional[a] - extent).ceil() as i32,
                            (-fractional[a] + extent).floor() as i32,
                        )
                    });
                    let count = bounds
                        .iter()
                        .map(|(lo, hi)| (i64::from(*hi) - i64::from(*lo) + 1).max(0) as usize)
                        .collect::<Vec<_>>();
                    if product(&count)? > 1_000_000 {
                        return Err(invalid(
                            "Wannier nearest-replica search exceeds its geometric work bound",
                        ));
                    }
                    let mut chosen = vec![];
                    for x in bounds[0].0..=bounds[0].1 {
                        for y in bounds[1].0..=bounds[1].1 {
                            for z in bounds[2].0..=bounds[2].1 {
                                let distance = (displacement
                                    + supercell * Vector3::new(x as f64, y as f64, z as f64))
                                .norm_squared();
                                if distance < best - 1e-8 {
                                    best = distance;
                                    chosen.clear();
                                }
                                if (distance - best).abs() < 1e-8 {
                                    chosen.push([x, y, z]);
                                }
                            }
                        }
                    }
                    if chosen.is_empty() {
                        return Err(invalid("Wannier replica search found no image"));
                    }
                    let weight = 1. / chosen.len() as f64;
                    let mut list = Vec::with_capacity(chosen.len());
                    for image in chosen {
                        let mut vec_out = [0; 3];
                        for a in 0..3 {
                            vec_out[a] =
                                i32::try_from(vector[a] as i64 + image[a] as i64 * size[a] as i64)
                                    .map_err(|_| crate::Error::Overflow)?;
                        }
                        unique.insert(vec_out, 0);
                        list.push((vec_out, weight));
                    }
                    raw_assignments.push(list);
                }
            }
        }
        let mut vectors = Vec::with_capacity(unique.len());
        for (idx, (vec_key, slot)) in unique.iter_mut().enumerate() {
            *slot = idx;
            vectors.push(*vec_key);
        }
        let assignments = raw_assignments
            .into_iter()
            .map(|list| list.into_iter().map(|(v, w)| (unique[&v], w)).collect())
            .collect();
        Ok(Self {
            cell,
            size,
            functions,
            vectors,
            assignments,
        })
    }

    /// Scatter BvK-indexed blocks `[r * J * J + i * J + j]` into replica-indexed `Matrix` tables.
    pub fn scatter(&self, ctx: &Context, bvk_blocks: &[C64]) -> Result<Vec<Matrix>> {
        let n = product(&self.size)?;
        let j_dim = self.functions;
        if bvk_blocks.len() != product(&[n, j_dim, j_dim])? {
            return Err(invalid("Wannier replica scatter block length mismatch"));
        }
        let mut out = Vec::with_capacity(self.vectors.len());
        let mut _guards = Vec::with_capacity(self.vectors.len());
        for _ in 0..self.vectors.len() {
            _guards.push(ctx.reserve(product(&[j_dim, j_dim, 16])?)?);
            out.push(Matrix::zeros(j_dim, j_dim));
        }
        for r in 0..n {
            for i in 0..j_dim {
                for j in 0..j_dim {
                    let val = bvk_blocks[(r * j_dim + i) * j_dim + j];
                    for &(target, weight) in &self.assignments[(r * j_dim + i) * j_dim + j] {
                        out[target][(i, j)] += val * weight;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Pack a replica matrix slice back into a single flat `Buffer<C64>` (`vectors` major, row-major `J x J`).
    pub fn pack(&self, ctx: &Context, matrices: &[Matrix]) -> Result<Buffer<C64>> {
        let j_dim = self.functions;
        let mut values = ctx.allocate::<C64>(product(&[self.vectors.len(), j_dim, j_dim])?)?;
        for (r, mat) in matrices.iter().enumerate() {
            for i in 0..j_dim {
                for j in 0..j_dim {
                    values.as_mut_slice()[(r * j_dim + i) * j_dim + j] = mat[(i, j)];
                }
            }
        }
        Ok(values)
    }
}

/// Real-space tight-binding Wannier representation of `H(R)` and `A_a(R)`.
#[derive(Clone, Debug)]
pub struct Model {
    /// Lattice vectors as columns (Bohr).
    pub lattice: Matrix3<f64>,
    /// Reciprocal lattice vectors `G_j` as columns (1/Bohr), `G^T L = 2 pi I`.
    pub reciprocal: Matrix3<f64>,
    pub functions: usize,
    pub vectors: Vec<[i32; 3]>,
    pub h_r: Vec<Matrix>,
    pub a_r: [Vec<Matrix>; 3],
    /// Raw non-Hermitian asymmetry `max_{a, R} || A_a(R) - A_a(-R)^H ||_F` before Hermitization.
    pub position_asymmetry: f64,
}

fn fourier_mesh_to_bvk(size: [usize; 3], functions: usize, k_blocks: &[Matrix]) -> Vec<C64> {
    let n = k_blocks.len();
    let mut out = vec![C64::default(); n * functions * functions];
    for r in 0..n {
        let index = [
            r / (size[1] * size[2]),
            (r / size[2]) % size[1],
            r % size[2],
        ];
        for (k, mat) in k_blocks.iter().enumerate() {
            let frac = [
                (k / (size[1] * size[2])) as f64 / size[0] as f64,
                ((k / size[2]) % size[1]) as f64 / size[1] as f64,
                (k % size[2]) as f64 / size[2] as f64,
            ];
            let phase = C64::from_polar(
                1. / n as f64,
                -2. * PI * (0..3).map(|a| frac[a] * index[a] as f64).sum::<f64>(),
            );
            for i in 0..functions {
                for j in 0..functions {
                    out[(r * functions + i) * functions + j] += phase * mat[(i, j)];
                }
            }
        }
    }
    out
}

/// Hermitize `[Vec<Matrix>; 3]` (`A(R) <- 0.5 (A(R) + A(-R)^H)`) and return `(asymmetry, hermitized)`.
fn hermitize_positions(vectors: &[[i32; 3]], raw: [Vec<Matrix>; 3]) -> (f64, [Vec<Matrix>; 3]) {
    let index: BTreeMap<[i32; 3], usize> =
        vectors.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut worst = 0_f64;
    let out = std::array::from_fn(|axis| {
        let mats = &raw[axis];
        let functions = mats.first().map_or(0, Matrix::nrows);
        vectors
            .iter()
            .enumerate()
            .map(|(r, &vec_r)| {
                let neg = [-vec_r[0], -vec_r[1], -vec_r[2]];
                let partner = index
                    .get(&neg)
                    .map(|&nr| mats[nr].adjoint())
                    .unwrap_or_else(|| Matrix::zeros(functions, functions));
                worst = worst.max((&mats[r] - &partner).norm());
                (&mats[r] + &partner) * C64::new(0.5, 0.)
            })
            .collect()
    });
    (worst, out)
}

impl Model {
    /// Construct from Wannier-gauge mesh links `M_tilde(q, b) = U_q^+ M(q, b) U_{q+b}` using one
    /// shared [`ReplicaTable`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_rotated_links(
        ctx: &Context,
        replica: &ReplicaTable,
        mesh_vectors: &[[f64; 3]],
        mesh_weights: &[f64],
        rotated_links: &[Matrix],
        centers: &[[f64; 3]],
        h_r: Vec<Matrix>,
    ) -> Result<Self> {
        let points = product(&replica.size)?;
        let nb = mesh_vectors.len();
        let functions = replica.functions;
        if mesh_weights.len() != nb
            || rotated_links.len() != product(&[points, nb])?
            || centers.len() != functions
            || h_r.len() != replica.vectors.len()
        {
            return Err(invalid(
                "Wannier position-matrix inputs mismatch mesh sizes",
            ));
        }
        let mut a_k: [Vec<Matrix>; 3] = std::array::from_fn(|_| Vec::with_capacity(points));
        for k in 0..points {
            let mut mats: [Matrix; 3] =
                std::array::from_fn(|_| Matrix::zeros(functions, functions));
            for b in 0..nb {
                let m = &rotated_links[k * nb + b];
                let w = mesh_weights[b];
                let bv = mesh_vectors[b];
                for axis in 0..3 {
                    let fac = C64::new(0., w * bv[axis]);
                    for i in 0..functions {
                        for j in 0..functions {
                            let val = m[(i, j)]
                                - if i == j {
                                    C64::new(1., 0.)
                                } else {
                                    C64::default()
                                };
                            mats[axis][(i, j)] += val * fac;
                        }
                    }
                }
            }
            for axis in 0..3 {
                a_k[axis].push(mats[axis].clone());
            }
        }
        let mut raw_mats: [Vec<Matrix>; 3] = std::array::from_fn(|_| Vec::new());
        for axis in 0..3 {
            let mut bvk = fourier_mesh_to_bvk(replica.size, functions, &a_k[axis]);
            // Replace R = 0 diagonal by exact spread-branch Wannier centers (WYSV 2006, Sec. III).
            for i in 0..functions {
                bvk[i * functions + i] = C64::new(centers[i][axis], 0.);
            }
            raw_mats[axis] = replica.scatter(ctx, &bvk)?;
        }
        let (position_asymmetry, a_r) = hermitize_positions(&replica.vectors, raw_mats);
        Self::from_tight_binding(
            replica.cell,
            replica.vectors.clone(),
            h_r,
            a_r,
            position_asymmetry,
        )
    }

    /// Construct directly from replica-indexed R-space blocks `H(R)` and `A_a(R)`.
    pub fn from_tight_binding(
        lattice: Matrix3<f64>,
        vectors: Vec<[i32; 3]>,
        h_r: Vec<Matrix>,
        a_r: [Vec<Matrix>; 3],
        position_asymmetry: f64,
    ) -> Result<Self> {
        let functions = h_r.first().map_or(0, Matrix::nrows);
        if functions == 0
            || vectors.is_empty()
            || h_r.len() != vectors.len()
            || a_r.iter().any(|v| v.len() != vectors.len())
            || h_r
                .iter()
                .chain(a_r.iter().flatten())
                .any(|m| m.shape() != (functions, functions))
        {
            return Err(invalid("invalid Wannier tight-binding interpolation model"));
        }
        let reciprocal = 2.
            * PI
            * lattice
                .try_inverse()
                .ok_or_else(|| invalid("singular lattice in Wannier model"))?
                .transpose();
        Ok(Self {
            lattice,
            reciprocal,
            functions,
            vectors,
            h_r,
            a_r,
            position_asymmetry,
        })
    }

    /// Evaluate Wannier-gauge operators `H(k)`, `dH/dk_a`, `A_a(k)` and bare axial curvature
    /// `Omega_bare_c(k)` (`c = 0..3` -> `(yz, zx, xy)`) at fractional k-point `point`.
    pub fn evaluate_k(&self, point: [f64; 3]) -> PointOperators {
        let j = self.functions;
        let mut h_k = Matrix::zeros(j, j);
        let mut dh_dk: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(j, j));
        let mut a_k: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(j, j));
        let mut omega_bare: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(j, j));
        let cyclic = [(1, 2, 0), (2, 0, 1), (0, 1, 2)];

        for (r, vec_r) in self.vectors.iter().enumerate() {
            let phase = C64::from_polar(
                1.,
                2. * PI * (0..3).map(|a| point[a] * vec_r[a] as f64).sum::<f64>(),
            );
            let r_cart =
                self.lattice * Vector3::new(vec_r[0] as f64, vec_r[1] as f64, vec_r[2] as f64);
            h_k += &self.h_r[r] * phase;
            for axis in 0..3 {
                dh_dk[axis] += &self.h_r[r] * (phase * C64::new(0., r_cart[axis]));
                a_k[axis] += &self.a_r[axis][r] * phase;
            }
            for &(a, b, c) in &cyclic {
                let term = &self.a_r[b][r] * C64::new(0., r_cart[a])
                    - &self.a_r[a][r] * C64::new(0., r_cart[b]);
                omega_bare[c] += term * phase;
            }
        }
        h_k = (&h_k + h_k.adjoint()) * C64::new(0.5, 0.);
        for axis in 0..3 {
            dh_dk[axis] = (&dh_dk[axis] + dh_dk[axis].adjoint()) * C64::new(0.5, 0.);
            a_k[axis] = (&a_k[axis] + a_k[axis].adjoint()) * C64::new(0.5, 0.);
            omega_bare[axis] = (&omega_bare[axis] + omega_bare[axis].adjoint()) * C64::new(0.5, 0.);
        }
        PointOperators {
            h_k,
            dh_dk,
            a_k,
            omega_bare,
        }
    }

    /// Diagonalize `H(k)` at each fractional k-point, returning `(eigenvalues, eigenvectors)`.
    pub fn interpolate_with_vectors(
        &self,
        ctx: &Context,
        kpoints: &[[f64; 3]],
    ) -> Result<(Buffer<f64>, Vec<Matrix>)> {
        if kpoints.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid(
                "Wannier interpolation needs finite fractional k-points",
            ));
        }
        let j = self.functions;
        let _work = ctx.reserve(product(&[j, j, 16, 8])?)?;
        let mut energies = ctx.allocate::<f64>(product(&[kpoints.len(), j])?)?;
        let mut vectors = Vec::with_capacity(kpoints.len());
        for (k, point) in kpoints.iter().enumerate() {
            let mut h = Matrix::zeros(j, j);
            for (r, vec_r) in self.vectors.iter().enumerate() {
                let phase = C64::from_polar(
                    1.,
                    2. * PI * (0..3).map(|a| point[a] * vec_r[a] as f64).sum::<f64>(),
                );
                h += &self.h_r[r] * phase;
            }
            if (&h - h.adjoint()).norm() > 1e-7 * h.norm().max(1.) {
                return Err(invalid(
                    "Wannier interpolation produced a non-Hermitian Hamiltonian",
                ));
            }
            let (e, u) = eigen((&h + h.adjoint()) * C64::new(0.5, 0.))?;
            energies.as_mut_slice()[k * j..(k + 1) * j].copy_from_slice(e.as_slice());
            vectors.push(u);
        }
        Ok((energies, vectors))
    }

    /// Evaluate the WYSV band-sum Berry curvature axial vector `[Omega_yz, Omega_zx, Omega_xy]`
    /// (Bohr^2) and band eigenvalues (Hartree) at fractional k-point `point`.
    pub fn berry_curvature_at(
        &self,
        point: [f64; 3],
        fermi: f64,
        smearing: Smearing,
        width: f64,
        degeneracy: f64,
    ) -> Result<([f64; 3], Vec<f64>)> {
        let j = self.functions;
        let ops = self.evaluate_k(point);
        let (energies, u) = eigen(ops.h_k)?;
        let f: Vec<f64> = energies
            .iter()
            .map(|&e| {
                if width > 0.0 {
                    degeneracy * smearing.occupation((e - fermi) / width)
                } else if e <= fermi {
                    degeneracy
                } else {
                    0.0
                }
            })
            .collect();

        let v_bar: [Matrix; 3] = std::array::from_fn(|a| u.adjoint() * &ops.dh_dk[a] * &u);
        let a_bar: [Matrix; 3] = std::array::from_fn(|a| u.adjoint() * &ops.a_k[a] * &u);
        let o_bar: [Matrix; 3] = std::array::from_fn(|a| u.adjoint() * &ops.omega_bare[a] * &u);

        let mut d_mat: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(j, j));
        for n in 0..j {
            for m in 0..j {
                let de = energies[m] - energies[n];
                if de.abs() >= DEGENERATE_GAP {
                    for a in 0..3 {
                        d_mat[a][(n, m)] = v_bar[a][(n, m)] / de;
                    }
                }
            }
        }

        let cyclic = [(1, 2, 0), (2, 0, 1), (0, 1, 2)];
        let mut omega = [0.; 3];
        for &(a, b, c) in &cyclic {
            let mut sum = 0.;
            for n in 0..j {
                sum += f[n] * o_bar[c][(n, n)].re;
            }
            for n in 0..j {
                for m in 0..j {
                    let df = f[m] - f[n];
                    if df == 0. {
                        continue;
                    }
                    let da_ab =
                        d_mat[a][(n, m)] * a_bar[b][(m, n)] - d_mat[b][(n, m)] * a_bar[a][(m, n)];
                    let dd = C64::i() * d_mat[a][(n, m)] * d_mat[b][(m, n)];
                    sum += df * (da_ab.re + dd.re);
                }
            }
            omega[c] = sum;
        }
        Ok((omega, energies.as_slice().to_vec()))
    }
}

/// Wannier-gauge operators at one k-point (`H`, `dH/dk`, `A`, `Omega_bare`).
pub struct PointOperators {
    pub h_k: Matrix,
    pub dh_dk: [Matrix; 3],
    pub a_k: [Matrix; 3],
    /// Axial components `(Omega_{yz}, Omega_{zx}, Omega_{xy})` of `partial_a A_b - partial_b A_a`.
    pub omega_bare: [Matrix; 3],
}

/// Compute Wannier-interpolated anomalous Hall conductivity over one or two spin channels, returning
/// Berry's `Hall` result struct for unified conversion/presentation (`B2`/`W4`).
#[allow(clippy::too_many_arguments)]
pub fn anomalous_hall(
    ctx: &Context,
    channels: &[&Model],
    size: [usize; 3],
    target_electrons: f64,
    fermi_scf: f64,
    smearing: Smearing,
    width: f64,
    degeneracy: f64,
    refine: bool,
) -> Result<Hall> {
    if channels.is_empty() {
        return Err(invalid("Wannier anomalous Hall needs at least one channel"));
    }
    if size.iter().any(|&n| n < 2) || product(&size).is_err() {
        return Err(invalid(format!(
            "Wannier anomalous Hall needs at least 2 k-points per direction; got {size:?}"
        )));
    }
    let reciprocal = channels[0].reciprocal;
    let mesh = MeshIndex::new(size, [0.; 3])?;
    let n_pts = mesh.points();

    // 1. Collect interpolated eigenvalues on `size` to solve Fermi level and identify insulation.
    let mut levels: Vec<Vec<f64>> = Vec::with_capacity(channels.len() * n_pts);
    for ch in channels {
        let points: Vec<[f64; 3]> = (0..n_pts).map(|k| mesh.point(k)).collect();
        let (e_buf, _) = ch.interpolate_with_vectors(ctx, &points)?;
        let j = ch.functions;
        for k in 0..n_pts {
            levels.push(e_buf.as_slice()[k * j..(k + 1) * j].to_vec());
        }
    }
    let min_gap = levels
        .iter()
        .map(|l| {
            l.windows(2)
                .map(|w| w[1] - w[0])
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::INFINITY, f64::min);
    let _ = min_gap;

    let weights = vec![1. / n_pts as f64; levels.len()];
    let max_capacity = channels
        .iter()
        .map(|c| c.functions as f64 * degeneracy)
        .sum::<f64>();
    let mu = if (target_electrons - max_capacity).abs() < 1e-6 || !(width.is_finite() && width > 0.)
    {
        fermi_scf
    } else {
        spectrum::fermi(
            &levels,
            &weights,
            target_electrons,
            smearing,
            width,
            degeneracy,
        )?
    };

    let electrons_at = |level_mu: f64| -> f64 {
        levels
            .iter()
            .flatten()
            .map(|&e| {
                if width > 0. {
                    degeneracy * smearing.occupation((e - level_mu) / width)
                } else if e <= level_mu {
                    degeneracy
                } else {
                    0.
                }
            })
            .sum::<f64>()
            / n_pts as f64
    };
    let scf_electron_error = electrons_at(fermi_scf) - target_electrons;

    // Detect insulating subspace when every state at every k-point is effectively integer-filled or empty.
    let metallic = levels.iter().flatten().any(|&e| {
        let occ = if width > 0. {
            smearing.occupation((e - mu) / width)
        } else if e <= mu {
            1.
        } else {
            0.
        };
        occ > 1e-4 && occ < 1. - 1e-4
    });

    // 2. Integrate axial curvature across the dense uniform mesh (with optional adaptive sub-grid).
    let evaluate_mesh = |mesh_size: [usize; 3],
                         allow_refine: bool|
     -> Result<(Vec<Vector3<f64>>, f64)> {
        let m_idx = MeshIndex::new(mesh_size, [0.; 3])?;
        let m_pts = m_idx.points();
        let inv_m = 1. / m_pts as f64;
        let sub_offsets: Vec<[f64; 3]> = if allow_refine {
            let s = ADAPTIVE_SUBGRID as f64;
            let mut offs = Vec::with_capacity(ADAPTIVE_SUBGRID.pow(3));
            for i in 0..ADAPTIVE_SUBGRID {
                for j in 0..ADAPTIVE_SUBGRID {
                    for k in 0..ADAPTIVE_SUBGRID {
                        offs.push([
                            ((i as f64 + 0.5) / s - 0.5) / mesh_size[0] as f64,
                            ((j as f64 + 0.5) / s - 0.5) / mesh_size[1] as f64,
                            ((k as f64 + 0.5) / s - 0.5) / mesh_size[2] as f64,
                        ]);
                    }
                }
            }
            offs
        } else {
            vec![]
        };
        let mut channel_means = vec![Vector3::zeros(); channels.len()];
        let mut max_abs = 0_f64;
        for (s, ch) in channels.iter().enumerate() {
            let mut sum = Vector3::zeros();
            for k in 0..m_pts {
                let center = m_idx.point(k);
                let (omega, _) = ch.berry_curvature_at(center, mu, smearing, width, degeneracy)?;
                let v = Vector3::from(omega);
                let mag = v.norm();
                max_abs = max_abs.max(mag);
                if allow_refine && mag > ADAPTIVE_CURVATURE_THRESHOLD {
                    let mut sub_avg = Vector3::zeros();
                    for off in &sub_offsets {
                        let pt = [center[0] + off[0], center[1] + off[1], center[2] + off[2]];
                        let (sub_o, _) =
                            ch.berry_curvature_at(pt, mu, smearing, width, degeneracy)?;
                        sub_avg += Vector3::from(sub_o);
                    }
                    sum += sub_avg * (inv_m / sub_offsets.len() as f64);
                } else {
                    sum += v * inv_m;
                }
            }
            channel_means[s] = sum;
        }
        Ok((channel_means, max_abs))
    };

    let (channel_means, max_abs) = evaluate_mesh(size, refine)?;
    let face_vectors: [Vector3<f64>; 3] = std::array::from_fn(|c| {
        let a = (c + 1) % 3;
        let b = (c + 2) % 3;
        reciprocal.column(a).cross(&reciprocal.column(b)) / (2. * PI)
    });
    let numbers: [f64; 3] = std::array::from_fn(|c| {
        channel_means
            .iter()
            .map(|m| face_vectors[c].dot(m))
            .sum::<f64>()
    });
    let layers: [Vec<Vec<f64>>; 3] = std::array::from_fn(|c| {
        channel_means
            .iter()
            .map(|m| vec![face_vectors[c].dot(m)])
            .collect()
    });
    let (vector, sigma) = hall::conductivity_of(&reciprocal, numbers);

    let coarse = if metallic && size.iter().all(|n| n % 2 == 0 && *n >= COARSE_MESH) {
        let half = size.map(|n| n / 2);
        let (c_means, _) = evaluate_mesh(half, false)?;
        let c_nums: [f64; 3] =
            std::array::from_fn(|c| c_means.iter().map(|m| face_vectors[c].dot(m)).sum::<f64>());
        Some(hall::conductivity_of(&reciprocal, c_nums).1)
    } else {
        None
    };

    let mut warnings = Vec::new();
    let to_s_cm = Quantity::Conductivity.factor(false);
    let largest = sigma.iter().flatten().fold(0_f64, |m, v| m.max(v.abs()));
    if metallic {
        match coarse {
            None => warnings.push(format!(
                "the Wannier-interpolated Hall conductivity of a metal converges with the dense \
                 k mesh, and the mesh {size:?} gives no stride-2 convergence estimate (use even \
                 sizes >= {COARSE_MESH})"
            )),
            Some(coarse_sigma) => {
                let difference = sigma
                    .iter()
                    .flatten()
                    .zip(coarse_sigma.iter().flatten())
                    .fold(0_f64, |m, (a, b)| m.max((a - b).abs()));
                if difference > CONVERGENCE_WARNING * largest && difference > 1e-9 {
                    warnings.push(format!(
                        "the Wannier-interpolated Hall conductivity is not yet converged on {size:?}: \
                         largest component {:.4e} S/cm differs by {:.4e} S/cm on stride-2 sub-mesh",
                        largest * to_s_cm,
                        difference * to_s_cm
                    ));
                }
            }
        }
    }

    let total: [i64; 3] = if metallic {
        [0; 3]
    } else {
        numbers.map(|v| (v / degeneracy).round() as i64)
    };
    let chern: [Vec<i64>; 3] = if metallic {
        std::array::from_fn(|_| vec![])
    } else {
        std::array::from_fn(|c| {
            channel_means
                .iter()
                .map(|m| (face_vectors[c].dot(m) / degeneracy).round() as i64)
                .collect()
        })
    };

    Ok(Hall {
        size,
        metallic,
        chern,
        total,
        numbers,
        layers,
        vector,
        sigma,
        sigma_coarse: coarse,
        max_flux: max_abs,
        minimum: if metallic { f64::NAN } else { 1.0 },
        cap: 0,
        merged: 0,
        plaquettes: n_pts * 3,
        fermi: mu,
        scf_electron_error,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replica_table_maps_home_and_is_closed_under_inversion() {
        let cell = Matrix3::identity() * 4.0;
        let table = ReplicaTable::build(cell, [3, 3, 3], &[[0., 0., 0.]]).unwrap();
        let set: std::collections::BTreeSet<[i32; 3]> = table.vectors.iter().copied().collect();
        for &v in &table.vectors {
            assert!(set.contains(&[-v[0], -v[1], -v[2]]), "missing -{v:?}");
        }
        assert!(set.contains(&[0, 0, 0]));
    }
}
