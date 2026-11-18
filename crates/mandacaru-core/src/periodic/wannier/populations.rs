//! Generalized Pipek-Mezey orbital localization for crystals (Jonsson, Lehtola, Puska, Jonsson,
//! JCTC 13, 460 (2017)): maximizes `sum_n sum_{A, R} (q^{A,R}_{nn})^p` (`p = 2` or `4`) over
//! Born-von Karman supercell atomic populations `q^{A,R}`, in either Mulliken or Lowdin scheme
//! using the full PAW AO overlap matrix `S(k)` (`KPoint::overlap`, `KPoint::lowdin`).
use super::{invalid, Matrix};
use crate::{
    periodic::{bloch_mesh::MeshIndex, setup::PeriodicSetup},
    product,
    quantum::C64,
    Context, Result,
};
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Atomic population partitioning scheme for periodic Pipek-Mezey localization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopulationScheme {
    Mulliken,
    Lowdin,
}

impl PopulationScheme {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "mulliken" => Ok(Self::Mulliken),
            "lowdin" => Ok(Self::Lowdin),
            _ => Err(invalid(format!(
                "Pipek-Mezey population scheme must be 'mulliken' or 'lowdin'; got {name:?}"
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Mulliken => "mulliken",
            Self::Lowdin => "lowdin",
        }
    }
}

/// Disjoint partition of AO basis rows `0..m` across the crystal's atoms `0..atoms`.
#[derive(Clone, Debug)]
pub struct AtomPartition {
    pub ao_count: usize,
    pub atom_rows: Vec<Vec<usize>>,
}

impl AtomPartition {
    /// Build directly from `PeriodicSetup::shells` and `PeriodicSetup::centers`.
    pub fn from_setup(setup: &PeriodicSetup) -> Result<Self> {
        let ao_count = setup.shells.functions();
        let n_atoms = setup.centers.len();
        if n_atoms == 0 || ao_count == 0 {
            return Err(invalid(
                "Pipek-Mezey localization requires at least one atom and one atomic orbital",
            ));
        }
        let mut atom_rows = vec![Vec::new(); n_atoms];
        for shell in setup.shells.shells() {
            let center = Vector3::from(shell.center);
            let mut best = None;
            for (atom, atom_center) in setup.centers.iter().enumerate() {
                let d2 = (center - Vector3::from(*atom_center)).norm_squared();
                if best.is_none_or(|(_, bd2)| d2 < bd2) {
                    best = Some((atom, d2));
                }
            }
            let (atom, dist2) = best.ok_or_else(|| invalid("setup has no atom centers"))?;
            if dist2 > 1e-10 {
                return Err(invalid(
                    "AO shell center does not coincide with any setup atom center",
                ));
            }
            atom_rows[atom].extend_from_slice(&shell.rows);
        }
        Ok(Self {
            ao_count,
            atom_rows,
        })
    }

    /// Direct partition from pre-grouped atom AO index lists (model-free test helper).
    pub fn from_atom_rows(ao_count: usize, atom_rows: Vec<Vec<usize>>) -> Result<Self> {
        if ao_count == 0 || atom_rows.is_empty() {
            return Err(invalid("invalid atomic orbital partition"));
        }
        let mut seen = vec![false; ao_count];
        for rows in &atom_rows {
            for &r in rows {
                if r >= ao_count || std::mem::replace(&mut seen[r], true) {
                    return Err(invalid(
                        "atomic orbital partition must assign every AO to exactly one atom",
                    ));
                }
            }
        }
        if seen.iter().any(|&v| !v) {
            return Err(invalid("atomic orbital partition left unassigned AOs"));
        }
        Ok(Self {
            ao_count,
            atom_rows,
        })
    }
}

/// Apply `(I_{components} kron op)` to a `(components * m) x bands` coefficient matrix.
fn apply_block_op(op: &Matrix, c: &Matrix, m: usize, components: usize) -> Matrix {
    let bands = c.ncols();
    let mut out = Matrix::zeros(components * m, bands);
    for s in 0..components {
        let block = op * c.rows(s * m, m);
        out.rows_mut(s * m, m).copy_from(&block);
    }
    out
}

/// Prepared k-mesh population operators across the Born-von Karman supercell.
pub struct PopulationMesh {
    pub size: [usize; 3],
    pub points: usize,
    pub bands: usize,
    pub ao_count: usize,
    pub components: usize,
    pub partition: AtomPartition,
    pub scheme: PopulationScheme,
    pub exponent: usize,
    /// `left[k]` and `right[k]` each have shape `(components * ao_count, bands)`.
    /// Mulliken: `left = C_k`, `right = S_k C_k`.
    /// Lowdin:   `left = right = S_k^{1/2} C_k` (`S_k * S_k^{-1/2}`).
    left: Vec<Matrix>,
    right: Vec<Matrix>,
    symmetric: bool,
}

impl PopulationMesh {
    /// Build from frame AO coefficients `coefficients[k]` (`(components * m) x bands`) and per-point
    /// `(overlap, lowdin)` (`m x m` Hermitian `S_k`, `S_k^{-1/2}`).
    pub fn new(
        size: [usize; 3],
        components: usize,
        coefficients: &[&Matrix],
        metrics: &[(&Matrix, &Matrix)],
        partition: AtomPartition,
        scheme: PopulationScheme,
        exponent: usize,
    ) -> Result<Self> {
        if ![2, 4].contains(&exponent) {
            return Err(invalid(format!(
                "Pipek-Mezey exponent must be 2 or 4; got {exponent}"
            )));
        }
        let mesh = MeshIndex::new(size, [0.; 3])?;
        let points = mesh.points();
        if coefficients.len() != points || metrics.len() != points || components == 0 {
            return Err(invalid("Pipek-Mezey mesh metrics do not match k-mesh size"));
        }
        let m = partition.ao_count;
        let bands = coefficients[0].ncols();
        if bands == 0 {
            return Err(invalid("Pipek-Mezey localization needs at least one band"));
        }
        let mut left = Vec::with_capacity(points);
        let mut right = Vec::with_capacity(points);
        for (c_k, &(s_k, s_inv_sqrt_k)) in coefficients.iter().zip(metrics) {
            if c_k.shape() != (components * m, bands)
                || s_k.shape() != (m, m)
                || s_inv_sqrt_k.shape() != (m, m)
            {
                return Err(invalid(
                    "Pipek-Mezey AO coefficients or overlap matrix dimensions mismatch",
                ));
            }
            match scheme {
                PopulationScheme::Mulliken => {
                    left.push((*c_k).clone());
                    right.push(apply_block_op(s_k, c_k, m, components));
                }
                PopulationScheme::Lowdin => {
                    // S^{1/2} = S * S^{-1/2}.
                    let s_half =
                        (s_k * s_inv_sqrt_k + (s_k * s_inv_sqrt_k).adjoint()) * C64::new(0.5, 0.);
                    let w = apply_block_op(&s_half, c_k, m, components);
                    left.push(w.clone());
                    right.push(w);
                }
            }
        }
        Ok(Self {
            size,
            points,
            bands,
            ao_count: m,
            components,
            partition,
            scheme,
            exponent,
            left,
            right,
            symmetric: scheme == PopulationScheme::Lowdin,
        })
    }

    /// Evaluate the minimization objective `-P(U) = -sum_{n, A, R} (q^{A,R}_{nn})^p` and its
    /// exact anti-Hermitian Riemannian gradient `G_k` at every mesh point (`U_k -> U_k exp(W_k)`).
    pub fn evaluate(&self, ctx: &Context, gauge: &[Matrix]) -> Result<(f64, Vec<Matrix>)> {
        let functions = gauge.first().map_or(0, Matrix::ncols);
        if gauge.len() != self.points
            || functions == 0
            || functions > self.bands
            || gauge.iter().any(|u| u.shape() != (self.bands, functions))
        {
            return Err(invalid("invalid Wannier gauge shape for Pipek-Mezey"));
        }
        let rows = self.components * self.ao_count;
        let _work = ctx.reserve(product(&[self.points, rows, functions, 16, 4])?)?;
        let n_f = self.points as f64;
        let inv_n = 1. / n_f;
        let mesh = MeshIndex::new(self.size, [0.; 3])?;

        // 1. Transform Bloch states into the Wannier gauge at each mesh point.
        let wk: Vec<Matrix> = self.left.iter().zip(gauge).map(|(l, u)| l * u).collect();
        let zk: Vec<Matrix> = if self.symmetric {
            Vec::new()
        } else {
            self.right.iter().zip(gauge).map(|(r, u)| r * u).collect()
        };

        let frac_k: Vec<[f64; 3]> = (0..self.points).map(|k| mesh.point(k)).collect();
        let idx_r: Vec<[usize; 3]> = (0..self.points).map(|r| mesh.indices(r)).collect();

        // 2. Supercell Fourier transform to W_R, Z_R and accumulate -P(U) and real-space forces.
        let p_exp = self.exponent as i32;
        let p_f = self.exponent as f64;
        let mut total_pop = 0.;
        let mut gw_r = vec![Matrix::zeros(rows, functions); self.points];
        let mut gz_r = if self.symmetric {
            Vec::new()
        } else {
            vec![Matrix::zeros(rows, functions); self.points]
        };

        for r in 0..self.points {
            let rv = idx_r[r];
            let mut wr = Matrix::zeros(rows, functions);
            let mut zr = Matrix::zeros(rows, functions);
            for k in 0..self.points {
                let phase = C64::from_polar(
                    inv_n,
                    -2. * PI
                        * (frac_k[k][0] * rv[0] as f64
                            + frac_k[k][1] * rv[1] as f64
                            + frac_k[k][2] * rv[2] as f64),
                );
                wr += &wk[k] * phase;
                if !self.symmetric {
                    zr += &zk[k] * phase;
                }
            }
            let z_ref = if self.symmetric { &wr } else { &zr };

            for atom_aos in &self.partition.atom_rows {
                if atom_aos.is_empty() {
                    continue;
                }
                for n in 0..functions {
                    let mut q_nn = 0.;
                    for s in 0..self.components {
                        let base = s * self.ao_count;
                        for &mu in atom_aos {
                            let idx = base + mu;
                            q_nn += (wr[(idx, n)].conj() * z_ref[(idx, n)]).re;
                        }
                    }
                    total_pop += q_nn.powi(p_exp);
                    let d_n = -p_f * q_nn.powi(p_exp - 1);
                    let half_d = C64::new(0.5 * d_n, 0.);
                    if self.symmetric {
                        let full_d = C64::new(d_n, 0.);
                        for s in 0..self.components {
                            let base = s * self.ao_count;
                            for &mu in atom_aos {
                                let idx = base + mu;
                                gw_r[r][(idx, n)] += wr[(idx, n)] * full_d;
                            }
                        }
                    } else {
                        for s in 0..self.components {
                            let base = s * self.ao_count;
                            for &mu in atom_aos {
                                let idx = base + mu;
                                gw_r[r][(idx, n)] += zr[(idx, n)] * half_d;
                                gz_r[r][(idx, n)] += wr[(idx, n)] * half_d;
                            }
                        }
                    }
                }
            }
        }

        // 3. Inverse Fourier transform forces back to k-space and project onto the unitary tangent.
        let mut grad = Vec::with_capacity(self.points);
        for k in 0..self.points {
            let kf = frac_k[k];
            let mut gwk = Matrix::zeros(rows, functions);
            let mut gzk = Matrix::zeros(rows, functions);
            for r in 0..self.points {
                let rv = idx_r[r];
                let phase = C64::from_polar(
                    1.,
                    2. * PI * (kf[0] * rv[0] as f64 + kf[1] * rv[1] as f64 + kf[2] * rv[2] as f64),
                );
                gwk += &gw_r[r] * phase;
                if !self.symmetric {
                    gzk += &gz_r[r] * phase;
                }
            }
            let raw_f = if self.symmetric {
                // symmetric Lowdin already doubled gw_r by using `full_d` instead of `2 * half_d`:
                gauge[k].adjoint() * (self.left[k].adjoint() * &gwk) * C64::new(inv_n, 0.)
            } else {
                gauge[k].adjoint()
                    * (self.left[k].adjoint() * &gwk + self.right[k].adjoint() * &gzk)
                    * C64::new(inv_n, 0.)
            };
            grad.push(&raw_f - raw_f.adjoint());
        }

        Ok((-total_pop, grad))
    }
}
