//! Spinor & Bloch band irrep classification, connectivity tracking, irrep-projected eigensolver,
//! double-group optical transition selection rules (Waves DG-2 & DG-3), and live PAW crystal
//! double-group adapters.
use super::setup::{symmetry::rotation_tables, PeriodicSetup};
use super::space_group::{
    double_group::{CorepKind, DoubleGroup},
    find, Cell,
};
use super::wannier::sampling::lattice as crystal_lattice;
use super::{invalid, solver, Crystal, KPoint, PeriodicResult};
use crate::math::linalg::eigen;
use crate::math::storage::MatrixSource;
use crate::quantum::C64;
use crate::Result;
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;

/// One degenerate band cluster at `k` classified under [`DoubleGroup`].
#[derive(Clone, Debug)]
pub struct BandCluster {
    pub lo: usize,
    pub hi: usize,
    pub energy: f64,
    pub irreps: Vec<(usize, f64)>,
    pub dominant_irrep: usize,
    pub corep: CorepKind,
}

/// Band irrep assignment and unitarity residuals for one `k`-point.
#[derive(Clone, Debug)]
pub struct BandIrreps {
    pub clusters: Vec<BandCluster>,
    pub max_unitarity_error: f64,
}

/// Output of [`irrep_blocked_diagonalize`]: sorted eigenvalues, AO eigenvectors, Loewdin
/// eigenvectors, and per-band irrep index.
#[derive(Clone, Debug)]
pub struct IrrepSolved {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: DMatrix<C64>,
    pub lowdin_eigenvectors: DMatrix<C64>,
    pub band_irreps: Vec<usize>,
}

/// One invariant double-group irrep subspace for a single `k`-point (`q_ao = S^{-1/2} q_low`).
#[derive(Clone, Debug)]
pub struct IrrepBlock {
    pub irrep: usize,
    pub q_low: DMatrix<C64>,
    pub q_ao: DMatrix<C64>,
}

/// Precomputed Loewdin-orthonormal symmetry projectors reused across SCF iterations to
/// block-diagonalize `H(k)` under `DoubleGroup`.
#[derive(Clone, Debug)]
pub struct IrrepProjectors {
    pub double_group: DoubleGroup,
    pub blocks: Vec<IrrepBlock>,
    pub basis_dim: usize,
}

impl IrrepProjectors {
    /// Construct complete orthonormal irrep blocks from `dg_k`, `overlap`, and `ao_actions`.
    pub fn from_actions(
        dg_k: &DoubleGroup,
        overlap: &DMatrix<C64>,
        ao_actions: &[DMatrix<C64>],
    ) -> Result<Self> {
        let n = overlap.nrows();
        if overlap.ncols() != n
            || ao_actions.len() != dg_k.unitary.len()
            || ao_actions.iter().any(|m| m.shape() != (n, n))
        {
            return Err(invalid(
                "irrep-projector construction dimensions do not match",
            ));
        }
        let s_herm = (overlap + overlap.adjoint()) * C64::new(0.5, 0.);
        let (s_vals, s_vecs) = eigen(s_herm)?;
        if s_vals.iter().any(|&v| v <= 1e-10) {
            return Err(invalid("overlap matrix is singular in irrep projectors"));
        }
        let inv_sqrt = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            n,
            s_vals.iter().map(|&v| C64::new(1. / v.sqrt(), 0.)),
        ));
        let lowdin = &s_vecs * inv_sqrt * s_vecs.adjoint();
        // Unitary Lowdin representation `U_low[g] = S^{-1/2} (S D_ao[g]) S^{-1/2}`.
        let u_low: Vec<DMatrix<C64>> = ao_actions
            .iter()
            .map(|d| &lowdin * overlap * d * &lowdin)
            .collect();
        let inv_order = 1. / dg_k.unitary.len() as f64;
        let mut blocks = Vec::new();
        let mut total_cols = 0usize;
        for irrep in &dg_k.irreps {
            let mut p_alpha = DMatrix::<C64>::zeros(n, n);
            let scale = irrep.dimension as f64 * inv_order;
            for (u, &chi) in u_low.iter().zip(&irrep.characters) {
                p_alpha += u * (chi.conj() * scale);
            }
            let p_alpha = (&p_alpha + p_alpha.adjoint()) * C64::new(0.5, 0.);
            let (p_vals, p_vecs) = eigen(p_alpha)?;
            if p_vals
                .iter()
                .any(|&v| v.abs() > 1e-4 && (v - 1.0).abs() > 1e-4)
            {
                return Err(invalid(
                    "irrep projector eigenvalues deviated from {0, 1}; D_ao not unitary under S",
                ));
            }
            let cols: Vec<usize> = (0..n).filter(|&i| p_vals[i] > 0.5).collect();
            if cols.is_empty() {
                continue;
            }
            let q_low = DMatrix::from_fn(n, cols.len(), |r, c| p_vecs[(r, cols[c])]);
            let q_ao = &lowdin * &q_low;
            total_cols += cols.len();
            blocks.push(IrrepBlock {
                irrep: irrep.index,
                q_low,
                q_ao,
            });
        }
        if total_cols != n {
            return Err(invalid(
                "irrep subspaces do not span the full basis dimension; check D_ao compatibility with S",
            ));
        }
        Ok(Self {
            double_group: dg_k.clone(),
            blocks,
            basis_dim: n,
        })
    }

    /// Precompute `IrrepProjectors` for a PAW `Crystal` `KPoint` when crystal symmetry operations
    /// are present and reduce the generalized Hermitian solve (`|G_k| > 2`, overlap residual `< 1e-6`).
    pub fn for_crystal_point(c: &Crystal, point: &KPoint) -> Result<Option<Self>> {
        let has_ops = c
            .description
            .as_ref()
            .and_then(|d| d.operations.as_ref())
            .is_some();
        if !c.spinor || !has_ops {
            return Ok(None);
        }
        let dg_k = crystal_double_group(c, point.fractional)?;
        if dg_k.unitary.len() <= 2 {
            return Ok(None);
        }
        let ao = paw_ao_actions(&dg_k, c)?;
        let overlap = full_overlap(c, point);
        if paw_overlap_residual(&overlap, &ao) > 1e-6 {
            return Ok(None);
        }
        match Self::from_actions(&dg_k, &overlap, &ao) {
            Ok(proj) => Ok(Some(proj)),
            Err(_) => Ok(None),
        }
    }

    /// Block-diagonalize `hamiltonian` in the precomputed invariant irrep subspaces and reconstruct
    /// energy-sorted AO and Loewdin eigenvectors.
    pub fn solve_hamiltonian(&self, hamiltonian: &DMatrix<C64>) -> Result<IrrepSolved> {
        let n = self.basis_dim;
        if hamiltonian.shape() != (n, n) {
            return Err(invalid(
                "Hamiltonian dimension mismatch in IrrepProjectors::solve_hamiltonian",
            ));
        }
        let mut collected: Vec<(f64, Vec<C64>, Vec<C64>, usize)> = Vec::with_capacity(n);
        for block in &self.blocks {
            let h_sub = block.q_ao.adjoint() * hamiltonian * &block.q_ao;
            let h_sub = (&h_sub + h_sub.adjoint()) * C64::new(0.5, 0.);
            let (vals, vecs) = eigen(h_sub)?;
            let ao_vecs = &block.q_ao * &vecs;
            let low_vecs = &block.q_low * &vecs;
            for b in 0..vals.len() {
                let ao_col: Vec<C64> = (0..n).map(|r| ao_vecs[(r, b)]).collect();
                let low_col: Vec<C64> = (0..n).map(|r| low_vecs[(r, b)]).collect();
                collected.push((vals[b], ao_col, low_col, block.irrep));
            }
        }
        collected.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut eigenvalues = Vec::with_capacity(n);
        let mut eigenvectors = DMatrix::<C64>::zeros(n, n);
        let mut lowdin_eigenvectors = DMatrix::<C64>::zeros(n, n);
        let mut band_irreps = Vec::with_capacity(n);
        for (col_idx, (e, ao_col, low_col, irr)) in collected.into_iter().enumerate() {
            eigenvalues.push(e);
            band_irreps.push(irr);
            for r in 0..n {
                eigenvectors[(r, col_idx)] = ao_col[r];
                lowdin_eigenvectors[(r, col_idx)] = low_col[r];
            }
        }
        Ok(IrrepSolved {
            eigenvalues,
            eigenvectors,
            lowdin_eigenvectors,
            band_irreps,
        })
    }
}

/// Build unitary representation matrices `U_band[g] = C^H * S * D_ao[g] * C` on a degenerate cluster
/// and classify it against `dg_k.irreps` (`ao_actions[g]` corresponds to `dg_k.unitary[g]`).
pub fn classify_bloch_bands(
    dg_k: &DoubleGroup,
    energies: &[f64],
    overlap: &DMatrix<C64>,
    coeffs: &DMatrix<C64>,
    ao_actions: &[DMatrix<C64>],
    degeneracy_tol: f64,
) -> Result<BandIrreps> {
    let n_basis = overlap.nrows();
    let n_bands = coeffs.ncols();
    if overlap.ncols() != n_basis
        || coeffs.nrows() != n_basis
        || energies.len() != n_bands
        || ao_actions.len() != dg_k.unitary.len()
        || ao_actions.iter().any(|m| m.shape() != (n_basis, n_basis))
    {
        return Err(invalid(
            "band irrep classification inputs have inconsistent dimensions",
        ));
    }
    let s_d_ops: Vec<DMatrix<C64>> = ao_actions.iter().map(|d| overlap * d).collect();
    let mut clusters = Vec::new();
    let mut max_unitarity_error = 0.0_f64;
    let mut lo = 0;
    while lo < n_bands {
        let mut hi = lo + 1;
        while hi < n_bands && (energies[hi] - energies[lo]).abs() <= degeneracy_tol {
            hi += 1;
        }
        let d = hi - lo;
        let c_sub = coeffs.columns(lo, d);
        let mut mats = Vec::with_capacity(dg_k.unitary.len());
        for sd in &s_d_ops {
            let m_g = c_sub.adjoint() * sd * &c_sub;
            let err = (&m_g * m_g.adjoint() - DMatrix::<C64>::identity(d, d))
                .iter()
                .map(|c| c.norm())
                .fold(0., f64::max);
            max_unitarity_error = max_unitarity_error.max(err);
            mats.push(m_g);
        }
        let irreps = dg_k.classify_representation(&mats)?;
        let dominant_irrep = irreps
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|x| x.0)
            .ok_or_else(|| invalid("band cluster failed to match any double-group irrep"))?;
        let corep = dg_k.irreps[dominant_irrep].corep;
        clusters.push(BandCluster {
            lo,
            hi,
            energy: energies[lo..hi].iter().sum::<f64>() / d as f64,
            irreps,
            dominant_irrep,
            corep,
        });
        lo = hi;
    }
    Ok(BandIrreps {
        clusters,
        max_unitarity_error,
    })
}

/// Symmetry-projected block-diagonalized generalized eigensolver for `(H, S)` under `dg_k` (Wave DG-3):
/// projects Lowdin orthonormal basis `S^{-1/2}` onto each irrep subspace
/// `P_alpha = (d_alpha / |G|) sum_g chi_alpha*(g) S^{-1/2} (S D_ao[g]) S^{-1/2}`
/// and diagonalizes decoupled invariant blocks `H_alpha = Q_alpha^H H Q_alpha`.
pub fn irrep_blocked_diagonalize(
    dg_k: &DoubleGroup,
    hamiltonian: &DMatrix<C64>,
    overlap: &DMatrix<C64>,
    ao_actions: &[DMatrix<C64>],
) -> Result<IrrepSolved> {
    IrrepProjectors::from_actions(dg_k, overlap, ao_actions)?.solve_hamiltonian(hamiltonian)
}

/// Track band-cluster connectivity between two nearby `k`-points via Bloch overlap `|C_a^H S C_b|`,
/// returning `(cluster_a, cluster_b, total_overlap_weight)` pairs.
pub fn track_band_connectivity(
    clusters_a: &BandIrreps,
    coeffs_a: &DMatrix<C64>,
    overlap_ab: &DMatrix<C64>,
    clusters_b: &BandIrreps,
    coeffs_b: &DMatrix<C64>,
) -> Vec<(usize, usize, f64)> {
    let over = coeffs_a.adjoint() * overlap_ab * coeffs_b;
    let mut matches = Vec::with_capacity(clusters_a.clusters.len());
    for (ia, ca) in clusters_a.clusters.iter().enumerate() {
        let mut best_b = 0;
        let mut best_w = -1.0_f64;
        for (ib, cb) in clusters_b.clusters.iter().enumerate() {
            let mut w = 0.0_f64;
            for r in ca.lo..ca.hi {
                for c in cb.lo..cb.hi {
                    w += over[(r, c)].norm_sqr();
                }
            }
            let norm_w = w / (ca.hi - ca.lo).max(1) as f64;
            if norm_w > best_w {
                best_w = norm_w;
                best_b = ib;
            }
        }
        matches.push((ia, best_b, best_w));
    }
    matches
}

/// Optical/multipole double-group selection mask (row-major `nv x nc`) for transitions
/// `v_irrep -> c_irrep` mediated by any operator component transforming as `dipole_irreps`.
pub fn dipole_selection_mask(
    dg: &DoubleGroup,
    v_irreps: &[usize],
    c_irreps: &[usize],
    dipole_irreps: &[usize],
) -> Result<Vec<bool>> {
    let mut out = Vec::with_capacity(v_irreps.len() * c_irreps.len());
    for &vi in v_irreps {
        for &ci in c_irreps {
            let mut allowed = false;
            for &di in dipole_irreps {
                if dg.allows_transition(vi, di, ci)? {
                    allowed = true;
                    break;
                }
            }
            out.push(allowed);
        }
    }
    Ok(out)
}

/// Result of classifying a converged PAW crystal `k`-point under its crystallographic or magnetic
/// double little group.
#[derive(Clone, Debug)]
pub struct CrystalBandIrreps {
    pub double_group: DoubleGroup,
    pub irreps: BandIrreps,
    /// `max_g ||D_ao[g]^H S D_ao[g] - S||_inf` verifying PAW overlap invariance.
    pub overlap_residual: f64,
}

/// Full `(n_basis, n_basis)` Bloch overlap matrix (`I_2 \otimes S` for spinor crystals,
/// `S` for scalar crystals).
pub fn full_overlap(c: &Crystal, point: &KPoint) -> DMatrix<C64> {
    if !c.spinor {
        return point.overlap.clone();
    }
    let m = c.m;
    let mut s_full = DMatrix::<C64>::zeros(2 * m, 2 * m);
    for s in 0..2 {
        s_full
            .view_mut((s * m, s * m), (m, m))
            .copy_from(&point.overlap);
    }
    s_full
}

/// Maximum `||D_ao[g]^H S D_ao[g] - S||_inf` across all unitary little-group operations `g`.
pub fn paw_overlap_residual(overlap: &DMatrix<C64>, ao_actions: &[DMatrix<C64>]) -> f64 {
    ao_actions
        .iter()
        .map(|d| {
            (d.adjoint() * overlap * d - overlap)
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max)
}

/// Build `DoubleGroup` at `k_frac` from `Crystal` (uses `description.operations` when present,
/// otherwise derives spatial space-group operations on `description.centers` for nonmagnetic or
/// time-reversal-invariant crystals).
pub fn crystal_double_group(c: &Crystal, k_frac: [f64; 3]) -> Result<DoubleGroup> {
    let desc = c.description.as_ref().ok_or_else(|| {
        invalid("crystal double-group analysis needs the setup description block")
    })?;
    let lattice = crystal_lattice(&c.grid);
    if let Some(ops) = &desc.operations {
        return DoubleGroup::from_description(ops, &lattice, Some(k_frac));
    }
    if c.spinor && !c.time_reversal {
        return Err(invalid(
            "magnetic spinor crystals without stored operations require magnetic symmetry classification",
        ));
    }
    let mut labels = Vec::with_capacity(desc.centers.len());
    let mut kind_ids: HashMap<&str, u32> = HashMap::new();
    let mut fractional = Vec::with_capacity(desc.centers.len());
    let inv = lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular crystal lattice"))?;
    for (i, center) in desc.centers.iter().enumerate() {
        let next_id = kind_ids.len() as u32 + 1;
        let id = *kind_ids
            .entry(desc.datasets[i].0.as_str())
            .or_insert(next_id);
        labels.push(id);
        let f = inv * Vector3::from(*center);
        fractional.push([f[0], f[1], f[2]]);
    }
    let ops = find(
        &Cell {
            lattice,
            fractional: &fractional,
            labels: &labels,
        },
        1e-4,
    )?;
    DoubleGroup::from_operations(&ops, &lattice, None, Some(k_frac))
}

/// Construct exact PAW AO transformation matrices `D_ao[g]` (`(2m, 2m)` spin-major if `c.spinor`,
/// `(m, m)` if scalar) for every unitary element `g` in `dg_k` using complex Condon-Shortley
/// harmonic rotations (`Y_lm(R^-1 u) = sum_{m'} Y_{lm'}(u) T^l_{m m'}(R)^*`), Bloch lattice shifts
/// `exp(-2 pi i k . n_a)`, and `SU(2)` spinor matrices.
pub fn paw_ao_actions_from_layout(
    dg_k: &DoubleGroup,
    lattice: &Matrix3<f64>,
    centers: &[[f64; 3]],
    atom_of: &[usize],
    l_of: &[usize],
    m_of: &[i32],
    radial_of: &[usize],
    spinor: bool,
) -> Result<Vec<DMatrix<C64>>> {
    let m = atom_of.len();
    if l_of.len() != m || m_of.len() != m || radial_of.len() != m {
        return Err(invalid(
            "PAW AO basis metadata slices must have equal length",
        ));
    }
    if centers.is_empty() || m == 0 {
        return Err(invalid("PAW AO basis cannot be empty"));
    }
    let inv_lattice = lattice
        .try_inverse()
        .ok_or_else(|| invalid("singular crystal lattice in PAW AO action"))?;
    let frac_centers: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| inv_lattice * Vector3::from(*c))
        .collect();
    // Rank distinct radial shells within each `(atom, l)` so DZ/TZ shells stay uniquely matched.
    let mut shell_rank = vec![0usize; m];
    let mut seen_tables: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut slot_by_key: HashMap<(usize, usize, usize, i32), Vec<usize>> = HashMap::new();
    for orb in 0..m {
        let key_al = (atom_of[orb], l_of[orb]);
        let tables = seen_tables.entry(key_al).or_default();
        let rank = match tables.iter().position(|&t| t == radial_of[orb]) {
            Some(pos) => pos,
            None => {
                tables.push(radial_of[orb]);
                tables.len() - 1
            }
        };
        shell_rank[orb] = rank;
        slot_by_key
            .entry((atom_of[orb], l_of[orb], rank, m_of[orb]))
            .or_default()
            .push(orb);
    }
    let max_l = *l_of.iter().max().unwrap_or(&0);
    let mut actions = Vec::with_capacity(dg_k.unitary.len());
    for el in &dg_k.unitary {
        let w_f = el.rotation.map(f64::from);
        let tau = Vector3::from(el.translation);
        let tables = rotation_tables(&el.cartesian, max_l, 1e-6)?;
        let mut atom_map = Vec::with_capacity(centers.len());
        let mut atom_phase = Vec::with_capacity(centers.len());
        for frac_a in &frac_centers {
            let moved = w_f * frac_a + tau;
            let mut best = None;
            let mut best_dist = f64::INFINITY;
            let mut best_shift = Vector3::zeros();
            for (b, frac_b) in frac_centers.iter().enumerate() {
                let diff = moved - frac_b;
                let shift = diff.map(f64::round);
                let dist = (lattice * (diff - shift)).norm();
                if dist < best_dist {
                    best_dist = dist;
                    best = Some(b);
                    best_shift = shift;
                }
            }
            if best_dist > 1e-3 {
                return Err(invalid(
                    "symmetry operation does not map crystal atomic centers onto themselves",
                ));
            }
            atom_map.push(best.expect("nonempty centers"));
            let dot = dg_k.k_frac[0] * best_shift[0]
                + dg_k.k_frac[1] * best_shift[1]
                + dg_k.k_frac[2] * best_shift[2];
            atom_phase.push(C64::from_polar(1., -2. * PI * dot));
        }

        let mut d_spatial = DMatrix::<C64>::zeros(m, m);
        for orb_a in 0..m {
            let (a, l, m_a, rank) = (atom_of[orb_a], l_of[orb_a], m_of[orb_a], shell_rank[orb_a]);
            let b = atom_map[a];
            let phase = atom_phase[a];
            let t_l = &tables[l];
            let col_a = (m_a + l as i32) as usize;
            for m_b in -(l as i32)..=(l as i32) {
                let row_b = (m_b + l as i32) as usize;
                let coeff = phase * t_l[(col_a, row_b)].conj();
                if coeff.norm() < 1e-14 {
                    continue;
                }
                let targets = slot_by_key.get(&(b, l, rank, m_b)).ok_or_else(|| {
                    invalid("mapped atom lacks matching PAW basis shell under symmetry")
                })?;
                if targets.len() != 1 {
                    return Err(invalid("ambiguous PAW basis shell slot under symmetry"));
                }
                d_spatial[(targets[0], orb_a)] = coeff;
            }
        }

        if spinor {
            let mut d_full = DMatrix::<C64>::zeros(2 * m, 2 * m);
            for s_out in 0..2 {
                for s_in in 0..2 {
                    let spin_coeff = el.su2[s_out][s_in];
                    if spin_coeff == C64::new(0., 0.) {
                        continue;
                    }
                    let mut view = d_full.view_mut((s_out * m, s_in * m), (m, m));
                    view.copy_from(&(&d_spatial * spin_coeff));
                }
            }
            actions.push(d_full);
        } else {
            actions.push(d_spatial);
        }
    }
    Ok(actions)
}

/// Convenience wrapper for [`paw_ao_actions_from_layout`] reading AO metadata directly from a
/// [`Crystal`]'s `Description`.
pub fn paw_ao_actions(dg_k: &DoubleGroup, c: &Crystal) -> Result<Vec<DMatrix<C64>>> {
    let desc = c
        .description
        .as_ref()
        .ok_or_else(|| invalid("PAW AO actions require the crystal description block"))?;
    let lattice = crystal_lattice(&c.grid);
    let l_of: Vec<usize> = desc
        .function_table
        .iter()
        .map(|&t| desc.tables[t].l)
        .collect();
    paw_ao_actions_from_layout(
        dg_k,
        &lattice,
        &desc.centers,
        &desc.function_atom,
        &l_of,
        &desc.function_m,
        &desc.function_table,
        c.spinor,
    )
}

fn point_exchange(
    c: &Crystal,
    result: &PeriodicResult,
    point: &KPoint,
    spin: usize,
) -> Result<Option<DMatrix<C64>>> {
    if c.screened_exchange()?.is_some() {
        Ok(Some(c.hybrid_exchange_at_k(
            &result.matrices,
            point,
            spin,
            result.options.exxdiv_treatment,
        )?))
    } else {
        Ok(None)
    }
}

/// Assemble `(hamiltonian, full_overlap)` in the AO (or spin-major spinor AO) basis for `point`
/// and spin channel `spin` from a converged `PeriodicResult`.
pub fn crystal_hamiltonian_and_overlap(
    c: &Crystal,
    result: &PeriodicResult,
    point: &KPoint,
    spin: usize,
) -> Result<(DMatrix<C64>, DMatrix<C64>)> {
    if spin >= c.spins {
        return Err(invalid(format!(
            "spin index {spin} out of range for crystal with {} spin channel(s)",
            c.spins
        )));
    }
    let n = c.grid.index.points();
    let exchange = point_exchange(c, result, point, spin)?;
    let potential = result.potential.as_slice();
    let kinetic = result
        .kinetic_potential
        .as_ref()
        .map(|v| &v.as_slice()[spin * n..(spin + 1) * n]);
    let hubbard = result.hubbard.as_ref().map(|h| &h.operators[spin]);
    let h = if c.spinor {
        super::spinor::hamiltonian(
            c,
            point,
            potential,
            result.moment_potential.as_slice(),
            kinetic,
            hubbard,
            exchange.as_ref(),
        )?
    } else {
        solver::hamiltonian(
            c,
            point,
            &potential[spin * n..(spin + 1) * n],
            result.moment_potential.as_slice(),
            kinetic,
            hubbard,
            exchange.as_ref(),
            None,
        )?
    };
    Ok((h, full_overlap(c, point)))
}

/// Classify a `KPoint`'s converged eigenvalues and AO eigenvectors under `dg_k`.
fn classify_point_states(
    c: &Crystal,
    point: &KPoint,
    k_frac: [f64; 3],
    energies: &[f64],
    coeffs: &DMatrix<C64>,
    degeneracy_tol: f64,
) -> Result<CrystalBandIrreps> {
    let dg_k = crystal_double_group(c, k_frac)?;
    let ao_actions = paw_ao_actions(&dg_k, c)?;
    let overlap = full_overlap(c, point);
    let overlap_residual = paw_overlap_residual(&overlap, &ao_actions);
    let irreps = classify_bloch_bands(
        &dg_k,
        energies,
        &overlap,
        coeffs,
        &ao_actions,
        degeneracy_tol,
    )?;
    Ok(CrystalBandIrreps {
        double_group: dg_k,
        irreps,
        overlap_residual,
    })
}

/// Classify the converged Bloch bands at mesh point `k_index` (`spin` channel) under `dg_k`.
pub fn classify_crystal_kpoint(
    c: &Crystal,
    result: &PeriodicResult,
    spin: usize,
    k_index: usize,
    degeneracy_tol: f64,
) -> Result<CrystalBandIrreps> {
    if spin >= c.spins || k_index >= c.points.len() {
        return Err(invalid("spin or k_index out of bounds for crystal result"));
    }
    let point = &c.points[k_index];
    let idx = spin * c.points.len() + k_index;
    let coeffs = result.coefficients[idx].matrix(&c.context)?;
    classify_point_states(
        c,
        point,
        point.fractional,
        &result.eigenvalues[idx],
        &coeffs,
        degeneracy_tol,
    )
}

/// Solve converged eigenvalues (including `c.reference`) and AO eigenvectors at an assembled
/// off-mesh `point` for `spin` channel.
fn solve_point_states(
    c: &Crystal,
    result: &PeriodicResult,
    point: &KPoint,
    spin: usize,
) -> Result<(Vec<f64>, DMatrix<C64>)> {
    if spin >= c.spins {
        return Err(invalid("spin index out of range for crystal"));
    }
    let n = c.grid.index.points();
    let exchange = point_exchange(c, result, point, spin)?;
    let potential = result.potential.as_slice();
    let (eps, coeffs, _) = solver::diagonalize_with_exchange(
        c,
        point,
        if c.spinor {
            potential
        } else {
            &potential[spin * n..(spin + 1) * n]
        },
        result.moment_potential.as_slice(),
        result
            .kinetic_potential
            .as_ref()
            .map(|v| &v.as_slice()[spin * n..(spin + 1) * n]),
        result.hubbard.as_ref().map(|h| &h.operators[spin]),
        exchange.as_ref(),
    )?;
    let energies: Vec<f64> = eps.iter().map(|&e| e + c.reference).collect();
    Ok((energies, coeffs))
}

/// Solve converged states at an arbitrary high-symmetry `k_frac` using `setup` and classify
/// every degenerate Bloch band cluster under the crystallographic/magnetic double little group.
pub fn classify_crystal_at(
    c: &Crystal,
    setup: &PeriodicSetup,
    result: &PeriodicResult,
    spin: usize,
    k_frac: [f64; 3],
    degeneracy_tol: f64,
) -> Result<(CrystalBandIrreps, Vec<f64>, DMatrix<C64>)> {
    if spin >= c.spins {
        return Err(invalid("spin index out of range for crystal"));
    }
    let points = setup.matrices_at(&c.context, &[k_frac], &[1.0])?;
    let point = &points[0];
    let (energies, coeffs) = solve_point_states(c, result, point, spin)?;
    let classified = classify_point_states(c, point, k_frac, &energies, &coeffs, degeneracy_tol)?;
    Ok((classified, energies, coeffs))
}

/// One point along a symmetry-labeled Bloch band path (`links_from_prev` gives
/// `(cluster_prev, cluster_cur, overlap_weight)` from the previous step via [`track_band_connectivity`]).
#[derive(Clone, Debug)]
pub struct KPathStepIrreps {
    pub k_frac: [f64; 3],
    pub classification: CrystalBandIrreps,
    pub energies: Vec<f64>,
    pub links_from_prev: Vec<(usize, usize, f64)>,
}

/// Solve converged Bloch states along `path_k`, classify every step under its crystallographic or
/// magnetic double little group `G_k`, and track continuous band-cluster connectivity across
/// consecutive steps along the path.
pub fn classify_kpath(
    c: &Crystal,
    setup: &PeriodicSetup,
    result: &PeriodicResult,
    spin: usize,
    path_k: &[[f64; 3]],
    degeneracy_tol: f64,
) -> Result<Vec<KPathStepIrreps>> {
    if spin >= c.spins {
        return Err(invalid("spin index out of range for crystal"));
    }
    if path_k.is_empty() {
        return Err(invalid("band path k-point list cannot be empty"));
    }
    let weights = vec![1.0 / path_k.len() as f64; path_k.len()];
    let points = setup.matrices_at(&c.context, path_k, &weights)?;
    let mut out = Vec::with_capacity(path_k.len());
    let mut prev: Option<(BandIrreps, DMatrix<C64>)> = None;
    for (&k_frac, point) in path_k.iter().zip(&points) {
        let (energies, coeffs) = solve_point_states(c, result, point, spin)?;
        let classification =
            classify_point_states(c, point, k_frac, &energies, &coeffs, degeneracy_tol)?;
        let links_from_prev = match &prev {
            Some((prev_irreps, prev_coeffs)) => {
                let overlap = full_overlap(c, point);
                track_band_connectivity(
                    prev_irreps,
                    prev_coeffs,
                    &overlap,
                    &classification.irreps,
                    &coeffs,
                )
            }
            None => Vec::new(),
        };
        prev = Some((classification.irreps.clone(), coeffs));
        out.push(KPathStepIrreps {
            k_frac,
            classification,
            energies,
            links_from_prev,
        });
    }
    Ok(out)
}

/// Cartesian electric-dipole selection masks between valence `[0..n_valence]` and conduction
/// `[n_valence..n_valence + n_conduction]` Bloch states at wavevector `k`.
#[derive(Clone, Debug)]
pub struct DipoleSelection {
    pub double_group: DoubleGroup,
    pub irreps: BandIrreps,
    pub overlap_residual: f64,
    pub n_valence: usize,
    pub n_conduction: usize,
    /// Per-axis dipole weights (`W_xx, W_yy, W_zz`, row-major `n_valence x n_conduction`).
    pub weights: Vec<[f64; 3]>,
    /// Per-axis selection masks (`[x, y, z]`, each row-major `n_valence x n_conduction`).
    pub allowed_axes: [Vec<bool>; 3],
    /// Combined mask (`allowed_x || allowed_y || allowed_z`, row-major `n_valence x n_conduction`).
    pub allowed: Vec<bool>,
}

/// Build Cartesian dipole selection rules (`nv x nc`) from an already-classified `k`-point.
pub fn dipole_selection_from_classification(
    classified: &CrystalBandIrreps,
    n_valence: usize,
) -> Result<DipoleSelection> {
    let n_bands = classified.irreps.clusters.last().map_or(0, |cl| cl.hi);
    if n_valence == 0 || n_valence >= n_bands {
        return Err(invalid(format!(
            "dipole selection needs 0 < n_valence < n_bands; got n_valence={n_valence}, n_bands={n_bands}"
        )));
    }
    let n_conduction = n_bands - n_valence;
    let mut band_decomp: Vec<&[(usize, f64)]> = vec![&[]; n_bands];
    for cl in &classified.irreps.clusters {
        for b in cl.lo..cl.hi {
            band_decomp[b] = &cl.irreps;
        }
    }
    let total = n_valence * n_conduction;
    let mut weights = Vec::with_capacity(total);
    let mut allowed_axes: [Vec<bool>; 3] = [
        Vec::with_capacity(total),
        Vec::with_capacity(total),
        Vec::with_capacity(total),
    ];
    let mut allowed = Vec::with_capacity(total);
    const WEIGHT_TOL: f64 = 1e-6;
    for v in 0..n_valence {
        for c in n_valence..n_bands {
            let mut w_vc = [0.0_f64; 3];
            for &(vi, wv) in band_decomp[v] {
                if wv <= 1e-4 {
                    continue;
                }
                for &(ci, wc) in band_decomp[c] {
                    if wc <= 1e-4 {
                        continue;
                    }
                    let ax = classified.double_group.dipole_axis_weights(vi, ci)?;
                    for a in 0..3 {
                        w_vc[a] += wv * wc * ax[a].max(0.0);
                    }
                }
            }
            let ax_ok = [
                w_vc[0] > WEIGHT_TOL,
                w_vc[1] > WEIGHT_TOL,
                w_vc[2] > WEIGHT_TOL,
            ];
            for a in 0..3 {
                allowed_axes[a].push(ax_ok[a]);
            }
            allowed.push(ax_ok[0] || ax_ok[1] || ax_ok[2]);
            weights.push(w_vc);
        }
    }
    Ok(DipoleSelection {
        double_group: classified.double_group.clone(),
        irreps: classified.irreps.clone(),
        overlap_residual: classified.overlap_residual,
        n_valence,
        n_conduction,
        weights,
        allowed_axes,
        allowed,
    })
}

fn default_valence_count(c: &Crystal, energies: &[f64]) -> usize {
    if c.spinor || c.spins == 2 {
        (c.electrons.round() as usize).clamp(1, energies.len().saturating_sub(1))
    } else {
        ((c.electrons / 2.0).round() as usize).clamp(1, energies.len().saturating_sub(1))
    }
}

/// Classify a converged mesh `k_index` and compute its Cartesian dipole selection masks between
/// occupied and unoccupied bands.
pub fn crystal_dipole_selection(
    c: &Crystal,
    result: &PeriodicResult,
    spin: usize,
    k_index: usize,
    degeneracy_tol: f64,
) -> Result<DipoleSelection> {
    let classified = classify_crystal_kpoint(c, result, spin, k_index, degeneracy_tol)?;
    let idx = spin * c.points.len() + k_index;
    let nv = default_valence_count(c, &result.eigenvalues[idx]);
    dipole_selection_from_classification(&classified, nv)
}

/// Solve converged states at arbitrary `k_frac` using `setup` and compute its double-group
/// Cartesian dipole selection masks.
pub fn crystal_dipole_selection_at(
    c: &Crystal,
    setup: &PeriodicSetup,
    result: &PeriodicResult,
    spin: usize,
    k_frac: [f64; 3],
    degeneracy_tol: f64,
) -> Result<DipoleSelection> {
    let (classified, energies, _) =
        classify_crystal_at(c, setup, result, spin, k_frac, degeneracy_tol)?;
    let nv = default_valence_count(c, &energies);
    dipole_selection_from_classification(&classified, nv)
}
