//! Sparse grid spatial index and coset translation search in reduced coordinates.
use crate::periodic::invalid;
use crate::Result;
use nalgebra::Matrix3;
use std::collections::HashMap;

/// Sparse periodic spatial grid in reduced fractional coordinates.
pub struct SparseGrid {
    bin_sizes: [f64; 3],
    n_bins: [i64; 3],
    cells: HashMap<[i64; 3], Vec<usize>>,
}

impl SparseGrid {
    pub fn new(xb: &[[f64; 3]], b: &Matrix3<f64>, tol: f64) -> Result<Self> {
        let b_inv = b
            .try_inverse()
            .ok_or_else(|| invalid("singular reduced basis"))?;
        let b_inv_t = b_inv.transpose();

        let mut bin_sizes = [0.0f64; 3];
        let mut n_bins = [0i64; 3];

        for i in 0..3 {
            let recip_len = b_inv_t.row(i).norm();
            let height = 1.0 / recip_len;
            let bin_i = tol / height;
            bin_sizes[i] = bin_i;
            let n = (1.0 / bin_i).ceil() as i64;
            n_bins[i] = n.max(1);
        }

        let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (idx, p) in xb.iter().enumerate() {
            let key = Self::bin_key(p, &bin_sizes, &n_bins);
            cells.entry(key).or_default().push(idx);
        }

        Ok(Self {
            bin_sizes,
            n_bins,
            cells,
        })
    }

    #[inline]
    fn bin_key(p: &[f64; 3], bin_sizes: &[f64; 3], n_bins: &[i64; 3]) -> [i64; 3] {
        std::array::from_fn(|i| {
            let wrapped = p[i].rem_euclid(1.0);
            let b = (wrapped / bin_sizes[i]).floor() as i64;
            b.rem_euclid(n_bins[i])
        })
    }

    /// Find an atom matching position y with the given label within distance tol.
    pub fn find_match(
        &self,
        y: &[f64; 3],
        label: u32,
        xb: &[[f64; 3]],
        labels: &[u32],
        b: &Matrix3<f64>,
        tol: f64,
    ) -> Option<usize> {
        let center = Self::bin_key(y, &self.bin_sizes, &self.n_bins);
        let mut best_idx = None;
        let mut best_dist = tol;

        for d0 in -1..=1 {
            let b0 = (center[0] + d0).rem_euclid(self.n_bins[0]);
            for d1 in -1..=1 {
                let b1 = (center[1] + d1).rem_euclid(self.n_bins[1]);
                for d2 in -1..=1 {
                    let b2 = (center[2] + d2).rem_euclid(self.n_bins[2]);
                    if let Some(candidates) = self.cells.get(&[b0, b1, b2]) {
                        for &cand in candidates {
                            if labels[cand] != label {
                                continue;
                            }
                            let dx = xb[cand][0] - y[0];
                            let dy = xb[cand][1] - y[1];
                            let dz = xb[cand][2] - y[2];
                            let d = [
                                dx - dx.round(),
                                dy - dy.round(),
                                dz - dz.round(),
                            ];
                            let vx = d[0] * b[(0, 0)] + d[1] * b[(1, 0)] + d[2] * b[(2, 0)];
                            let vy = d[0] * b[(0, 1)] + d[1] * b[(1, 1)] + d[2] * b[(2, 1)];
                            let vz = d[0] * b[(0, 2)] + d[1] * b[(1, 2)] + d[2] * b[(2, 2)];
                            let dist = (vx * vx + vy * vy + vz * vz).sqrt();
                            if dist <= best_dist {
                                best_dist = dist;
                                best_idx = Some(cand);
                            }
                        }
                    }
                }
            }
        }
        best_idx
    }
}

/// A representative symmetry operation in the reduced basis with its atom permutation.
#[derive(Clone, Debug)]
pub struct OperationReduced {
    pub rotation: Matrix3<i32>,
    pub translation: [f64; 3],
    pub atom_map: Vec<usize>,
}

/// Pure translations and distinct coset representatives in reduced coordinates.
pub struct CosetDecomposition {
    pub pure_translations: Vec<OperationReduced>,
    pub coset_representatives: Vec<OperationReduced>,
}

/// Find pure translations and coset representatives for given candidate rotations.
pub fn find_cosets(
    xb: &[[f64; 3]],
    labels: &[u32],
    b: &Matrix3<f64>,
    candidate_rotations: &[Matrix3<i32>],
    tol: f64,
) -> Result<CosetDecomposition> {
    let n_atoms = xb.len();
    if n_atoms == 0 {
        return Err(invalid("cell must contain at least one atom"));
    }
    if n_atoms != labels.len() {
        return Err(invalid("coordinates and labels length mismatch"));
    }

    let grid = SparseGrid::new(xb, b, tol)?;

    // Identify rarest label
    let mut label_counts: HashMap<u32, usize> = HashMap::new();
    for &l in labels {
        *label_counts.entry(l).or_default() += 1;
    }
    let (&rare_label, _) = label_counts
        .iter()
        .min_by_key(|(&l, &cnt)| (cnt, l))
        .ok_or_else(|| invalid("no labels provided"))?;

    let anchor_idx = labels
        .iter()
        .position(|&l| l == rare_label)
        .ok_or_else(|| invalid("anchor atom not found"))?;
    let x_anchor = xb[anchor_idx];

    let same_rare_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, &l)| if l == rare_label { Some(i) } else { None })
        .collect();

    let mut pure_translations = Vec::new();
    let mut coset_representatives = Vec::new();

    // Verify identity operation first to gather all pure translations
    let identity = Matrix3::identity();
    for &j in &same_rare_indices {
        let t_cand = std::array::from_fn(|c| (xb[j][c] - x_anchor[c]).rem_euclid(1.0));
        if let Some(op) = verify_operation(&identity, &t_cand, xb, labels, b, tol, &grid) {
            pure_translations.push(op);
        }
    }

    if pure_translations.is_empty() {
        return Err(invalid("identity operation failed verification"));
    }

    // Now find one coset representative for each candidate rotation
    for w in candidate_rotations {
        if *w == identity {
            // Representative of identity is (I, [0, 0, 0])
            let zero_op = pure_translations
                .iter()
                .find(|op| {
                    op.translation[0].abs() < 1e-6
                        && op.translation[1].abs() < 1e-6
                        && op.translation[2].abs() < 1e-6
                })
                .cloned()
                .unwrap_or_else(|| pure_translations[0].clone());
            coset_representatives.push(zero_op);
            continue;
        }

        // Image of anchor atom under rotation: x_anchor * W^T
        let img_anchor = [
            x_anchor[0] * w[(0, 0)] as f64 + x_anchor[1] * w[(0, 1)] as f64 + x_anchor[2] * w[(0, 2)] as f64,
            x_anchor[0] * w[(1, 0)] as f64 + x_anchor[1] * w[(1, 1)] as f64 + x_anchor[2] * w[(1, 2)] as f64,
            x_anchor[0] * w[(2, 0)] as f64 + x_anchor[1] * w[(2, 1)] as f64 + x_anchor[2] * w[(2, 2)] as f64,
        ];

        let mut found = None;
        for &j in &same_rare_indices {
            let t_cand = std::array::from_fn(|c| (xb[j][c] - img_anchor[c]).rem_euclid(1.0));
            if let Some(op) = verify_operation(w, &t_cand, xb, labels, b, tol, &grid) {
                found = Some(op);
                break; // Stop at first valid translation for this rotation
            }
        }

        if let Some(op) = found {
            coset_representatives.push(op);
        }
    }

    Ok(CosetDecomposition {
        pure_translations,
        coset_representatives,
    })
}

/// Verify if (W, t) maps the cell onto itself bijectively within tol, and refine t.
fn verify_operation(
    w: &Matrix3<i32>,
    t_raw: &[f64; 3],
    xb: &[[f64; 3]],
    labels: &[u32],
    b: &Matrix3<f64>,
    tol: f64,
    grid: &SparseGrid,
) -> Option<OperationReduced> {
    let n = xb.len();
    let mut atom_map = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut residuals = [0.0f64; 3];

    for a in 0..n {
        let xa = xb[a];
        let ya = [
            xa[0] * w[(0, 0)] as f64 + xa[1] * w[(0, 1)] as f64 + xa[2] * w[(0, 2)] as f64 + t_raw[0],
            xa[0] * w[(1, 0)] as f64 + xa[1] * w[(1, 1)] as f64 + xa[2] * w[(1, 2)] as f64 + t_raw[1],
            xa[0] * w[(2, 0)] as f64 + xa[1] * w[(2, 1)] as f64 + xa[2] * w[(2, 2)] as f64 + t_raw[2],
        ];

        let target_idx = grid.find_match(&ya, labels[a], xb, labels, b, tol)?;
        if visited[target_idx] {
            return None; // Not a bijection
        }
        visited[target_idx] = true;
        atom_map.push(target_idx);

        let dx = xb[target_idx][0] - ya[0];
        let dy = xb[target_idx][1] - ya[1];
        let dz = xb[target_idx][2] - ya[2];
        residuals[0] += dx - dx.round();
        residuals[1] += dy - dy.round();
        residuals[2] += dz - dz.round();
    }

    // Refine translation by mean residual
    let inv_n = 1.0 / n as f64;
    let mut t_refined = [
        (t_raw[0] + residuals[0] * inv_n).rem_euclid(1.0),
        (t_raw[1] + residuals[1] * inv_n).rem_euclid(1.0),
        (t_raw[2] + residuals[2] * inv_n).rem_euclid(1.0),
    ];

    // Snap to 0 if within snap threshold of 0 or 1
    let snap = 1e-6;
    for c in 0..3 {
        if t_refined[c] < snap || t_refined[c] > 1.0 - snap {
            t_refined[c] = 0.0;
        }
    }

    Some(OperationReduced {
        rotation: *w,
        translation: t_refined,
        atom_map,
    })
}
