//! A deterministic frame for exactly degenerate orbital clusters.
//!
//! The SCF fixes an orbital only up to the freedom its eigenproblem leaves: inside a cluster of
//! orbitals whose labels (orbital energies of a canonical frame, natural occupations of a
//! natural-orbital frame) agree within [`DEGENERACY_TOLERANCE`], any orthogonal rotation is as
//! good an answer as the one the eigensolver returned, and which one it returns changes with
//! the roundoff of the matrix. A truncated ansatz (a short operator sequence, an active space
//! that cuts the cluster) is not invariant under that rotation, so its energy would jump
//! between nearly identical geometries.
//!
//! [`fix_cluster_gauge`] removes the freedom: inside each cluster it diagonalizes fixed
//! lab-frame operators (quadrupole-type moments about the nuclear charge centroid, with
//! distinct axis weights), orders the orbitals by the eigenvalues and fixes each sign by a
//! fixed linear functional of the orbital function itself. Nothing in the result depends on
//! the incoming rotation of the cluster. The helper is pure over the orbitals, labels,
//! occupation domains and a moment provider; [`lab_moments`] is the provider that integrates
//! the operators on the molecular grid.
use super::{invalid, response::DEGENERACY_TOLERANCE, MeanField, MolecularIntegrals};
use crate::math::{
    linalg::real_eigen,
    storage::{Precision, SampleBlocks},
};
use crate::{product, quantum::C64, Result};
use nalgebra::{DMatrix, DVector};

/// The operator matrices and orbital functionals of a block of orbitals, in the block's own
/// basis (the `k` columns handed to the provider).
pub struct Moments {
    /// Real symmetric `k x k` operators in the order they are tried; the first resolves what it
    /// can, each next one only the eigenspaces still degenerate. Empty when the provider was
    /// asked for the functionals only.
    pub operators: Vec<DMatrix<f64>>,
    /// Linear functionals `<f|psi_c>` of the orbital functions (a `k` vector each) with the
    /// norm of `f`; the first one that is not negligible fixes each orbital's sign.
    pub functionals: Vec<(DVector<f64>, f64)>,
    /// The relative separation of two operator eigenvalues below which they count as one.
    pub resolution: f64,
}

/// Which of two orbitals' functionals counts as nonzero, relative to the norm of the functional.
const SIGN_THRESHOLD: f64 = 1e-6;

/// The clusters of `labels`: inside each occupation domain (a range of orbital indices that
/// the model keeps apart), the orbitals whose labels all lie within [`DEGENERACY_TOLERANCE`]
/// of the lowest one, each cluster of two or more as its ascending indices. A cluster the
/// domains split is two clusters (mixing an occupied orbital with an empty one changes the
/// state, not the frame).
pub fn clusters(labels: &[f64], domains: &[Vec<usize>]) -> Result<Vec<Vec<usize>>> {
    if labels.iter().any(|v| !v.is_finite()) {
        return Err(invalid("orbital labels must be finite"));
    }
    let mut out = Vec::new();
    for domain in domains {
        if domain.iter().any(|&i| i >= labels.len()) {
            return Err(invalid("an occupation domain reaches past the orbitals"));
        }
        let mut order: Vec<usize> = domain.clone();
        order.sort_by(|&a, &b| labels[a].total_cmp(&labels[b]).then(a.cmp(&b)));
        let mut current: Vec<usize> = Vec::new();
        for &i in &order {
            if current
                .first()
                .is_some_and(|&f| (labels[i] - labels[f]).abs() >= DEGENERACY_TOLERANCE)
            {
                out.push(std::mem::take(&mut current));
            }
            current.push(i);
        }
        out.push(current);
    }
    out.retain(|c| c.len() >= 2);
    for cluster in &mut out {
        cluster.sort_unstable();
    }
    Ok(out)
}

/// Flip each column of `u` (orbitals as combinations of the block's columns) whose first
/// non-negligible functional is negative: the sign of an orbital is then a property of its
/// function, not of the frame it was handed in.
fn orient(functionals: &[(DVector<f64>, f64)], u: &mut DMatrix<f64>) -> Result<()> {
    for c in 0..u.ncols() {
        let sign = functionals
            .iter()
            .map(|(f, scale)| (u.column(c).dot(f), *scale))
            .find(|(value, scale)| value.abs() > SIGN_THRESHOLD * scale)
            .map(|(value, _)| value);
        match sign {
            Some(value) if value < 0. => u.column_mut(c).scale_mut(-1.),
            Some(_) => {}
            None => {
                return Err(crate::Error::Unsupported(
                    "an orbital has no lab-frame moment that fixes its sign".into(),
                ))
            }
        }
    }
    Ok(())
}

/// The orthogonal `k x k` matrix that takes the cluster's orbitals to the deterministic frame.
fn resolve(moments: &Moments) -> Result<DMatrix<f64>> {
    let k = moments.operators.first().map_or(0, |q| q.nrows());
    if k == 0
        || moments
            .operators
            .iter()
            .any(|q| q.shape() != (k, k) || q.iter().any(|v| !v.is_finite()))
        || moments.functionals.iter().any(|(f, s)| {
            f.len() != k || !s.is_finite() || *s <= 0. || f.iter().any(|v| !v.is_finite())
        })
    {
        return Err(invalid(
            "cluster moments must be finite and match the cluster",
        ));
    }
    let mut u = DMatrix::<f64>::identity(k, k);
    // Groups of column positions of `u` that no operator has separated yet, in frame order.
    let mut groups: Vec<Vec<usize>> = vec![(0..k).collect()];
    for q in &moments.operators {
        let scale = q.iter().fold(1f64, |a, v| a.max(v.abs()));
        let gap = moments.resolution * scale;
        let mut next = Vec::with_capacity(groups.len());
        for group in groups {
            if group.len() == 1 {
                next.push(group);
                continue;
            }
            let sub = DMatrix::from_fn(k, group.len(), |r, c| u[(r, group[c])]);
            let projected = sub.transpose() * q * &sub;
            let (values, vectors) = real_eigen((&projected + projected.transpose()) * 0.5)?;
            let rotated = &sub * &vectors;
            for (j, &c) in group.iter().enumerate() {
                u.set_column(c, &rotated.column(j));
            }
            let mut start = 0;
            for j in 1..=group.len() {
                if j == group.len() || values[j] - values[j - 1] >= gap {
                    next.push(group[start..j].to_vec());
                    start = j;
                }
            }
        }
        groups = next;
    }
    if let Some(group) = groups.iter().find(|g| g.len() > 1) {
        return Err(crate::Error::Unsupported(format!(
            "{} orbitals of an exactly degenerate cluster stay degenerate under every lab-frame \
             operator tried (quadrupole, cubic), so no deterministic frame exists for them",
            group.len()
        )));
    }
    orient(&moments.functionals, &mut u)?;
    Ok(u)
}

/// Put every cluster of `orbitals` (columns over the orthonormal frame, labelled by `labels`,
/// kept apart by `domains`) into its deterministic frame and give every other orbital of the
/// domains its deterministic sign, in place; return the number of clusters changed. `moments`
/// is asked for operators only when a cluster exists (`operators` true), and once for the
/// functionals of the orbitals outside every cluster (`operators` false), whose columns are
/// otherwise untouched but for a sign.
pub fn fix_cluster_gauge(
    orbitals: &mut DMatrix<C64>,
    labels: &[f64],
    domains: &[Vec<usize>],
    moments: &mut dyn FnMut(&DMatrix<f64>, bool) -> Result<Moments>,
) -> Result<usize> {
    if labels.len() != orbitals.ncols() {
        return Err(invalid("a cluster frame needs one label per orbital"));
    }
    let found = clusters(labels, domains)?;
    for cluster in &found {
        let scale = cluster
            .iter()
            .flat_map(|&c| {
                orbitals
                    .column(c)
                    .iter()
                    .map(|v| v.norm())
                    .collect::<Vec<_>>()
            })
            .fold(1f64, f64::max);
        if cluster.iter().any(|&c| {
            orbitals
                .column(c)
                .iter()
                .any(|v| v.im.abs() > 1e-10 * scale)
        }) {
            return Err(invalid(
                "the cluster frame of degenerate orbitals needs real orbitals",
            ));
        }
        let block = DMatrix::from_fn(orbitals.nrows(), cluster.len(), |r, c| {
            orbitals[(r, cluster[c])].re
        });
        let u = resolve(&moments(&block, true)?)?;
        let rotated = &block * u;
        for (c, &target) in cluster.iter().enumerate() {
            for r in 0..orbitals.nrows() {
                orbitals[(r, target)] = C64::new(rotated[(r, c)], 0.);
            }
        }
    }
    // Every other orbital of the domains: only its sign.
    let single: Vec<usize> = domains
        .iter()
        .flatten()
        .copied()
        .filter(|c| !found.iter().any(|f| f.contains(c)))
        .collect();
    if !single.is_empty() {
        let block = DMatrix::from_fn(orbitals.nrows(), single.len(), |r, c| {
            orbitals[(r, single[c])]
        });
        if block.iter().any(|v| v.im.abs() > 1e-10 * v.norm().max(1.)) {
            return Err(invalid(
                "the sign frame of the orbitals needs real orbitals",
            ));
        }
        let real = block.map(|v| v.re);
        let mut flips = DMatrix::<f64>::identity(single.len(), single.len());
        orient(&moments(&real, false)?.functionals, &mut flips)?;
        for (c, &target) in single.iter().enumerate() {
            if flips[(c, c)] < 0. {
                orbitals.column_mut(target).scale_mut(-1.);
            }
        }
    }
    Ok(found.len())
}

/// The lab-frame operators, as functions of the position `u` from the nuclear charge centroid
/// (Bohr): a quadrupole with distinct axis weights, the off-diagonal quadrupole, two cubic
/// moments. Each is tried only on what the one before left degenerate.
fn operators(u: [f64; 3]) -> [f64; 4] {
    let [x, y, z] = u;
    [
        x * x + 1.7 * y * y + 2.9 * z * z,
        x * y + 1.3 * y * z + 1.9 * z * x,
        x * x * x + 1.4 * y * y * y + 2.1 * z * z * z,
        x * x * y + 1.3 * y * y * z + 1.7 * z * z * x,
    ]
}

/// Three fixed generic polynomials under a Gaussian envelope: linear functionals of an orbital
/// function whose sign is the orbital's own (unlike the sign of a coefficient in some other
/// frame). The next takes over when the one before vanishes by symmetry.
fn functionals(u: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = u;
    let envelope = (-(x * x + y * y + z * z) / 18.).exp();
    [
        (1. + 0.9 * x
            + 1.3 * y
            + 1.7 * z
            + 0.31 * x * y
            + 0.47 * y * z
            + 0.59 * z * x
            + 0.23 * x * x
            + 0.37 * y * y
            + 0.53 * z * z)
            * envelope,
        (0.7 - 1.1 * x + 0.6 * y - 0.8 * z + 0.43 * x * y - 0.29 * y * z
            + 0.17 * z * x
            + 0.13 * x * x * x
            + 0.19 * y * y * y
            + 0.11 * z * z * z)
            * envelope,
        (-0.4 + 0.5 * x + 1.2 * y - 1.5 * z - 0.61 * x * y + 0.37 * y * z - 0.23 * z * x
            + 0.29 * x * x
            - 0.41 * y * y
            + 0.17 * z * z
            + 0.07 * x * y * z)
            * envelope,
    ]
}

/// The moment provider of a molecule: [`operators`] and [`functionals`] about the nuclear
/// charge centroid, integrated on the grid of `integrals` over the cluster's functions (the
/// orthonormal samples times the cluster's coefficients, streamed in point blocks).
pub fn lab_moments(
    integrals: &MolecularIntegrals,
) -> impl FnMut(&DMatrix<f64>, bool) -> Result<Moments> + '_ {
    let charge: f64 = integrals
        .geometry
        .numbers()
        .iter()
        .map(|&z| f64::from(z))
        .sum();
    let mut center = [0.; 3];
    for (&z, r) in integrals
        .geometry
        .numbers()
        .iter()
        .zip(integrals.geometry.positions())
    {
        for axis in 0..3 {
            center[axis] += f64::from(z) * r[axis] / charge;
        }
    }
    move |orbitals, with_operators| {
        let ctx = &integrals.context;
        let m = integrals.m();
        let k = orbitals.ncols();
        if orbitals.nrows() != m {
            return Err(invalid(
                "cluster orbitals must be columns over the molecular basis",
            ));
        }
        let points = integrals.grid.index.points();
        let dv = integrals.grid.index.dv();
        let mut blocks = SampleBlocks::new(ctx, &integrals.orthonormal_samples, m, points)?;
        let stride = blocks.stride();
        let _work = ctx.reserve(product(&[stride, k + 16, 8])?)?;
        let mut operator_sums = vec![DMatrix::<f64>::zeros(k, k); 4 * usize::from(with_operators)];
        let mut functional_sums = vec![DVector::<f64>::zeros(k); 3];
        let mut functional_norms = [0f64; 3];
        let mut weights = [
            vec![0f64; stride],
            vec![0f64; stride],
            vec![0f64; stride],
            vec![0f64; stride],
        ];
        let mut linear = [vec![0f64; stride], vec![0f64; stride], vec![0f64; stride]];
        for start in (0..points).step_by(stride) {
            let count = blocks.load(start)?;
            let samples = nalgebra::DMatrixView::from_slice_with_strides_generic(
                blocks.values(),
                nalgebra::Dyn(count),
                nalgebra::Dyn(m),
                nalgebra::Const::<1>,
                nalgebra::Dyn(stride),
            );
            let psi: DMatrix<f64> = &samples * orbitals;
            for i in 0..count {
                let p = integrals.grid.point(start + i);
                let u = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                let (o, f) = (
                    if with_operators {
                        operators(u)
                    } else {
                        [0.; 4]
                    },
                    functionals(u),
                );
                for a in 0..4 {
                    weights[a][i] = o[a];
                }
                for a in 0..3 {
                    linear[a][i] = f[a];
                    functional_norms[a] += f[a] * f[a];
                }
            }
            for (sum, w) in operator_sums.iter_mut().zip(&weights) {
                let weighted = DMatrix::from_fn(count, k, |i, c| psi[(i, c)] * w[i]);
                *sum += psi.transpose() * weighted;
            }
            for (sum, f) in functional_sums.iter_mut().zip(&linear) {
                for c in 0..k {
                    sum[c] += (0..count).map(|i| psi[(i, c)] * f[i]).sum::<f64>();
                }
            }
        }
        Ok(Moments {
            operators: operator_sums
                .into_iter()
                .map(|q| (&q + q.transpose()) * (0.5 * dv))
                .collect(),
            functionals: functional_sums
                .into_iter()
                .zip(functional_norms)
                .map(|(f, n)| (f * dv, (n * dv).sqrt()))
                .collect(),
            resolution: match integrals.samples.precision() {
                Precision::Float64 => 1e-6,
                Precision::Float32 => 1e-3,
            },
        })
    }
}

impl MeanField {
    /// Whether the frame of this reference can be put into the deterministic frame of
    /// [`fix_cluster_gauge`]: a collinear determinant of real orbitals on an isolated grid.
    /// Spinors (every Kramers pair is a cluster, which needs a spin-aware operator), a periodic
    /// grid (a position operator is undefined on it) and a state that is not a determinant
    /// (fractional or non-aufbau occupations) are not covered: their clusters stay as the SCF
    /// returned them.
    pub(super) fn frame_covered(&self, integrals: &MolecularIntegrals) -> bool {
        matches!(self.method.as_str(), "rhf" | "uhf")
            && !integrals.grid.periodic
            && integrals.supercell.is_none()
            && !self.is_ensemble()
            && self.aufbau_inversion.is_none()
            && self.model_orbitals.iter().all(|v| v.im.abs() <= 1e-10)
    }
    /// Put the canonical orbitals' exactly degenerate clusters into the deterministic frame of
    /// [`fix_cluster_gauge`]. The labels are the orbital energies of a restricted frame, with
    /// the doubly occupied and the empty orbitals as domains, and the natural occupations of an
    /// unrestricted frame, with the doubly occupied, singly occupied and empty natural orbitals
    /// as domains. Every model, active-space selection, force and ansatz handover reads
    /// `model_orbitals`, so they all see the one frame. A frame with no cluster is not touched,
    /// and costs no grid pass. Not covered: a frame [`Self::frame_covered`] excludes.
    ///
    /// An unrestricted frame has clusters that no symmetry protects: `P_a + P_b` has rank at
    /// most `n_a + n_b`, so the empty natural orbitals are one exact zero-occupation cluster (and
    /// the doubly occupied ones, of a spin-symmetric state, one cluster at two). Their frame is
    /// deterministic here. Measured on linear H3 UHF (`tests/frame_force_gap.rs`), the production
    /// energy surface of an active space cutting the zero-occupation cluster equals the
    /// frame-independent polar-aligned oracle to 1e-3 meV/A, so this frame costs the force
    /// nothing; the analytic force of that case differs from both by about 20 meV/A, a defect of
    /// the force whose cause is open (pinned by that test). Symmetry-protected clusters under
    /// symmetry-preserving steps are exact.
    pub(super) fn fix_frame(&mut self, integrals: &MolecularIntegrals) -> Result<()> {
        if !self.frame_covered(integrals) {
            return Ok(());
        }
        let m = self.model_orbitals.ncols();
        let (labels, domains): (Vec<f64>, Vec<Vec<usize>>) = if self.method == "rhf" {
            let n = self.particles[0].min(m);
            (
                self.energies[0].clone(),
                vec![(0..n).collect(), (n..m).collect()],
            )
        } else {
            let low = self.particles[0].min(self.particles[1]).min(m);
            let high = self.particles[0].max(self.particles[1]).min(m);
            (
                self.natural_occupations.clone(),
                vec![
                    (0..low).collect(),
                    (low..high).collect(),
                    (high..m).collect(),
                ],
            )
        };
        if labels.len() != m {
            return Err(invalid("the model orbitals and their labels disagree"));
        }
        // A cluster no operator resolves is not an error of the SCF, whose energy does not
        // depend on the frame: it is recorded, and a model or active space refuses it. Any
        // other failure is the SCF's.
        match fix_cluster_gauge(
            &mut self.model_orbitals,
            &labels,
            &domains,
            &mut lab_moments(integrals),
        ) {
            Ok(_) => {}
            Err(crate::Error::Unsupported(reason)) => self.frame_error = Some(reason),
            Err(error) => return Err(error),
        }
        Ok(())
    }
    /// An error when the degenerate clusters of the model orbitals have no deterministic frame
    /// (a truncated ansatz built on them would depend on the eigensolver's roundoff).
    pub fn require_frame(&self) -> Result<()> {
        match &self.frame_error {
            Some(reason) => Err(invalid(format!(
                "the model orbitals have a degenerate cluster without a deterministic frame, so a \
                 truncated ansatz on them would depend on the eigensolver's roundoff: {reason}"
            ))),
            None => Ok(()),
        }
    }
}
