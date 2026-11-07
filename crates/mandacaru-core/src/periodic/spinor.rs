//! Two-component Bloch states with a vector magnetization and separable SOC.
//! Spin-major ordering: alpha orbitals followed by beta orbitals.
use super::{invalid, solver, Crystal, KPoint};
use crate::{
    math::{
        fft::Fft3,
        linalg::{davidson, eigen, rmm_diis},
        projection::{hermitian_density_tiles_accumulate, ComplexProjector},
        storage::MatrixSource,
    },
    quantum::C64,
    Buffer, Result,
};
use nalgebra::{DMatrix, Matrix3, Rotation3, UnitQuaternion, SVD};
type Matrix = DMatrix<C64>;

/// Rotate the local density into its eigenframe, differentiate the collinear
/// functional there, and rotate the functional derivative back. In a GGA the
/// derivatives are taken on the eigenvalue fields before this final rotation.
pub fn exchange_correlation(
    c: &Crystal,
    fft: &mut Fft3,
    fields: &[f64],
) -> Result<super::fields::Terms> {
    if c.time_reversal {
        return solver::collinear_xc(c, fft, fields, None, 1);
    }
    let n = c.grid.index.points();
    if fields.len() != 4 * n || fields.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "spinor XC needs charge and three finite magnetization fields",
        ));
    }
    let mut rotated = c.context.allocate::<f64>(2 * n)?;
    let mut norms = c.context.allocate::<f64>(n)?;
    for k in 0..n {
        let norm = (1..4)
            .map(|a| fields[a * n + k].powi(2))
            .sum::<f64>()
            .sqrt();
        norms.as_mut_slice()[k] = norm;
        rotated.as_mut_slice()[k] = 0.5 * (fields[k] + norm);
        rotated.as_mut_slice()[n + k] = 0.5 * (fields[k] - norm);
    }
    let terms = solver::collinear_xc(c, fft, rotated.as_slice(), None, 2)?;
    let mut potential = c.context.allocate::<f64>(4 * n)?;
    for k in 0..n {
        let up = terms.potential.as_slice()[k];
        let down = terms.potential.as_slice()[n + k];
        potential.as_mut_slice()[k] = 0.5 * (up + down);
        if norms.as_slice()[k] > 1e-14 {
            let radial = 0.5 * (up - down) / norms.as_slice()[k];
            for a in 1..4 {
                potential.as_mut_slice()[a * n + k] = radial * fields[a * n + k];
            }
        }
    }
    Ok(super::fields::Terms {
        energy: terms.energy,
        potential,
        kinetic_potential: None,
        // The eigenframe arrays of the rotated spin channels (the stress's spectral-gradient term).
        gradients: terms.gradients,
        flux: terms.flux,
    })
}

/// The Lowdin transformation of one k-point, block diagonal in spin for a spinor.
pub(super) fn lowdin_block(point: &KPoint, m: usize, spinor: bool) -> Matrix {
    if !spinor {
        return point.lowdin.clone();
    }
    Matrix::from_fn(2 * m, 2 * m, |i, j| {
        if i / m == j / m {
            point.lowdin[(i % m, j % m)]
        } else {
            C64::default()
        }
    })
}

pub fn diagonalize(
    c: &Crystal,
    point: &KPoint,
    fields: &[f64],
    moments: &[C64],
) -> Result<(Vec<f64>, Matrix, Matrix)> {
    // The exact exchange K(k) of a hybrid is not part of (fields, moments): refuse, never drop it.
    if c.screened_exchange()?.is_some() {
        return Err(invalid(
            "spinor diagonalize without the exact exchange of a hybrid crystal; \
             use diagonalize_with_exchange",
        ));
    }
    diagonalize_with_exchange(c, point, fields, moments, None, None)
}

pub fn diagonalize_with_exchange(
    c: &Crystal,
    point: &KPoint,
    fields: &[f64],
    moments: &[C64],
    hubbard: Option<&Matrix>,
    exchange: Option<&Matrix>,
) -> Result<(Vec<f64>, Matrix, Matrix)> {
    diagonalize_with_solver(
        c,
        point,
        fields,
        moments,
        hubbard,
        exchange,
        solver::Eigensolver::Direct,
        1e-2,
        0.0,
        None,
    )
}

pub fn diagonalize_with_solver(
    c: &Crystal,
    point: &KPoint,
    fields: &[f64],
    moments: &[C64],
    hubbard: Option<&Matrix>,
    exchange: Option<&Matrix>,
    eigensolver: solver::Eigensolver,
    switch_tolerance: f64,
    current_residual: f64,
    previous_y: Option<&Matrix>,
) -> Result<(Vec<f64>, Matrix, Matrix)> {
    let m = c.m;
    let n = c.grid.index.points();
    if fields.len() != c.field_channels() * n
        || moments.len() != c.channels
        || point.spin_orbit.is_none()
    {
        return Err(invalid(
            "spinor Hamiltonian needs four local potential fields and all moment potentials",
        ));
    }
    let mut operator = Matrix::from_fn(c.p, c.p, |i, j| {
        (0..c.channels)
            .map(|a| moments[a] * c.blocks.as_slice()[(a * c.p + i) * c.p + j])
            .sum()
    });
    if let Some(extra) = hubbard {
        operator += extra;
    }
    let fixed = &point.fixed + &point.projections * operator * point.projections.adjoint();
    let mut h = Matrix::zeros(2 * m, 2 * m);
    for s in 0..2 {
        h.view_mut((s * m, s * m), (m, m)).copy_from(&fixed);
    }
    if let Some(soc) = &point.spin_orbit {
        h += soc;
    }
    let mut projector = ComplexProjector::streaming(&c.context, &point.psi, m, n)?;
    let mut scaled = c.context.allocate::<f64>(n)?;
    // Each projected Pauli component acts on its corresponding spin blocks.
    for a in 0..c.field_channels() {
        for k in 0..n {
            scaled.as_mut_slice()[k] = fields[a * n + k] * c.grid.index.dv();
        }
        let values = projector.project(scaled.as_slice())?;
        for i in 0..m {
            for j in 0..m {
                let value = values[i + j * m];
                match a {
                    0 => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] += value;
                    }
                    1 => {
                        h[(i, m + j)] += value;
                        h[(m + i, j)] += value;
                    }
                    2 => {
                        h[(i, m + j)] -= C64::i() * value;
                        h[(m + i, j)] += C64::i() * value;
                    }
                    _ => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] -= value;
                    }
                }
            }
        }
    }
    drop(projector);
    if let Some(k) = exchange {
        h -= k * C64::new(c.require_screened_exchange()?.1, 0.);
    }
    let lowdin = lowdin_block(point, m, true);
    let h = (&h + h.adjoint()) * C64::new(0.5, 0.);
    let h_ortho = lowdin.adjoint() * h * &lowdin;
    let h_ortho = (&h_ortho + h_ortho.adjoint()) * C64::new(0.5, 0.);

    let (eps, y) = match eigensolver {
        solver::Eigensolver::Direct => eigen(h_ortho)?,
        solver::Eigensolver::Davidson => davidson(&h_ortho, previous_y, 1e-6, 4)?,
        solver::Eigensolver::RmmDiis => {
            if let Some(y_prev) = previous_y {
                rmm_diis(&h_ortho, y_prev, 1e-6, 3)?
            } else {
                eigen(h_ortho)?
            }
        }
        solver::Eigensolver::Mixed => {
            if current_residual < switch_tolerance {
                if let Some(y_prev) = previous_y {
                    rmm_diis(&h_ortho, y_prev, 1e-6, 3)?
                } else {
                    davidson(&h_ortho, previous_y, 1e-6, 4)?
                }
            } else {
                davidson(&h_ortho, previous_y, 1e-6, 4)?
            }
        }
    };
    Ok((eps, &lowdin * &y, y))
}

/// Spatial Hermitian matrices for (n, mx, my, mz); public for the spinor force. This retains transverse
/// coherences; dropping the off-diagonal spin blocks would discard SOC.
pub fn component(d: &Matrix, m: usize, a: usize) -> Matrix {
    Matrix::from_fn(m, m, |i, j| match a {
        0 => d[(i, j)] + d[(m + i, m + j)],
        1 => d[(i, m + j)] + d[(m + i, j)],
        2 => C64::i() * (d[(i, m + j)] - d[(m + i, j)]),
        _ => d[(i, j)] - d[(m + i, m + j)],
    })
}
pub fn density<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<(Buffer<f64>, Buffer<C64>)> {
    if matrices.len() != c.points.len() {
        return Err(invalid(
            "spinor density matrices must match the Bloch basis and k-point mesh",
        ));
    }
    let n = c.grid.index.points();
    let p = c.p;
    let mut rho = c.context.allocate::<f64>(c.field_channels() * n)?;
    let mut projected = Matrix::zeros(p, p);
    for (point, d) in c.points.iter().zip(matrices) {
        let d = d.matrix(&c.context)?;
        if d.shape() != (2 * c.m, 2 * c.m) {
            return Err(invalid("spinor matrix dimensions disagree with the basis"));
        }
        for a in 0..c.field_channels() {
            let d = component(&d, c.m, a);
            if a == 0 {
                projected += point.projections.adjoint()
                    * &d
                    * &point.projections
                    * C64::new(point.weight, 0.);
            }
            hermitian_density_tiles_accumulate(
                &c.context,
                &point.psi,
                &d,
                n,
                point.weight,
                &mut rho.as_mut_slice()[a * n..(a + 1) * n],
            )?;
        }
    }
    let mut q = c.context.allocate::<C64>(c.channels)?;
    for a in 0..c.channels {
        q.as_mut_slice()[a] = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| projected[(i, j)] * c.blocks.as_slice()[(a * p + j) * p + i])
            .sum();
    }
    if c.time_reversal {
        let mut charge = c.context.allocate::<f64>(n)?;
        solver::sym_field(c, rho.as_slice(), charge.as_mut_slice());
        rho = charge;
        let mut moments = c.context.allocate::<C64>(c.channels)?;
        for a in 0..c.channels {
            moments.as_mut_slice()[a] = (0..c.channels)
                .map(|b| c.qsym.as_slice()[a * c.channels + b] * q.as_slice()[b])
                .sum();
        }
        q = moments;
    }
    Ok((rho, q))
}
pub fn fixed_energy(point: &KPoint, d: &Matrix, m: usize) -> f64 {
    (component(d, m, 0) * &point.fixed).trace().re
        + point
            .spin_orbit
            .as_ref()
            .map_or(0., |soc| (d * soc).trace().re)
}
pub fn spin_orbit_energy<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<f64> {
    c.points
        .iter()
        .zip(matrices)
        .map(|(point, d)| {
            let d = d.matrix(&c.context)?;
            Ok(point.weight
                * point
                    .spin_orbit
                    .as_ref()
                    .map_or(0., |soc| (&*d * soc).trace().re))
        })
        .sum()
}
/// The stiffness (Hartree per radian squared) below which the orientation of a magnetic state is
/// left to the SCF. Measured on N next to Ne (`nne_ldafr`, anisotropy tensor `diag(-6.8e-7, -6.9e-7,
/// -5e-9)` Ha): a state `0.026` rad off its optimum, with a smallest stiffness of `6.9e-7` Ha/rad^2,
/// left a transverse density residual of `4.0e-7` (L1), i.e. about `0.02 * stiffness * angle`. A
/// stiffness of `1e-10` leaves under `2e-12` per radian of angle error, three decades below the
/// density tolerance `1e-9`, so a softer mode cannot hold up the stopping rule and its orientation
/// (which would be fixed by noise) is not touched.
pub const ORIENTATION_STIFFNESS_FLOOR: f64 = 1e-10;

/// The anisotropy tensor of the spin-orbit energy of the density matrices `matrices` under a RIGID
/// spin rotation `D -> (1 x U) D (1 x U^+)`: with `D = 1/2 sum_a sigma_a x d_a` (`d_a =
/// component(D, a)`) and `h^SO = 1/2 sum_a sigma_a x h_a`, a rotation `d_a -> sum_b R_ab d_b` makes
/// `E_SO(R) = const + sum_ab R_ab M_ab`, `M_ab = 1/2 sum_k w_k tr(d_b h_a)`.
fn anisotropy<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<Matrix3<f64>> {
    let mut tensor = Matrix3::<f64>::zeros();
    for (point, d) in c.points.iter().zip(matrices) {
        let Some(soc) = &point.spin_orbit else {
            continue;
        };
        let d = d.matrix(&c.context)?;
        let h: Vec<Matrix> = (1..4).map(|a| component(soc, c.m, a)).collect();
        for b in 0..3 {
            let db = component(&d, c.m, b + 1);
            for (a, ha) in h.iter().enumerate() {
                // tr(d_b h_a) for two Hermitian matrices is real: sum_ij d_b[i, j] h_a[j, i].
                let trace: f64 = (0..c.m)
                    .flat_map(|i| (0..c.m).map(move |j| (i, j)))
                    .map(|(i, j)| (db[(i, j)] * ha[(j, i)]).re)
                    .sum();
                tensor[(a, b)] += 0.5 * point.weight * trace;
            }
        }
    }
    Ok(tensor)
}

/// The rigid spin rotation `U` (an SU(2) matrix, `[[u00, u01], [u10, u11]]`) that minimizes the
/// spin-orbit energy of `matrices` at their fixed spatial structure, or `None` when the state needs
/// none: a time-reversal-symmetric crystal, a crystal without a spin-orbit matrix, an orientation
/// already optimal, or an orientation whose stiffness is below
/// [`ORIENTATION_STIFFNESS_FLOOR`].
///
/// Every other term of the energy (kinetic, Hartree, the semilocal and the screened exchange, the
/// spheres, the entropy) is invariant under a global spin rotation, so the optimum over the
/// rotation is the Procrustes problem `min tr(R^T M)`, solved in closed form by the SVD of `-M`.
/// The orientation of a state is a nearly neutral mode of the SCF map: on `nne_ldafr` the transverse
/// density residual stayed at `4.0e-7` (L1) for 1000 iterations of plain linear mixing (weights
/// `0.25` and `0.8`) and under Pulay mixing of 2 to 10 vectors, with the energy converged to
/// `1e-13`, and the Pulay coefficients reached `|alpha|` of 142. Mixing cannot resolve the mode
/// from the noise of the others, so the output state is put at the optimum orientation instead,
/// which turns the mode into one the mixer contracts at once (205, 98 and 131 iterations from
/// `z`, `x` and `y`). At a fixed point the optimum is the identity, so the fixed points of the SCF
/// are those it had; the SCF now relaxes the orientation to the nearest optimum of the SOC
/// energy at fixed spatial structure, and a state with its direction constrained is not available.
/// The coefficients and the density matrices are rotated, not the eigenvalues or the potentials:
/// at convergence the rotation is the identity, a result that stopped at the iteration limit is not
/// internally consistent in this one respect.
pub fn optimal_rotation<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
) -> Result<Option<[[C64; 2]; 2]>> {
    if c.time_reversal || !c.spinor || c.points.iter().all(|p| p.spin_orbit.is_none()) {
        return Ok(None);
    }
    let tensor = anisotropy(c, matrices)?;
    let svd = SVD::new(-tensor, true, true);
    let (Some(u), Some(vt)) = (svd.u, svd.v_t) else {
        return Ok(None);
    };
    let mut singular = svd.singular_values.as_slice().to_vec();
    let sign = (u * vt).determinant().signum();
    // The proper rotation flips the sign of the smallest singular value when `det(U V^T) < 0`.
    let smallest = (0..3)
        .min_by(|&i, &j| singular[i].total_cmp(&singular[j]))
        .unwrap_or(0);
    singular[smallest] *= sign;
    let stiffness = (0..3)
        .map(|i| singular[(i + 1) % 3] + singular[(i + 2) % 3])
        .fold(f64::INFINITY, f64::min);
    if !(stiffness.is_finite() && stiffness > ORIENTATION_STIFFNESS_FLOOR) {
        return Ok(None);
    }
    let flip = Matrix3::from_diagonal(&nalgebra::Vector3::new(1., 1., sign));
    let rotation = Rotation3::from_matrix_unchecked(u * flip * vt);
    let q = UnitQuaternion::from_rotation_matrix(&rotation);
    let (w, x, y, z) = (q.w, q.i, q.j, q.k);
    if x * x + y * y + z * z < 1e-30 {
        return Ok(None);
    }
    let i = C64::i();
    Ok(Some([
        [C64::new(w, 0.) - i * z, -i * x - C64::new(y, 0.)],
        [-i * x + C64::new(y, 0.), C64::new(w, 0.) + i * z],
    ]))
}

/// `(U x 1) A` for a matrix with spin-major rows (`A_s = sum_t U_st A_t`).
pub fn rotate_rows(a: &Matrix, m: usize, u: &[[C64; 2]; 2]) -> Matrix {
    let mut out = Matrix::zeros(a.nrows(), a.ncols());
    for s in 0..2 {
        for t in 0..2 {
            let block = a.view((t * m, 0), (m, a.ncols())).into_owned() * u[s][t];
            let mut target = out.view_mut((s * m, 0), (m, a.ncols()));
            target += block;
        }
    }
    out
}

/// `(1 x U) D (1 x U^+)`: the density matrix of the rigidly rotated state.
pub fn rotate_density(d: &Matrix, m: usize, u: &[[C64; 2]; 2]) -> Matrix {
    let left = rotate_rows(d, m, u);
    rotate_rows(&left.adjoint(), m, u).adjoint()
}

/// Total moment uses the PAW overlap metric, including augmentation charge.
pub fn magnetic_moment<M: MatrixSource>(c: &Crystal, matrices: &[M]) -> Result<f64> {
    if c.time_reversal {
        return Ok(0.);
    }
    let squared = (1..4)
        .map(|a| {
            c.points
                .iter()
                .zip(matrices)
                .map(|(point, d)| {
                    let d = d.matrix(&c.context)?;
                    Ok(point.weight * (component(&d, c.m, a) * &point.overlap).trace().re)
                })
                .sum::<Result<f64>>()
                .map(|v| v.powi(2))
        })
        .sum::<Result<f64>>()?;
    Ok(squared.sqrt())
}
