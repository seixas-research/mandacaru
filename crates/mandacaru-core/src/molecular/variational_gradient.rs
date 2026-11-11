//! The analytic nuclear gradient of a variational state on a PAW-LCAO reference (MG-ZA).
//!
//! With the state's reduced density matrices held fixed, the energy is
//! `E(R) = sum gamma_pq h_pq + 1/2 sum Gamma_pqrs <pq|rs> + E_nn` in the model orbitals; the frozen
//! orbitals enter as an idempotent part of the density matrices over `T = frozen + active`, so
//! one formalism with the bare one-body operator holds. The two-body density is written as
//! Coulomb forms: `Gamma_{(pr),(qs)} = Gamma_pqrs`, symmetrized over `p <-> r`, `q <-> s` and the
//! two pairs (the symmetries of `(pr|qs)` with real orbitals), has the eigenpairs
//! `(lambda_k, x_k)`, and `1/2 sum Gamma <pq|rs> = 1/2 sum_k lambda_k (rho_k | rho_k)` with
//! `rho_k = sum x_k,pr phi_p phi_r`. [`MolecularIntegrals::fixed_density_gradient`] differentiates
//! that at fixed raw-AO coefficients.
use super::invalid;
use crate::Result;
use nalgebra::DMatrix;

/// A one-body density `one` (`P`), an energy-weighted density `weighted` (`W`, the weight of
/// `-tr[W dS]`) and a two-body density as Coulomb forms `(lambda_k, x_k)` over the orbitals
/// `frame` (`m x r`), every matrix in the raw (non-orthogonal) atomic-orbital basis.
pub struct FixedDensities {
    pub one: DMatrix<f64>,
    pub weighted: DMatrix<f64>,
    pub frame: DMatrix<f64>,
    pub two: Vec<(f64, DMatrix<f64>)>,
}

/// The spin-summed density matrices over `T = frozen + active` (the `frozen` doubly occupied
/// orbitals first): `gamma_ii = 2`, `Gamma_ijkl = 4 d_ik d_jl - 2 d_il d_jk` on the frozen block,
/// the Coulomb (`2 gamma`) and exchange (`-gamma`) blocks between a frozen orbital and the active
/// density, and the active `gamma`, `Gamma` (`na x na`, `na^4`, row-major) unchanged.
pub fn extended_rdms(
    frozen: usize,
    active_one: &[f64],
    active_two: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let na = (active_one.len() as f64).sqrt().round() as usize;
    if na * na != active_one.len() || active_two.len() != na.pow(4) {
        return Err(invalid("active density matrices must be na x na and na^4"));
    }
    let t = frozen + na;
    let mut one = vec![0.; t * t];
    let mut two = vec![0.; t.pow(4)];
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * t + q) * t + r) * t + s;
    for i in 0..frozen {
        one[i * t + i] = 2.;
        for j in 0..frozen {
            two[at(i, j, i, j)] += 4.;
            two[at(i, j, j, i)] -= 2.;
        }
    }
    for q in 0..na {
        for s in 0..na {
            let g = active_one[q * na + s];
            one[(frozen + q) * t + frozen + s] = g;
            for i in 0..frozen {
                let (q, s) = (frozen + q, frozen + s);
                two[at(i, q, i, s)] += 2. * g;
                two[at(q, i, s, i)] += 2. * g;
                two[at(i, q, s, i)] -= g;
                two[at(q, i, i, s)] -= g;
            }
            for r in 0..na {
                for u in 0..na {
                    two[at(frozen + q, frozen + s, frozen + r, frozen + u)] =
                        active_two[((q * na + s) * na + r) * na + u];
                }
            }
        }
    }
    Ok((one, two))
}

/// The Coulomb forms of a spin-summed two-body density `Gamma_pqrs` over `t` orbitals
/// (energy `1/2 sum Gamma_pqrs <pq|rs>`): the eigenpairs of the symmetrized pair matrix, those
/// below `1e-12` of the largest magnitude dropped.
pub fn two_body_forms(two: &[f64], t: usize) -> Result<Vec<(f64, DMatrix<f64>)>> {
    if two.len() != t.pow(4) || two.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "the two-body density must be finite with t^4 entries",
        ));
    }
    let at = |p: usize, q: usize, r: usize, s: usize| two[((p * t + q) * t + r) * t + s];
    let pairs = t * t;
    // G[(p r), (q s)] = Gamma_pqrs, symmetrized.
    let raw = DMatrix::from_fn(pairs, pairs, |a, b| {
        let (p, r, q, s) = (a / t, a % t, b / t, b % t);
        0.25 * (at(p, q, r, s) + at(r, q, p, s) + at(p, s, r, q) + at(r, s, p, q))
    });
    let symmetric = (&raw + raw.transpose()) * 0.5;
    // The verified eigensolver: an unverified one can leave the reconstruction off by far more
    // than roundoff, which the force contraction then amplifies.
    let (values, vectors) = crate::math::linalg::real_eigen(symmetric)?;
    let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
    let mut forms = Vec::new();
    for (k, &lambda) in values.iter().enumerate() {
        if lambda.abs() <= 1e-12 * largest {
            continue;
        }
        let v = vectors.column(k);
        forms.push((lambda, DMatrix::from_fn(t, t, |p, r| v[p * t + r])));
    }
    Ok(forms)
}

/// The two-electron integrals of the canonical frame, dense (`m^4`, from the Cholesky factors),
/// with the chemists' accessor `(pq|rs)`.
struct Frame {
    m: usize,
    phys: Vec<f64>,
}
impl Frame {
    fn chem(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let m = self.m;
        self.phys[((p * m + r) * m + q) * m + s]
    }
    /// `J(X)_pq = sum_rs X_rs (pq|rs)` and `K(X)_pq = sum_rs X_rs (pr|sq)` of a symmetric `X`.
    fn coulomb_exchange(&self, x: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        let m = self.m;
        let mut j = DMatrix::<f64>::zeros(m, m);
        let mut k = DMatrix::<f64>::zeros(m, m);
        for p in 0..m {
            for q in 0..m {
                let (mut sj, mut sk) = (0., 0.);
                for r in 0..m {
                    for s in 0..m {
                        let v = x[(r, s)];
                        if v != 0. {
                            sj += v * self.chem(p, q, r, s);
                            sk += v * self.chem(p, r, s, q);
                        }
                    }
                }
                j[(p, q)] = sj;
                k[(p, q)] = sk;
            }
        }
        (j, k)
    }
    /// `(J - K/2)(X)_pq = sum_rs X_rs [(pq|rs) - (pr|sq)/2]` of a symmetric `X`.
    fn fock(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let m = self.m;
        DMatrix::from_fn(m, m, |p, q| {
            let mut sum = 0.;
            for r in 0..m {
                for s in 0..m {
                    let v = x[(r, s)];
                    if v != 0. {
                        sum += v * (self.chem(p, q, r, s) - 0.5 * self.chem(p, r, s, q));
                    }
                }
            }
            sum
        })
    }
}

fn sym(x: &DMatrix<f64>) -> DMatrix<f64> {
    (x + x.transpose()) * 0.5
}

/// `R^(4) Gamma`: a two-body density over `t` orbitals (`t^4`, row-major) carried to the `m`
/// orbitals of the frame whose columns `rt` (`m x t`) express them.
fn lift_two(two_t: Vec<f64>, rt: &DMatrix<f64>) -> Vec<f64> {
    let (m, t) = rt.shape();
    let mut current = two_t;
    let mut shape = [t; 4];
    for axis in 0..4 {
        let old = shape;
        shape[axis] = m;
        let mut next = vec![0.; shape.iter().product()];
        for (flat, out) in next.iter_mut().enumerate() {
            let mut index = [0; 4];
            let mut rest = flat;
            for a in (0..4).rev() {
                index[a] = rest % shape[a];
                rest /= shape[a];
            }
            let target = index[axis];
            let mut sum = 0.;
            for source in 0..t {
                index[axis] = source;
                let o_flat = ((index[0] * old[1] + index[1]) * old[2] + index[2]) * old[3] + index[3];
                sum += rt[(target, source)] * current[o_flat];
            }
            *out = sum;
        }
        current = next;
    }
    current
}

/// `Gamma` (`m^4`, physicists' order) of `sum_pq L_pq sum_k [2 (pq|kk) - (pk|kq)]` over the
/// occupied `k < o`: the two-electron part of a contraction `sum L f` with the Fock matrix.
fn add_fock_two_body(gamma: &mut [f64], m: usize, o: usize, l: &DMatrix<f64>) {
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
    for p in 0..m {
        for q in 0..m {
            let v = l[(p, q)];
            if v == 0. {
                continue;
            }
            for k in 0..o {
                gamma[at(p, k, q, k)] += 4. * v;
                gamma[at(p, k, k, q)] -= 2. * v;
            }
        }
    }
}

/// `4 sum_{r, k < o} Y_rk M_rk` as a matrix of the coefficients of `Y`: the response of a Fock
/// contraction to the change of the Hartree-Fock density, `M = (J - K/2)(L)`.
fn add_density_response(g: &mut DMatrix<f64>, o: usize, m_fock: &DMatrix<f64>, scale: f64) {
    for r in 0..g.nrows() {
        for k in 0..o {
            g[(r, k)] += 4. * scale * m_fock[(r, k)];
        }
    }
}

/// The density matrices, frames and labels the analytic gradient consumes.
pub struct VariationalDensities {
    /// The densities of the explicit derivative at fixed raw-AO coefficients.
    pub densities: FixedDensities,
    /// `sum gamma h + 1/2 sum Gamma <pq|rs> + E_nn` of the state, rebuilt from the integrals.
    pub energy: f64,
    /// The largest `|f_ai|` of the reference (its stationarity) and the CPHF residual.
    pub stationarity: f64,
    pub cphf_residual: f64,
}

/// The relaxed densities of a variational state on a closed-shell Hartree-Fock reference: the
/// state's `gamma`, `Gamma` (active, spin-summed) in the model orbitals, the response of the
/// natural-orbital rotation of an MP2 active space (eigenvector adjoint, zero inside an exactly
/// degenerate cluster: the parallel-transport gauge of the displaced frames), the MP2
/// amplitudes' Lagrangian, the canonical occupied-occupied and virtual-virtual rotations and
/// one CPHF solve for the occupied-virtual ones, all folded into fixed raw-AO densities.
pub fn relaxed_densities(
    integrals: &super::MolecularIntegrals,
    reference: &super::MeanField,
    model: &super::transform::Model,
    active_one: &[f64],
    active_two: &[f64],
) -> Result<VariationalDensities> {
    use super::response::DEGENERACY_TOLERANCE;
    use crate::quantum::C64;
    let unsupported = |what: &str| {
        crate::Error::Unsupported(format!(
            "the analytic variational force does not support {what} yet"
        ))
    };
    if reference.method == "uhf" && reference.functional.is_none() {
        return unrestricted(integrals, reference, model, active_one, active_two);
    }
    if reference.method != "rhf" || reference.functional.is_some() {
        return Err(unsupported(
            "a reference other than Hartree-Fock (a Kohn-Sham or spinor reference)",
        ));
    }
    reference.require_determinant("the analytic variational force")?;
    if !reference.converged || model.spinor || reference.particles[0] != reference.particles[1] {
        return Err(invalid(
            "the analytic variational force needs a converged closed-shell collinear reference",
        ));
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let o = reference.particles[0];
    let c = reference.model_orbitals.map(|v| v.re);
    if reference.model_orbitals.iter().any(|v| v.im.abs() > 1e-10) || c.shape() != (m, m) {
        return Err(invalid(
            "the analytic variational force needs real reference orbitals",
        ));
    }
    let eps = &reference.energies[0];
    let rotation = model.orbital_rotation.as_ref();
    let r_full = rotation.map_or_else(|| DMatrix::identity(m, m), |r| r.matrix.map(|v| v.re));
    let t_indices: Vec<usize> = model
        .frozen_indices
        .iter()
        .chain(&model.orbital_indices)
        .copied()
        .collect();
    let t = t_indices.len();
    if model.frozen_indices.iter().any(|&i| i >= o) || t_indices.iter().any(|&i| i >= m) {
        return Err(invalid(
            "frozen orbitals must be occupied and every model orbital in the basis",
        ));
    }
    let (one_t, two_t) = extended_rdms(model.frozen_indices.len(), active_one, active_two)?;
    let _memory = ctx.reserve(crate::product(&[m, m, m, m, 8, 4])?)?;
    // The state's densities in the canonical frame A: gamma^A = R_T gamma R_T^T, Gamma^A = R_T^(4) Gamma.
    let rt = DMatrix::from_fn(m, t, |i, j| r_full[(i, t_indices[j])]);
    let gamma_t = DMatrix::from_row_slice(t, t, &one_t);
    let mut gamma_a = &rt * gamma_t * rt.transpose();
    let mut two_a = lift_two(two_t, &rt);
    let eri = integrals.eri()?;
    let cc = c.map(|v| C64::new(v, 0.));
    let phys = super::transform::four(ctx, &eri, [&cc, &cc, &cc, &cc])?;
    let frame = Frame {
        m,
        phys: phys.as_slice().iter().map(|v| v.re).collect(),
    };
    drop(phys);
    let h = c.transpose() * DMatrix::from_row_slice(m, m, integrals.one.as_slice()) * &c;
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
    // The generalized Fock F_rp = sum h_rq gamma_qp + sum <rq|st> Gamma_pqst and the energy.
    let mut fock = &h * &gamma_a;
    let mut energy = integrals.nuclear + (h.component_mul(&gamma_a)).sum();
    for p in 0..m {
        for q in 0..m {
            for s in 0..m {
                for u in 0..m {
                    let density = two_a[at(p, q, s, u)];
                    if density == 0. {
                        continue;
                    }
                    energy += 0.5 * density * frame.phys[at(p, q, s, u)];
                    for r in 0..m {
                        fock[(r, p)] += frame.phys[at(r, q, s, u)] * density;
                    }
                }
            }
        }
    }
    let fock_hf = DMatrix::from_fn(m, m, |p, q| {
        h[(p, q)]
            + (0..o)
                .map(|k| 2. * frame.chem(p, q, k, k) - frame.chem(p, k, k, q))
                .sum::<f64>()
    });
    let stationarity = (0..o)
        .flat_map(|i| (o..m).map(move |a| (i, a)))
        .fold(0f64, |w, (i, a)| w.max(fock_hf[(a, i)].abs()));
    if stationarity > 1e-6 {
        return Err(invalid(format!(
            "the reference is not stationary: its occupied-virtual Fock elements reach {stationarity:.2e}, above the \
             1e-6 the CPHF response assumes"
        )));
    }
    let mut g = &fock * 2.;
    // Lambda: the coefficients of the fixed-orbital Fock matrix f^(x); sigma: of S^(x) in frame A.
    let mut lambda = DMatrix::<f64>::zeros(m, m);
    let mut sigma = DMatrix::<f64>::zeros(m, m);
    if let Some(rotation) = rotation {
        let n = &rotation.occupations;
        let v = m - o;
        // The natural-orbital adjoint: dE = <D-bar, dD> per block (canonical coordinates).
        let q_phi = r_full.transpose() * (&fock - fock.transpose()) * &r_full;
        let keep: Vec<usize> = (0..o)
            .filter(|i| !model.frozen_indices.contains(i))
            .collect();
        let virtuals: Vec<usize> = (o..m).collect();
        let adjoint = |block: &[usize]| {
            let k = block.len();
            let rb = DMatrix::from_fn(k, k, |i, j| r_full[(block[i], block[j])]);
            let y = DMatrix::from_fn(k, k, |a, b| {
                let (na, nb) = (n[block[a]], n[block[b]]);
                if a == b || (nb - na).abs() < DEGENERACY_TOLERANCE {
                    0.
                } else {
                    q_phi[(block[a], block[b])] / (nb - na)
                }
            });
            &rb * y * rb.transpose()
        };
        let bar_vv = adjoint(&virtuals);
        let bar_k = adjoint(&keep);
        let mut bar_oo = DMatrix::<f64>::zeros(o, o);
        for (i, &a) in keep.iter().enumerate() {
            for (j, &b) in keep.iter().enumerate() {
                bar_oo[(a, b)] = bar_k[(i, j)];
            }
        }
        // MP2 amplitudes t_ijab = (ia|jb) / D_ijab and the adjoint of D(t).
        let idx = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
        let delta =
            |i: usize, j: usize, a: usize, b: usize| eps[i] + eps[j] - eps[o + a] - eps[o + b];
        let mut amp = vec![0.; o * o * v * v];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        amp[idx(i, j, a, b)] = frame.chem(i, o + a, j, o + b) / delta(i, j, a, b);
                    }
                }
            }
        }
        let tilde = |x: &[f64], i, j, a, b| 2. * x[idx(i, j, a, b)] - x[idx(i, j, b, a)];
        // The selector's own correlation energy: canonical MP2 only (a DLPNO selection has other
        // amplitudes, whose response is not implemented).
        let mut correlation = 0.;
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        correlation += tilde(&amp, i, j, a, b) * frame.chem(i, o + a, j, o + b);
                    }
                }
            }
        }
        // The eigenvector adjoint needs the natural orbitals to diagonalize the MP2 densities.
        let mut dvv = DMatrix::<f64>::zeros(v, v);
        let mut doo = DMatrix::<f64>::zeros(o, o);
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let x = tilde(&amp, i, j, a, b);
                        for c in 0..v {
                            dvv[(a, c)] += x * amp[idx(i, j, c, b)]
                                + tilde(&amp, i, j, c, b) * amp[idx(i, j, a, b)];
                        }
                        for k in 0..o {
                            doo[(i, k)] -= x * amp[idx(k, j, a, b)]
                                + tilde(&amp, k, j, a, b) * amp[idx(i, j, a, b)];
                        }
                    }
                }
            }
        }
        let off_diagonal = |density: &DMatrix<f64>, block: &[usize], offset: usize| {
            let k = block.len();
            let rb = DMatrix::from_fn(k, k, |i, j| r_full[(block[i], block[j])]);
            let local =
                DMatrix::from_fn(k, k, |i, j| density[(block[i] - offset, block[j] - offset)]);
            let rotated = rb.transpose() * local * rb;
            (0..k)
                .flat_map(|i| (0..k).map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .fold(0f64, |w, (i, j)| w.max(rotated[(i, j)].abs()))
        };
        let residual = off_diagonal(&dvv, &virtuals, o).max(off_diagonal(&doo, &keep, 0));
        if residual > 1e-11 {
            return Err(invalid(format!(
                "the natural orbitals of the active space do not diagonalize its MP2 densities (off-diagonal \
                 {residual:.2e}): the eigenvector response needs exact eigenvectors"
            )));
        }
        if (correlation - rotation.correlation_energy).abs()
            > 1e-9 * rotation.correlation_energy.abs().max(1e-3)
        {
            return Err(unsupported(
                "the natural-orbital response of this selection: it is implemented for canonical MP2, and these \
                 natural orbitals do not come from it (a dlpno-mp2 selection)",
            ));
        }
        // L = <bar_oo, D_oo> + <bar_vv, D_vv> = B(t, t); dL/dt = d1 B + d2 B.
        let mut vt = vec![0.; amp.len()];
        let mut wt = vec![0.; amp.len()];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let (mut x, mut y) = (0., 0.);
                        for k in 0..o {
                            x -= 2. * bar_oo[(i, k)] * amp[idx(k, j, a, b)];
                            y -= 2. * bar_oo[(i, k)] * tilde(&amp, k, j, a, b);
                        }
                        for e in 0..v {
                            x += 2. * bar_vv[(a, e)] * amp[idx(i, j, e, b)];
                            y += 2. * bar_vv[(a, e)] * tilde(&amp, i, j, e, b);
                        }
                        vt[idx(i, j, a, b)] = x;
                        wt[idx(i, j, a, b)] = y;
                    }
                }
            }
        }
        let mut bar = vec![0.; amp.len()];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let dl =
                            2. * vt[idx(i, j, a, b)] - vt[idx(i, j, b, a)] + wt[idx(i, j, a, b)];
                        bar[idx(i, j, a, b)] = dl / delta(i, j, a, b);
                    }
                }
            }
        }
        let bar: Vec<f64> = (0..amp.len())
            .map(|f| {
                let b = f % v;
                let a = (f / v) % v;
                let j = (f / (v * v)) % o;
                let i = f / (v * v * o);
                0.5 * (bar[idx(i, j, a, b)] + bar[idx(j, i, b, a)])
            })
            .collect();
        // sum bar (ia|jb)^(x): Gamma_{i j a b} += 2 bar; its orbital response; the energies.
        let mut coefficient = vec![0.; m];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let value = bar[idx(i, j, a, b)];
                        two_a[at(i, j, o + a, o + b)] += 2. * value;
                        coefficient[i] -= 2. * value * amp[idx(i, j, a, b)];
                        coefficient[o + a] += 2. * value * amp[idx(i, j, a, b)];
                        for r in 0..m {
                            g[(r, i)] += 2. * value * frame.chem(r, o + a, j, o + b);
                            g[(r, o + a)] += 2. * value * frame.chem(i, r, j, o + b);
                        }
                    }
                }
            }
        }
        let diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&coefficient));
        for p in 0..m {
            g[(p, p)] += 2. * eps[p] * coefficient[p];
        }
        add_density_response(&mut g, o, &frame.fock(&diagonal), 1.);
        lambda += diagonal;
    }
    // The canonical occupied-occupied and virtual-virtual rotations.
    let antisymmetric = &g - g.transpose();
    let mut omega = DMatrix::<f64>::zeros(m, m);
    for p in 0..m {
        for q in 0..m {
            if p != q && (p < o) == (q < o) && (eps[p] - eps[q]).abs() >= DEGENERACY_TOLERANCE {
                omega[(p, q)] = if p > q {
                    antisymmetric[(p, q)] / (eps[p] - eps[q])
                } else {
                    antisymmetric[(q, p)] / (eps[q] - eps[p])
                };
            }
        }
    }
    let omega_fock = frame.fock(&omega);
    add_density_response(&mut g, o, &omega_fock, -0.5);
    lambda -= &omega * 0.5;
    sigma += DMatrix::from_fn(m, m, |p, q| 0.25 * omega[(p, q)] * (eps[p] + eps[q]));
    // CPHF: (eps_a - eps_i + A) z = G_ai - G_ia.
    let v = m - o;
    let nov = o * v;
    let mut hessian = DMatrix::<f64>::zeros(nov, nov);
    for b in 0..v {
        for j in 0..o {
            let mut unit = DMatrix::<f64>::zeros(m, m);
            unit[(o + b, j)] = 2.;
            unit[(j, o + b)] = 2.;
            let response = frame.fock(&unit);
            for a in 0..v {
                for i in 0..o {
                    hessian[(a * o + i, b * o + j)] = response[(o + a, i)];
                }
            }
            hessian[(b * o + j, b * o + j)] += eps[o + b] - eps[j];
        }
    }
    let rhs = nalgebra::DVector::from_fn(nov, |ai, _| {
        let (a, i) = (ai / o, ai % o);
        g[(o + a, i)] - g[(i, o + a)]
    });
    // No occupied-virtual pair (every orbital occupied, or none): nothing to solve.
    let z = if nov == 0 {
        nalgebra::DVector::zeros(0)
    } else {
        sym(&hessian)
            .lu()
            .solve(&rhs)
            .ok_or_else(|| invalid("the CPHF orbital Hessian is singular"))?
    };
    let cphf_residual = (sym(&hessian) * &z - &rhs).norm();
    let mut zs = DMatrix::<f64>::zeros(m, m);
    for a in 0..v {
        for i in 0..o {
            zs[(o + a, i)] = z[a * o + i];
            zs[(i, o + a)] = z[a * o + i];
        }
    }
    lambda -= &zs * 0.5;
    sigma += DMatrix::from_fn(m, m, |p, q| 0.25 * zs[(p, q)] * (eps[p] + eps[q]));
    let z_fock = frame.fock(&zs);
    for r in 0..m {
        for k in 0..o {
            sigma[(r, k)] += z_fock[(r, k)];
        }
    }
    sigma -= &g * 0.5;
    // The explicit densities at fixed raw-AO coefficients A = lowdin C.
    gamma_a += &lambda;
    add_fock_two_body(&mut two_a, m, o, &lambda);
    let a = &integrals.lowdin * &c;
    let densities = FixedDensities {
        one: &a * sym(&gamma_a) * a.transpose(),
        weighted: -(&a * sym(&sigma) * a.transpose()),
        frame: a,
        two: two_body_forms(&two_a, m)?,
    };
    Ok(VariationalDensities {
        densities,
        energy,
        stationarity,
        cphf_residual,
    })
}

/// `sum over clusters` of the Lowdin connection: inside an exactly degenerate cluster of a frame
/// with orthonormal-frame coefficients `v`, the displaced frame is aligned in those coefficients
/// (the parallel-transport gauge of the displaced frames), so the cluster's physical rotation is
/// the antisymmetric part of `S^{1/2} dS^{-1/2}`, a function of the overlap derivative alone.
/// `gradient` holds `G_pq - G_qp` on the cluster pairs (zero elsewhere); the result is the matrix
/// `M` (raw atomic-orbital basis) of the contribution `sum M_mn dS_mn`.
fn lowdin_gauge(
    integrals: &super::MolecularIntegrals,
    v: &DMatrix<f64>,
    gradient: &DMatrix<f64>,
) -> Result<DMatrix<f64>> {
    let m = integrals.m();
    let s = DMatrix::from_row_slice(m, m, integrals.overlap.as_slice());
    let (values, u) = crate::math::linalg::real_eigen((&s + s.transpose()) * 0.5)?;
    // Sum_pq G_pq (V^T omega V)_pq = sum_ij B_ij omega_ij, B = V G V^T, in the eigenbasis of S.
    let b = u.transpose() * (v * gradient * v.transpose()) * &u;
    let root: Vec<f64> = values.iter().map(|x| x.sqrt()).collect();
    let mut prime = DMatrix::<f64>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            if i != j {
                prime[(i, j)] = 0.25 * b[(i, j)] * (root[j] - root[i])
                    / (root[i] * root[j] * (root[i] + root[j]));
            }
        }
    }
    Ok(&u * sym(&prime) * u.transpose())
}

/// The relaxed densities of a variational state on an unrestricted Hartree-Fock reference. The
/// model orbitals are the natural orbitals of the total density `P = P_a + P_b`, which depend on
/// the spin orbitals through their occupied spaces only: the response is their eigenvector
/// response (`dN = 0` off the diagonal of `N = Phi^T S P S Phi`), the Lowdin gauge inside exactly
/// degenerate clusters, and one CPHF solve of both spins' occupied-virtual rotations.
fn unrestricted(
    integrals: &super::MolecularIntegrals,
    reference: &super::MeanField,
    model: &super::transform::Model,
    active_one: &[f64],
    active_two: &[f64],
) -> Result<VariationalDensities> {
    use super::response::DEGENERACY_TOLERANCE;
    use crate::quantum::C64;
    reference.require_determinant("the analytic variational force")?;
    if !reference.converged
        || model.spinor
        || reference.coefficients.len() != 2
        || reference.energies.len() != 2
    {
        return Err(invalid(
            "the analytic variational force needs a converged collinear unrestricted reference",
        ));
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let real = |x: &DMatrix<C64>| -> Result<DMatrix<f64>> {
        if x.iter().any(|v| v.im.abs() > 1e-10) || x.shape() != (m, m) {
            return Err(invalid(
                "the analytic variational force needs real reference orbitals",
            ));
        }
        Ok(x.map(|v| v.re))
    };
    let c = [
        real(&reference.coefficients[0])?,
        real(&reference.coefficients[1])?,
    ];
    let eps = [&reference.energies[0], &reference.energies[1]];
    let nocc = reference.particles;
    let v = real(&reference.model_orbitals)?;
    let n = &reference.natural_occupations;
    if n.len() != m {
        return Err(invalid(
            "the unrestricted reference has no natural occupations",
        ));
    }
    let t_indices: Vec<usize> = model
        .frozen_indices
        .iter()
        .chain(&model.orbital_indices)
        .copied()
        .collect();
    let t = t_indices.len();
    let (one_t, two_t) = extended_rdms(model.frozen_indices.len(), active_one, active_two)?;
    let _memory = ctx.reserve(crate::product(&[m, m, m, m, 8, 8])?)?;
    let at = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
    // The state's densities in the natural-orbital frame V: the model orbitals are `V R`, `R` the
    // rotation of an MP2 active space (the identity for an energy selection).
    let rotation = model.orbital_rotation.as_ref();
    let r_full = rotation.map_or_else(|| DMatrix::identity(m, m), |r| r.matrix.map(|x| x.re));
    if t_indices.iter().any(|&i| i >= m) {
        return Err(invalid("every model orbital must be in the basis"));
    }
    let rt = DMatrix::from_fn(m, t, |i, j| r_full[(i, t_indices[j])]);
    let gamma = &rt * DMatrix::from_row_slice(t, t, &one_t) * rt.transpose();
    let two = lift_two(two_t, &rt);
    let eri = integrals.eri()?;
    let vc = v.map(|x| C64::new(x, 0.));
    let phys = super::transform::four(ctx, &eri, [&vc, &vc, &vc, &vc])?;
    let frame = Frame {
        m,
        phys: phys.as_slice().iter().map(|x| x.re).collect(),
    };
    drop(phys);
    let one_ortho = DMatrix::from_row_slice(m, m, integrals.one.as_slice());
    let h = v.transpose() * &one_ortho * &v;
    // The generalized Fock F_rp = sum h_rq gamma_qp + sum <rq|su> Gamma_pqsu and the energy.
    let generalized = |gamma: &DMatrix<f64>, two: &[f64]| {
        let mut fock = &h * gamma;
        let mut energy = integrals.nuclear + (h.component_mul(gamma)).sum();
        for p in 0..m {
            for q in 0..m {
                for s in 0..m {
                    for u in 0..m {
                        let density = two[at(p, q, s, u)];
                        if density == 0. {
                            continue;
                        }
                        energy += 0.5 * density * frame.phys[at(p, q, s, u)];
                        for r in 0..m {
                            fock[(r, p)] += frame.phys[at(r, q, s, u)] * density;
                        }
                    }
                }
            }
        }
        (fock, energy)
    };
    let (fock, energy) = generalized(&gamma, &two);
    // An MP2 active space: R is a function of the V-frame integrals alone (the open-shell MP2 of
    // `correlation::open_shell_natural_orbitals` on the determinant of the lowest n_a, n_b V
    // orbitals), so its response is a pair of effective densities in V added to the state's,
    // after which V's own response below is unchanged.
    let (gamma, mut two, fock) = match rotation {
        None => (gamma, two, fock),
        Some(rotation) => {
            let high = nocc[0].max(nocc[1]);
            let occupations = &rotation.occupations;
            if occupations.len() != m || high >= m {
                return Err(invalid("the MP2 rotation does not match the unrestricted reference"));
            }
            // The adjoint of the virtual natural orbitals (eigenvectors of the MP2 density's
            // block from `high` up), zero inside an exactly degenerate cluster.
            let q_phi = r_full.transpose() * (&fock - fock.transpose()) * &r_full;
            let k = m - high;
            let rb = DMatrix::from_fn(k, k, |i, j| r_full[(high + i, high + j)]);
            let y = DMatrix::from_fn(k, k, |a, b| {
                let (na, nb) = (occupations[high + a], occupations[high + b]);
                if a == b || (nb - na).abs() < DEGENERACY_TOLERANCE {
                    0.
                } else {
                    q_phi[(high + a, high + b)] / (nb - na)
                }
            });
            let block = &rb * y * rb.transpose();
            let mut d_bar = DMatrix::<f64>::zeros(m, m);
            d_bar.view_mut((high, high), (k, k)).copy_from(&block);
            // The selector also rotates the active doubly occupied orbitals (below min(n_a, n_b),
            // frozen ones excepted) by the eigenvectors of the MP2 density's occupied block: the
            // same adjoint there.
            let keep: Vec<usize> = (0..nocc[0].min(nocc[1]))
                .filter(|i| !model.frozen_indices.contains(i))
                .collect();
            let kk = keep.len();
            let rk = DMatrix::from_fn(kk, kk, |i, j| r_full[(keep[i], keep[j])]);
            let yk = DMatrix::from_fn(kk, kk, |a, b| {
                let (na, nb) = (occupations[keep[a]], occupations[keep[b]]);
                if a == b || (nb - na).abs() < DEGENERACY_TOLERANCE {
                    0.
                } else {
                    q_phi[(keep[a], keep[b])] / (nb - na)
                }
            });
            let occupied_block = &rk * yk * rk.transpose();
            for (i, &p) in keep.iter().enumerate() {
                for (j, &q) in keep.iter().enumerate() {
                    d_bar[(p, q)] = occupied_block[(i, j)];
                }
            }
            let (p_bar, two_bar) = open_shell_mp2_lagrangian(
                &frame,
                &h,
                nocc,
                &d_bar,
                &r_full,
                &keep,
                rotation.correlation_energy,
            )?;
            let gamma = gamma + p_bar;
            let two: Vec<f64> = two.iter().zip(&two_bar).map(|(a, b)| a + b).collect();
            let (fock, _) = generalized(&gamma, &two);
            (gamma, two, fock)
        }
    };
    // Everything in the natural-orbital frame: T_s = V^T C_s maps spin-frame matrices into it.
    let tmap = [v.transpose() * &c[0], v.transpose() * &c[1]];
    let occupied = |s: usize| {
        let mut e = DMatrix::<f64>::zeros(m, m);
        for k in 0..nocc[s] {
            e[(k, k)] = 1.;
        }
        &tmap[s] * e * tmap[s].transpose()
    };
    let p_spin = [occupied(0), occupied(1)];
    let p_total = &p_spin[0] + &p_spin[1];
    let (j_total, _) = frame.coulomb_exchange(&p_total);
    let spin_fock: Vec<DMatrix<f64>> = (0..2)
        .map(|s| {
            let (_, k) = frame.coulomb_exchange(&p_spin[s]);
            tmap[s].transpose() * (&h + &j_total - k) * &tmap[s]
        })
        .collect();
    let stationarity = (0..2)
        .flat_map(|s| (0..nocc[s]).flat_map(move |i| (nocc[s]..m).map(move |a| (s, i, a))))
        .fold(0f64, |w, (s, i, a)| w.max(spin_fock[s][(a, i)].abs()));
    if stationarity > 1e-6 {
        return Err(invalid(format!(
            "the reference is not stationary: its occupied-virtual Fock elements reach {stationarity:.2e}, above the \
             1e-6 the CPHF response assumes"
        )));
    }
    let g_phi = &fock * 2.;
    let antisymmetric = &g_phi - g_phi.transpose();
    // Off a degenerate cluster: K_pq = -(S_pq (n_p + n_q) / 2 + dP_pq) / (n_p - n_q).
    let mut omega = DMatrix::<f64>::zeros(m, m);
    let mut cluster = DMatrix::<f64>::zeros(m, m);
    for p in 0..m {
        for q in 0..p {
            if (n[p] - n[q]).abs() >= DEGENERACY_TOLERANCE {
                let w = antisymmetric[(p, q)] / (n[p] - n[q]);
                omega[(p, q)] = w;
                omega[(q, p)] = w;
            } else {
                cluster[(p, q)] = antisymmetric[(p, q)];
                cluster[(q, p)] = -antisymmetric[(p, q)];
            }
        }
    }
    // Sigma: coefficients of S^(x) in the natural-orbital frame (phi) and each spin frame.
    let mut sigma_phi =
        -&g_phi * 0.5 - DMatrix::from_fn(m, m, |p, q| 0.25 * omega[(p, q)] * (n[p] + n[q]));
    // dP_phi = T_s Delta_s T_s^T: G_s[r, k] += -(T_s^T omega T_s)[r, k] for the occupied k.
    let mut g_spin: Vec<DMatrix<f64>> = (0..2)
        .map(|s| {
            let mapped = tmap[s].transpose() * &omega * &tmap[s];
            DMatrix::from_fn(m, m, |r, k| if k < nocc[s] { -mapped[(r, k)] } else { 0. })
        })
        .collect();
    let mut sigma_spin: Vec<DMatrix<f64>> = g_spin.iter().map(|g| -g * 0.5).collect();
    let mut lambda_spin = [DMatrix::<f64>::zeros(m, m), DMatrix::<f64>::zeros(m, m)];
    // CPHF over both spins' occupied-virtual rotations.
    let pairs: Vec<(usize, usize, usize)> = (0..2)
        .flat_map(|s| (nocc[s]..m).flat_map(move |a| (0..nocc[s]).map(move |i| (s, a, i))))
        .collect();
    let size = pairs.len();
    let mut hessian = DMatrix::<f64>::zeros(size, size);
    for (column, &(tau, b, j)) in pairs.iter().enumerate() {
        let mut delta = DMatrix::<f64>::zeros(m, m);
        delta[(b, j)] = 1.;
        delta[(j, b)] = 1.;
        let d_phi = &tmap[tau] * delta * tmap[tau].transpose();
        let (jd, kd) = frame.coulomb_exchange(&d_phi);
        for s in 0..2 {
            let response = if s == tau { &jd - &kd } else { jd.clone() };
            let local = tmap[s].transpose() * response * &tmap[s];
            for (row, &(sr, a, i)) in pairs.iter().enumerate() {
                if sr == s {
                    hessian[(row, column)] += local[(a, i)];
                }
            }
        }
        hessian[(column, column)] += eps[tau][b] - eps[tau][j];
    }
    let rhs = nalgebra::DVector::from_fn(size, |row, _| {
        let (s, a, i) = pairs[row];
        g_spin[s][(a, i)] - g_spin[s][(i, a)]
    });
    let symmetric = sym(&hessian);
    // No occupied-virtual pair in either spin (a filled or empty channel): nothing to solve.
    let z = if size == 0 {
        nalgebra::DVector::zeros(0)
    } else {
        symmetric
            .clone()
            .lu()
            .solve(&rhs)
            .ok_or_else(|| invalid("the CPHF orbital Hessian is singular"))?
    };
    let cphf_residual = (&symmetric * &z - &rhs).norm();
    let mut z_spin = [DMatrix::<f64>::zeros(m, m), DMatrix::<f64>::zeros(m, m)];
    for (row, &(s, a, i)) in pairs.iter().enumerate() {
        z_spin[s][(a, i)] = z[row];
        z_spin[s][(i, a)] = z[row];
    }
    // -z . R: the explicit Fock derivative, the overlap terms and the Fock response's overlap part.
    let z_phi = [
        &tmap[0] * &z_spin[0] * tmap[0].transpose(),
        &tmap[1] * &z_spin[1] * tmap[1].transpose(),
    ];
    let (jz, _) = frame.coulomb_exchange(&(&z_phi[0] + &z_phi[1]));
    for s in 0..2 {
        lambda_spin[s] -= &z_spin[s] * 0.5;
        sigma_spin[s] += DMatrix::from_fn(m, m, |p, q| {
            0.25 * z_spin[s][(p, q)] * (eps[s][p] + eps[s][q])
        });
        let (_, kz) = frame.coulomb_exchange(&z_phi[s]);
        let response = tmap[s].transpose() * (&jz - kz) * &tmap[s];
        for r in 0..m {
            for k in 0..nocc[s] {
                sigma_spin[s][(r, k)] += 0.5 * response[(r, k)];
            }
        }
    }
    let _ = &mut g_spin;
    // The explicit densities in the natural-orbital frame.
    let mut gamma_phi = gamma;
    for s in 0..2 {
        let mapped = &tmap[s] * &lambda_spin[s] * tmap[s].transpose();
        gamma_phi += &mapped;
        // sum L f_s^(x) with f_s = h + J(P) - K(P_s): Gamma_{p r q s} += 2 L_pq P_rs, Gamma_{p s r q} -= 2 L_pq P_s,rs.
        for p in 0..m {
            for q in 0..m {
                let l = mapped[(p, q)];
                if l == 0. {
                    continue;
                }
                for r in 0..m {
                    for u in 0..m {
                        two[at(p, r, q, u)] += 2. * l * p_total[(r, u)];
                        two[at(p, u, r, q)] -= 2. * l * p_spin[s][(r, u)];
                    }
                }
            }
        }
    }
    let phi = &integrals.lowdin * &v;
    let mut sigma_ao = &phi * sym(&sigma_phi) * phi.transpose();
    for s in 0..2 {
        let a = &integrals.lowdin * &c[s];
        sigma_ao += &a * sym(&sigma_spin[s]) * a.transpose();
    }
    sigma_ao += lowdin_gauge(integrals, &v, &cluster)?;
    let _ = &mut sigma_phi;
    let densities = FixedDensities {
        one: &phi * sym(&gamma_phi) * phi.transpose(),
        weighted: -sigma_ao,
        frame: phi,
        two: two_body_forms(&two, m)?,
    };
    Ok(VariationalDensities {
        densities,
        energy,
        stationarity,
        cphf_residual,
    })
}

/// `<ij|ab> = sum A_pi B_qj C_ra D_sb <pq|rs>` of the frame's integrals over four sets of
/// orbitals given as columns (`m x k`), row-major `(k_A, k_B, k_C, k_D)`.
fn block4(frame: &Frame, c: [&DMatrix<f64>; 4]) -> Vec<f64> {
    let m = frame.m;
    let k: Vec<usize> = c.iter().map(|x| x.ncols()).collect();
    // Contract the last index first: (p, q, r, s) -> (p, q, r, b) -> ... -> (i, j, a, b).
    let mut current = frame.phys.clone();
    let mut shape = [m; 4];
    for axis in (0..4).rev() {
        let old = shape;
        shape[axis] = k[axis];
        let mut next = vec![0.; shape.iter().product()];
        for (flat, out) in next.iter_mut().enumerate() {
            let mut index = [0; 4];
            let mut rest = flat;
            for a in (0..4).rev() {
                index[a] = rest % shape[a];
                rest /= shape[a];
            }
            let target = index[axis];
            let mut sum = 0.;
            for source in 0..m {
                index[axis] = source;
                let o_flat = ((index[0] * old[1] + index[1]) * old[2] + index[2]) * old[3] + index[3];
                sum += c[axis][(source, target)] * current[o_flat];
            }
            *out = sum;
        }
        current = next;
    }
    current
}

/// One spin channel of the open-shell MP2 reference in the V frame: semicanonical occupied and
/// virtual orbitals (columns over V), their energies and the singles.
struct OpenShellSpin {
    co: DMatrix<f64>,
    cv: DMatrix<f64>,
    eo: Vec<f64>,
    ev: Vec<f64>,
    t1: DMatrix<f64>,
}

/// The response of an open-shell MP2 active space (`correlation::open_shell_natural_orbitals`) to
/// the V-frame integrals: given `d_bar`, the adjoint of the spin-summed MP2 density on its
/// virtual block and on the active doubly occupied orbitals `keep` (the selector diagonalizes both;
/// `dE = <d_bar, d gamma_MP2>`), the effective one-body density `P` and two-body
/// density `Gamma` (V frame, energy `1/2 sum Gamma <pq|rs>`) with
/// `<d_bar, d gamma_MP2> = <P, dh> + 1/2 <Gamma, d<pq|rs>>`.
///
/// The density is an orbital-invariant function of each spin's Fock blocks `f^s = h + J[D_a +
/// D_b] - K[D_s]` (`D_s` the lowest `n_s` V orbitals) and of the integrals: the amplitude
/// equations `sum_b t_ib f_ba - sum_j f_ij t_ja + f_ia = 0` (singles), and the same with two
/// virtual and two occupied indices for the same-spin (antisymmetrized) and opposite-spin pairs,
/// are solved in the semicanonical blocks, where they are diagonal; the multipliers
/// `tbar = (dL/dt) / D` then give the Fock-block and integral coefficients, carried back to V.
/// The selector's density and correlation energy are rebuilt here and must match (the natural
/// orbitals diagonalize the virtual block to 1e-11, the energy to 1e-9 relative), or it refuses.
/// The open-shell MP2 of [`open_shell_mp2_lagrangian`] in the V frame: semicanonical spins, the
/// same-spin and opposite-spin amplitudes with their denominators, the correlation energy and the
/// spin-summed second-order density (as `correlation::open_shell_natural_orbitals` builds it).
struct OpenShellMp2 {
    spins: Vec<OpenShellSpin>,
    same: Vec<(Vec<f64>, Vec<f64>)>,
    t_ab: Vec<f64>,
    d_ab: Vec<f64>,
    correlation: f64,
    density: DMatrix<f64>,
}

fn open_shell_mp2(frame: &Frame, h: &DMatrix<f64>, nocc: [usize; 2]) -> Result<OpenShellMp2> {
    let m = frame.m;
    let high = nocc[0].max(nocc[1]);
    let occupied = |s: usize, k: usize| if k < nocc[s] { 1. } else { 0. };
    let fock: Vec<DMatrix<f64>> = (0..2)
        .map(|s| {
            sym(&DMatrix::from_fn(m, m, |p, q| {
                h[(p, q)]
                    + (0..high)
                        .map(|k| {
                            (occupied(0, k) + occupied(1, k)) * frame.chem(p, q, k, k)
                                - occupied(s, k) * frame.chem(p, k, k, q)
                        })
                        .sum::<f64>()
            }))
        })
        .collect();
    let gap = |value: f64| -> Result<f64> {
        if value.abs() < 1e-10 {
            Err(invalid(
                "open-shell MP2 met a vanishing orbital-energy denominator (a degenerate frontier)",
            ))
        } else {
            Ok(value)
        }
    };
    let mut spins = Vec::with_capacity(2);
    for (s, f) in fock.iter().enumerate() {
        let (o, v) = (nocc[s], m - nocc[s]);
        let mut co = DMatrix::<f64>::zeros(m, o);
        let mut cv = DMatrix::<f64>::zeros(m, v);
        let (mut eo, mut ev) = (vec![], vec![]);
        if o > 0 {
            let (e, w) = crate::math::linalg::real_eigen(f.view((0, 0), (o, o)).into_owned())?;
            co.view_mut((0, 0), (o, o)).copy_from(&w);
            eo = e;
        }
        if v > 0 {
            let (e, w) = crate::math::linalg::real_eigen(f.view((o, o), (v, v)).into_owned())?;
            cv.view_mut((o, 0), (v, v)).copy_from(&w);
            ev = e;
        }
        let f_ov = co.transpose() * f * &cv;
        let mut t1 = DMatrix::<f64>::zeros(o, v);
        for i in 0..o {
            for a in 0..v {
                t1[(i, a)] = f_ov[(i, a)] / gap(eo[i] - ev[a])?;
            }
        }
        spins.push(OpenShellSpin { co, cv, eo, ev, t1 });
    }
    let mut correlation = 0.;
    for (s, spin) in spins.iter().enumerate() {
        let f_ov = spin.co.transpose() * &fock[s] * &spin.cv;
        correlation += f_ov.component_mul(&spin.t1).sum();
    }
    // Same-spin pairs: t_ijab = <ij||ab> / D_ijab, the full antisymmetric array.
    let mut same = Vec::with_capacity(2);
    for spin in &spins {
        let (o, v) = (spin.eo.len(), spin.ev.len());
        let g = if o > 0 && v > 0 { block4(frame, [&spin.co, &spin.co, &spin.cv, &spin.cv]) } else { vec![] };
        let at = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
        let mut t = vec![0.; g.len()];
        let mut denominator = vec![0.; g.len()];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let anti = g[at(i, j, a, b)] - g[at(i, j, b, a)];
                        let d = spin.eo[i] + spin.eo[j] - spin.ev[a] - spin.ev[b];
                        let value = if anti == 0. { 0. } else { anti / gap(d)? };
                        t[at(i, j, a, b)] = value;
                        denominator[at(i, j, a, b)] = d;
                        correlation += 0.25 * anti * value;
                    }
                }
            }
        }
        same.push((t, denominator));
    }
    let (alpha, beta) = (&spins[0], &spins[1]);
    let (oa, va, ob, vb) = (alpha.eo.len(), alpha.ev.len(), beta.eo.len(), beta.ev.len());
    let g_ab = if oa > 0 && ob > 0 && va > 0 && vb > 0 {
        block4(frame, [&alpha.co, &beta.co, &alpha.cv, &beta.cv])
    } else {
        vec![]
    };
    let ab = |i: usize, j: usize, a: usize, b: usize| ((i * ob + j) * va + a) * vb + b;
    let mut t_ab = vec![0.; g_ab.len()];
    let mut d_ab = vec![0.; g_ab.len()];
    if !g_ab.is_empty() {
        for i in 0..oa {
            for j in 0..ob {
                for a in 0..va {
                    for b in 0..vb {
                        let d = alpha.eo[i] + beta.eo[j] - alpha.ev[a] - beta.ev[b];
                        let value = g_ab[ab(i, j, a, b)] / gap(d)?;
                        t_ab[ab(i, j, a, b)] = value;
                        d_ab[ab(i, j, a, b)] = d;
                        correlation += g_ab[ab(i, j, a, b)] * value;
                    }
                }
            }
        }
    }
    // The spin-summed second-order density in V (as the selector builds it), and its check.
    let mut density = DMatrix::<f64>::zeros(m, m);
    for (s, spin) in spins.iter().enumerate() {
        let (o, v) = (spin.eo.len(), spin.ev.len());
        let t = &same[s].0;
        let at = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
        let mut vir = spin.t1.transpose() * &spin.t1;
        let mut occ = -(&spin.t1 * spin.t1.transpose());
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let x = t[at(i, j, a, b)];
                        if x == 0. {
                            continue;
                        }
                        for c in 0..v {
                            vir[(a, c)] += 0.5 * x * t[at(i, j, c, b)];
                        }
                        for k in 0..o {
                            occ[(i, k)] -= 0.5 * x * t[at(k, j, a, b)];
                        }
                    }
                }
            }
        }
        if !t_ab.is_empty() {
            for i in 0..oa {
                for j in 0..ob {
                    for a in 0..va {
                        for b in 0..vb {
                            let x = t_ab[ab(i, j, a, b)];
                            if s == 0 {
                                for c in 0..va {
                                    vir[(a, c)] += x * t_ab[ab(i, j, c, b)];
                                }
                                for k in 0..oa {
                                    occ[(i, k)] -= x * t_ab[ab(k, j, a, b)];
                                }
                            } else {
                                for c in 0..vb {
                                    vir[(b, c)] += x * t_ab[ab(i, j, a, c)];
                                }
                                for k in 0..ob {
                                    occ[(j, k)] -= x * t_ab[ab(i, k, a, b)];
                                }
                            }
                        }
                    }
                }
            }
        }
        density += &spin.co * occ * spin.co.transpose() + &spin.cv * vir * spin.cv.transpose();
    }
    let density = sym(&density);
    Ok(OpenShellMp2 { spins, same, t_ab, d_ab, correlation, density })
}

fn open_shell_mp2_lagrangian(
    frame: &Frame,
    h: &DMatrix<f64>,
    nocc: [usize; 2],
    d_bar: &DMatrix<f64>,
    rotation: &DMatrix<f64>,
    keep: &[usize],
    correlation_energy: f64,
) -> Result<(DMatrix<f64>, Vec<f64>)> {
    let m = frame.m;
    let high = nocc[0].max(nocc[1]);
    let occupied = |s: usize, k: usize| if k < nocc[s] { 1. } else { 0. };
    let OpenShellMp2 { spins, same, t_ab, d_ab, correlation, density } = open_shell_mp2(frame, h, nocc)?;
    if (correlation - correlation_energy).abs() > 1e-9 * correlation_energy.abs().max(1e-3) {
        return Err(crate::Error::Unsupported(format!(
            "the natural-orbital response of this unrestricted selection: the rebuilt open-shell MP2 correlation \
             energy {correlation:.12e} is not the selector's {correlation_energy:.12e}"
        )));
    }
    let (alpha, beta) = (&spins[0], &spins[1]);
    let (oa, va, ob, vb) = (alpha.eo.len(), alpha.ev.len(), beta.eo.len(), beta.ev.len());
    let ab = |i: usize, j: usize, a: usize, b: usize| ((i * ob + j) * va + a) * vb + b;
    let k = m - high;
    let rb = DMatrix::from_fn(k, k, |i, j| rotation[(high + i, high + j)]);
    let rotated = rb.transpose() * density.view((high, high), (k, k)) * &rb;
    let residual = (0..k)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .fold(0f64, |w, (i, j)| w.max(rotated[(i, j)].abs()));
    let kk = keep.len();
    let rk = DMatrix::from_fn(kk, kk, |i, j| rotation[(keep[i], keep[j])]);
    let occupied_rotated =
        rk.transpose() * DMatrix::from_fn(kk, kk, |i, j| density[(keep[i], keep[j])]) * &rk;
    let residual = (0..kk)
        .flat_map(|i| (0..kk).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .fold(residual, |w, (i, j)| w.max(occupied_rotated[(i, j)].abs()));
    if residual > 1e-11 {
        return Err(invalid(format!(
            "the natural orbitals of the unrestricted active space do not diagonalize its open-shell MP2 density \
             (off-diagonal {residual:.2e}): the eigenvector response needs exact eigenvectors"
        )));
    }
    // The multipliers: tbar = (dL_D / dt) / D in the semicanonical blocks.
    let bar_v: Vec<DMatrix<f64>> = spins.iter().map(|s| s.cv.transpose() * d_bar * &s.cv).collect();
    let bar_o: Vec<DMatrix<f64>> = spins.iter().map(|s| s.co.transpose() * d_bar * &s.co).collect();
    let mut p_bar = DMatrix::<f64>::zeros(m, m);
    let mut two_bar = vec![0.; m.pow(4)];
    let at4 = |p: usize, q: usize, r: usize, s: usize| ((p * m + q) * m + r) * m + s;
    // Carry the coefficients `c` of `<ij|ab>` over four orbital sets back to V: Gamma += 2 c.
    let mut add_two = |c: &[f64], sets: [&DMatrix<f64>; 4]| {
        let k: Vec<usize> = sets.iter().map(|x| x.ncols()).collect();
        let mut current = c.to_vec();
        let mut shape = [k[0], k[1], k[2], k[3]];
        for axis in 0..4 {
            let old = shape;
            shape[axis] = m;
            let mut next = vec![0.; shape.iter().product()];
            for (flat, out) in next.iter_mut().enumerate() {
                let mut index = [0; 4];
                let mut rest = flat;
                for a in (0..4).rev() {
                    index[a] = rest % shape[a];
                    rest /= shape[a];
                }
                let target = index[axis];
                let mut sum = 0.;
                for source in 0..k[axis] {
                    index[axis] = source;
                    let o_flat = ((index[0] * old[1] + index[1]) * old[2] + index[2]) * old[3] + index[3];
                    sum += sets[axis][(target, source)] * current[o_flat];
                }
                *out = sum;
            }
            current = next;
        }
        for (g, x) in two_bar.iter_mut().zip(&current) {
            *g += 2. * x;
        }
    };
    let mut f_bar: Vec<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> = spins
        .iter()
        .map(|s| {
            let (o, v) = (s.eo.len(), s.ev.len());
            (DMatrix::zeros(o, o), DMatrix::zeros(v, v), DMatrix::zeros(o, v))
        })
        .collect();
    for (s, spin) in spins.iter().enumerate() {
        let (o, v) = (spin.eo.len(), spin.ev.len());
        let bar = &bar_v[s];
        let bar_occ = &bar_o[s];
        // Singles: dL_D/dt1 = 2 t1 bar_v - 2 bar_o t1 (the occupied density is -t1 t1^T).
        let x1 = &spin.t1 * bar * 2. - bar_occ * &spin.t1 * 2.;
        let tbar1 = DMatrix::from_fn(o, v, |i, a| x1[(i, a)] / (spin.eo[i] - spin.ev[a]));
        let (oo, vv, ov) = &mut f_bar[s];
        *ov += &tbar1;
        *vv += spin.t1.transpose() * &tbar1;
        *oo -= &tbar1 * spin.t1.transpose();
        // Same-spin pairs: dL_D/dt_ijab = sum_e bar_ae t_ijeb.
        let (t, denominator) = &same[s];
        if t.is_empty() {
            continue;
        }
        let at = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
        let mut tbar = vec![0.; t.len()];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        // Virtual: sum_e bar_ae t_ijeb; occupied (-1/2 sum t_ikab t_jkab): -sum_l bar_il t_ljab.
                        let x: f64 = (0..v).map(|e| bar[(a, e)] * t[at(i, j, e, b)]).sum::<f64>()
                            - (0..o).map(|l| bar_occ[(i, l)] * t[at(l, j, a, b)]).sum::<f64>();
                        tbar[at(i, j, a, b)] = if x == 0. { 0. } else { x / denominator[at(i, j, a, b)] };
                    }
                }
            }
        }
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let y = tbar[at(i, j, a, b)];
                        if y == 0. {
                            continue;
                        }
                        // R_ijab: sum_c f_ac t_ijcb + f_bc t_ijac - sum_k f_ki t_kjab + f_kj t_ikab.
                        for c in 0..v {
                            vv[(a, c)] += y * t[at(i, j, c, b)];
                            vv[(b, c)] += y * t[at(i, j, a, c)];
                        }
                        for k in 0..o {
                            oo[(k, i)] -= y * t[at(k, j, a, b)];
                            oo[(k, j)] -= y * t[at(i, k, a, b)];
                        }
                    }
                }
            }
        }
        let c: Vec<f64> = (0..t.len())
            .map(|f| {
                let (b, a, j, i) = (f % v, (f / v) % v, (f / (v * v)) % o, f / (v * v * o));
                tbar[at(i, j, a, b)] - tbar[at(i, j, b, a)]
            })
            .collect();
        add_two(&c, [&spin.co, &spin.co, &spin.cv, &spin.cv]);
    }
    if !t_ab.is_empty() {
        let (bar_a, bar_b) = (&bar_v[0], &bar_v[1]);
        let (occ_a, occ_b) = (&bar_o[0], &bar_o[1]);
        let mut tbar = vec![0.; t_ab.len()];
        for i in 0..oa {
            for j in 0..ob {
                for a in 0..va {
                    for c in 0..vb {
                        let x: f64 = 2. * (0..va).map(|b| bar_a[(a, b)] * t_ab[ab(i, j, b, c)]).sum::<f64>()
                            + 2. * (0..vb).map(|d| bar_b[(c, d)] * t_ab[ab(i, j, a, d)]).sum::<f64>()
                            - 2. * (0..oa).map(|k| occ_a[(i, k)] * t_ab[ab(k, j, a, c)]).sum::<f64>()
                            - 2. * (0..ob).map(|k| occ_b[(j, k)] * t_ab[ab(i, k, a, c)]).sum::<f64>();
                        tbar[ab(i, j, a, c)] = x / d_ab[ab(i, j, a, c)];
                    }
                }
            }
        }
        for i in 0..oa {
            for j in 0..ob {
                for a in 0..va {
                    for b in 0..vb {
                        let y = tbar[ab(i, j, a, b)];
                        if y == 0. {
                            continue;
                        }
                        // R_ijab: sum_c f^a_ac t_ijcb + f^b_bc t_ijac - sum_k f^a_ki t_kjab + f^b_kj t_ikab.
                        for c in 0..va {
                            f_bar[0].1[(a, c)] += y * t_ab[ab(i, j, c, b)];
                        }
                        for c in 0..vb {
                            f_bar[1].1[(b, c)] += y * t_ab[ab(i, j, a, c)];
                        }
                        for k in 0..oa {
                            f_bar[0].0[(k, i)] -= y * t_ab[ab(k, j, a, b)];
                        }
                        for k in 0..ob {
                            f_bar[1].0[(k, j)] -= y * t_ab[ab(i, k, a, b)];
                        }
                    }
                }
            }
        }
        add_two(&tbar, [&alpha.co, &beta.co, &alpha.cv, &beta.cv]);
    }
    // The Fock-block coefficients in V, then f^s = h + J[D_a + D_b] - K[D_s] into P and Gamma.
    for (s, spin) in spins.iter().enumerate() {
        let (oo, vv, ov) = &f_bar[s];
        let l = sym(&(&spin.co * oo * spin.co.transpose() + &spin.cv * vv * spin.cv.transpose()
            + &spin.co * ov * spin.cv.transpose()));
        p_bar += &l;
        for p in 0..m {
            for q in 0..m {
                let value = l[(p, q)];
                if value == 0. {
                    continue;
                }
                for r in 0..high {
                    two_bar[at4(p, r, q, r)] += 2. * value * (occupied(0, r) + occupied(1, r));
                    two_bar[at4(p, r, r, q)] -= 2. * value * occupied(s, r);
                }
            }
        }
    }
    // Symmetrized over the eight permutations of a real <pq|rs> (p <-> r, q <-> s, the pairs):
    // the energy is unchanged, and the generalized Fock of the caller, which differentiates the
    // first index only, needs a density with the symmetries of a reduced density matrix.
    let mut symmetric = vec![0.; two_bar.len()];
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for u in 0..m {
                    symmetric[at4(p, q, r, u)] = 0.125
                        * (two_bar[at4(p, q, r, u)]
                            + two_bar[at4(r, q, p, u)]
                            + two_bar[at4(p, u, r, q)]
                            + two_bar[at4(r, u, p, q)]
                            + two_bar[at4(q, p, u, r)]
                            + two_bar[at4(u, p, q, r)]
                            + two_bar[at4(q, r, u, p)]
                            + two_bar[at4(u, r, q, p)]);
                }
            }
        }
    }
    Ok((p_bar, symmetric))
}
