//! The two-body tensor of a periodic cell from its pair densities and their potentials.
//!
//! `raw[i, j] = sum_g rho_i(g) phi_j(g)` over the grid, every pair `i = (a, b)` of the `M` orbitals,
//! then the chemists' tensor in physicists' order scaled by the volume element, optionally rotated
//! by `X` (the Loewdin orthonormalization) and `V` (molecular orbitals): `out[p,q,r,s] = sum X[a,p]
//! X[b,q] X[c,r] X[d,s] eri[a,b,c,d]`, then the same with `V`.
use crate::periodic::invalid;
use crate::{product, Context, Result};

/// The `(M, M, M, M)` tensor of `m` orbitals from `rho` and `phi`, each `(M^2, points)` row-major.
#[allow(clippy::too_many_arguments)]
pub fn pair_two_body(
    ctx: &Context,
    rho: &[f64],
    phi: &[f64],
    points: usize,
    dv: f64,
    x: Option<&[f64]>,
    v: Option<&[f64]>,
    m: usize,
) -> Result<Vec<f64>> {
    let m2 = product(&[m, m])?;
    if rho.len() != product(&[m2, points])? || phi.len() != rho.len() || !dv.is_finite() {
        return Err(invalid("the pair densities and potentials must both be (M^2, points)"));
    }
    // The raw product, the tensor and the four rotation scratches.
    let _memory = ctx.reserve(product(&[m2, m2, 6, std::mem::size_of::<f64>()])?)?;
    let mut raw = vec![0.0_f64; m2 * m2];
    for i in 0..m2 {
        let row = &rho[i * points..(i + 1) * points];
        for j in 0..m2 {
            let col = &phi[j * points..(j + 1) * points];
            raw[i * m2 + j] = row.iter().zip(col).map(|(a, b)| a * b).sum();
        }
    }
    let raw_slice = raw.as_slice();
    let mut eri = vec![0.0_f64; m * m * m * m];
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                for d in 0..m {
                    // chemists: row = a*M + c, col = b*M + d
                    // physicists: eri[a, b, c, d] = chem[a, c, b, d] = raw[(a*m + b)*m2 + c*m + d]
                    eri[((a * m + b) * m + c) * m + d] =
                        raw_slice[(a * m + b) * m2 + c * m + d] * dv;
                }
            }
        }
    }

    // Transform with X if provided
    let mut cur = eri;
    if let Some(x_mat) = x {
        if x_mat.len() != m * m {
            return Err(invalid("X must be (M, M)"));
        }
        let x_slice = x_mat;
        // cur[p, q, r, s] = sum_{a,b,c,d} X[a, p] X[b, q] X[c, r] X[d, s] cur[a, b, c, d]
        // 4 successive contractions
        let mut t1 = vec![0.0_f64; m * m * m * m];
        for a in 0..m {
            for p in 0..m {
                let x_ap = x_slice[a * m + p];
                if x_ap == 0.0 {
                    continue;
                }
                for b in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            t1[((p * m + b) * m + c) * m + d] +=
                                x_ap * cur[((a * m + b) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t2 = vec![0.0_f64; m * m * m * m];
        for b in 0..m {
            for q in 0..m {
                let x_bq = x_slice[b * m + q];
                if x_bq == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            t2[((p * m + q) * m + c) * m + d] +=
                                x_bq * t1[((p * m + b) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t3 = vec![0.0_f64; m * m * m * m];
        for c in 0..m {
            for r in 0..m {
                let x_cr = x_slice[c * m + r];
                if x_cr == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for q in 0..m {
                        for d in 0..m {
                            t3[((p * m + q) * m + r) * m + d] +=
                                x_cr * t2[((p * m + q) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t4 = vec![0.0_f64; m * m * m * m];
        for d in 0..m {
            for s in 0..m {
                let x_ds = x_slice[d * m + s];
                if x_ds == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for q in 0..m {
                        for r in 0..m {
                            t4[((p * m + q) * m + r) * m + s] +=
                                x_ds * t3[((p * m + q) * m + r) * m + d];
                        }
                    }
                }
            }
        }
        cur = t4;
    }

    // Transform with V if provided
    if let Some(v_mat) = v {
        if v_mat.len() != m * m {
            return Err(invalid("V must be (M, M)"));
        }
        let v_slice = v_mat;
        let mut t1 = vec![0.0_f64; m * m * m * m];
        for a in 0..m {
            for p in 0..m {
                let v_ap = v_slice[a * m + p];
                if v_ap == 0.0 {
                    continue;
                }
                for b in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            t1[((p * m + b) * m + c) * m + d] +=
                                v_ap * cur[((a * m + b) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t2 = vec![0.0_f64; m * m * m * m];
        for b in 0..m {
            for q in 0..m {
                let v_bq = v_slice[b * m + q];
                if v_bq == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            t2[((p * m + q) * m + c) * m + d] +=
                                v_bq * t1[((p * m + b) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t3 = vec![0.0_f64; m * m * m * m];
        for c in 0..m {
            for r in 0..m {
                let v_cr = v_slice[c * m + r];
                if v_cr == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for q in 0..m {
                        for d in 0..m {
                            t3[((p * m + q) * m + r) * m + d] +=
                                v_cr * t2[((p * m + q) * m + c) * m + d];
                        }
                    }
                }
            }
        }
        let mut t4 = vec![0.0_f64; m * m * m * m];
        for d in 0..m {
            for s in 0..m {
                let v_ds = v_slice[d * m + s];
                if v_ds == 0.0 {
                    continue;
                }
                for p in 0..m {
                    for q in 0..m {
                        for r in 0..m {
                            t4[((p * m + q) * m + r) * m + s] +=
                                v_ds * t3[((p * m + q) * m + r) * m + d];
                        }
                    }
                }
            }
        }
        cur = t4;
    }

    Ok(cur)
}

/// `A = X V` as a complex `(M, M)` row-major matrix: `X` the real Loewdin orthonormalization and
/// `V` the molecular orbitals, either absent for the identity.
pub fn orbital_rotation(
    m: usize,
    x: Option<&[f64]>,
    v: Option<&[num_complex::Complex64]>,
) -> Result<Vec<num_complex::Complex64>> {
    use num_complex::Complex64 as C64;
    let mm = product(&[m, m])?;
    if x.is_some_and(|x| x.len() != mm) || v.is_some_and(|v| v.len() != mm) {
        return Err(invalid("the orbital rotations must be (M, M)"));
    }
    let mut a: Vec<C64> = match x {
        Some(x) => x.iter().map(|&value| C64::new(value, 0.)).collect(),
        None => (0..mm).map(|k| C64::new(if k / m == k % m { 1. } else { 0. }, 0.)).collect(),
    };
    if let Some(v) = v {
        let mut out = vec![C64::new(0., 0.); mm];
        for i in 0..m {
            for k in 0..m {
                let left = a[i * m + k];
                for j in 0..m {
                    out[i * m + j] += left * v[k * m + j];
                }
            }
        }
        a = out;
    }
    Ok(a)
}

/// `rho(g) = Re sum_pq D_pq conj(phi_p(g)) phi_q(g)` with `phi = A^T psi`: the density of the state
/// whose one-body density matrix is `D` in the orbitals `A` of the real functions `psi`
/// (`(M, points)` row-major).
pub fn rotated_density(
    ctx: &Context,
    psi: &[f64],
    points: usize,
    a: &[num_complex::Complex64],
    d: &[f64],
    m: usize,
) -> Result<Vec<f64>> {
    use num_complex::Complex64 as C64;
    let mm = product(&[m, m])?;
    if psi.len() != product(&[m, points])? || a.len() != mm || d.len() != mm {
        return Err(invalid("psi is (M, points), A and D are (M, M)"));
    }
    let _memory = ctx.reserve(product(&[m, points, std::mem::size_of::<C64>()])?
        .checked_add(product(&[points, std::mem::size_of::<f64>()])?)
        .ok_or(crate::Error::Overflow)?)?;
    let mut phi = vec![C64::new(0., 0.); m * points];
    for p in 0..m {
        for mu in 0..m {
            let coefficient = a[mu * m + p];
            if coefficient == C64::new(0., 0.) {
                continue;
            }
            for (out, &value) in phi[p * points..(p + 1) * points].iter_mut()
                .zip(&psi[mu * points..(mu + 1) * points])
            {
                *out += coefficient * value;
            }
        }
    }
    let mut density = vec![0.; points];
    for p in 0..m {
        for q in 0..m {
            let weight = d[p * m + q];
            if weight == 0. {
                continue;
            }
            for g in 0..points {
                density[g] += weight * (phi[p * points + g].conj() * phi[q * points + g]).re;
            }
        }
    }
    Ok(density)
}
