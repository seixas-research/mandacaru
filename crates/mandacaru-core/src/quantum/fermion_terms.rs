//! Second-quantized terms of the molecular Hamiltonian from free integral arrays.
//!
//! `H = sum h_pq a+_p a_q + 1/2 sum <pq|rs> a+_p a+_q a_s a_r` in physicists' notation, as an
//! ordered list of `(ladder product, coefficient)`: the one-body entries above `tol` in `(p, q)`
//! order, then the two-body ones in `(p, q, r, s)` order, so every later sum over the terms runs
//! in a fixed order. Entries at or below `tol` in magnitude are skipped.
use super::C64;
use crate::{product, Context, Error, Result};

/// One term: the ladder operators `(mode, is_creation)` left to right, and its coefficient.
pub type FermionTerm = (Vec<(usize, bool)>, C64);

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn one_body(h: &[C64], n: usize, tol: f64, out: &mut Vec<FermionTerm>) {
    for p in 0..n {
        for q in 0..n {
            let value = h[p * n + q];
            if value.norm() > tol {
                out.push((vec![(p, true), (q, false)], value));
            }
        }
    }
}
fn bytes(terms: usize) -> Result<usize> {
    product(&[terms, 160])
}
fn count_above(values: &[C64], tol: f64) -> usize {
    values.iter().filter(|v| v.norm() > tol).count()
}

/// Terms of spin-orbital integrals: `one` is `(n, n)` and `two` the `(n, n, n, n)` tensor
/// `<pq|rs>`.
pub fn from_spin_orbital(
    ctx: &Context,
    n: usize,
    one: &[C64],
    two: Option<&[C64]>,
    tol: f64,
) -> Result<Vec<FermionTerm>> {
    let n4 = product(&[n, n, n, n])?;
    if one.len() != n * n || two.is_some_and(|g| g.len() != n4) {
        return Err(invalid("the integral arrays disagree with the orbital count"));
    }
    let count = count_above(one, tol) + two.map_or(0, |g| count_above(g, tol));
    let _memory = ctx.reserve(bytes(count)?)?;
    let mut out = Vec::with_capacity(count);
    one_body(one, n, tol, &mut out);
    if let Some(g) = two {
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let value = g[((p * n + q) * n + r) * n + s];
                        if value.norm() > tol {
                            out.push((
                                vec![(p, true), (q, true), (s, false), (r, false)],
                                C64::new(0.5 * value.re, 0.5 * value.im),
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Terms of spatial integrals over `m` orbitals, spin-orbital `P = p + sigma m`.
///
/// `one` is `(m, m)` (the same for both spins) or `(2, m, m)` when `resolved`, and `spin` an
/// optional `(2m, 2m)` one-body term added to the spin-orbital block. `two` is `(m, m, m, m)`
/// with `<pq|rs>` for both spin pairs, or `(2, 2, m, m, m, m)` when `resolved`, whose
/// `[sigma, tau]` block has electron 1 in spin `sigma` and electron 2 in spin `tau`. The
/// `(2m)^4` spin-orbital tensor is never formed: each spatial slab is scanned once per spin
/// block it populates.
pub fn from_spatial(
    ctx: &Context,
    m: usize,
    resolved: bool,
    one: &[C64],
    spin: Option<&[C64]>,
    two: Option<&[C64]>,
    tol: f64,
) -> Result<Vec<FermionTerm>> {
    let n = 2 * m;
    let m2 = product(&[m, m])?;
    let m4 = product(&[m2, m2])?;
    let spins = if resolved { 2 } else { 1 };
    if one.len() != spins * m2
        || spin.is_some_and(|s| s.len() != n * n)
        || two.is_some_and(|g| g.len() != spins * spins * m4)
    {
        return Err(invalid("the integral arrays disagree with the orbital count"));
    }
    let mut h = vec![C64::default(); n * n];
    for sigma in 0..2 {
        let block = &one[if resolved { sigma * m2 } else { 0 }..][..m2];
        for p in 0..m {
            for q in 0..m {
                h[(sigma * m + p) * n + sigma * m + q] = block[p * m + q];
            }
        }
    }
    if let Some(extra) = spin {
        for (value, add) in h.iter_mut().zip(extra) {
            *value += add;
        }
    }
    let slab = |sigma: usize, tau: usize, p: usize| -> &[C64] {
        let two = two.expect("slab of absent two-body integrals");
        let start = if resolved {
            (((sigma * 2 + tau) * m + p) * m2) * m
        } else {
            p * m2 * m
        };
        &two[start..start + m2 * m]
    };
    let mut count = count_above(&h, tol);
    if two.is_some() {
        for big_p in 0..n {
            for tau in 0..2 {
                count += count_above(slab(big_p / m, tau, big_p % m), tol);
            }
        }
    }
    let _memory = ctx.reserve(bytes(count)?)?;
    let mut out = Vec::with_capacity(count);
    one_body(&h, n, tol, &mut out);
    if two.is_some() {
        for big_p in 0..n {
            let (sigma, p) = (big_p / m, big_p % m);
            for tau in 0..2 {
                let shift = tau * m;
                let block = slab(sigma, tau, p);
                for q in 0..m {
                    for r in 0..m {
                        for s in 0..m {
                            let value = block[(q * m + r) * m + s];
                            if value.norm() > tol {
                                out.push((
                                    vec![
                                        (big_p, true),
                                        (q + shift, true),
                                        (s + shift, false),
                                        (r + sigma * m, false),
                                    ],
                                    C64::new(0.5 * value.re, 0.5 * value.im),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `strength * direction` for a Cartesian unit `direction` (norm one to 1e-8; a non-unit one,
/// the zero vector included, is refused rather than rescaled) and a finite, nonnegative
/// `strength` in atomic units.
pub fn field_vector(direction: [f64; 3], strength: f64) -> Result<[f64; 3]> {
    if direction.iter().any(|v| !v.is_finite()) {
        return Err(invalid("field_direction must be a finite real Cartesian unit vector"));
    }
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if (norm - 1.).abs() > 1e-8 {
        return Err(invalid("field_direction must have unit norm"));
    }
    if !strength.is_finite() || strength < 0. {
        return Err(invalid("field_strength must be finite and nonnegative"));
    }
    Ok(direction.map(|v| strength * v))
}

/// The one-body matrix and constant of the length-gauge dipole coupling `V = -E . mu`, with
/// `mu = mu_nuc - sum_pq <p|r|q> a+_p a_q` in atomic units: `h_pq = E . <p|r|q>` (the electronic
/// potential has a positive sign) and the constant `-E . mu_nuc`. `positions` are the three
/// `(m, m)` Hermitian matrices `<p|r_a|q>` (Bohr), row-major one after another. With `spatial`
/// the matrix is copied into the alpha and beta blocks of the `2m` spin orbitals (alpha first,
/// Mandacaru's order); otherwise the matrices already act on spin orbitals. Returns
/// `(h, modes, constant)`, `h` row-major `modes x modes`.
pub fn dipole_one_body(
    positions: &[C64],
    m: usize,
    field: [f64; 3],
    nuclear: [f64; 3],
    spatial: bool,
) -> Result<(Vec<C64>, usize, f64)> {
    if m == 0 || positions.len() != product(&[3, m, m])? {
        return Err(invalid("position_integrals must have shape (3, m, m), m > 0"));
    }
    let hermitian = positions.iter().all(|v| v.re.is_finite() && v.im.is_finite())
        && (0..3).all(|a| {
            let block = &positions[a * m * m..(a + 1) * m * m];
            (0..m).all(|p| (0..m).all(|q| (block[p * m + q] - block[q * m + p].conj()).norm() <= 1e-12))
        });
    if !hermitian {
        return Err(invalid("position_integrals must be finite and Hermitian"));
    }
    if field.iter().chain(&nuclear).any(|v| !v.is_finite()) {
        return Err(invalid("field and nuclear_dipole must be finite real Cartesian vectors"));
    }
    let modes = if spatial { 2 * m } else { m };
    let mut h = vec![C64::default(); modes * modes];
    for p in 0..m {
        for q in 0..m {
            let value: C64 = (0..3).map(|a| positions[(a * m + p) * m + q] * field[a]).sum();
            h[p * modes + q] = value;
            if spatial {
                h[(p + m) * modes + (q + m)] = value;
            }
        }
    }
    let constant = -(0..3).map(|a| field[a] * nuclear[a]).sum::<f64>();
    Ok((h, modes, constant))
}
