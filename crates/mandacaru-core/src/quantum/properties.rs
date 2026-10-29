//! Spin-orbital RDMs in physicists' notation, evaluated in occupation space.
use super::{Basis, Encoding, C64};
use crate::{product, Buffer, Context, Error, Result};

fn ladder(n: usize, mut ket: u64, factors: &[(usize, bool)]) -> Option<(u64, f64)> {
    let mut sign = 1.;
    for &(j, creation) in factors.iter().rev() {
        let mask = 1u64 << (n - 1 - j);
        if (ket & mask != 0) == creation {
            return None;
        }
        if j > 0 && (ket >> (n - j)).count_ones() % 2 != 0 {
            sign = -sign;
        }
        ket ^= mask;
    }
    Some((ket, sign))
}
fn expectation(
    encoding: &Encoding,
    basis: &Basis,
    state: &[C64],
    factors: &[(usize, bool)],
) -> C64 {
    let mut value = C64::default();
    for (&index, amplitude) in basis.indices.as_slice().iter().zip(state) {
        if let Some((target, sign)) = ladder(encoding.modes, encoding.decode(index), factors) {
            let encoded = encoding.encode(target);
            // Reduced registers represent only the fixed parity subspace.
            if encoding.decode(encoded) != target {
                continue;
            }
            if let Some(row) = basis.index(encoded) {
                value += state[row].conj() * amplitude * sign;
            }
        }
    }
    value
}
pub struct Rdms {
    pub one: Buffer<C64>,
    pub two: Option<Buffer<C64>>,
}
/// Spin-summed spatial density matrices without allocating a spin-orbital
/// four-index tensor. Occupied annihilators and allowed empty creators are
/// enumerated once per determinant, rather than scanning every zero RDM term.
pub struct SpatialRdms {
    pub m: usize,
    pub one: Buffer<C64>,
    pub two: Buffer<C64>,
}
pub fn spatial_rdms(
    ctx: &Context,
    encoding: &Encoding,
    basis: &Basis,
    state: &[C64],
) -> Result<SpatialRdms> {
    if encoding.modes % 2 != 0
        || state.len() != basis.len()
        || state.iter().any(|c| !c.re.is_finite() || !c.im.is_finite())
    {
        return Err(Error::Invalid(
            "spatial RDMs need a finite state in an even spin-blocked register".into(),
        ));
    }
    let n = encoding.modes;
    let m = n / 2;
    let mut one = ctx.allocate::<C64>(product(&[m, m])?)?;
    let mut two = ctx.allocate::<C64>(product(&[m, m, m, m])?)?;
    let locate = |occupation| {
        let encoded = encoding.encode(occupation);
        if encoding.decode(encoded) != occupation {
            None
        } else {
            basis.index(encoded)
        }
    };
    for (&index, &amplitude) in basis.indices.as_slice().iter().zip(state) {
        if amplitude == C64::new(0., 0.) {
            continue;
        }
        let ket = encoding.decode(index);
        for r in 0..n {
            let Some((first, sign_r)) = ladder(n, ket, &[(r, false)]) else {
                continue;
            };
            let spin_r = r / m * m;
            for p in spin_r..spin_r + m {
                if let Some((target, sign_p)) = ladder(n, first, &[(p, true)]) {
                    if let Some(row) = locate(target) {
                        one.as_mut_slice()[(p % m) * m + r % m] +=
                            state[row].conj() * amplitude * (sign_r * sign_p);
                    }
                }
            }
            for s in 0..n {
                let Some((second, sign_s)) = ladder(n, first, &[(s, false)]) else {
                    continue;
                };
                let spin_s = s / m * m;
                for q in spin_s..spin_s + m {
                    let Some((third, sign_q)) = ladder(n, second, &[(q, true)]) else {
                        continue;
                    };
                    for p in spin_r..spin_r + m {
                        if let Some((target, sign_p)) = ladder(n, third, &[(p, true)]) {
                            if let Some(row) = locate(target) {
                                let at = (((p % m) * m + q % m) * m + r % m) * m + s % m;
                                two.as_mut_slice()[at] += state[row].conj()
                                    * amplitude
                                    * (sign_r * sign_s * sign_q * sign_p);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(SpatialRdms { m, one, two })
}
pub fn rdms(
    ctx: &Context,
    encoding: &Encoding,
    basis: &Basis,
    state: &[C64],
    two_body: bool,
) -> Result<Rdms> {
    if state.len() != basis.len() || state.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
        return Err(Error::Invalid(
            "RDM state must be finite and match the basis".into(),
        ));
    }
    let n = encoding.modes;
    let mut one = ctx.allocate(product(&[n, n])?)?;
    let mut two = if two_body {
        Some(ctx.allocate(product(&[n, n, n, n])?)?)
    } else {
        None
    };
    for p in 0..n {
        for q in 0..n {
            one.as_mut_slice()[p * n + q] =
                expectation(encoding, basis, state, &[(p, true), (q, false)]);
            if let Some(two) = two.as_mut() {
                if p != q {
                    for r in 0..n {
                        for s in 0..n {
                            if r != s {
                                two.as_mut_slice()[((p * n + q) * n + r) * n + s] = expectation(
                                    encoding,
                                    basis,
                                    state,
                                    &[(p, true), (q, true), (s, false), (r, false)],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Rdms { one, two })
}
