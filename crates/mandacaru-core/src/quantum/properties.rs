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
        if (ket >> (n - j)).count_ones() % 2 != 0 {
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
            if let Ok(row) = basis.indices.as_slice().binary_search(&encoded) {
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
