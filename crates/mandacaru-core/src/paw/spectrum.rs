//! Radial generalized PAW eigenstates on uniform interior nodes.
use super::invalid;
use crate::{
    finite,
    math::separable::{self, Eigenpair, Pencil},
    product, Context, Result,
};

pub struct Channel<'a> {
    pub radii: &'a [f64],
    pub potential: &'a [f64],
    /// Projectors in u(r)=r R(r) form, row-major (rank, nodes).
    pub projectors: &'a [f64],
    pub coupling: &'a [f64],
    pub overlap: &'a [f64],
    pub rank: usize,
}

pub fn eigenpair(
    ctx: &Context,
    channel: Channel<'_>,
    state: usize,
    vectors: bool,
) -> Result<Eigenpair> {
    let n = channel.radii.len();
    let rank = channel.rank;
    if n < 2
        || channel.potential.len() != n
        || channel.projectors.len() != product(&[n, rank])?
        || channel.coupling.len() != product(&[rank, rank])?
        || channel.overlap.len() != product(&[rank, rank])?
    {
        return Err(invalid("invalid radial PAW channel dimensions"));
    }
    finite(channel.radii, "radial PAW nodes")?;
    let h = channel.radii[0];
    if h <= 0.
        || channel
            .radii
            .iter()
            .enumerate()
            .any(|(i, &r)| (r - (i + 1) as f64 * h).abs() > 1e-10 * (1. + r.abs()))
    {
        return Err(invalid(
            "radial PAW nodes must be uniformly spaced interior nodes starting at h",
        ));
    }
    let mut diagonal = ctx.allocate(n)?;
    let mut off = ctx.allocate(n - 1)?;
    let mut u = ctx.allocate(product(&[n, rank])?)?;
    for i in 0..n {
        diagonal.as_mut_slice()[i] = 1. / h.powi(2) + channel.potential[i];
        for j in 0..rank {
            u.as_mut_slice()[i * rank + j] = h.sqrt() * channel.projectors[j * n + i];
        }
    }
    off.as_mut_slice().fill(-0.5 / h.powi(2));
    let mut out = separable::eigenpair(
        ctx,
        Pencil {
            diagonal: diagonal.as_slice(),
            off_diagonal: off.as_slice(),
            projectors: u.as_slice(),
            coupling: channel.coupling,
            metric: channel.overlap,
            rank,
        },
        state,
        vectors,
    )?;
    if let Some(vector) = out.vector.as_mut() {
        for v in vector.as_mut_slice() {
            *v /= h.sqrt();
        }
    }
    Ok(out)
}
