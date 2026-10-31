//! PAW shell weights for the DFT+U occupation matrix `n = W^T R W`.
//!
//! A state `psi` has the one-center expansion `sum_i phi_i <p_i|psi>` inside the
//! sphere, so its amplitude on the bound all-electron partial wave `chi` of a
//! channel is `sum_i w_i <p_i|psi>` with `w_i = <chi|phi_i>` over the sphere,
//! `chi` normalized to one inside the channel's cutoff radius. With
//! `R = <p|psi><psi|p>` over the shell's projectors, ordered `m` then radial
//! index, the occupation matrix is `n_{mm'} = sum_ij w_i R_{(m,i),(m',j)} w_j`.
//!
//! This is the partial-wave projection of the deleted Python route, kept with its
//! normalization on purpose: the projector basis is not orthonormal and the part
//! of the orbital outside the sphere is not counted, so a filled shell has
//! `n < 1` (Ni 3d 0.951, Gd 4f 0.458). `n` is a sphere-truncated projection, not
//! an electron count. The weights are taken in the basis the projections were
//! built in: `n` does not depend on that choice (`w_raw = B^-1 w_dual` and the
//! raw projections are `B^T` times the dual ones).
use super::{
    invalid,
    onecenter::{Basis, Channel},
};
use crate::{math::spline::simpson, Buffer, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Largest number of partial waves per channel the weights accept.
const MAX_WAVES: usize = 8;

/// The geometry a channel's integrals share: the grid points inside the sphere and
/// the reference wave.
struct Reference<'a> {
    inside: usize,
    chi: &'a [f64],
}

fn reference<'a>(
    channel: &'a Channel<'_>,
    r: &[f64],
    occupations: &[f64],
) -> Result<Reference<'a>> {
    let grid = r.len();
    let waves = channel.waves;
    if waves == 0 || waves > MAX_WAVES {
        return Err(invalid(format!(
            "PAW shell: a channel has one to {MAX_WAVES} partial waves, not {waves}"
        )));
    }
    if grid < 2 || r.iter().any(|x| !x.is_finite()) || r.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid(
            "PAW shell: the radial grid must be finite and strictly increasing",
        ));
    }
    if channel.ae.len() != waves.saturating_mul(grid)
        || channel.vanderbilt.len() != waves.saturating_mul(waves)
        || occupations.len() != waves
    {
        return Err(invalid(
            "PAW shell: partial waves, Vanderbilt matrix or occupations do not match the grid \
             (occupations hold one number per partial wave; a library entry that stores only the \
             scalar channel occupation means [occupation, 0, ...])",
        ));
    }
    if occupations.iter().any(|f| !f.is_finite() || *f < 0.) {
        return Err(invalid("PAW shell: occupations must be finite and >= 0"));
    }
    if !channel.r_cut.is_finite() || channel.r_cut <= r[0] {
        return Err(invalid(
            "PAW shell: the cutoff radius must be finite and beyond the first grid point",
        ));
    }
    let inside = r.partition_point(|x| *x <= channel.r_cut);
    if inside < 2 {
        return Err(invalid(
            "PAW shell: fewer than two grid points inside the sphere",
        ));
    }
    // The first occupied wave is the reference orbital. A channel with none has no
    // bound orbital to project on: raise, never substitute wave 0.
    let occupied = occupations.iter().position(|f| *f > 0.).ok_or_else(|| {
        invalid("PAW shell: the channel has no occupied reference wave to project on")
    })?;
    Ok(Reference {
        inside,
        chi: &channel.ae[occupied * grid..(occupied + 1) * grid],
    })
}

/// `int chi^2 r^2` over the first `count` grid points, through `scratch`.
fn square_norm(r: &[f64], chi: &[f64], count: usize, scratch: &mut Buffer<f64>) -> Result<f64> {
    let work = &mut scratch.as_mut_slice()[..count];
    for p in 0..count {
        work[p] = chi[p] * chi[p] * (r[p] * r[p]);
    }
    simpson(&r[..count], work)
}

/// The sphere share of the reference orbital, `int_0^{r_c} chi^2 r^2 / int chi^2 r^2`
/// (the second over the whole grid): the largest `n` a filled shell can reach in
/// this projection when the orbital is the dataset's own reference.
pub fn sphere_fraction(
    ctx: &Context,
    channel: &Channel<'_>,
    r: &[f64],
    occupations: &[f64],
) -> Result<f64> {
    let reference = reference(channel, r, occupations)?;
    let mut scratch = ctx.allocate::<f64>(r.len())?;
    let inner = square_norm(r, reference.chi, reference.inside, &mut scratch)?;
    let full = square_norm(r, reference.chi, r.len(), &mut scratch)?;
    if !(inner > 0. && full > 0.) {
        return Err(invalid(
            "PAW shell: the reference wave has no norm inside the sphere",
        ));
    }
    Ok(inner / full)
}

/// Electrons of the reference atom in the channel, the sum of its occupations: the shell's
/// starting electron count in the DFT+U seed.
pub fn reference_electrons(occupations: &[f64]) -> Result<f64> {
    if occupations.iter().any(|f| !f.is_finite() || *f < 0.) {
        return Err(invalid("PAW shell: occupations must be finite and >= 0"));
    }
    Ok(occupations.iter().sum())
}

/// `w_i = <chi|phi_i>` over the augmentation sphere, one weight per projector `i` of
/// the channel in the projector `basis`, with `chi` the first occupied all-electron
/// partial wave normalized to one inside `r_cut`.
///
/// `occupations` are the channel's reference occupations (one per partial wave); the
/// first positive one marks the reference orbital. A library entry that stores only the
/// scalar channel occupation (lda-sr Nd and Eu 4f and 6s, seven lda-fr channels) has the list
/// `[occupation, 0, ...]`: the library reader pads it so, the caller passes the padded list, and the reference is wave 0 as it was
/// in the old Python. A channel with no occupied wave at all (27 d and f channels of lda-sr and 10 of
/// lda-fr, the unoccupied shells, for example Ba 5d) raises; the old Python took wave 0
/// silently there, which is a scattering wave and would define a different shell.
///
/// `channel.ps` is not read. The raw-basis waves `phi^raw_i = sum_k (B^-1)_{ik} phi_k`
/// are formed one at a time in a single grid-sized buffer from `ctx`.
pub fn weights(
    ctx: &Context,
    channel: &Channel<'_>,
    r: &[f64],
    occupations: &[f64],
    basis: Basis,
) -> Result<Vec<f64>> {
    let reference = reference(channel, r, occupations)?;
    let (grid, count) = (r.len(), reference.inside);
    if channel.vanderbilt.iter().any(|x| !x.is_finite()) {
        return Err(invalid("PAW shell: the Vanderbilt matrix is not finite"));
    }
    let transform = channel.transform(basis)?;
    if transform
        .as_ref()
        .is_some_and(|t| t.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "PAW shell: the projector basis change is not finite",
        ));
    }
    let mut scratch = ctx.allocate::<f64>(count)?;
    let norm = square_norm(r, reference.chi, count, &mut scratch)?;
    if !(norm > 0.) {
        return Err(invalid(format!(
            "PAW shell: the channel has no bound partial wave to project on (norm {norm:e})"
        )));
    }
    let root = norm.sqrt();
    let mut out = Vec::new();
    out.try_reserve_exact(channel.waves)
        .map_err(|_| crate::Error::Allocation)?;
    for i in 0..channel.waves {
        let work = scratch.as_mut_slice();
        for p in 0..count {
            let wave = match &transform {
                None => channel.ae[i * grid + p],
                Some(t) => {
                    (0..channel.waves).fold(0., |s, k| s + t[(i, k)] * channel.ae[k * grid + p])
                }
            };
            work[p] = reference.chi[p] * wave * (r[p] * r[p]);
        }
        out.push(simpson(&r[..count], &work[..count])? / root);
    }
    Ok(out)
}

/// `n = W^T R W` of one shell. `projected` is the `(2l+1)p x (2l+1)p` matrix
/// `R = <p|psi><psi|p>` over the shell's projectors ordered `m` (from `-l` to `l`)
/// and then radial index, `p = weights.len()`; the result is the `(2l+1) x (2l+1)`
/// Hermitian occupation matrix (the caller halves `R` for a restricted run). The output
/// is `(2l+1)^2` complex numbers, at most `17^2 * 16` bytes = 4.6 KB for `l <= 8`; it is
/// small, bounded and returned to the caller, so it is not charged to a `Context`.
pub fn occupation(
    l: usize,
    weights: &[f64],
    projected: &DMatrix<Complex64>,
) -> Result<DMatrix<Complex64>> {
    let p = weights.len();
    let m = 2 * l + 1;
    if p == 0 || p > MAX_WAVES || l > 8 || weights.iter().any(|w| !w.is_finite()) {
        return Err(invalid(
            "PAW shell: the weights must be one to eight finite numbers and l <= 8",
        ));
    }
    if projected.nrows() != m * p || projected.ncols() != m * p {
        return Err(invalid(format!(
            "PAW shell: R is {} x {}, expected {} x {} for l = {l} with {p} radial projectors",
            projected.nrows(),
            projected.ncols(),
            m * p,
            m * p
        )));
    }
    if projected
        .iter()
        .any(|x| !x.re.is_finite() || !x.im.is_finite())
    {
        return Err(invalid("PAW shell: R is not finite"));
    }
    Ok(DMatrix::from_fn(m, m, |a, b| {
        let mut sum = Complex64::new(0., 0.);
        for i in 0..p {
            for j in 0..p {
                sum += projected[(a * p + i, b * p + j)] * (weights[i] * weights[j]);
            }
        }
        sum
    }))
}
