//! The bound spectrum of a channel's smooth Hamiltonian
//! `T_l + v^scr + sum |p_i> D^scr_ij <p_j|` with the metric `1 + sum |p_i> q_ij <p_j|`.
//!
//! The generalized radial problem is solved on uniform interior nodes by
//! [`crate::paw::spectrum::eigenpair`]. Eigenvalues of a channel are the 3-point Laplacian's at
//! two node spacings, `stride` and `2 * stride` times [`SPACING`], Richardson-extrapolated,
//! `(4 e_fine - e_coarse) / 3` (the discretization error is second order). The box reaches
//! out to where the outermost occupied smooth wave has decayed to [`EXTENT_FLOOR`] of its peak.
use super::view::{Channel, Smooth};
use crate::{
    paw::{
        checks::spectrum_extent,
        confinement::{ChannelData, Prepared},
        invalid,
    },
    Context, Result,
};

/// Node spacing (Bohr) of the finest grid the spectrum is solved on, per unit of stride.
pub const SPACING: f64 = 0.005;
/// Stride of the finer of the two grids (so the nodes are `0.01` and `0.02` Bohr apart).
pub const STRIDE: f64 = 2.;
/// The box ends where the outermost occupied wave has decayed to this fraction of its peak.
pub const EXTENT_FLOOR: f64 = 1e-5;
/// The box radius is clipped to this interval (Bohr).
pub const EXTENT_BOUNDS: (f64, f64) = (12., 24.);
/// Spacing (Bohr) and reach (Bohr) of the single grid the `j` branches' lowest level is solved on.
pub const BRANCH_SPACING: f64 = 0.004;
pub const BRANCH_REACH: f64 = 22.;

/// The nodes `radial_step, 2 radial_step, ..., floor(r_max / radial_step) radial_step` of the
/// radial box (`np.arange(1, int(r_max / step) + 1) * step`).
pub fn nodes(r_max: f64, radial_step: f64) -> Vec<f64> {
    (1..=(r_max / radial_step) as usize)
        .map(|i| i as f64 * radial_step)
        .collect()
}

/// The channel's splines on the dataset's table, ready to be evaluated on any uniform grid.
pub fn prepared<'a>(
    ctx: &Context,
    smooth: &'a Smooth,
    channel: &'a Channel,
) -> Result<Prepared<'a>> {
    if smooth.r.as_slice().len() * channel.rank != channel.projectors.as_slice().len() {
        return Err(invalid(
            "diagnose: the projectors are not on the dataset's radial table",
        ));
    }
    Prepared::new(
        ctx,
        ChannelData {
            table: smooth.r.as_slice(),
            local: smooth.v_local_screened.as_slice(),
            projectors: channel.projectors.as_slice(),
            rank: channel.rank,
            coupling: &channel.coupling,
            overlap: &channel.overlap,
            wave: channel.pseudo_waves[0].as_slice(),
            l: channel.l,
            r_cut: channel.r_cut,
        },
    )
}

/// Radius of the spectrum's box: where the outermost occupied smooth wave (a semicore
/// channel's valence wave, not its compact one) has decayed to [`EXTENT_FLOOR`].
pub fn extent(smooth: &Smooth, channel: &Channel) -> Result<f64> {
    let occupied = channel.occupied_energies().len();
    spectrum_extent(
        smooth.r.as_slice(),
        channel.pseudo_waves[occupied - 1].as_slice(),
        EXTENT_FLOOR,
        EXTENT_BOUNDS.0,
        EXTENT_BOUNDS.1,
    )
}

/// The lowest `states` eigenvalues of the channel (Hartree), Richardson-extrapolated.
pub fn eigenvalues(
    ctx: &Context,
    prepared: &Prepared,
    r_max: f64,
    states: usize,
) -> Result<Vec<f64>> {
    let fine = nodes(r_max, STRIDE * SPACING);
    let coarse = nodes(r_max, 2. * STRIDE * SPACING);
    (0..states)
        .map(|state| {
            let f = prepared.solve_on(ctx, &fine, None, state, false)?.0;
            let c = prepared.solve_on(ctx, &coarse, None, state, false)?.0;
            Ok((4. * f - c) / 3.)
        })
        .collect()
}

/// The lowest `states` eigenvalues of a channel on the box its occupied waves ask for.
pub fn spectrum(
    ctx: &Context,
    smooth: &Smooth,
    channel: &Channel,
    states: usize,
) -> Result<Vec<f64>> {
    let prepared = prepared(ctx, smooth, channel)?;
    eigenvalues(ctx, &prepared, extent(smooth, channel)?, states)
}

/// The levels the ghost test reads: two, three when the channel has two occupied references
/// (a semicore state below the valence one).
pub fn ghost_levels(ctx: &Context, smooth: &Smooth, channel: &Channel) -> Result<Vec<f64>> {
    let occupied = channel.occupations.iter().filter(|&&o| o > 0.).count();
    spectrum(ctx, smooth, channel, if occupied > 1 { 3 } else { 2 })
}

/// The lowest level of a `j` branch minus its Dirac reference energy (Hartree): one grid
/// ([`BRANCH_SPACING`], out to [`BRANCH_REACH`]), no extrapolation, as the generator measures it.
pub fn branch_level_error(ctx: &Context, smooth: &Smooth, channel: &Channel) -> Result<f64> {
    let prepared = prepared(ctx, smooth, channel)?;
    // `np.arange(1, int(22 / 0.004))`: the end is exclusive.
    let count = (BRANCH_REACH / BRANCH_SPACING) as usize;
    let grid = (1..count)
        .map(|i| i as f64 * BRANCH_SPACING)
        .collect::<Vec<_>>();
    let level = prepared.solve_on(ctx, &grid, None, 0, false)?.0;
    Ok(level - channel.energies[0])
}
