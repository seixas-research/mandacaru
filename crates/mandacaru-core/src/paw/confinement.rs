//! Confined pseudo-atomic orbitals of a PAW channel: the generalized radial
//! problem on two uniform grids with a Richardson extrapolation, the cutoff
//! radius that realizes an `energy_shift`, and the orbital tabulated on the
//! dataset's own grid. The potential and the grids are `basis::confinement`.
use super::{invalid, spectrum};
use crate::{
    basis::confinement::{
        grid, potential, Potential, BRACKET_GROWTH, RADIUS_TOLERANCE, TABLE_FRACTION,
    },
    math::{roots::brent, spline::Cubic},
    product,
    units::HARTREE_TO_EV,
    Buffer, Context, Result,
};

/// One angular-momentum channel of a PAW dataset, as arrays on the dataset grid.
#[derive(Clone, Copy)]
pub struct ChannelData<'a> {
    /// The dataset's radial grid.
    pub table: &'a [f64],
    /// The screened local potential on `table`.
    pub local: &'a [f64],
    /// Projectors `p(r)` on `table`, row-major `(rank, table.len())`.
    pub projectors: &'a [f64],
    pub rank: usize,
    /// Screened coupling `D` and overlap correction `q`, `(rank, rank)`.
    pub coupling: &'a [f64],
    pub overlap: &'a [f64],
    /// The channel's bound smooth wave on `table`: the sign reference.
    pub wave: &'a [f64],
    pub l: usize,
    /// The channel's cutoff radius.
    pub r_cut: f64,
}

/// The channel's splines, built once and evaluated on any uniform grid.
pub struct Prepared<'a> {
    data: ChannelData<'a>,
    local: Cubic,
    projectors: Vec<Cubic>,
    wave: Cubic,
}

fn spline(ctx: &Context, knots: &[f64], values: &[f64]) -> Result<Cubic> {
    Cubic::new(ctx, ctx.copy_slice(knots)?, ctx.copy_slice(values)?)
}

impl<'a> Prepared<'a> {
    pub fn new(ctx: &Context, data: ChannelData<'a>) -> Result<Self> {
        let n = data.table.len();
        if data.local.len() != n
            || data.wave.len() != n
            || data.projectors.len() != product(&[data.rank, n])?
            || data.coupling.len() != product(&[data.rank, data.rank])?
            || data.overlap.len() != product(&[data.rank, data.rank])?
            || !data.r_cut.is_finite()
            || data.r_cut <= 0.
        {
            return Err(invalid("invalid PAW channel arrays for a confined orbital"));
        }
        Ok(Self {
            local: spline(ctx, data.table, data.local)?,
            projectors: (0..data.rank)
                .map(|j| spline(ctx, data.table, &data.projectors[j * n..(j + 1) * n]))
                .collect::<Result<_>>()?,
            wave: spline(ctx, data.table, data.wave)?,
            data,
        })
    }
    /// Last point of the dataset grid.
    pub fn table_end(&self) -> f64 {
        *self.data.table.last().expect("validated table")
    }
    /// The potential (with the centrifugal term) and the `u`-form projectors
    /// `p(r) r` on the nodes `r`.
    pub fn operator_on(&self, ctx: &Context, r: &[f64]) -> Result<(Buffer<f64>, Buffer<f64>)> {
        let n = r.len();
        let l = self.data.l as f64;
        let mut v = ctx.allocate::<f64>(n)?;
        let mut p = ctx.allocate::<f64>(product(&[self.data.rank, n])?)?;
        for (i, &x) in r.iter().enumerate() {
            v.as_mut_slice()[i] = self.local.value_slope(x).0 + l * (l + 1.) / (2. * x * x);
        }
        for (j, spline) in self.projectors.iter().enumerate() {
            for (i, &x) in r.iter().enumerate() {
                p.as_mut_slice()[j * n + i] = spline.value_slope(x).0 * x;
            }
        }
        Ok((v, p))
    }
    /// The `state`-th eigenpair of the generalized problem on the uniform
    /// interior nodes `r`, with `extra` added to the potential.
    pub fn solve_on(
        &self,
        ctx: &Context,
        r: &[f64],
        extra: Option<&[f64]>,
        state: usize,
        vectors: bool,
    ) -> Result<(f64, Option<Buffer<f64>>)> {
        let (mut v, p) = self.operator_on(ctx, r)?;
        if let Some(extra) = extra {
            for (v, e) in v.as_mut_slice().iter_mut().zip(extra) {
                *v += e;
            }
        }
        let out = spectrum::eigenpair(
            ctx,
            spectrum::Channel {
                radii: r,
                potential: v.as_slice(),
                projectors: p.as_slice(),
                coupling: self.data.coupling,
                overlap: self.data.overlap,
                rank: self.data.rank,
            },
            state,
            vectors,
        )?;
        Ok((out.energy, out.vector))
    }
    /// `+1` if `u` overlaps the stored bound wave positively on `r`, else `-1`.
    pub fn sign_of(&self, r: &[f64], u: &[f64]) -> f64 {
        let y = |i: usize| u[i] * self.wave.value_slope(r[i]).0 * r[i];
        let h = r[1] - r[0];
        let total: f64 = (1..r.len()).map(|i| h * (y(i) + y(i - 1)) / 2.).sum();
        if total < 0. {
            -1.
        } else {
            1.
        }
    }
}

fn check_wall(prepared: &Prepared, r_c: f64) -> Result<()> {
    if r_c.is_finite() && r_c > 0. && r_c < prepared.table_end() {
        Ok(())
    } else {
        Err(invalid(
            "the confinement radius must lie inside the dataset's radial table",
        ))
    }
}

struct Solved {
    r: Vec<f64>,
    u: Option<Buffer<f64>>,
    energy: f64,
}

fn solve(
    ctx: &Context,
    prepared: &Prepared,
    r_c: f64,
    options: &Potential,
    state: usize,
    vectors: bool,
) -> Result<[Solved; 2]> {
    options.check()?;
    check_wall(prepared, r_c)?;
    let mut out = Vec::with_capacity(2);
    for doubling in 0..2 {
        let r = grid(r_c, doubling);
        let wall = potential(&r, r_c, options.amplitude, options.inner_fraction);
        let (energy, u) = prepared.solve_on(ctx, &r, Some(&wall), state, vectors)?;
        out.push(Solved { r, u, energy });
    }
    let coarse = out.pop().expect("two grids");
    let fine = out.pop().expect("two grids");
    Ok([fine, coarse])
}

/// The `state`-th eigenvalue (Hartree) of the channel confined at `r_c`,
/// Richardson-extrapolated from the two grids.
pub fn energy(
    ctx: &Context,
    prepared: &Prepared,
    r_c: f64,
    options: &Potential,
    state: usize,
) -> Result<f64> {
    let [fine, coarse] = solve(ctx, prepared, r_c, options, state, false)?;
    Ok((4. * fine.energy - coarse.energy) / 3.)
}

/// `(r, u, energy)` of the confined channel's `state`-th eigenstate: `u = r R`
/// on the coarse grid (wall excluded), Richardson-extrapolated pointwise,
/// normalized in the dataset's metric and signed like its bound wave.
pub fn eigenstate(
    ctx: &Context,
    prepared: &Prepared,
    r_c: f64,
    options: &Potential,
    state: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let [fine, coarse] = solve(ctx, prepared, r_c, options, state, true)?;
    let signed = |s: &Solved| {
        let u = s.u.as_ref().expect("vectors requested").as_slice();
        let sign = prepared.sign_of(&s.r, u);
        u.iter().map(|v| sign * v).collect::<Vec<_>>()
    };
    let (u_f, u_c) = (signed(&fine), signed(&coarse));
    let u = (0..u_c.len())
        .map(|i| (4. * u_f[2 * i + 1] - u_c[i]) / 3.)
        .collect();
    Ok((coarse.r, u, (4. * fine.energy - coarse.energy) / 3.))
}

/// Eigenvalue (Hartree) of the channel's free bound state in the dataset's own
/// operator: the confined problem with the wall at the end of the radial table.
pub fn free_energy(ctx: &Context, prepared: &Prepared, state: usize) -> Result<f64> {
    energy(
        ctx,
        prepared,
        TABLE_FRACTION * prepared.table_end(),
        &Potential::default(),
        state,
    )
}

/// What the radius search found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Radius {
    /// The cutoff radius (Bohr) realizing the shift.
    Found(f64),
    /// The shift would cut into the sphere and `tightest` was set: the tightest
    /// wall the sphere allows.
    Tightest(f64),
    /// The shift is too large to reach outside the sphere; the largest usable
    /// shift (eV).
    TooLarge { largest_ev: f64 },
    /// The shift is too small to reach inside the table (whose reach, Bohr).
    TooSmall { reach: f64 },
}

/// Cutoff radius at which the channel's eigenvalue lies `shift_ev` above
/// `free` (Hartree): bracketed by an outward scan from the channel's cutoff
/// radius, refined by Brent's method.
pub fn radius(
    ctx: &Context,
    prepared: &Prepared,
    free: f64,
    shift_ev: f64,
    options: &Potential,
    tightest: bool,
    state: usize,
) -> Result<Radius> {
    if !(shift_ev.is_finite() && shift_ev > 0.) {
        return Err(invalid(
            "the confinement radius needs a positive energy shift",
        ));
    }
    options.check()?;
    let shift = shift_ev * (1. / HARTREE_TO_EV);
    let target = free + shift;
    let excess = |r_c: f64| Ok(energy(ctx, prepared, r_c, options, state)? - target);
    let mut low = prepared.data.r_cut / options.inner_fraction;
    let end = TABLE_FRACTION * prepared.table_end();
    let f_low = excess(low)?;
    if f_low <= 0. {
        return Ok(if tightest {
            Radius::Tightest(low)
        } else {
            Radius::TooLarge {
                largest_ev: (f_low + shift) * HARTREE_TO_EV,
            }
        });
    }
    let mut high = low;
    loop {
        high = (BRACKET_GROWTH * high).min(end);
        if excess(high)? < 0. {
            break;
        }
        if high >= end {
            return Ok(Radius::TooSmall { reach: end });
        }
        low = high;
    }
    Ok(Radius::Found(brent(
        excess,
        low,
        high,
        RADIUS_TOLERANCE,
        4. * f64::EPSILON,
        100,
    )?))
}

/// A confined first zeta on the dataset's grid.
#[derive(Clone, Debug)]
pub struct Orbital {
    pub r_c: f64,
    /// Confined eigenvalue (Hartree).
    pub energy: f64,
    /// Free eigenvalue (Hartree).
    pub free_energy: f64,
    /// `R(r)` on the dataset grid, zero from `r_c` on.
    pub radial: Vec<f64>,
}

/// The orbital of the channel confined so that its level lies `shift_ev` above
/// the free one (`free_energy`, or solved when `None`). A shift too large to
/// reach takes the tightest wall; one too small to reach is an error carrying
/// the table's reach.
pub fn orbital(
    ctx: &Context,
    prepared: &Prepared,
    shift_ev: f64,
    options: &Potential,
    state: usize,
    given_free: Option<f64>,
) -> Result<std::result::Result<Orbital, f64>> {
    let free = match given_free {
        Some(value) => value,
        None => free_energy(ctx, prepared, state)?,
    };
    let r_c = match radius(ctx, prepared, free, shift_ev, options, true, state)? {
        Radius::Found(r) | Radius::Tightest(r) => r,
        Radius::TooSmall { reach } => return Ok(Err(reach)),
        Radius::TooLarge { .. } => unreachable!("the tightest wall was requested"),
    };
    let (r, u, level) = eigenstate(ctx, prepared, r_c, options, state)?;
    // u(0) = u(r_c) = 0 close the table; R = u / r is then regular at the origin.
    let mut knots = Vec::with_capacity(r.len() + 2);
    knots.push(0.);
    knots.extend_from_slice(&r);
    knots.push(r_c);
    let mut values = Vec::with_capacity(u.len() + 2);
    values.push(0.);
    values.extend_from_slice(&u);
    values.push(0.);
    let curve = spline(ctx, &knots, &values)?;
    let radial = prepared
        .data
        .table
        .iter()
        .map(|&x| {
            if x < r_c {
                curve.value_slope(x.min(r_c)).0 / x
            } else {
                0.
            }
        })
        .collect();
    Ok(Ok(Orbital {
        r_c,
        energy: level,
        free_energy: free,
        radial,
    }))
}
