//! The ionic local potential of a dataset between its table points.
//!
//! A not-a-knot cubic spline of `v_local` (the interpolant of
//! `scipy.interpolate.CubicSpline`, so the potential and its slope are C^2 and
//! the PAW-LCAO force has no kink at a table point). Below the first point the
//! potential is flat (slope 0); beyond the last it is the ionic tail
//! `-Z/r` (slope `Z/r^2`), `Z` being the valence charge.

use super::PawDataset;
use crate::math::spline::Cubic;
use crate::{Buffer, Context, Result};

/// Smallest radius the tail divides by.
const TAIL_FLOOR: f64 = 1e-12;

#[derive(Debug)]
pub struct LocalPotential {
    spline: Cubic,
    first: f64,
    last: f64,
    charge: f64,
}

impl PawDataset {
    /// The local potential `v_local(r)` as a callable, built on copies of the
    /// table reserved from `ctx`.
    pub fn local_potential(&self, ctx: &Context) -> Result<LocalPotential> {
        LocalPotential::from_table(
            ctx,
            self.r.as_slice(),
            self.v_local.as_slice(),
            self.valence_charge,
        )
    }
}

impl LocalPotential {
    /// The potential of a table `v(r)` (strictly increasing `r`, at least four points) whose
    /// ionic charge is `charge`: the callable a record builds, for a caller that holds the
    /// table without the record.
    pub fn from_table(ctx: &Context, r: &[f64], v: &[f64], charge: f64) -> Result<Self> {
        let (first, last) = match (r.first(), r.last()) {
            (Some(&a), Some(&b)) => (a, b),
            _ => return Err(crate::paw::invalid("the local potential needs a table")),
        };
        Ok(Self {
            spline: Cubic::new(ctx, ctx.copy_slice(r)?, ctx.copy_slice(v)?)?,
            first,
            last,
            charge,
        })
    }

    /// `v_local` at `radius` (Bohr).
    pub fn value(&self, radius: f64) -> f64 {
        if radius <= self.last {
            self.spline.value_slope(radius.max(self.first)).0
        } else {
            -self.charge / radius.max(TAIL_FLOOR)
        }
    }

    /// `d v_local / dr` at `radius`: zero below the first point.
    pub fn slope(&self, radius: f64) -> f64 {
        if radius <= self.last {
            if radius < self.first {
                0.0
            } else {
                self.spline.value_slope(radius).1
            }
        } else {
            self.charge / radius.max(TAIL_FLOOR).powi(2)
        }
    }

    /// [`LocalPotential::value`] and [`LocalPotential::slope`] at every radius,
    /// into budgeted buffers.
    pub fn evaluate(&self, ctx: &Context, radii: &[f64]) -> Result<(Buffer<f64>, Buffer<f64>)> {
        let mut values = ctx.allocate::<f64>(radii.len())?;
        let mut slopes = ctx.allocate::<f64>(radii.len())?;
        for ((v, s), radius) in values
            .as_mut_slice()
            .iter_mut()
            .zip(slopes.as_mut_slice())
            .zip(radii)
        {
            *v = self.value(*radius);
            *s = self.slope(*radius);
        }
        Ok((values, slopes))
    }
}
