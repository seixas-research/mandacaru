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
        let knots = ctx.copy_slice(self.r.as_slice())?;
        let values = ctx.copy_slice(self.v_local.as_slice())?;
        Ok(LocalPotential {
            spline: Cubic::new(ctx, knots, values)?,
            first: self.r.as_slice()[0],
            last: *self
                .r
                .as_slice()
                .last()
                .expect("a dataset grid is never empty"),
            charge: self.valence_charge,
        })
    }
}

impl LocalPotential {
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
