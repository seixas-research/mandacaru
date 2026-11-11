//! The non-collinear remainder of the spheres' semilocal short-range exchange.
//!
//! A magnetic spinor crystal has a vector magnetization `m = (n_1, n_2, n_3)` in every sphere,
//! not a pair of spin densities. The grid's own functional follows the local-rotation
//! prescription (`periodic::spinor::exchange_correlation`): at every point rotate to the axis of
//! `m`, so the spin densities are `n_pm = (n_0 +- |m|) / 2`. The sphere term is the AE-minus-smooth
//! difference of that same functional, so it must use the same prescription:
//!
//! ```text
//! P = (F[2 n_+] + F[2 n_-]) / 2 - F[n_0],    n_pm = max((n_0 +- |m|) / 2, 0),    n_0 -> max(n_0, 0)
//! ```
//!
//! The total in the third term is the UNCLAMPED `n_0` (clamped only at the density floor), so that
//! the potential below is the exact derivative of this energy also where a `n_pm` clamp is active.
//! The inputs are the four Pauli components `Q^a` of the projected density (`a = 0` charge,
//! `1..3` the magnetization), real parts, in the projector layout: `n_a(r) = sum_t S^a_t pair_t(r) /
//! 4 pi` with `S^a_t` the sum of `Q^a[i, j]` over the entries of table `t`, exactly as the
//! collinear [`Polarization::evaluate`] builds a spin density.
//!
//! The operators `O^a = dP / dQ^a` carry the same product-table slope as the density does: the
//! gradient part of `O^a` uses the slope of `c * mhat_a * pair`, not `mhat_a' pair + mhat_a pair'`,
//! which makes the operator the derivative of the discrete energy to round-off. `mhat = m / |m|`
//! is zero where `|m| <= 1e-12 n_0`, and `|m|` itself is zeroed there, so a density without
//! magnetization has `n_+ = n_- = n_0 / 2` and `P = 0`, `O = 0` exactly. The lift to the `2p x 2p`
//! spinor operator is in `periodic::screened_onecenter`.
use super::{Exchange, Polarization};
use crate::{
    paw::{
        invalid,
        onecenter_radial::{gradient_first_edge, trapezoid},
    },
    Context, Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// `|m| <= MAGNETIZATION_FLOOR n_0` counts as no magnetization at a node.
const MAGNETIZATION_FLOOR: f64 = 1e-12;

/// What the non-collinear remainder adds to a screened hybrid's semilocal short-range term
/// (before the `-fraction` of the hybrid).
pub struct VectorTerms {
    /// `P`: zero for a density without magnetization.
    pub energy: f64,
    /// `F[max(n_0, 0)]`, the unpolarized short-range exchange of the sphere's total density
    /// (all-electron minus smooth plus the smooth core).
    pub unpolarized: f64,
    /// `O^0, O^1, O^2, O^3 = dP / dQ^a` (real, symmetric) in the projector layout.
    pub operators: [DMatrix<f64>; 4],
}

/// The radial fields of one density (all-electron or smooth) and the clamp indicators.
struct Local {
    /// `mhat_a(r)`.
    direction: [Vec<f64>; 3],
    /// `max((n_0 +- |m|) / 2, 0)` and the indicator of the unclamped branch.
    plus: (Vec<f64>, Vec<f64>),
    minus: (Vec<f64>, Vec<f64>),
    /// `max(n_0, 0)` and its indicator.
    total: (Vec<f64>, Vec<f64>),
}

impl Polarization {
    /// `S^a_t`: the sum of `q[(i, j)]` over the entries `(i, j)` of each pair table.
    fn table_sums(&self, q: &DMatrix<f64>) -> Vec<f64> {
        let mut sums = vec![0.; self.tables.len()];
        for e in &self.entries {
            sums[e.table] += q[(e.a, e.b)];
        }
        sums
    }

    /// The unclamped radial field `sum_t sums_t pair_t(r) / 4 pi`.
    fn field(&self, sums: &[f64], smooth: bool) -> Vec<f64> {
        let mut out = vec![0.; self.r.len()];
        for (sum, table) in sums.iter().zip(&self.tables) {
            let pair = if smooth { &table.ps } else { &table.ae };
            for (slot, value) in out.iter_mut().zip(pair) {
                *slot += sum * value / (4. * PI);
            }
        }
        out
    }

    fn local(&self, sums: &[Vec<f64>; 4], smooth: bool) -> Local {
        let mut n0 = self.field(&sums[0], smooth);
        if smooth {
            // The grid's smooth core is part of the charge, never of the magnetization.
            for (slot, core) in n0.iter_mut().zip(&self.smooth_core) {
                *slot += core;
            }
        }
        let m: [Vec<f64>; 3] = std::array::from_fn(|a| self.field(&sums[a + 1], smooth));
        let nodes = n0.len();
        let mut magnitude = vec![0.; nodes];
        for k in 0..nodes {
            let norm = (m[0][k] * m[0][k] + m[1][k] * m[1][k] + m[2][k] * m[2][k]).sqrt();
            if norm > MAGNETIZATION_FLOOR * n0[k].max(0.) {
                magnitude[k] = norm;
            }
        }
        let branch = |sign: f64| -> (Vec<f64>, Vec<f64>) {
            let raw: Vec<f64> = (0..nodes)
                .map(|k| (n0[k] + sign * magnitude[k]) / 2.)
                .collect();
            (
                raw.iter().map(|v| v.max(0.)).collect(),
                raw.iter().map(|v| f64::from(*v > 0.)).collect(),
            )
        };
        Local {
            direction: std::array::from_fn(|a| {
                (0..nodes)
                    .map(|k| {
                        if magnitude[k] > 0. {
                            m[a][k] / magnitude[k]
                        } else {
                            0.
                        }
                    })
                    .collect()
            }),
            plus: branch(1.),
            minus: branch(-1.),
            total: (
                n0.iter().map(|v| v.max(0.)).collect(),
                n0.iter().map(|v| f64::from(*v > 0.)).collect(),
            ),
        }
    }

    /// `int r^2 [local * y + gradient * y'] dr` over the sphere, the slope being the one the
    /// density uses.
    fn response(&self, field: &Exchange, y: &[f64]) -> Result<f64> {
        let slope = gradient_first_edge(&self.r, y)?;
        let integrand: Vec<f64> = (0..self.inside)
            .map(|k| (field.local[k] * y[k] + field.gradient[k] * slope[k]) * self.r[k] * self.r[k])
            .collect();
        Ok(trapezoid(&self.r[..self.inside], &integrand))
    }

    fn check_components(&self, q: &[DMatrix<f64>; 4]) -> Result<()> {
        if q.iter().any(|m| m.shape() != (self.size, self.size)) {
            return Err(invalid(
                "a Pauli component of the projected density does not match the projector layout",
            ));
        }
        Ok(())
    }

    /// `P`, `F[n_0]` and `O^a = dP/dQ^a` for the real parts `q[a]` of the four Pauli components
    /// of the sphere's projected density (`Q^0 = Q_uu + Q_dd`, `Q^3 = Q_uu - Q_dd`, ...).
    pub fn evaluate_vector(&self, ctx: &Context, q: &[DMatrix<f64>; 4]) -> Result<VectorTerms> {
        self.check_components(q)?;
        let sums: [Vec<f64>; 4] = std::array::from_fn(|a| self.table_sums(&q[a]));
        let mut energy = 0.;
        let mut unpolarized = 0.;
        let mut potentials = vec![vec![0.; self.tables.len()]; 4];
        for (smooth, sign) in [(false, 1.), (true, -1.)] {
            let local = self.local(&sums, smooth);
            let doubled = |n: &[f64]| n.iter().map(|v| 2. * v).collect::<Vec<_>>();
            let plus = self.exchange(ctx, &doubled(&local.plus.0))?;
            let minus = self.exchange(ctx, &doubled(&local.minus.0))?;
            let total = self.exchange(ctx, &local.total.0)?;
            energy += sign * (0.5 * (plus.energy + minus.energy) - total.energy);
            unpolarized += sign * total.energy;
            for (t, table) in self.tables.iter().enumerate() {
                let pair = if smooth { &table.ps } else { &table.ae };
                let weighted = |indicator: &[f64], direction: Option<&[f64]>| -> Vec<f64> {
                    (0..self.r.len())
                        .map(|k| indicator[k] * direction.map_or(1., |d| d[k]) * pair[k])
                        .collect()
                };
                let charge = [
                    (0.5, &plus, &local.plus.1),
                    (0.5, &minus, &local.minus.1),
                    (-1., &total, &local.total.1),
                ];
                for (weight, field, indicator) in charge {
                    potentials[0][t] +=
                        sign * weight * self.response(field, &weighted(indicator, None))?;
                }
                for a in 0..3 {
                    let direction = Some(local.direction[a].as_slice());
                    let up = self.response(&plus, &weighted(&local.plus.1, direction))?;
                    let down = self.response(&minus, &weighted(&local.minus.1, direction))?;
                    potentials[a + 1][t] += sign * 0.5 * (up - down);
                }
            }
        }
        let matrix = |values: &[f64]| {
            let mut out = DMatrix::zeros(self.size, self.size);
            for e in &self.entries {
                out[(e.a, e.b)] = values[e.table];
            }
            out
        };
        Ok(VectorTerms {
            energy,
            unpolarized,
            operators: std::array::from_fn(|a| matrix(&potentials[a])),
        })
    }

    /// The spherical-average magnetization of the sphere relative to its charge,
    /// `int r^2 |m| dr / int r^2 n_0 dr` on the all-electron density, from the real parts of the
    /// Pauli components. It is zero (to round-off) for a time-reversal-symmetric density, whatever
    /// the cross-spin blocks `Q_ud` hold: the cross block carries the spin-orbit mixing, not a
    /// moment.
    pub fn magnetization_ratio(&self, q: &[DMatrix<f64>; 4]) -> Result<f64> {
        self.check_components(q)?;
        let n0 = self.field(&self.table_sums(&q[0]), false);
        let m: [Vec<f64>; 3] =
            std::array::from_fn(|a| self.field(&self.table_sums(&q[a + 1]), false));
        let nodes = &self.r[..self.inside];
        let weight = |f: &dyn Fn(usize) -> f64| {
            let integrand: Vec<f64> = (0..self.inside)
                .map(|k| f(k) * self.r[k] * self.r[k])
                .collect();
            trapezoid(nodes, &integrand)
        };
        let moment = weight(&|k| (m[0][k].powi(2) + m[1][k].powi(2) + m[2][k].powi(2)).sqrt());
        let charge = weight(&|k| n0[k].max(0.));
        Ok(if moment == 0. {
            0.
        } else if charge > 0. {
            moment / charge
        } else {
            f64::INFINITY
        })
    }
}
