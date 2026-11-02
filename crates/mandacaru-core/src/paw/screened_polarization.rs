//! The spin dependence of the spheres' semilocal short-range exchange.
//!
//! The frozen one-center term of a screened hybrid, `X0 + V . (D - D0)`
//! (see [`super::screened_atom`]), is first order in the TOTAL density about the
//! unpolarized reference atom: it hands both spins the same potential. The
//! exchange functional is not that simple. Its exact spin-scaling relation,
//! `E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn]) / 2`, makes the polarized
//! functional differ from the unpolarized one at the total density by a
//! quantity that is even in the magnetization and second order in it:
//!
//! `P(D_up, D_dn) = (F[2 n_up] + F[2 n_dn]) / 2 - F[n_up + n_dn]`,
//!
//! where `F` is the short-range hole-model exchange of the all-electron partial
//! waves minus that of the smooth ones, and `n_s` is the spherical average of
//! the sphere's spin density. `P` vanishes for `D_up = D_dn`, and so does its
//! derivative, so a closed shell keeps the frozen term unchanged; an open shell
//! gains the polarization of the spheres. It is evaluated exactly (not
//! expanded) on the current density matrices, with its own potential, so the
//! Fock matrices stay the derivative of the energy.
use super::{
    invalid,
    onecenter_radial::{gradient_first_edge, trapezoid},
    onecenter_terms::nodes_inside,
    pair_tables::{check, Projector, SpeciesView},
};
use crate::{models::xc::exchange_hole_batch, Context, Result};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// The product of two partial waves of one channel, all-electron and smooth,
/// with the radial derivative of each.
struct PairTable {
    ae: Vec<f64>,
    ps: Vec<f64>,
    d_ae: Vec<f64>,
    d_ps: Vec<f64>,
}

/// One entry `(a, b)` of the projector layout contributing to a pair table.
#[derive(Clone, Copy)]
struct Entry {
    a: usize,
    b: usize,
    table: usize,
}

/// The radial tables of one species' sphere, at one screening.
pub struct Polarization {
    omega: f64,
    /// The radial nodes up to the end of the integration (one more than `inside`,
    /// for the slope at the last node).
    r: Vec<f64>,
    /// The nodes the energies and potentials are integrated over.
    inside: usize,
    tables: Vec<PairTable>,
    entries: Vec<Entry>,
    size: usize,
}

/// What the polarization of one sphere adds to a screened hybrid's semilocal
/// short-range term (before the `-fraction` of the hybrid).
pub struct PolarizationTerms {
    /// `P(D_up, D_dn)`: zero for equal spin densities.
    pub energy: f64,
    /// `F[n_up + n_dn]`, the unpolarized short-range exchange of the sphere's
    /// current total density (all-electron minus smooth), for comparison with the
    /// frozen linearization.
    pub unpolarized: f64,
    /// `dP/dD_up` and `dP/dD_dn` in the projector layout.
    pub up: DMatrix<f64>,
    pub down: DMatrix<f64>,
}

/// A radial field and its short-range hole-model exchange: energy and the
/// two potential pieces (`d/dn` and `2 d/dsigma . n'`).
struct Exchange {
    energy: f64,
    local: Vec<f64>,
    gradient: Vec<f64>,
}

impl Polarization {
    /// The tables of `species` for the projector `layout`.
    pub fn new(species: &SpeciesView<'_>, layout: &[Projector], omega: f64) -> Result<Self> {
        check(species)?;
        let radius = species
            .channels
            .iter()
            .map(|c| c.radius)
            .fold(f64::NEG_INFINITY, f64::max);
        // The nodes of the sphere, and one more: the slope of the density at the
        // node on the sphere's edge sees the last smooth-wave node inside it, so
        // the all-electron and smooth densities agree only from the next node on,
        // where the integrand of their difference vanishes.
        let inside = (nodes_inside(species.knots, radius) + 1).min(species.knots.len() - 1);
        let upto = (inside + 1).min(species.knots.len());
        if inside < 3 || upto < 3 {
            return Err(invalid("a PAW sphere needs at least three radial nodes"));
        }
        let r = species.knots[..upto].to_vec();
        let mut tables = Vec::new();
        let mut slot = Vec::new();
        for (channel, view) in species.channels.iter().enumerate() {
            let waves = view.ae.len();
            for i in 0..waves {
                for j in 0..waves {
                    let ae: Vec<f64> = (0..upto).map(|k| view.ae[i][k] * view.ae[j][k]).collect();
                    let ps: Vec<f64> = (0..upto).map(|k| view.ps[i][k] * view.ps[j][k]).collect();
                    tables.push(PairTable {
                        d_ae: gradient_first_edge(&r, &ae)?,
                        d_ps: gradient_first_edge(&r, &ps)?,
                        ae,
                        ps,
                    });
                    slot.push((channel, i, j));
                }
            }
        }
        let mut entries = Vec::new();
        for (a, pa) in layout.iter().enumerate() {
            for (b, pb) in layout.iter().enumerate() {
                if pa.channel == pb.channel && pa.m == pb.m {
                    let table = slot
                        .iter()
                        .position(|s| *s == (pa.channel, pa.radial, pb.radial))
                        .ok_or_else(|| invalid("a projector has no partial-wave pair table"))?;
                    entries.push(Entry { a, b, table });
                }
            }
        }
        Ok(Self {
            omega,
            r,
            inside,
            tables,
            entries,
            size: layout.len(),
        })
    }

    /// The spherical-average spin density `sum_ij (sum_m D_ij) w_i w_j / 4 pi` of
    /// the sphere, all-electron or smooth, from `d` in the projector layout.
    fn density(&self, d: &DMatrix<f64>, smooth: bool) -> Vec<f64> {
        let mut sums = vec![0.; self.tables.len()];
        for e in &self.entries {
            sums[e.table] += d[(e.a, e.b)];
        }
        let mut out = vec![0.; self.r.len()];
        for (sum, table) in sums.iter().zip(&self.tables) {
            let pair = if smooth { &table.ps } else { &table.ae };
            for (slot, value) in out.iter_mut().zip(pair) {
                *slot += sum * value / (4. * PI);
            }
        }
        for value in &mut out {
            *value = value.max(0.);
        }
        out
    }

    fn exchange(&self, ctx: &Context, density: &[f64]) -> Result<Exchange> {
        let slope = gradient_first_edge(&self.r, density)?;
        let values: Vec<f64> = density
            .iter()
            .zip(&slope)
            .flat_map(|(n, s)| [*n, s * s])
            .collect();
        let hole = exchange_hole_batch(ctx, &values, self.omega)?;
        let hole = hole.as_slice();
        let n = self.inside;
        let integrand: Vec<f64> = (0..n)
            .map(|k| hole[3 * k] * 4. * PI * self.r[k] * self.r[k])
            .collect();
        Ok(Exchange {
            energy: trapezoid(&self.r[..n], &integrand),
            local: (0..n).map(|k| hole[3 * k + 1]).collect(),
            gradient: (0..n).map(|k| 2. * hole[3 * k + 2] * slope[k]).collect(),
        })
    }

    /// `P`, `F[n_up + n_dn]` and the derivatives of `P` for the spin density
    /// matrices `up` and `down` in the projector layout (the electrons of each
    /// spin, summed over `m` and over the sphere's channels).
    pub fn evaluate(
        &self,
        ctx: &Context,
        up: &DMatrix<f64>,
        down: &DMatrix<f64>,
    ) -> Result<PolarizationTerms> {
        if up.shape() != (self.size, self.size) || down.shape() != (self.size, self.size) {
            return Err(invalid(
                "a spin density does not match the projector layout",
            ));
        }
        let mut energy = 0.;
        let mut unpolarized = 0.;
        let mut potentials = [vec![0.; self.tables.len()], vec![0.; self.tables.len()]];
        for (smooth, sign) in [(false, 1.), (true, -1.)] {
            let (n_up, n_down) = (self.density(up, smooth), self.density(down, smooth));
            let doubled = |n: &[f64]| n.iter().map(|v| 2. * v).collect::<Vec<_>>();
            let total: Vec<f64> = n_up.iter().zip(&n_down).map(|(a, b)| a + b).collect();
            let fields = [
                self.exchange(ctx, &doubled(&n_up))?,
                self.exchange(ctx, &doubled(&n_down))?,
                self.exchange(ctx, &total)?,
            ];
            energy += sign * (0.5 * (fields[0].energy + fields[1].energy) - fields[2].energy);
            unpolarized += sign * fields[2].energy;
            for spin in 0..2 {
                let (scaled, whole) = (&fields[spin], &fields[2]);
                for (value, table) in potentials[spin].iter_mut().zip(&self.tables) {
                    let (pair, slope) = if smooth {
                        (&table.ps, &table.d_ps)
                    } else {
                        (&table.ae, &table.d_ae)
                    };
                    let integrand: Vec<f64> = (0..self.inside)
                        .map(|k| {
                            ((scaled.local[k] - whole.local[k]) * pair[k]
                                + (scaled.gradient[k] - whole.gradient[k]) * slope[k])
                                * self.r[k]
                                * self.r[k]
                        })
                        .collect();
                    *value += sign * trapezoid(&self.r[..self.inside], &integrand);
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
        Ok(PolarizationTerms {
            energy,
            unpolarized,
            up: matrix(&potentials[0]),
            down: matrix(&potentials[1]),
        })
    }
}
