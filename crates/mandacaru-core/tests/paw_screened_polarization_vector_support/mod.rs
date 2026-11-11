//! Shared by `paw_screened_polarization_vector*`: a toy species (two channels of two partial waves
//! on a uniform grid), Pauli-component density matrices in its projector layout and an independent
//! oracle of the non-collinear remainder. The oracle rebuilds the radial fields from the waves,
//! applies the local-rotation prescription pointwise on the whole grid and calls the hole model
//! directly: it shares no code with `Polarization::evaluate_vector`.
#![allow(dead_code)]

use mandacaru_core::{
    models::xc::exchange_hole_batch,
    paw::{
        onecenter_radial::{gradient_first_edge, trapezoid},
        pair_tables::{layout, ChannelView, Projector, SpeciesView},
        screened_polarization::Polarization,
    },
    Context,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

pub const RADIUS: f64 = 1.;
pub const OMEGA: f64 = 0.11;
pub type Pauli = [DMatrix<f64>; 4];

pub struct Waves {
    pub knots: Vec<f64>,
    pub ae: Vec<Vec<Vec<f64>>>,
    pub ps: Vec<Vec<Vec<f64>>>,
    pub reference: Vec<DMatrix<f64>>,
    /// The smooth core of the nonlinear core correction (zeros: none).
    pub core: Vec<f64>,
}

/// The smooth waves are the all-electron ones outside the sphere and a softened copy inside it.
pub fn waves() -> Waves {
    let knots: Vec<f64> = (1..=700).map(|k| 0.02 * k as f64).collect();
    let (mut ae, mut ps) = (Vec::new(), Vec::new());
    for l in 0..2usize {
        let channel: Vec<Vec<f64>> = vec![
            knots
                .iter()
                .map(|&r| 1.5 * r.powi(l as i32) * (-r).exp())
                .collect(),
            knots
                .iter()
                .map(|&r| 1.1 * r.powi(l as i32) * (-0.6 * r).exp())
                .collect(),
        ];
        ps.push(
            channel
                .iter()
                .map(|wave| {
                    knots
                        .iter()
                        .zip(wave)
                        .map(|(&r, &w)| {
                            let x = (r / RADIUS).min(1.);
                            w * (1. - 0.6 * (1. - x * x).powi(2))
                        })
                        .collect()
                })
                .collect(),
        );
        ae.push(channel);
    }
    let reference = vec![
        DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.1, 0.15]),
        DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.2]) / 3.,
    ];
    let core = vec![0.; knots.len()];
    Waves {
        knots,
        ae,
        ps,
        reference,
        core,
    }
}

pub fn view(w: &Waves) -> SpeciesView<'_> {
    SpeciesView {
        knots: &w.knots,
        smooth_core: &w.core,
        compensation_radius: RADIUS,
        channels: (0..2)
            .map(|l| ChannelView {
                l,
                radius: RADIUS,
                ae: w.ae[l].iter().map(Vec::as_slice).collect(),
                ps: w.ps[l].iter().map(Vec::as_slice).collect(),
                reference: &w.reference[l],
            })
            .collect(),
    }
}

pub fn context() -> Context {
    Context::new(Some(1), 512 << 20).unwrap()
}

/// The toy sphere: waves, projector layout and the polarization tables.
pub struct Toy {
    pub waves: Waves,
    pub projectors: Vec<Projector>,
    pub polarization: Polarization,
}
pub fn toy() -> Toy {
    toy_with(waves())
}
/// The toy species with a smooth core on the smooth side of the sphere, as a nonlinear core
/// correction places it: a bump of peak density `peak` that vanishes beyond 0.9 Bohr, so that the
/// whole-grid oracle and the sphere's own integration extent see the same difference.
pub fn toy_with_core(peak: f64) -> Toy {
    let mut waves = waves();
    waves.core = waves
        .knots
        .iter()
        .map(|r| {
            if *r < 0.9 {
                peak * (1. - (r / 0.9).powi(2)).powi(3)
            } else {
                0.
            }
        })
        .collect();
    toy_with(waves)
}
fn toy_with(waves: Waves) -> Toy {
    let projectors = layout(&view(&waves));
    let polarization = Polarization::new(&view(&waves), &projectors, OMEGA).unwrap();
    Toy {
        waves,
        projectors,
        polarization,
    }
}

impl Toy {
    pub fn size(&self) -> usize {
        self.projectors.len()
    }
    /// The reference density matrix in the projector layout (diagonal in `m`).
    pub fn reference(&self) -> DMatrix<f64> {
        let p = self.size();
        DMatrix::from_fn(p, p, |a, b| {
            let (pa, pb) = (&self.projectors[a], &self.projectors[b]);
            if pa.channel == pb.channel && pa.m == pb.m {
                self.waves.reference[pa.channel][(pa.radial, pb.radial)]
            } else {
                0.
            }
        })
    }
    /// The reference pattern, each element weighted by `0.5 + sin(seed (3 i + 7 j + 1))`
    /// (symmetric, sign-varying, so that the radial shape differs from the reference's).
    pub fn varied(&self, seed: f64) -> DMatrix<f64> {
        let d0 = self.reference();
        DMatrix::from_fn(self.size(), self.size(), |a, b| {
            let (i, j) = (a.min(b) as f64, a.max(b) as f64);
            d0[(a, b)] * (0.5 + (seed * (3. * i + 7. * j + 1.)).sin())
        })
    }
    /// A magnetized sphere: charge `1.0 d0` and a vector of Pauli components of the sizes
    /// `amplitude`, each with its own radial shape.
    pub fn pauli(&self, amplitude: [f64; 3]) -> Pauli {
        [
            self.reference(),
            self.varied(0.37) * amplitude[0],
            self.varied(0.91) * amplitude[1],
            self.varied(1.43) * amplitude[2],
        ]
    }
    /// `q + e dq` for every component.
    pub fn shifted(&self, q: &Pauli, dq: &Pauli, e: f64) -> Pauli {
        std::array::from_fn(|a| &q[a] + &dq[a] * e)
    }
}

pub fn scaled(q: &Pauli, t: f64) -> Pauli {
    std::array::from_fn(|a| if a == 0 { q[0].clone() } else { &q[a] * t })
}

/// `sum_a tr(O^a dq^a)`.
pub fn contract(o: &Pauli, dq: &Pauli) -> f64 {
    (0..4).map(|a| o[a].component_mul(&dq[a]).sum()).sum()
}

/// The form of the third term of the oracle.
#[derive(Clone, Copy, PartialEq)]
pub enum Total {
    /// `F[max(n_0, 0)]`: the unclamped total.
    Unclamped,
    /// `F[n_+ + n_-]` of the clamped branches (the form of the collinear `evaluate`).
    ClampedSum,
}

/// What the oracle saw on the grid.
pub struct Seen {
    pub energy: f64,
    /// Nodes (inside the sphere) where a branch `n_pm` was clamped.
    pub clamped: usize,
    /// Sign changes of `m_z` along the radius (over the nodes where `|m_z| > 1e-10`).
    pub sign_changes: usize,
}

/// `P` from the radial fields, all-electron minus smooth, on the whole grid.
pub fn oracle(ctx: &Context, toy: &Toy, q: &Pauli, total: Total) -> Seen {
    let w = &toy.waves;
    let mut seen = Seen {
        energy: 0.,
        clamped: 0,
        sign_changes: 0,
    };
    for (waves, sign) in [(&w.ae, 1.), (&w.ps, -1.)] {
        let field = |a: usize| -> Vec<f64> {
            (0..w.knots.len())
                .map(|k| {
                    let mut value = 0.;
                    for (i, pi) in toy.projectors.iter().enumerate() {
                        for (j, pj) in toy.projectors.iter().enumerate() {
                            if pi.channel == pj.channel && pi.m == pj.m {
                                value += q[a][(i, j)]
                                    * waves[pi.channel][pi.radial][k]
                                    * waves[pj.channel][pj.radial][k]
                                    / (4. * PI);
                            }
                        }
                    }
                    value
                })
                .collect()
        };
        let (mut n0, mx, my, mz) = (field(0), field(1), field(2), field(3));
        if sign < 0. {
            // The grid's smooth core is charge, never magnetization.
            for (n, core) in n0.iter_mut().zip(&w.core) {
                *n += core;
            }
        }
        let nodes = n0.len();
        let mag: Vec<f64> = (0..nodes)
            .map(|k| (mx[k] * mx[k] + my[k] * my[k] + mz[k] * mz[k]).sqrt())
            .collect();
        let plus: Vec<f64> = (0..nodes)
            .map(|k| ((n0[k] + mag[k]) / 2.).max(0.))
            .collect();
        let minus: Vec<f64> = (0..nodes)
            .map(|k| ((n0[k] - mag[k]) / 2.).max(0.))
            .collect();
        let whole: Vec<f64> = match total {
            Total::Unclamped => n0.iter().map(|v| v.max(0.)).collect(),
            Total::ClampedSum => (0..nodes).map(|k| plus[k] + minus[k]).collect(),
        };
        let exchange = |n: &[f64]| -> f64 {
            let slope = gradient_first_edge(&w.knots, n).unwrap();
            let values: Vec<f64> = n
                .iter()
                .zip(&slope)
                .flat_map(|(n, s)| [*n, s * s])
                .collect();
            let hole = exchange_hole_batch(ctx, &values, OMEGA).unwrap();
            let integrand: Vec<f64> = (0..nodes)
                .map(|k| hole.as_slice()[3 * k] * 4. * PI * w.knots[k] * w.knots[k])
                .collect();
            trapezoid(&w.knots, &integrand)
        };
        let double = |n: &[f64]| n.iter().map(|v| 2. * v).collect::<Vec<_>>();
        seen.energy += sign
            * (0.5 * (exchange(&double(&plus)) + exchange(&double(&minus))) - exchange(&whole));
        if !(sign < 0.) {
            let inside = w.knots.iter().take_while(|r| **r <= RADIUS).count();
            seen.clamped = (0..inside).filter(|&k| n0[k] - mag[k] < 0.).count();
            let signs: Vec<f64> = mz.iter().filter(|v| v.abs() > 1e-10).copied().collect();
            seen.sign_changes = signs.windows(2).filter(|s| s[0] * s[1] < 0.).count();
        }
    }
    seen
}

/// Fourth-order central difference of `f` along `dq` at `q`.
pub fn derivative(toy: &Toy, q: &Pauli, dq: &Pauli, step: f64, f: &dyn Fn(&Pauli) -> f64) -> f64 {
    let at = |e: f64| f(&toy.shifted(q, dq, e));
    (-at(2. * step) + 8. * at(step) - 8. * at(-step) + at(-2. * step)) / (12. * step)
}
