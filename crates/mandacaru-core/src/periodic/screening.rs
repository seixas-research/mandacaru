//! Static independent-particle response and the symmetric RPA Dyson equation.
//! Transition densities include the caller's complete PAW augmentation.
use super::{invalid, spectrum::Smearing};
use crate::{
    math::{
        gemm::{self, MatrixMut, MatrixRef},
        linalg::eigen,
    },
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

pub struct Options {
    pub cell_volume: f64,
    pub kpoints: usize,
    pub fermi: f64,
    pub width: f64,
    pub degeneracy: f64,
    pub smearing: Smearing,
    pub constrained: bool,
}
/// Energy and fractional target-space weight of a single Kohn-Sham state.
#[derive(Clone, Copy)]
pub struct State {
    pub energy: f64,
    pub target: f64,
}
pub struct Polarizability {
    context: Context,
    options: Options,
    vectors: usize,
    count: usize,
    densities: Buffer<C64>,
    weights: Buffer<f64>,
    left: Buffer<f64>,
    right: Buffer<f64>,
    real: Buffer<f64>,
    imaginary: Buffer<f64>,
    _packing: Reservation,
}
pub struct ScreenedInteraction {
    pub inverse_dielectric: DMatrix<C64>,
    pub correction: DMatrix<C64>,
    _memory: Reservation,
}
impl Polarizability {
    pub fn new(ctx: &Context, vectors: usize, options: Options) -> Result<Self> {
        if vectors == 0
            || options.kpoints == 0
            || !options.cell_volume.is_finite()
            || options.cell_volume <= 0.
            || !options.fermi.is_finite()
            || !options.width.is_finite()
            || options.width <= 0.
            || !options.degeneracy.is_finite()
            || options.degeneracy <= 0.
        {
            return Err(invalid("invalid static response dimensions or occupations"));
        }
        let batch = 128;
        Ok(Self {
            context: ctx.clone(),
            options,
            vectors,
            count: 0,
            densities: ctx.allocate(product(&[batch, vectors])?)?,
            weights: ctx.allocate(batch)?,
            left: ctx.allocate(product(&[batch, vectors])?)?,
            right: ctx.allocate(product(&[batch, vectors])?)?,
            real: ctx.allocate(product(&[vectors, vectors])?)?,
            imaginary: ctx.allocate(product(&[vectors, vectors])?)?,
            _packing: ctx.reserve(product(&[batch, vectors, 16])?)?,
        })
    }
    /// Both directions of the band-pair sum must be supplied. Degenerate
    /// transitions take the analytic smearing derivative, including intraband
    /// response; constrained RPA removes the product of target weights.
    pub fn add(&mut self, first: State, second: State, density: &[C64]) -> Result<()> {
        for state in [first, second] {
            if !state.energy.is_finite()
                || !state.target.is_finite()
                || state.target < -1e-10
                || state.target > 1. + 1e-10
            {
                return Err(invalid(
                    "invalid static response energy or target-space weight",
                ));
            }
        }
        if density.len() != self.vectors
            || density
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid(
                "transition density must match the response reciprocal space",
            ));
        }
        let options = &self.options;
        let delta = first.energy - second.energy;
        let mut weight = if delta.abs() < 1e-9 {
            options.degeneracy
                * options.smearing.slope(
                    (0.5 * first.energy + 0.5 * second.energy - options.fermi) / options.width,
                )
                / options.width
        } else {
            options.degeneracy
                * (options
                    .smearing
                    .occupation((first.energy - options.fermi) / options.width)
                    - options
                        .smearing
                        .occupation((second.energy - options.fermi) / options.width))
                / delta
        };
        if options.constrained {
            weight *= 1. - first.target.clamp(0., 1.) * second.target.clamp(0., 1.);
        }
        if !weight.is_finite() {
            return Err(invalid("nonfinite static response transition weight"));
        }
        if weight.abs() <= 1e-12 {
            return Ok(());
        }
        let start = self.count * self.vectors;
        self.densities.as_mut_slice()[start..start + self.vectors].copy_from_slice(density);
        self.weights.as_mut_slice()[self.count] =
            weight / (options.kpoints as f64 * options.cell_volume);
        self.count += 1;
        if self.count == self.weights.as_slice().len() {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        if self.count == 0 {
            return Ok(());
        }
        let n = self.vectors;
        let batch = self.weights.as_slice().len();
        for (left_imag, right_imag, imaginary, sign) in [
            (false, false, false, 1.),
            (true, true, false, 1.),
            (true, false, true, 1.),
            (false, true, true, -1.),
        ] {
            for t in 0..self.count {
                for g in 0..n {
                    let rho = self.densities.as_slice()[t * n + g];
                    self.left.as_mut_slice()[t * n + g] = if left_imag { rho.im } else { rho.re };
                    self.right.as_mut_slice()[g * batch + t] =
                        self.weights.as_slice()[t] * if right_imag { rho.im } else { rho.re };
                }
            }
            let target = if imaginary {
                &mut self.imaginary
            } else {
                &mut self.real
            };
            gemm::multiply(
                MatrixRef {
                    values: self.left.as_slice(),
                    rows: n,
                    columns: self.count,
                    leading: n,
                },
                MatrixRef {
                    values: self.right.as_slice(),
                    rows: self.count,
                    columns: n,
                    leading: batch,
                },
                MatrixMut {
                    values: target.as_mut_slice(),
                    rows: n,
                    columns: n,
                    leading: n,
                },
                sign,
                1.,
            )?;
        }
        self.count = 0;
        Ok(())
    }
    pub fn finish(mut self) -> Result<(DMatrix<C64>, Reservation)> {
        self.flush()?;
        let memory = self
            .context
            .reserve(product(&[self.vectors, self.vectors, 16])?)?;
        let values = DMatrix::from_fn(self.vectors, self.vectors, |i, j| {
            C64::new(
                self.real.as_slice()[i + j * self.vectors],
                self.imaginary.as_slice()[i + j * self.vectors],
            )
        });
        Ok((values, memory))
    }
}

/// The G=0 periodic Coulomb component is excluded here. A truncated fragment
/// interaction's head is determined separately from small nonzero q states.
pub fn dyson(
    ctx: &Context,
    chi: &DMatrix<C64>,
    wavevectors: &[[f64; 3]],
) -> Result<ScreenedInteraction> {
    let n = wavevectors.len();
    if n == 0
        || chi.shape() != (n, n)
        || wavevectors.iter().flatten().any(|v| !v.is_finite())
        || chi.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        || (chi - chi.adjoint()).norm() > 1e-9 * chi.norm().max(1.)
    {
        return Err(invalid(
            "static response must be finite and Hermitian on its reciprocal vectors",
        ));
    }
    let memory = ctx.reserve(product(&[n, n, 16, 2])?)?;
    let _work = ctx.reserve(product(&[n, n, 16, 10])?)?;
    ctx.run(|| {
        let coulomb: Vec<_> = wavevectors
            .iter()
            .map(|v| {
                let q2 = v.iter().map(|x| x * x).sum::<f64>();
                if q2 < 1e-14 {
                    0.
                } else {
                    (4. * PI / q2).sqrt()
                }
            })
            .collect();
        let dielectric = DMatrix::from_fn(n, n, |i, j| {
            C64::new(f64::from(i == j), 0.) - coulomb[i] * chi[(i, j)] * coulomb[j]
        });
        let (e, u) = eigen(dielectric)?;
        if e[0] <= 1e-12 {
            return Err(invalid(
                "static dielectric response is singular or unstable",
            ));
        }
        let inverse =
            &u * DMatrix::from_diagonal(&DVector::from_iterator(
                n,
                e.iter().map(|v| C64::new(1. / v, 0.)),
            )) * u.adjoint();
        let correction = DMatrix::from_fn(n, n, |i, j| {
            coulomb[i] * (inverse[(i, j)] - C64::new(f64::from(i == j), 0.)) * coulomb[j]
        });
        Ok(ScreenedInteraction {
            inverse_dielectric: inverse,
            correction,
            _memory: memory,
        })
    })
}
