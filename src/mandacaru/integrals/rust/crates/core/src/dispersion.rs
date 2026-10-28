//! Independent dispersion building blocks in atomic units.
//!
//! BJ pair damping, neutral-reference ATM geometry and constrained Gaussian
//! electronegativity equilibration. These are not a complete D4 model: no
//! elemental reference data, coordination interpolation or periodic model is
//! bundled here. Caldeweyher et al., doi:10.1063/1.5090222 and
//! doi:10.1039/D0CP00502A specify the scientific model.
use crate::{finite, product, Buffer, Context, Error, Geometry, Result};
use nalgebra::{DMatrix, DVector};
use std::{
    f64::consts::PI,
    ops::{Add, Div, Mul, Neg, Sub},
};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

#[derive(Clone, Copy)]
struct Dual<const N: usize> {
    v: f64,
    d: [f64; N],
}
impl<const N: usize> Dual<N> {
    fn c(v: f64) -> Self {
        Self { v, d: [0.; N] }
    }
    fn x(v: f64, axis: usize) -> Self {
        let mut x = Self::c(v);
        x.d[axis] = 1.;
        x
    }
    fn pow(self, p: f64) -> Self {
        Self {
            v: self.v.powf(p),
            d: self.d.map(|d| p * self.v.powf(p - 1.) * d),
        }
    }
}
impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: self.d.map(|d| -d),
        }
    }
}
impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: std::array::from_fn(|i| (self.d[i] - v * b.d[i]) / b.v),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Damping {
    pub s6: f64,
    pub s8: f64,
    pub a1: f64,
    pub a2: f64,
}
impl Damping {
    fn validate(self) -> Result<()> {
        finite(&[self.s6, self.s8, self.a1, self.a2], "dispersion damping")?;
        if self.a1 < 0. || self.a2 <= 0. {
            return Err(invalid(
                "BJ damping requires nonnegative a1 and positive a2",
            ));
        }
        Ok(())
    }
}
/// Pair energy and total partials in (distance,C6,C8), including the
/// coefficient dependence of the damping radius.
pub fn pair(distance: f64, c6: f64, c8: f64, damping: Damping) -> Result<(f64, [f64; 3])> {
    damping.validate()?;
    if [distance, c6, c8]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(invalid(
            "dispersion pair distance and coefficients must be positive",
        ));
    }
    let r = Dual::<3>::x(distance, 0);
    let c6 = Dual::x(c6, 1);
    let c8 = Dual::x(c8, 2);
    let radius = Dual::c(damping.a1) * (c8 / c6).pow(0.5) + Dual::c(damping.a2);
    let e = -Dual::c(damping.s6) * c6 / (r.pow(6.) + radius.pow(6.))
        - Dual::c(damping.s8) * c8 / (r.pow(8.) + radius.pow(8.));
    if !e.v.is_finite() || e.d.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite dispersion pair"));
    }
    Ok((e.v, e.d))
}
/// ATM energy and partials in (rAB,rBC,rCA,C6AB^0,C6BC^0,C6CA^0,dAB,dBC,dCA).
/// C6 inputs here are the neutral-reference coefficients, not the charged
/// coefficients of the pair energy.
pub fn three_body(
    distances: [f64; 3],
    neutral_c6: [f64; 3],
    radii: [f64; 3],
    s9: f64,
) -> Result<(f64, [f64; 9])> {
    if !s9.is_finite()
        || distances
            .iter()
            .chain(&neutral_c6)
            .chain(&radii)
            .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(invalid(
            "ATM distances, neutral coefficients and radii must be positive",
        ));
    }
    for i in 0..3 {
        if distances[i] > distances[(i + 1) % 3] + distances[(i + 2) % 3] {
            return Err(invalid("ATM distances must form a triangle"));
        }
    }
    let [x, y, z] = std::array::from_fn(|i| Dual::<9>::x(distances[i], i));
    let c9 = (0..3)
        .fold(Dual::c(1.), |p, i| p * Dual::x(neutral_c6[i], i + 3))
        .pow(0.5);
    let damping_product = (0..3).fold(Dual::c(1.), |p, i| p * Dual::x(radii[i], i + 6));
    let product = x * y * z;
    let cos_a = (x * x + z * z - y * y) / (Dual::c(2.) * x * z);
    let cos_b = (x * x + y * y - z * z) / (Dual::c(2.) * x * y);
    let cos_c = (y * y + z * z - x * x) / (Dual::c(2.) * y * z);
    let damp =
        Dual::c(1.) / (Dual::c(1.) + Dual::c(6.) * (product / damping_product).pow(-16. / 3.));
    let e = Dual::c(s9) * c9 * (Dual::c(1.) + Dual::c(3.) * cos_a * cos_b * cos_c) * damp
        / product.pow(3.);
    if !e.v.is_finite() || e.d.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite ATM energy"));
    }
    Ok((e.v, e.d))
}

pub struct ChargeParameters<'a> {
    pub electronegativities: &'a [f64],
    pub hardnesses: &'a [f64],
    pub widths: &'a [f64],
    /// Optional d(chi_i)/d(R_j,axis), row-major (n,n,3). This is necessary
    /// when the caller's electronegativities depend on coordination.
    pub electronegativity_gradient: Option<&'a [f64]>,
}
pub struct Charges {
    pub values: Buffer<f64>,
    /// dq_i/d(R_j,axis), row-major (n,n,3), Bohr^-1.
    pub response: Buffer<f64>,
}
/// Isolated constrained Gaussian charge equilibration and its analytic full
/// coordinate response for the supplied environment-dependent chi values.
pub fn equilibrate(
    ctx: &Context,
    geometry: &Geometry,
    parameters: ChargeParameters<'_>,
    charge: f64,
) -> Result<Charges> {
    ctx.run(|| {
        let n = geometry.numbers().len();
        let stride = product(&[n, 3])?;
        let response_size = product(&[n, stride])?;
        let size = n.checked_add(1).ok_or(Error::Overflow)?;
        let _matrix_memory = ctx.reserve(product(&[size, size, 8, 6])?)?;
        let chi = parameters.electronegativities;
        let eta = parameters.hardnesses;
        let widths = parameters.widths;
        if geometry.pbc().iter().any(|p| *p)
            || !charge.is_finite()
            || [chi.len(), eta.len(), widths.len()]
                .iter()
                .any(|len| *len != n)
            || parameters
                .electronegativity_gradient
                .is_some_and(|v| v.len() != response_size || v.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid(
                "charge equilibration requires isolated geometry and matching parameters",
            ));
        }
        for (v, label) in [
            (chi, "electronegativities"),
            (eta, "hardnesses"),
            (widths, "Gaussian widths"),
        ] {
            finite(v, label)?;
        }
        if widths.iter().any(|a| *a <= 0.) {
            return Err(invalid("Gaussian charge widths must be positive"));
        }
        let mut a = DMatrix::zeros(size, size);
        for i in 0..n {
            a[(i, i)] = eta[i] + (2. / PI).sqrt() / widths[i];
            a[(i, n)] = 1.;
            a[(n, i)] = 1.;
            for j in 0..i {
                let r = (0..3)
                    .map(|axis| {
                        (geometry.positions()[i][axis] - geometry.positions()[j][axis]).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                if r <= 1e-10 {
                    return Err(invalid("coincident charge-equilibration sites"));
                }
                let gamma = (widths[i].powi(2) + widths[j].powi(2)).sqrt().recip();
                a[(i, j)] = libm::erf(gamma * r) / r;
                a[(j, i)] = a[(i, j)];
            }
        }
        if n > 1 {
            let constrained = DMatrix::from_fn(n - 1, n - 1, |i, j| {
                a[(i, j)] - a[(i, n - 1)] - a[(n - 1, j)] + a[(n - 1, n - 1)]
            });
            if constrained.cholesky().is_none() {
                return Err(invalid(
                    "charge energy is not positive on the charge-conserving subspace",
                ));
            }
        }
        let lu = a.lu();
        let mut rhs = DVector::from_fn(size, |i, _| if i == n { charge } else { -chi[i] });
        let q = lu
            .solve(&rhs)
            .ok_or_else(|| invalid("singular constrained charge system"))?;
        if q.iter().any(|v| !v.is_finite())
            || (q.iter().take(n).sum::<f64>() - charge).abs() > 1e-9 * (1. + charge.abs())
        {
            return Err(invalid(
                "charge equilibration did not conserve total charge",
            ));
        }
        let mut out = Charges {
            values: ctx.allocate::<f64>(n)?,
            response: ctx.allocate::<f64>(response_size)?,
        };
        out.values
            .as_mut_slice()
            .copy_from_slice(&q.as_slice()[..n]);
        for atom in 0..n {
            for axis in 0..3 {
                rhs.fill(0.);
                if let Some(gradient) = parameters.electronegativity_gradient {
                    for i in 0..n {
                        rhs[i] = -gradient[i * stride + atom * 3 + axis];
                    }
                }
                for j in 0..n {
                    if j != atom {
                        let delta = std::array::from_fn::<_, 3, _>(|a| {
                            geometry.positions()[atom][a] - geometry.positions()[j][a]
                        });
                        let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
                        let gamma = (widths[atom].powi(2) + widths[j].powi(2)).sqrt().recip();
                        let derivative = (2. * gamma / PI.sqrt() * (-(gamma * r).powi(2)).exp()
                            / r
                            - libm::erf(gamma * r) / (r * r))
                            * delta[axis]
                            / r;
                        rhs[atom] -= derivative * q[j];
                        rhs[j] -= derivative * q[atom];
                    }
                }
                let dq = lu
                    .solve(&rhs)
                    .ok_or_else(|| invalid("singular charge response"))?;
                for i in 0..n {
                    out.response.as_mut_slice()[i * stride + atom * 3 + axis] = dq[i];
                }
            }
        }
        finite(out.response.as_slice(), "charge coordinate response")?;
        Ok(out)
    })
}
