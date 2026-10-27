//! Discrete spin LDA/PBE. Forward derivatives of the energy expression and
//! skew-adjoint grid differences make the potential a derivative by construction.
use super::{invalid, SpatialGrid};
use crate::{product, Buffer, Context, Result};
use std::{
    f64::consts::PI,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Clone, Copy)]
struct Jet {
    v: f64,
    d: [f64; 5],
}
impl Jet {
    fn c(v: f64) -> Self {
        Self { v, d: [0.; 5] }
    }
    fn variable(v: f64, axis: usize) -> Self {
        let mut out = Self::c(v);
        out.d[axis] = 1.;
        out
    }
    fn unary(self, v: f64, derivative: f64) -> Self {
        Self {
            v,
            d: self.d.map(|d| d * derivative),
        }
    }
    fn pow(self, p: f64) -> Self {
        self.unary(self.v.powf(p), p * self.v.powf(p - 1.))
    }
    fn ln(self) -> Self {
        self.unary(self.v.ln(), self.v.recip())
    }
    fn ln1p(self) -> Self {
        self.unary(self.v.ln_1p(), (1. + self.v).recip())
    }
    fn expm1(self) -> Self {
        self.unary(self.v.exp_m1(), self.v.exp())
    }
}
impl Add for Jet {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl Neg for Jet {
    type Output = Self;
    fn neg(self) -> Self {
        self.unary(-self.v, -1.)
    }
}
impl Sub for Jet {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for Jet {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for Jet {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: std::array::from_fn(|i| (self.d[i] - v * b.d[i]) / b.v),
        }
    }
}
macro_rules! scalar_ops {($trait:ident,$method:ident,$op:tt) => {
    impl $trait<f64> for Jet {type Output=Self;fn $method(self,b:f64)->Self {self $op Self::c(b)}}
    impl $trait<Jet> for f64 {type Output=Jet;fn $method(self,b:Jet)->Jet {Jet::c(self) $op b}}
};}
scalar_ops!(Add,add,+);
scalar_ops!(Sub,sub,-);
scalar_ops!(Mul,mul,*);
scalar_ops!(Div,div,/);

pub fn functional(name: &str) -> Result<&'static str> {
    match name.trim().to_lowercase().replace(['-', '_'], "").as_str() {
        "lda" | "pz" | "pz81" | "ldapz" => Ok("lda"),
        "pbe" | "gga" | "ggapbe" => Ok("pbe"),
        _ => Err(invalid("native XC supports LDA/PZ81 and PBE")),
    }
}
fn pz(rs: Jet, ferro: bool) -> Jet {
    let (gamma, b1, b2, a, b) = if ferro {
        (-0.0843, 1.3981, 0.2611, 0.01555, -0.0269)
    } else {
        (-0.1423, 1.0529, 0.3334, 0.0311, -0.048)
    };
    if rs.v >= 1. {
        return gamma / (1. + b1 * rs.pow(0.5) + b2 * rs);
    }
    let denominator = 1. + b1 + b2;
    let d = gamma / denominator - b;
    let c = -gamma * (b1 / 2. + b2) / (denominator * denominator) - a - d;
    a * rs.ln() + b + c * rs * rs.ln() + d * rs
}
fn pw(rs: Jet, params: [f64; 6]) -> Jet {
    let [a, a1, b1, b2, b3, b4] = params;
    let q = 2. * a * (b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs * rs);
    -2. * a * (1. + a1 * rs) * (1. / q).ln1p()
}
/// f and its five partials with respect to (up, down, uu, ud, dd).
pub fn partials(name: &str, values: [f64; 5]) -> Result<(f64, [f64; 5])> {
    partials_key(functional(name)?, values)
}
pub(super) fn partials_key(name: &str, values: [f64; 5]) -> Result<(f64, [f64; 5])> {
    partials_relativistic(name, values, false)
}
pub(crate) fn partials_relativistic(
    name: &str,
    values: [f64; 5],
    relativistic: bool,
) -> Result<(f64, [f64; 5])> {
    if values.iter().any(|v| !v.is_finite())
        || values[0] < 0.
        || values[1] < 0.
        || values[2] < 0.
        || values[4] < 0.
    {
        return Err(invalid(
            "XC densities and diagonal gradient contractions must be finite and nonnegative",
        ));
    }
    let [up, dn, suu, sud, sdd] = std::array::from_fn(|i| Jet::variable(values[i], i));
    let beta = 0.06672455060314922;
    let gamma = (1. - 2f64.ln()) / (PI * PI);
    let mut f = Jet::c(0.);
    for (rho, sigma) in [(up * 2., suu * 4.), (dn * 2., sdd * 4.)] {
        if rho.v <= 2e-12 {
            continue;
        }
        let mut fx = -0.75 * (3. / PI).cbrt() * rho.pow(4. / 3.);
        if name == "pbe" {
            let kf = (3. * PI * PI * rho).pow(1. / 3.);
            let s2 = sigma / (4. * kf * kf * rho * rho);
            fx = fx * (1. + 0.804 - 0.804 / (1. + (beta * PI * PI / 3.) * s2 / 0.804));
        }
        if relativistic {
            let beta = (3. * PI * PI * rho).pow(1. / 3.) / 137.035999177;
            let factor = if beta.v < 1e-2 {
                1. - (2. / 3.) * beta.pow(2.) + (2. / 5.) * beta.pow(4.)
            } else {
                let asinh = beta.unary(beta.v.asinh(), 1. / (1. + beta.v * beta.v).sqrt());
                let f = (1. + beta * beta).pow(0.5) / beta - asinh / (beta * beta);
                1. - 1.5 * f * f
            };
            fx = fx * factor;
        }
        f = f + 0.5 * fx;
    }
    let rho = up + dn;
    if rho.v > 1e-12 {
        let z = (1. - 1e-12) * ((up - dn) / rho);
        let fz = ((1. + z).pow(4. / 3.) + (1. - z).pow(4. / 3.) - 2.) / (2f64.powf(4. / 3.) - 2.);
        let rs = (3. / (4. * PI) / rho).pow(1. / 3.);
        let ec = if name == "lda" {
            let para = pz(rs, false);
            para + fz * (pz(rs, true) - para)
        } else {
            let para = pw(rs, [0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]);
            let ferro = pw(rs, [0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517]);
            let stiffness = -pw(rs, [0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671]);
            para + stiffness * fz * (1. - z.pow(4.)) / 1.709921 + (ferro - para) * fz * z.pow(4.)
        };
        let mut correlation = ec;
        if name == "pbe" {
            let phi = 0.5 * ((1. + z).pow(2. / 3.) + (1. - z).pow(2. / 3.));
            let kf = (3. * PI * PI * rho).pow(1. / 3.);
            let t2 = (suu + 2. * sud + sdd) / (4. * phi * phi * (4. / PI * kf) * rho * rho);
            let ratio = beta / gamma;
            let a = ratio / (-ec / (gamma * phi.pow(3.))).expm1();
            let at = a * t2;
            correlation = correlation
                + gamma * phi.pow(3.) * (ratio * t2 * (1. + at) / (1. + at + at * at)).ln1p();
        }
        f = f + rho * correlation;
    }
    if !f.v.is_finite() || f.d.iter().any(|d| !d.is_finite()) {
        return Err(invalid("nonfinite XC evaluation"));
    }
    Ok((f.v, f.d))
}
/// Molecular orthorhombic ninth-point periodic-wrap stencil, also at boundaries.
pub fn derivative(grid: &SpatialGrid, field: &[f64], axis: usize, out: &mut [f64]) -> Result<()> {
    let n = grid.index.points();
    if axis > 2 || field.len() != n || out.len() != n {
        return Err(invalid("invalid XC derivative shape"));
    }
    if (0..3).any(|a| (0..3).any(|b| a != b && grid.step[(a, b)].abs() > 1e-14)) || grid.periodic {
        return Err(invalid(
            "native PBE currently requires a molecular orthorhombic grid",
        ));
    }
    let shape = grid.index.shape();
    let stride = [shape[1] * shape[2], shape[2], 1][axis];
    let length = shape[axis];
    for (k, out) in out.iter_mut().enumerate() {
        let index = k / stride % length;
        let base = k - index * stride;
        *out = [0.8, -0.2, 4. / 105., -1. / 280.]
            .iter()
            .enumerate()
            .map(|(m, c)| {
                let ahead = (index + m + 1) % length;
                let back = (index + length - (m + 1) % length) % length;
                c * (field[base + ahead * stride] - field[base + back * stride])
            })
            .sum::<f64>()
            / grid.step[(axis, axis)];
    }
    Ok(())
}
pub struct Xc {
    pub energy: f64,
    pub potentials: Buffer<f64>,
}
pub fn evaluate(ctx: &Context, grid: &SpatialGrid, densities: &[f64], name: &str) -> Result<Xc> {
    let name = functional(name)?;
    let n = grid.index.points();
    if densities.len() != product(&[2, n])? {
        return Err(invalid("XC requires two grid densities"));
    }
    let mut potentials = ctx.allocate::<f64>(2 * n)?;
    let mut gradients = ctx.allocate::<f64>(6 * n)?;
    if name == "pbe" {
        for s in 0..2 {
            for a in 0..3 {
                derivative(
                    grid,
                    &densities[s * n..(s + 1) * n],
                    a,
                    &mut gradients.as_mut_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                )?;
            }
        }
    }
    let mut flux = ctx.allocate::<f64>(6 * n)?;
    let mut work = ctx.allocate::<f64>(n)?;
    let mut energy = 0.;
    for k in 0..n {
        let weighted = densities[k] + densities[n + k] > 1e-10;
        let mut sigma = [0.; 3];
        if weighted {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(3 + a) * n + k];
                sigma[0] += u * u;
                sigma[1] += u * d;
                sigma[2] += d * d;
            }
        }
        let (f, df) = partials_key(
            name,
            [densities[k], densities[n + k], sigma[0], sigma[1], sigma[2]],
        )?;
        energy += f;
        potentials.as_mut_slice()[k] = df[0];
        potentials.as_mut_slice()[n + k] = df[1];
        if weighted && name == "pbe" {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(3 + a) * n + k];
                flux.as_mut_slice()[a * n + k] = 2. * df[2] * u + df[3] * d;
                flux.as_mut_slice()[(3 + a) * n + k] = 2. * df[4] * d + df[3] * u;
            }
        }
    }
    if name == "pbe" {
        for s in 0..2 {
            for a in 0..3 {
                derivative(
                    grid,
                    &flux.as_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                    a,
                    work.as_mut_slice(),
                )?;
                for k in 0..n {
                    potentials.as_mut_slice()[s * n + k] -= work.as_slice()[k];
                }
            }
        }
    }
    Ok(Xc {
        energy: energy * grid.index.dv(),
        potentials,
    })
}
