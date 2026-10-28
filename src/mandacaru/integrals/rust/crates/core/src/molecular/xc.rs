//! Spin-resolved local, gradient, meta-GGA and screened exchange kernels.
//! Forward derivatives of the energy expression and
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
    d: [f64; 7],
}
impl Jet {
    fn c(v: f64) -> Self {
        Self { v, d: [0.; 7] }
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
    fn exp(self) -> Self {
        self.unary(self.v.exp(), self.v.exp())
    }
    fn tanh(self) -> Self {
        let v = self.v.tanh();
        self.unary(v, 1. - v * v)
    }
    fn erfc(self) -> Self {
        self.unary(
            libm::erfc(self.v),
            -2. / PI.sqrt() * (-self.v * self.v).exp(),
        )
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
    match name {
        "lda" => Ok("lda"),
        "pbe" => Ok("pbe"),
        _ => Err(invalid("expected canonical XC name lda or pbe")),
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
    Ok((f.v, std::array::from_fn(|i| f.d[i])))
}
/// Energy per volume and derivatives in (up, down, uu, ud, dd, tau_up, tau_down).
/// Meta-GGA evaluation is separate from the density-only grid API: callers
/// must supply the kinetic-energy densities rather than silently substituting.
pub fn point_partials(name: &str, values: [f64; 7], relativistic: bool) -> Result<(f64, [f64; 7])> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("point XC variables must be finite"));
    }
    match name {
        "lda" | "pbe" => {
            let (energy, derivative) =
                partials_relativistic(name, std::array::from_fn(|i| values[i]), relativistic)?;
            Ok((
                energy,
                std::array::from_fn(|i| if i < 5 { derivative[i] } else { 0. }),
            ))
        }
        "hse06" => hse_partials(values, relativistic),
        "r2scan" => {
            if relativistic {
                return Err(invalid("relativistic exchange is defined for lda and pbe"));
            }
            if values.iter().any(|v| !v.is_finite())
                || [0, 1, 2, 4, 5, 6].iter().any(|&i| values[i] < 0.)
            {
                return Err(invalid("point XC requires finite physical densities, diagonal contractions and kinetic densities"));
            }
            let [up, dn, suu, sud, sdd, tu, td] =
                std::array::from_fn(|i| Jet::variable(values[i], i));
            let total = up + dn;
            if total.v <= 1e-12 {
                return Ok((0., [0.; 7]));
            }
            let mut f = Jet::c(0.);
            for (rho, sigma, tau) in [(2. * up, 4. * suu, 2. * tu), (2. * dn, 4. * sdd, 2. * td)] {
                if rho.v > 2e-12 {
                    f = f + 0.5 * r2_exchange(rho, sigma, tau);
                }
            }
            let sigma = suu + 2. * sud + sdd;
            if sigma.v < 0. {
                return Err(invalid(
                    "total density-gradient contraction must be nonnegative",
                ));
            }
            let z = (1. - 1e-12) * (up - dn) / total;
            f = f + r2_correlation(total, z, sigma, tu + td);
            if !f.v.is_finite() || f.d.iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite meta-GGA evaluation"));
            }
            Ok((f.v, f.d))
        }
        _ => Err(invalid(
            "point XC requires the canonical name lda, pbe, r2scan or hse06",
        )),
    }
}
fn hole_parameters(rho: Jet, sigma: Jet) -> (Jet, Jet, Jet, Jet) {
    let kf = (3. * PI * PI * rho).pow(1. / 3.);
    let mut t = sigma / (4. * kf * kf * rho * rho);
    if t.v > 1. {
        let s = t.pow(0.5);
        let x = 8.572844 - s;
        let softplus = if x.v > 0. {
            x + (-x).exp().ln1p()
        } else {
            x.exp().ln1p()
        };
        let bent = 8.572844 - softplus + (1. - 8.572844f64).exp().ln_1p();
        t = bent * bent;
    }
    let t2 = t * t;
    let h = t * (0.00979681 + 0.0410834 * t)
        / (1. + 0.187440 * t2 + 0.00120824 * t.pow(2.5) + 0.0347188 * t * t2);
    let hs = t * h;
    let f = 6.4753871 * h + 0.47965830;
    let eg = if t.v <= 0.08 * 0.08 {
        -0.02628417880 - 0.07117647788 * t + 0.08534541323 * t2
    } else {
        let p = 0.57786348 + hs;
        let x = (9. / (4. * 1.0161144) * hs).pow(0.5);
        let scaled_erfc = positive_erfcx(x);
        let bracket = 0.75 * PI
            + PI.sqrt()
                * (15. * -0.051955731
                    + 6. * -0.077215461 * (f * t + 1.) * p
                    + 4. * -0.37170836 * p * p
                    + 8. * 1.0161144 * p.pow(3.))
                / (16. * p.pow(3.5))
            - 0.75 * PI * 1.0161144f64.sqrt() * scaled_erfc;
        -16. / 15. * bracket * p.pow(3.5) / (PI.sqrt() * t)
    };
    (t, hs, f, eg)
}
fn positive_erfcx(x: Jet) -> Jet {
    if x.v < 8. {
        let value = (x.v * x.v).exp() * libm::erfc(x.v);
        x.unary(value, 2. * x.v * value - 2. / PI.sqrt())
    } else {
        let mut fraction = x;
        for i in (1..=64).rev() {
            fraction = x + (0.5 * i as f64) / fraction;
        }
        1. / (PI.sqrt() * fraction)
    }
}
fn e1_plus_log(x: Jet) -> Jet {
    const EULER: f64 = 0.577_215_664_901_532_9;
    if x.v == 0. {
        return Jet::c(-EULER);
    }
    if x.v < 3. {
        let mut term = x;
        let mut sum = x;
        for k in 2..=40 {
            term = -term * x / k as f64;
            sum = sum + term / k as f64;
        }
        x.exp() * (sum - EULER) - x.expm1() * x.ln()
    } else {
        let mut f = x + 81.;
        for k in (1..=40).rev() {
            f = x + (2 * k - 1) as f64 - (k * k) as f64 / f;
        }
        1. / f + x.ln()
    }
}
fn hole_exchange(rho: Jet, sigma: Jet, omega: f64) -> Result<Jet> {
    let (t, hs, f, eg) = hole_parameters(rho, sigma);
    let p = 0.57786348 + hs;
    let enhancement = if omega == 0. {
        let c = 9. / (4. * 1.0161144);
        let long = 0.5 * 1.0161144 * (e1_plus_log(c * hs) - c.ln() - p.ln());
        let gaussian = -4. / 9.
            * (-0.37170836 * p * p - 0.077215461 * p * (1. + f * t) + 2. * (-0.051955731 + eg * t))
            / p.pow(3.);
        -8. / 9. * long + gaussian
    } else {
        let nu = omega / (3. * PI * PI * rho).pow(1. / 3.);
        let integrand = |y: f64| {
            let y2 = y * y;
            let c = 9. / (4. * 1.0161144);
            let bracket = -y2 / (c + y2) - (-0.57786348 * y2).exp_m1();
            let long = -1.0161144 / y * (-hs * y2).exp() * bracket;
            let gaussian = y
                * (-p * y2).exp()
                * (-0.37170836 - 0.077215461 * (1. + f * t) * y2
                    + (-0.051955731 + eg * t) * y2 * y2);
            (long + gaussian) * (nu * y).erfc()
        };
        let maximum = 8.5 / nu.v;
        if !maximum.is_finite() {
            return Err(invalid("screened exchange integration range overflows"));
        }
        let mut low = 0.;
        let mut high = maximum.min(1.);
        let mut sum = Jet::c(0.);
        let mut budget = 16384;
        while low < maximum {
            sum = sum + integrate_hole(&integrand, low, high, 0, &mut budget)?;
            low = high;
            high = (2. * high).min(maximum);
        }
        -8. / 9. * sum
    };
    Ok(-0.75 * (3. / PI).cbrt() * rho.pow(4. / 3.) * enhancement)
}
fn gauss_nodes<const N: usize>() -> [(f64, f64); N] {
    std::array::from_fn(|i| {
        let mut z = (PI * (i as f64 + 0.75) / (N as f64 + 0.5)).cos();
        let mut derivative = 0.;
        for _ in 0..64 {
            let (mut a, mut b) = (1., z);
            for k in 2..=N {
                let p = ((2 * k - 1) as f64 * z * b - (k - 1) as f64 * a) / k as f64;
                a = b;
                b = p;
            }
            derivative = N as f64 * (z * b - a) / (z * z - 1.);
            let next = z - b / derivative;
            if (next - z).abs() < 2e-16 {
                z = next;
                break;
            }
            z = next;
        }
        (z, 2. / ((1. - z * z) * derivative * derivative))
    })
}
fn integrate_hole(
    f: &impl Fn(f64) -> Jet,
    lo: f64,
    hi: f64,
    depth: usize,
    budget: &mut usize,
) -> Result<Jet> {
    static LOW: std::sync::OnceLock<[(f64, f64); 16]> = std::sync::OnceLock::new();
    static HIGH: std::sync::OnceLock<[(f64, f64); 32]> = std::sync::OnceLock::new();
    if *budget == 0 {
        return Err(invalid("screened exchange quadrature budget exhausted"));
    }
    *budget -= 1;
    let mid = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let eval = |nodes: &[(f64, f64)]| {
        nodes.iter().fold(Jet::c(0.), |sum, &(x, w)| {
            sum + w * half * f(mid + half * x)
        })
    };
    let coarse = eval(LOW.get_or_init(gauss_nodes::<16>));
    let fine = eval(HIGH.get_or_init(gauss_nodes::<32>));
    let good = std::iter::once((coarse.v, fine.v))
        .chain(coarse.d.into_iter().zip(fine.d))
        .all(|(a, b)| (a - b).abs() <= 1e-12 + 1e-10 * b.abs());
    if good {
        return Ok(fine);
    }
    if depth >= 16 {
        return Err(invalid("screened exchange quadrature did not converge"));
    }
    Ok(integrate_hole(f, lo, mid, depth + 1, budget)?
        + integrate_hole(f, mid, hi, depth + 1, budget)?)
}
fn hse_partials(values: [f64; 7], relativistic: bool) -> Result<(f64, [f64; 7])> {
    hybrid_partials(values, relativistic, 0.11, 0.25)
}
fn hybrid_partials(
    values: [f64; 7],
    relativistic: bool,
    omega: f64,
    fraction: f64,
) -> Result<(f64, [f64; 7])> {
    if relativistic || [0, 1, 2, 4].iter().any(|&i| values[i] < 0.) {
        return Err(invalid("hse06 needs nonnegative spin densities and diagonal gradients, without relativistic exchange"));
    }
    let (energy, d) = partials_relativistic("pbe", std::array::from_fn(|i| values[i]), false)?;
    let mut result = Jet {
        v: energy,
        d: std::array::from_fn(|i| if i < 5 { d[i] } else { 0. }),
    };
    let variables: [Jet; 7] = std::array::from_fn(|i| Jet::variable(values[i], i));
    for (rho, sigma) in [
        (2. * variables[0], 4. * variables[2]),
        (2. * variables[1], 4. * variables[4]),
    ] {
        if rho.v <= 2e-12 {
            continue;
        }
        let kf = (3. * PI * PI * rho).pow(1. / 3.);
        let s2 = sigma / (4. * kf * kf * rho * rho);
        let beta = 0.06672455060314922;
        let pbe = -0.75
            * (3. / PI).cbrt()
            * rho.pow(4. / 3.)
            * (1. + 0.804 - 0.804 / (1. + (beta * PI * PI / 3.) * s2 / 0.804));
        result = result + 0.5 * (hole_exchange(rho, sigma, 0.)? - pbe);
        if fraction != 0. {
            result = result - 0.5 * fraction * hole_exchange(rho, sigma, omega)?;
        }
    }
    if !result.v.is_finite() || result.d.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite screened hybrid evaluation"));
    }
    Ok((result.v, result.d))
}
/// Full or short-range unpolarized exchange hole: input (rho,sigma),
/// output (energy density, d/drho, d/dsigma).
pub fn exchange_hole_batch(ctx: &Context, values: &[f64], omega: f64) -> Result<Buffer<f64>> {
    ctx.run(|| {
        if !omega.is_finite()
            || omega < 0.
            || values.len() % 2 != 0
            || values.iter().any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(invalid(
                "exchange hole requires nonnegative finite density, sigma and screening",
            ));
        }
        let mut out = ctx.allocate::<f64>(product(&[values.len() / 2, 3])?)?;
        for (x, y) in values
            .chunks_exact(2)
            .zip(out.as_mut_slice().chunks_exact_mut(3))
        {
            if x[0] > 1e-12 {
                let result = hole_exchange(Jet::variable(x[0], 0), Jet::variable(x[1], 1), omega)?;
                y.copy_from_slice(&[result.v, result.d[0], result.d[1]]);
                if y.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("nonfinite exchange hole result"));
                }
            }
        }
        Ok(out)
    })
}
/// Hybrid semilocal intermediate with explicit range and exact-exchange fraction.
pub fn hybrid_batch(
    ctx: &Context,
    values: &[f64],
    omega: f64,
    fraction: f64,
) -> Result<Buffer<f64>> {
    ctx.run(|| {
        if !omega.is_finite()
            || omega < 0.
            || !fraction.is_finite()
            || !(0. ..=1.).contains(&fraction)
            || values.len() % 7 != 0
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "invalid hybrid fields, screening or exact-exchange fraction",
            ));
        }
        let mut out = ctx.allocate::<f64>(product(&[values.len() / 7, 8])?)?;
        for (x, y) in values
            .chunks_exact(7)
            .zip(out.as_mut_slice().chunks_exact_mut(8))
        {
            let (f, d) = hybrid_partials(x.try_into().unwrap(), false, omega, fraction)?;
            y[0] = f;
            y[1..].copy_from_slice(&d);
        }
        Ok(out)
    })
}
pub fn point_batch(
    ctx: &Context,
    name: &str,
    values: &[f64],
    relativistic: bool,
) -> Result<Buffer<f64>> {
    ctx.run(|| {
        if !["lda", "pbe", "r2scan", "hse06"].contains(&name) {
            return Err(invalid("unknown canonical point XC functional"));
        }
        if values.len() % 7 != 0 {
            return Err(invalid(
                "point XC input must contain seven variables per row",
            ));
        }
        let mut out = ctx.allocate::<f64>(product(&[values.len() / 7, 8])?)?;
        for (input, output) in values
            .chunks_exact(7)
            .zip(out.as_mut_slice().chunks_exact_mut(8))
        {
            let (energy, partials) = point_partials(name, input.try_into().unwrap(), relativistic)?;
            output[0] = energy;
            output[1..].copy_from_slice(&partials);
        }
        Ok(out)
    })
}
const R2_CX: [f64; 8] = [
    1.,
    -0.667,
    -0.4445555,
    -0.663086601049,
    1.451297044490,
    -0.887998041597,
    0.234528941479,
    -0.023185843322,
];
const R2_CC: [f64; 8] = [
    1.,
    -0.64,
    -0.4352,
    -1.535685604549,
    3.061560252175,
    -1.915710236206,
    0.516884468372,
    -0.051848879792,
];
fn slope(c: &[f64; 8]) -> f64 {
    c.iter().enumerate().map(|(i, c)| i as f64 * c).sum()
}
fn alpha_floor(a: Jet) -> Jet {
    if a.v < 0. {
        0.5 * (a / 0.5).tanh()
    } else {
        a
    }
}
fn interpolation(a: Jet, c: &[f64; 8], tail: f64, scale: f64) -> Jet {
    if a.v > 2.5 {
        -scale * (tail / (1. - a)).exp()
    } else {
        c.iter().rev().fold(Jet::c(0.), |value, &c| value * a + c)
    }
}
fn r2_exchange(rho: Jet, sigma: Jet, tau: Jet) -> Jet {
    let kf2 = (3. * PI * PI * rho).pow(2. / 3.);
    let p = sigma / (4. * kf2 * rho * rho);
    let tw = sigma / (8. * rho);
    let alpha = alpha_floor((tau - tw) / (0.3 * kf2 * rho + 0.001 * tw));
    let damping = (-p * p / 0.361f64.powi(4)).exp();
    let restored = -slope(&R2_CX) * (1. - 1.174);
    let x = ((20. / 27. + 5. * 0.001 / 3.) * restored * damping + 10. / 81.) * p;
    let h1 = 1. + 0.065 - 0.065 / (1. + x / 0.065);
    let fx = interpolation(alpha, &R2_CX, 0.8, 1.24);
    let safe = if p.v < 1e-30 { Jet::c(1e-30) } else { p };
    let gx = 1. - (-4.9479 / safe.pow(0.25)).exp();
    -0.75 * (3. / PI).cbrt() * rho.pow(4. / 3.) * (h1 + fx * (1.174 - h1)) * gx
}
fn pw_slope(rs: Jet, params: [f64; 6]) -> Jet {
    let [a, a1, b1, b2, b3, b4] = params;
    let q = 2. * a * (b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs * rs);
    let dq = a * (b1 / rs.pow(0.5) + 2. * b2 + 3. * b3 * rs.pow(0.5) + 4. * b4 * rs);
    -2. * a * a1 * (1. / q).ln1p() + 2. * a * (1. + a1 * rs) * dq / (q * q + q)
}
fn polarized_pw(rs: Jet, z: Jet, derivative: bool) -> Jet {
    let fz = ((1. + z).pow(4. / 3.) + (1. - z).pow(4. / 3.) - 2.) / (2f64.powf(4. / 3.) - 2.);
    let evaluate = if derivative { pw_slope } else { pw };
    let para = evaluate(rs, [0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]);
    let ferro = evaluate(rs, [0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517]);
    let stiffness = -evaluate(rs, [0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671]);
    para + stiffness * fz * (1. - z.pow(4.)) / 1.709921 + (ferro - para) * fz * z.pow(4.)
}
fn r2_correlation(rho: Jet, z: Jet, sigma: Jet, tau: Jet) -> Jet {
    let gamma = (1. - 2f64.ln()) / (PI * PI);
    let kf2 = (3. * PI * PI * rho).pow(2. / 3.);
    let p = sigma / (4. * kf2 * rho * rho);
    let tw = sigma / (8. * rho);
    let ds = 0.5 * ((1. + z).pow(5. / 3.) + (1. - z).pow(5. / 3.));
    let dx = 0.5 * ((1. + z).pow(4. / 3.) + (1. - z).pow(4. / 3.));
    let phi = 0.5 * ((1. + z).pow(2. / 3.) + (1. - z).pow(2. / 3.));
    let phi3 = phi.pow(3.);
    let alpha = alpha_floor((tau - tw) / (0.3 * kf2 * rho * ds + 0.001 * tw));
    let damping = (-p * p / 0.361f64.powi(4)).exp();
    let rs = (3. / (4. * PI * rho)).pow(1. / 3.);
    let e1 = polarized_pw(rs, z, false);
    let de1 = polarized_pw(rs, z, true);
    let gc = (1. - 2.3631 * (dx - 1.)) * (1. - z.pow(12.));
    let den = 1. + 0.0889 * rs.pow(0.5) + 0.125541 * rs;
    let e0 = -0.0285764 / den;
    let de0 = 0.0285764 * (0.5 * 0.0889 / rs.pow(0.5) + 0.125541) / (den * den);
    let w1 = (-e1 / (gamma * phi3)).expm1();
    let beta = 0.066725 * (1. + 0.1 * rs) / (1. + 0.1778 * rs);
    let ks2 = 4. * (3. * PI * PI * rho).pow(1. / 3.) / PI;
    let t2 = sigma / (4. * phi * phi * ks2 * rho * rho);
    let y = beta / (gamma * w1) * t2;
    let dy = slope(&R2_CC) / (27. * gamma * ds * phi3 * w1)
        * (20. * rs * (de0 * gc - de1) - 45. * 0.001 * (e0 * gc - e1))
        * p
        * damping;
    let g = (1. + 4. * (y - dy)).pow(-0.25);
    let e1_full = e1 + gamma * phi3 * (w1 * (1. - g)).ln1p();
    let w0 = (-e0 / 0.0285764).expm1();
    let g_inf = (1. + 4. * 0.128026 * p).pow(-0.25);
    let e0_full = (e0 + 0.0285764 * (w0 * (1. - g_inf)).ln1p()) * gc;
    rho * (e1_full + interpolation(alpha, &R2_CC, 1.5, 0.7) * (e0_full - e1_full))
}
/// Molecular orthorhombic ninth-point periodic-wrap stencil, also at boundaries.
pub fn derivative(grid: &SpatialGrid, field: &[f64], axis: usize, out: &mut [f64]) -> Result<()> {
    let n = grid.index.points();
    if axis > 2 || field.len() != n || out.len() != n {
        return Err(invalid("invalid XC derivative shape"));
    }
    if (0..3).any(|a| (0..3).any(|b| a != b && grid.step[(a, b)].abs() > 1e-14)) || grid.periodic {
        return Err(invalid(
            "native molecular XC derivatives require an orthorhombic isolated grid",
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
    pub tau_potentials: Option<Buffer<f64>>,
}
pub fn evaluate(ctx: &Context, grid: &SpatialGrid, densities: &[f64], name: &str) -> Result<Xc> {
    let name = functional(name)?;
    evaluate_fields(ctx, grid, densities, None, name)
}
pub fn evaluate_fields(
    ctx: &Context,
    grid: &SpatialGrid,
    densities: &[f64],
    kinetic: Option<&[f64]>,
    name: &str,
) -> Result<Xc> {
    let (name, nonlocal_b) = match name {
        "pbe+rvv10" => ("pbe", Some(6.3)),
        "r2scan+rvv10" => ("r2scan", Some(11.95)),
        _ => (name, None),
    };
    if !["lda", "pbe", "r2scan", "hse06"].contains(&name) {
        return Err(invalid("unknown canonical XC functional"));
    }
    let n = grid.index.points();
    if densities.len() != product(&[2, n])?
        || kinetic.is_some_and(|t| {
            t.len() != densities.len() || t.iter().any(|v| !v.is_finite() || *v < 0.)
        })
    {
        return Err(invalid("XC requires two grid densities"));
    }
    if name == "r2scan" && kinetic.is_none() {
        return Err(invalid("r2scan requires both kinetic-energy densities"));
    }
    let mut potentials = ctx.allocate::<f64>(2 * n)?;
    let mut tau_potentials = if name == "r2scan" {
        Some(ctx.allocate::<f64>(2 * n)?)
    } else {
        None
    };
    let mut gradients = ctx.allocate::<f64>(6 * n)?;
    if name != "lda" {
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
    let nonlocal = if let Some(b) = nonlocal_b {
        let mut total = ctx.allocate::<f64>(n)?;
        let mut sigma = ctx.allocate::<f64>(n)?;
        for k in 0..n {
            total.as_mut_slice()[k] = densities[k] + densities[n + k];
            if total.as_slice()[k] > 1e-10 {
                sigma.as_mut_slice()[k] = (0..3)
                    .map(|a| {
                        (gradients.as_slice()[a * n + k] + gradients.as_slice()[(3 + a) * n + k])
                            .powi(2)
                    })
                    .sum();
            }
        }
        Some(super::nonlocal::evaluate(
            ctx,
            grid,
            total.as_slice(),
            sigma.as_slice(),
            super::nonlocal::Options {
                b,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
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
        let (f, mut df) = if name == "r2scan" && !weighted {
            (0., [0.; 7])
        } else {
            point_partials(
                name,
                [
                    densities[k],
                    densities[n + k],
                    sigma[0],
                    sigma[1],
                    sigma[2],
                    kinetic.map(|v| v[k]).unwrap_or(0.),
                    kinetic.map(|v| v[n + k]).unwrap_or(0.),
                ],
                false,
            )?
        };
        if let Some(ref nonlocal) = nonlocal {
            df[0] += nonlocal.density.as_slice()[k];
            df[1] += nonlocal.density.as_slice()[k];
            df[2] += nonlocal.sigma.as_slice()[k];
            df[3] += 2. * nonlocal.sigma.as_slice()[k];
            df[4] += nonlocal.sigma.as_slice()[k];
        }
        energy += f;
        potentials.as_mut_slice()[k] = df[0];
        potentials.as_mut_slice()[n + k] = df[1];
        if let Some(ref mut tau) = tau_potentials {
            tau.as_mut_slice()[k] = df[5];
            tau.as_mut_slice()[n + k] = df[6];
        }
        if weighted && name != "lda" {
            for a in 0..3 {
                let u = gradients.as_slice()[a * n + k];
                let d = gradients.as_slice()[(3 + a) * n + k];
                flux.as_mut_slice()[a * n + k] = 2. * df[2] * u + df[3] * d;
                flux.as_mut_slice()[(3 + a) * n + k] = 2. * df[4] * d + df[3] * u;
            }
        }
    }
    if name != "lda" {
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
        energy: energy * grid.index.dv() + nonlocal.map_or(0., |v| v.energy),
        potentials,
        tau_potentials,
    })
}
