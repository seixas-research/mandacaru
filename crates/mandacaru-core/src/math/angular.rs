//! Normalized real spherical harmonics in polar and Cartesian coordinates.
use std::f64::consts::PI;

/// Complex harmonics with the Condon--Shortley phase. The same Cartesian
/// recurrence supplies real and complex bases, including their axis limits.
pub fn complex(l: usize, m: i32, point: [f64; 3]) -> num_complex::Complex64 {
    use num_complex::Complex64;
    if m == 0 {
        return Complex64::new(cartesian(l, 0, point), 0.);
    }
    let order = m.unsigned_abs() as i32;
    let sign = if order % 2 == 0 { 1. } else { -1. };
    let positive = Complex64::new(cartesian(l, order, point), cartesian(l, -order, point))
        * (sign / 2f64.sqrt());
    if m > 0 {
        positive
    } else {
        sign * positive.conj()
    }
}

fn factorial(n: i32) -> f64 {
    (2..=n).map(|n| n as f64).product()
}
fn parity(n: i32) -> f64 {
    if n % 2 == 0 {
        1.
    } else {
        -1.
    }
}
fn wigner(j: [i32; 3], m: [i32; 3]) -> f64 {
    let [a, b, c] = j;
    let [x, y, z] = m;
    if x + y + z != 0
        || x.abs() > a
        || y.abs() > b
        || z.abs() > c
        || a + b < c
        || a + c < b
        || b + c < a
    {
        return 0.;
    }
    let triangle = factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c)
        / factorial(a + b + c + 1);
    let magnetic = factorial(a + x)
        * factorial(a - x)
        * factorial(b + y)
        * factorial(b - y)
        * factorial(c + z)
        * factorial(c - z);
    let lo = 0.max(b - c - x).max(a - c + y);
    let hi = (a + b - c).min(a - x).min(b + y);
    let sum = (lo..=hi)
        .map(|k| {
            parity(k)
                / (factorial(k)
                    * factorial(a + b - c - k)
                    * factorial(a - x - k)
                    * factorial(b + y - k)
                    * factorial(c - b + x + k)
                    * factorial(c - a - y + k))
        })
        .sum::<f64>();
    parity(a - b - z) * (triangle * magnetic).sqrt() * sum
}

/// Integral conj(Y_LM) conj(Y_l1m1) Y_l2m2. Selection rules are enforced
/// before the finite Racah sum. The supported PAW shells have l <= 8.
pub fn gaunt(level: [usize; 3], order: [i32; 3]) -> crate::Result<f64> {
    let [l, a, b] = level;
    let [m, x, y] = order;
    if a > 8
        || b > 8
        || l > 16
        || order
            .iter()
            .zip(level)
            .any(|(m, l)| m.unsigned_abs() as usize > l)
    {
        return Err(crate::Error::Invalid(
            "invalid angular momenta in PAW Gaunt coefficient".into(),
        ));
    }
    if m + x != y || (l + a + b) % 2 != 0 || l > a + b || l < a.abs_diff(b) {
        return Ok(0.);
    }
    let prefactor = (((2 * l + 1) * (2 * a + 1) * (2 * b + 1)) as f64 / (4. * PI)).sqrt();
    let j = [l as i32, a as i32, b as i32];
    Ok(parity(m + x) * prefactor * wigner(j, [0, 0, 0]) * wigner(j, [-m, -x, y]))
}

/// Cartesian evaluation avoids recovering angles only to take their sines and
/// cosines again. At the origin the polar-axis convention is used.
pub fn cartesian(l: usize, m: i32, point: [f64; 3]) -> f64 {
    let r = point.iter().map(|v| v * v).sum::<f64>().sqrt();
    normalized(
        l,
        m,
        if r == 0. {
            [0., 0., 1.]
        } else {
            point.map(|v| v / r)
        },
    )
}
/// A direction already normalized by a radial sampler. s, p and d shells
/// reduce to Cartesian polynomials; higher shells use the same normalized
/// associated-Legendre recurrence in complex Cartesian form.
pub(crate) fn normalized(l: usize, m: i32, [x, y, z]: [f64; 3]) -> f64 {
    let order = m.unsigned_abs() as usize;
    if order > l {
        return 0.;
    }
    match (l, m) {
        (0, 0) => 1. / (4. * PI).sqrt(),
        (1, -1) => (3. / (4. * PI)).sqrt() * y,
        (1, 0) => (3. / (4. * PI)).sqrt() * z,
        (1, 1) => (3. / (4. * PI)).sqrt() * x,
        (2, -2) => (15. / (4. * PI)).sqrt() * x * y,
        (2, -1) => (15. / (4. * PI)).sqrt() * y * z,
        (2, 0) => (5. / (16. * PI)).sqrt() * (3. * z * z - 1.),
        (2, 1) => (15. / (4. * PI)).sqrt() * x * z,
        (2, 2) => (15. / (16. * PI)).sqrt() * (x * x - y * y),
        _ => {
            use num_complex::Complex64;
            let mut scale = 1. / (4. * PI).sqrt();
            for k in 1..=order {
                scale *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt();
            }
            let mut value = Complex64::new(x, y).powu(order as u32) * scale;
            if l > order {
                let mut previous = value;
                value *= ((2 * order + 3) as f64).sqrt() * z;
                for degree in order + 2..=l {
                    let d = degree as f64;
                    let o = order as f64;
                    let a = ((4. * d * d - 1.) / (d * d - o * o)).sqrt();
                    let b = (((d - 1.).powi(2) - o * o) / (4. * (d - 1.).powi(2) - 1.)).sqrt();
                    let next = a * (z * value - b * previous);
                    previous = value;
                    value = next;
                }
            }
            if m == 0 {
                value.re
            } else {
                2f64.sqrt()
                    * if order % 2 == 0 { 1. } else { -1. }
                    * if m > 0 { value.re } else { value.im }
            }
        }
    }
}

pub fn real(l: usize, m: i32, theta: f64, phi: f64) -> f64 {
    let order = m.unsigned_abs() as usize;
    let cos_t = theta.cos();
    let mut legendre = 1. / (4. * PI).sqrt();
    for k in 1..=order {
        legendre *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt() * theta.sin();
    }
    if l > order {
        let mut previous = legendre;
        legendre *= ((2 * order + 3) as f64).sqrt() * cos_t;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4. * d * d - 1.) / (d * d - o * o)).sqrt();
            let b = (((d - 1.).powi(2) - o * o) / (4. * (d - 1.).powi(2) - 1.)).sqrt();
            let next = a * (cos_t * legendre - b * previous);
            previous = legendre;
            legendre = next;
        }
    }
    if m == 0 {
        legendre
    } else {
        2f64.sqrt()
            * if order % 2 == 0 { 1. } else { -1. }
            * legendre
            * if m > 0 {
                (order as f64 * phi).cos()
            } else {
                (order as f64 * phi).sin()
            }
    }
}
