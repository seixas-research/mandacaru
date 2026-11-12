//! Spherical harmonics of a point given by its Cartesian offset from a center: the polar angles,
//! the orthonormal harmonic (real `S_lm` or complex `Y_l^m`, the standard stable recurrence of the
//! fully normalized associated Legendre functions) and its analytic gradient. The radial
//! orbitals of `basis::sampling` and the angular bindings of the interface share these.
use num_complex::Complex64;
use std::f64::consts::PI;

/// Offsets closer than this to a function's center are moved out to it along z: the gradient of
/// `R(r) Y_lm` is a smooth Cartesian function there, but its direction-dependent pieces are 0/0.
pub const NODE_OFFSET: f64 = 1e-12;
/// Regularizes `1/r` and the polar angle at the nucleus.
pub const R_EPS: f64 = 1e-15;

/// `(r, theta, phi)` of the offset `(x, y, z)`: `r` is the true radius (used to mask a hard
/// cutoff); the angles use a floored radius so they stay finite at the origin.
pub fn polar(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let r_safe = if r < R_EPS { R_EPS } else { r };
    let z_ratio = (z / r_safe).clamp(-1.0, 1.0);
    (r, z_ratio.acos(), y.atan2(x))
}

/// `(real value, complex value)` of the harmonic of degree `l` and order `m` at the angles
/// `(theta, phi)`: the real one fills the first entry for `real`, the complex one the second.
pub fn harmonic(
    l: usize,
    m: i32,
    theta: f64,
    phi: f64,
    real: bool,
) -> (f64, Complex64) {
    let order = m.unsigned_abs() as usize;
    if order > l {
        return (0.0, Complex64::default());
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut legendre = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=order {
        legendre *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt() * sin_t;
    }
    if l > order {
        let mut previous = legendre;
        legendre *= ((2 * order + 3) as f64).sqrt() * cos_t;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4.0 * d * d - 1.0) / (d * d - o * o)).sqrt();
            let b = (((d - 1.0).powi(2) - o * o) / (4.0 * (d - 1.0).powi(2) - 1.0)).sqrt();
            let next = a * (cos_t * legendre - b * previous);
            previous = legendre;
            legendre = next;
        }
    }
    if real {
        let val = if m == 0 {
            legendre
        } else {
            let trig = if m > 0 {
                (order as f64 * phi).cos()
            } else {
                (order as f64 * phi).sin()
            };
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            2.0_f64.sqrt() * sign * legendre * trig
        };
        (val, Complex64::default())
    } else {
        let exp_term = Complex64::from_polar(1.0, order as f64 * phi);
        let mut value = exp_term * legendre;
        if m < 0 {
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            value = value.conj() * sign;
        }
        (0.0, value)
    }
}

/// `(real value, real gradient, complex value, complex gradient)` of the harmonic at the Cartesian
/// offset `(x, y, z)` (`r > 0`), written as `Q_lm(z/r) ((x + iy)/r)^m`, a form smooth in Cartesian
/// coordinates, so there is no pole on the z axis.
pub fn harmonic_gradient(
    l: usize,
    m: i32,
    x: f64,
    y: f64,
    z: f64,
    real: bool,
) -> (f64, [f64; 3], Complex64, [Complex64; 3]) {
    let order = m.unsigned_abs() as usize;
    let r = (x * x + y * y + z * z).sqrt();
    let u = z / r;
    let w = Complex64::new(x / r, y / r);

    let mut q0 = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=order {
        q0 *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt();
    }
    let d0 = 0.0;
    let (q, dq) = if l == order {
        (q0, d0)
    } else {
        let mut q0_cur = q0;
        let mut d0_cur = d0;
        let mut q1 = ((2 * order + 3) as f64).sqrt() * u * q0;
        let mut d1 = ((2 * order + 3) as f64).sqrt() * q0;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4.0 * d * d - 1.0) / (d * d - o * o)).sqrt();
            let b = (((d - 1.0).powi(2) - o * o) / (4.0 * (d - 1.0).powi(2) - 1.0)).sqrt();
            let next_q = a * (u * q1 - b * q0_cur);
            let next_d = a * (q1 + u * d1 - b * d0_cur);
            q0_cur = q1;
            d0_cur = d1;
            q1 = next_q;
            d1 = next_d;
        }
        (q1, d1)
    };

    let wm = w.powu(order as u32);
    let wm1 = if order > 0 {
        w.powu((order - 1) as u32)
    } else {
        Complex64::default()
    };
    let mut value = wm * q;

    let coords = [x, y, z];
    let e_axes = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
    ];
    let mut grad = [Complex64::default(); 3];
    for axis in 0..3 {
        let xa = coords[axis];
        let du = (((if axis == 2 { 1.0 } else { 0.0 }) - u * xa / r)) / r;
        let dw = (e_axes[axis] - w * (xa / r)) / r;
        grad[axis] = wm * (dq * du) + (wm1 * dw) * (q * order as f64);
    }

    if real {
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        let scale = 2.0_f64.sqrt() * sign;
        if m == 0 {
            (
                value.re,
                [grad[0].re, grad[1].re, grad[2].re],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        } else if m > 0 {
            (
                scale * value.re,
                [scale * grad[0].re, scale * grad[1].re, scale * grad[2].re],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        } else {
            (
                scale * value.im,
                [scale * grad[0].im, scale * grad[1].im, scale * grad[2].im],
                Complex64::default(),
                [Complex64::default(); 3],
            )
        }
    } else {
        if m < 0 {
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            value = value.conj() * sign;
            for g in grad.iter_mut() {
                *g = g.conj() * sign;
            }
        }
        (0.0, [0.0; 3], value, grad)
    }
}
