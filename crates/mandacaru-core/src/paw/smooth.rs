//! The Bessel basis of the smooth partial waves and the record a channel keeps
//! of its expansion; the optimization itself is [`super::pseudize`].
use super::invalid;
use crate::{Buffer, Result};
use mandacaru_kernels::two_center::spherical_bessel as j;
use nalgebra::DMatrix as M;

pub struct SmoothWaves {
    pub wavevectors: Vec<f64>,
    pub coefficients: Buffer<f64>,
    pub norms: M<f64>,
    pub residuals: Vec<f64>,
}
pub fn bessel(l: usize, x: f64) -> [f64; 4] {
    let value = j(l as i32, x);
    let first = if l == 0 {
        -j(1, x)
    } else {
        j(l as i32 - 1, x) - (l + 1) as f64 * value / x
    };
    let ll = (l * (l + 1)) as f64;
    let second = -2. * first / x - (1. - ll / (x * x)) * value;
    let third = 2. * first / (x * x)
        - 2. * second / x
        - 2. * ll * value / x.powi(3)
        - (1. - ll / (x * x)) * first;
    [value, first, second, third]
}
pub fn wavevectors(l: usize, radius: f64, count: usize) -> Result<Vec<f64>> {
    if l > 8 || !radius.is_finite() || radius <= 0. || !(5..=32).contains(&count) {
        return Err(invalid("invalid PAW Bessel basis"));
    }
    let mut roots = vec![];
    for derivative in [false, true] {
        let f = |x| bessel(l, x)[usize::from(derivative)];
        let mut lower = 0.1;
        let mut last = f(lower);
        let mut found = 0;
        while lower < 400. && found < count {
            let upper = lower + 0.01;
            let next = f(upper);
            if last * next < 0. {
                let (mut a, mut b) = (lower, upper);
                let mut fa = last;
                for _ in 0..60 {
                    let c = 0.5 * (a + b);
                    if c == a || c == b {
                        break;
                    }
                    let fc = f(c);
                    if fa * fc <= 0. {
                        b = c;
                    } else {
                        a = c;
                        fa = fc;
                    }
                }
                roots.push(0.5 * (a + b) / radius);
                found += 1;
            }
            lower = upper;
            last = next;
        }
    }
    roots.sort_by(f64::total_cmp);
    if roots.len() < count {
        return Err(invalid("insufficient Bessel matching roots"));
    }
    roots.truncate(count);
    Ok(roots)
}
