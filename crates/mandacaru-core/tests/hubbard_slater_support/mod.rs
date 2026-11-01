//! Helpers of `hubbard_slater.rs`: the frozen-reference loader, the double-sphere quadrature,
//! determinant energies from the tensor and the exact hydrogenic Slater integrals.
#![allow(dead_code)]
use mandacaru_core::math::angular;
use mandacaru_core::math::radial::gauss_legendre;
use mandacaru_core::models::hubbard_slater::{Basis, Coupling};
use num_complex::Complex64 as C64;
use serde_json::Value;
use std::f64::consts::PI;

// ---- reference file ------------------------------------------------------------------------

pub fn reference() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/hubbard/slater.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("frozen reference {path} is missing ({e}); copy the freeze_hubbard_slater.py output")
    });
    serde_json::from_str(&text).unwrap()
}
pub fn fraction(s: &str) -> f64 {
    match s.split_once('/') {
        Some((n, d)) => n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap(),
        None => s
            .parse()
            .unwrap_or_else(|_| panic!("unparsable reference number {s:?}")),
    }
}
/// The six shapes `freeze_hubbard_slater.py` writes: `N/N`, `-N/N`, `sqrt(N/N)`, `-sqrt(N/N)`,
/// `N/N*sqrt(N)`, `-N/N*sqrt(N)`; anything else panics.
pub fn number(s: &str) -> f64 {
    let (sign, s) = s.strip_prefix('-').map_or((1., s), |r| (-1., r));
    let (head, root) = match s.split_once("*sqrt(") {
        Some((h, r)) => (
            h,
            r.strip_suffix(')').unwrap().parse::<f64>().unwrap().sqrt(),
        ),
        None => (s, 1.),
    };
    let value = match head.strip_prefix("sqrt(") {
        Some(inner) => fraction(inner.strip_suffix(')').unwrap()).sqrt(),
        None => fraction(head),
    };
    sign * value * root
}
pub fn name(l: usize) -> &'static str {
    if l == 2 {
        "d"
    } else {
        "f"
    }
}
pub fn dense(entry: &Value, size: usize) -> Vec<f64> {
    let mut out = vec![0.; size];
    for (i, v) in entry["idx"]
        .as_array()
        .unwrap()
        .iter()
        .zip(entry["val"].as_array().unwrap())
    {
        out[i.as_u64().unwrap() as usize] = number(v.as_str().unwrap());
    }
    out
}
pub fn spread(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}

/// Per-`k` coefficients of `E_Hub = 1/2 sum (<ij|v|ij> - delta_ss' <ij|v|ji>)` and of
/// `E_dc = U/2 N(N-1) - J/2 sum_s N_s(N_s-1)` for a determinant (orbital, spin) list.
pub fn determinant(c: &Coupling, occ: &[(usize, bool)]) -> (Vec<f64>, Vec<f64>) {
    let n = occ.len() as f64;
    let n_up = occ.iter().filter(|o| o.1).count() as f64;
    let n_dn = n - n_up;
    let w = c.j_weights().unwrap();
    let mut e_int = vec![0.; w.len()];
    for (i, &k) in c.orders().iter().enumerate() {
        for &(a, sa) in occ {
            for &(b, sb) in occ {
                let direct = c.value(k, [a, b, a, b]).unwrap();
                let exchange = if sa == sb {
                    c.value(k, [a, b, b, a]).unwrap()
                } else {
                    0.
                };
                e_int[i] += 0.5 * (direct - exchange);
            }
        }
    }
    let e_dc = c
        .orders()
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            let hartree = if k == 0 { n * (n - 1.) / 2. } else { 0. };
            hartree - w[i] * (n_up * (n_up - 1.) + n_dn * (n_dn - 1.)) / 2.
        })
        .collect();
    (e_int, e_dc)
}

pub fn legendre(k: usize, x: f64) -> f64 {
    let (mut a, mut b) = (1., x);
    if k == 0 {
        return 1.;
    }
    for n in 2..=k {
        (a, b) = (
            b,
            ((2 * n - 1) as f64 * x * b - (n - 1) as f64 * a) / n as f64,
        );
    }
    b
}

/// `<ab|P_k(cos gamma)|cd>` by a product Gauss-Legendre x trapezoid rule on the sphere, exact for
/// every polynomial degree used (8 nodes in cos(theta): degree 15; 16 in phi: orders < 16).
pub fn quadrature(l: usize, basis: Basis, k: usize) -> Vec<f64> {
    let p = 2 * l + 1;
    let (nodes, n_phi) = (gauss_legendre(8).unwrap(), 16);
    let mut points = Vec::new();
    for &(x, w) in &nodes {
        for j in 0..n_phi {
            let (theta, phi) = (x.acos(), 2. * PI * j as f64 / n_phi as f64);
            points.push((theta, phi, w * 2. * PI / n_phi as f64));
        }
    }
    let n = points.len();
    let orbital = |r: usize, &(theta, phi, _): &(f64, f64, f64)| -> C64 {
        let m = r as i32 - l as i32;
        match basis {
            Basis::Complex => angular::complex(
                l,
                m,
                [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ],
            ),
            Basis::Real => C64::new(angular::real(l, m, theta, phi), 0.),
        }
    };
    let phis: Vec<Vec<C64>> = (0..p)
        .map(|r| points.iter().map(|pt| orbital(r, pt)).collect())
        .collect();
    // pair[(a, c)][i] = w_i conj(phi_a) phi_c at point i
    let pair = |a: usize, c: usize| -> Vec<C64> {
        (0..n)
            .map(|i| phis[a][i].conj() * phis[c][i] * points[i].2)
            .collect()
    };
    let unit = |&(t, f, _): &(f64, f64, f64)| [t.sin() * f.cos(), t.sin() * f.sin(), t.cos()];
    let kernel: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (u, v) = (unit(&points[ij / n]), unit(&points[ij % n]));
            legendre(k, u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
        })
        .collect();
    let pairs: Vec<Vec<C64>> = (0..p * p).map(|ac| pair(ac / p, ac % p)).collect();
    // (K g_bd)[i], then contract with f_ac
    let mapped: Vec<Vec<C64>> = pairs
        .iter()
        .map(|g| {
            (0..n)
                .map(|i| (0..n).map(|j| g[j] * kernel[i * n + j]).sum())
                .collect()
        })
        .collect();
    let mut out = vec![0.; p.pow(4)];
    for ac in 0..p * p {
        for bd in 0..p * p {
            let value: C64 = (0..n).map(|i| pairs[ac][i] * mapped[bd][i]).sum();
            assert!(value.im.abs() < 1e-12);
            // a = ac / p, c = ac % p, b = bd / p, d = bd % p
            let (a, cc, b, d) = (ac / p, ac % p, bd / p, bd % p);
            out[((a * p + b) * p + cc) * p + d] = value.re;
        }
    }
    out
}

/// `F^k` of the radial density `x^(2l+2) exp(-x)` (hydrogenic n = l+1 shells: the ratios do not
/// depend on the exponent) by the exact piecewise-integrated double integral.
pub fn hydrogenic(l: usize, k: usize) -> f64 {
    let fact = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
    let n = 2 * l + 2;
    let inner: f64 = (0..=n + k)
        .map(|j| fact(n - k - 1 + j) / (fact(j) * 2f64.powi((n - k + j) as i32)))
        .sum();
    let first = fact(n + k) * (fact(n - k - 1) - inner);
    let second = fact(n - k - 1)
        * (0..n - k)
            .map(|j| fact(n + k + j) / (fact(j) * 2f64.powi((n + k + j + 1) as i32)))
            .sum::<f64>();
    first + second
}
