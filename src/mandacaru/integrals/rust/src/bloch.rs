use std::f64::consts::PI;

pub const BLOCK: usize = 256;

#[inline(always)]
pub fn radial_value(r: f64, breaks: &[f64], coeffs: &[f64], nb: usize, rc: f64, l: i32) -> f64 {
    let x0 = breaks[0];
    let x_end = breaks[nb - 1];
    if r > x_end || r >= rc {
        return 0.0;
    }
    let rr = if r < x0 { x0 } else { r };
    let mut lo = 0usize;
    let mut hi = nb - 2;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if breaks[mid] <= rr {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let c = &coeffs[4 * lo..4 * lo + 4];
    let t = rr - breaks[lo];
    let mut value = ((c[0] * t + c[1]) * t + c[2]) * t + c[3];
    if r < x0 && x0 > 0.0 && l > 0 {
        value *= (if r > 0.0 { r / x0 } else { 0.0 }).powi(l);
    }
    value
}

#[inline(always)]
pub fn radial_slope(r: f64, breaks: &[f64], coeffs: &[f64], nb: usize, rc: f64, l: i32) -> f64 {
    let x0 = breaks[0];
    let x_end = breaks[nb - 1];
    if r > x_end || r >= rc {
        return 0.0;
    }
    if r < x0 && x0 > 0.0 {
        if l == 0 {
            return 0.0;
        }
        let edge = radial_value(x0, breaks, coeffs, nb, rc, l);
        return (l as f64) * edge * (r / x0).powi(l - 1) / x0;
    }
    let mut lo = 0usize;
    let mut hi = nb - 2;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if breaks[mid] <= r {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let c = &coeffs[4 * lo..4 * lo + 4];
    let t = r - breaks[lo];
    (3.0 * c[0] * t + 2.0 * c[1]) * t + c[2]
}

#[inline(always)]
pub fn legendre(l: i32, m: i32, c: f64, s: f64) -> f64 {
    let mut p = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=m {
        p *= -((2.0 * k as f64 + 1.0) / (2.0 * k as f64)).sqrt() * s;
    }
    if l == m {
        return p;
    }
    let mut previous = p;
    p = (2.0 * m as f64 + 3.0).sqrt() * c * p;
    for d in (m + 2)..=l {
        let a = ((4.0 * (d * d) as f64 - 1.0) / ((d * d - m * m) as f64)).sqrt();
        let b = ((((d - 1) * (d - 1) - m * m) as f64)
            / (4.0 * ((d - 1) * (d - 1)) as f64 - 1.0)).sqrt();
        let next = a * (c * p - b * previous);
        previous = p;
        p = next;
    }
    p
}

#[inline(always)]
pub fn legendre_q(l: i32, m: i32, u: f64, q: &mut f64, dq: &mut f64) {
    let mut q0 = 1.0 / (4.0 * PI).sqrt();
    for k in 1..=m {
        q0 *= -((2.0 * k as f64 + 1.0) / (2.0 * k as f64)).sqrt();
    }
    let d0 = 0.0;
    if l == m {
        *q = q0;
        *dq = d0;
        return;
    }
    let mut q1 = (2.0 * m as f64 + 3.0).sqrt() * u * q0;
    let mut d1 = (2.0 * m as f64 + 3.0).sqrt() * q0;
    let mut prev_q0 = q0;
    let mut prev_d0 = d0;
    for d in (m + 2)..=l {
        let a = ((4.0 * (d * d) as f64 - 1.0) / ((d * d - m * m) as f64)).sqrt();
        let b = ((((d - 1) * (d - 1) - m * m) as f64)
            / (4.0 * ((d - 1) * (d - 1)) as f64 - 1.0)).sqrt();
        let q2 = a * (u * q1 - b * prev_q0);
        let d2 = a * (q1 + u * d1 - b * prev_d0);
        prev_q0 = q1;
        prev_d0 = d1;
        q1 = q2;
        d1 = d2;
    }
    *q = q1;
    *dq = d1;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }
    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    #[inline(always)]
    pub fn mul_real(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

#[inline(always)]
pub fn gradient_value(
    dx: f64,
    dy: f64,
    dz: f64,
    r: f64,
    axis: i32,
    l: i32,
    order: i32,
    r_val: f64,
    dr: f64,
    breaks: &[f64],
    coeffs: &[f64],
    nb: usize,
    rc: f64,
) -> Complex64 {
    if r < 1e-12 {
        if l != 1 {
            return Complex64::new(0.0, 0.0);
        }
        let slope = radial_slope(0.0, breaks, coeffs, nb, rc, l);
        let c0 = 1.0 / (4.0 * PI).sqrt();
        let g = if order == 0 {
            if axis == 2 {
                Complex64::new(3.0f64.sqrt() * c0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        } else {
            let pref = -1.5f64.sqrt() * c0;
            if axis == 0 {
                Complex64::new(pref, 0.0)
            } else if axis == 1 {
                Complex64::new(0.0, pref)
            } else {
                Complex64::new(0.0, 0.0)
            }
        };
        Complex64::new(g.re * slope, g.im * slope)
    } else {
        let x = [dx, dy, dz];
        let u = dz / r;
        let w = Complex64::new(dx / r, dy / r);
        let mut q = 0.0;
        let mut dq = 0.0;
        legendre_q(l, order, u, &mut q, &mut dq);
        let mut wm = Complex64::new(1.0, 0.0);
        let mut wm1 = Complex64::new(1.0, 0.0);
        for _ in 0..order {
            wm1 = wm;
            wm = wm.mul(w);
        }
        let du = ((if axis == 2 { 1.0 } else { 0.0 }) - u * x[axis as usize] / r) / r;
        let e = if axis == 0 {
            Complex64::new(1.0, 0.0)
        } else if axis == 1 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        let dw = Complex64::new(
            (e.re - w.re * x[axis as usize] / r) / r,
            (e.im - w.im * x[axis as usize] / r) / r,
        );
        let y = Complex64::new(q * wm.re, q * wm.im);
        let dy_ = if order > 0 {
            let term2 = Complex64::new(
                q * (order as f64) * (wm1.re * dw.re - wm1.im * dw.im),
                q * (order as f64) * (wm1.re * dw.im + wm1.im * dw.re),
            );
            Complex64::new(dq * du * wm.re + term2.re, dq * du * wm.im + term2.im)
        } else {
            Complex64::new(dq * du * wm.re, dq * du * wm.im)
        };
        Complex64::new(
            dr * (x[axis as usize] / r) * y.re + r_val * dy_.re,
            dr * (x[axis as usize] / r) * y.im + r_val * dy_.im,
        )
    }
}

#[inline(always)]
fn image_values(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    j0: usize,
    n: usize,
    center: &[f64],
    shift: &[f64],
    support2: f64,
    derivative: i32,
    l: i32,
    ms: &[i32],
    breaks: &[f64],
    coeffs: &[f64],
    rc: f64,
    values: &mut [Complex64],
    inside: &mut [usize],
) -> usize {
    let nm = ms.len();
    let nb = breaks.len();
    let mut n_inside = 0;
    for jj in 0..n {
        let j = j0 + jj;
        for i in 0..nm {
            values[i * BLOCK + jj] = Complex64::new(0.0, 0.0);
        }
        let dx = x[j] - center[0] - shift[0];
        let dy = y[j] - center[1] - shift[1];
        let dz = z[j] - center[2] - shift[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 >= support2 {
            continue;
        }
        inside[n_inside] = jj;
        n_inside += 1;
        let r = r2.sqrt();
        if derivative >= 0 {
            let r_val = if r >= 1e-12 {
                radial_value(r, breaks, coeffs, nb, rc, l)
            } else {
                0.0
            };
            let dr = if r >= 1e-12 {
                radial_slope(r, breaks, coeffs, nb, rc, l)
            } else {
                0.0
            };
            for i in 0..nm {
                let m_val = ms[i];
                let order = m_val.abs();
                let mut g = gradient_value(
                    dx, dy, dz, r, derivative, l, order, r_val, dr, breaks, coeffs, nb, rc,
                );
                if m_val < 0 {
                    let sign = if (order & 1) != 0 { -1.0 } else { 1.0 };
                    g = Complex64::new(sign * g.re, -sign * g.im);
                }
                values[i * BLOCK + jj] = g;
            }
            continue;
        }
        let radial = radial_value(r, breaks, coeffs, nb, rc, l);
        if radial == 0.0 {
            continue;
        }
        let rho = (dx * dx + dy * dy).sqrt();
        let cos_t = if r > 0.0 { dz / r } else { 0.0 };
        let sin_t = if r > 0.0 { rho / r } else { 1.0 };
        let unit = if rho > 0.0 {
            Complex64::new(dx / rho, dy / rho)
        } else {
            Complex64::new(1.0, 0.0)
        };
        for i in 0..nm {
            let m_val = ms[i];
            let order = m_val.abs();
            let mut e = Complex64::new(1.0, 0.0);
            for _ in 0..order {
                e = e.mul(unit);
            }
            let p_lm = legendre(l, order, cos_t, sin_t);
            let mut y_lm = Complex64::new(p_lm * e.re, p_lm * e.im);
            if m_val < 0 {
                let sign = if (order & 1) != 0 { -1.0 } else { 1.0 };
                y_lm = Complex64::new(sign * y_lm.re, -sign * y_lm.im);
            }
            values[i * BLOCK + jj] = Complex64::new(radial * y_lm.re, radial * y_lm.im);
        }
    }
    n_inside
}

#[inline(always)]
fn block_hits(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    j0: usize,
    n: usize,
    center: &[f64],
    translations: &[f64],
    n_images: usize,
    support2: f64,
    hits: &mut [usize],
) -> usize {
    let mut lo = [x[j0], y[j0], z[j0]];
    let mut hi = [x[j0], y[j0], z[j0]];
    for j in (j0 + 1)..(j0 + n) {
        let p = [x[j], y[j], z[j]];
        for a in 0..3 {
            if p[a] < lo[a] { lo[a] = p[a]; }
            if p[a] > hi[a] { hi[a] = p[a]; }
        }
    }
    let mut n_hits = 0;
    for r in 0..n_images {
        let mut gap2 = 0.0;
        for a in 0..3 {
            let c = center[a] + translations[3 * r + a];
            let g = if c < lo[a] { lo[a] - c } else if c > hi[a] { c - hi[a] } else { 0.0 };
            gap2 += g * g;
        }
        if gap2 < support2 {
            hits[n_hits] = r;
            n_hits += 1;
        }
    }
    n_hits
}

pub fn bloch_block() -> i64 {
    BLOCK as i64
}

pub unsafe fn real_shell(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    npts: i64,
    center: *const f64,
    support: f64,
    breaks: *const f64,
    coeffs: *const f64,
    nb: i32,
    rc: f64,
    l: i32,
    derivative: i32,
    ms: *const i32,
    rows: *const i64,
    nm: i32,
    _m: i64,
    out: *mut f64,
) {
    let support2 = support * support;
    let sqrt2 = std::f64::consts::SQRT_2;
    let npts_usize = npts as usize;
    let nb_usize = nb as usize;
    let nm_usize = nm as usize;

    let x_slice = std::slice::from_raw_parts(x, npts_usize);
    let y_slice = std::slice::from_raw_parts(y, npts_usize);
    let z_slice = std::slice::from_raw_parts(z, npts_usize);
    let center_slice = std::slice::from_raw_parts(center, 3);
    let breaks_slice = std::slice::from_raw_parts(breaks, nb_usize);
    let coeffs_slice = std::slice::from_raw_parts(coeffs, (nb_usize - 1) * 4);
    let ms_slice = std::slice::from_raw_parts(ms, nm_usize);
    let rows_slice = if !rows.is_null() {
        Some(std::slice::from_raw_parts(rows, nm_usize))
    } else {
        None
    };

    let n_threads = crate::runtime::thread_count();
    let chunk_size = (npts_usize + n_threads - 1) / n_threads;
    let out_ptr = out as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let start = t * chunk_size;
            let end = std::cmp::min(start + chunk_size, npts_usize);
            if start >= end {
                continue;
            }

            crate::runtime::spawn(s, move || {
                let out_raw = out_ptr as *mut f64;
                for j in start..end {
                    let dx = x_slice[j] - center_slice[0];
                    let dy = y_slice[j] - center_slice[1];
                    let dz = z_slice[j] - center_slice[2];
                    let r2 = dx * dx + dy * dy + dz * dz;

                    if r2 >= support2 {
                        for i in 0..nm_usize {
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = 0.0;
                            }
                        }
                        continue;
                    }

                    let r = r2.sqrt();
                    if derivative >= 0 {
                        let r_val = if r >= 1e-12 {
                            radial_value(r, breaks_slice, coeffs_slice, nb_usize, rc, l)
                        } else {
                            0.0
                        };
                        let dr = if r >= 1e-12 {
                            radial_slope(r, breaks_slice, coeffs_slice, nb_usize, rc, l)
                        } else {
                            0.0
                        };
                        for i in 0..nm_usize {
                            let m_val = ms_slice[i];
                            let order = m_val.abs();
                            let g = gradient_value(
                                dx, dy, dz, r, derivative, l, order, r_val, dr,
                                breaks_slice, coeffs_slice, nb_usize, rc,
                            );
                            let val = if order == 0 {
                                g.re
                            } else {
                                let scale = if (order & 1) != 0 { -sqrt2 } else { sqrt2 };
                                scale * (if m_val > 0 { g.re } else { g.im })
                            };
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = val;
                            }
                        }
                    } else {
                        let radial = radial_value(r, breaks_slice, coeffs_slice, nb_usize, rc, l);
                        if radial == 0.0 {
                            for i in 0..nm_usize {
                                let row = match rows_slice {
                                    Some(r) => r[i],
                                    None => i as i64,
                                };
                                unsafe {
                                    *out_raw.add(row as usize * npts_usize + j) = 0.0;
                                }
                            }
                            continue;
                        }
                        let rho = (dx * dx + dy * dy).sqrt();
                        let cos_t = if r > 0.0 { dz / r } else { 0.0 };
                        let sin_t = if r > 0.0 { rho / r } else { 1.0 };
                        let unit = if rho > 0.0 {
                            Complex64::new(dx / rho, dy / rho)
                        } else {
                            Complex64::new(1.0, 0.0)
                        };

                        for i in 0..nm_usize {
                            let m_val = ms_slice[i];
                            let order = m_val.abs();
                            let mut e = Complex64::new(1.0, 0.0);
                            for _ in 0..order {
                                e = e.mul(unit);
                            }
                            let p_lm = legendre(l, order, cos_t, sin_t);
                            let val = if order == 0 {
                                radial * p_lm
                            } else {
                                let scale = if (order & 1) != 0 { -sqrt2 } else { sqrt2 };
                                radial * (scale * p_lm * (if m_val > 0 { e.re } else { e.im }))
                            };
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = val;
                            }
                        }
                    }
                }
            });
        }
    });
}

pub unsafe fn real_shell_f32(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    npts: i64,
    center: *const f64,
    support: f64,
    breaks: *const f64,
    coeffs: *const f64,
    nb: i32,
    rc: f64,
    l: i32,
    derivative: i32,
    ms: *const i32,
    rows: *const i64,
    nm: i32,
    _m: i64,
    out: *mut f32,
) {
    let support2 = support * support;
    let sqrt2 = std::f64::consts::SQRT_2;
    let npts_usize = npts as usize;
    let nb_usize = nb as usize;
    let nm_usize = nm as usize;

    let x_slice = std::slice::from_raw_parts(x, npts_usize);
    let y_slice = std::slice::from_raw_parts(y, npts_usize);
    let z_slice = std::slice::from_raw_parts(z, npts_usize);
    let center_slice = std::slice::from_raw_parts(center, 3);
    let breaks_slice = std::slice::from_raw_parts(breaks, nb_usize);
    let coeffs_slice = std::slice::from_raw_parts(coeffs, (nb_usize - 1) * 4);
    let ms_slice = std::slice::from_raw_parts(ms, nm_usize);
    let rows_slice = if !rows.is_null() {
        Some(std::slice::from_raw_parts(rows, nm_usize))
    } else {
        None
    };

    let n_threads = crate::runtime::thread_count();
    let chunk_size = (npts_usize + n_threads - 1) / n_threads;
    let out_ptr = out as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let start = t * chunk_size;
            let end = std::cmp::min(start + chunk_size, npts_usize);
            if start >= end {
                continue;
            }

            crate::runtime::spawn(s, move || {
                let out_raw = out_ptr as *mut f32;
                for j in start..end {
                    let dx = x_slice[j] - center_slice[0];
                    let dy = y_slice[j] - center_slice[1];
                    let dz = z_slice[j] - center_slice[2];
                    let r2 = dx * dx + dy * dy + dz * dz;

                    if r2 >= support2 {
                        for i in 0..nm_usize {
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = 0.0;
                            }
                        }
                        continue;
                    }

                    let r = r2.sqrt();
                    if derivative >= 0 {
                        let r_val = if r >= 1e-12 {
                            radial_value(r, breaks_slice, coeffs_slice, nb_usize, rc, l)
                        } else {
                            0.0
                        };
                        let dr = if r >= 1e-12 {
                            radial_slope(r, breaks_slice, coeffs_slice, nb_usize, rc, l)
                        } else {
                            0.0
                        };
                        for i in 0..nm_usize {
                            let m_val = ms_slice[i];
                            let order = m_val.abs();
                            let g = gradient_value(
                                dx, dy, dz, r, derivative, l, order, r_val, dr,
                                breaks_slice, coeffs_slice, nb_usize, rc,
                            );
                            let val = if order == 0 {
                                g.re
                            } else {
                                let scale = if (order & 1) != 0 { -sqrt2 } else { sqrt2 };
                                scale * (if m_val > 0 { g.re } else { g.im })
                            };
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = val as f32;
                            }
                        }
                    } else {
                        let radial = radial_value(r, breaks_slice, coeffs_slice, nb_usize, rc, l);
                        if radial == 0.0 {
                            for i in 0..nm_usize {
                                let row = match rows_slice {
                                    Some(r) => r[i],
                                    None => i as i64,
                                };
                                unsafe {
                                    *out_raw.add(row as usize * npts_usize + j) = 0.0;
                                }
                            }
                            continue;
                        }
                        let rho = (dx * dx + dy * dy).sqrt();
                        let cos_t = if r > 0.0 { dz / r } else { 0.0 };
                        let sin_t = if r > 0.0 { rho / r } else { 1.0 };
                        let unit = if rho > 0.0 {
                            Complex64::new(dx / rho, dy / rho)
                        } else {
                            Complex64::new(1.0, 0.0)
                        };

                        for i in 0..nm_usize {
                            let m_val = ms_slice[i];
                            let order = m_val.abs();
                            let mut e = Complex64::new(1.0, 0.0);
                            for _ in 0..order {
                                e = e.mul(unit);
                            }
                            let p_lm = legendre(l, order, cos_t, sin_t);
                            let val = if order == 0 {
                                radial * p_lm
                            } else {
                                let scale = if (order & 1) != 0 { -sqrt2 } else { sqrt2 };
                                radial * (scale * p_lm * (if m_val > 0 { e.re } else { e.im }))
                            };
                            let row = match rows_slice {
                                Some(r) => r[i],
                                None => i as i64,
                            };
                            unsafe {
                                *out_raw.add(row as usize * npts_usize + j) = val as f32;
                            }
                        }
                    }
                }
            });
        }
    });
}

pub unsafe fn bloch_shell(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    npts: i64,
    center: *const f64,
    translations: *const f64,
    n_images: i32,
    phases: *const Complex64,
    nk: i32,
    support: f64,
    breaks: *const f64,
    coeffs: *const f64,
    nb: i32,
    rc: f64,
    l: i32,
    derivative: i32,
    ms: *const i32,
    rows: *const i64,
    nm: i32,
    m_dim: i64,
    out: *mut Complex64,
) {
    let support2 = support * support;
    let npts_usize = npts as usize;
    let nb_usize = nb as usize;
    let nm_usize = nm as usize;
    let n_images_usize = n_images as usize;
    let nk_usize = nk as usize;
    let m_dim_usize = m_dim as usize;

    let x_slice = std::slice::from_raw_parts(x, npts_usize);
    let y_slice = std::slice::from_raw_parts(y, npts_usize);
    let z_slice = std::slice::from_raw_parts(z, npts_usize);
    let center_slice = std::slice::from_raw_parts(center, 3);
    let trans_slice = std::slice::from_raw_parts(translations, 3 * n_images_usize);
    let phases_slice = std::slice::from_raw_parts(phases, nk_usize * n_images_usize);
    let breaks_slice = std::slice::from_raw_parts(breaks, nb_usize);
    let coeffs_slice = std::slice::from_raw_parts(coeffs, (nb_usize - 1) * 4);
    let ms_slice = std::slice::from_raw_parts(ms, nm_usize);
    let rows_slice = std::slice::from_raw_parts(rows, nm_usize);

    let n_blocks = (npts_usize + BLOCK - 1) / BLOCK;
    let n_threads = crate::runtime::thread_count();
    let blocks_per_thread = (n_blocks + n_threads - 1) / n_threads;

    let out_ptr = out as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let b_start = t * blocks_per_thread;
            let b_end = std::cmp::min(b_start + blocks_per_thread, n_blocks);
            if b_start >= b_end {
                continue;
            }

            crate::runtime::spawn(s, move || {
                let out_raw = out_ptr as *mut Complex64;
                let mut hits = vec![0usize; n_images_usize.max(1)];
                let mut inside = [0usize; BLOCK];
                let mut buffer = vec![Complex64::default(); n_images_usize.max(1) * nm_usize * BLOCK];

                for b in b_start..b_end {
                    let j0 = b * BLOCK;
                    let n = std::cmp::min(j0 + BLOCK, npts_usize) - j0;
                    let n_hits = block_hits(
                        x_slice, y_slice, z_slice, j0, n, center_slice,
                        trans_slice, n_images_usize, support2, &mut hits,
                    );
                    if n_hits == 0 {
                        continue;
                    }

                    for h in 0..n_hits {
                        let shift = &trans_slice[3 * hits[h]..3 * hits[h] + 3];
                        let buf_offset = h * nm_usize * BLOCK;
                        image_values(
                            x_slice, y_slice, z_slice, j0, n, center_slice, shift,
                            support2, derivative, l, ms_slice, breaks_slice, coeffs_slice,
                            rc, &mut buffer[buf_offset..buf_offset + nm_usize * BLOCK],
                            &mut inside,
                        );
                    }

                    for k in 0..nk_usize {
                        for i in 0..nm_usize {
                            let target_offset = (k * m_dim_usize + rows_slice[i] as usize) * npts_usize + j0;
                            let target = unsafe { out_raw.add(target_offset) };
                            for h in 0..n_hits {
                                let phase = phases_slice[k * n_images_usize + hits[h]];
                                let val_offset = (h * nm_usize + i) * BLOCK;
                                for jj in 0..n {
                                    let v = buffer[val_offset + jj];
                                    let prod = phase.mul(v);
                                    unsafe {
                                        let curr = *target.add(jj);
                                        *target.add(jj) = curr.add(prod);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

pub unsafe fn bloch_project(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    npts: i64,
    center: *const f64,
    translations: *const f64,
    n_images: i32,
    support: f64,
    breaks: *const f64,
    coeffs: *const f64,
    nb: i32,
    rc: f64,
    l: i32,
    derivative: i32,
    ms: *const i32,
    nm: i32,
    cols: *const Complex64,
    p: i32,
    out: *mut Complex64,
) {
    let support2 = support * support;
    let npts_usize = npts as usize;
    let nb_usize = nb as usize;
    let nm_usize = nm as usize;
    let n_images_usize = n_images as usize;
    let p_usize = p as usize;

    let x_slice = std::slice::from_raw_parts(x, npts_usize);
    let y_slice = std::slice::from_raw_parts(y, npts_usize);
    let z_slice = std::slice::from_raw_parts(z, npts_usize);
    let center_slice = std::slice::from_raw_parts(center, 3);
    let trans_slice = std::slice::from_raw_parts(translations, 3 * n_images_usize);
    let breaks_slice = std::slice::from_raw_parts(breaks, nb_usize);
    let coeffs_slice = std::slice::from_raw_parts(coeffs, (nb_usize - 1) * 4);
    let ms_slice = std::slice::from_raw_parts(ms, nm_usize);
    let cols_slice = std::slice::from_raw_parts(cols, p_usize * npts_usize);

    let total_size = n_images_usize * nm_usize * p_usize;
    let out_slice = std::slice::from_raw_parts_mut(out, total_size.max(1));
    for item in out_slice.iter_mut() {
        *item = Complex64::default();
    }

    let n_blocks = (npts_usize + BLOCK - 1) / BLOCK;
    let n_threads = crate::runtime::thread_count();
    let blocks_per_thread = (n_blocks + n_threads - 1) / n_threads;

    let sync_mutex = std::sync::Mutex::new(());
    let out_ptr = out as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let b_start = t * blocks_per_thread;
            let b_end = std::cmp::min(b_start + blocks_per_thread, n_blocks);
            if b_start >= b_end {
                continue;
            }

            let sync_ref = &sync_mutex;
            crate::runtime::spawn(s, move || {
                let out_raw = out_ptr as *mut Complex64;
                let mut hits = vec![0usize; n_images_usize.max(1)];
                let mut inside = [0usize; BLOCK];
                let mut buffer = vec![Complex64::default(); nm_usize * BLOCK];
                let mut local_out = vec![Complex64::default(); total_size.max(1)];

                for b in b_start..b_end {
                    let j0 = b * BLOCK;
                    let n = std::cmp::min(j0 + BLOCK, npts_usize) - j0;
                    let n_hits = block_hits(
                        x_slice, y_slice, z_slice, j0, n, center_slice,
                        trans_slice, n_images_usize, support2, &mut hits,
                    );
                    for h in 0..n_hits {
                        let r_idx = hits[h];
                        let shift = &trans_slice[3 * r_idx..3 * r_idx + 3];
                        let n_inside = image_values(
                            x_slice, y_slice, z_slice, j0, n, center_slice, shift,
                            support2, derivative, l, ms_slice, breaks_slice, coeffs_slice,
                            rc, &mut buffer, &mut inside,
                        );

                        let target_base = (r_idx * nm_usize) * p_usize;
                        for i in 0..nm_usize {
                            for col_idx in 0..p_usize {
                                let col_offset = col_idx * npts_usize + j0;
                                let mut sum = Complex64::new(0.0, 0.0);
                                for t_idx in 0..n_inside {
                                    let jj = inside[t_idx];
                                    let phi_conj = buffer[i * BLOCK + jj].conj();
                                    let col_val = cols_slice[col_offset + jj];
                                    sum = sum.add(phi_conj.mul(col_val));
                                }
                                let idx = target_base + i * p_usize + col_idx;
                                local_out[idx] = local_out[idx].add(sum);
                            }
                        }
                    }
                }

                let _guard = sync_ref.lock().unwrap();
                for k in 0..total_size {
                    unsafe {
                        let curr = *out_raw.add(k);
                        *out_raw.add(k) = curr.add(local_out[k]);
                    }
                }
            });
        }
    });
}
