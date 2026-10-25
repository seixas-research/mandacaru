// file: src/one_body.rs
//
// Real-space one-body integral kernels in Rust (C ABI).

use crate::bloch::Complex64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline(always)]
    pub fn new(re: f32, im: f32) -> Self {
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
}

pub fn get_num_threads() -> usize {
    if let Ok(val) = std::env::var("MANDACARU_NUM_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    if let Ok(val) = std::env::var("OMP_NUM_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

#[inline(always)]
fn idx3(i: usize, j: usize, k: usize, npts: usize) -> usize {
    (i * npts + j) * npts + k
}

#[inline(always)]
fn idx3g(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    (i * ny + j) * nz + k
}

fn laplacian_fd_slice(field: &[Complex64], lap: &mut [Complex64], npts: usize, dx: f64) {
    let inv_dx2 = 1.0 / (dx * dx);
    for i in 0..npts {
        for j in 0..npts {
            for k in 0..npts {
                let g = idx3(i, j, k, npts);
                let f_g = field[g];
                let mut acc_re = -6.0 * f_g.re;
                let mut acc_im = -6.0 * f_g.im;

                if i + 1 < npts {
                    let v = field[idx3(i + 1, j, k, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }
                if i > 0 {
                    let v = field[idx3(i - 1, j, k, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }
                if j + 1 < npts {
                    let v = field[idx3(i, j + 1, k, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }
                if j > 0 {
                    let v = field[idx3(i, j - 1, k, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }
                if k + 1 < npts {
                    let v = field[idx3(i, j, k + 1, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }
                if k > 0 {
                    let v = field[idx3(i, j, k - 1, npts)];
                    acc_re += v.re;
                    acc_im += v.im;
                }

                lap[g] = Complex64::new(acc_re * inv_dx2, acc_im * inv_dx2);
            }
        }
    }
}

fn laplacian_general_slice(
    field: &[Complex64],
    lap: &mut [Complex64],
    nx: usize,
    ny: usize,
    nz: usize,
    ginv: &[f64],
) {
    let g00 = ginv[0];
    let g11 = ginv[4];
    let g22 = ginv[8];
    let g01 = ginv[1];
    let g02 = ginv[2];
    let g12 = ginv[5];
    let diag = -2.0 * (g00 + g11 + g22);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let g = idx3g(i, j, k, ny, nz);
                let f_g = field[g];
                let mut acc_re = diag * f_g.re;
                let mut acc_im = diag * f_g.im;

                if i + 1 < nx {
                    let v = field[idx3g(i + 1, j, k, ny, nz)];
                    acc_re += g00 * v.re;
                    acc_im += g00 * v.im;
                }
                if i > 0 {
                    let v = field[idx3g(i - 1, j, k, ny, nz)];
                    acc_re += g00 * v.re;
                    acc_im += g00 * v.im;
                }
                if j + 1 < ny {
                    let v = field[idx3g(i, j + 1, k, ny, nz)];
                    acc_re += g11 * v.re;
                    acc_im += g11 * v.im;
                }
                if j > 0 {
                    let v = field[idx3g(i, j - 1, k, ny, nz)];
                    acc_re += g11 * v.re;
                    acc_im += g11 * v.im;
                }
                if k + 1 < nz {
                    let v = field[idx3g(i, j, k + 1, ny, nz)];
                    acc_re += g22 * v.re;
                    acc_im += g22 * v.im;
                }
                if k > 0 {
                    let v = field[idx3g(i, j, k - 1, ny, nz)];
                    acc_re += g22 * v.re;
                    acc_im += g22 * v.im;
                }

                if g01 != 0.0 {
                    let c = 0.5 * g01;
                    if i + 1 < nx && j + 1 < ny {
                        let v = field[idx3g(i + 1, j + 1, k, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                    if i + 1 < nx && j > 0 {
                        let v = field[idx3g(i + 1, j - 1, k, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if i > 0 && j + 1 < ny {
                        let v = field[idx3g(i - 1, j + 1, k, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if i > 0 && j > 0 {
                        let v = field[idx3g(i - 1, j - 1, k, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                }

                if g02 != 0.0 {
                    let c = 0.5 * g02;
                    if i + 1 < nx && k + 1 < nz {
                        let v = field[idx3g(i + 1, j, k + 1, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                    if i + 1 < nx && k > 0 {
                        let v = field[idx3g(i + 1, j, k - 1, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if i > 0 && k + 1 < nz {
                        let v = field[idx3g(i - 1, j, k + 1, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if i > 0 && k > 0 {
                        let v = field[idx3g(i - 1, j, k - 1, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                }

                if g12 != 0.0 {
                    let c = 0.5 * g12;
                    if j + 1 < ny && k + 1 < nz {
                        let v = field[idx3g(i, j + 1, k + 1, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                    if j + 1 < ny && k > 0 {
                        let v = field[idx3g(i, j + 1, k - 1, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if j > 0 && k + 1 < nz {
                        let v = field[idx3g(i, j - 1, k + 1, ny, nz)];
                        acc_re -= c * v.re;
                        acc_im -= c * v.im;
                    }
                    if j > 0 && k > 0 {
                        let v = field[idx3g(i, j - 1, k - 1, ny, nz)];
                        acc_re += c * v.re;
                        acc_im += c * v.im;
                    }
                }

                lap[g] = Complex64::new(acc_re, acc_im);
            }
        }
    }
}

pub unsafe fn laplacian_real(
    field: *const f64,
    lap: *mut f64,
    n: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    ginv: *const f64,
) {
    let n_usize = n as usize;
    let nx_usize = nx as usize;
    let ny_usize = ny as usize;
    let nz_usize = nz as usize;
    let ngrid = nx_usize * ny_usize * nz_usize;

    let ginv_slice = std::slice::from_raw_parts(ginv, 9);
    let g00 = ginv_slice[0];
    let g11 = ginv_slice[4];
    let g22 = ginv_slice[8];
    let g01 = ginv_slice[1];
    let g02 = ginv_slice[2];
    let g12 = ginv_slice[5];
    let diag = -2.0 * (g00 + g11 + g22);

    let n_threads = get_num_threads();
    let total_planes = n_usize * nx_usize;
    let planes_per_thread = (total_planes + n_threads - 1) / n_threads;

    let field_ptr = field as usize;
    let lap_ptr = lap as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let p_start = t * planes_per_thread;
            let p_end = std::cmp::min(p_start + planes_per_thread, total_planes);
            if p_start >= p_end {
                continue;
            }

            s.spawn(move || {
                let f_base = field_ptr as *const f64;
                let l_base = lap_ptr as *mut f64;

                for p in p_start..p_end {
                    let f = p / nx_usize;
                    let i = p % nx_usize;

                    let u = f_base.add(f * ngrid);
                    let out = l_base.add(f * ngrid);

                    for j in 0..ny_usize {
                        for k in 0..nz_usize {
                            let g = idx3g(i, j, k, ny_usize, nz_usize);
                            let mut acc = diag * *u.add(g);

                            let mut s_val = 0.0;
                            if i + 1 < nx_usize {
                                s_val += *u.add(idx3g(i + 1, j, k, ny_usize, nz_usize));
                            }
                            if i > 0 {
                                s_val += *u.add(idx3g(i - 1, j, k, ny_usize, nz_usize));
                            }
                            acc += g00 * s_val;

                            s_val = 0.0;
                            if j + 1 < ny_usize {
                                s_val += *u.add(idx3g(i, j + 1, k, ny_usize, nz_usize));
                            }
                            if j > 0 {
                                s_val += *u.add(idx3g(i, j - 1, k, ny_usize, nz_usize));
                            }
                            acc += g11 * s_val;

                            s_val = 0.0;
                            if k + 1 < nz_usize {
                                s_val += *u.add(idx3g(i, j, k + 1, ny_usize, nz_usize));
                            }
                            if k > 0 {
                                s_val += *u.add(idx3g(i, j, k - 1, ny_usize, nz_usize));
                            }
                            acc += g22 * s_val;

                            if g01 != 0.0 {
                                let mut c = 0.0;
                                if i + 1 < nx_usize && j + 1 < ny_usize {
                                    c += *u.add(idx3g(i + 1, j + 1, k, ny_usize, nz_usize));
                                }
                                if i + 1 < nx_usize && j > 0 {
                                    c -= *u.add(idx3g(i + 1, j - 1, k, ny_usize, nz_usize));
                                }
                                if i > 0 && j + 1 < ny_usize {
                                    c -= *u.add(idx3g(i - 1, j + 1, k, ny_usize, nz_usize));
                                }
                                if i > 0 && j > 0 {
                                    c += *u.add(idx3g(i - 1, j - 1, k, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g01 * c;
                            }

                            if g02 != 0.0 {
                                let mut c = 0.0;
                                if i + 1 < nx_usize && k + 1 < nz_usize {
                                    c += *u.add(idx3g(i + 1, j, k + 1, ny_usize, nz_usize));
                                }
                                if i + 1 < nx_usize && k > 0 {
                                    c -= *u.add(idx3g(i + 1, j, k - 1, ny_usize, nz_usize));
                                }
                                if i > 0 && k + 1 < nz_usize {
                                    c -= *u.add(idx3g(i - 1, j, k + 1, ny_usize, nz_usize));
                                }
                                if i > 0 && k > 0 {
                                    c += *u.add(idx3g(i - 1, j, k - 1, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g02 * c;
                            }

                            if g12 != 0.0 {
                                let mut c = 0.0;
                                if j + 1 < ny_usize && k + 1 < nz_usize {
                                    c += *u.add(idx3g(i, j + 1, k + 1, ny_usize, nz_usize));
                                }
                                if j + 1 < ny_usize && k > 0 {
                                    c -= *u.add(idx3g(i, j + 1, k - 1, ny_usize, nz_usize));
                                }
                                if j > 0 && k + 1 < nz_usize {
                                    c -= *u.add(idx3g(i, j - 1, k + 1, ny_usize, nz_usize));
                                }
                                if j > 0 && k > 0 {
                                    c += *u.add(idx3g(i, j - 1, k - 1, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g12 * c;
                            }

                            *out.add(g) = acc;
                        }
                    }
                }
            });
        }
    });
}

pub unsafe fn laplacian_real_f32(
    field: *const f32,
    lap: *mut f32,
    n: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    ginv: *const f64,
) {
    let n_usize = n as usize;
    let nx_usize = nx as usize;
    let ny_usize = ny as usize;
    let nz_usize = nz as usize;
    let ngrid = nx_usize * ny_usize * nz_usize;

    let ginv_slice = std::slice::from_raw_parts(ginv, 9);
    let g00 = ginv_slice[0] as f32;
    let g11 = ginv_slice[4] as f32;
    let g22 = ginv_slice[8] as f32;
    let g01 = ginv_slice[1] as f32;
    let g02 = ginv_slice[2] as f32;
    let g12 = ginv_slice[5] as f32;
    let diag = -2.0 * (g00 + g11 + g22);

    let n_threads = get_num_threads();
    let total_planes = n_usize * nx_usize;
    let planes_per_thread = (total_planes + n_threads - 1) / n_threads;

    let field_ptr = field as usize;
    let lap_ptr = lap as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let p_start = t * planes_per_thread;
            let p_end = std::cmp::min(p_start + planes_per_thread, total_planes);
            if p_start >= p_end {
                continue;
            }

            s.spawn(move || {
                let f_base = field_ptr as *const f32;
                let l_base = lap_ptr as *mut f32;

                for p in p_start..p_end {
                    let f = p / nx_usize;
                    let i = p % nx_usize;

                    let u = f_base.add(f * ngrid);
                    let out = l_base.add(f * ngrid);

                    for j in 0..ny_usize {
                        for k in 0..nz_usize {
                            let g = idx3g(i, j, k, ny_usize, nz_usize);
                            let mut acc = diag * *u.add(g);

                            let mut s_val = 0.0f32;
                            if i + 1 < nx_usize {
                                s_val += *u.add(idx3g(i + 1, j, k, ny_usize, nz_usize));
                            }
                            if i > 0 {
                                s_val += *u.add(idx3g(i - 1, j, k, ny_usize, nz_usize));
                            }
                            acc += g00 * s_val;

                            s_val = 0.0f32;
                            if j + 1 < ny_usize {
                                s_val += *u.add(idx3g(i, j + 1, k, ny_usize, nz_usize));
                            }
                            if j > 0 {
                                s_val += *u.add(idx3g(i, j - 1, k, ny_usize, nz_usize));
                            }
                            acc += g11 * s_val;

                            s_val = 0.0f32;
                            if k + 1 < nz_usize {
                                s_val += *u.add(idx3g(i, j, k + 1, ny_usize, nz_usize));
                            }
                            if k > 0 {
                                s_val += *u.add(idx3g(i, j, k - 1, ny_usize, nz_usize));
                            }
                            acc += g22 * s_val;

                            if g01 != 0.0 {
                                let mut c = 0.0f32;
                                if i + 1 < nx_usize && j + 1 < ny_usize {
                                    c += *u.add(idx3g(i + 1, j + 1, k, ny_usize, nz_usize));
                                }
                                if i + 1 < nx_usize && j > 0 {
                                    c -= *u.add(idx3g(i + 1, j - 1, k, ny_usize, nz_usize));
                                }
                                if i > 0 && j + 1 < ny_usize {
                                    c -= *u.add(idx3g(i - 1, j + 1, k, ny_usize, nz_usize));
                                }
                                if i > 0 && j > 0 {
                                    c += *u.add(idx3g(i - 1, j - 1, k, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g01 * c;
                            }

                            if g02 != 0.0 {
                                let mut c = 0.0f32;
                                if i + 1 < nx_usize && k + 1 < nz_usize {
                                    c += *u.add(idx3g(i + 1, j, k + 1, ny_usize, nz_usize));
                                }
                                if i + 1 < nx_usize && k > 0 {
                                    c -= *u.add(idx3g(i + 1, j, k - 1, ny_usize, nz_usize));
                                }
                                if i > 0 && k + 1 < nz_usize {
                                    c -= *u.add(idx3g(i - 1, j, k + 1, ny_usize, nz_usize));
                                }
                                if i > 0 && k > 0 {
                                    c += *u.add(idx3g(i - 1, j, k - 1, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g02 * c;
                            }

                            if g12 != 0.0 {
                                let mut c = 0.0f32;
                                if j + 1 < ny_usize && k + 1 < nz_usize {
                                    c += *u.add(idx3g(i, j + 1, k + 1, ny_usize, nz_usize));
                                }
                                if j + 1 < ny_usize && k > 0 {
                                    c -= *u.add(idx3g(i, j + 1, k - 1, ny_usize, nz_usize));
                                }
                                if j > 0 && k + 1 < nz_usize {
                                    c -= *u.add(idx3g(i, j - 1, k + 1, ny_usize, nz_usize));
                                }
                                if j > 0 && k > 0 {
                                    c += *u.add(idx3g(i, j - 1, k - 1, ny_usize, nz_usize));
                                }
                                acc += 0.5 * g12 * c;
                            }

                            *out.add(g) = acc;
                        }
                    }
                }
            });
        }
    });
}

pub unsafe fn one_body(
    psi: *const Complex64,
    vext: *const f64,
    m: i32,
    npts: i32,
    dx: f64,
    out_t: *mut Complex64,
    out_v: *mut Complex64,
) {
    let m_usize = m as usize;
    let npts_usize = npts as usize;
    let ngrid = npts_usize * npts_usize * npts_usize;
    let d_v = dx * dx * dx;

    let n_threads = get_num_threads();
    let kets_per_thread = (m_usize + n_threads - 1) / n_threads;

    let psi_slice = std::slice::from_raw_parts(psi, m_usize * ngrid);
    let vext_slice = std::slice::from_raw_parts(vext, ngrid);

    let out_t_ptr = out_t as usize;
    let out_v_ptr = out_v as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let b_start = t * kets_per_thread;
            let b_end = std::cmp::min(b_start + kets_per_thread, m_usize);
            if b_start >= b_end {
                continue;
            }

            s.spawn(move || {
                let out_t_raw = out_t_ptr as *mut Complex64;
                let out_v_raw = out_v_ptr as *mut Complex64;

                let mut lap_b = vec![Complex64::default(); ngrid];

                for b in b_start..b_end {
                    let pb = &psi_slice[b * ngrid..(b + 1) * ngrid];
                    laplacian_fd_slice(pb, &mut lap_b, npts_usize, dx);

                    for a in 0..m_usize {
                        let pa = &psi_slice[a * ngrid..(a + 1) * ngrid];
                        let mut t_acc = Complex64::default();
                        let mut v_acc = Complex64::default();

                        for g in 0..ngrid {
                            let ca = pa[g].conj();
                            let l_term = lap_b[g].mul_real(-0.5);
                            t_acc = t_acc.add(ca.mul(l_term));

                            let v_term = pb[g].mul_real(vext_slice[g]);
                            v_acc = v_acc.add(ca.mul(v_term));
                        }

                        *out_t_raw.add(a * m_usize + b) = t_acc.mul_real(d_v);
                        *out_v_raw.add(a * m_usize + b) = v_acc.mul_real(d_v);
                    }
                }
            });
        }
    });
}

pub unsafe fn one_body_general(
    psi: *const Complex64,
    vext: *const f64,
    m: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    ginv: *const f64,
    d_v: f64,
    out_t: *mut Complex64,
    out_v: *mut Complex64,
) {
    let m_usize = m as usize;
    let nx_usize = nx as usize;
    let ny_usize = ny as usize;
    let nz_usize = nz as usize;
    let ngrid = nx_usize * ny_usize * nz_usize;

    let n_threads = get_num_threads();
    let kets_per_thread = (m_usize + n_threads - 1) / n_threads;

    let psi_slice = std::slice::from_raw_parts(psi, m_usize * ngrid);
    let vext_slice = std::slice::from_raw_parts(vext, ngrid);
    let ginv_slice = std::slice::from_raw_parts(ginv, 9);

    let out_t_ptr = out_t as usize;
    let out_v_ptr = out_v as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let b_start = t * kets_per_thread;
            let b_end = std::cmp::min(b_start + kets_per_thread, m_usize);
            if b_start >= b_end {
                continue;
            }

            s.spawn(move || {
                let out_t_raw = out_t_ptr as *mut Complex64;
                let out_v_raw = out_v_ptr as *mut Complex64;

                let mut lap_b = vec![Complex64::default(); ngrid];

                for b in b_start..b_end {
                    let pb = &psi_slice[b * ngrid..(b + 1) * ngrid];
                    laplacian_general_slice(pb, &mut lap_b, nx_usize, ny_usize, nz_usize, ginv_slice);

                    for a in 0..m_usize {
                        let pa = &psi_slice[a * ngrid..(a + 1) * ngrid];
                        let mut t_acc = Complex64::default();
                        let mut v_acc = Complex64::default();

                        for g in 0..ngrid {
                            let ca = pa[g].conj();
                            let l_term = lap_b[g].mul_real(-0.5);
                            t_acc = t_acc.add(ca.mul(l_term));

                            let v_term = pb[g].mul_real(vext_slice[g]);
                            v_acc = v_acc.add(ca.mul(v_term));
                        }

                        *out_t_raw.add(a * m_usize + b) = t_acc.mul_real(d_v);
                        *out_v_raw.add(a * m_usize + b) = v_acc.mul_real(d_v);
                    }
                }
            });
        }
    });
}

pub unsafe fn kb_project(
    psi: *const Complex64,
    chi: *const Complex64,
    m: i32,
    p: i32,
    ngrid: i64,
    d_v: f64,
    out_p: *mut Complex64,
) {
    let m_usize = m as usize;
    let p_usize = p as usize;
    let ngrid_usize = ngrid as usize;

    let psi_slice = std::slice::from_raw_parts(psi, m_usize * ngrid_usize);
    let chi_slice = std::slice::from_raw_parts(chi, p_usize * ngrid_usize);

    let n_threads = get_num_threads();
    let total_pairs = m_usize * p_usize;
    let pairs_per_thread = (total_pairs + n_threads - 1) / n_threads;

    let out_p_ptr = out_p as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let pair_start = t * pairs_per_thread;
            let pair_end = std::cmp::min(pair_start + pairs_per_thread, total_pairs);
            if pair_start >= pair_end {
                continue;
            }

            s.spawn(move || {
                let out_p_raw = out_p_ptr as *mut Complex64;

                for pair in pair_start..pair_end {
                    let a = pair / p_usize;
                    let pj = pair % p_usize;

                    let pa = &psi_slice[a * ngrid_usize..(a + 1) * ngrid_usize];
                    let cp = &chi_slice[pj * ngrid_usize..(pj + 1) * ngrid_usize];

                    let mut acc = Complex64::default();
                    for g in 0..ngrid_usize {
                        acc = acc.add(pa[g].conj().mul(cp[g]));
                    }

                    *out_p_raw.add(pair) = acc.mul_real(d_v);
                }
            });
        }
    });
}

pub unsafe fn kb_project_f32(
    psi: *const Complex32,
    chi: *const Complex32,
    m: i32,
    p: i32,
    ngrid: i64,
    d_v: f64,
    out_p: *mut Complex64,
) {
    let m_usize = m as usize;
    let p_usize = p as usize;
    let ngrid_usize = ngrid as usize;

    let psi_slice = std::slice::from_raw_parts(psi, m_usize * ngrid_usize);
    let chi_slice = std::slice::from_raw_parts(chi, p_usize * ngrid_usize);

    let n_threads = get_num_threads();
    let total_pairs = m_usize * p_usize;
    let pairs_per_thread = (total_pairs + n_threads - 1) / n_threads;

    let out_p_ptr = out_p as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let pair_start = t * pairs_per_thread;
            let pair_end = std::cmp::min(pair_start + pairs_per_thread, total_pairs);
            if pair_start >= pair_end {
                continue;
            }

            s.spawn(move || {
                let out_p_raw = out_p_ptr as *mut Complex64;

                for pair in pair_start..pair_end {
                    let a = pair / p_usize;
                    let pj = pair % p_usize;

                    let pa = &psi_slice[a * ngrid_usize..(a + 1) * ngrid_usize];
                    let cp = &chi_slice[pj * ngrid_usize..(pj + 1) * ngrid_usize];

                    // Accumulate in f64 for mixed precision
                    let mut acc_re = 0.0f64;
                    let mut acc_im = 0.0f64;
                    for g in 0..ngrid_usize {
                        let ca_re = pa[g].re as f64;
                        let ca_im = -pa[g].im as f64;
                        let cp_re = cp[g].re as f64;
                        let cp_im = cp[g].im as f64;

                        acc_re += ca_re * cp_re - ca_im * cp_im;
                        acc_im += ca_re * cp_im + ca_im * cp_re;
                    }

                    *out_p_raw.add(pair) = Complex64::new(acc_re * d_v, acc_im * d_v);
                }
            });
        }
    });
}
