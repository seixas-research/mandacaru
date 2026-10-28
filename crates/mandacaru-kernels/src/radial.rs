// file: src/radial.rs
//
// Radial kernels of atom and pseudopotential generation in Rust.
// Numerov integration and tridiagonal Sturm bisection with twisted factorization.

pub const MANDACARU_RADIAL_ABI: i32 = 2;

#[no_mangle]
pub unsafe extern "C" fn mandacaru_radial_abi_version() -> i32 {
    MANDACARU_RADIAL_ABI
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_numerov_outward(
    n: i32,
    f_ptr: *const f64,
    s_ptr: *const f64,
    h2: f64,
    start: i32,
    u_ptr: *mut f64,
) {
    let n_u = n as usize;
    let st = start as usize;
    let f = std::slice::from_raw_parts(f_ptr, n_u);
    let s = std::slice::from_raw_parts(s_ptr, n_u);
    let u = std::slice::from_raw_parts_mut(u_ptr, n_u);

    let c = h2 / 12.0;
    for i in (st + 1)..(n_u - 1) {
        let a_im1 = 1.0 - h2 * f[i - 1] / 12.0;
        let b_i = 2.0 * (1.0 + 5.0 * h2 * f[i] / 12.0);
        let a_ip1 = 1.0 - h2 * f[i + 1] / 12.0;
        u[i + 1] =
            (b_i * u[i] - a_im1 * u[i - 1] + c * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / a_ip1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_numerov_inward(
    n: i32,
    f_ptr: *const f64,
    h2: f64,
    stop: i32,
    u_ptr: *mut f64,
) {
    let n_u = n as usize;
    let st = stop as usize;
    let f = std::slice::from_raw_parts(f_ptr, n_u);
    let u = std::slice::from_raw_parts_mut(u_ptr, n_u);

    for i in ((st + 1)..=(n_u - 2)).rev() {
        let a_ip1 = 1.0 - h2 * f[i + 1] / 12.0;
        let b_i = 2.0 * (1.0 + 5.0 * h2 * f[i] / 12.0);
        let a_im1 = 1.0 - h2 * f[i - 1] / 12.0;
        u[i - 1] = (b_i * u[i] - a_ip1 * u[i + 1]) / a_im1;
    }
}

#[inline(always)]
fn sturm_count(n: usize, d: &[f64], e2: &[f64], x: f64, pivmin: f64) -> i32 {
    let mut count = 0;
    let mut q = d[0] - x;
    if q.abs() < pivmin {
        q = -pivmin;
    }
    if q < 0.0 {
        count += 1;
    }
    for i in 1..n {
        q = d[i] - x - e2[i - 1] / q;
        if q.abs() < pivmin {
            q = -pivmin;
        }
        if q < 0.0 {
            count += 1;
        }
    }
    count
}

fn shifted_solve(
    n: usize,
    d: &[f64],
    e: &[f64],
    lambda: f64,
    x: &mut [f64],
    dl: &mut [f64],
    dd: &mut [f64],
    du: &mut [f64],
    du2: &mut [f64],
    tiny: f64,
) {
    for i in 0..n {
        dd[i] = d[i] - lambda;
    }
    for i in 0..(n - 1) {
        dl[i] = e[i];
        du[i] = e[i];
    }
    for i in 0..(n - 2) {
        du2[i] = 0.0;
    }

    for i in 0..(n - 1) {
        if dd[i].abs() >= dl[i].abs() {
            if dd[i] == 0.0 {
                dd[i] = tiny;
            }
            let m = dl[i] / dd[i];
            dd[i + 1] -= m * du[i];
            x[i + 1] -= m * x[i];
            if i < n - 2 {
                du2[i] = 0.0;
            }
            dl[i] = m;
        } else {
            let m = dd[i] / dl[i];
            dd[i] = dl[i];
            let t = dd[i + 1];
            dd[i + 1] = du[i] - m * t;
            if i < n - 2 {
                du2[i] = du[i + 1];
                du[i + 1] = -m * du2[i];
            }
            du[i] = t;
            let xt = x[i];
            x[i] = x[i + 1];
            x[i + 1] = xt - m * x[i + 1];
            dl[i] = m;
        }
    }
    if dd[n - 1] == 0.0 {
        dd[n - 1] = tiny;
    }

    // Back substitution
    x[n - 1] /= dd[n - 1];
    if n > 1 {
        x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / dd[n - 2];
    }
    for i in (0..(n - 2)).rev() {
        x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i];
    }
}

fn twisted_vector(
    n: usize,
    d: &[f64],
    e: &[f64],
    lambda: f64,
    pivmin: f64,
    dplus: &mut [f64],
    dminus: &mut [f64],
    x: &mut [f64],
) -> i32 {
    dplus[0] = d[0] - lambda;
    if dplus[0].abs() < pivmin {
        dplus[0] = -pivmin;
    }
    for i in 1..n {
        dplus[i] = d[i] - lambda - e[i - 1] * e[i - 1] / dplus[i - 1];
        if dplus[i].abs() < pivmin {
            dplus[i] = -pivmin;
        }
    }

    dminus[n - 1] = d[n - 1] - lambda;
    if dminus[n - 1].abs() < pivmin {
        dminus[n - 1] = -pivmin;
    }
    for i in (0..(n - 1)).rev() {
        dminus[i] = d[i] - lambda - e[i] * e[i] / dminus[i + 1];
        if dminus[i].abs() < pivmin {
            dminus[i] = -pivmin;
        }
    }

    let mut twist = 0;
    let mut best = f64::INFINITY;
    for i in 0..n {
        let gamma = (dplus[i] + dminus[i] - (d[i] - lambda)).abs();
        if gamma < best {
            best = gamma;
            twist = i;
        }
    }

    x[twist] = 1.0;
    for i in (0..twist).rev() {
        x[i] = -(e[i] / dplus[i]) * x[i + 1];
    }
    for i in (twist + 1)..n {
        x[i] = -(e[i - 1] / dminus[i]) * x[i - 1];
    }

    let mut norm = 0.0f64;
    for i in 0..n {
        norm += x[i] * x[i];
    }
    norm = norm.sqrt();
    if !(norm > 0.0) || !norm.is_finite() {
        return -3;
    }
    for i in 0..n {
        x[i] /= norm;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_tridiagonal_eigenpair(
    n: i32,
    d_ptr: *const f64,
    e_ptr: *const f64,
    k: i32,
    guess: f64,
    val_out: *mut f64,
    vec_out: *mut f64,
) -> i32 {
    let n_u = n as usize;
    let k_u = k as usize;
    if n < 1 || k < 0 || k_u >= n_u {
        return -1;
    }

    let d = std::slice::from_raw_parts(d_ptr, n_u);
    if n_u == 1 {
        *val_out = d[0];
        *vec_out = 1.0;
        return 0;
    }
    let e = std::slice::from_raw_parts(e_ptr, n_u - 1);
    let vector = std::slice::from_raw_parts_mut(vec_out, n_u);

    let mut e2 = vec![0.0f64; n_u - 1];
    let mut dl = vec![0.0f64; n_u];
    let mut dd = vec![0.0f64; n_u];
    let mut du = vec![0.0f64; n_u];
    let mut du2 = vec![0.0f64; n_u];
    let mut x = vec![0.0f64; n_u];

    let mut lo = d[0] - e[0].abs();
    let mut hi = d[0] + e[0].abs();
    let mut scale = 0.0f64;

    for i in 0..n_u {
        let left = if i > 0 { e[i - 1].abs() } else { 0.0 };
        let right = if i < n_u - 1 { e[i].abs() } else { 0.0 };
        let lower = d[i] - left - right;
        let upper = d[i] + left + right;
        if lower < lo {
            lo = lower;
        }
        if upper > hi {
            hi = upper;
        }
        let row_scale = d[i].abs() + left + right;
        if row_scale > scale {
            scale = row_scale;
        }
    }

    for i in 0..(n_u - 1) {
        e2[i] = e[i] * e[i];
    }

    let dbl_min = f64::MIN_POSITIVE;
    let dbl_eps = f64::EPSILON;
    let pivmin = dbl_min * if scale > 1.0 { scale } else { 1.0 };
    let pad = 2.0 * dbl_eps * scale + pivmin;
    lo -= pad;
    hi += pad;

    let tiny = dbl_eps * scale + pivmin;
    let full_lo = lo;
    let full_hi = hi;

    if guess.is_finite() && guess > lo && guess < hi {
        let mut step = 1e-4 * (1.0f64).max(guess.abs());
        let mut a = guess - step;
        let mut b = guess + step;
        while a > lo && sturm_count(n_u, d, &e2, a, pivmin) > k {
            step *= 4.0;
            a = guess - step;
        }
        step = 1e-4 * (1.0f64).max(guess.abs());
        while b < hi && sturm_count(n_u, d, &e2, b, pivmin) <= k {
            step *= 4.0;
            b = guess + step;
        }
        if a > lo {
            lo = a;
        }
        if b < hi {
            hi = b;
        }
    }

    for _it in 0..256 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if sturm_count(n_u, d, &e2, mid, pivmin) > k {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo <= 1e-6 * (1.0f64).max(mid.abs()) {
            break;
        }
    }

    let mut lambda = 0.5 * (lo + hi);
    let inv_sqrt_n = 1.0 / (n_u as f64).sqrt();
    for i in 0..n_u {
        vector[i] = inv_sqrt_n;
    }

    let mut accepted = false;
    for it in 0..8 {
        x.copy_from_slice(vector);
        shifted_solve(
            n_u, d, e, lambda, &mut x, &mut dl, &mut dd, &mut du, &mut du2, tiny,
        );
        let mut norm = 0.0f64;
        for i in 0..n_u {
            norm += x[i] * x[i];
        }
        norm = norm.sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            break;
        }
        for i in 0..n_u {
            x[i] /= norm;
        }
        vector.copy_from_slice(&x);

        let mut rq = 0.0f64;
        for i in 0..n_u {
            let mut tx = d[i] * vector[i];
            if i > 0 {
                tx += e[i - 1] * vector[i - 1];
            }
            if i < n_u - 1 {
                tx += e[i] * vector[i + 1];
            }
            rq += vector[i] * tx;
        }

        if rq < lo - 1e-3 * (hi - lo) - pad || rq > hi + 1e-3 * (hi - lo) + pad {
            break;
        }

        let change = (rq - lambda).abs();
        lambda = rq;
        if it >= 1 && change <= 8.0 * dbl_eps * scale {
            let margin = 64.0 * dbl_eps * scale + pivmin;
            accepted = sturm_count(n_u, d, &e2, rq - margin, pivmin) <= k
                && sturm_count(n_u, d, &e2, rq + margin, pivmin) > k;
            break;
        }
    }

    if !accepted {
        lo = full_lo;
        hi = full_hi;
        for _it in 0..256 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if sturm_count(n_u, d, &e2, mid, pivmin) > k {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lambda = 0.5 * (lo + hi);
        for i in 0..n_u {
            vector[i] = inv_sqrt_n;
        }
        for it in 0..6 {
            x.copy_from_slice(vector);
            shifted_solve(
                n_u, d, e, lambda, &mut x, &mut dl, &mut dd, &mut du, &mut du2, tiny,
            );
            let mut norm = 0.0f64;
            for i in 0..n_u {
                norm += x[i] * x[i];
            }
            norm = norm.sqrt();
            if !(norm > 0.0) || !norm.is_finite() {
                break;
            }
            let mut overlap = 0.0f64;
            for i in 0..n_u {
                x[i] /= norm;
                overlap += x[i] * vector[i];
            }
            vector.copy_from_slice(&x);
            if it > 0 && (overlap.abs() - 1.0).abs() < 1e-14 {
                break;
            }
        }
    }

    *val_out = lambda;
    twisted_vector(n_u, d, e, lambda, pivmin, &mut dl, &mut dd, vector)
}
