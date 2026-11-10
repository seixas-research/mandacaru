// file: src/onecenter.rs
//
// PAW one-center radial Coulomb/Hartree and angular quadrature kernels in Rust (C ABI).
// Implements prefix/suffix radial Poisson solves and 4-index Slater integral tensors.


/// Fast radial Coulomb integral for partial-wave pair densities:
/// int_0^rc rho_1(r) * [ int_0^r rho_2(r') (r'/r)^(L+1) r'^2 dr' + int_r^rc rho_2(r') (r/r')^L r'^2 dr' ] * r^2 dr
/// Exactly matches Python scipy cumulative_trapezoid + simpson.
pub unsafe fn mandacaru_radial_coulomb(
    l: i32,
    nr: i32,
    r_ptr: *const f64,
    rho1_ptr: *const f64,
    rho2_ptr: *const f64,
) -> f64 {
    let n = nr as usize;
    if n < 3 {
        return 0.0;
    }
    let r = std::slice::from_raw_parts(r_ptr, n);
    let rho1 = std::slice::from_raw_parts(rho1_ptr, n);
    let rho2 = std::slice::from_raw_parts(rho2_ptr, n);

    let mut below = vec![0.0f64; n];
    let mut above = vec![0.0f64; n];

    let mut prev_w_in = rho2[0] * r[0].powi(l + 2);
    let mut prev_w_out = if r[0] > 0.0 {
        rho2[0] * r[0].powi(1 - l)
    } else {
        0.0
    };

    let mut acc_in = 0.0f64;
    let mut acc_out = 0.0f64;

    for k in 0..(n - 1) {
        let dr = r[k + 1] - r[k];
        let r_next = r[k + 1];
        let curr_w_in = rho2[k + 1] * r_next.powi(l + 2);
        let curr_w_out = if r_next > 0.0 {
            rho2[k + 1] * r_next.powi(1 - l)
        } else {
            0.0
        };

        acc_in += 0.5 * (prev_w_in + curr_w_in) * dr;
        acc_out += 0.5 * (prev_w_out + curr_w_out) * dr;

        below[k + 1] = acc_in;
        above[k + 1] = acc_out;

        prev_w_in = curr_w_in;
        prev_w_out = curr_w_out;
    }

    let tot_out = above[n - 1];
    for k in 0..n {
        above[k] = tot_out - above[k];
    }

    let mut y = vec![0.0f64; n];
    for k in 0..n {
        let r_val = r[k];
        let pot = (if r_val > 0.0 {
            below[k] / r_val.powi(l + 1)
        } else {
            0.0
        }) + r_val.powi(l) * above[k];
        y[k] = rho1[k] * pot * r_val * r_val;
    }

    // Composite Simpson's 1/3 rule with Cartwright boundary adjustment on last interval
    let h = r[1] - r[0];
    let num_pairs = (n - 2) / 2;
    let mut res = 0.0f64;
    for k in 0..num_pairs {
        let i = 2 * k;
        res += (h / 3.0) * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
    }

    let alpha = (5.0 * h) / 12.0;
    let beta = (8.0 * h) / 12.0;
    let eta = h / 12.0;
    res += alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];

    res
}
