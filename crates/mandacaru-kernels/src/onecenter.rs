// file: src/onecenter.rs
//
// PAW one-center radial Coulomb/Hartree and angular quadrature kernels in Rust (C ABI).
// Implements prefix/suffix radial Poisson solves and 4-index Slater integral tensors.

use std::f64::consts::PI;

/// Radial Poisson solver for a multipole L:
/// Computes v_L(r) from pair density rho(r) = u_1(r) * u_2(r):
/// I_in(r) = int_0^r rho(r') * r'^(L+2) dr'
/// I_out(r) = int_r^rc rho(r') / r'^(L-1) dr'
/// v_L(r) = (4pi / (2L + 1)) * ( I_in(r) / r^(L+1) + r^L * I_out(r) )
#[no_mangle]
pub unsafe extern "C" fn mandacaru_radial_poisson_solve(
    l: i32,
    nr: i32,
    r_ptr: *const f64,
    dr_ptr: *const f64,
    rho_ptr: *const f64,
    v_out_ptr: *mut f64,
) {
    let n = nr as usize;
    let r = std::slice::from_raw_parts(r_ptr, n);
    let dr = std::slice::from_raw_parts(dr_ptr, n);
    let rho = std::slice::from_raw_parts(rho_ptr, n);
    let v_out = std::slice::from_raw_parts_mut(v_out_ptr, n);

    let mut i_in = vec![0.0f64; n];
    let mut i_out = vec![0.0f64; n];

    // Forward pass for I_in
    let mut acc_in = 0.0f64;
    for i in 0..n {
        let r_val = r[i];
        let term = rho[i] * r_val.powi(l + 2) * dr[i];
        acc_in += term;
        i_in[i] = acc_in;
    }

    // Backward pass for I_out
    let mut acc_out = 0.0f64;
    for i in (0..n).rev() {
        let r_val = r[i];
        let term = if r_val > 1e-12 {
            rho[i] / r_val.powi(l - 1) * dr[i]
        } else {
            0.0
        };
        acc_out += term;
        i_out[i] = acc_out;
    }

    let pref = 4.0 * PI / (2 * l + 1) as f64;
    for i in 0..n {
        let r_val = r[i];
        if r_val > 1e-12 {
            let part1 = i_in[i] / r_val.powi(l + 1);
            let part2 = r_val.powi(l) * i_out[i];
            v_out[i] = pref * (part1 + part2);
        } else {
            v_out[i] = 0.0;
        }
    }
}

/// Compute 4-index Slater Coulomb tensor W[i, j, k, l] for a given multipole L:
/// W[i, j, k, l] = int_0^rc u_k(r) * u_l(r) * v_L[u_i * u_j](r) dr
#[no_mangle]
pub unsafe extern "C" fn mandacaru_radial_slater_tensor(
    l: i32,
    n_waves: i32,
    nr: i32,
    r_ptr: *const f64,
    dr_ptr: *const f64,
    waves_ptr: *const f64, // [n_waves * nr]
    w_out_ptr: *mut f64,   // [n_waves * n_waves * n_waves * n_waves]
) {
    let nw = n_waves as usize;
    let n = nr as usize;
    let r = std::slice::from_raw_parts(r_ptr, n);
    let dr = std::slice::from_raw_parts(dr_ptr, n);

    let stride_i = nw * nw * nw;
    let stride_j = nw * nw;
    let stride_k = nw;

    // Allocate scratch for v_L
    let mut rho_ij = vec![0.0f64; n];
    let mut v_ij = vec![0.0f64; n];

    for i in 0..nw {
        let u_i = std::slice::from_raw_parts(waves_ptr.add(i * n), n);
        for j in 0..nw {
            let u_j = std::slice::from_raw_parts(waves_ptr.add(j * n), n);
            for p in 0..n {
                rho_ij[p] = u_i[p] * u_j[p];
            }

            // Solve radial Poisson for pair (i, j)
            mandacaru_radial_poisson_solve(
                l,
                nr,
                r.as_ptr(),
                dr.as_ptr(),
                rho_ij.as_ptr(),
                v_ij.as_mut_ptr(),
            );

            for k in 0..nw {
                let u_k = std::slice::from_raw_parts(waves_ptr.add(k * n), n);
                for l_idx in 0..nw {
                    let u_l = std::slice::from_raw_parts(waves_ptr.add(l_idx * n), n);

                    let mut integral = 0.0f64;
                    for p in 0..n {
                        integral += u_k[p] * u_l[p] * v_ij[p] * dr[p];
                    }

                    let out_idx = i * stride_i + j * stride_j + k * stride_k + l_idx;
                    *w_out_ptr.add(out_idx) = integral;
                }
            }
        }
    }
}

/// Fast radial Coulomb integral for partial-wave pair densities:
/// int_0^rc rho_1(r) * [ int_0^r rho_2(r') (r'/r)^(L+1) r'^2 dr' + int_r^rc rho_2(r') (r/r')^L r'^2 dr' ] * r^2 dr
/// Exactly matches Python scipy cumulative_trapezoid + simpson.
#[no_mangle]
pub unsafe extern "C" fn mandacaru_radial_coulomb(
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
