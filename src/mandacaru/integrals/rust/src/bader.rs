// file: src/bader.rs
//
// Steepest-ascent trajectories of a density for the Bader partition.

use crate::one_body::get_num_threads;

#[inline(always)]
fn interp(r: f64, x: &[f64], y: &[f64], n: usize) -> f64 {
    if r <= x[0] {
        return y[0];
    }
    if r >= x[n - 1] {
        return y[n - 1];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x[mid] <= r {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (r - x[lo]) / (x[hi] - x[lo]);
    y[lo] + t * (y[hi] - y[lo])
}

#[inline(always)]
fn nearbyint(x: f64) -> f64 {
    // round half to even, matching C99 nearbyint / np.round
    let r = x.round();
    if (x - r).abs() == 0.5 {
        if (r as i64) % 2 != 0 {
            if x > r {
                r + 1.0
            } else {
                r - 1.0
            }
        } else {
            r
        }
    } else {
        r
    }
}

#[inline(always)]
fn offset(
    x: &[f64; 3],
    a: usize,
    periodic: bool,
    positions: &[f64],
    frac_atoms: &[f64],
    l_mat: &[f64],
    l_inv: &[f64],
    v: &mut [f64; 3],
) {
    if !periodic {
        for i in 0..3 {
            v[i] = x[i] - positions[3 * a + i];
        }
        return;
    }
    let mut delta = [0.0; 3];
    for i in 0..3 {
        let mut f = 0.0;
        for j in 0..3 {
            f += l_inv[3 * i + j] * x[j];
        }
        delta[i] = f - frac_atoms[3 * a + i];
        delta[i] -= nearbyint(delta[i]);
    }
    for i in 0..3 {
        v[i] = l_mat[3 * i] * delta[0]
            + l_mat[3 * i + 1] * delta[1]
            + l_mat[3 * i + 2] * delta[2];
    }
}

#[inline(always)]
fn trilinear(f: &[f64], p: &[f64; 3], n: &[usize; 3], periodic: bool) -> f64 {
    let mut i0 = [0usize; 3];
    let mut i1 = [0usize; 3];
    let mut t = [0.0; 3];

    for a in 0..3 {
        let fl = p[a].floor();
        t[a] = p[a] - fl;
        let mut i = fl as i64;
        let na = n[a] as i64;

        if periodic {
            i = i.rem_euclid(na);
            i0[a] = i as usize;
            i1[a] = ((i + 1) % na) as usize;
        } else {
            if i < 0 {
                i = 0;
                t[a] = 0.0;
            }
            if i > na - 1 {
                i = na - 1;
                t[a] = 0.0;
            }
            i0[a] = i as usize;
            i1[a] = if i + 1 < na { (i + 1) as usize } else { (na - 1) as usize };
        }
    }

    let mut out = 0.0;
    for c in 0..8 {
        let ix = if (c & 4) != 0 { i1[0] } else { i0[0] };
        let iy = if (c & 2) != 0 { i1[1] } else { i0[1] };
        let iz = if (c & 1) != 0 { i1[2] } else { i0[2] };

        let w = (if (c & 4) != 0 { t[0] } else { 1.0 - t[0] })
            * (if (c & 2) != 0 { t[1] } else { 1.0 - t[1] })
            * (if (c & 1) != 0 { t[2] } else { 1.0 - t[2] });

        out += w * f[(ix * n[1] + iy) * n[2] + iz];
    }
    out
}

pub unsafe fn bader_ascent(
    starts: *const f64,
    n_starts: i64,
    gx: *const f64,
    gy: *const f64,
    gz: *const f64,
    n0: i32,
    n1: i32,
    n2: i32,
    periodic: i32,
    a_mat: *const f64,
    a_inv: *const f64,
    origin: *const f64,
    l_mat: *const f64,
    l_inv: *const f64,
    positions: *const f64,
    frac_atoms: *const f64,
    n_atoms: i32,
    core_offsets: *const i64,
    core_r: *const f64,
    core_slope: *const f64,
    core_support: *const f64,
    step_length: f64,
    capture: f64,
    max_steps: i32,
    stall_window: i32,
    stall_distance: f64,
    owner: *mut i32,
    final_index: *mut f64,
) {
    let n = [n0 as usize, n1 as usize, n2 as usize];
    let ngrid = n[0] * n[1] * n[2];
    let n_starts_usize = n_starts as usize;
    let n_atoms_usize = n_atoms as usize;
    let is_periodic = periodic != 0;

    let starts_slice = std::slice::from_raw_parts(starts, n_starts_usize * 3);
    let gx_slice = std::slice::from_raw_parts(gx, ngrid);
    let gy_slice = std::slice::from_raw_parts(gy, ngrid);
    let gz_slice = std::slice::from_raw_parts(gz, ngrid);
    let fields = [gx_slice, gy_slice, gz_slice];

    let a_slice = std::slice::from_raw_parts(a_mat, 9);
    let a_inv_slice = std::slice::from_raw_parts(a_inv, 9);
    let origin_slice = std::slice::from_raw_parts(origin, 3);
    let l_slice = std::slice::from_raw_parts(l_mat, 9);
    let l_inv_slice = std::slice::from_raw_parts(l_inv, 9);
    let pos_slice = std::slice::from_raw_parts(positions, n_atoms_usize * 3);
    let frac_slice = std::slice::from_raw_parts(frac_atoms, n_atoms_usize * 3);
    let core_offsets_slice = std::slice::from_raw_parts(core_offsets, n_atoms_usize + 1);
    let total_core = core_offsets_slice[n_atoms_usize] as usize;
    let core_r_slice = std::slice::from_raw_parts(core_r, total_core);
    let core_slope_slice = std::slice::from_raw_parts(core_slope, total_core);
    let core_support_slice = std::slice::from_raw_parts(core_support, n_atoms_usize);

    let n_threads = get_num_threads();
    let starts_per_thread = (n_starts_usize + n_threads - 1) / n_threads;

    let owner_ptr = owner as usize;
    let final_ptr = final_index as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let s_start = t * starts_per_thread;
            let s_end = std::cmp::min(s_start + starts_per_thread, n_starts_usize);
            if s_start >= s_end {
                continue;
            }

            s.spawn(move || {
                let owner_raw = owner_ptr as *mut i32;
                let final_raw = final_ptr as *mut f64;

                for s_idx in s_start..s_end {
                    let mut p = [
                        starts_slice[3 * s_idx],
                        starts_slice[3 * s_idx + 1],
                        starts_slice[3 * s_idx + 2],
                    ];
                    let mut unwrapped = [p[0], p[1], p[2]];
                    let mut anchor = [p[0], p[1], p[2]];
                    let mut result = -1i32;

                    for count in 1..=max_steps {
                        if count % stall_window == 0 {
                            let mut moved = 0.0;
                            for i in 0..3 {
                                let mut c = 0.0;
                                for d in 0..3 {
                                    c += a_slice[3 * i + d] * (unwrapped[d] - anchor[d]);
                                }
                                moved += c * c;
                            }
                            if moved.sqrt() < stall_distance {
                                break;
                            }
                            for d in 0..3 {
                                anchor[d] = unwrapped[d];
                            }
                        }

                        let mut x = [0.0; 3];
                        for i in 0..3 {
                            x[i] = origin_slice[i]
                                + a_slice[3 * i] * p[0]
                                + a_slice[3 * i + 1] * p[1]
                                + a_slice[3 * i + 2] * p[2];
                        }

                        // Capture: nearest nucleus
                        let mut v = [0.0; 3];
                        let mut best = f64::INFINITY;
                        let mut nearest = 0i32;
                        for a in 0..n_atoms_usize {
                            offset(&x, a, is_periodic, pos_slice, frac_slice, l_slice, l_inv_slice, &mut v);
                            let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                            if d < best {
                                best = d;
                                nearest = a as i32;
                            }
                        }
                        if best < capture {
                            result = nearest;
                            break;
                        }

                        // Gradient Cartesian
                        let mut gi = [0.0; 3];
                        for d in 0..3 {
                            gi[d] = trilinear(fields[d], &p, &n, is_periodic);
                        }
                        let mut g = [0.0; 3];
                        for j in 0..3 {
                            g[j] = gi[0] * a_inv_slice[j]
                                + gi[1] * a_inv_slice[3 + j]
                                + gi[2] * a_inv_slice[6 + j];
                        }
                        for a in 0..n_atoms_usize {
                            let lo = core_offsets_slice[a] as usize;
                            let hi = core_offsets_slice[a + 1] as usize;
                            if hi == lo {
                                continue;
                            }
                            offset(&x, a, is_periodic, pos_slice, frac_slice, l_slice, l_inv_slice, &mut v);
                            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                            if r < core_support_slice[a] && r > 1e-12 {
                                let slope = interp(r, &core_r_slice[lo..hi], &core_slope_slice[lo..hi], hi - lo) / r;
                                for j in 0..3 {
                                    g[j] += slope * v[j];
                                }
                            }
                        }

                        let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                        let scale = if norm > 0.0 { step_length / norm } else { 0.0 };

                        for d in 0..3 {
                            let di = scale * (a_inv_slice[3 * d] * g[0]
                                + a_inv_slice[3 * d + 1] * g[1]
                                + a_inv_slice[3 * d + 2] * g[2]);
                            p[d] += di;
                            unwrapped[d] += di;
                            if is_periodic {
                                let nd = n[d] as f64;
                                p[d] = p[d].rem_euclid(nd);
                            } else {
                                if p[d] < 0.0 {
                                    p[d] = 0.0;
                                }
                                let max_val = n[d] as f64 - 1.0;
                                if p[d] > max_val {
                                    p[d] = max_val;
                                }
                            }
                        }
                    }

                    unsafe {
                        *owner_raw.add(s_idx) = result;
                        *final_raw.add(3 * s_idx) = p[0];
                        *final_raw.add(3 * s_idx + 1) = p[1];
                        *final_raw.add(3 * s_idx + 2) = p[2];
                    }
                }
            });
        }
    });
}
