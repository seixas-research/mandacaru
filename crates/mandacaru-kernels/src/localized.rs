// file: src/localized.rs
//
// High-performance localized subgrid quadrature kernels in Rust (C ABI).
// Operates strictly on atom-centered bounding box intersections to eliminate
// global (M, N_grid) array instantiations and minimize memory footprint.

use crate::one_body::get_num_threads;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SubBox {
    pub x_start: i32,
    pub x_stop: i32,
    pub y_start: i32,
    pub y_stop: i32,
    pub z_start: i32,
    pub z_stop: i32,
}

impl SubBox {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.x_start >= self.x_stop || self.y_start >= self.y_stop || self.z_start >= self.z_stop
    }

    #[inline(always)]
    pub fn intersect(&self, other: &SubBox) -> SubBox {
        SubBox {
            x_start: self.x_start.max(other.x_start),
            x_stop: self.x_stop.min(other.x_stop),
            y_start: self.y_start.max(other.y_start),
            y_stop: self.y_stop.min(other.y_stop),
            z_start: self.z_start.max(other.z_start),
            z_stop: self.z_stop.min(other.z_stop),
        }
    }

    #[inline(always)]
    pub fn len_x(&self) -> usize {
        (self.x_stop - self.x_start).max(0) as usize
    }

    #[inline(always)]
    pub fn len_y(&self) -> usize {
        (self.y_stop - self.y_start).max(0) as usize
    }

    #[inline(always)]
    pub fn len_z(&self) -> usize {
        (self.z_stop - self.z_start).max(0) as usize
    }

    #[inline(always)]
    pub fn volume(&self) -> usize {
        self.len_x() * self.len_y() * self.len_z()
    }
}

// =========================================================================
// Real Localized Density
// =========================================================================

pub unsafe fn localized_density_real(
    nx: i32,
    ny: i32,
    nz: i32,
    n_atoms: i32,
    boxes_ptr: *const i32,            // [n_atoms * 6]
    n_orbs_ptr: *const i32,           // [n_atoms]
    orb_offsets_ptr: *const i32,      // [n_atoms]
    atom_psi_ptrs: *const *const f64, // [n_atoms]
    p_ao: *const f64,                 // [m_total * m_total]
    m_total: i32,
    out_rho: *mut f64, // [nx * ny * nz]
) {
    let nx_u = nx as usize;
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let n_at = n_atoms as usize;
    let m_tot = m_total as usize;

    let boxes_slice = std::slice::from_raw_parts(boxes_ptr, n_at * 6);
    let n_orbs = std::slice::from_raw_parts(n_orbs_ptr, n_at);
    let orb_offsets = std::slice::from_raw_parts(orb_offsets_ptr, n_at);
    let raw_psi_ptrs = std::slice::from_raw_parts(atom_psi_ptrs, n_at);
    let psi_ptrs_usize: Vec<usize> = raw_psi_ptrs.iter().map(|&p| p as usize).collect();

    let mut sub_boxes = Vec::with_capacity(n_at);
    for a in 0..n_at {
        sub_boxes.push(SubBox {
            x_start: boxes_slice[a * 6],
            x_stop: boxes_slice[a * 6 + 1],
            y_start: boxes_slice[a * 6 + 2],
            y_stop: boxes_slice[a * 6 + 3],
            z_start: boxes_slice[a * 6 + 4],
            z_stop: boxes_slice[a * 6 + 5],
        });
    }

    let p_ao_usize = p_ao as usize;
    let out_rho_usize = out_rho as usize;

    // Parallelize over X-slices to avoid write conflicts
    let n_threads = get_num_threads();
    let x_per_thread = (nx_u + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let x_t_start = t * x_per_thread;
            let x_t_end = (x_t_start + x_per_thread).min(nx_u);
            if x_t_start >= x_t_end {
                continue;
            }

            let sub_boxes_ref = &sub_boxes;
            let psi_ref = &psi_ptrs_usize;

            crate::runtime::spawn(s, move || {
                let p_ao_base = p_ao_usize as *const f64;
                let out_rho_base = out_rho_usize as *mut f64;

                for a in 0..n_at {
                    let box_a = sub_boxes_ref[a];
                    let m_a = n_orbs[a] as usize;
                    if m_a == 0 || box_a.is_empty() {
                        continue;
                    }
                    let off_a = orb_offsets[a] as usize;
                    let psi_a_ptr = psi_ref[a] as *const f64;
                    let sx_len_a = box_a.len_x();
                    let sy_len_a = box_a.len_y();
                    let sz_len_a = box_a.len_z();
                    let vol_a = sx_len_a * sy_len_a * sz_len_a;

                    for b in 0..n_at {
                        let box_b = sub_boxes_ref[b];
                        let m_b = n_orbs[b] as usize;
                        if m_b == 0 || box_b.is_empty() {
                            continue;
                        }

                        let inter = box_a.intersect(&box_b);
                        if inter.is_empty() {
                            continue;
                        }

                        // Clip intersection to this thread's x-range
                        let x0 = (inter.x_start as usize).max(x_t_start);
                        let x1 = (inter.x_stop as usize).min(x_t_end);
                        if x0 >= x1 {
                            continue;
                        }

                        let off_b = orb_offsets[b] as usize;
                        let psi_b_ptr = psi_ref[b] as *const f64;
                        let sx_len_b = box_b.len_x();
                        let sy_len_b = box_b.len_y();
                        let sz_len_b = box_b.len_z();
                        let vol_b = sx_len_b * sy_len_b * sz_len_b;

                        // Check max element in P_ab
                        let mut p_max = 0.0f64;
                        for mu in 0..m_a {
                            for nu in 0..m_b {
                                let val =
                                    (*p_ao_base.add((off_a + mu) * m_tot + (off_b + nu))).abs();
                                if val > p_max {
                                    p_max = val;
                                }
                            }
                        }
                        if p_max < 1e-15 {
                            continue;
                        }

                        let y0 = inter.y_start as usize;
                        let y1 = inter.y_stop as usize;
                        let z0 = inter.z_start as usize;
                        let z1 = inter.z_stop as usize;

                        for x in x0..x1 {
                            let dx_a = x - box_a.x_start as usize;
                            let dx_b = x - box_b.x_start as usize;

                            for y in y0..y1 {
                                let dy_a = y - box_a.y_start as usize;
                                let dy_b = y - box_b.y_start as usize;

                                let sub_base_a = (dx_a * sy_len_a + dy_a) * sz_len_a;
                                let sub_base_b = (dx_b * sy_len_b + dy_b) * sz_len_b;
                                let grid_base = (x * ny_u + y) * nz_u;

                                for z in z0..z1 {
                                    let dz_a = z - box_a.z_start as usize;
                                    let dz_b = z - box_b.z_start as usize;
                                    let sub_idx_a = sub_base_a + dz_a;
                                    let sub_idx_b = sub_base_b + dz_b;
                                    let grid_idx = grid_base + z;

                                    let mut term = 0.0f64;
                                    for mu in 0..m_a {
                                        let psi_val_a = *psi_a_ptr.add(mu * vol_a + sub_idx_a);
                                        let mut t_mu = 0.0f64;
                                        let p_row = p_ao_base.add((off_a + mu) * m_tot + off_b);
                                        for nu in 0..m_b {
                                            let psi_val_b = *psi_b_ptr.add(nu * vol_b + sub_idx_b);
                                            t_mu += *p_row.add(nu) * psi_val_b;
                                        }
                                        term += psi_val_a * t_mu;
                                    }

                                    *out_rho_base.add(grid_idx) += term;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

// =========================================================================
// Real Localized Potential Matrix
// =========================================================================

pub unsafe fn localized_potential_matrix_real(
    _nx: i32,
    ny: i32,
    nz: i32,
    d_v: f64,
    n_atoms: i32,
    boxes_ptr: *const i32,
    n_orbs_ptr: *const i32,
    orb_offsets_ptr: *const i32,
    atom_psi_ptrs: *const *const f64,
    v_grid: *const f64, // [nx * ny * nz]
    m_total: i32,
    out_v_ao: *mut f64, // [m_total * m_total]
) {
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let n_at = n_atoms as usize;
    let m_tot = m_total as usize;

    let boxes_slice = std::slice::from_raw_parts(boxes_ptr, n_at * 6);
    let n_orbs = std::slice::from_raw_parts(n_orbs_ptr, n_at);
    let orb_offsets = std::slice::from_raw_parts(orb_offsets_ptr, n_at);
    let raw_psi_ptrs = std::slice::from_raw_parts(atom_psi_ptrs, n_at);
    let psi_ptrs_usize: Vec<usize> = raw_psi_ptrs.iter().map(|&p| p as usize).collect();

    let mut sub_boxes = Vec::with_capacity(n_at);
    for a in 0..n_at {
        sub_boxes.push(SubBox {
            x_start: boxes_slice[a * 6],
            x_stop: boxes_slice[a * 6 + 1],
            y_start: boxes_slice[a * 6 + 2],
            y_stop: boxes_slice[a * 6 + 3],
            z_start: boxes_slice[a * 6 + 4],
            z_stop: boxes_slice[a * 6 + 5],
        });
    }

    // Build list of valid pairs (a, b) with a <= b
    let mut pairs = Vec::new();
    for a in 0..n_at {
        let box_a = sub_boxes[a];
        if n_orbs[a] == 0 || box_a.is_empty() {
            continue;
        }
        for b in a..n_at {
            let box_b = sub_boxes[b];
            if n_orbs[b] == 0 || box_b.is_empty() {
                continue;
            }
            let inter = box_a.intersect(&box_b);
            if !inter.is_empty() {
                pairs.push((a, b, inter));
            }
        }
    }

    let n_pairs = pairs.len();
    if n_pairs == 0 {
        return;
    }

    let v_grid_usize = v_grid as usize;
    let out_v_ao_usize = out_v_ao as usize;

    let n_threads = get_num_threads();
    let pairs_per_thread = (n_pairs + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let p_start = t * pairs_per_thread;
            let p_end = (p_start + pairs_per_thread).min(n_pairs);
            if p_start >= p_end {
                continue;
            }

            let pairs_slice = &pairs[p_start..p_end];
            let sub_boxes_ref = &sub_boxes;
            let psi_ref = &psi_ptrs_usize;

            crate::runtime::spawn(s, move || {
                let v_grid_base = v_grid_usize as *const f64;
                let out_v_ao_base = out_v_ao_usize as *mut f64;

                for &(a, b, inter) in pairs_slice {
                    let box_a = sub_boxes_ref[a];
                    let box_b = sub_boxes_ref[b];
                    let m_a = n_orbs[a] as usize;
                    let m_b = n_orbs[b] as usize;
                    let off_a = orb_offsets[a] as usize;
                    let off_b = orb_offsets[b] as usize;

                    let psi_a_ptr = psi_ref[a] as *const f64;
                    let psi_b_ptr = psi_ref[b] as *const f64;

                    let sx_len_a = box_a.len_x();
                    let sy_len_a = box_a.len_y();
                    let sz_len_a = box_a.len_z();
                    let vol_a = sx_len_a * sy_len_a * sz_len_a;

                    let sx_len_b = box_b.len_x();
                    let sy_len_b = box_b.len_y();
                    let sz_len_b = box_b.len_z();
                    let vol_b = sx_len_b * sy_len_b * sz_len_b;

                    let x0 = inter.x_start as usize;
                    let x1 = inter.x_stop as usize;
                    let y0 = inter.y_start as usize;
                    let y1 = inter.y_stop as usize;
                    let z0 = inter.z_start as usize;
                    let z1 = inter.z_stop as usize;

                    // Compute block (m_a x m_b)
                    let mut block = vec![0.0f64; m_a * m_b];

                    for x in x0..x1 {
                        let dx_a = x - box_a.x_start as usize;
                        let dx_b = x - box_b.x_start as usize;

                        for y in y0..y1 {
                            let dy_a = y - box_a.y_start as usize;
                            let dy_b = y - box_b.y_start as usize;

                            let sub_base_a = (dx_a * sy_len_a + dy_a) * sz_len_a;
                            let sub_base_b = (dx_b * sy_len_b + dy_b) * sz_len_b;
                            let grid_base = (x * ny_u + y) * nz_u;

                            for z in z0..z1 {
                                let dz_a = z - box_a.z_start as usize;
                                let dz_b = z - box_b.z_start as usize;
                                let sub_idx_a = sub_base_a + dz_a;
                                let sub_idx_b = sub_base_b + dz_b;
                                let grid_idx = grid_base + z;

                                let v_w = *v_grid_base.add(grid_idx) * d_v;

                                for mu in 0..m_a {
                                    let psi_a_val = *psi_a_ptr.add(mu * vol_a + sub_idx_a);
                                    let p_a_w = psi_a_val * v_w;
                                    let b_row = mu * m_b;
                                    for nu in 0..m_b {
                                        let psi_b_val = *psi_b_ptr.add(nu * vol_b + sub_idx_b);
                                        block[b_row + nu] += p_a_w * psi_b_val;
                                    }
                                }
                            }
                        }
                    }

                    // Accumulate block into out_v_ao (disjoint between threads)
                    for mu in 0..m_a {
                        for nu in 0..m_b {
                            let val = block[mu * m_b + nu];
                            *out_v_ao_base.add((off_a + mu) * m_tot + (off_b + nu)) += val;
                            if a != b {
                                *out_v_ao_base.add((off_b + nu) * m_tot + (off_a + mu)) += val;
                            }
                        }
                    }
                }
            });
        }
    });
}

// =========================================================================
// Real Localized One-Body Cross (Pulay Force Derivations)
// =========================================================================

pub unsafe fn localized_cross_real(
    moving_atom: i32,
    _nx: i32,
    ny: i32,
    nz: i32,
    d_v: f64,
    n_atoms: i32,
    boxes_ptr: *const i32,
    n_orbs_ptr: *const i32,
    orb_offsets_ptr: *const i32,
    dpsi_moving_ptr: *const f64, // [m_moving * vol_moving]
    atom_psi_ptrs: *const *const f64,
    v_grid: *const f64,
    m_total: i32,
    out_cross: *mut f64, // [m_total * m_total]
) {
    let a = moving_atom as usize;
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let n_at = n_atoms as usize;
    let m_tot = m_total as usize;

    let boxes_slice = std::slice::from_raw_parts(boxes_ptr, n_at * 6);
    let n_orbs = std::slice::from_raw_parts(n_orbs_ptr, n_at);
    let orb_offsets = std::slice::from_raw_parts(orb_offsets_ptr, n_at);
    let raw_psi_ptrs = std::slice::from_raw_parts(atom_psi_ptrs, n_at);
    let psi_ptrs_usize: Vec<usize> = raw_psi_ptrs.iter().map(|&p| p as usize).collect();

    let box_a = SubBox {
        x_start: boxes_slice[a * 6],
        x_stop: boxes_slice[a * 6 + 1],
        y_start: boxes_slice[a * 6 + 2],
        y_stop: boxes_slice[a * 6 + 3],
        z_start: boxes_slice[a * 6 + 4],
        z_stop: boxes_slice[a * 6 + 5],
    };

    let m_a = n_orbs[a] as usize;
    if m_a == 0 || box_a.is_empty() {
        return;
    }
    let off_a = orb_offsets[a] as usize;
    let sx_len_a = box_a.len_x();
    let sy_len_a = box_a.len_y();
    let sz_len_a = box_a.len_z();
    let vol_a = sx_len_a * sy_len_a * sz_len_a;

    let dpsi_moving_usize = dpsi_moving_ptr as usize;
    let v_grid_usize = v_grid as usize;
    let out_cross_usize = out_cross as usize;

    // Parallelize over ket atoms b
    let n_threads = get_num_threads();
    let b_per_thread = (n_at + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let b_start = t * b_per_thread;
            let b_end = (b_start + b_per_thread).min(n_at);
            if b_start >= b_end {
                continue;
            }

            let psi_ref = &psi_ptrs_usize;

            crate::runtime::spawn(s, move || {
                let dpsi_moving_base = dpsi_moving_usize as *const f64;
                let v_grid_base = v_grid_usize as *const f64;
                let out_cross_base = out_cross_usize as *mut f64;

                for b in b_start..b_end {
                    let box_b = SubBox {
                        x_start: boxes_slice[b * 6],
                        x_stop: boxes_slice[b * 6 + 1],
                        y_start: boxes_slice[b * 6 + 2],
                        y_stop: boxes_slice[b * 6 + 3],
                        z_start: boxes_slice[b * 6 + 4],
                        z_stop: boxes_slice[b * 6 + 5],
                    };
                    let m_b = n_orbs[b] as usize;
                    if m_b == 0 || box_b.is_empty() {
                        continue;
                    }

                    let inter = box_a.intersect(&box_b);
                    if inter.is_empty() {
                        continue;
                    }

                    let off_b = orb_offsets[b] as usize;
                    let psi_b_ptr = psi_ref[b] as *const f64;
                    let sx_len_b = box_b.len_x();
                    let sy_len_b = box_b.len_y();
                    let sz_len_b = box_b.len_z();
                    let vol_b = sx_len_b * sy_len_b * sz_len_b;

                    let x0 = inter.x_start as usize;
                    let x1 = inter.x_stop as usize;
                    let y0 = inter.y_start as usize;
                    let y1 = inter.y_stop as usize;
                    let z0 = inter.z_start as usize;
                    let z1 = inter.z_stop as usize;

                    let mut block = vec![0.0f64; m_a * m_b];

                    for x in x0..x1 {
                        let dx_a = x - box_a.x_start as usize;
                        let dx_b = x - box_b.x_start as usize;

                        for y in y0..y1 {
                            let dy_a = y - box_a.y_start as usize;
                            let dy_b = y - box_b.y_start as usize;

                            let sub_base_a = (dx_a * sy_len_a + dy_a) * sz_len_a;
                            let sub_base_b = (dx_b * sy_len_b + dy_b) * sz_len_b;
                            let grid_base = (x * ny_u + y) * nz_u;

                            for z in z0..z1 {
                                let dz_a = z - box_a.z_start as usize;
                                let dz_b = z - box_b.z_start as usize;
                                let sub_idx_a = sub_base_a + dz_a;
                                let sub_idx_b = sub_base_b + dz_b;
                                let grid_idx = grid_base + z;

                                let v_w = *v_grid_base.add(grid_idx) * d_v;

                                for mu in 0..m_a {
                                    let dpsi_val = *dpsi_moving_base.add(mu * vol_a + sub_idx_a);
                                    let p_w = dpsi_val * v_w;
                                    let b_row = mu * m_b;
                                    for nu in 0..m_b {
                                        let psi_b_val = *psi_b_ptr.add(nu * vol_b + sub_idx_b);
                                        block[b_row + nu] += p_w * psi_b_val;
                                    }
                                }
                            }
                        }
                    }

                    // Write block into out_cross (disjoint column blocks for different b)
                    for mu in 0..m_a {
                        for nu in 0..m_b {
                            *out_cross_base.add((off_a + mu) * m_tot + (off_b + nu)) =
                                block[mu * m_b + nu];
                        }
                    }
                }
            });
        }
    });
}
