//! Low-level grid sampling, radial equations, localized contractions and analysis.
//! Raw-pointer entry points require the contracts documented in SAFETY.md.

pub mod bader;
pub mod bloch;
pub mod localized;
pub mod one_body;
pub mod onecenter;
pub mod radial;
pub mod runtime;
pub mod two_center;

use bloch::Complex64;
use one_body::Complex32;

/// Identifies the Rust implementation without guessing from optional kernels.
#[no_mangle]
pub extern "C" fn mandacaru_backend_abi_version() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn mandacaru_thread_policy_version() -> i32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_num_threads() -> i32 {
    one_body::get_num_threads() as i32
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_one_body(
    psi: *const Complex64,
    vext: *const f64,
    m: i32,
    npts: i32,
    dx: f64,
    out_t: *mut Complex64,
    out_v: *mut Complex64,
) {
    one_body::one_body(psi, vext, m, npts, dx, out_t, out_v);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_one_body_general(
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
    one_body::one_body_general(psi, vext, m, nx, ny, nz, ginv, d_v, out_t, out_v);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_laplacian_real(
    field: *const f64,
    lap: *mut f64,
    n: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    ginv: *const f64,
) {
    one_body::laplacian_real(field, lap, n, nx, ny, nz, ginv);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_laplacian_real_f32(
    field: *const f32,
    lap: *mut f32,
    n: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    ginv: *const f64,
) {
    one_body::laplacian_real_f32(field, lap, n, nx, ny, nz, ginv);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_kb_project(
    psi: *const Complex64,
    chi: *const Complex64,
    m: i32,
    p: i32,
    ngrid: i64,
    d_v: f64,
    out_p: *mut Complex64,
) {
    one_body::kb_project(psi, chi, m, p, ngrid, d_v, out_p);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_kb_project_f32(
    psi: *const Complex32,
    chi: *const Complex32,
    m: i32,
    p: i32,
    ngrid: i64,
    d_v: f64,
    out_p: *mut Complex64,
) {
    one_body::kb_project_f32(psi, chi, m, p, ngrid, d_v, out_p);
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_bloch_shell(
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
    bloch::bloch_shell(
        x,
        y,
        z,
        npts,
        center,
        translations,
        n_images,
        phases,
        nk,
        support,
        breaks,
        coeffs,
        nb,
        rc,
        l,
        derivative,
        ms,
        rows,
        nm,
        m_dim,
        out,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_real_shell(
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
    m_dim: i64,
    out: *mut f64,
) {
    bloch::real_shell(
        x, y, z, npts, center, support, breaks, coeffs, nb, rc, l, derivative, ms, rows, nm, m_dim,
        out,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_real_shell_f32(
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
    m_dim: i64,
    out: *mut f32,
) {
    bloch::real_shell_f32(
        x, y, z, npts, center, support, breaks, coeffs, nb, rc, l, derivative, ms, rows, nm, m_dim,
        out,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_bloch_block() -> i64 {
    bloch::bloch_block()
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_bloch_project(
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
    bloch::bloch_project(
        x,
        y,
        z,
        npts,
        center,
        translations,
        n_images,
        support,
        breaks,
        coeffs,
        nb,
        rc,
        l,
        derivative,
        ms,
        nm,
        cols,
        p,
        out,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_bader_ascent(
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
    bader::bader_ascent(
        starts,
        n_starts,
        gx,
        gy,
        gz,
        n0,
        n1,
        n2,
        periodic,
        a_mat,
        a_inv,
        origin,
        l_mat,
        l_inv,
        positions,
        frac_atoms,
        n_atoms,
        core_offsets,
        core_r,
        core_slope,
        core_support,
        step_length,
        capture,
        max_steps,
        stall_window,
        stall_distance,
        owner,
        final_index,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_localized_density_real(
    nx: i32,
    ny: i32,
    nz: i32,
    n_atoms: i32,
    boxes_ptr: *const i32,
    n_orbs_ptr: *const i32,
    orb_offsets_ptr: *const i32,
    atom_psi_ptrs: *const *const f64,
    p_ao: *const f64,
    m_total: i32,
    out_rho: *mut f64,
) {
    localized::localized_density_real(
        nx,
        ny,
        nz,
        n_atoms,
        boxes_ptr,
        n_orbs_ptr,
        orb_offsets_ptr,
        atom_psi_ptrs,
        p_ao,
        m_total,
        out_rho,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_localized_potential_matrix_real(
    nx: i32,
    ny: i32,
    nz: i32,
    d_v: f64,
    n_atoms: i32,
    boxes_ptr: *const i32,
    n_orbs_ptr: *const i32,
    orb_offsets_ptr: *const i32,
    atom_psi_ptrs: *const *const f64,
    v_grid: *const f64,
    m_total: i32,
    out_v_ao: *mut f64,
) {
    localized::localized_potential_matrix_real(
        nx,
        ny,
        nz,
        d_v,
        n_atoms,
        boxes_ptr,
        n_orbs_ptr,
        orb_offsets_ptr,
        atom_psi_ptrs,
        v_grid,
        m_total,
        out_v_ao,
    );
}

#[no_mangle]
pub unsafe extern "C" fn mandacaru_localized_cross_real(
    moving_atom: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    d_v: f64,
    n_atoms: i32,
    boxes_ptr: *const i32,
    n_orbs_ptr: *const i32,
    orb_offsets_ptr: *const i32,
    dpsi_moving_ptr: *const f64,
    atom_psi_ptrs: *const *const f64,
    v_grid: *const f64,
    m_total: i32,
    out_cross: *mut f64,
) {
    localized::localized_cross_real(
        moving_atom,
        nx,
        ny,
        nz,
        d_v,
        n_atoms,
        boxes_ptr,
        n_orbs_ptr,
        orb_offsets_ptr,
        dpsi_moving_ptr,
        atom_psi_ptrs,
        v_grid,
        m_total,
        out_cross,
    );
}
