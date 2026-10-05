/* file: mandacaru_integrals.h
 *
 * This code is part of Mandacaru.
 * MIT License
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * High-performance, basis-agnostic real-space integral backend.
 *
 * The kernels operate on *sampled function values* on a uniform cubic grid,
 * never on analytic orbital forms.  This is what makes them agnostic to the
 * basis: hydrogen-like orbitals, Wannier functions or any localized function
 * are all just complex arrays here: pre-sampled, psi[i * ngrid + g], passed
 * zero-copy from NumPy complex128 == C99 double _Complex.
 *
 * Parallelism: OpenMP over matrix-element / grid indices (shared read-only
 * grids, no communication).  See the .c file for the schedule rationale.
 */
#ifndef MANDACARU_INTEGRALS_H
#define MANDACARU_INTEGRALS_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of OpenMP threads the kernels run with (1 when built without OpenMP).
 * Lets the Python layer report the core count used by the integral backend. */
int mandacaru_num_threads(void);

/* One-body matrices for M sampled functions on a cubic grid of npts^3 nodes.
 *
 *   T[a*M + b] = <psi_a| -1/2 nabla^2 |psi_b>   (7-point FD Laplacian)
 *   V[a*M + b] = <psi_a|      Vext     |psi_b>
 *
 * psi    : (M * ngrid) complex, row-major (function-major).
 * Vext   : (ngrid)     real, external potential sampled on the grid.
 * dx     : grid spacing (dV = dx^3).
 * out_T, out_V : (M * M) complex, caller-allocated.
 */
void mandacaru_one_body(const double _Complex *psi,
                      const double *Vext,
                      int M, int npts, double dx,
                      double _Complex *out_T,
                      double _Complex *out_V);

/* One-body matrices on a *general* grid: per-axis node counts (nx, ny, nz) and
 * an arbitrary (anisotropic and/or non-orthogonal) geometry.  The grid geometry
 * enters only through
 *
 *   ginv : (9) row-major 3x3 inverse metric (step^T step)^{-1}, which carries
 *          the 1/length^2 units of the Laplacian (diag(1/dx^2, 1/dy^2, 1/dz^2)
 *          for an orthorhombic grid, with non-zero off-diagonals when skewed);
 *   dV   : voxel volume |det(step)|.
 *
 * This is the general counterpart of mandacaru_one_body (which is the fast path
 * for a cubic grid) and supports varying resolution along each axis and
 * non-orthogonal unit cells.  out_T, out_V : (M * M) complex, caller-allocated.
 */
void mandacaru_one_body_general(const double _Complex *psi,
                              const double *Vext,
                              int M, int nx, int ny, int nz,
                              const double *ginv, double dV,
                              double _Complex *out_T,
                              double _Complex *out_V);

/* Kleinman-Bylander projector overlaps.
 *
 * The nonlocal part of a norm-conserving pseudopotential is a sum of rank-one
 * terms,  V_NL = sum_p |chi_p> E_p <chi_p|,  so the only grid work it needs is
 * the overlap of every basis function with every projector:
 *
 *     out_P[a * P + p] = dV * sum_g conj(psi_a[g]) * chi_p[g].
 *
 * The nonlocal matrix is then the small outer product P diag(E) P^dagger,
 * assembled by the caller.  This is O(M * P * ngrid) rather than the
 * O(M^2 * ngrid) a semilocal form would cost -- the whole point of the
 * Kleinman-Bylander transformation.
 *
 * psi   : (M * ngrid) complex basis samples.
 * chi   : (P * ngrid) complex projector samples.
 * out_P : (M * P) complex, caller-allocated.
 */
void mandacaru_kb_project(const double _Complex *psi,
                        const double _Complex *chi,
                        int M, int P, long ngrid, double dV,
                        double _Complex *out_P);

/* Bloch sums of one shell of R(r) Y_lm functions (mandacaru_bloch.c):
 * out[k, rows[i], j] += sum_R phases[k, R] R(|r_j - c - R|) Y_l^{ms[i]}, or
 * of its Cartesian derivative along `derivative` (0, 1, 2; -1 for none). */
void mandacaru_bloch_shell(const double *x, const double *y, const double *z,
                           long npts, const double *center,
                           const double *translations, int n_images,
                           const double _Complex *phases, int nk,
                           double support,
                           const double *breaks, const double *coeffs, int nb,
                           double rc, int l, int derivative,
                           const int *ms, const long *rows, int nm, long M,
                           double _Complex *out);

/* Steepest-ascent trajectories for the Bader partition (mandacaru_bader.c):
 * owner[s] is the captured atom, or -1 (stalled / out of steps) with the
 * stopping point in final_index. */
void mandacaru_bader_ascent(const double *starts, long n_starts,
                            const double *gx, const double *gy,
                            const double *gz, int n0, int n1, int n2,
                            int periodic, const double *A,
                            const double *A_inv, const double *origin,
                            const double *L, const double *L_inv,
                            const double *positions, const double *frac_atoms,
                            int n_atoms, const long *core_offsets,
                            const double *core_r, const double *core_slope,
                            const double *core_support, double step_length,
                            double capture, int max_steps, int stall_window,
                            double stall_distance, int *owner,
                            double *final_index);

#ifdef __cplusplus
}
#endif

#endif /* MANDACARU_INTEGRALS_H */
