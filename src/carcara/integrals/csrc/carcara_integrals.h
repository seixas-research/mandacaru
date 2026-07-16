/* file: carcara_integrals.h
 *
 * This code is part of Carcara.
 * MIT License
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * High-performance, basis-agnostic real-space integral backend.
 *
 * The kernels operate on *sampled function values* on a uniform cubic grid,
 * never on analytic orbital forms.  This is what makes them agnostic to the
 * basis: hydrogen-like orbitals, Wannier functions or any localized function
 * are all just complex arrays here.  Two consumption paths are supported:
 *
 *   1. Pre-sampled arrays  (psi[i * ngrid + g])  -- the default, zero-copy from
 *      NumPy complex128 == C99 double _Complex.
 *   2. On-the-fly evaluation through a function pointer (carcara_basis_fn),
 *      for grids too large to store M full fields in memory.
 *
 * Parallelism: OpenMP over matrix-element / grid indices (shared read-only
 * grids, no communication).  See the .c file for the schedule rationale.
 */
#ifndef CARCARA_INTEGRALS_H
#define CARCARA_INTEGRALS_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Signature for on-the-fly basis evaluation: fills `out` (length ngrid) with
 * the value of basis function `i` at the supplied grid coordinates.  `ctx` is
 * an opaque user pointer (e.g. a struct of quantum numbers / Wannier tables). */
typedef void (*carcara_basis_fn)(int i,
                                 const double *x, const double *y,
                                 const double *z, int ngrid,
                                 double _Complex *out, void *ctx);

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
void carcara_one_body(const double _Complex *psi,
                      const double *Vext,
                      int M, int npts, double dx,
                      double _Complex *out_T,
                      double _Complex *out_V);

/* Two-body electron-repulsion tensor (physicists' notation <ab|cd>):
 *
 *   eri[((a*M + b)*M + c)*M + d] =
 *       \int\int conj(psi_a(1)) psi_c(1) (1/r12) conj(psi_b(2)) psi_d(2) dV1 dV2
 *
 * i.e. electron 1 carries the index pair (a, c) and electron 2 the pair (b, d).
 *
 * Computed as: for each density pair rho_bd(2) build its Coulomb potential
 * Phi_bd(1) on the grid (the O(ngrid^2) hotspot, OpenMP-parallel), then
 * contract against every rho_ac(1).  `softening` regularizes r12 -> 0.
 *
 * xg, yg, zg : (ngrid) node coordinates.  dV = dx^3.
 * out_eri    : (M^4) complex, caller-allocated.
 */
void carcara_two_body(const double _Complex *psi,
                      const double *xg, const double *yg, const double *zg,
                      int M, int ngrid, double dV, double softening,
                      double _Complex *out_eri);

/* Optional helper: sample all M functions via a callback into `psi`
 * (M * ngrid).  Lets callers stream a basis (e.g. Wannier) into the same
 * kernels without materializing it in Python. */
void carcara_sample_basis(carcara_basis_fn fn, int M,
                          const double *xg, const double *yg, const double *zg,
                          int ngrid, double _Complex *psi, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CARCARA_INTEGRALS_H */
