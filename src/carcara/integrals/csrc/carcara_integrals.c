/* file: carcara_integrals.c
 *
 * This code is part of Carcara.
 * MIT License
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * OpenMP-parallel real-space one- and two-body integral kernels.
 *
 * Why OpenMP (not MPI) here:
 *   The integral matrices are embarrassingly parallel over (matrix-element or
 *   grid) indices, and every thread reads the *same* sampled grids -- a
 *   shared-memory, read-only workload with no communication.  `#pragma omp
 *   parallel for schedule(dynamic)` over the outer index load-balances the work
 *   with essentially zero overhead.  MPI would add message passing and array
 *   partitioning for no gain on a single node; it only pays off once the grid
 *   no longer fits in one node's memory (fine 6-D two-body meshes).  The loops
 *   below are written so that step is a later domain-decomposition + Allreduce,
 *   without touching the numerical core.
 */
#include "carcara_integrals.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Number of OpenMP threads the kernels run with (1 without OpenMP). */
int carcara_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Flat index into a cubic (npts^3) grid, row-major (i,j,k). */
static inline long idx3(int i, int j, int k, int npts) {
    return ((long)i * npts + j) * npts + k;
}

/* 7-point finite-difference Laplacian of one sampled field into `lap`.
 * Localized functions are assumed to have decayed to ~0 at the box faces, so
 * out-of-range neighbors contribute nothing. */
static void laplacian_fd(const double _Complex *field, double _Complex *lap,
                         int npts, double dx) {
    const double inv_dx2 = 1.0 / (dx * dx);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < npts; ++i) {
        for (int j = 0; j < npts; ++j) {
            for (int k = 0; k < npts; ++k) {
                const long g = idx3(i, j, k, npts);
                double _Complex acc = -6.0 * field[g];
                if (i + 1 < npts) acc += field[idx3(i + 1, j, k, npts)];
                if (i - 1 >= 0)   acc += field[idx3(i - 1, j, k, npts)];
                if (j + 1 < npts) acc += field[idx3(i, j + 1, k, npts)];
                if (j - 1 >= 0)   acc += field[idx3(i, j - 1, k, npts)];
                if (k + 1 < npts) acc += field[idx3(i, j, k + 1, npts)];
                if (k - 1 >= 0)   acc += field[idx3(i, j, k - 1, npts)];
                lap[g] = acc * inv_dx2;
            }
        }
    }
}

/* Flat index into a general (nx*ny*nz) grid, row-major (i,j,k). */
static inline long idx3g(int i, int j, int k, int ny, int nz) {
    return ((long)i * ny + j) * nz + k;
}

/* General finite-difference Laplacian on an anisotropic and/or non-orthogonal
 * grid.  With the grid step vectors s_m (columns of `step`) and the metric
 * G = step^T step, the Laplacian in integer index coordinates is
 *
 *     nabla^2 f = sum_{a,b} (G^{-1})_{ab} d_a d_b f,
 *
 * where d_a is a unit-spacing finite difference along index axis a.  Diagonal
 * terms use the 3-point second difference; off-diagonal (cross) terms use the
 * 4-point mixed difference [f(+a+b) - f(+a-b) - f(-a+b) + f(-a-b)] / 4, weighted
 * by 2 (G^{-1})_{ab}.  `ginv` is the row-major 3x3 inverse metric; it already
 * carries the 1/length^2 units, so no separate dx^2 division is needed.  For a
 * cubic grid ginv = diag(1/dx^2) and this reduces to the 7-point kernel above.
 * Out-of-range neighbors are treated as zero (localized functions decay to ~0
 * at the box faces). */
static void laplacian_general(const double _Complex *field, double _Complex *lap,
                              int nx, int ny, int nz, const double *ginv) {
    const double g00 = ginv[0], g11 = ginv[4], g22 = ginv[8];
    const double g01 = ginv[1], g02 = ginv[2], g12 = ginv[5];
    const double diag = -2.0 * (g00 + g11 + g22);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const long g = idx3g(i, j, k, ny, nz);
                double _Complex acc = diag * field[g];
                /* Diagonal second derivatives. */
                if (i + 1 < nx) acc += g00 * field[idx3g(i + 1, j, k, ny, nz)];
                if (i - 1 >= 0) acc += g00 * field[idx3g(i - 1, j, k, ny, nz)];
                if (j + 1 < ny) acc += g11 * field[idx3g(i, j + 1, k, ny, nz)];
                if (j - 1 >= 0) acc += g11 * field[idx3g(i, j - 1, k, ny, nz)];
                if (k + 1 < nz) acc += g22 * field[idx3g(i, j, k + 1, ny, nz)];
                if (k - 1 >= 0) acc += g22 * field[idx3g(i, j, k - 1, ny, nz)];
                /* Cross derivatives: coefficient 2*g_ab * 1/4 = g_ab/2. */
                if (g01 != 0.0) {
                    const double c = 0.5 * g01;
                    if (i + 1 < nx && j + 1 < ny) acc += c * field[idx3g(i+1, j+1, k, ny, nz)];
                    if (i + 1 < nx && j - 1 >= 0) acc -= c * field[idx3g(i+1, j-1, k, ny, nz)];
                    if (i - 1 >= 0 && j + 1 < ny) acc -= c * field[idx3g(i-1, j+1, k, ny, nz)];
                    if (i - 1 >= 0 && j - 1 >= 0) acc += c * field[idx3g(i-1, j-1, k, ny, nz)];
                }
                if (g02 != 0.0) {
                    const double c = 0.5 * g02;
                    if (i + 1 < nx && k + 1 < nz) acc += c * field[idx3g(i+1, j, k+1, ny, nz)];
                    if (i + 1 < nx && k - 1 >= 0) acc -= c * field[idx3g(i+1, j, k-1, ny, nz)];
                    if (i - 1 >= 0 && k + 1 < nz) acc -= c * field[idx3g(i-1, j, k+1, ny, nz)];
                    if (i - 1 >= 0 && k - 1 >= 0) acc += c * field[idx3g(i-1, j, k-1, ny, nz)];
                }
                if (g12 != 0.0) {
                    const double c = 0.5 * g12;
                    if (j + 1 < ny && k + 1 < nz) acc += c * field[idx3g(i, j+1, k+1, ny, nz)];
                    if (j + 1 < ny && k - 1 >= 0) acc -= c * field[idx3g(i, j+1, k-1, ny, nz)];
                    if (j - 1 >= 0 && k + 1 < nz) acc -= c * field[idx3g(i, j-1, k+1, ny, nz)];
                    if (j - 1 >= 0 && k - 1 >= 0) acc += c * field[idx3g(i, j-1, k-1, ny, nz)];
                }
                lap[g] = acc;
            }
        }
    }
}

void carcara_one_body_general(const double _Complex *psi,
                              const double *Vext,
                              int M, int nx, int ny, int nz,
                              const double *ginv, double dV,
                              double _Complex *out_T,
                              double _Complex *out_V) {
    const long ngrid = (long)nx * ny * nz;

    double _Complex *lap = malloc((size_t)M * ngrid * sizeof(*lap));
    if (!lap) return;
    for (int b = 0; b < M; ++b)
        laplacian_general(psi + (long)b * ngrid, lap + (long)b * ngrid,
                          nx, ny, nz, ginv);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
    for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
            const double _Complex *pa = psi + (long)a * ngrid;
            const double _Complex *pb = psi + (long)b * ngrid;
            const double _Complex *lb = lap + (long)b * ngrid;
            double _Complex t = 0.0, v = 0.0;
            for (long g = 0; g < ngrid; ++g) {
                const double _Complex ca = conj(pa[g]);
                t += ca * (-0.5 * lb[g]);
                v += ca * Vext[g] * pb[g];
            }
            out_T[(long)a * M + b] = t * dV;
            out_V[(long)a * M + b] = v * dV;
        }
    }
    free(lap);
}

void carcara_one_body(const double _Complex *psi,
                      const double *Vext,
                      int M, int npts, double dx,
                      double _Complex *out_T,
                      double _Complex *out_V) {
    const long ngrid = (long)npts * npts * npts;
    const double dV = dx * dx * dx;

    /* Precompute the Laplacian of every ket once: reused for all bras. */
    double _Complex *lap = malloc((size_t)M * ngrid * sizeof(*lap));
    if (!lap) return;
    for (int b = 0; b < M; ++b)
        laplacian_fd(psi + (long)b * ngrid, lap + (long)b * ngrid, npts, dx);

    /* Matrix elements: parallel over the flattened (a,b) index.  Dynamic
     * scheduling keeps threads busy if some reductions are cheaper. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
    for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
            const double _Complex *pa = psi + (long)a * ngrid;
            const double _Complex *pb = psi + (long)b * ngrid;
            const double _Complex *lb = lap + (long)b * ngrid;
            double _Complex t = 0.0, v = 0.0;
            for (long g = 0; g < ngrid; ++g) {
                const double _Complex ca = conj(pa[g]);
                t += ca * (-0.5 * lb[g]);
                v += ca * Vext[g] * pb[g];
            }
            out_T[(long)a * M + b] = t * dV;
            out_V[(long)a * M + b] = v * dV;
        }
    }
    free(lap);
}

void carcara_kb_project(const double _Complex *psi,
                        const double _Complex *chi,
                        int M, int P, long ngrid, double dV,
                        double _Complex *out_P) {
    /* Parallel over the flattened (basis, projector) pair.  Each entry is an
     * independent reduction over the grid, so there is nothing to synchronize. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int a = 0; a < M; ++a) {
        for (int p = 0; p < P; ++p) {
            const double _Complex *pa = psi + (long)a * ngrid;
            const double _Complex *cp = chi + (long)p * ngrid;
            double _Complex acc = 0.0;
            for (long g = 0; g < ngrid; ++g)
                acc += conj(pa[g]) * cp[g];
            out_P[(long)a * P + p] = acc * dV;
        }
    }
}

void carcara_two_body(const double _Complex *psi,
                      const double *xg, const double *yg, const double *zg,
                      int M, int ngrid, double dV, double softening,
                      double _Complex *out_eri) {
    const double soft2 = softening * softening;
    double _Complex *rho2 = malloc((size_t)ngrid * sizeof(*rho2));
    double _Complex *phi  = malloc((size_t)ngrid * sizeof(*phi));
    if (!rho2 || !phi) { free(rho2); free(phi); return; }

    for (int b = 0; b < M; ++b) {
        for (int d = 0; d < M; ++d) {
            const double _Complex *pb = psi + (long)b * ngrid;
            const double _Complex *pd = psi + (long)d * ngrid;
            for (long g = 0; g < ngrid; ++g)
                rho2[g] = conj(pb[g]) * pd[g];

            /* Coulomb potential of density rho_bd -- the O(ngrid^2) hotspot.
             * Each target point g1 is independent -> parallelize over g1.
             * (Production path: replace this double sum by an FFT/multigrid
             *  Poisson solve, O(ngrid log ngrid).) */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
            for (long g1 = 0; g1 < ngrid; ++g1) {
                const double x1 = xg[g1], y1 = yg[g1], z1 = zg[g1];
                double _Complex acc = 0.0;
                for (long g2 = 0; g2 < ngrid; ++g2) {
                    const double dxr = x1 - xg[g2];
                    const double dyr = y1 - yg[g2];
                    const double dzr = z1 - zg[g2];
                    double r = sqrt(dxr * dxr + dyr * dyr + dzr * dzr + soft2);
                    if (r < 1e-15) r = 1e-15;
                    acc += rho2[g2] / r;
                }
                phi[g1] = acc * dV;
            }

            /* Contract Phi_bd against every density rho_ac. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
            for (int a = 0; a < M; ++a) {
                for (int c = 0; c < M; ++c) {
                    const double _Complex *pa = psi + (long)a * ngrid;
                    const double _Complex *pc = psi + (long)c * ngrid;
                    double _Complex acc = 0.0;
                    for (long g = 0; g < ngrid; ++g)
                        acc += conj(pa[g]) * pc[g] * phi[g];
                    const long e = (((long)a * M + b) * M + c) * M + d;
                    out_eri[e] = acc * dV;
                }
            }
        }
    }
    free(rho2);
    free(phi);
}

void carcara_sample_basis(carcara_basis_fn fn, int M,
                          const double *xg, const double *yg, const double *zg,
                          int ngrid, double _Complex *psi, void *ctx) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < M; ++i)
        fn(i, xg, yg, zg, ngrid, psi + (long)i * ngrid, ctx);
}
