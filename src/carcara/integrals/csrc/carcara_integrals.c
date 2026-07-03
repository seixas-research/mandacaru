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
