/* file: mandacaru_bader.c
 *
 * This code is part of Mandacaru.
 * MIT License
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * Steepest-ascent trajectories of a density for the Bader partition
 * (algorithms/charges.py: bader_populations).
 *
 * Each trajectory climbs the density in fixed steps until it comes within
 * the capture radius of a nucleus.  The direction is the gradient: the
 * valence part interpolated trilinearly from its values at the grid nodes
 * (index components, periodic or clamped at a box's faces), plus each
 * frozen core's analytic slope from the atom's nearest image.  Every
 * `stall_window` steps a trajectory that has moved less than
 * `stall_distance` -- oscillating about a saddle -- is given up, as one is
 * that runs out of steps; the caller assigns those (owner -1, with the
 * point where they stopped).
 *
 * Trajectories are independent: one per loop iteration, in parallel.
 */
#include "mandacaru_integrals.h"

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* np.interp: linear, clamped at both ends. */
static double interp(double r, const double *x, const double *y, long n) {
    if (r <= x[0])
        return y[0];
    if (r >= x[n - 1])
        return y[n - 1];
    long lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const long mid = (lo + hi) / 2;
        if (x[mid] <= r)
            lo = mid;
        else
            hi = mid;
    }
    const double t = (r - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + t * (y[hi] - y[lo]);
}

/* Offset from atom `a`'s nearest image to the Cartesian point x. */
static void offset(const double *x, int a, int periodic,
                   const double *positions, const double *frac_atoms,
                   const double *L, const double *L_inv, double *v) {
    if (!periodic) {
        for (int i = 0; i < 3; ++i)
            v[i] = x[i] - positions[3 * a + i];
        return;
    }
    double delta[3];
    for (int i = 0; i < 3; ++i) {
        double f = 0.0;
        for (int j = 0; j < 3; ++j)
            f += L_inv[3 * i + j] * x[j];
        delta[i] = f - frac_atoms[3 * a + i];
        delta[i] -= nearbyint(delta[i]);       /* half to even, as np.round */
    }
    for (int i = 0; i < 3; ++i)
        v[i] = L[3 * i] * delta[0] + L[3 * i + 1] * delta[1]
               + L[3 * i + 2] * delta[2];
}

/* Trilinear interpolation of a node field at a (folded or clipped) index. */
static double trilinear(const double *f, const double *p, const int *n,
                        int periodic) {
    long i0[3], i1[3];
    double t[3];
    for (int a = 0; a < 3; ++a) {
        const double fl = floor(p[a]);
        t[a] = p[a] - fl;
        long i = (long)fl;
        if (periodic) {
            i %= n[a];
            if (i < 0)
                i += n[a];
            i0[a] = i;
            i1[a] = (i + 1) % n[a];
        } else {
            if (i < 0) { i = 0; t[a] = 0.0; }
            if (i > n[a] - 1) { i = n[a] - 1; t[a] = 0.0; }
            i0[a] = i;
            i1[a] = i + 1 < n[a] ? i + 1 : n[a] - 1;
        }
    }
    double out = 0.0;
    for (int c = 0; c < 8; ++c) {
        const long ix = (c & 4) ? i1[0] : i0[0];
        const long iy = (c & 2) ? i1[1] : i0[1];
        const long iz = (c & 1) ? i1[2] : i0[2];
        const double w = ((c & 4) ? t[0] : 1.0 - t[0])
                         * ((c & 2) ? t[1] : 1.0 - t[1])
                         * ((c & 1) ? t[2] : 1.0 - t[2]);
        out += w * f[(ix * n[1] + iy) * n[2] + iz];
    }
    return out;
}

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
                            double *final_index) {
    const int n[3] = {n0, n1, n2};
    const double *fields[3] = {gx, gy, gz};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (long s = 0; s < n_starts; ++s) {
        double p[3] = {starts[3 * s], starts[3 * s + 1], starts[3 * s + 2]};
        double unwrapped[3] = {p[0], p[1], p[2]};
        double anchor[3] = {p[0], p[1], p[2]};
        int result = -1;
        for (int count = 1; count <= max_steps; ++count) {
            if (count % stall_window == 0) {
                double moved = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double c = 0.0;
                    for (int d = 0; d < 3; ++d)
                        c += A[3 * i + d] * (unwrapped[d] - anchor[d]);
                    moved += c * c;
                }
                if (sqrt(moved) < stall_distance)
                    break;
                for (int d = 0; d < 3; ++d)
                    anchor[d] = unwrapped[d];
            }
            double x[3];
            for (int i = 0; i < 3; ++i)
                x[i] = origin[i] + A[3 * i] * p[0] + A[3 * i + 1] * p[1]
                       + A[3 * i + 2] * p[2];
            /* Capture: the nearest nucleus (first on a tie, as argmin). */
            double v[3], best = INFINITY;
            int nearest = 0;
            for (int a = 0; a < n_atoms; ++a) {
                offset(x, a, periodic, positions, frac_atoms, L, L_inv, v);
                const double d = sqrt(v[0] * v[0] + v[1] * v[1]
                                      + v[2] * v[2]);
                if (d < best) {
                    best = d;
                    nearest = a;
                }
            }
            if (best < capture) {
                result = nearest;
                break;
            }
            /* The gradient, Cartesian. */
            double gi[3], g[3];
            for (int d = 0; d < 3; ++d)
                gi[d] = trilinear(fields[d], p, n, periodic);
            for (int j = 0; j < 3; ++j)
                g[j] = gi[0] * A_inv[j] + gi[1] * A_inv[3 + j]
                       + gi[2] * A_inv[6 + j];
            for (int a = 0; a < n_atoms; ++a) {
                const long lo = core_offsets[a], hi = core_offsets[a + 1];
                if (hi == lo)
                    continue;
                offset(x, a, periodic, positions, frac_atoms, L, L_inv, v);
                const double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                if (r < core_support[a] && r > 1e-12) {
                    const double slope = interp(r, core_r + lo,
                                                core_slope + lo, hi - lo) / r;
                    for (int j = 0; j < 3; ++j)
                        g[j] += slope * v[j];
                }
            }
            const double norm = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            const double scale = norm > 0.0 ? step_length / norm : 0.0;
            for (int d = 0; d < 3; ++d) {
                const double di = scale * (A_inv[3 * d] * g[0]
                                           + A_inv[3 * d + 1] * g[1]
                                           + A_inv[3 * d + 2] * g[2]);
                p[d] += di;
                unwrapped[d] += di;
                if (periodic) {
                    p[d] = fmod(p[d], (double)n[d]);
                    if (p[d] < 0.0)
                        p[d] += n[d];
                    if (p[d] >= n[d])
                        p[d] = 0.0;
                } else {
                    if (p[d] < 0.0) p[d] = 0.0;
                    if (p[d] > n[d] - 1.0) p[d] = n[d] - 1.0;
                }
            }
        }
        owner[s] = result;
        for (int d = 0; d < 3; ++d)
            final_index[3 * s + d] = p[d];
    }
}
