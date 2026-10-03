/* file: mandacaru_bloch.c
 *
 * This code is part of Mandacaru.
 * MIT License
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * Bloch sums of atom-centered functions R(r) Y_lm on a crystal grid.
 *
 * One call handles one shell: the m components that share a radial table and
 * a center.  For every grid point and every lattice image within the support
 * it evaluates the radial spline once and each Y_lm once, and accumulates
 * the image's k phases into the output -- what the NumPy path does with a
 * Python loop over images, two transcendental calls per point for the
 * angles and a sparse product per function.
 *
 * Threads own grid points, so no two write the same output element.
 *
 * The radial function is the piecewise cubic of a SciPy CubicSpline
 * (`breaks`, `coeffs` row i = c[0..3] for (r - x_i)^3 .. (r - x_i)^0),
 * continued below the first break as R(x_0) (r / x_0)^l and zero beyond the
 * last break or at r >= `rc`.  The spherical harmonics are the orthonormal
 * complex ones with the Condon-Shortley phase, by the same recurrence as
 * `basis/_angular.py:spherical_harmonic`.
 *
 * With `derivative` = 0, 1 or 2 the kernel sums the Cartesian derivative
 * d/dx, d/dy or d/dz of the function instead -- analytically, writing
 *     Y_l^m = Q_lm(z / r) ((x + i y) / r)^m          (m >= 0),
 * Q_lm the associated Legendre function without its sin^m factor: a form
 * smooth in Cartesian coordinates, with no pole on the z axis.
 */
#include "mandacaru_integrals.h"

#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Value of the radial spline at r (see the file comment). */
static double radial_value(double r, const double *breaks,
                           const double *coeffs, int nb, double rc, int l) {
    const double x0 = breaks[0], x_end = breaks[nb - 1];
    if (r > x_end || r >= rc)
        return 0.0;
    const double rr = r < x0 ? x0 : r;
    /* Interval i with breaks[i] <= rr < breaks[i+1] (the last one closed). */
    int lo = 0, hi = nb - 2;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (breaks[mid] <= rr)
            lo = mid;
        else
            hi = mid - 1;
    }
    const double *c = coeffs + 4 * (long)lo;
    const double t = rr - breaks[lo];
    double value = ((c[0] * t + c[1]) * t + c[2]) * t + c[3];
    if (r < x0 && x0 > 0.0 && l > 0)
        value *= pow(r > 0.0 ? r / x0 : 0.0, (double)l);
    return value;
}

/* Fully normalized associated Legendre function P_l^m(cos theta), m >= 0,
 * including the Condon-Shortley phase and the 1/sqrt(4 pi) of Y_lm. */
static double legendre(int l, int m, double c, double s) {
    double p = 1.0 / sqrt(4.0 * M_PI);
    for (int k = 1; k <= m; ++k)
        p *= -sqrt((2.0 * k + 1.0) / (2.0 * k)) * s;
    if (l == m)
        return p;
    double previous = p;
    p = sqrt(2.0 * m + 3.0) * c * p;
    for (int d = m + 2; d <= l; ++d) {
        const double a = sqrt((4.0 * d * d - 1.0) / ((double)d * d - (double)m * m));
        const double b = sqrt(((d - 1.0) * (d - 1.0) - (double)m * m)
                              / (4.0 * (d - 1.0) * (d - 1.0) - 1.0));
        const double next = a * (c * p - b * previous);
        previous = p;
        p = next;
    }
    return p;
}

/* Q_lm(u) and dQ_lm/du (m >= 0): P_l^m(u) = Q_lm(u) (1 - u^2)^(m/2), with
 * the normalization and phase of `legendre`. */
static void legendre_q(int l, int m, double u, double *Q, double *dQ) {
    double q0 = 1.0 / sqrt(4.0 * M_PI);
    for (int k = 1; k <= m; ++k)
        q0 *= -sqrt((2.0 * k + 1.0) / (2.0 * k));
    double d0 = 0.0;
    if (l == m) {
        *Q = q0;
        *dQ = d0;
        return;
    }
    double q1 = sqrt(2.0 * m + 3.0) * u * q0;
    double d1 = sqrt(2.0 * m + 3.0) * q0;
    for (int d = m + 2; d <= l; ++d) {
        const double a = sqrt((4.0 * d * d - 1.0) / ((double)d * d - (double)m * m));
        const double b = sqrt(((d - 1.0) * (d - 1.0) - (double)m * m)
                              / (4.0 * (d - 1.0) * (d - 1.0) - 1.0));
        const double q2 = a * (u * q1 - b * q0);
        const double d2 = a * (q1 + u * d1 - b * d0);
        q0 = q1; d0 = d1;
        q1 = q2; d1 = d2;
    }
    *Q = q1;
    *dQ = d1;
}

/* dR/dr of the radial spline (see `radial_value`). */
static double radial_slope(double r, const double *breaks,
                           const double *coeffs, int nb, double rc, int l) {
    const double x0 = breaks[0], x_end = breaks[nb - 1];
    if (r > x_end || r >= rc)
        return 0.0;
    if (r < x0 && x0 > 0.0) {
        if (l == 0)
            return 0.0;
        const double edge = radial_value(x0, breaks, coeffs, nb, rc, l);
        return l * edge * pow(r / x0, (double)(l - 1)) / x0;
    }
    int lo = 0, hi = nb - 2;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (breaks[mid] <= r)
            lo = mid;
        else
            hi = mid - 1;
    }
    const double *c = coeffs + 4 * (long)lo;
    const double t = r - breaks[lo];
    return (3.0 * c[0] * t + 2.0 * c[1]) * t + c[2];
}

/* Component `axis` of grad(R Y_l^m) at the offset (dx, dy, dz). */
static double _Complex gradient_value(double dx, double dy, double dz,
                                      double r, int axis, int l, int m,
                                      const double *breaks,
                                      const double *coeffs, int nb,
                                      double rc) {
    const int order = abs(m);
    double _Complex g;
    if (r < 1e-12) {
        /* At the nucleus only a p function has a gradient: R ~ R'(0) r, and
         * r Y_1^m is linear (sqrt(3) c z, or c_1 (x +- i y)). */
        if (l != 1)
            return 0.0;
        const double slope = radial_slope(0.0, breaks, coeffs, nb, rc, l);
        const double c0 = 1.0 / sqrt(4.0 * M_PI);
        if (order == 0)
            g = axis == 2 ? sqrt(3.0) * c0 : 0.0;
        else
            g = -sqrt(1.5) * c0 * (axis == 0 ? 1.0 : (axis == 1 ? I : 0.0));
        g *= slope;
    } else {
        const double x[3] = {dx, dy, dz};
        const double u = dz / r;
        const double _Complex w = (dx + I * dy) / r;
        double Q, dQ;
        legendre_q(l, order, u, &Q, &dQ);
        double _Complex wm = 1.0, wm1 = 1.0;   /* w^m, w^(m-1) */
        for (int k = 0; k < order; ++k) {
            wm1 = wm;
            wm *= w;
        }
        const double du = ((axis == 2 ? 1.0 : 0.0) - u * x[axis] / r) / r;
        const double _Complex e = axis == 0 ? 1.0 : (axis == 1 ? I : 0.0);
        const double _Complex dw = (e - w * x[axis] / r) / r;
        const double _Complex y = Q * wm;
        const double _Complex dy_ = dQ * du * wm
                                    + (order ? Q * order * wm1 * dw : 0.0);
        const double R = radial_value(r, breaks, coeffs, nb, rc, l);
        const double dR = radial_slope(r, breaks, coeffs, nb, rc, l);
        g = dR * (x[axis] / r) * y + R * dy_;
    }
    if (m < 0)
        g = ((order & 1) ? -1.0 : 1.0) * conj(g);
    return g;
}

void mandacaru_bloch_shell(const double *x, const double *y, const double *z,
                           long npts, const double *center,
                           const double *translations, int n_images,
                           const double _Complex *phases, int nk,
                           double support,
                           const double *breaks, const double *coeffs, int nb,
                           double rc, int l, int derivative,
                           const int *ms, const long *rows, int nm, long M,
                           double _Complex *out) {
    const double support2 = support * support;
    /* Points are taken in blocks of consecutive indices -- rows of a grid,
     * or 1-Bohr bins of a quadrature sphere, spatially compact.  An image is
     * skipped for a whole block when the support sphere misses the block's
     * bounding box; the images that reach it are evaluated once into a
     * buffer, and each k-point then accumulates its phases over contiguous
     * runs of points -- the k loop innermost per point wrote nk x M values
     * at a stride of npts and was memory-bound. */
    enum { BLOCK = 256 };
    const long n_blocks = (npts + BLOCK - 1) / BLOCK;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int *hits = malloc(sizeof(int) * (n_images > 0 ? n_images : 1));
        long capacity = 0;
        double _Complex *buffer = NULL;      /* (hit, m, point in block) */
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long b = 0; b < n_blocks; ++b) {
            const long j0 = b * BLOCK;
            const long n = (j0 + BLOCK < npts ? j0 + BLOCK : npts) - j0;
            double lo[3] = {x[j0], y[j0], z[j0]};
            double hi[3] = {x[j0], y[j0], z[j0]};
            for (long j = j0 + 1; j < j0 + n; ++j) {
                const double p[3] = {x[j], y[j], z[j]};
                for (int a = 0; a < 3; ++a) {
                    if (p[a] < lo[a]) lo[a] = p[a];
                    if (p[a] > hi[a]) hi[a] = p[a];
                }
            }
            int n_hits = 0;
            for (int R = 0; R < n_images; ++R) {
                double gap2 = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double c = center[a] + translations[3 * R + a];
                    const double g = c < lo[a] ? lo[a] - c
                                   : (c > hi[a] ? c - hi[a] : 0.0);
                    gap2 += g * g;
                }
                if (gap2 < support2)
                    hits[n_hits++] = R;
            }
            if (n_hits == 0)
                continue;
            const long need = (long)n_hits * nm * BLOCK;
            if (need > capacity) {
                free(buffer);
                buffer = malloc(sizeof(double _Complex) * need);
                capacity = need;
            }
            for (int h = 0; h < n_hits; ++h) {
                const int R = hits[h];
                double _Complex *values = buffer + (long)h * nm * BLOCK;
                for (long jj = 0; jj < n; ++jj) {
                    const long j = j0 + jj;
                    for (int i = 0; i < nm; ++i)
                        values[i * BLOCK + jj] = 0.0;
                    const double dx = x[j] - center[0] - translations[3 * R];
                    const double dy = y[j] - center[1] - translations[3 * R + 1];
                    const double dz = z[j] - center[2] - translations[3 * R + 2];
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= support2)
                        continue;
                    const double r = sqrt(r2);
                    if (derivative >= 0) {
                        for (int i = 0; i < nm; ++i)
                            values[i * BLOCK + jj] = gradient_value(
                                dx, dy, dz, r, derivative, l, ms[i], breaks,
                                coeffs, nb, rc);
                        continue;
                    }
                    const double radial = radial_value(r, breaks, coeffs, nb,
                                                       rc, l);
                    if (radial == 0.0)
                        continue;
                    /* The angles of the NumPy path: theta = arccos(dz / r)
                     * (pi / 2 at the nucleus), phi = arctan2(dy, dx) (0 on
                     * the z axis). */
                    const double rho = sqrt(dx * dx + dy * dy);
                    const double cos_t = r > 0.0 ? dz / r : 0.0;
                    const double sin_t = r > 0.0 ? rho / r : 1.0;
                    const double _Complex unit = rho > 0.0
                        ? (dx + I * dy) / rho : 1.0;
                    for (int i = 0; i < nm; ++i) {
                        const int order = abs(ms[i]);
                        double _Complex e = 1.0;
                        for (int k = 0; k < order; ++k)
                            e *= unit;
                        double _Complex y_lm = legendre(l, order, cos_t, sin_t)
                                               * e;
                        if (ms[i] < 0)
                            y_lm = ((order & 1) ? -1.0 : 1.0) * conj(y_lm);
                        values[i * BLOCK + jj] = radial * y_lm;
                    }
                }
            }
            for (int k = 0; k < nk; ++k) {
                for (int i = 0; i < nm; ++i) {
                    double _Complex *target = out + ((long)k * M + rows[i])
                                              * npts + j0;
                    for (int h = 0; h < n_hits; ++h) {
                        const double _Complex phase =
                            phases[(long)k * n_images + hits[h]];
                        const double _Complex *values =
                            buffer + ((long)h * nm + i) * BLOCK;
                        for (long jj = 0; jj < n; ++jj)
                            target[jj] += phase * values[jj];
                    }
                }
            }
        }
        free(buffer);
        free(hits);
    }
}
