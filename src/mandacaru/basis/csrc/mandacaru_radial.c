/*
 * mandacaru_radial.c -- the radial kernels of atom and pseudopotential
 * generation, in C.
 *
 * Profiling an iron PAW-LCAO generation puts 67 % of its time in the k-th
 * eigenpair of a symmetric tridiagonal matrix (the uniform-grid radial
 * equation, solved thousands of times through the relativistic M(eps) loop)
 * and another 19 % in two Numerov recursions written as Python loops.  These
 * are the three functions below.  The Python side (basis/radial_backend.py)
 * loads this library with ctypes and keeps its own reference implementations
 * as the fallback, so every number can be checked against them.
 *
 * Build: cc -std=c11 -O2 -fPIC -shared mandacaru_radial.c -o libmandacaru_radial.so -lm
 * (radial_backend.build_radial_backend() does this on first use.)
 *
 * No -ffast-math: the Sturm count relies on IEEE signed zeros and the
 * reproducibility tests compare against the reference kernels to 1e-12.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "mandacaru_radial.h"

/* ------------------------------------------------------------------------ */
/* Numerov recursions.                                                       */
/* ------------------------------------------------------------------------ */

/* u'' = f u + s, outward from u[start], u[start+1] (already set in u).
 * Same arithmetic, in the same order, as oncv.numerov_outward. */
void mandacaru_numerov_outward(int n, const double *f, const double *s,
                               double h2, int start, double *u)
{
    const double c = h2 / 12.0;
    for (int i = start + 1; i < n - 1; ++i) {
        const double a_im1 = 1.0 - h2 * f[i - 1] / 12.0;
        const double b_i = 2.0 * (1.0 + 5.0 * h2 * f[i] / 12.0);
        const double a_ip1 = 1.0 - h2 * f[i + 1] / 12.0;
        u[i + 1] = (b_i * u[i] - a_im1 * u[i - 1]
                    + c * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / a_ip1;
    }
}

/* Homogeneous inward recursion from u[n-1], u[n-2] (already set) down to
 * index stop.  Same arithmetic as oncv._numerov_inward. */
void mandacaru_numerov_inward(int n, const double *f, double h2, int stop,
                              double *u)
{
    for (int i = n - 2; i > stop; --i) {
        const double a_ip1 = 1.0 - h2 * f[i + 1] / 12.0;
        const double b_i = 2.0 * (1.0 + 5.0 * h2 * f[i] / 12.0);
        const double a_im1 = 1.0 - h2 * f[i - 1] / 12.0;
        u[i - 1] = (b_i * u[i] - a_ip1 * u[i + 1]) / a_im1;
    }
}

/* ------------------------------------------------------------------------ */
/* k-th eigenpair of a symmetric tridiagonal matrix.                         */
/* ------------------------------------------------------------------------ */

/* Number of eigenvalues strictly below x (Sturm sequence of the LDL^T
 * factorization of T - x I).  A zero pivot is replaced by a tiny one of the
 * matrix's scale, as LAPACK's dstebz does. */
static int sturm_count(int n, const double *d, const double *e2, double x,
                       double pivmin)
{
    int count = 0;
    double q = d[0] - x;
    if (fabs(q) < pivmin) q = -pivmin;
    if (q < 0.0) ++count;
    for (int i = 1; i < n; ++i) {
        q = d[i] - x - e2[i - 1] / q;
        if (fabs(q) < pivmin) q = -pivmin;
        if (q < 0.0) ++count;
    }
    return count;
}

/* Solve (T - lambda I) x = b in place by Gaussian elimination with partial
 * pivoting (the tridiagonal algorithm of LAPACK's dgtsv): the factor has a
 * second superdiagonal.  A pivot that is exactly zero is nudged, which is
 * what inverse iteration wants -- the solution is then just very large. */
static void shifted_solve(int n, const double *d, const double *e,
                          double lambda, double *x, double *dl, double *dd,
                          double *du, double *du2, double tiny)
{
    for (int i = 0; i < n; ++i) dd[i] = d[i] - lambda;
    for (int i = 0; i < n - 1; ++i) { dl[i] = e[i]; du[i] = e[i]; }
    for (int i = 0; i < n - 2; ++i) du2[i] = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        if (fabs(dd[i]) >= fabs(dl[i])) {
            if (dd[i] == 0.0) dd[i] = tiny;
            const double m = dl[i] / dd[i];
            dd[i + 1] -= m * du[i];
            x[i + 1] -= m * x[i];
            if (i < n - 2) du2[i] = 0.0;
            dl[i] = m;
        } else {
            /* swap rows i and i+1 */
            const double m = dd[i] / dl[i];
            dd[i] = dl[i];
            const double t = dd[i + 1];
            dd[i + 1] = du[i] - m * t;
            if (i < n - 2) {
                du2[i] = du[i + 1];
                du[i + 1] = -m * du2[i];
            }
            du[i] = t;
            const double xt = x[i];
            x[i] = x[i + 1];
            x[i + 1] = xt - m * x[i + 1];
            dl[i] = m;
        }
    }
    if (dd[n - 1] == 0.0) dd[n - 1] = tiny;
    /* back substitution */
    x[n - 1] /= dd[n - 1];
    if (n > 1) x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / dd[n - 2];
    for (int i = n - 3; i >= 0; --i)
        x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i];
}

/* Eigenvector of T at an (accurate) eigenvalue lambda by twisted
 * factorization, as in the MRRR algorithm: the top-down pivots D+ and the
 * bottom-up pivots D- of T - lambda I meet at the index r where
 * gamma_r = D+_r + D-_r - (d_r - lambda) is smallest, and from there every
 * component is a product of ratios -e/D taken in the direction the solution
 * decays.  No subtraction of nearly equal numbers is involved, so a component
 * of 1e-40 has full relative accuracy and the right sign -- which inverse
 * iteration cannot give below eps * max|x|: oxygen's 1s came out with a
 * spurious node at 7.3 Bohr, amplitude 1e-22.  Returns 0, or -3 when the
 * products overflowed or vanished and x is not a usable vector. */
static int twisted_vector(int n, const double *d, const double *e,
                           double lambda, double pivmin, double *dplus,
                           double *dminus, double *x)
{
    dplus[0] = d[0] - lambda;
    if (fabs(dplus[0]) < pivmin) dplus[0] = -pivmin;
    for (int i = 1; i < n; ++i) {
        dplus[i] = d[i] - lambda - e[i - 1] * e[i - 1] / dplus[i - 1];
        if (fabs(dplus[i]) < pivmin) dplus[i] = -pivmin;
    }
    dminus[n - 1] = d[n - 1] - lambda;
    if (fabs(dminus[n - 1]) < pivmin) dminus[n - 1] = -pivmin;
    for (int i = n - 2; i >= 0; --i) {
        dminus[i] = d[i] - lambda - e[i] * e[i] / dminus[i + 1];
        if (fabs(dminus[i]) < pivmin) dminus[i] = -pivmin;
    }
    int twist = 0;
    double best = INFINITY;
    for (int i = 0; i < n; ++i) {
        const double gamma = fabs(dplus[i] + dminus[i] - (d[i] - lambda));
        if (gamma < best) { best = gamma; twist = i; }
    }
    x[twist] = 1.0;
    for (int i = twist - 1; i >= 0; --i)
        x[i] = -(e[i] / dplus[i]) * x[i + 1];
    for (int i = twist + 1; i < n; ++i)
        x[i] = -(e[i - 1] / dminus[i]) * x[i - 1];
    double norm = 0.0;
    for (int i = 0; i < n; ++i) norm += x[i] * x[i];
    norm = sqrt(norm);
    if (!(norm > 0.0) || !isfinite(norm)) return -3;
    for (int i = 0; i < n; ++i) x[i] /= norm;
    return 0;
}

/* The k-th smallest eigenvalue (k = 0, 1, ...) of the symmetric tridiagonal
 * matrix with diagonal d[0..n-1] and off-diagonal e[0..n-2], and its unit
 * eigenvector.  `guess` (NaN for none) is where to start looking.  Bisection on the Sturm count to full double precision, then
 * inverse iteration at that shift.  Returns 0 on success, -1 on bad input,
 * -2 when memory could not be allocated, -3 when the eigenvector came out
 * non-finite or zero (never silently). */
int mandacaru_tridiagonal_eigenpair(int n, const double *d, const double *e,
                                    int k, double guess, double *value,
                                    double *vector)
{
    if (n < 1 || k < 0 || k >= n) return -1;
    if (n == 1) { *value = d[0]; vector[0] = 1.0; return 0; }

    double *work = (double *)malloc(sizeof(double) * (size_t)(6 * n));
    if (work == NULL) return -2;
    double *e2 = work, *dl = work + n, *dd = work + 2 * n, *du = work + 3 * n,
           *du2 = work + 4 * n, *x = work + 5 * n;

    /* Gershgorin interval and the matrix scale. */
    double lo = d[0] - fabs(e[0]), hi = d[0] + fabs(e[0]), scale = 0.0;
    for (int i = 0; i < n; ++i) {
        const double left = (i > 0) ? fabs(e[i - 1]) : 0.0;
        const double right = (i < n - 1) ? fabs(e[i]) : 0.0;
        if (d[i] - left - right < lo) lo = d[i] - left - right;
        if (d[i] + left + right > hi) hi = d[i] + left + right;
        if (fabs(d[i]) + left + right > scale) scale = fabs(d[i]) + left + right;
    }
    for (int i = 0; i < n - 1; ++i) e2[i] = e[i] * e[i];
    const double pivmin = DBL_MIN * (scale > 1.0 ? scale : 1.0);
    const double pad = 2.0 * DBL_EPSILON * scale + pivmin;
    lo -= pad;
    hi += pad;

    /* count(lo) <= k < count(hi): the k-th eigenvalue is in [lo, hi).
     * Bisect only until it is bracketed to 1e-6 (relative); Rayleigh-quotient
     * inverse iteration then converges cubically.  Radial levels are far
     * apart on that scale, so the bracket holds one eigenvalue. */
    const double tiny = DBL_EPSILON * scale + pivmin;
    double full_lo = lo, full_hi = hi;
    /* A guess (the previous eigenvalue of a self-consistent loop) brackets
     * the eigenvalue in a few counts instead of ~30 halvings of the
     * Gershgorin interval.  NaN means none. */
    if (isfinite(guess) && guess > lo && guess < hi) {
        double step = 1e-4 * fmax(1.0, fabs(guess));
        double a = guess - step, b = guess + step;
        while (a > lo && sturm_count(n, d, e2, a, pivmin) > k) {
            step *= 4.0;
            a = guess - step;
        }
        step = 1e-4 * fmax(1.0, fabs(guess));
        while (b < hi && sturm_count(n, d, e2, b, pivmin) <= k) {
            step *= 4.0;
            b = guess + step;
        }
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }
    for (int it = 0; it < 256; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;         /* no representable midpoint */
        if (sturm_count(n, d, e2, mid, pivmin) > k) hi = mid;
        else lo = mid;
        if (hi - lo <= 1e-6 * fmax(1.0, fabs(mid))) break;
    }
    double lambda = 0.5 * (lo + hi);
    for (int i = 0; i < n; ++i) vector[i] = 1.0 / sqrt((double)n);
    int accepted = 0;
    for (int it = 0; it < 8; ++it) {
        memcpy(x, vector, sizeof(double) * (size_t)n);
        shifted_solve(n, d, e, lambda, x, dl, dd, du, du2, tiny);
        double norm = 0.0;
        for (int i = 0; i < n; ++i) norm += x[i] * x[i];
        norm = sqrt(norm);
        if (!(norm > 0.0) || !isfinite(norm)) break;
        for (int i = 0; i < n; ++i) x[i] /= norm;
        memcpy(vector, x, sizeof(double) * (size_t)n);
        /* Rayleigh quotient x^T T x */
        double rq = 0.0;
        for (int i = 0; i < n; ++i) {
            double tx = d[i] * x[i];
            if (i > 0) tx += e[i - 1] * x[i - 1];
            if (i < n - 1) tx += e[i] * x[i + 1];
            rq += x[i] * tx;
        }
        if (rq < lo - 1e-3 * (hi - lo) - pad || rq > hi + 1e-3 * (hi - lo) + pad)
            break;                                 /* wandered to a neighbor */
        /* The Rayleigh quotient of a matrix of norm `scale` is only good to
         * about eps * scale -- the same backward error LAPACK's bisection
         * reaches -- so that is the convergence test, not an ulp of rq. */
        const double change = fabs(rq - lambda);
        lambda = rq;
        if (it >= 1 && change <= 8.0 * DBL_EPSILON * scale) {
            /* The bracket is assumed to hold one eigenvalue; confirm that
             * the one converged to is the k-th, not a close neighbor. */
            const double margin = 64.0 * DBL_EPSILON * scale + pivmin;
            accepted = sturm_count(n, d, e2, rq - margin, pivmin) <= k
                       && sturm_count(n, d, e2, rq + margin, pivmin) > k;
            break;
        }
    }
    if (!accepted) {
        /* Fall back to bisection to full precision and plain inverse
         * iteration at that shift. */
        lo = full_lo; hi = full_hi;
        for (int it = 0; it < 256; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            if (sturm_count(n, d, e2, mid, pivmin) > k) hi = mid;
            else lo = mid;
        }
        lambda = 0.5 * (lo + hi);
        for (int i = 0; i < n; ++i) vector[i] = 1.0 / sqrt((double)n);
        for (int it = 0; it < 6; ++it) {
            memcpy(x, vector, sizeof(double) * (size_t)n);
            shifted_solve(n, d, e, lambda, x, dl, dd, du, du2, tiny);
            double norm = 0.0;
            for (int i = 0; i < n; ++i) norm += x[i] * x[i];
            norm = sqrt(norm);
            if (!(norm > 0.0) || !isfinite(norm)) break;
            double overlap = 0.0;
            for (int i = 0; i < n; ++i) {
                x[i] /= norm;
                overlap += x[i] * vector[i];
            }
            memcpy(vector, x, sizeof(double) * (size_t)n);
            if (it > 0 && fabs(fabs(overlap) - 1.0) < 1e-14) break;
        }
    }
    *value = lambda;
    /* The vector inverse iteration produced has the right shape but noise
     * below eps * max|x|; recompute it from the converged eigenvalue. */
    const int status = twisted_vector(n, d, e, lambda, pivmin, dl, dd, vector);
    free(work);
    return status;
}

int mandacaru_radial_abi_version(void) { return MANDACARU_RADIAL_ABI; }
