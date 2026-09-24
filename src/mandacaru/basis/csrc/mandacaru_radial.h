/* mandacaru_radial.h -- radial kernels of atom and pseudopotential generation. */
#ifndef MANDACARU_RADIAL_H
#define MANDACARU_RADIAL_H

/* Bumped whenever a signature changes; the loader refuses a stale library. */
#define MANDACARU_RADIAL_ABI 2

void mandacaru_numerov_outward(int n, const double *f, const double *s,
                               double h2, int start, double *u);
void mandacaru_numerov_inward(int n, const double *f, double h2, int stop,
                              double *u);
int mandacaru_tridiagonal_eigenpair(int n, const double *d, const double *e,
                                    int k, double guess, double *value,
                                    double *vector);
int mandacaru_radial_abi_version(void);

#endif
