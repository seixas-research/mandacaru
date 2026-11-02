//! The spinor kinetic-energy density (`periodic::setup::spinor_tau`, the WP-X port of
//! `PeriodicPAW.spinor_kinetic_energy_density`): its trace and vector part against the definition
//! written out with explicit Pauli matrices and point-by-point loops, and the collinear limit
//! `P = diag(P_a, P_b)` against the scalar densities of each spin. The Python freeze cannot be
//! replayed: it holds the gradients of the first k-point only, and the density needs every one,
//! so these identities (which the freeze satisfies to 0.0) are the gate.
use mandacaru_core::{periodic::setup::spinor_tau::spinor_kinetic_density, Context};
use num_complex::Complex64;

const M: usize = 3;
const POINTS: usize = 17;
const WEIGHTS: [f64; 3] = [0.25, 0.5, 0.25];

struct Stream(u64);
impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
    fn complex(&mut self) -> Complex64 {
        Complex64::new(self.next(), self.next())
    }
}

/// A Hermitian positive semidefinite `size x size` matrix, row-major.
fn psd(stream: &mut Stream, size: usize) -> Vec<Complex64> {
    let a: Vec<Complex64> = (0..size * 3).map(|_| stream.complex()).collect();
    let mut out = vec![Complex64::default(); size * size];
    for i in 0..size {
        for j in 0..size {
            out[i * size + j] = (0..3)
                .map(|r| a[i * 3 + r] * a[j * 3 + r].conj())
                .sum::<Complex64>()
                / 3.;
        }
    }
    out
}

fn gradients(stream: &mut Stream) -> Vec<Vec<Complex64>> {
    WEIGHTS
        .iter()
        .map(|_| (0..3 * M * POINTS).map(|_| stream.complex()).collect())
        .collect()
}

/// `T_{st}(x) = 1/2 sum_k w sum_axis sum_ab P[(a,s),(b,t)] d_a d_b^*`, point by point.
fn explicit(
    matrices: &[Vec<Complex64>],
    grads: &[Vec<Complex64>],
    x: usize,
) -> [[Complex64; 2]; 2] {
    let mut t = [[Complex64::default(); 2]; 2];
    for (k, w) in WEIGHTS.iter().enumerate() {
        for axis in 0..3 {
            let d = |a: usize| grads[k][(axis * M + a) * POINTS + x];
            for s in 0..2 {
                for u in 0..2 {
                    for a in 0..M {
                        for b in 0..M {
                            t[s][u] += matrices[k][(s * M + a) * 2 * M + u * M + b]
                                * d(a)
                                * d(b).conj()
                                * (0.5 * w);
                        }
                    }
                }
            }
        }
    }
    t
}

fn call(matrices: &[Vec<Complex64>], grads: &[Vec<Complex64>]) -> (Vec<f64>, Vec<f64>) {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let p: Vec<&[Complex64]> = matrices.iter().map(|v| v.as_slice()).collect();
    let g: Vec<&[Complex64]> = grads.iter().map(|v| v.as_slice()).collect();
    let out = spinor_kinetic_density(&ctx, M, POINTS, &WEIGHTS, &p, &g).unwrap();
    (out.tau, out.tau_m)
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}

#[test]
fn the_trace_and_the_vector_part_are_the_pauli_traces_of_the_density_matrix() {
    let mut stream = Stream(11);
    let matrices: Vec<Vec<Complex64>> = WEIGHTS.iter().map(|_| psd(&mut stream, 2 * M)).collect();
    let grads = gradients(&mut stream);
    let (tau, tau_m) = call(&matrices, &grads);
    let (mut want_tau, mut want_m) = (vec![], vec![vec![]; 3]);
    for x in 0..POINTS {
        let t = explicit(&matrices, &grads, x);
        want_tau.push((t[0][0] + t[1][1]).re);
        // m_a = tr(T sigma_a) with explicit Pauli matrices sigma[a][row][column].
        let sigma = [
            [
                [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                [Complex64::new(1., 0.), Complex64::new(0., 0.)],
            ],
            [
                [Complex64::new(0., 0.), Complex64::new(0., -1.)],
                [Complex64::new(0., 1.), Complex64::new(0., 0.)],
            ],
            [
                [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                [Complex64::new(0., 0.), Complex64::new(-1., 0.)],
            ],
        ];
        for a in 0..3 {
            let trace: Complex64 = (0..2)
                .flat_map(|s| (0..2).map(move |u| (s, u)))
                .map(|(s, u)| t[s][u] * sigma[a][u][s])
                .sum();
            want_m[a].push(trace.re);
        }
    }
    let scale = want_tau.iter().fold(0f64, |m, v| m.max(v.abs()));
    let miss_tau = worst(&tau, &want_tau);
    let miss_m = (0..3)
        .map(|a| worst(&tau_m[a * POINTS..(a + 1) * POINTS], &want_m[a]))
        .fold(0., f64::max);
    println!(
        "spinor tau: trace {miss_tau:.2e}, vector part {miss_m:.2e} (array maximum {scale:.2e})"
    );
    assert!(scale > 1e-2 && miss_tau < 1e-13 * scale && miss_m < 1e-13 * scale);
    // The vector part is not zero (a wrong block or a transposed matrix would show).
    let transverse = tau_m[..2 * POINTS].iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(transverse > 1e-2 * scale);
    // Controls: P transposed (a conjugate partner) and the lower off-diagonal block in place of the upper.
    let transposed: Vec<Vec<Complex64>> = matrices
        .iter()
        .map(|p| {
            (0..4 * M * M)
                .map(|i| p[(i % (2 * M)) * 2 * M + i / (2 * M)])
                .collect()
        })
        .collect();
    let (_, other) = call(&transposed, &grads);
    assert!(worst(&other, &tau_m) > 1e2 * 1e-13 * scale);
}

#[test]
fn the_collinear_limit_is_the_sum_and_the_difference_of_the_scalar_densities() {
    let mut stream = Stream(23);
    let (pa, pb): (Vec<_>, Vec<_>) = WEIGHTS
        .iter()
        .map(|_| (psd(&mut stream, M), psd(&mut stream, M)))
        .unzip();
    let grads = gradients(&mut stream);
    let blocks: Vec<Vec<Complex64>> = pa
        .iter()
        .zip(&pb)
        .map(|(a, b)| {
            let mut p = vec![Complex64::default(); 4 * M * M];
            for i in 0..M {
                for j in 0..M {
                    p[i * 2 * M + j] = a[i * M + j];
                    p[(M + i) * 2 * M + M + j] = b[i * M + j];
                }
            }
            p
        })
        .collect();
    let (tau, tau_m) = call(&blocks, &grads);
    // The scalar density of one spin: 1/2 sum_k w sum_axis Re sum_ab P_ab d_a d_b^*.
    let scalar = |p: &[Vec<Complex64>]| -> Vec<f64> {
        (0..POINTS)
            .map(|x| {
                let mut sum = 0.;
                for (k, w) in WEIGHTS.iter().enumerate() {
                    for axis in 0..3 {
                        let d = |a: usize| grads[k][(axis * M + a) * POINTS + x];
                        for a in 0..M {
                            for b in 0..M {
                                sum += 0.5 * w * (p[k][a * M + b] * d(a) * d(b).conj()).re;
                            }
                        }
                    }
                }
                sum
            })
            .collect()
    };
    let (up, down) = (scalar(&pa), scalar(&pb));
    let total: Vec<f64> = up.iter().zip(&down).map(|(u, d)| u + d).collect();
    let diff: Vec<f64> = up.iter().zip(&down).map(|(u, d)| u - d).collect();
    let scale = total.iter().fold(0f64, |m, v| m.max(v.abs()));
    let miss = worst(&tau, &total)
        .max(worst(&tau_m[2 * POINTS..], &diff))
        .max(tau_m[..2 * POINTS].iter().fold(0f64, |m, v| m.max(v.abs())));
    println!("collinear limit: {miss:.2e} of {scale:.2e}");
    assert!(scale > 1e-2 && miss < 1e-13 * scale);
    // Control: the difference of the spins is far from their sum.
    assert!(worst(&diff, &total) > 1e-2 * scale);
}

#[test]
fn inconsistent_inputs_raise() {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let p = vec![Complex64::default(); 4 * M * M];
    let g = vec![Complex64::default(); 3 * M * POINTS];
    assert!(spinor_kinetic_density(&ctx, M, POINTS, &[1.], &[&p], &[&g[..g.len() - 1]]).is_err());
    assert!(spinor_kinetic_density(&ctx, M, POINTS, &[1.], &[&p[1..]], &[&g]).is_err());
    assert!(spinor_kinetic_density(&ctx, M, POINTS, &[], &[], &[]).is_err());
}
