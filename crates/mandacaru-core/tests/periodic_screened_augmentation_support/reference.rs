//! An independent build of K(k): a direct Fourier sum of the pair densities at the wave
//! vector `Q` of each grid index, with the compensation charges from the raw definition
//! (explicit loops over projectors and channels), no FFT, no `pair_shapes`, no `moments`.
//! Only the kernel values, the Bloch squares and the shape transforms are shared with the
//! code under test (each has its own gate elsewhere).
use super::Matrix;
use mandacaru_core::{
    math::{fft::frequency, linalg::eigen, reciprocal},
    periodic::{
        screened_exchange::{gygi_baldereschi_screened_kernel, screened_kernel, ExxDivTreatment},
        setup::shapes::{Shapes, Sites},
        Crystal,
    },
};
use nalgebra::Vector3;
use num_complex::Complex64;
use std::f64::consts::PI;

pub struct Compensation<'a> {
    pub sites: &'a Sites<'a>,
    pub shapes: &'a Shapes,
    /// The setup's alias-averaged hats `(channels, n)`, used where `k_diff = 0` instead of the
    /// shape transforms at the wrapped vector (the choice the log records).
    pub alias_hats: Option<&'a [Complex64]>,
}

/// `K(k_target)` of one spin channel from the density matrices `d` (the solver's, restricted
/// ones carry the spin degeneracy).
pub fn exchange(
    c: &Crystal,
    d: &[Matrix],
    spin: usize,
    target: usize,
    omega: f64,
    exxdiv: ExxDivTreatment,
    compensation: Option<&Compensation>,
) -> Matrix {
    let (m, p, n) = (c.m, c.p, c.grid.index.points());
    let components = if c.spinor { 2 } else { 1 };
    let nb = components * m;
    let nk = c.points.len();
    let shape = c.grid.index.shape();
    let dv = c.grid.index.dv();
    let volume = dv * n as f64;
    let scale = if c.spins == 1 && !c.spinor { 0.5 } else { 1.0 };
    let b = reciprocal::reciprocal_cell(&c.grid).unwrap();
    let binv = b.try_inverse().unwrap();
    let point = &c.points[target];
    let u_target: Vec<Vec<Complex64>> = {
        let samples = point.psi.read(&c.context).unwrap();
        (0..m)
            .map(|mu| {
                (0..n)
                    .map(|r| {
                        let x = c.grid.point(r);
                        let angle = -(0..3).map(|a| point.k[a] * x[a]).sum::<f64>();
                        samples[mu * n + r] * Complex64::from_polar(1., angle)
                    })
                    .collect()
            })
            .collect()
    };
    let mut k_matrix = Matrix::zeros(nb, nb);
    for q in 0..nk {
        let source = &c.points[q];
        let (values, vectors) = eigen(&d[spin * nk + q] * Complex64::new(scale, 0.)).unwrap();
        let samples = source.psi.read(&c.context).unwrap();
        let k_diff = [
            source.k[0] - point.k[0],
            source.k[1] - point.k[1],
            source.k[2] - point.k[2],
        ];
        let squares = reciprocal::bloch(&c.context, &c.grid, k_diff).unwrap();
        let kappa = binv * Vector3::from(k_diff);
        for (state, &occupation) in values.iter().enumerate() {
            if occupation <= 1e-10 {
                continue;
            }
            let coefficient: Vec<Complex64> = (0..nb)
                .map(|mu| vectors[(mu, state)] * occupation.sqrt())
                .collect();
            // u_n(r) = psi_n(r) exp(-i q.r), one per spin component
            let u_state: Vec<Vec<Complex64>> = (0..components)
                .map(|comp| {
                    (0..n)
                        .map(|r| {
                            let x = c.grid.point(r);
                            let angle = -(0..3).map(|a| source.k[a] * x[a]).sum::<f64>();
                            (0..m)
                                .map(|mu| coefficient[comp * m + mu] * samples[mu * n + r])
                                .sum::<Complex64>()
                                * Complex64::from_polar(1., angle)
                        })
                        .collect()
                })
                .collect();
            // a_j = <p_j|psi_n>, then Q^a_{mu n} = sum_{ij} C(k)[mu, i] B_a[i, j] a_j.
            let moments: Vec<Vec<Complex64>> = match compensation {
                None => vec![],
                Some(comp) => {
                    let a: Vec<Vec<Complex64>> = (0..components)
                        .map(|comp| {
                            (0..p)
                                .map(|j| {
                                    (0..m)
                                        .map(|mu| {
                                            source.projections[(mu, j)].conj()
                                                * coefficient[comp * m + mu]
                                        })
                                        .sum()
                                })
                                .collect()
                        })
                        .collect();
                    (0..nb)
                        .map(|row| {
                            let (component, mu) = (row / m, row % m);
                            (0..comp.sites.channels.len())
                                .map(|ch| {
                                    let mut total = Complex64::default();
                                    for i in 0..p {
                                        for j in 0..p {
                                            total += point.projections[(mu, i)]
                                                * c.blocks.as_slice()[(ch * p + i) * p + j]
                                                * a[component][j];
                                        }
                                    }
                                    total
                                })
                                .collect()
                        })
                        .collect()
                }
            };
            for at in 0..n {
                let index = [
                    at / (shape[1] * shape[2]),
                    at / shape[2] % shape[1],
                    at % shape[2],
                ];
                let wrapped = Vector3::from_fn(|a, _| {
                    let size = shape[a] as f64;
                    let x = frequency(index[a], shape[a]) + kappa[a];
                    x - size * ((x + 0.5 * size + 1e-9) / size).floor()
                });
                let wave = b * wrapped;
                let g = wave - Vector3::from(k_diff);
                let q2 = squares.squares.as_slice()[at];
                let kernel = if q2 > 1e-12 {
                    screened_kernel(q2, omega)
                } else {
                    match exxdiv {
                        ExxDivTreatment::GygiBaldereschi => gygi_baldereschi_screened_kernel(
                            (6. * PI * PI * source.weight / volume).cbrt(),
                            omega,
                        ),
                        ExxDivTreatment::None => screened_kernel(0., omega),
                    }
                };
                let phase: Vec<Complex64> = (0..n)
                    .map(|r| {
                        let x = c.grid.point(r);
                        Complex64::from_polar(1., -(g[0] * x[0] + g[1] * x[1] + g[2] * x[2]))
                    })
                    .collect();
                let mut hats =
                    vec![Complex64::default(); compensation.map_or(0, |s| s.sites.channels.len())];
                if let Some(comp) = compensation {
                    comp.shapes
                        .transforms(comp.sites, &[wave.into()], &mut hats, 1)
                        .unwrap();
                    if let (Some(alias), true) = (comp.alias_hats, k_diff == [0.; 3]) {
                        for (a, hat) in hats.iter_mut().enumerate() {
                            *hat = alias[a * n + at];
                        }
                    }
                }
                let rho: Vec<Complex64> = (0..nb)
                    .map(|row| {
                        let (comp_index, mu) = (row / m, row % m);
                        let grid: Complex64 = (0..n)
                            .map(|r| u_target[mu][r].conj() * u_state[comp_index][r] * phase[r])
                            .sum::<Complex64>()
                            / n as f64;
                        let comp: Complex64 = if compensation.is_some() {
                            (0..hats.len())
                                .map(|a| moments[row][a] * hats[a])
                                .sum::<Complex64>()
                                / volume
                        } else {
                            Complex64::default()
                        };
                        grid + comp
                    })
                    .collect();
                for i in 0..nb {
                    for j in 0..nb {
                        k_matrix[(i, j)] +=
                            rho[i] * rho[j].conj() * (source.weight * volume * kernel);
                    }
                }
            }
        }
    }
    k_matrix
}
