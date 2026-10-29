//! Local MP2 in Boys-localized occupied orbitals and pair-natural domains.
use super::{correlation::Mp2Result, invalid, transform, MeanField, MolecularIntegrals};
use crate::{math::linalg::real_eigen, product, quantum::C64, Reservation, Result};
use nalgebra::DMatrix;
#[derive(Clone)]
pub struct Options {
    pub full_domains: bool,
    pub pno_cutoff: f64,
    pub doi_cutoff: f64,
    pub domain_population: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            full_domains: false,
            pno_cutoff: 1e-8,
            doi_cutoff: 1e-2,
            domain_population: 0.02,
        }
    }
}
pub struct LocalResult {
    pub mp2: Mp2Result,
    pub pno_counts: Vec<usize>,
    pub iterations: usize,
    pub domains: Vec<Vec<usize>>,
    _memory: Reservation,
}
struct Pair {
    i: usize,
    j: usize,
    c: DMatrix<f64>,
    eps: Vec<f64>,
    k: DMatrix<f64>,
    t: DMatrix<f64>,
}
fn canonical_domain(p: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (values, vectors) = real_eigen(p.transpose() * p)?;
    let cutoff = 1e-7 * values.iter().copied().fold(1_f64, f64::max);
    let keep: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > cutoff)
        .map(|(i, _)| i)
        .collect();
    if keep.is_empty() {
        return Err(invalid(
            "a local MP2 domain has no independent virtual orbitals",
        ));
    }
    Ok(p * DMatrix::from_fn(p.ncols(), keep.len(), |i, j| {
        vectors[(i, keep[j])] / values[keep[j]].sqrt()
    }))
}
fn amplitude(pair: &Pair, i: usize, j: usize) -> DMatrix<f64> {
    if i <= j {
        pair.t.clone()
    } else {
        pair.t.transpose()
    }
}
pub fn calculate(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    frozen: &[usize],
    options: &Options,
) -> Result<LocalResult> {
    if reference.method != "rhf"
        || reference.functional.is_some()
        || reference.particles[0] != reference.particles[1]
    {
        return Err(invalid(
            "local MP2 needs closed-shell Hartree-Fock orbitals",
        ));
    }
    if [
        options.pno_cutoff,
        options.doi_cutoff,
        options.domain_population,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(invalid(
            "local-correlation cutoffs must be nonnegative and finite",
        ));
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let o = reference.particles[0];
    let v = m - o;
    if o == 0 || v == 0 || frozen.iter().any(|&i| i >= o) {
        return Err(invalid(
            "local MP2 needs occupied/virtual orbitals and valid frozen indices",
        ));
    }
    if reference.model_orbitals.iter().any(|x| x.im.abs() > 1e-10) {
        return Err(invalid("local MP2 requires real orbitals"));
    }
    let correlated: Vec<_> = (0..o).filter(|i| !frozen.contains(i)).collect();
    let n = correlated.len();
    let np = n * (n + 1) / 2;
    let memory = ctx.reserve(product(&[np.max(1), m, m, 8, 12])?)?;
    let _work = ctx.reserve(
        product(&[4096, m, 8, 4])?
            .checked_add(product(&[m, m, 8, 24])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let c = reference.model_orbitals.map(|x| x.re);
    let g = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let mut position: [DMatrix<f64>; 3] = std::array::from_fn(|_| DMatrix::zeros(m, m));
    for start in (0..g).step_by(4096) {
        let size = (g - start).min(4096);
        let samples = DMatrix::from_fn(size, m, |i, j| {
            integrals.orthonormal_samples.as_slice()[j * g + start + i]
        });
        for (axis, r) in position.iter_mut().enumerate() {
            let weighted = DMatrix::from_fn(size, m, |i, j| {
                samples[(i, j)] * integrals.grid.point(start + i)[axis] * dv
            });
            *r += samples.transpose() * weighted;
        }
    }
    for r in &mut position {
        *r = c.transpose() * r.clone() * &c;
    }
    let mut rotation = DMatrix::<f64>::identity(n, n);
    let mut centers: Vec<_> = position
        .iter()
        .map(|r| DMatrix::from_fn(n, n, |i, j| r[(correlated[i], correlated[j])]))
        .collect();
    let mut localized = false;
    for _ in 0..200 {
        let mut largest = 0_f64;
        for i in 0..n {
            for j in i + 1..n {
                let a: Vec<_> = centers.iter().map(|r| r[(i, j)]).collect();
                let b: Vec<_> = centers.iter().map(|r| r[(i, i)] - r[(j, j)]).collect();
                let aa = a.iter().map(|x| x * x).sum::<f64>()
                    - 0.25 * b.iter().map(|x| x * x).sum::<f64>();
                let bb = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>();
                if aa.abs() < 1e-14 && bb.abs() < 1e-14 {
                    continue;
                }
                let angle = 0.25 * bb.atan2(-aa);
                largest = largest.max(angle.abs());
                if angle.abs() < 1e-10 {
                    continue;
                }
                let (s, q) = angle.sin_cos();
                for r in &mut centers {
                    for k in 0..n {
                        let (a, b) = (r[(i, k)], r[(j, k)]);
                        r[(i, k)] = q * a + s * b;
                        r[(j, k)] = -s * a + q * b;
                    }
                    for k in 0..n {
                        let (a, b) = (r[(k, i)], r[(k, j)]);
                        r[(k, i)] = q * a + s * b;
                        r[(k, j)] = -s * a + q * b;
                    }
                }
                for k in 0..n {
                    let (a, b) = (rotation[(k, i)], rotation[(k, j)]);
                    rotation[(k, i)] = q * a + s * b;
                    rotation[(k, j)] = -s * a + q * b;
                }
            }
        }
        if largest < 1e-10 {
            localized = true;
            break;
        }
    }
    if !localized {
        return Err(invalid("Boys localization did not converge"));
    }
    let lmo = DMatrix::from_fn(m, n, |i, j| {
        correlated
            .iter()
            .position(|&p| p == i)
            .map_or(0., |k| rotation[(k, j)])
    });
    let lmo_ao = &c * &lmo;
    let fock = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(reference.energies[0].clone()));
    let occupied_fock = lmo.transpose() * &fock * &lmo;
    let mut paos = c.transpose();
    for i in 0..o {
        paos.row_mut(i).fill(0.);
    }
    let alive: Vec<_> = (0..m).filter(|&j| paos.column(j).norm() > 1e-8).collect();
    let paos = DMatrix::from_fn(m, alive.len(), |i, j| {
        paos[(i, alive[j])] / paos.column(alive[j]).norm()
    });
    let owners: Vec<_> = alive
        .iter()
        .map(|&mu| integrals.basis.orbitals[mu].atom)
        .collect();
    let mut doi = DMatrix::<f64>::zeros(n, m);
    for start in (0..g).step_by(4096) {
        let size = (g - start).min(4096);
        let samples = DMatrix::from_fn(size, m, |i, j| {
            integrals.orthonormal_samples.as_slice()[j * g + start + i]
        });
        let local = (&samples * &lmo_ao).map(|x| x * x);
        doi += local.transpose() * samples.map(|x| x * x) * dv;
    }
    doi = doi.map(|x| x.max(0.).sqrt());
    let atoms = integrals.geometry.numbers().len();
    let populations = &c * &lmo;
    let domains: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            if options.full_domains {
                return (0..atoms).collect();
            }
            let population: Vec<_> = (0..atoms)
                .map(|atom| {
                    (0..m)
                        .filter(|&mu| integrals.basis.orbitals[mu].atom == atom)
                        .map(|mu| populations[(mu, i)].powi(2))
                        .sum::<f64>()
                })
                .collect();
            let largest = (0..atoms)
                .max_by(|&a, &b| population[a].total_cmp(&population[b]))
                .unwrap();
            (0..atoms)
                .filter(|&atom| {
                    atom == largest
                        || population[atom] >= options.domain_population
                        || alive
                            .iter()
                            .enumerate()
                            .any(|(j, &mu)| owners[j] == atom && doi[(i, mu)] >= options.doi_cutoff)
                })
                .collect()
        })
        .collect();
    let eri = integrals.eri()?;
    let mut pairs = Vec::new();
    let mut lookup = vec![usize::MAX; n * n];
    for i in 0..n {
        for j in i..n {
            let domain: Vec<_> = (0..alive.len())
                .filter(|&mu| domains[i].contains(&owners[mu]) || domains[j].contains(&owners[mu]))
                .collect();
            let p = DMatrix::from_fn(m, domain.len(), |a, b| paos[(a, domain[b])]);
            let q = canonical_domain(&p)?;
            let (eps, vectors) = real_eigen(q.transpose() * &fock * &q)?;
            let q = q * vectors;
            let ci = lmo_ao.columns(i, 1).map(|x| C64::new(x, 0.));
            let cj = lmo_ao.columns(j, 1).map(|x| C64::new(x, 0.));
            let cv = (&c * &q).map(|x| C64::new(x, 0.));
            let block = transform::four(ctx, &eri, [&ci, &cj, &cv, &cv])?;
            let size = q.ncols();
            let k = DMatrix::from_fn(size, size, |a, b| block.as_slice()[a * size + b].re);
            let denominator = |a: usize, b: usize| {
                eps[a] + eps[b] - occupied_fock[(i, i)] - occupied_fock[(j, j)]
            };
            if (0..size).any(|a| (0..size).any(|b| denominator(a, b).abs() < 1e-6)) {
                return Err(invalid("local MP2 denominator is below 1e-6 Ha"));
            }
            let t = DMatrix::from_fn(size, size, |a, b| -k[(a, b)] / denominator(a, b));
            let tt = &t * 2. - t.transpose();
            let density =
                (tt.transpose() * &t + &tt * t.transpose()) * (2. / if i == j { 2. } else { 1. });
            let (occupations, vectors) = real_eigen((&density + density.transpose()) * 0.5)?;
            let mut keep: Vec<_> = occupations
                .iter()
                .enumerate()
                .filter(|(_, v)| options.pno_cutoff == 0. || **v >= options.pno_cutoff)
                .map(|(a, _)| a)
                .collect();
            if keep.is_empty() {
                keep.push(size - 1);
            }
            let pno = &q * DMatrix::from_fn(size, keep.len(), |a, b| vectors[(a, keep[b])]);
            let (ep, w) = real_eigen(pno.transpose() * &fock * &pno)?;
            let pno = pno * w;
            let transform = q.transpose() * &pno;
            let k = transform.transpose() * k * &transform;
            let size = pno.ncols();
            let t = DMatrix::from_fn(size, size, |a, b| {
                -k[(a, b)] / (ep[a] + ep[b] - occupied_fock[(i, i)] - occupied_fock[(j, j)])
            });
            lookup[i * n + j] = pairs.len();
            lookup[j * n + i] = pairs.len();
            pairs.push(Pair {
                i,
                j,
                c: pno,
                eps: ep,
                k,
                t,
            });
        }
    }
    let mut iterations = 0;
    let mut converged = pairs.is_empty();
    for iteration in 0..200 {
        let mut next = Vec::new();
        let mut largest = 0_f64;
        for pair in &pairs {
            let size = pair.eps.len();
            let mut residual = &pair.k
                + DMatrix::from_fn(size, size, |a, b| {
                    (pair.eps[a] + pair.eps[b]) * pair.t[(a, b)]
                });
            for k in 0..n {
                if occupied_fock[(pair.i, k)].abs() > 1e-6 {
                    let other = &pairs[lookup[k * n + pair.j]];
                    let s = pair.c.transpose() * &other.c;
                    residual -= (&s * amplitude(other, k, pair.j) * s.transpose())
                        * occupied_fock[(pair.i, k)];
                }
                if occupied_fock[(k, pair.j)].abs() > 1e-6 {
                    let other = &pairs[lookup[pair.i * n + k]];
                    let s = pair.c.transpose() * &other.c;
                    residual -= (&s * amplitude(other, pair.i, k) * s.transpose())
                        * occupied_fock[(k, pair.j)];
                }
            }
            largest = largest.max(residual.amax());
            next.push(
                &pair.t
                    - DMatrix::from_fn(size, size, |a, b| {
                        residual[(a, b)]
                            / (pair.eps[a] + pair.eps[b]
                                - occupied_fock[(pair.i, pair.i)]
                                - occupied_fock[(pair.j, pair.j)])
                    }),
            );
        }
        for (pair, t) in pairs.iter_mut().zip(next) {
            pair.t = t;
        }
        iterations = iteration + 1;
        if largest < 1e-8 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(invalid("coupled local MP2 amplitudes did not converge"));
    }
    let mut energy = 0.;
    let mut virtual_density = DMatrix::<f64>::zeros(m, m);
    for pair in &pairs {
        energy += (if pair.i == pair.j { 1. } else { 2. })
            * pair
                .k
                .component_mul(&(&pair.t * 2. - pair.t.transpose()))
                .sum();
        for t in [
            Some(pair.t.clone()),
            if pair.i != pair.j {
                Some(pair.t.transpose())
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
        {
            virtual_density +=
                (&pair.c * ((&t * 2. - t.transpose()) * t.transpose()) * pair.c.transpose()) * 2.;
        }
    }
    let mut occupied_density = DMatrix::<f64>::identity(o, o) * 2.;
    let local_density = DMatrix::from_fn(n, n, |i, j| {
        let mut value = 0.;
        for k in 0..n {
            let ik = &pairs[lookup[i * n + k]];
            let jk = &pairs[lookup[j * n + k]];
            let a = amplitude(ik, i, k);
            let b = amplitude(jk, j, k);
            let s = ik.c.transpose() * &jk.c;
            value -= 2.
                * (&a * 2. - a.transpose())
                    .component_mul(&(&s * b * s.transpose()))
                    .sum();
        }
        value
    });
    let correction =
        &rotation * ((&local_density + local_density.transpose()) * 0.5) * rotation.transpose();
    for i in 0..n {
        for j in 0..n {
            occupied_density[(correlated[i], correlated[j])] += correction[(i, j)];
        }
    }
    let virtual_density = virtual_density.view((o, o), (v, v)).into_owned();
    let (values, vectors) = real_eigen((&virtual_density + virtual_density.transpose()) * 0.5)?;
    let mut transform = ctx.allocate::<f64>(product(&[m, m])?)?;
    for i in 0..m {
        for j in 0..m {
            transform.as_mut_slice()[i * m + j] = if i < o && j < o {
                f64::from(i == j)
            } else if i >= o && j >= o {
                vectors[(i - o, v - 1 - (j - o))]
            } else {
                0.
            };
        }
    }
    let mut density = ctx.allocate::<f64>(product(&[o, o])?)?;
    for i in 0..o {
        for j in 0..o {
            density.as_mut_slice()[i * o + j] = occupied_density[(i, j)];
        }
    }
    let occupations: Vec<_> = (0..o).map(|i| occupied_density[(i, i)]).collect();
    let values: Vec<_> = values.into_iter().rev().collect();
    Ok(LocalResult {
        mp2: Mp2Result {
            correlation_energy: energy,
            occupied_occupations: ctx.copy_slice(&occupations)?,
            virtual_occupations: ctx.copy_slice(&values)?,
            rotation: transform,
            occupied_density: density,
        },
        pno_counts: pairs.iter().map(|p| p.eps.len()).collect(),
        iterations,
        domains,
        _memory: memory,
    })
}
