//! Molecular orbital symmetry representations and whole-multiplet selection.
use super::{invalid, MeanField, MolecularIntegrals};
use crate::{math::symmetry, product, quantum::C64, Reservation, Result};
use nalgebra::{DMatrix, Vector3};
pub struct OrbitalSymmetry {
    pub sets: Vec<Vec<usize>>,
    pub characters: Vec<Vec<f64>>,
    pub order: usize,
    _memory: Reservation,
}
fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}
pub fn orbitals(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    rotation: &DMatrix<C64>,
) -> Result<OrbitalSymmetry> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let (center, operations) = symmetry::operations(
        integrals.geometry.positions(),
        integrals.geometry.numbers(),
        0.02,
    )?;
    let points = product(&[integrals.geometry.numbers().len(), 400])?;
    let memory = ctx.reserve(product(&[operations.len(), m, m, 16])?)?;
    let _work = ctx.reserve(
        product(&[points, m, 8, 5])?
            .checked_add(product(&[m, m, 16, 12])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut samples = Vec::with_capacity(points);
    let mut seed = 0_u64;
    let mut uniform = || {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    for atom in integrals.geometry.positions() {
        for _ in 0..400 {
            let p = Vector3::from(*atom)
                + Vector3::from_fn(|_, _| {
                    (-2. * uniform().ln()).sqrt()
                        * (2. * std::f64::consts::PI * uniform()).cos()
                        * 1.5
                });
            samples.push(p);
        }
    }
    let phi = DMatrix::from_fn(points, m, |i, j| {
        integrals.basis.orbitals[j].evaluate(samples[i].into())
    });
    let svd = phi.svd(true, true);
    let scale = svd.singular_values.amax();
    let inverse = svd.pseudo_inverse(1e-12 * scale).map_err(invalid)?;
    let coefficients =
        integrals.lowdin.map(|v| C64::new(v, 0.)) * &reference.model_orbitals * rotation;
    let overlap = DMatrix::from_fn(m, m, |i, j| {
        C64::new(integrals.overlap.as_slice()[i * m + j], 0.)
    });
    let mut representations = Vec::new();
    let mut parent: Vec<_> = (0..m).collect();
    for operation in &operations {
        let moved: Vec<_> = samples
            .iter()
            .map(|p| center + operation.transpose() * (p - center))
            .collect();
        let transformed = DMatrix::from_fn(points, m, |i, j| {
            integrals.basis.orbitals[j].evaluate(moved[i].into())
        });
        let ao = &inverse * transformed;
        let representation =
            coefficients.adjoint() * &overlap * ao.map(|v| C64::new(v, 0.)) * &coefficients;
        for p in 0..m {
            for q in p + 1..m {
                if representation[(p, q)]
                    .norm()
                    .max(representation[(q, p)].norm())
                    > 0.3
                {
                    let a = root(&parent, p);
                    let b = root(&parent, q);
                    parent[a] = b;
                }
            }
        }
        representations.push(representation);
    }
    let mut sets: Vec<Vec<usize>> = vec![];
    for p in 0..m {
        if let Some(set) = sets
            .iter_mut()
            .find(|set| root(&parent, set[0]) == root(&parent, p))
        {
            set.push(p);
        } else {
            sets.push(vec![p]);
        }
    }
    let characters = sets
        .iter()
        .map(|set| {
            representations
                .iter()
                .map(|u| set.iter().map(|&p| u[(p, p)].re).sum())
                .collect()
        })
        .collect();
    Ok(OrbitalSymmetry {
        sets,
        characters,
        order: operations.len(),
        _memory: memory,
    })
}
impl OrbitalSymmetry {
    pub fn select(
        &self,
        occupied: usize,
        doubly: usize,
        frozen: &[usize],
        plain: &[usize],
        occupations: &[f64],
        threshold: Option<f64>,
    ) -> Result<Vec<usize>> {
        for set in &self.sets {
            if set.iter().any(|&i| i < doubly) && set.iter().any(|&i| i >= occupied) {
                return Err(invalid("the mean-field reference mixes occupied and virtual orbitals in one symmetry multiplet"));
            }
            let cut = set.iter().filter(|i| frozen.contains(i)).count();
            if cut > 0 && cut < set.iter().filter(|&&i| i < occupied).count() {
                return Err(invalid("the frozen core splits a symmetry multiplet"));
            }
        }
        let cap = plain.iter().filter(|&&i| i >= occupied).count();
        let mut active: Vec<_> = plain.iter().copied().filter(|&i| i < occupied).collect();
        let virtuals: Vec<_> = self
            .sets
            .iter()
            .enumerate()
            .filter(|(_, set)| set.iter().all(|&i| i >= occupied))
            .filter(|(_, set)| threshold.is_none_or(|t| set.iter().any(|&i| occupations[i] >= t)))
            .collect();
        let reaches = |a: usize| {
            self.sets
                .iter()
                .enumerate()
                .filter(|(_, set)| set.iter().any(|&i| i < occupied && !frozen.contains(&i)))
                .any(|(i, _)| {
                    self.characters[i]
                        .iter()
                        .zip(&self.characters[a])
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
                        / self.order as f64
                        > 0.5
                })
        };
        let mut chosen = Vec::new();
        for (index, set) in &virtuals {
            if chosen.len() + set.len() <= cap {
                chosen.extend_from_slice(set);
                let _ = index;
            }
        }
        if cap > 0
            && !active.is_empty()
            && !virtuals
                .iter()
                .any(|(index, set)| set.iter().all(|i| chosen.contains(i)) && reaches(*index))
        {
            let (_,required)=virtuals.iter().find(|(index,set)|set.len()<=cap && reaches(*index)).ok_or_else(||invalid("no complete virtual symmetry multiplet fits the requested ground-state active space"))?;
            chosen = required.to_vec();
            for (_, set) in &virtuals {
                if !set.iter().any(|i| chosen.contains(i)) && chosen.len() + set.len() <= cap {
                    chosen.extend_from_slice(set);
                }
            }
        }
        active.extend(chosen);
        active.sort_unstable();
        Ok(active)
    }
}
