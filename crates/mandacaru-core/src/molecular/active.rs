//! Integral-direct orbital selection and frozen-natural-orbital Hamiltonians.
use super::{correlation, invalid, scf, transform, MeanField, MolecularIntegrals};
use crate::{math::linalg::real_eigen, product, quantum::C64, Result};
use nalgebra::DMatrix;
use serde_json::Value;
use std::sync::Arc;

fn indices(value: &Value, m: usize) -> Result<Vec<usize>> {
    let mut out = if let Some(n) = value.as_u64() {
        if n > m as u64 {
            return Err(invalid("orbital count exceeds the basis"));
        }
        (0..n as usize).collect()
    } else {
        value
            .as_array()
            .ok_or_else(|| invalid("orbitals must be counts or index lists"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&i| i < m)
                    .ok_or_else(|| invalid("invalid orbital index"))
            })
            .collect::<Result<Vec<_>>>()?
    };
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

impl MeanField {
    /// Build only the selected four-index tensor. MP2 first requests an oovv
    /// block from Coulomb factors, never a full source-MO tensor.
    pub fn active_model(
        &self,
        integrals: &MolecularIntegrals,
        specification: &str,
    ) -> Result<transform::Model> {
        let spec: Value =
            serde_json::from_str(specification).map_err(|e| invalid(e.to_string()))?;
        let object = spec
            .as_object()
            .ok_or_else(|| invalid("active_space must be an object"))?;
        if object.keys().any(|k| {
            ![
                "method",
                "orbitals",
                "threshold",
                "frozen",
                "symmetry",
                "correlating_pairs",
            ]
            .contains(&k.as_str())
        }) {
            return Err(invalid("unknown active_space option"));
        }
        for key in ["correlating_pairs"] {
            if !spec[key].is_null() && spec[key] != false {
                return Err(invalid(format!(
                    "native active-space {key} expansion is not yet implemented"
                )));
            }
        }
        let symmetry = spec["symmetry"].as_bool().unwrap_or(false);
        if !spec["symmetry"].is_null() && spec["symmetry"].as_bool().is_none() {
            return Err(invalid("active-space symmetry must be a boolean"));
        }
        if symmetry && self.particles[0] != self.particles[1] {
            return Err(invalid(
                "native symmetry-constrained selection currently requires a closed-shell reference",
            ));
        }
        let method = if spec["method"].is_null() {
            "energy"
        } else {
            spec["method"]
                .as_str()
                .ok_or_else(|| invalid("active-space method must be a string"))?
        };
        if !["energy", "mp2", "dlpno-mp2"].contains(&method) {
            return Err(invalid(format!(
                "native active-space method {method:?} is not yet implemented"
            )));
        }
        if self.method == "ghf" {
            return Err(invalid("spatial active spaces do not apply to GHF spinors"));
        }
        if !self.converged {
            return Err(invalid(
                "active spaces require a converged mean-field state",
            ));
        }
        if method != "energy" && (self.method != "rhf" || self.functional.is_some()) {
            return Err(invalid(
                "native MP2 orbital selection requires a closed-shell Hartree-Fock reference",
            ));
        }
        let m = integrals.m();
        let occupied = self.particles[0].max(self.particles[1]);
        let doubly = self.particles[0].min(self.particles[1]);
        let mut frozen = match &spec["frozen"] {
            Value::Null => vec![],
            Value::String(s) if s == "auto" => {
                let n = if integrals.corrections.is_some() {
                    0
                } else {
                    integrals
                        .geometry
                        .numbers()
                        .iter()
                        .map(|z| match z {
                            0..=2 => 0,
                            3..=10 => 1,
                            11..=18 => 5,
                            19..=36 => 9,
                            37..=54 => 18,
                            55..=86 => 27,
                            _ => 43,
                        })
                        .sum()
                };
                (0..n).collect()
            }
            value => indices(value, m)?,
        };
        if frozen.iter().any(|&i| i >= doubly) {
            return Err(invalid("frozen orbitals must be doubly occupied"));
        }
        let threshold = match &spec["threshold"] {
            Value::Null | Value::Bool(false) => None,
            Value::Bool(true) => Some(1e-3),
            v => Some(
                v.as_f64()
                    .filter(|v| v.is_finite() && *v > 0. && *v < 2.)
                    .ok_or_else(|| {
                        invalid("occupation threshold must be positive and below two")
                    })?,
            ),
        };
        if threshold.is_some() && method == "energy" {
            return Err(invalid("an occupation threshold requires MP2 ranking"));
        }
        if method != "energy" && spec["orbitals"].is_array() {
            return Err(invalid("explicit orbital indices require energy ordering"));
        }
        let ctx = &integrals.context;
        let _workspace = ctx.reserve(product(&[m, m, 16, 16])?)?;
        let eri = integrals.eri()?;
        let mut rotation = DMatrix::<C64>::identity(m, m);
        let mut occupations = vec![];
        let mut correlation_energy = None;
        let mut occupied_density = None;
        if method != "energy" {
            if occupied == 0 || occupied == m {
                return Err(invalid("MP2 selection needs occupied and virtual orbitals"));
            }
            let c = &self.model_orbitals;
            if c.iter().any(|v| v.im.abs() > 1e-10) {
                return Err(invalid("MP2 selection requires real reference orbitals"));
            }
            let mp2 = if method == "dlpno-mp2" {
                super::local_correlation::calculate(integrals, self, &frozen, &Default::default())?
                    .mp2
            } else {
                let co = c.columns(0, occupied).into_owned();
                let cv = c.columns(occupied, m - occupied).into_owned();
                let block = transform::four(ctx, &eri, [&co, &co, &cv, &cv])?;
                if block.as_slice().iter().any(|v| v.im.abs() > 1e-10) {
                    return Err(invalid("MP2 selection requires real integrals"));
                }
                let mut real = ctx.allocate::<f64>(block.as_slice().len())?;
                for (out, value) in real.as_mut_slice().iter_mut().zip(block.as_slice()) {
                    *out = value.re;
                }
                let canonical: Vec<_> = (0..m * m)
                    .map(|i| if i / m == i % m { 1. } else { 0. })
                    .collect();
                correlation::from_block(
                    ctx,
                    real.as_slice(),
                    &self.energies[0],
                    occupied,
                    &canonical,
                )?
            };
            rotation = DMatrix::from_fn(m, m, |i, j| {
                C64::new(mp2.rotation.as_slice()[i * m + j], 0.)
            });
            occupations.extend_from_slice(mp2.occupied_occupations.as_slice());
            occupations.extend_from_slice(mp2.virtual_occupations.as_slice());
            correlation_energy = Some(mp2.correlation_energy);
            occupied_density = Some(mp2.occupied_density);
            if let Some(t) = threshold {
                for (i, &n) in occupations[..doubly].iter().enumerate() {
                    if 2. - n < t && !frozen.contains(&i) {
                        frozen.push(i);
                    }
                }
                frozen.sort_unstable();
            }
        }
        let candidates: Vec<_> = (0..m).filter(|i| !frozen.contains(i)).collect();
        let mut active = if spec["orbitals"].is_null() {
            candidates
        } else if spec["orbitals"].is_array() {
            indices(&spec["orbitals"], m)?
        } else if let Some(n) = spec["orbitals"].as_u64() {
            if n == 0 || n > candidates.len() as u64 {
                return Err(invalid(
                    "active count exceeds remaining orbitals or is zero",
                ));
            }
            candidates[..n as usize].to_vec()
        } else if let Some(counts) = spec["orbitals"].as_object() {
            if counts
                .keys()
                .any(|k| !["occupied", "virtual"].contains(&k.as_str()))
            {
                return Err(invalid("orbital counts accept occupied and virtual"));
            }
            let available = doubly - frozen.len();
            let get = |key: &str, default: usize| -> Result<usize> {
                counts.get(key).map_or(Ok(default), |v| {
                    v.as_u64()
                        .and_then(|i| usize::try_from(i).ok())
                        .filter(|&i| i <= default)
                        .ok_or_else(|| invalid("active count exceeds available orbitals"))
                })
            };
            let keep = get("occupied", available)?;
            let virtuals = get("virtual", m - occupied)?;
            let extra: Vec<_> = (0..doubly)
                .filter(|i| !frozen.contains(i))
                .take(available - keep)
                .collect();
            frozen.extend(extra);
            frozen.sort_unstable();
            (0..occupied + virtuals)
                .filter(|i| !frozen.contains(i))
                .collect()
        } else {
            return Err(invalid("invalid active orbital selection"));
        };
        if let Some(t) = threshold.filter(|_| !symmetry) {
            active.retain(|&i| i < occupied || occupations[i] >= t);
        }
        if active.is_empty()
            || active.iter().any(|i| frozen.contains(i))
            || (0..occupied).any(|i| !frozen.contains(&i) && !active.contains(&i))
        {
            return Err(invalid(
                "occupied orbitals must be frozen or active, and the active space must be nonempty",
            ));
        }
        if let Some(density) = occupied_density {
            let keep: Vec<_> = (0..doubly).filter(|i| !frozen.contains(i)).collect();
            if !keep.is_empty() {
                let d = DMatrix::from_fn(keep.len(), keep.len(), |i, j| {
                    density.as_slice()[keep[i] * doubly + keep[j]]
                });
                let (values, vectors) = real_eigen(d)?;
                for (j, &target) in keep.iter().enumerate() {
                    occupations[target] = values[keep.len() - 1 - j];
                    for (i, &source) in keep.iter().enumerate() {
                        rotation[(source, target)] = C64::new(vectors[(i, keep.len() - 1 - j)], 0.);
                    }
                }
            }
        }
        let mut symmetry_sets = Vec::new();
        if symmetry {
            let symmetry = super::symmetry::orbitals(integrals, self, &rotation)?;
            active =
                symmetry.select(occupied, doubly, &frozen, &active, &occupations, threshold)?;
            symmetry_sets = symmetry.sets.clone();
            if active.is_empty() {
                return Err(invalid(
                    "symmetry and occupation thresholds leave an empty active space",
                ));
            }
        }
        let selected = DMatrix::from_fn(m, active.len(), |i, j| rotation[(i, active[j])]);
        let c = &self.model_orbitals * selected;
        let h = DMatrix::from_fn(m, m, |i, j| {
            C64::new(integrals.one.as_slice()[i * m + j], 0.)
        });
        let mut effective = h.clone();
        let mut constant = integrals.nuclear;
        if !frozen.is_empty() {
            let core =
                DMatrix::from_fn(m, frozen.len(), |i, j| self.model_orbitals[(i, frozen[j])]);
            let density = &core * core.adjoint();
            let (j, k) = scf::jk(&eri, m, &density);
            let potential = j * C64::new(2., 0.) - k;
            constant += (density.clone() * (h * C64::new(2., 0.) + &potential))
                .trace()
                .re;
            effective += potential;
        }
        let n = active.len();
        let h = c.adjoint() * effective * &c;
        let mut one = ctx.allocate::<C64>(product(&[n, n])?)?;
        for i in 0..n {
            for j in 0..n {
                one.as_mut_slice()[i * n + j] = h[(i, j)];
            }
        }
        let two = transform::four(ctx, &eri, [&c; 4])?;
        let mut model = transform::Model::new(
            ctx,
            n,
            [
                self.particles[0] - frozen.len(),
                self.particles[1] - frozen.len(),
            ],
            one,
            two,
            constant,
            false,
        )?;
        model.orbital_indices = active;
        model.frozen_indices = frozen;
        model.orbital_symmetry_sets = symmetry_sets;
        if let Some(energy) = correlation_energy {
            model.orbital_rotation = Some(Arc::new(transform::OrbitalRotation {
                matrix: rotation,
                occupations,
                correlation_energy: energy,
                _memory: ctx.reserve(
                    product(&[m, m, 16])?
                        .checked_add(product(&[m, 8])?)
                        .ok_or(crate::Error::Overflow)?,
                )?,
            }));
        }
        Ok(model)
    }
}
