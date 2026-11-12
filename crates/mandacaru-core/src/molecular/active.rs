//! Integral-direct orbital selection and frozen-natural-orbital Hamiltonians.
use super::double_factorized::{DoubleFactorized, Representation};
use super::{correlation, frame, invalid, scf, transform, MeanField, MolecularIntegrals};
use crate::{math::linalg::real_eigen, product, quantum::C64, Result};
use nalgebra::DMatrix;
use serde_json::Value;
use std::sync::Arc;

/// The active-space selection methods, in one place (the bibliography sweep reads them).
pub const METHODS: &[&str] = &["energy", "mp2", "dlpno-mp2"];

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
    /// Put the clusters of the natural-orbital `rotation` (columns over the model orbitals)
    /// that lie inside one `domain` of orbital indices into the deterministic frame of
    /// [`frame::fix_cluster_gauge`]; their labels are the natural `occupations`. The
    /// moments are those of the orbitals `model_orbitals * rotation`.
    fn fix_rotation_frame(
        &self,
        integrals: &MolecularIntegrals,
        rotation: &mut DMatrix<C64>,
        occupations: &[f64],
        domain: Vec<usize>,
    ) -> Result<()> {
        let basis = self.model_orbitals.map(|v| v.re);
        let mut lab = frame::lab_moments(integrals);
        let mut in_model_frame =
            |block: &DMatrix<f64>, operators: bool| lab(&(&basis * block), operators);
        frame::fix_cluster_gauge(rotation, occupations, &[domain], &mut in_model_frame)?;
        Ok(())
    }
    /// Build only the selected four-index tensor. MP2 first requests an oovv
    /// block from Coulomb factors, never a full source-MO tensor.
    pub fn active_model(
        &self,
        integrals: &MolecularIntegrals,
        specification: &str,
    ) -> Result<transform::Model> {
        self.active_model_with(integrals, specification, Representation::Dense)
    }
    /// [`Self::active_model`] with its two-body part in `representation`; the double-factorized
    /// one rotates the two-electron factors into the active orbitals and never forms the
    /// tensor.
    pub fn active_model_with(
        &self,
        integrals: &MolecularIntegrals,
        specification: &str,
        representation: Representation,
    ) -> Result<transform::Model> {
        self.require_determinant("an active-space selection")?;
        self.require_frame()?;
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
        let pairs = match &spec["correlating_pairs"] {
            Value::Null => false,
            Value::Bool(b) => *b,
            _ => return Err(invalid("active-space correlating_pairs must be a boolean")),
        };
        let symmetry = spec["symmetry"].as_bool().unwrap_or(false);
        if pairs && symmetry {
            return Err(crate::Error::Unsupported(
                "active-space correlating_pairs together with symmetry is not implemented; use one \
                 of them"
                    .into(),
            ));
        }
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
        if !METHODS.contains(&method) {
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
        let open_shell = self.method == "uhf";
        if method != "energy" && (!["rhf", "uhf"].contains(&self.method.as_str()) || self.functional.is_some()) {
            return Err(invalid(
                "native MP2 orbital selection requires a Hartree-Fock reference (rhf or uhf)",
            ));
        }
        if open_shell && method == "dlpno-mp2" {
            return Err(invalid(
                "the dlpno-mp2 active-space method is closed-shell only; an open-shell reference \
                 takes method 'mp2'",
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
        // The requested core: never paired (it is not a bonding orbital), unlike an occupied
        // orbital a count or a threshold freezes below.
        let core = frozen.clone();
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
            let mp2 = if open_shell {
                correlation::open_shell_natural_orbitals(
                    integrals,
                    c,
                    self.particles[0],
                    self.particles[1],
                )?
            } else if method == "dlpno-mp2" {
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
            // The virtual natural orbitals are arbitrary inside a degenerate occupation: fix
            // their frame before anything (the threshold, the partners) reads them.
            if self.frame_covered(integrals) {
                self.fix_rotation_frame(
                    integrals,
                    &mut rotation,
                    &occupations,
                    (occupied..m).collect(),
                )?;
            }
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
        if let Some(setup) = &integrals.paw_setup {
            crate::basis::diffuse::refuse_energy_ranking(
                method,
                setup.diffuse_functions(),
                m - occupied,
                active.iter().filter(|&&i| i >= occupied).count(),
            )?;
        }
        if pairs {
            let paired: Vec<usize> = (0..doubly).filter(|i| !core.contains(i)).collect();
            let partners = correlating_partners(
                integrals,
                &(&self.model_orbitals * &rotation),
                &self.particles,
                &paired,
                &(occupied..m).collect::<Vec<_>>(),
            )?;
            let required: Vec<usize> = partners
                .iter()
                .filter(|(i, _)| !frozen.contains(i))
                .map(|&(_, a)| a)
                .collect();
            if spec["orbitals"].is_array() {
                // An explicit list is the user's choice: checked, not changed.
                let missing: Vec<_> = partners
                    .iter()
                    .filter(|(i, a)| !frozen.contains(i) && !active.contains(a))
                    .collect();
                if !missing.is_empty() {
                    return Err(invalid(format!(
                        "the active_space 'orbitals' list leaves out the correlating partner of \
                         an active occupied orbital (occupied, virtual): {missing:?}; add those \
                         virtual orbitals or drop 'correlating_pairs'"
                    )));
                }
            } else if !spec["orbitals"].is_null() || threshold.is_some() {
                let barred: Vec<usize> = partners
                    .iter()
                    .filter(|(i, _)| frozen.contains(i))
                    .map(|&(_, a)| a)
                    .collect();
                // The register keeps the virtual count it asked for (no cap for a threshold
                // alone): the required partners first, then the rest in rank order.
                let current: Vec<usize> = active.iter().copied().filter(|&i| i >= occupied).collect();
                let cap = (!spec["orbitals"].is_null()).then_some(current.len());
                if cap.is_some_and(|c| required.len() > c) {
                    return Err(invalid(format!(
                        "the active space asks for {} virtual orbitals and its active occupied \
                         orbitals need {} correlating partners; enlarge it or drop \
                         'correlating_pairs'",
                        current.len(),
                        required.len()
                    )));
                }
                let mut chosen = required.clone();
                for a in occupied..m {
                    if cap.is_some_and(|c| chosen.len() >= c) {
                        break;
                    }
                    if chosen.contains(&a)
                        || barred.contains(&a)
                        || threshold.is_some_and(|t| occupations[a] < t)
                    {
                        continue;
                    }
                    chosen.push(a);
                }
                active.retain(|&i| i < occupied);
                active.extend(chosen);
                active.sort_unstable();
            }
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
                // The occupied block of the MP2 density, `occupied` rows (equal to `doubly` for
                // a closed shell); only the active doubly occupied orbitals mix, which keeps
                // the reference determinant.
                let d = DMatrix::from_fn(keep.len(), keep.len(), |i, j| {
                    density.as_slice()[keep[i] * occupied + keep[j]]
                });
                let (values, vectors) = real_eigen(d)?;
                for (j, &target) in keep.iter().enumerate() {
                    occupations[target] = values[keep.len() - 1 - j];
                    for (i, &source) in keep.iter().enumerate() {
                        rotation[(source, target)] = C64::new(vectors[(i, keep.len() - 1 - j)], 0.);
                    }
                }
                // Likewise the occupied natural orbitals of a degenerate density block.
                if self.frame_covered(integrals) {
                    self.fix_rotation_frame(integrals, &mut rotation, &occupations, keep.clone())?;
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
        let particles = [
            self.particles[0] - frozen.len(),
            self.particles[1] - frozen.len(),
        ];
        let mut model = match representation {
            Representation::Dense => {
                let two = transform::four(ctx, &eri, [&c; 4])?;
                transform::Model::new(ctx, n, particles, one, two, constant, false)?
            }
            Representation::DoubleFactorized { truncation } => {
                let factorized = DoubleFactorized::from_eri(ctx, &eri, &c, truncation)?;
                transform::Model::double_factorized(ctx, particles, one, factorized, constant)?
            }
        };
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

/// The correlating partner of each occupied orbital in `occupied`: the one-to-one assignment
/// of maximum total weight `|<ii|aa>| / (F_aa - F_ii)` over the `virtuals` (an exchange-type
/// integral, large when `i` and `a` share a region of space with a node between them, over the
/// energy the pair excitation `i^2 -> a^2` costs). `orbitals` are the selection's orbitals
/// (columns over the orthonormal basis); `F` is the Fock operator of the reference occupations
/// (2, 1 or 0 per orbital, the open shell's singly occupied orbitals 1).
pub fn correlating_partners(
    integrals: &MolecularIntegrals,
    orbitals: &DMatrix<C64>,
    particles: &[usize; 2],
    occupied: &[usize],
    virtuals: &[usize],
) -> Result<Vec<(usize, usize)>> {
    if occupied.is_empty() || virtuals.is_empty() {
        return Ok(vec![]);
    }
    let ctx = &integrals.context;
    let m = integrals.m();
    let eri = integrals.eri()?;
    let (high, low) = (particles[0].max(particles[1]), particles[0].min(particles[1]));
    let mut density = DMatrix::<C64>::zeros(m, m);
    for p in 0..high {
        let weight = if p < low { 2. } else { 1. };
        let c = orbitals.column(p);
        density += c * c.adjoint() * C64::new(weight, 0.);
    }
    let (j, k) = scf::jk(&eri, m, &density);
    let h = DMatrix::from_fn(m, m, |i, j| C64::new(integrals.one.as_slice()[i * m + j], 0.));
    let fock = h + j - k * C64::new(0.5, 0.);
    let diagonal = |p: usize| (orbitals.column(p).adjoint() * &fock * orbitals.column(p))[(0, 0)].re;
    let co = DMatrix::from_fn(m, occupied.len(), |r, c| orbitals[(r, occupied[c])]);
    let cv = DMatrix::from_fn(m, virtuals.len(), |r, c| orbitals[(r, virtuals[c])]);
    let (no, nv) = (occupied.len(), virtuals.len());
    let block = transform::four(ctx, &eri, [&co, &co, &cv, &cv])?;
    let size = no.max(nv);
    let mut cost = DMatrix::<f64>::zeros(size, size);
    for (r, &i) in occupied.iter().enumerate() {
        for (c, &a) in virtuals.iter().enumerate() {
            let exchange = block.as_slice()[((r * no + r) * nv + c) * nv + c].norm();
            let gap = (diagonal(a) - diagonal(i)).max(1e-6);
            cost[(r, c)] = -exchange / gap;
        }
    }
    let assigned = crate::math::assignment::minimum(ctx, &cost)?;
    let mut pairs: Vec<(usize, usize)> = (0..no)
        .filter(|&r| assigned[r] < nv)
        .map(|r| (occupied[r], virtuals[assigned[r]]))
        .collect();
    pairs.sort_unstable();
    Ok(pairs)
}
