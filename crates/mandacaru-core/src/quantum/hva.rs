//! Hamiltonian variational groups with shared-angle product-formula stages.
use super::{algebra::Generator, CachedProblem, Operator, C64};
use crate::{product, Error, Reservation, Result};
pub struct HVA {
    stages: Vec<Vec<(Operator, f64)>>,
    pub seed_angle: f64,
    pub trotter: bool,
    _memory: Reservation,
}
pub struct Expanded<'a> {
    pub operators: Vec<&'a Operator>,
    pub parameters: Vec<f64>,
    pub links: Vec<(usize, f64)>,
}
impl HVA {
    pub fn expand<'a>(
        &'a self,
        pool: &'a [Generator],
        indices: &[usize],
        parameters: &[f64],
    ) -> Result<Expanded<'a>> {
        if indices.len() != parameters.len() || indices.iter().any(|&i| i >= pool.len()) {
            return Err(Error::Invalid(
                "HVA logical parameters do not match the group sequence".into(),
            ));
        }
        let mut out = Expanded {
            operators: vec![],
            parameters: vec![],
            links: vec![],
        };
        for (j, (&i, &t)) in indices.iter().zip(parameters).enumerate() {
            if !t.is_finite() {
                return Err(Error::Invalid("HVA parameters must be finite".into()));
            }
            if self.trotter {
                for (op, weight) in &self.stages[i] {
                    out.operators.push(op);
                    out.parameters.push(t * weight);
                    out.links.push((j, *weight));
                }
            } else {
                out.operators.push(&pool[i].operator);
                out.parameters.push(t);
                out.links.push((j, 1.));
            }
        }
        Ok(out)
    }
}
impl CachedProblem {
    pub fn use_hva(
        &mut self,
        layers: usize,
        mode: &str,
        order: usize,
        steps: usize,
        seed_angle: f64,
    ) -> Result<()> {
        if layers == 0
            || layers > 1000
            || !["exact", "trotter"].contains(&mode)
            || ![1, 2].contains(&order)
            || steps == 0
            || !seed_angle.is_finite()
        {
            return Err(Error::Invalid("HVA needs positive layers/steps, exact or trotter evolution, order 1 or 2 and a finite seed angle".into()));
        }
        if self.hamiltonian.decomposition.is_empty() {
            return Err(Error::Invalid("HVA requires the physical one-/two-body decomposition of a native molecular Hamiltonian".into()));
        }
        let count = product(&[layers, self.hamiltonian.decomposition.len()])?;
        let stage_count = product(&[
            layers,
            steps,
            order,
            self.hamiltonian
                .decomposition
                .iter()
                .map(|g| g.terms.len())
                .sum::<usize>(),
        ])?;
        if stage_count > 1_000_000 {
            return Err(Error::Invalid(
                "HVA product-formula schedule exceeds one million stages".into(),
            ));
        }
        let memory = self.context.reserve(product(&[stage_count, 128])?)?;
        let pool_memory = self.context.reserve(product(&[count, 192])?)?;
        let mut pool = Vec::new();
        let mut stages = Vec::new();
        for layer in 0..layers {
            for (index, group) in self.hamiltonian.decomposition.iter().enumerate() {
                if !group.is_hermitian() {
                    return Err(Error::Invalid("HVA groups must be Hermitian".into()));
                }
                let operator = Operator::from_terms(
                    &self.context,
                    group.n,
                    group
                        .terms
                        .iter()
                        .map(|t| super::algebra::Term {
                            coefficient: -C64::new(0., 1.) * t.coefficient,
                            ..*t
                        })
                        .collect(),
                )?;
                self.basis.check_invariant(&operator)?;
                pool.push(Generator {
                    label: format!("HVA({layer},{index})"),
                    operator,
                });
                let mut sequence = Vec::new();
                if mode == "trotter" {
                    for _ in 0..steps {
                        let weight = 1. / (steps * order) as f64;
                        let mut terms: Vec<_> = group.terms.iter().collect();
                        if order == 2 {
                            terms.extend(group.terms.iter().rev());
                        }
                        for term in terms {
                            let operator = Operator::from_terms(
                                &self.context,
                                group.n,
                                vec![super::algebra::Term {
                                    coefficient: -C64::new(0., 1.) * term.coefficient,
                                    ..*term
                                }],
                            )?;
                            self.basis.check_invariant(&operator)?;
                            sequence.push((operator, weight));
                        }
                    }
                }
                stages.push(sequence);
            }
        }
        self.pool = pool;
        self.pool_screenable = vec![true; count];
        self.growth_members = vec![vec![]; count];
        self._pool_metadata = pool_memory;
        self.hva = Some(HVA {
            stages,
            seed_angle,
            trotter: mode == "trotter",
            _memory: memory,
        });
        Ok(())
    }
}
