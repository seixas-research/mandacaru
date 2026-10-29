use super::{integrals::Eri, invalid};
use crate::quantum::{algebra::Term, CachedProblem, Encoding, Operator, C64};
use crate::{product, Buffer, Context, Reservation, Result};
use nalgebra::DMatrix;

pub fn four(ctx: &Context, eri: &Eri, coefficients: [&DMatrix<C64>; 4]) -> Result<Buffer<C64>> {
    let m = eri.m();
    let sizes = coefficients.map(DMatrix::ncols);
    let [np, nq, nr, ns] = sizes;
    let n = *sizes.iter().max().unwrap();
    if coefficients.iter().any(|c| c.nrows() != m) {
        return Err(invalid("MO transformation dimensions disagree"));
    }
    if let Eri::Augmented {
        base, correction, ..
    } = eri
    {
        let mut out = four(ctx, base, coefficients)?;
        let _work = ctx.reserve(product(&[n.max(m), n.max(m), 16, 8])?)?;
        for (factor, &weight) in correction.factors.iter().zip(&correction.weights) {
            let b = factor.map(|v| C64::new(v, 0.));
            let left = coefficients[0].adjoint() * &b * coefficients[2];
            let right = coefficients[1].adjoint() * b * coefficients[3];
            for p in 0..np {
                for q in 0..nq {
                    for r in 0..nr {
                        for s in 0..ns {
                            out.as_mut_slice()[((p * nq + q) * nr + r) * ns + s] +=
                                weight * left[(p, r)] * right[(q, s)];
                        }
                    }
                }
            }
        }
        return Ok(out);
    }
    if let Eri::Cholesky { rank, factors, .. } = eri {
        let _work = ctx.reserve(product(&[n.max(m), n.max(m), 16, 8])?)?;
        let mut out = ctx.allocate::<C64>(product(&sizes)?)?;
        for column in 0..*rank {
            let b = DMatrix::from_fn(m, m, |p, q| {
                C64::new(
                    factors.as_slice()[super::integrals::pair(p, q) * rank + column],
                    0.,
                )
            });
            let left = coefficients[0].adjoint() * &b * coefficients[2];
            let right = coefficients[1].adjoint() * b * coefficients[3];
            for p in 0..np {
                for q in 0..nq {
                    for r in 0..nr {
                        for s in 0..ns {
                            out.as_mut_slice()[((p * nq + q) * nr + r) * ns + s] +=
                                left[(p, r)] * right[(q, s)];
                        }
                    }
                }
            }
        }
        return Ok(out);
    }
    let mut current = ctx.allocate::<C64>(product(&[np, m, m, m])?)?;
    for p in 0..np {
        for b in 0..m {
            for c in 0..m {
                for d in 0..m {
                    current.as_mut_slice()[((p * m + b) * m + c) * m + d] = (0..m)
                        .map(|a| coefficients[0][(a, p)].conj() * eri.value(a, b, c, d))
                        .sum();
                }
            }
        }
    }
    let shapes = [[np, m, m, m], [np, nq, m, m], [np, nq, nr, m], sizes];
    for axis in 1..4 {
        let old = shapes[axis - 1];
        let new = shapes[axis];
        let mut next = ctx.allocate::<C64>(product(&new)?)?;
        for flat in 0..next.as_slice().len() {
            let mut index = [0; 4];
            let mut cursor = flat;
            for a in (0..4).rev() {
                index[a] = cursor % new[a];
                cursor /= new[a];
            }
            let target = index[axis];
            let mut value = C64::default();
            for source in 0..m {
                index[axis] = source;
                let old_flat =
                    ((index[0] * old[1] + index[1]) * old[2] + index[2]) * old[3] + index[3];
                let c = coefficients[axis][(source, target)];
                value += current.as_slice()[old_flat] * if axis < 2 { c.conj() } else { c };
            }
            next.as_mut_slice()[flat] = value;
        }
        current = next;
    }
    Ok(current)
}
pub struct Model {
    pub m: usize,
    pub particles: [usize; 2],
    pub one: Buffer<C64>,
    pub two: Buffer<C64>,
    pub constant: f64,
    pub spinor: bool,
    pub orbital_indices: Vec<usize>,
    pub frozen_indices: Vec<usize>,
    /// Full source-MO rotation, shared by reduced models and density views.
    pub orbital_rotation: Option<std::sync::Arc<OrbitalRotation>>,
    pub orbital_symmetry_sets: Vec<Vec<usize>>,
    _memory: Reservation,
}
pub struct OrbitalRotation {
    pub matrix: DMatrix<C64>,
    pub occupations: Vec<f64>,
    pub correlation_energy: f64,
    pub(crate) _memory: Reservation,
}
impl Model {
    pub fn save(&self, ctx: &Context, path: &std::path::Path, mapping: &str) -> Result<()> {
        let operator = self.operator(ctx, mapping)?;
        let terms: Vec<_> = operator
            .labels()
            .into_iter()
            .map(|(s, c)| serde_json::json!([s, c.re, c.im]))
            .collect();
        let groups:Vec<_>=operator.decomposition.iter().map(|g|serde_json::json!({"num_qubits":g.n,"terms":g.labels().into_iter().map(|(s,c)|serde_json::json!([s,c.re,c.im])).collect::<Vec<_>>()})).collect();
        let value = serde_json::json!({"format":"mandacaru-qubit-hamiltonian","version":1,"num_qubits":operator.n,"mapping":mapping,"num_particles":self.particles,"n_spatial_orbitals":self.m,"metadata":{"energy_unit":"Ha","source_engine":"rust","groups":groups},"terms":terms});
        crate::persistence::Document::parse(
            ctx,
            &value.to_string(),
            crate::persistence::HAMILTONIAN,
        )?
        .save(path)
    }
    pub fn new(
        ctx: &Context,
        m: usize,
        particles: [usize; 2],
        one: Buffer<C64>,
        two: Buffer<C64>,
        constant: f64,
        spinor: bool,
    ) -> Result<Self> {
        let n = if spinor { 2 * m } else { m };
        if one.as_slice().len() != product(&[n, n])?
            || two.as_slice().len() != product(&[n, n, n, n])?
            || particles.iter().any(|p| *p > m)
            || !constant.is_finite()
        {
            return Err(invalid("invalid molecular model dimensions or occupation"));
        }
        Ok(Self {
            m,
            particles,
            one,
            two,
            constant,
            spinor,
            orbital_indices: (0..n).collect(),
            frozen_indices: vec![],
            orbital_rotation: None,
            orbital_symmetry_sets: Vec::new(),
            _memory: ctx.reserve(product(&[n, n, 64])?)?,
        })
    }
    /// Frozen doubly occupied spatial orbitals and an explicit active subspace.
    pub fn reduce(&self, ctx: &Context, frozen: &[usize], active: &[usize]) -> Result<Self> {
        if active.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("active orbital indices must be strictly ascending so reference occupations keep their order"));
        }
        if self.spinor {
            return Err(invalid(
                "spatial frozen-core reduction does not apply to GHF spinors",
            ));
        }
        let mut seen = vec![false; self.m];
        for &i in frozen.iter().chain(active) {
            if i >= self.m || seen[i] {
                return Err(invalid(
                    "frozen/active orbitals must be distinct valid indices",
                ));
            }
            seen[i] = true;
        }
        if frozen
            .iter()
            .any(|&i| i >= self.particles[0].min(self.particles[1]))
            || active.is_empty()
            || frozen.len() > self.particles[0].min(self.particles[1])
        {
            return Err(invalid(
                "frozen core requires doubly occupied orbitals and a nonempty active space",
            ));
        }
        if (0..self.particles[0].max(self.particles[1])).any(|i| !seen[i]) {
            return Err(invalid("occupied orbitals must be frozen or active"));
        }
        let m = self.m;
        let n = active.len();
        let h = |p, q| self.one.as_slice()[p * m + q];
        let g = |p, q, r, s| self.two.as_slice()[((p * m + q) * m + r) * m + s];
        let mut constant = self.constant;
        for &i in frozen {
            constant += 2. * h(i, i).re;
            for &j in frozen {
                constant += (2. * g(i, j, i, j) - g(i, j, j, i)).re;
            }
        }
        let mut one = ctx.allocate(product(&[n, n])?)?;
        let mut two = ctx.allocate(product(&[n, n, n, n])?)?;
        for (a, &p) in active.iter().enumerate() {
            for (b, &q) in active.iter().enumerate() {
                one.as_mut_slice()[a * n + b] = h(p, q)
                    + frozen
                        .iter()
                        .map(|&i| 2. * g(p, i, q, i) - g(p, i, i, q))
                        .sum::<C64>();
                for (c, &r) in active.iter().enumerate() {
                    for (d, &s) in active.iter().enumerate() {
                        two.as_mut_slice()[((a * n + b) * n + c) * n + d] = g(p, q, r, s);
                    }
                }
            }
        }
        let mut result = Self::new(
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
        result.orbital_indices = active.iter().map(|&i| self.orbital_indices[i]).collect();
        result.frozen_indices = self
            .frozen_indices
            .iter()
            .copied()
            .chain(frozen.iter().map(|&i| self.orbital_indices[i]))
            .collect();
        result.orbital_rotation = self.orbital_rotation.clone();
        result.orbital_symmetry_sets = self.orbital_symmetry_sets.clone();
        Ok(result)
    }
    pub fn fermion_terms(&self) -> Vec<(Vec<(usize, bool)>, C64)> {
        let mut out = vec![(vec![], C64::new(self.constant, 0.))];
        let n = if self.spinor { 2 * self.m } else { self.m };
        for p in 0..n {
            for q in 0..n {
                let h = self.one.as_slice()[p * n + q];
                if h.norm() > 1e-12 {
                    for spin in 0..if self.spinor { 1 } else { 2 } {
                        out.push((vec![(p + spin * n, true), (q + spin * n, false)], h));
                    }
                }
                for r in 0..n {
                    for s in 0..n {
                        let g = self.two.as_slice()[((p * n + q) * n + r) * n + s];
                        if g.norm() > 1e-12 {
                            for alpha in 0..if self.spinor { 1 } else { 2 } {
                                for beta in 0..if self.spinor { 1 } else { 2 } {
                                    out.push((
                                        vec![
                                            (p + alpha * n, true),
                                            (q + beta * n, true),
                                            (s + beta * n, false),
                                            (r + alpha * n, false),
                                        ],
                                        0.5 * g,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
    pub fn operator(&self, ctx: &Context, mapping: &str) -> Result<Operator> {
        if self.spinor && mapping == "parity_reduced" {
            return Err(invalid("GHF spinors do not have a fixed alpha parity"));
        }
        let encoding = Encoding::new(mapping, 2 * self.m, self.particles)?;
        // Budget the finite polynomial expansion before constructing it.
        let _terms = ctx.reserve(product(&[
            2 * self.m,
            2 * self.m,
            2 * self.m,
            2 * self.m,
            384,
        ])?)?;
        let mut terms = Vec::new();
        let mut groups = [Vec::new(), Vec::new()];
        for (ladders, coefficient) in self.fermion_terms() {
            let mapped = encoding.monomial(ctx, &ladders)?;
            let entries: Vec<_> = mapped
                .terms
                .iter()
                .map(|t| Term {
                    coefficient: t.coefficient * coefficient,
                    ..*t
                })
                .collect();
            if !ladders.is_empty() {
                groups[usize::from(ladders.len() == 4)].extend_from_slice(&entries);
            }
            terms.extend(entries);
        }
        let mut operator = Operator::from_terms(ctx, encoding.n_qubits(), terms)?;
        for terms in groups {
            let group = Operator::from_terms(ctx, encoding.n_qubits(), terms)?;
            if !group.terms.is_empty() {
                operator.decomposition.push(group);
            }
        }
        Ok(operator)
    }
    pub fn problem(
        &self,
        ctx: &Context,
        mapping: &str,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<CachedProblem> {
        if self.spinor {
            return Err(invalid(
                "the native GHF model needs a spin-orbit excitation pool before a quantum solve",
            ));
        }
        let encoding = Encoding::new(mapping, 2 * self.m, self.particles)?;
        let h = self.operator(ctx, mapping)?;
        CachedProblem::new(ctx, encoding, h, sector, dense, singles)
    }
}
