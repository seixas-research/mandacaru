//! Random circuit fidelities and the binned Haar KL divergence.
use super::{evolution, CachedProblem};
use crate::{product, Buffer, Context, Error, Result};
pub fn dimension(qubits: usize, particles: Option<[usize; 2]>) -> Result<usize> {
    if qubits > 62 {
        return Err(Error::Invalid(
            "expressibility supports at most 62 qubits".into(),
        ));
    }
    if let Some([a, b]) = particles {
        if qubits % 2 != 0 || a > qubits / 2 || b > qubits / 2 {
            return Err(Error::Invalid(
                "particle-sector dimension needs an even register and valid spin counts".into(),
            ));
        }
        product(&[
            super::algebra::choose(qubits / 2, a)?,
            super::algebra::choose(qubits / 2, b)?,
        ])
    } else {
        1usize.checked_shl(qubits as u32).ok_or(Error::Overflow)
    }
}
pub fn kl(ctx: &Context, fidelities: &[f64], bins: usize, dimension: usize) -> Result<f64> {
    if fidelities.is_empty()
        || bins < 2
        || dimension == 0
        || fidelities
            .iter()
            .any(|f| !f.is_finite() || !(0. - 1e-12..=1. + 1e-12).contains(f))
    {
        return Err(Error::Invalid(
            "KL divergence needs fidelities in [0,1], at least two bins and a positive dimension"
                .into(),
        ));
    }
    if dimension == 1 {
        if fidelities.iter().all(|f| (f - 1.).abs() < 1e-10) {
            return Ok(0.);
        }
        return Err(Error::Invalid(
            "a one-dimensional ansatz must have unit fidelities".into(),
        ));
    }
    let mut counts = ctx.allocate::<usize>(bins)?;
    for f in fidelities {
        let bin = ((f.clamp(0., 1.) * bins as f64) as usize).min(bins - 1);
        counts.as_mut_slice()[bin] += 1;
    }
    let mut value = 0.;
    for (i, &count) in counts.as_slice().iter().enumerate() {
        if count == 0 {
            continue;
        }
        let a = i as f64 / bins as f64;
        let b = (i + 1) as f64 / bins as f64;
        let lo = (dimension as f64 - 1.) * (-a).ln_1p();
        let hi = (dimension as f64 - 1.) * (-b).ln_1p();
        let log_haar = lo + (-(hi - lo).exp_m1()).ln();
        let p = count as f64 / fidelities.len() as f64;
        value += p * (p.ln() - log_haar);
    }
    Ok(value.max(0.))
}
impl CachedProblem {
    pub fn sample_fidelities(
        &self,
        indices: &[usize],
        pairs: usize,
        seed: u64,
        product_form: bool,
    ) -> Result<Buffer<f64>> {
        if pairs == 0 {
            return Err(Error::Invalid(
                "fidelity sampling needs a positive pair count".into(),
            ));
        }
        self.context.run(|| {
            let _work = self.context.reserve(product(&[indices.len(), 16])?)?;
            let mut samples = self.context.allocate::<f64>(pairs)?;
            let mut state = seed;
            let mut parameters = vec![0.; indices.len()];
            let mut sample = || -> Result<Buffer<super::C64>> {
                for theta in &mut parameters {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    z ^= z >> 31;
                    *theta = 2. * std::f64::consts::PI * (z >> 11) as f64 / (1u64 << 53) as f64;
                }
                self.prepare_state(indices, &parameters, product_form)
            };
            for value in samples.as_mut_slice() {
                let a = sample()?;
                let b = sample()?;
                *value = evolution::inner(a.as_slice(), b.as_slice())
                    .norm_sqr()
                    .clamp(0., 1.);
            }
            Ok(samples)
        })
    }
}
