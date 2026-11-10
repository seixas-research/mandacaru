//! The pairwise ranker of VALSQA's decision A (2026-10-01): a Bradley-Terry model on the
//! standardized insertion descriptors, `P(mu gains more than nu) = sigma(w . (x_mu - x_nu))`, fitted
//! on every pair of insertions measured at one state.
//!
//! A candidate's score is `w . x`: within a state it ranks the candidates, which is the proposal's
//! decision. The objective is the summed logistic loss of the pairs plus `ridge / 2 |w|^2`, with
//! `ridge = 2 l2 pairs` (the mean loss plus `l2 |w|^2`). It is strictly convex, so damped Newton
//! steps with a backtracking line search reach its unique minimizer.
use crate::{Error, Result};
use nalgebra::{SMatrix, SVector};

/// The number of insertion descriptors (`graph::Candidates::descriptors` columns).
pub const DESCRIPTORS: usize = 9;
type Vector = SVector<f64, DESCRIPTORS>;
type Square = SMatrix<f64, DESCRIPTORS, DESCRIPTORS>;

/// A fitted ranker: the weights on the descriptors standardized by `mean` and `std`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranker {
    pub weights: [f64; DESCRIPTORS],
    pub mean: [f64; DESCRIPTORS],
    pub std: [f64; DESCRIPTORS],
    /// The number of ordered pairs it was fitted on.
    pub pairs: usize,
}

/// One state's candidates: their descriptor rows and targets (the compressed energy change; lower
/// is a larger decrease).
pub struct Sample<'a> {
    pub descriptors: &'a [[f64; DESCRIPTORS]],
    pub targets: &'a [f64],
}

fn softplus(x: f64) -> f64 {
    if x > 0. {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}

impl Ranker {
    /// Higher means a larger predicted energy decrease.
    pub fn score(&self, descriptors: &[f64; DESCRIPTORS]) -> f64 {
        (0..DESCRIPTORS).map(|c| (descriptors[c] - self.mean[c]) / self.std[c] * self.weights[c]).sum()
    }

    /// The ranker of `samples`, or `None` when no state has two candidates with different targets.
    pub fn fit(samples: &[Sample], l2: f64) -> Result<Option<Self>> {
        if !l2.is_finite() || l2 <= 0. {
            return Err(Error::Invalid("the ranker's l2 must be positive and finite".into()));
        }
        let mut count = 0.;
        let mut mean = [0.; DESCRIPTORS];
        for sample in samples {
            if sample.descriptors.len() != sample.targets.len()
                || sample.targets.iter().chain(sample.descriptors.iter().flatten()).any(|v| !v.is_finite())
            {
                return Err(Error::Invalid("a ranker sample needs finite descriptors, one row per target".into()));
            }
            for row in sample.descriptors {
                count += 1.;
                for c in 0..DESCRIPTORS {
                    mean[c] += row[c];
                }
            }
        }
        if count == 0. {
            return Ok(None);
        }
        mean.iter_mut().for_each(|m| *m /= count);
        let mut std = [0.; DESCRIPTORS];
        for row in samples.iter().flat_map(|s| s.descriptors) {
            for c in 0..DESCRIPTORS {
                std[c] += (row[c] - mean[c]).powi(2) / count;
            }
        }
        for s in &mut std {
            *s = if s.sqrt() > 1e-12 { s.sqrt() } else { 1. };
        }
        let standard = |row: &[f64; DESCRIPTORS]| Vector::from_fn(|c, _| (row[c] - mean[c]) / std[c]);
        let mut differences = Vec::new();
        for sample in samples {
            for a in 0..sample.targets.len() {
                for b in a + 1..sample.targets.len() {
                    let (ta, tb) = (sample.targets[a], sample.targets[b]);
                    if ta == tb {
                        continue;
                    }
                    let (better, worse) = if ta < tb { (a, b) } else { (b, a) };
                    differences.push(standard(&sample.descriptors[better]) - standard(&sample.descriptors[worse]));
                }
            }
        }
        if differences.is_empty() {
            return Ok(None);
        }
        let ridge = 2. * l2 * differences.len() as f64;
        let objective = |w: &Vector| -> f64 {
            differences.iter().map(|d| softplus(-d.dot(w))).sum::<f64>() + 0.5 * ridge * w.norm_squared()
        };
        let mut w = Vector::zeros();
        for _ in 0..100 {
            let mut gradient = ridge * w;
            let mut hessian = Square::identity() * ridge;
            for d in &differences {
                let margin = d.dot(&w);
                gradient -= d * sigmoid(-margin);
                let p = sigmoid(margin);
                hessian += d * d.transpose() * (p * (1. - p));
            }
            if gradient.amax() <= 1e-13 * (1. + ridge) {
                break;
            }
            let step = hessian
                .cholesky()
                .ok_or_else(|| Error::Invalid("the ranker's curvature is not positive definite".into()))?
                .solve(&-gradient);
            let (start, mut t) = (objective(&w), 1.);
            while objective(&(w + step * t)) > start + 1e-4 * t * gradient.dot(&step) && t > 1e-12 {
                t *= 0.5;
            }
            w += step * t;
        }
        Ok(Some(Self { weights: w.into(), mean, std, pairs: differences.len() }))
    }
}
