//! Spectral broadening shared by molecular and crystal observables.
use super::invalid;
use crate::{finite, product, Buffer, Context, Result};

pub struct DosInput<'a> {
    pub shape: [usize; 3],
    pub levels: &'a [f64],
    pub weights: &'a [f64],
}

pub fn gaussian_dos(
    ctx: &Context,
    input: DosInput<'_>,
    axis: &[f64],
    width: f64,
    degeneracy: f64,
) -> Result<Buffer<f64>> {
    let [spins, points, bands] = input.shape;
    if input.shape.contains(&0)
        || input.levels.len() != product(&input.shape)?
        || input.weights.len() != points
        || axis.is_empty()
        || !width.is_finite()
        || width <= 0.
        || !degeneracy.is_finite()
        || degeneracy <= 0.
        || input.weights.iter().any(|w| *w < 0.)
        || (input.weights.iter().sum::<f64>() - 1.).abs() > 1e-10
    {
        return Err(invalid(
            "DOS needs matching spectral arrays, normalized weights and positive width/degeneracy",
        ));
    }
    finite(input.levels, "spectral levels")?;
    finite(input.weights, "spectral weights")?;
    finite(axis, "spectral axis")?;
    let mut out = ctx.allocate::<f64>(axis.len())?;
    let norm = degeneracy / (width * (2. * std::f64::consts::PI).sqrt());
    if !norm.is_finite() {
        return Err(invalid(
            "DOS width or degeneracy overflows its normalization",
        ));
    }
    for (i, &energy) in axis.iter().enumerate() {
        for s in 0..spins {
            for (k, &weight) in input.weights.iter().enumerate() {
                for b in 0..bands {
                    let delta = (energy - input.levels[(s * points + k) * bands + b]) / width;
                    out.as_mut_slice()[i] += weight * norm * (-0.5 * delta * delta).exp();
                }
            }
        }
    }
    finite(out.as_slice(), "density of states")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broadened_spectrum_conserves_state_count_and_spin_weight() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let axis: Vec<_> = (0..4001).map(|i| -2. + i as f64 * 0.001).collect();
        let levels = [-0.3, 0.4, -0.2, 0.6];
        let weights = [0.25, 0.75];
        let out = gaussian_dos(
            &ctx,
            DosInput {
                shape: [1, 2, 2],
                levels: &levels,
                weights: &weights,
            },
            &axis,
            0.1,
            2.,
        )
        .unwrap();
        let integral = out
            .as_slice()
            .windows(2)
            .map(|w| (w[0] + w[1]) * 0.0005)
            .sum::<f64>();
        assert!((integral - 4.).abs() < 1e-12);
        assert!(gaussian_dos(
            &ctx,
            DosInput {
                shape: [1, 2, 2],
                levels: &levels,
                weights: &weights
            },
            &axis,
            0.,
            2.
        )
        .is_err());
    }
}
