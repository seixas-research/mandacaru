use crate::{finite, product, Error, Result};

/// Rectangular index domain and positive-definite inverse metric, in Bohr.
#[derive(Clone, Debug)]
pub struct Grid {
    shape: [usize; 3],
    metric: [f64; 9],
    dv: f64,
    points: usize,
}

impl Grid {
    pub fn new(shape: [usize; 3], metric: [f64; 9], dv: f64) -> Result<Self> {
        if shape.iter().any(|&n| n == 0 || n > i32::MAX as usize) {
            return Err(Error::Invalid(
                "grid dimensions must be positive and fit int32".into(),
            ));
        }
        let points = product(&shape)?;
        product(&[points, 16])?
            .checked_add(1)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(Error::Overflow)?;
        finite(&metric, "inverse metric")?;
        if !dv.is_finite() || dv <= 0.0 {
            return Err(Error::Invalid(
                "grid volume element must be positive and finite".into(),
            ));
        }
        let mut l = [0.0; 9];
        for i in 0..3 {
            for j in 0..=i {
                if (metric[i * 3 + j] - metric[j * 3 + i]).abs()
                    > 1e-12 * metric[i * 3 + j].abs().max(1.0)
                {
                    return Err(Error::Invalid("inverse metric must be symmetric".into()));
                }
                let v =
                    metric[i * 3 + j] - (0..j).map(|k| l[i * 3 + k] * l[j * 3 + k]).sum::<f64>();
                if i == j {
                    if v <= 0.0 || !v.is_finite() {
                        return Err(Error::Invalid(
                            "inverse metric must be positive definite".into(),
                        ));
                    }
                    l[i * 3 + j] = v.sqrt();
                } else {
                    l[i * 3 + j] = v / l[j * 3 + j];
                }
            }
        }
        Ok(Self {
            shape,
            metric,
            dv,
            points,
        })
    }
    pub fn orthogonal(shape: [usize; 3], spacing: [f64; 3]) -> Result<Self> {
        if spacing.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(Error::Invalid(
                "grid spacing must be positive and finite".into(),
            ));
        }
        Self::new(
            shape,
            [
                1.0 / spacing[0].powi(2),
                0.0,
                0.0,
                0.0,
                1.0 / spacing[1].powi(2),
                0.0,
                0.0,
                0.0,
                1.0 / spacing[2].powi(2),
            ],
            spacing.iter().product(),
        )
    }
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
    pub fn points(&self) -> usize {
        self.points
    }
    pub fn metric(&self) -> &[f64; 9] {
        &self.metric
    }
    pub fn dv(&self) -> f64 {
        self.dv
    }
}
