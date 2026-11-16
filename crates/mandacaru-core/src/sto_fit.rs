//! Unit Slater orbitals fitted by contracted Gaussians (the STO-nG reference fits that
//! `test/migration/export_gaussian_fits.py` freezes): the Slater-Gaussian overlaps by 64-point
//! Gauss-Legendre quadrature and the projected-overlap maximization by Nelder-Mead from three
//! even-tempered starts.
use crate::{quantum::optimizer, Context, Error, Result};
use nalgebra::{DMatrix, DVector};

fn gamma_half_integer(l: usize) -> f64 {
    let mut g = std::f64::consts::PI.sqrt();
    for i in 0..=l {
        g *= (i as f64) + 0.5;
    }
    g
}

const GL64_X: [f64; 64] = [
    -9.99305041735772170e-01,
    -9.96340116771955220e-01,
    -9.91013371476744287e-01,
    -9.83336253884625977e-01,
    -9.73326827789910975e-01,
    -9.61008799652053658e-01,
    -9.46411374858402765e-01,
    -9.29569172131939570e-01,
    -9.10522137078502825e-01,
    -8.89315445995114140e-01,
    -8.65999398154092770e-01,
    -8.40629296252580316e-01,
    -8.13265315122797539e-01,
    -7.83972358943341385e-01,
    -7.52819907260531940e-01,
    -7.19881850171610771e-01,
    -6.85236313054233270e-01,
    -6.48965471254657311e-01,
    -6.11155355172393278e-01,
    -5.71895646202634000e-01,
    -5.31279464019894565e-01,
    -4.89403145707052956e-01,
    -4.46366017253464087e-01,
    -4.02270157963991570e-01,
    -3.57220158337668126e-01,
    -3.11322871990210970e-01,
    -2.64687162208767424e-01,
    -2.17423643740007083e-01,
    -1.69644420423992803e-01,
    -1.21462819296120558e-01,
    -7.29931217877990424e-02,
    -2.43502926634244360e-02,
    2.43502926634244360e-02,
    7.29931217877990424e-02,
    1.21462819296120558e-01,
    1.69644420423992803e-01,
    2.17423643740007083e-01,
    2.64687162208767424e-01,
    3.11322871990210970e-01,
    3.57220158337668126e-01,
    4.02270157963991570e-01,
    4.46366017253464087e-01,
    4.89403145707052956e-01,
    5.31279464019894565e-01,
    5.71895646202634000e-01,
    6.11155355172393278e-01,
    6.48965471254657311e-01,
    6.85236313054233270e-01,
    7.19881850171610771e-01,
    7.52819907260531940e-01,
    7.83972358943341385e-01,
    8.13265315122797539e-01,
    8.40629296252580316e-01,
    8.65999398154092770e-01,
    8.89315445995114140e-01,
    9.10522137078502825e-01,
    9.29569172131939570e-01,
    9.46411374858402765e-01,
    9.61008799652053658e-01,
    9.73326827789910975e-01,
    9.83336253884625977e-01,
    9.91013371476744287e-01,
    9.96340116771955220e-01,
    9.99305041735772170e-01,
];

const GL64_W: [f64; 64] = [
    1.78328072169426418e-03,
    4.14703326056464847e-03,
    6.50445796897843737e-03,
    8.84675982636336060e-03,
    1.11681394601310085e-02,
    1.34630478967191197e-02,
    1.57260304760251102e-02,
    1.79517157756972877e-02,
    2.01348231535300390e-02,
    2.22701738083829447e-02,
    2.43527025687110855e-02,
    2.63774697150548944e-02,
    2.83396726142594244e-02,
    3.02346570724024884e-02,
    3.20579283548514116e-02,
    3.38051618371418769e-02,
    3.54722132568822263e-02,
    3.70551285402400191e-02,
    3.85501531786156051e-02,
    3.99537411327204536e-02,
    4.12625632426234859e-02,
    4.24735151236535421e-02,
    4.35837245293234712e-02,
    4.45905581637566009e-02,
    4.54916279274181004e-02,
    4.62847965813143747e-02,
    4.69681828162100065e-02,
    4.75401657148303430e-02,
    4.79993885964583034e-02,
    4.83447622348029335e-02,
    4.85754674415034282e-02,
    4.86909570091396890e-02,
    4.86909570091396890e-02,
    4.85754674415034282e-02,
    4.83447622348029335e-02,
    4.79993885964583034e-02,
    4.75401657148303430e-02,
    4.69681828162100065e-02,
    4.62847965813143747e-02,
    4.54916279274181004e-02,
    4.45905581637566009e-02,
    4.35837245293234712e-02,
    4.24735151236535421e-02,
    4.12625632426234859e-02,
    3.99537411327204536e-02,
    3.85501531786156051e-02,
    3.70551285402400191e-02,
    3.54722132568822263e-02,
    3.38051618371418769e-02,
    3.20579283548514116e-02,
    3.02346570724024884e-02,
    2.83396726142594244e-02,
    2.63774697150548944e-02,
    2.43527025687110855e-02,
    2.22701738083829447e-02,
    2.01348231535300390e-02,
    1.79517157756972877e-02,
    1.57260304760251102e-02,
    1.34630478967191197e-02,
    1.11681394601310085e-02,
    8.84675982636336060e-03,
    6.50445796897843737e-03,
    4.14703326056464847e-03,
    1.78328072169426418e-03,
];

/// `int_0^inf r^q exp(-alpha r^2 - r) dr` by 64-point Gauss-Legendre quadrature.
pub fn slater_moment_single(alpha: f64, q: usize) -> f64 {
    if !alpha.is_finite() || alpha <= 0.0 {
        return f64::NAN;
    }
    if alpha >= 0.2 {
        let sa = alpha.sqrt();
        let mut integ = 0.0;
        for i in 0..64 {
            let u = 3.5 * (GL64_X[i] + 1.0);
            let w = 3.5 * GL64_W[i];
            integ += w * u.powi(q as i32) * (-u * u - u / sa).exp();
        }
        integ * alpha.powf(-((q as f64) + 1.0) / 2.0)
    } else {
        let mut integ = 0.0;
        for i in 0..64 {
            let r = 22.5 * (GL64_X[i] + 1.0);
            let w = 22.5 * GL64_W[i];
            integ += w * r.powi(q as i32) * (-alpha * r * r - r).exp();
        }
        integ
    }
}

/// The overlap of a normalized `l` Gaussian of exponent `alpha` with the unit `(n, l)` Slater
/// orbital.
pub fn slater_projection_single(alpha: f64, n: usize, l: usize) -> f64 {
    let p = l as f64 + 1.5;
    let g = gamma_half_integer(l);
    let norm = (2.0 * (2.0 * alpha).powf(p) / g).sqrt();
    let mut factorial = 1.0;
    for k in 1..=(2 * n) {
        factorial *= k as f64;
    }
    let sto_norm = ((2.0f64).powi((2 * n + 1) as i32) / factorial).sqrt();
    let integ = slater_moment_single(alpha, n + l + 1);
    norm * sto_norm * integ
}

/// The `n_gaussians`-term fit of the unit `(n, l)` Slater orbital: `(exponents, coefficients,
/// overlap^2)`, the exponents in descending order.
pub fn fit_sto_reference(
    n: usize,
    l: usize,
    n_gaussians: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    if n_gaussians < 1 {
        return Err(Error::Invalid("n_gaussians must be >= 1".into()));
    }
    let p = l as f64 + 1.5;
    let center = 1.0 / ((n.saturating_sub(1).max(1)) as f64).powi(2);
    let k = n_gaussians;

    let projected_overlap = |alpha: &[f64]| -> Option<(f64, Vec<f64>)> {
        if alpha.iter().any(|&a| !a.is_finite() || a <= 0.0) {
            return None;
        }
        let mut mat_s = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let ai = alpha[i];
                let aj = alpha[j];
                let val = (2.0 * (ai * aj).sqrt() / (ai + aj)).powf(p);
                mat_s[(i, j)] = val;
            }
        }
        let mut vec_b = DVector::zeros(k);
        for i in 0..k {
            let bi = slater_projection_single(alpha[i], n, l);
            if !bi.is_finite() {
                return None;
            }
            vec_b[i] = bi;
        }
        let decomp = mat_s.lu();
        let c = decomp.solve(&vec_b)?;
        let val = vec_b.dot(&c);
        if !val.is_finite() || val > 1.0 + 1e-9 || val < 0.0 {
            return None;
        }
        Some((val, c.as_slice().to_vec()))
    };

    let neg_overlap_sq = |log_alpha: &[f64]| -> f64 {
        for &la in log_alpha {
            if la < -12.0 || la > 12.0 || !la.is_finite() {
                return 1.0;
            }
        }
        let alpha: Vec<f64> = log_alpha.iter().map(|&la| la.exp()).collect();
        match projected_overlap(&alpha) {
            Some((val, _)) => -val,
            None => 1.0,
        }
    };

    let ctx = Context::new(None, 64 * 1024 * 1024)?;
    let mut best_val = 1.0;
    let mut best_x = Vec::new();

    for &spread in &[1.6f64, 2.2, 3.0] {
        let mut t0 = Vec::with_capacity(k);
        for idx in 0..k {
            let offset = (idx as f64) - ((k as f64) - 1.0) / 2.0;
            t0.push((center * spread.powf(offset)).ln());
        }
        let options = optimizer::Options {
            method: optimizer::Method::NelderMead,
            tolerance: Some(1e-12),
            max_iterations: 20000,
            max_evaluations: 20000,
            seed: 0,
            track_best: true,
            patience: 5,
        };
        if let Ok(outcome) = optimizer::minimize(&ctx, &t0, &options, |x| {
            Ok((neg_overlap_sq(x), vec![0.0; x.len()]))
        }) {
            if outcome.energy < best_val {
                best_val = outcome.energy;
                best_x = outcome.parameters;
            }
        }
    }

    if best_x.is_empty() || best_val >= 0.0 {
        return Err(Error::Kernel(format!(
            "the STO-{}G fit of the ({}, {}) Slater orbital did not converge to a well-conditioned contraction; use fewer primitives",
            n_gaussians, n, l
        )));
    }

    let alpha: Vec<f64> = best_x.iter().map(|&la| la.exp()).collect();
    let (ov, c) = projected_overlap(&alpha)
        .ok_or_else(|| Error::Kernel("infeasible best fit parameters".into()))?;

    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| alpha[b].total_cmp(&alpha[a]));

    let sorted_alpha: Vec<f64> = indices.iter().map(|&i| alpha[i]).collect();
    let sorted_c: Vec<f64> = indices.iter().map(|&i| c[i]).collect();

    Ok((sorted_alpha, sorted_c, ov))
}
