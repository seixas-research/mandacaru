//! How badly the s channel misses a hydrogen 1s entering its sphere.
use super::view::Smooth;

/// Largest tolerated miss, as a fraction of that function's norm inside the sphere. Above it
/// the s projectors cannot represent a neighbor's orbital entering the sphere, and bonds come
/// out too long or collapse.
pub const INTRUDING_S_TOLERANCE: f64 = 1.0;

/// The miss `||f - g|| / ||f||` inside `r_c` of the `l = 0` part `f` of a hydrogen 1s centered
/// just outside the sphere (`r_c + 0.2` Bohr) against its expansion
/// `g = sum_i <p_i|f> phi~_i` in the channel's projectors and smooth waves: zero for a complete
/// set. `None` for a dataset without an s channel.
pub fn intruding_s_miss(smooth: &Smooth) -> Option<f64> {
    let channel = smooth.channel(0)?;
    let r = smooth.r.as_slice();
    let n = r.len();
    let step = r[1] - r[0];
    let inside = (0..n)
        .filter(|&i| r[i] > 0. && r[i] <= channel.r_cut)
        .collect::<Vec<_>>();
    let distance = channel.r_cut + 0.2;
    let primitive = |s: f64| -(s + 1.) * (-s).exp();
    let target = inside
        .iter()
        .map(|&i| {
            let x = r[i];
            (primitive(x + distance) - primitive((x - distance).abs()))
                / (2. * x * distance * std::f64::consts::PI.sqrt())
        })
        .collect::<Vec<_>>();
    let mut miss = target.clone();
    for (k, wave) in channel.pseudo_waves.iter().enumerate().take(channel.rank) {
        let projector = &channel.projectors.as_slice()[k * n..(k + 1) * n];
        let coefficient = inside
            .iter()
            .zip(&target)
            .map(|(&i, t)| projector[i] * t * r[i] * r[i])
            .sum::<f64>()
            * step;
        for (m, &i) in miss.iter_mut().zip(&inside) {
            *m -= coefficient * wave.as_slice()[i];
        }
    }
    let norm = |f: &[f64]| {
        (f.iter()
            .zip(&inside)
            .map(|(v, &i)| v * v * r[i] * r[i])
            .sum::<f64>()
            * step)
            .sqrt()
    };
    Some(norm(&miss) / norm(&target))
}

/// The repair search's acceptance of an otherwise clean PAW-LCAO construction: `Some(miss)`
/// when the miss reaches [`INTRUDING_S_TOLERANCE`] (strict: below it is clean), else `None`.
pub fn failed_acceptance(miss: Option<f64>) -> Option<f64> {
    miss.filter(|&m| !(m < INTRUDING_S_TOLERANCE))
}
