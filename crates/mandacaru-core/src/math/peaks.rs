//! Local maxima of a sampled curve.

/// The indices of the local maxima of `x` at least `height` high, in increasing order. A peak is a
/// point above its left neighbor; over a flat top it is the first point of the plateau, which must
/// then fall. The two end points are never peaks; a NaN sample
/// compares false with everything, so it is never one.
pub fn find_peaks(x: &[f64], height: f64) -> Vec<usize> {
    let n = x.len();
    let mut peaks = Vec::new();
    if n >= 3 {
        let mut i = 1;
        while i < n - 1 {
            if x[i] > x[i - 1] {
                let mut j = i;
                while j < n - 1 && x[j + 1] == x[i] {
                    j += 1;
                }
                if j < n - 1 && x[j + 1] < x[i] && x[i] >= height {
                    peaks.push(i);
                }
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }
    peaks
}
