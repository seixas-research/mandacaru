//! The rounding rules of the NumPy reductions and element operations a frozen result was computed
//! with, written out so a port reproduces them to the last bit.
//!
//! Each function states the NumPy expression it stands for. Where NumPy's formula is not the
//! textbook one it is the one reproduced here: the sum is pairwise, the complex magnitude is
//! `larger * sqrt(1 + (smaller / larger)^2)` with a fused multiply-add, and the division of a
//! complex number by a real one is `z * (1 / x)` through the complex quotient. The vectorized
//! builds of NumPy that use another formula (the libm magnitude of a CPU without AVX-512) are not
//! reproduced: these are the formulas of the Arm build. The default sort is an introsort whose order
//! among equal keys is part of its result, so [`argsort`] is that algorithm, not a stable sort.
use super::invalid;
use crate::Result;
use num_complex::{Complex32, Complex64};

/// `np.sum(values)`: the pairwise summation NumPy applies along a contiguous axis (blocks of eight
/// partial sums up to 128 elements, halves above), not the running sum.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |sum, &v| sum + v);
    }
    if n <= 128 {
        let mut r: [f64; 8] = std::array::from_fn(|j| values[j]);
        let mut i = 8;
        while i < n - n % 8 {
            for (j, partial) in r.iter_mut().enumerate() {
                *partial += values[i + j];
            }
            i += 8;
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[i..] {
            sum += v;
        }
        return sum;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// `np.max(values)`: `None` for no values, NaN when any value is NaN.
pub fn maximum(values: &[f64]) -> Option<f64> {
    let mut best = *values.first()?;
    for &v in values {
        if v.is_nan() {
            return Some(f64::NAN);
        }
        if v > best {
            best = v;
        }
    }
    Some(best)
}

/// `np.min(values)`: `None` for no values, NaN when any value is NaN.
pub fn minimum(values: &[f64]) -> Option<f64> {
    let mut best = *values.first()?;
    for &v in values {
        if v.is_nan() {
            return Some(f64::NAN);
        }
        if v < best {
            best = v;
        }
    }
    Some(best)
}

/// `np.argmax(values)`: the first index of the largest value, or of the first NaN.
pub fn argmax(values: &[f64]) -> Option<usize> {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return Some(i);
        }
        if v > values[best] {
            best = i;
        }
    }
    (!values.is_empty()).then_some(best)
}

/// `np.linspace(start, stop, num)` with the end point: `start + i * step`, the last point exactly
/// `stop` (`i / div * delta` when the step is zero).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut out: Vec<f64> = (0..num)
        .map(|i| {
            if step == 0.0 {
                (i as f64 / div) * delta + start
            } else {
                i as f64 * step + start
            }
        })
        .collect();
    out[num - 1] = stop;
    out
}

/// `np.percentile(values, q)` with the default linear interpolation: the order statistics at
/// `floor` and `floor + 1` of `(n - 1) q / 100`, joined as `a + (b - a) g` (`b - (b - a)(1 - g)` from
/// `g = 0.5`). NaN when a value is NaN; `q` in `[0, 100]`.
pub fn percentile(values: &[f64], q: f64) -> Result<f64> {
    if values.is_empty() || !(0.0..=100.0).contains(&q) {
        return Err(invalid("a percentile needs values and q in [0, 100]"));
    }
    if values.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let quantile = q / 100.0;
    let virtual_index = (n - 1) as f64 * quantile;
    // NumPy's bounds: at or above the last index both neighbors are the last element (-1), at or
    // below zero both are the first; gamma is measured from the bounded `previous`.
    let (mut previous, mut next) = (virtual_index.floor() as i64, virtual_index.floor() as i64 + 1);
    if virtual_index >= (n - 1) as f64 {
        (previous, next) = (-1, -1);
    }
    if virtual_index <= 0.0 {
        (previous, next) = (0, 0);
    }
    let gamma = virtual_index - previous as f64;
    let at = |i: i64| if i < 0 { sorted[n - 1] } else { sorted[i as usize] };
    let (a, b) = (at(previous), at(next));
    let diff = b - a;
    let lerp = a + diff * gamma;
    Ok(if gamma >= 0.5 { b - diff * (1.0 - gamma) } else { lerp })
}

/// `np.abs(z)` of a contiguous complex128 array: the SIMD magnitude `larger * sqrt(fma(r, r, 1))`
/// with `r = smaller / larger`, infinite when a part is, NaN when a part is NaN.
pub fn complex_abs(z: Complex64) -> f64 {
    let (re, im) = (z.re.abs(), z.im.abs());
    if re.is_infinite() || im.is_infinite() {
        return f64::INFINITY;
    }
    if re.is_nan() || im.is_nan() {
        return f64::NAN;
    }
    let (larger, smaller) = (re.max(im), re.min(im));
    let ratio = if larger == 0.0 { 0.0 } else { smaller / larger };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

/// `z / x` of a complex128 array by a Python float, as NumPy's complex quotient evaluates it for
/// `x + 0j`: `((a + b r) s, (b - a r) s)` with `r = 0 / x` and `s = 1 / (x + 0 r)`, which is
/// `z * (1 / x)` and not `z / x` componentwise.
pub fn divide_by_real(z: Complex64, x: f64) -> Complex64 {
    let (c, d) = (x, 0.0f64);
    if c.abs() >= d.abs() {
        if c.abs() == 0.0 && d.abs() == 0.0 {
            return Complex64::new(z.re / c.abs(), z.im / c.abs());
        }
        let rat = d / c;
        let scl = 1.0 / (c + d * rat);
        Complex64::new((z.re + z.im * rat) * scl, (z.im - z.re * rat) * scl)
    } else {
        let rat = c / d;
        let scl = 1.0 / (d + c * rat);
        Complex64::new((z.re * rat + z.im) * scl, (z.im * rat - z.re) * scl)
    }
}

/// [`divide_by_real`] for a complex64 array: the divisor is rounded to `f32` first and every
/// operation is `f32`.
pub fn divide_by_real_single(z: Complex32, x: f64) -> Complex32 {
    let (c, d) = (x as f32, 0.0f32);
    if c.abs() >= d.abs() {
        if c.abs() == 0.0 && d.abs() == 0.0 {
            return Complex32::new(z.re / c.abs(), z.im / c.abs());
        }
        let rat = d / c;
        let scl = 1.0f32 / (c + d * rat);
        Complex32::new((z.re + z.im * rat) * scl, (z.im - z.re * rat) * scl)
    } else {
        let rat = c / d;
        let scl = 1.0f32 / (d + c * rat);
        Complex32::new((z.re * rat + z.im) * scl, (z.im * rat - z.re) * scl)
    }
}

/// NumPy's ordering of floats: NaN after every number.
fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `np.argsort(values)` with the default kind: NumPy's introsort (median-of-three quicksort down to
/// partitions of sixteen elements, insertion sort below, heapsort when the recursion gets too deep).
/// It is not stable: the order of equal keys is the one this algorithm leaves, which is part of the
/// result a frozen run was computed with. The x86 builds of NumPy sort with a vectorized kernel
/// instead and order equal keys differently.
pub fn argsort(values: &[f64]) -> Vec<usize> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n < 2 {
        return order;
    }
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    let (mut pl, mut pr) = (0usize, n - 1);
    let mut depth = (usize::BITS - 1 - n.leading_zeros()) as i32 * 2;
    loop {
        if depth < 0 {
            heapsort(values, &mut order[pl..=pr]);
        } else {
            while pr - pl > 15 {
                let pm = pl + ((pr - pl) >> 1);
                if less(values[order[pm]], values[order[pl]]) {
                    order.swap(pm, pl);
                }
                if less(values[order[pr]], values[order[pm]]) {
                    order.swap(pr, pm);
                }
                if less(values[order[pm]], values[order[pl]]) {
                    order.swap(pm, pl);
                }
                let pivot = values[order[pm]];
                let (mut pi, mut pj) = (pl, pr - 1);
                order.swap(pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !less(values[order[pi]], pivot) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(pivot, values[order[pj]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    order.swap(pi, pj);
                }
                order.swap(pi, pr - 1);
                depth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, depth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, depth));
                    pl = pi + 1;
                }
            }
            for pi in pl + 1..=pr {
                let vi = order[pi];
                let vp = values[vi];
                let mut pj = pi;
                while pj > pl && less(vp, values[order[pj - 1]]) {
                    order[pj] = order[pj - 1];
                    pj -= 1;
                }
                order[pj] = vi;
            }
        }
        match stack.pop() {
            Some((left, right, remaining)) => {
                pl = left;
                pr = right;
                depth = remaining;
            }
            None => break,
        }
    }
    order
}

/// NumPy's heapsort of an index slice by the values they point to (the deep-recursion fallback).
fn heapsort(values: &[f64], order: &mut [usize]) {
    let mut n = order.len();
    // The heap is indexed from one, as in NumPy.
    let at = |order: &[usize], i: usize| order[i - 1];
    let mut l = n >> 1;
    while l > 0 {
        let tmp = at(order, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && less(values[at(order, j)], values[at(order, j + 1)]) {
                j += 1;
            }
            if less(values[tmp], values[at(order, j)]) {
                order[i - 1] = at(order, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        order[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = at(order, n);
        order[n - 1] = at(order, 1);
        n -= 1;
        let (mut i, mut j) = (1, 2);
        while j <= n {
            if j < n && less(values[at(order, j)], values[at(order, j + 1)]) {
                j += 1;
            }
            if less(values[tmp], values[at(order, j)]) {
                order[i - 1] = at(order, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        order[i - 1] = tmp;
    }
}
