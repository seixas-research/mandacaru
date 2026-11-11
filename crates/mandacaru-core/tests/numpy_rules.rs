//! The NumPy rounding rules (`math::numpy_rules`) and the peak finder (`math::peaks`): each
//! against a value worked out by hand or taken from the NumPy call it stands for.
use mandacaru_core::math::{numpy_rules as np, peaks::find_peaks};
use num_complex::{Complex32, Complex64};

#[test]
fn the_sum_is_pairwise_not_a_running_sum() {
    // Eight terms: ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) keeps the small terms that a
    // running sum loses against 1e16 (whose spacing is 2).
    let values = [1e16, 1., 1., 1., 1., 1., 1., 1.];
    assert_eq!(np::pairwise_sum(&values), 1e16 + 6.);
    assert_eq!(values.iter().sum::<f64>(), 1e16);
    // Under eight it is the running sum.
    assert_eq!(np::pairwise_sum(&[1e16, 1., 1.]), 1e16);
    assert_eq!(np::pairwise_sum(&[]), 0.);
    // Above 128 it halves: the two halves are summed pairwise and added.
    let long: Vec<f64> = (0..300).map(|i| i as f64).collect();
    assert_eq!(np::pairwise_sum(&long), (299. * 300.) / 2.);
}

#[test]
fn extrema_propagate_nan_and_argmax_takes_the_first() {
    assert_eq!(np::maximum(&[1., 3., 2.]), Some(3.));
    assert_eq!(np::minimum(&[2., -1., 5.]), Some(-1.));
    assert!(np::maximum(&[1., f64::NAN, 3.]).unwrap().is_nan());
    assert!(np::minimum(&[f64::NAN, 1.]).unwrap().is_nan());
    assert_eq!(np::maximum(&[]), None);
    assert_eq!(np::argmax(&[1., 7., 7., 3.]), Some(1));
    assert_eq!(np::argmax(&[1., f64::NAN, 9.]), Some(1));
    assert_eq!(np::argmax(&[]), None);
}

#[test]
fn linspace_steps_from_the_start_and_ends_on_the_stop() {
    assert_eq!(np::linspace(0., 1., 5), vec![0., 0.25, 0.5, 0.75, 1.]);
    assert_eq!(np::linspace(2., 2., 3), vec![2., 2., 2.]);
    assert_eq!(np::linspace(3., 9., 1), vec![3.]);
    assert!(np::linspace(0., 1., 0).is_empty());
    // The last point is exactly the stop, whatever the step's rounding.
    assert_eq!(*np::linspace(0., 1., 120).last().unwrap(), 1.);
}

#[test]
fn the_percentile_interpolates_linearly_between_order_statistics() {
    let v = [4., 1., 3., 2.];
    assert_eq!(np::percentile(&v, 0.).unwrap(), 1.);
    assert_eq!(np::percentile(&v, 100.).unwrap(), 4.);
    // (n - 1) q = 1.5: halfway between 2 and 3, from the upper side (gamma = 0.5).
    assert_eq!(np::percentile(&v, 50.).unwrap(), 2.5);
    assert_eq!(np::percentile(&[7.], 33.).unwrap(), 7.);
    assert!((np::percentile(&[0., 10.], 99.5).unwrap() - 9.95).abs() < 1e-12);
    assert!(np::percentile(&[1., f64::NAN], 50.).unwrap().is_nan());
    assert!(np::percentile(&[], 50.).is_err());
    assert!(np::percentile(&[1.], 101.).is_err());
}

#[test]
fn the_complex_magnitude_is_larger_times_the_root_of_one_plus_the_ratio_squared() {
    assert_eq!(np::complex_abs(Complex64::new(3., 4.)), 5.);
    assert_eq!(np::complex_abs(Complex64::new(-4., 3.)), 5.);
    assert_eq!(np::complex_abs(Complex64::new(0., 0.)), 0.);
    assert_eq!(np::complex_abs(Complex64::new(0., -2.)), 2.);
    assert_eq!(np::complex_abs(Complex64::new(f64::INFINITY, f64::NAN)), f64::INFINITY);
    assert!(np::complex_abs(Complex64::new(f64::NAN, 1.)).is_nan());
}

#[test]
fn a_complex_number_over_a_real_one_is_the_number_times_the_reciprocal() {
    // 7 * (1 / 3) = 2.333333333333333, one unit in the last place below 7 / 3.
    let q = np::divide_by_real(Complex64::new(7., 0.), 3.);
    assert_eq!(q.re, 7. * (1. / 3.));
    assert_ne!(q.re, 7. / 3.);
    assert_eq!(
        np::divide_by_real(Complex64::new(1., 2.), 4.),
        Complex64::new(0.25, 0.5)
    );
    // complex64: the divisor is rounded to f32 and the quotient is f32 throughout.
    let single = np::divide_by_real_single(Complex32::new(7., 0.), 3.);
    assert_eq!(single.re, 7.0f32 * (1.0f32 / 3.0f32));
    assert!(np::divide_by_real(Complex64::new(1., 1.), 0.).re.is_infinite());
}

#[test]
fn argsort_is_numpys_introsort_with_its_order_among_equal_keys() {
    assert_eq!(np::argsort(&[3., 1., 2.]), vec![1, 2, 0]);
    assert!(np::argsort(&[]).is_empty());
    // Up to fifteen elements the insertion sort keeps equal keys in their input order.
    assert_eq!(np::argsort(&[1., 1., 0., 1.]), vec![2, 0, 1, 3]);
    // NaN sorts last.
    assert_eq!(np::argsort(&[f64::NAN, 1., 0.]), vec![2, 1, 0]);
    // From sixteen the partitioning reorders equal keys; these orders are what NumPy returns.
    let a = [1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.];
    assert_eq!(
        np::argsort(&a),
        vec![1, 3, 4, 6, 11, 10, 0, 14, 13, 12, 8, 15, 7, 5, 2, 9, 16]
    );
    let b = [
        0.3, 0.1, 0.3, 0.3, 0.2, 0.1, 0.3, 0.0, 0.2, 0.3, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2,
    ];
    assert_eq!(
        np::argsort(&b),
        vec![7, 13, 10, 1, 17, 5, 14, 20, 18, 15, 11, 21, 8, 4, 12, 6, 16, 3, 2, 19, 9, 0]
    );
    let c: Vec<f64> = (0..64).map(|i| (i % 5) as f64).collect();
    assert_eq!(
        np::argsort(&c),
        vec![
            0, 25, 30, 15, 35, 40, 45, 10, 5, 55, 60, 50, 20, 46, 51, 21, 41, 36, 56, 26, 61, 31, 1, 6, 16, 11, 22, 2,
            57, 52, 7, 47, 12, 37, 42, 27, 32, 62, 17, 3, 48, 18, 8, 58, 63, 28, 23, 38, 33, 13, 53, 43, 59, 24, 4, 14,
            34, 29, 9, 44, 19, 39, 54, 49
        ]
    );
}

#[test]
fn argsort_reaches_numpys_heapsort_on_an_organ_pipe() {
    // The organ pipe 0, 1, ..., 128, ..., 1 drives the partitions past the depth limit, so the
    // heapsort finishes them: NumPy's first and last indices and its position-weighted checksum.
    let v: Vec<f64> = (0..256).map(|i: i32| i.min(256 - i) as f64).collect();
    let order = np::argsort(&v);
    assert_eq!(
        order[..24],
        [0, 255, 1, 2, 254, 3, 253, 4, 252, 5, 251, 250, 6, 249, 7, 248, 8, 247, 9, 246, 10, 245, 11, 244]
    );
    assert_eq!(order[248..], [124, 131, 125, 126, 130, 127, 129, 128]);
    assert_eq!(order.iter().enumerate().map(|(i, &x)| i * x).sum::<usize>(), 4177914);
}

#[test]
fn argsort_sorts_every_pattern_of_every_size() {
    // Ascending, descending, constant, organ-pipe and sawtooth inputs of every size from 0 to 200
    // and of 1000 and 4097 values: the result is always a sorting permutation.
    let patterns: Vec<Box<dyn Fn(usize) -> f64>> = vec![
        Box::new(|i| i as f64),
        Box::new(|i| -(i as f64)),
        Box::new(|_| 1.0),
        Box::new(|i| (i as f64 - 50.).abs()),
        Box::new(|i| (i % 7) as f64),
    ];
    for pattern in &patterns {
        for n in (0..200).chain([1000, 4097]) {
            let v: Vec<f64> = (0..n).map(|i| pattern(i)).collect();
            let order = np::argsort(&v);
            let mut seen = order.clone();
            seen.sort();
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
            assert!(order.windows(2).all(|w| v[w[0]] <= v[w[1]]));
        }
    }
}

#[test]
fn peaks_are_points_above_their_left_neighbor_that_the_curve_then_leaves() {
    let x = [0., 1., 0., 2., 2., 0., 3.];
    // A plateau counts once, at its first point; the last point is never a peak.
    assert_eq!(find_peaks(&x, f64::NEG_INFINITY), vec![1, 3]);
    assert_eq!(find_peaks(&x, 1.5), vec![3]);
    assert!(find_peaks(&[1., 2.], 0.).is_empty());
    assert!(find_peaks(&[0., 1., f64::NAN, 0.], 0.).is_empty());
}
