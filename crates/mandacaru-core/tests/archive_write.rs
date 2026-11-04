use mandacaru_core::{
    math::archive::Archive,
    Context,
};
use num_complex::Complex64;

#[test]
fn test_archive_create_empty_and_write_range() {
    let ctx = Context::new(Some(1), 100 * 1024 * 1024).unwrap();
    let n = 5000;
    let archive = Archive::<Complex64>::create_empty(&ctx, n).unwrap();

    let slice1: Vec<Complex64> = (0..1000)
        .map(|i| Complex64::new(i as f64, -(i as f64)))
        .collect();
    let slice2: Vec<Complex64> = (1000..n)
        .map(|i| Complex64::new((i * 2) as f64, (i * 3) as f64))
        .collect();

    // Positional out-of-order write
    archive.write_range(1000, &slice2).unwrap();
    archive.write_range(0, &slice1).unwrap();
    archive.finish().unwrap();

    let mut read_back = vec![Complex64::default(); n];
    archive.read_range_with(0, &mut read_back, |v| v).unwrap();

    for i in 0..1000 {
        assert_eq!(read_back[i], slice1[i]);
    }
    for i in 1000..n {
        assert_eq!(read_back[i], slice2[i - 1000]);
    }
}

#[test]
fn test_archive_write_range_bounds() {
    let ctx = Context::new(Some(1), 100 * 1024 * 1024).unwrap();
    let archive = Archive::<f64>::create_empty(&ctx, 100).unwrap();

    let valid = vec![1.23; 50];
    assert!(archive.write_range(0, &valid).is_ok());
    assert!(archive.write_range(50, &valid).is_ok());

    let out_of_bounds = vec![4.56; 51];
    assert!(archive.write_range(50, &out_of_bounds).is_err());
    assert!(archive.write_range(101, &[]).is_err());
}
