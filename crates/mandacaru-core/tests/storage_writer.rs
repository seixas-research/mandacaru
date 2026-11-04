use mandacaru_core::{
    math::storage::{ComplexSamples, Precision},
    periodic::setup::shells::{ArchiveRows, ComplexSamplesRows, RowSink},
    Context,
};
use num_complex::Complex64;

#[test]
fn test_complex_samples_zeros_and_fill_row() {
    let ctx = Context::new(Some(1), 100 * 1024 * 1024).unwrap();
    let rows = 4;
    let points = 256;
    let mut samples = ComplexSamples::zeros(&ctx, Precision::Float64, rows * points).unwrap();

    let row_data: Vec<Complex64> = (0..points)
        .map(|i| Complex64::new(i as f64 * 0.1, -(i as f64) * 0.2))
        .collect();

    samples.fill_row(2, points, &row_data).unwrap();

    let read = samples.read(&ctx).unwrap();
    let values = &read[..];

    // Rows 0, 1, 3 are zero
    for r in [0, 1, 3] {
        for p in 0..points {
            assert_eq!(values[r * points + p], Complex64::new(0.0, 0.0));
        }
    }
    // Row 2 matches row_data
    for p in 0..points {
        assert_eq!(values[2 * points + p], row_data[p]);
    }
}

#[test]
fn test_complex_samples_archive_writer_float64_and_float32() {
    let ctx = Context::new(Some(1), 100 * 1024 * 1024).unwrap();
    let rows = 5;
    let points = 128;

    // Test Float64
    {
        let mut writer = ComplexSamples::archive_writer(&ctx, Precision::Float64, rows, points).unwrap();
        for r in 0..rows {
            let row: Vec<Complex64> = (0..points)
                .map(|p| Complex64::new((r * 100 + p) as f64, (r * 10 + p) as f64))
                .collect();
            writer.push_row(&row).unwrap();
        }
        let samples = writer.finish().unwrap();
        assert_eq!(samples.precision(), Precision::Float64);
        let read = samples.read(&ctx).unwrap();
        for r in 0..rows {
            for p in 0..points {
                let expected = Complex64::new((r * 100 + p) as f64, (r * 10 + p) as f64);
                assert_eq!(read[r * points + p], expected);
            }
        }
    }

    // Test Float32 with bit-exact row-once rounding
    {
        let mut writer = ComplexSamples::archive_writer(&ctx, Precision::Float32, rows, points).unwrap();
        let mut all_expected = Vec::with_capacity(rows * points);
        for r in 0..rows {
            let row: Vec<Complex64> = (0..points)
                .map(|p| Complex64::new((r * 100 + p) as f64 * 0.123456789, (r * 10 + p) as f64 * 0.987654321))
                .collect();
            for &v in &row {
                all_expected.push(Complex64::new(v.re as f32 as f64, v.im as f32 as f64));
            }
            writer.push_row(&row).unwrap();
        }
        let samples = writer.finish().unwrap();
        assert_eq!(samples.precision(), Precision::Float32);
        let read = samples.read(&ctx).unwrap();
        for i in 0..rows * points {
            assert_eq!(read[i], all_expected[i], "bit-exact single-precision rounding match at index {}", i);
        }
    }
}

#[test]
fn test_complex_samples_row_sinks() {
    let ctx = Context::new(Some(1), 100 * 1024 * 1024).unwrap();
    let n_k = 2;
    let functions = 3;
    let points = 64;
    let total_rows = n_k * functions;

    // Test ComplexSamplesRows with resident samples
    let mut resident = ComplexSamples::zeros(&ctx, Precision::Float64, total_rows * points).unwrap();
    {
        let mut sink = ComplexSamplesRows::new(&mut resident, functions, points);
        for k in 0..n_k {
            for m in 0..functions {
                let row = vec![Complex64::new(k as f64, m as f64); points];
                sink.put(k, m, &row).unwrap();
            }
        }
    }
    let read = resident.read(&ctx).unwrap();
    for k in 0..n_k {
        for m in 0..functions {
            let idx = (k * functions + m) * points;
            assert_eq!(read[idx], Complex64::new(k as f64, m as f64));
        }
    }

    // Test ArchiveRows with positional archive writer
    let writer = ComplexSamples::archive_writer(&ctx, Precision::Float32, total_rows, points).unwrap();
    {
        let mut sink = ArchiveRows::new(&writer, functions, points);
        // Put in reverse k and m order to verify random-access / positional writes
        for k in (0..n_k).rev() {
            for m in (0..functions).rev() {
                let row = vec![Complex64::new((k + 1) as f64 * 0.5, (m + 1) as f64 * 0.25); points];
                sink.put(k, m, &row).unwrap();
            }
        }
    }
    let archived = writer.finish().unwrap();
    let read_archived = archived.read(&ctx).unwrap();
    for k in 0..n_k {
        for m in 0..functions {
            let idx = (k * functions + m) * points;
            let expected = Complex64::new(((k + 1) as f64 * 0.5) as f32 as f64, ((m + 1) as f64 * 0.25) as f32 as f64);
            assert_eq!(read_archived[idx], expected);
        }
    }
}
