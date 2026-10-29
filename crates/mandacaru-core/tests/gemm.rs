use mandacaru_core::math::gemm::{multiply, MatrixMut, MatrixRef};

#[test]
fn padded_rectangular_product_preserves_output_padding() {
    let (m, k, n) = (7, 13, 5);
    let (lda, ldb, ldc) = (m + 3, k + 2, m + 1);
    let mut a = vec![f64::NAN; lda * k];
    let mut b = vec![f64::NAN; ldb * n];
    let mut c = vec![77.; ldc * n];
    for j in 0..k {
        for i in 0..m {
            a[i + j * lda] = ((i + 3 * j) as f64).sin();
        }
    }
    for j in 0..n {
        for i in 0..k {
            b[i + j * ldb] = ((2 * i + j) as f64).cos();
        }
    }
    multiply(
        MatrixRef {
            values: &a,
            rows: m,
            columns: k,
            leading: lda,
        },
        MatrixRef {
            values: &b,
            rows: k,
            columns: n,
            leading: ldb,
        },
        MatrixMut {
            values: &mut c,
            rows: m,
            columns: n,
            leading: ldc,
        },
        -0.7,
        0.3,
    )
    .unwrap();
    for j in 0..n {
        for i in 0..m {
            let expected =
                -0.7 * (0..k).map(|p| a[i + p * lda] * b[p + j * ldb]).sum::<f64>() + 0.3 * 77.;
            assert!((c[i + j * ldc] - expected).abs() < 1e-12);
        }
        assert_eq!(c[m + j * ldc], 77.);
    }
}

#[test]
fn empty_inner_dimension_obeys_beta_and_invalid_storage_is_rejected() {
    for beta in [0., 2.] {
        let mut output = [if beta == 0. { f64::NAN } else { 3. }; 4];
        multiply(
            MatrixRef {
                values: &[],
                rows: 2,
                columns: 0,
                leading: 2,
            },
            MatrixRef {
                values: &[],
                rows: 0,
                columns: 2,
                leading: 0,
            },
            MatrixMut {
                values: &mut output,
                rows: 2,
                columns: 2,
                leading: 2,
            },
            1.,
            beta,
        )
        .unwrap();
        assert_eq!(output, [if beta == 0. { 0. } else { 6. }; 4]);
    }
    let mut output = [0.; 4];
    assert!(multiply(
        MatrixRef {
            values: &[1.; 3],
            rows: 2,
            columns: 2,
            leading: 2
        },
        MatrixRef {
            values: &[1.; 4],
            rows: 2,
            columns: 2,
            leading: 2
        },
        MatrixMut {
            values: &mut output,
            rows: 2,
            columns: 2,
            leading: 2
        },
        1.,
        0.,
    )
    .is_err());
}
