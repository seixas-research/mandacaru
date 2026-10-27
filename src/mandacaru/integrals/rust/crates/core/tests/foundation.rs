use mandacaru_core::{kernels, Complex64, Context, Error, Geometry, Grid, LengthUnit};

#[test]
fn geometry_normalizes_once_and_preserves_partial_periodicity() {
    let geometry = Geometry::new(
        vec![1],
        vec![[0.529177210903, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [true, true, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    assert!((geometry.positions()[0][0] - 1.0).abs() < 1e-14);
    assert_eq!(geometry.pbc(), [true, true, false]);
    assert!(Geometry::new(
        vec![0],
        vec![[0.0; 3]],
        [[0.0; 3]; 3],
        [false; 3],
        LengthUnit::Bohr
    )
    .is_err());
    assert!(Geometry::new(
        vec![1],
        vec![[0.0; 3]],
        [[0.0; 3]; 3],
        [true; 3],
        LengthUnit::Bohr
    )
    .is_err());
}

#[test]
fn grid_rejects_invalid_metrics_and_overflow_before_allocation() {
    assert!(Grid::orthogonal([0, 2, 2], [1.0; 3]).is_err());
    assert!(Grid::orthogonal([i32::MAX as usize; 3], [1.0; 3]).is_err());
    assert!(Grid::new([2; 3], [0.0; 9], 1.0).is_err());
    assert!(Grid::new([2; 3], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1.0).is_err());
}

#[test]
fn memory_budget_accounts_live_storage_and_rolls_back_failures() {
    let context = Context::new(Some(1), 24).unwrap();
    let observer = context.clone();
    let buffer = context.allocate::<f64>(3).unwrap();
    assert_eq!(context.memory_used(), 24);
    assert!(matches!(
        context.allocate::<f64>(1),
        Err(Error::Budget { .. })
    ));
    assert!(matches!(
        context.allocate::<f64>(usize::MAX),
        Err(Error::Overflow)
    ));
    drop(context);
    assert_eq!(observer.memory_used(), 24);
    drop(buffer);
    assert_eq!(observer.memory_used(), 0);
    assert_eq!(observer.memory_peak(), 24);
}

#[test]
fn shrinking_a_workspace_keeps_its_retained_storage_reserved() {
    let context = Context::new(Some(1), 32).unwrap();
    let mut reservation = context.reserve(24).unwrap();
    reservation.shrink_to(16).unwrap();
    let buffer = context.allocate::<f64>(2).unwrap();
    assert_eq!(context.memory_used(), 32);
    assert!(reservation.shrink_to(17).is_err());
    assert_eq!(context.memory_used(), 32);
    reservation.shrink_to(0).unwrap();
    assert_eq!(context.memory_used(), 16);
    drop(buffer);
    drop(reservation);
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn discrete_laplacian_and_one_body_obey_the_same_operator() {
    let context = Context::new(Some(2), 1 << 20).unwrap();
    let grid = Grid::orthogonal([3; 3], [1.0; 3]).unwrap();
    let mut field = vec![0.0; 27];
    field[13] = 1.0;
    let lap = kernels::laplacian(&context, &grid, 1, &field).unwrap();
    assert_eq!(lap.as_slice()[13], -6.0);
    assert_eq!(lap.as_slice().iter().sum::<f64>(), 0.0);
    let psi: Vec<_> = field.iter().map(|&r| Complex64::new(r, 0.0)).collect();
    let (t, v) = kernels::one_body(&context, &grid, 1, &psi, &vec![2.0; 27]).unwrap();
    assert_eq!(t.as_slice()[0].re, 3.0);
    assert_eq!(v.as_slice()[0].re, 2.0);
    assert!(context.workers_peak() <= 2);
    assert!(kernels::laplacian(&context, &grid, 2, &field).is_err());
}

#[test]
fn radial_eigenpair_satisfies_the_matrix_equation() {
    let context = Context::new(Some(1), 1 << 20).unwrap();
    let (value, vector) = kernels::eigenpair(&context, &[2.0; 3], &[-1.0; 2], 0).unwrap();
    assert!((value - (2.0 - 2.0_f64.sqrt())).abs() < 1e-12);
    let x = vector.as_slice();
    for i in 0..3 {
        let ax =
            2.0 * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - if i < 2 { x[i + 1] } else { 0.0 };
        assert!((ax - value * x[i]).abs() < 1e-12);
    }
    assert!(kernels::eigenpair(&context, &[], &[], 0).is_err());
    assert!(kernels::eigenpair(&context, &[1.0], &[], 1).is_err());
}

#[test]
fn rejected_kernel_inputs_leave_no_allocations() {
    let context = Context::new(Some(1), 1 << 20).unwrap();
    assert!(kernels::fourier_bessel(&context, 0, &[0.1, 0.2], &[1.0], &[1.0]).is_err());
    assert!(kernels::bessel(&context, -1, &[0.1], &[1.0]).is_err());
    assert!(kernels::numerov(&context, &[0.0; 2], &[0.0; 2], &[0.0; 2], 1.0, 0, false).is_err());
    let spline = kernels::Spline::new(&[0.0, 1.0], &[0.0, 0.0, 0.0, 1.0], 1.0, 0).unwrap();
    assert!(kernels::real_shell(
        &context,
        [&[0.0]; 3],
        [0.0; 3],
        1.0,
        &spline,
        &[i32::MIN],
        -1
    )
    .is_err());
    assert_eq!(context.memory_used(), 0);
}
