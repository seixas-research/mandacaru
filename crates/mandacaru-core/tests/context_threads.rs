//! The worker counts a context reports: the default a context takes and the hardware width.

use mandacaru_core::{default_threads, hardware_threads, Context};

#[test]
fn a_context_without_a_request_takes_the_default_and_a_request_overrides_it() {
    let default = default_threads();
    assert!((1..=mandacaru_kernels::runtime::MAX_THREADS).contains(&default));
    assert_eq!(Context::new(None, 1 << 20).unwrap().threads(), default);
    assert_eq!(Context::new(Some(3), 1 << 20).unwrap().threads(), 3);
}

#[test]
fn the_hardware_width_is_at_least_one_thread_and_does_not_follow_a_context() {
    let width = hardware_threads();
    assert!(width >= 1);
    let _one = Context::new(Some(1), 1 << 20).unwrap();
    assert_eq!(hardware_threads(), width);
}
