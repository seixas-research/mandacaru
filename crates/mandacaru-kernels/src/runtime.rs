//! Shared thread policy for the legacy ABI and checked calculation contexts.
//!
//! Context overrides are thread-local, restored on unwind, and propagated to
//! scoped workers. Independent calculations never mutate process environment.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{Scope, ScopedJoinHandle};

pub const MAX_THREADS: usize = 1024;

#[derive(Default, Debug)]
pub struct WorkerStats {
    active: AtomicUsize,
    peak: AtomicUsize,
    total: AtomicUsize,
}

impl WorkerStats {
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
    pub fn total(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct Policy {
    threads: usize,
    stats: Arc<WorkerStats>,
}

thread_local! {
    static POLICY: RefCell<Option<Policy>> = const { RefCell::new(None) };
}

struct Restore(Option<Policy>);
impl Drop for Restore {
    fn drop(&mut self) {
        POLICY.with(|p| *p.borrow_mut() = self.0.take());
    }
}

pub fn thread_count() -> usize {
    if let Some(n) = POLICY.with(|p| p.borrow().as_ref().map(|p| p.threads)) {
        return n;
    }
    for key in ["MANDACARU_NUM_THREADS", "OMP_NUM_THREADS"] {
        if let Some(n) = std::env::var(key)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            if n > 0 {
                return n.min(MAX_THREADS);
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(MAX_THREADS)
}

pub fn with_policy<T>(threads: usize, stats: Arc<WorkerStats>, f: impl FnOnce() -> T) -> T {
    assert!((1..=MAX_THREADS).contains(&threads));
    let previous = POLICY.with(|p| p.replace(Some(Policy { threads, stats })));
    let _restore = Restore(previous);
    f()
}

pub fn spawn<'scope, 'env, F, T>(
    scope: &'scope Scope<'scope, 'env>,
    f: F,
) -> ScopedJoinHandle<'scope, T>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    let policy = POLICY.with(|p| p.borrow().clone());
    scope.spawn(move || {
        if let Some(policy) = policy {
            struct Active(Arc<WorkerStats>);
            impl Drop for Active {
                fn drop(&mut self) {
                    self.0.active.fetch_sub(1, Ordering::Relaxed);
                }
            }
            let current = policy.stats.active.fetch_add(1, Ordering::Relaxed) + 1;
            policy.stats.peak.fetch_max(current, Ordering::Relaxed);
            policy.stats.total.fetch_add(1, Ordering::Relaxed);
            let _active = Active(policy.stats.clone());
            with_policy(policy.threads, policy.stats, f)
        } else {
            f()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_policy_survives_nested_calls_and_unwind() {
        let stats = Arc::new(WorkerStats::default());
        with_policy(2, stats.clone(), || {
            assert_eq!(thread_count(), 2);
            let _ = std::panic::catch_unwind(|| with_policy(1, stats.clone(), || panic!("test")));
            assert_eq!(thread_count(), 2);
            std::thread::scope(|s| {
                spawn(s, || assert_eq!(thread_count(), 2));
            });
        });
        assert_eq!(stats.total(), 1);
        assert_eq!(stats.peak(), 1);
    }

    #[test]
    fn concurrent_contexts_are_independent() {
        std::thread::scope(|s| {
            for n in [1, 3, 5] {
                s.spawn(move || {
                    with_policy(n, Arc::default(), || {
                        for _ in 0..100 {
                            assert_eq!(thread_count(), n);
                        }
                    })
                });
            }
        });
    }
}
