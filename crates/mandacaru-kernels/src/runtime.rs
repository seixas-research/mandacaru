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
    /// The calculation contexts this thread runs for, innermost last ([`enter_scope`]); a
    /// thread started by [`spawn`] inherits its parent's.
    static SCOPES: RefCell<Vec<ContextScope>> = const { RefCell::new(Vec::new()) };
}

/// One calculation context a thread runs for: its identity, the workers a kernel called from
/// the thread may take, and whether the thread runs one lane of a parallel region of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextScope {
    pub id: usize,
    pub width: usize,
    pub lane: bool,
}

/// Leaves the scope [`enter_scope`] entered, on drop (and on unwind).
pub struct ScopeGuard(usize);
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPES.with(|scopes| {
            let entry = scopes.borrow_mut().pop();
            debug_assert_eq!(entry.map(|scope| scope.id), Some(self.0));
        });
    }
}

/// Records on this thread that it runs for the context `id` with `width` workers (`lane`: as one
/// lane of a parallel region), until the guard drops.
pub fn enter_scope(id: usize, width: usize, lane: bool) -> ScopeGuard {
    SCOPES.with(|scopes| scopes.borrow_mut().push(ContextScope { id, width, lane }));
    ScopeGuard(id)
}

/// The innermost scope of the context `id` on this thread, if the thread runs for it.
pub fn scope_of(id: usize) -> Option<ContextScope> {
    SCOPES.with(|scopes| {
        scopes
            .borrow()
            .iter()
            .rev()
            .find(|scope| scope.id == id)
            .copied()
    })
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
    // The scopes go with the thread: a kernel's worker that calls back into its context sees the
    // width and the lane of the thread that started it, and re-enters instead of waiting for it.
    let scopes = SCOPES.with(|s| s.borrow().clone());
    scope.spawn(move || {
        SCOPES.with(|s| *s.borrow_mut() = scopes);
        if let Some(policy) = policy {
            as_worker(policy.threads, policy.stats, f)
        } else {
            f()
        }
    })
}

/// Runs `f` on the calling thread as one counted worker of `stats` under a policy of `threads`:
/// what [`spawn`] does in the thread it starts, for a thread that some other executor (a
/// persistent pool) owns. The kernels `f` calls take `threads` workers.
pub fn as_worker<T>(threads: usize, stats: Arc<WorkerStats>, f: impl FnOnce() -> T) -> T {
    struct Active(Arc<WorkerStats>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.active.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let current = stats.active.fetch_add(1, Ordering::Relaxed) + 1;
    stats.peak.fetch_max(current, Ordering::Relaxed);
    stats.total.fetch_add(1, Ordering::Relaxed);
    let _active = Active(stats.clone());
    with_policy(threads, stats, f)
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
    fn a_spawned_worker_inherits_the_scopes_of_its_parent() {
        let stats = Arc::new(WorkerStats::default());
        let _outer = enter_scope(7, 6, false);
        let _lane = enter_scope(7, 2, true);
        with_policy(2, stats, || {
            std::thread::scope(|s| {
                spawn(s, || {
                    assert_eq!(
                        scope_of(7),
                        Some(ContextScope {
                            id: 7,
                            width: 2,
                            lane: true
                        })
                    );
                    assert_eq!(scope_of(8), None);
                });
            });
        });
        drop(_lane);
        assert_eq!(scope_of(7).map(|scope| scope.width), Some(6));
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
