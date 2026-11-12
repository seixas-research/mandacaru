use crate::{math::archive::ScratchDir, product, Error, Result, WorkerStats};
use std::cell::RefCell;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
thread_local! {
    /// The contexts this thread runs for, innermost last, each with the width it runs at: the
    /// processor count on the thread that entered [`Context::run`], a task's share on a pool
    /// thread running one of its tasks ([`Context::map_tasks`]).
    static ACTIVE_CONTEXTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}
struct ActiveContext(usize);
impl ActiveContext {
    fn enter(id: usize, width: usize) -> Self {
        ACTIVE_CONTEXTS.with(|active| active.borrow_mut().push((id, width)));
        Self(id)
    }
}
impl Drop for ActiveContext {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|active| {
            let entry = active.borrow_mut().pop();
            debug_assert_eq!(entry.map(|(id, _)| id), Some(self.0));
        });
    }
}
fn active_width(id: usize) -> Option<usize> {
    ACTIVE_CONTEXTS.with(|active| {
        active
            .borrow()
            .iter()
            .rev()
            .find(|(entry, _)| *entry == id)
            .map(|(_, width)| *width)
    })
}

/// How a calculation's processors are arranged: `kpts` groups of k-point (and spin channel)
/// tasks, each split into `bands` groups of states, each running its kernels on `threads`
/// workers. The processor count is the product. A level a calculation does not have leaves its
/// processors to the levels below it: outside a k-point region every kernel runs on all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub kpts: usize,
    pub bands: usize,
    pub threads: usize,
}
impl Layout {
    /// The processor count, `kpts * bands * threads`.
    pub fn workers(&self) -> usize {
        self.kpts * self.bands * self.threads
    }
}

/// The level of a [`Layout`] a parallel region distributes over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Independent k-point and spin-channel tasks: `kpts` lanes.
    Kpts,
    /// Independent states inside one task: `bands` lanes.
    Bands,
    /// Independent pieces of one kernel (grid ranges): one lane per worker of the region.
    Threads,
}
impl Level {
    fn name(self) -> &'static str {
        match self {
            Self::Kpts => "kpts",
            Self::Bands => "bands",
            Self::Threads => "threads",
        }
    }
}

/// Size (bytes) at which a calculation's bulk arrays (Bloch orbital stacks, the mixing history,
/// the band matrices of a finished run) leave memory for the scratch folder when the caller sets
/// no threshold of its own.
pub const DEFAULT_OFFLOAD_BYTES: usize = 8 << 20;

#[derive(Debug)]
struct Memory {
    limit: usize,
    offload: usize,
    /// Bytes ever written to the scratch folder by this context's archives.
    archived: AtomicUsize,
    used: AtomicUsize,
    peak: AtomicUsize,
}

/// A shared, calculation-scoped allocation budget, independent of Python.
#[derive(Clone, Debug)]
pub struct Context {
    layout: Layout,
    memory: Arc<Memory>,
    workers: Arc<WorkerStats>,
    execution: Arc<Mutex<()>>,
    scratch: Arc<ScratchDir>,
    /// The persistent worker pool of the threaded kernels, built on first use ([`Context::pool`]).
    pool: Arc<std::sync::OnceLock<Option<rayon::ThreadPool>>>,
}

#[derive(Debug)]
pub struct Reservation {
    memory: Arc<Memory>,
    bytes: usize,
}
impl Reservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Release unused workspace capacity while continuously accounting for
    /// storage retained by a result. This operation cannot grow a reservation.
    pub fn shrink_to(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            return Err(Error::Invalid(
                "cannot grow a memory reservation with shrink_to".into(),
            ));
        }
        self.memory
            .used
            .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
        Ok(())
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.memory.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Rust-owned storage. Its budget reservation outlives the creating context.
#[derive(Debug)]
pub struct Buffer<T> {
    data: Vec<T>,
    _reservation: Reservation,
}
impl<T> Buffer<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    /// The storage and its budget reservation, separated so the storage can
    /// move into an owner (an array handed to Python) without a copy while
    /// the caller keeps the bytes charged for as long as it holds the guard.
    pub fn into_parts(self) -> (Vec<T>, Reservation) {
        (self.data, self._reservation)
    }
}

/// The worker count a context takes when none is requested: `MANDACARU_NUM_THREADS`, then
/// `OMP_NUM_THREADS`, then the parallelism the operating system offers this process.
pub fn default_threads() -> usize {
    mandacaru_kernels::runtime::thread_count()
}

/// The hardware threads this process may run on, whatever the environment asks for: the width at
/// which a dense product over a grid still gains from more threads.
pub fn hardware_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The workers each of `workers` concurrent jobs may use: the hardware threads shared out, at
/// least one. Without it every job opens a pool as wide as the machine (seven jobs ran 48 threads
/// each on 12 cores and built a library at a quarter of its speed).
pub fn worker_threads(workers: usize) -> usize {
    (hardware_threads() / workers.max(1)).max(1)
}

impl Context {
    pub fn new(threads: Option<usize>, memory_bytes: usize) -> Result<Self> {
        Self::with_offload(threads, memory_bytes, DEFAULT_OFFLOAD_BYTES)
    }
    /// A context whose bulk arrays of at least `offload_bytes` live on disk (`0`: every one does;
    /// `usize::MAX`: none does, short of the memory-limit rule of [`Context::offloads`]).
    pub fn with_offload(
        threads: Option<usize>,
        memory_bytes: usize,
        offload_bytes: usize,
    ) -> Result<Self> {
        Self::with_offload_and_scratch(threads, memory_bytes, offload_bytes, None)
    }
    /// Select the parent directory for this context's private scratch folder.
    pub fn with_offload_and_scratch(
        threads: Option<usize>,
        memory_bytes: usize,
        offload_bytes: usize,
        scratch_parent: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let threads = threads.unwrap_or_else(default_threads);
        if threads == 0 || threads > mandacaru_kernels::runtime::MAX_THREADS {
            return Err(Error::Invalid("threads must be in 1..1024".into()));
        }
        if memory_bytes == 0 || memory_bytes > isize::MAX as usize {
            return Err(Error::Invalid(
                "memory budget must be positive and addressable".into(),
            ));
        }
        Ok(Self {
            layout: Layout {
                kpts: 1,
                bands: 1,
                threads,
            },
            memory: Arc::new(Memory {
                limit: memory_bytes,
                offload: offload_bytes,
                archived: AtomicUsize::new(0),
                used: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }),
            workers: Arc::default(),
            execution: Arc::default(),
            scratch: Arc::new(ScratchDir::with_base(scratch_parent)),
            pool: Arc::default(),
        })
    }
    /// This context with its processors arranged as `kpts x bands x threads`, `threads` being the
    /// worker count it was created with. Call it before the context is shared: its pool is
    /// rebuilt for the new processor count.
    pub fn with_layout(mut self, kpts: usize, bands: usize) -> Result<Self> {
        let threads = self.layout.threads;
        let workers = kpts
            .checked_mul(bands)
            .and_then(|n| n.checked_mul(threads))
            .ok_or(Error::Overflow)?;
        if kpts == 0 || bands == 0 || workers > mandacaru_kernels::runtime::MAX_THREADS {
            return Err(Error::Invalid(format!(
                "parallelization kpts={kpts}, bands={bands}, threads={threads} needs every level \
                 at least 1 and at most {} processors in all",
                mandacaru_kernels::runtime::MAX_THREADS
            )));
        }
        self.layout = Layout {
            kpts,
            bands,
            threads,
        };
        self.pool = Arc::default();
        Ok(self)
    }
    /// The calculation's persistent pool of one worker per processor, built on its first parallel
    /// action and shared by every clone of this context; `None` for a one-processor context, or
    /// when the operating system refuses the threads. A kernel then runs on the calling thread,
    /// with the same numbers: every threaded kernel computes each row or block in a fixed order.
    pub(crate) fn pool(&self) -> Option<&rayon::ThreadPool> {
        if self.layout.workers() == 1 {
            return None;
        }
        self.pool
            .get_or_init(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(self.layout.workers())
                    .thread_name(|i| format!("mandacaru-worker-{i}"))
                    .build()
                    .ok()
            })
            .as_ref()
    }
    /// The arrangement of the processors.
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// The workers a kernel called from this thread may use: the processor count outside any
    /// parallel region, the share of one task inside a region of [`Context::map_tasks`] or
    /// [`Context::run_lanes`]. Every kernel sizes its workers by it, so a task never takes more
    /// than its share.
    pub fn width(&self) -> usize {
        active_width(self.id()).unwrap_or(self.layout.workers())
    }
    fn id(&self) -> usize {
        Arc::as_ptr(&self.execution) as usize
    }
    pub fn memory_limit(&self) -> usize {
        self.memory.limit
    }
    /// The size from which bulk arrays are written to the scratch folder.
    pub fn offload_threshold(&self) -> usize {
        self.memory.offload
    }
    /// Whether a bulk array of `bytes` lives on disk: at the context's offload threshold, and
    /// always above an eighth of the memory limit, which no stack may take.
    pub fn offloads(&self, bytes: usize) -> bool {
        bytes >= self.memory.offload || bytes > self.memory.limit / 8
    }
    /// Bytes this context's archives have written to the scratch folder (an evidence counter:
    /// it never decreases when an archive is dropped).
    pub fn archived_bytes(&self) -> usize {
        self.memory.archived.load(Ordering::Relaxed)
    }
    pub(crate) fn note_archived(&self, bytes: usize) {
        self.memory.archived.fetch_add(bytes, Ordering::Relaxed);
    }
    /// Returns the pages the allocator holds free to the operating system (see
    /// [`mandacaru_kernels::heap`]); `false` where the platform has no such call. Live arrays,
    /// the budget and every result are untouched.
    pub fn release_freed_memory(&self) -> bool {
        mandacaru_kernels::heap::release_freed_memory()
    }
    /// The calculation's scratch folder (created by the first archive, removed with the context).
    pub fn scratch(&self) -> &Arc<ScratchDir> {
        &self.scratch
    }
    pub fn memory_used(&self) -> usize {
        self.memory.used.load(Ordering::Acquire)
    }
    pub fn memory_peak(&self) -> usize {
        self.memory.peak.load(Ordering::Acquire)
    }
    pub fn workers_peak(&self) -> usize {
        self.workers.peak()
    }
    pub fn workers_total(&self) -> usize {
        self.workers.total()
    }
    /// Runs `f` as this context's calculation: one at a time per context, the kernels inside on
    /// [`Context::width`] workers. Re-entered from a thread that already runs for this context
    /// (a nested call, or a task of a parallel region) it keeps that thread's width.
    pub(crate) fn run<T>(&self, f: impl FnOnce() -> T) -> T {
        let id = self.id();
        if let Some(width) = active_width(id) {
            return mandacaru_kernels::runtime::with_policy(width, self.workers.clone(), f);
        }
        let _execution = self.execution.lock().unwrap_or_else(|e| e.into_inner());
        let width = self.layout.workers();
        let _active = ActiveContext::enter(id, width);
        mandacaru_kernels::runtime::with_policy(width, self.workers.clone(), f)
    }
    /// The lanes a region of `level` over `count` independent pieces runs on: the level's size in
    /// the layout (for [`Level::Threads`], the calling thread's width), at most `count`, and one
    /// when the context has no pool.
    pub fn lanes(&self, level: Level, count: usize) -> usize {
        if self.pool().is_none() {
            return 1;
        }
        self.configured_lanes(level).min(count).max(1)
    }
    fn configured_lanes(&self, level: Level) -> usize {
        match level {
            Level::Kpts => self.layout.kpts,
            Level::Bands => self.layout.bands,
            Level::Threads => self.width(),
        }
    }
    /// Refuses a region of `lanes` lanes of `level` when `per_lane` bytes each do not fit in the
    /// free budget, naming the largest lane count that does: a layout never silently runs fewer
    /// lanes than it asked for.
    pub fn check_lanes(&self, level: Level, lanes: usize, per_lane: usize) -> Result<()> {
        let needed = product(&[lanes, per_lane])?;
        let available = self.memory.limit.saturating_sub(self.memory_used());
        if needed <= available || lanes <= 1 {
            return Ok(());
        }
        let fits = (available / per_lane.max(1)).max(1);
        Err(Error::LayoutBudget(format!(
            "the parallel layout {}={lanes} needs {per_lane} bytes per task ({needed} in all) and \
             {available} bytes of the memory budget are free; the largest {} that fits is {fits}",
            level.name(),
            level.name()
        )))
    }
    /// Runs `f(lane, state)` once per entry of `states`, the entries on separate workers of the
    /// context's pool (on the calling thread without one, or with one entry). Each lane runs as
    /// one counted worker for this context with the width `width() / lanes of the level` (at
    /// least one), so a nested [`Context::run`] re-enters instead of waiting for the context and
    /// the kernels it calls take only that share. Returns the first error by lane order.
    pub fn run_lanes<S: Send>(
        &self,
        level: Level,
        states: &mut [S],
        f: impl Fn(usize, &mut S) -> Result<()> + Sync,
    ) -> Result<()> {
        // The region is one calculation of this context: the calling thread holds it (or already
        // runs for it) while the lanes run.
        self.run(|| {
            let lanes = states.len();
            let pool = self.pool().filter(|_| lanes > 1);
            let Some(pool) = pool else {
                for (lane, state) in states.iter_mut().enumerate() {
                    f(lane, state)?;
                }
                return Ok(());
            };
            let width = (self.width() / self.configured_lanes(level).max(1)).max(1);
            let id = self.id();
            let stats = self.workers.clone();
            use rayon::prelude::*;
            let outcomes: Vec<Result<()>> = pool.install(|| {
                states
                    .par_iter_mut()
                    .enumerate()
                    .map(|(lane, state)| {
                        let _active = ActiveContext::enter(id, width);
                        mandacaru_kernels::runtime::as_worker(width, stats.clone(), || {
                            f(lane, state)
                        })
                    })
                    .collect()
            });
            outcomes.into_iter().collect()
        })
    }
    /// `task(state, i)` for every `i` in `0..count`, the results in index order whatever lane ran
    /// them: the lanes (one per entry of `states`, see [`Context::run_lanes`]) take the next
    /// index as they finish one. A task is one serial computation on its lane's state, so a
    /// result depends neither on the lane count nor on which lane computed it. Returns the error
    /// of the smallest failing index.
    pub fn map_tasks<S: Send, T: Send>(
        &self,
        level: Level,
        states: &mut [S],
        count: usize,
        task: impl Fn(&mut S, usize) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        use std::sync::atomic::AtomicBool;
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let mut done: Vec<Vec<(usize, Result<T>)>> =
            (0..states.len()).map(|_| Vec::new()).collect();
        let mut lanes: Vec<(&mut S, &mut Vec<(usize, Result<T>)>)> =
            states.iter_mut().zip(done.iter_mut()).collect();
        self.run_lanes(level, &mut lanes, |_, (state, out)| {
            while !failed.load(Ordering::Relaxed) {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= count {
                    break;
                }
                let result = task(state, i);
                if result.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
                out.push((i, result));
            }
            Ok(())
        })?;
        drop(lanes);
        let mut slots: Vec<Option<Result<T>>> = (0..count).map(|_| None).collect();
        for (i, result) in done.into_iter().flatten() {
            slots[i] = Some(result);
        }
        let mut out = Vec::with_capacity(count);
        for slot in slots {
            match slot {
                Some(result) => out.push(result?),
                None => {
                    return Err(Error::Kernel(
                        "a parallel task was not run after another one failed".into(),
                    ))
                }
            }
        }
        Ok(out)
    }
    pub fn reserve(&self, bytes: usize) -> Result<Reservation> {
        let mut used = self.memory.used.load(Ordering::Acquire);
        loop {
            let next = used.checked_add(bytes).ok_or(Error::Overflow)?;
            if next > self.memory.limit {
                return Err(Error::Budget {
                    requested: bytes,
                    available: self.memory.limit - used,
                });
            }
            match self.memory.used.compare_exchange_weak(
                used,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.memory.peak.fetch_max(next, Ordering::Relaxed);
                    break;
                }
                Err(now) => used = now,
            }
        }
        Ok(Reservation {
            memory: self.memory.clone(),
            bytes,
        })
    }
    pub fn allocate<T: Default + Clone>(&self, len: usize) -> Result<Buffer<T>> {
        let bytes = product(&[len, size_of::<T>()])?;
        if bytes > isize::MAX as usize {
            return Err(Error::Overflow);
        }
        let reservation = self.reserve(bytes)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
        data.resize(len, T::default());
        Ok(Buffer {
            data,
            _reservation: reservation,
        })
    }
    pub fn copy_slice<T: Default + Clone + Copy>(&self, values: &[T]) -> Result<Buffer<T>> {
        let mut out = self.allocate(values.len())?;
        out.as_mut_slice().copy_from_slice(values);
        Ok(out)
    }
}
