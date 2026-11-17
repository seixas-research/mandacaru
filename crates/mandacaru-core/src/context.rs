use crate::{math::archive::ScratchDir, product, Error, Result, WorkerStats};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
// Which contexts a thread runs for, with the width it runs at and whether it is a lane of a
// parallel region, are the runtime's scopes (`mandacaru_kernels::runtime::enter_scope`): a
// kernel's own scoped workers inherit them, so they see their parent's share, not the
// processor count, and re-enter the context instead of waiting for it.
use mandacaru_kernels::runtime::{enter_scope, scope_of};

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
    /// The `auto` layout (P7): the processors, `kpts = bands = 1` in `layout`, spread over the
    /// k-point tasks of each region (see [`Context::with_auto_layout`]).
    auto: bool,
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

/// Storage whose capacity is laid out in virtual memory at once but charged to the budget as it
/// is written: the zero pages are mapped, not touched, so only what the owner writes is resident
/// (a factorization that does not know its rank in advance addresses a row per pair of the
/// largest rank it could reach and fills the rows' first entries). [`Lazy::charge`] makes the
/// bytes about to be written part of the reservation, [`Lazy::finish`] returns the first
/// `len` entries as a [`Buffer`] with exactly the bytes charged.
pub struct Lazy<T> {
    context: Context,
    data: Vec<T>,
    reservation: Reservation,
}
impl<T> Lazy<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    /// Charges `entries` more entries to the budget (the caller writes them next).
    pub fn charge(&mut self, entries: usize) -> Result<()> {
        let mut grown = self.context.reserve(product(&[entries, size_of::<T>()])?)?;
        self.reservation.bytes += std::mem::take(&mut grown.bytes);
        Ok(())
    }
    /// The first `len` entries, with the capacity beyond them released; the charge is cut to
    /// `len` entries.
    pub fn finish(mut self, len: usize) -> Result<Buffer<T>> {
        self.data.truncate(len);
        self.data.shrink_to_fit();
        self.reservation
            .shrink_to(product(&[len, size_of::<T>()])?)?;
        Ok(Buffer {
            data: self.data,
            _reservation: self.reservation,
        })
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
            auto: false,
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
        self.auto = false;
        Ok(self)
    }
    /// This context with the `auto` layout (P7), for a calculation that names no layout: every
    /// k-point region runs on as many lanes as it has tasks, up to the processor count, each lane
    /// with an equal share of the processors, and with fewer lanes when the free budget holds
    /// fewer (never a refusal: the same calculation on one lane runs). Results do not depend on
    /// the lanes (every task is a serial computation, read in a fixed order), so this layout gives
    /// the bits of any explicit one. Only for a context of one k-point and one band group.
    pub fn with_auto_layout(mut self) -> Result<Self> {
        if self.layout.kpts != 1 || self.layout.bands != 1 {
            return Err(Error::Invalid(
                "the auto layout spreads the threads over the k-point tasks itself; it takes no \
                 explicit kpts or bands"
                    .into(),
            ));
        }
        self.auto = true;
        Ok(self)
    }
    /// Whether this context runs the `auto` layout ([`Context::with_auto_layout`]).
    pub fn auto_layout(&self) -> bool {
        self.auto
    }
    /// The calculation's persistent pool of one worker per processor, built on its first parallel
    /// action and shared by every clone of this context; `None` for a one-processor context, or
    /// when the operating system refuses the threads. A kernel then runs on the calling thread,
    /// with the same numbers: every threaded kernel computes each row or block in a fixed order.
    ///
    /// Inside a lane of a parallel region (and in the workers its kernels start) there is no
    /// pool: a row or block region opened there runs on the lane's own thread, since the pool's
    /// threads carry no share of it. The lanes of a nested level are started by
    /// [`Context::run_lanes`] itself, which gives each its share.
    pub(crate) fn pool(&self) -> Option<&rayon::ThreadPool> {
        if scope_of(self.id()).is_some_and(|scope| scope.lane) {
            return None;
        }
        self.shared_pool()
    }
    fn shared_pool(&self) -> Option<&rayon::ThreadPool> {
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
        scope_of(self.id()).map_or(self.layout.workers(), |scope| scope.width)
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
        if let Some(width) = scope_of(id).map(|scope| scope.width) {
            return mandacaru_kernels::runtime::with_policy(width, self.workers.clone(), f);
        }
        let _execution = self.execution.lock().unwrap_or_else(|e| e.into_inner());
        let width = self.layout.workers();
        let _active = enter_scope(id, width, false);
        mandacaru_kernels::runtime::with_policy(width, self.workers.clone(), f)
    }
    /// The lanes a region of `level` over `count` independent pieces runs on: the level's size in
    /// the layout (for [`Level::Threads`], the calling thread's width), at most `count`, and one
    /// when the context has no pool.
    pub fn lanes(&self, level: Level, count: usize) -> usize {
        if self.shared_pool().is_none() {
            return 1;
        }
        self.configured_lanes(level).min(count).max(1)
    }
    fn configured_lanes(&self, level: Level) -> usize {
        match level {
            Level::Kpts if self.auto => self.width(),
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
    /// The lanes a region of `level` runs on when `per_lane` bytes each must fit in the free
    /// budget: `lanes` itself, refused as [`Context::check_lanes`] refuses it, except for the
    /// k-point level of the `auto` layout, which takes the largest count that fits (at least one)
    /// in the smaller of the context's budget and the machine's installed memory.
    pub fn fit_lanes(&self, level: Level, lanes: usize, per_lane: usize) -> Result<usize> {
        if self.auto && level == Level::Kpts {
            // The auto budget: the context's, and never more than the machine has installed.
            let installed = mandacaru_kernels::heap::installed_memory_bytes()
                .and_then(|b| usize::try_from(b).ok())
                .unwrap_or(usize::MAX);
            let available = self
                .memory
                .limit
                .min(installed)
                .saturating_sub(self.memory_used());
            return Ok(lanes.min(available / per_lane.max(1)).max(1));
        }
        self.check_lanes(level, lanes, per_lane)?;
        Ok(lanes)
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
            let pool = self.shared_pool().filter(|_| lanes > 1);
            let Some(pool) = pool else {
                for (lane, state) in states.iter_mut().enumerate() {
                    f(lane, state)?;
                }
                return Ok(());
            };
            // An auto k-point region shares the processors among the lanes it has; an explicit
            // level among the lanes the layout names.
            let divisor = if self.auto && level == Level::Kpts {
                lanes
            } else {
                self.configured_lanes(level)
            };
            let width = (self.width() / divisor.max(1)).max(1);
            let id = self.id();
            let stats = self.workers.clone();
            use rayon::prelude::*;
            let outcomes: Vec<Result<()>> = pool.install(|| {
                states
                    .par_iter_mut()
                    .enumerate()
                    .map(|(lane, state)| {
                        let _active = enter_scope(id, width, true);
                        mandacaru_kernels::runtime::as_worker(width, stats.clone(), || {
                            f(lane, state)
                        })
                    })
                    .collect()
            });
            outcomes.into_iter().collect()
        })
    }
    /// `task(i)` for every `i` in `0..count`, independent serial computations, the results in
    /// index order: on the layout's k-point lanes when it has more than one, else one lane per
    /// worker of the calling thread's width (the post-processing of a converged crystal, P6). A
    /// result depends neither on the lane count nor on the lane that ran it.
    pub fn map_independent<T: Send>(
        &self,
        count: usize,
        task: impl Fn(usize) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        let level = if self.layout.kpts > 1 || self.auto {
            Level::Kpts
        } else {
            Level::Threads
        };
        let lanes = self.lanes(level, count);
        self.map_tasks(level, &mut vec![(); lanes], count, |_, i| task(i))
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
    /// `len` zero entries mapped but not charged ([`Lazy`]; `vec![0.; len]` is a zeroed allocation,
    /// whose pages the operating system maps on first write); the owner charges what it writes.
    pub fn allocate_lazily(&self, len: usize) -> Result<Lazy<f64>> {
        let bytes = product(&[len, size_of::<f64>()])?;
        if bytes > isize::MAX as usize {
            return Err(Error::Overflow);
        }
        Ok(Lazy {
            context: self.clone(),
            data: vec![0f64; len],
            reservation: self.reserve(0)?,
        })
    }
    pub fn copy_slice<T: Default + Clone + Copy>(&self, values: &[T]) -> Result<Buffer<T>> {
        let mut out = self.allocate(values.len())?;
        out.as_mut_slice().copy_from_slice(values);
        Ok(out)
    }
}
