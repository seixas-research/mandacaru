use crate::{math::archive::ScratchDir, product, Error, Result, WorkerStats};
use std::cell::RefCell;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
thread_local! {static ACTIVE_CONTEXTS: RefCell<Vec<usize>>=const {RefCell::new(Vec::new())};}
struct ActiveContext(usize);
impl Drop for ActiveContext {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|active| {
            let id = active.borrow_mut().pop();
            debug_assert_eq!(id, Some(self.0));
        });
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
    threads: usize,
    memory: Arc<Memory>,
    workers: Arc<WorkerStats>,
    execution: Arc<Mutex<()>>,
    scratch: Arc<ScratchDir>,
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
            threads,
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
        })
    }
    pub fn threads(&self) -> usize {
        self.threads
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
    pub(crate) fn run<T>(&self, f: impl FnOnce() -> T) -> T {
        let id = Arc::as_ptr(&self.execution) as usize;
        if ACTIVE_CONTEXTS.with(|active| active.borrow().contains(&id)) {
            return mandacaru_kernels::runtime::with_policy(self.threads, self.workers.clone(), f);
        }
        let _execution = self.execution.lock().unwrap_or_else(|e| e.into_inner());
        ACTIVE_CONTEXTS.with(|active| active.borrow_mut().push(id));
        let _active = ActiveContext(id);
        mandacaru_kernels::runtime::with_policy(self.threads, self.workers.clone(), f)
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
