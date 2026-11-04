//! Immutable calculation-owned scratch arrays with bounded, budgeted reads.
//!
//! An archive is written once, on creation, into the calculation's scratch directory and is then
//! read by positional I/O only: no seek, no per-read `open`, and no heap allocation (a read goes
//! through a fixed stack block into the caller's slice). The file lives in a per-calculation
//! folder under the configured scratch directory or OS temporary directory ([`ScratchDir`]:
//! `TMPDIR` is honored and folders named `.mandacaru` are refused) and is removed with its owner.
//!
//! Descriptors are bounded: an archive keeps its file open for its whole life while fewer than
//! [`MAX_OPEN_HANDLES`] archives of the process do, and opens the file for the duration of a
//! single read otherwise, so a hundred-k-point calculation cannot exhaust the descriptor limit of
//! the process.
use crate::{product, Buffer, Context, Error, Reservation, Result};
use num_complex::{Complex32, Complex64};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    marker::PhantomData,
    path::{Component, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

/// A fixed-size element of a scratch file, stored little-endian behind the file header.
pub trait Scalar: Copy + Default + Send + Sync + 'static {
    const BYTES: usize;
    /// Names the element type in the file header: a record of another type is refused.
    const TAG: [u8; 4];
    fn encode(self, bytes: &mut [u8]);
    fn decode(bytes: &[u8]) -> Self;
}
impl Scalar for Complex64 {
    const BYTES: usize = 16;
    const TAG: [u8; 4] = *b"CF64";
    fn encode(self, bytes: &mut [u8]) {
        bytes[..8].copy_from_slice(&self.re.to_le_bytes());
        bytes[8..].copy_from_slice(&self.im.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        Self::new(
            f64::from_le_bytes(bytes[..8].try_into().unwrap()),
            f64::from_le_bytes(bytes[8..].try_into().unwrap()),
        )
    }
}
impl Scalar for Complex32 {
    const BYTES: usize = 8;
    const TAG: [u8; 4] = *b"CF32";
    fn encode(self, bytes: &mut [u8]) {
        bytes[..4].copy_from_slice(&self.re.to_le_bytes());
        bytes[4..].copy_from_slice(&self.im.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        Self::new(
            f32::from_le_bytes(bytes[..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..].try_into().unwrap()),
        )
    }
}
impl Scalar for f64 {
    const BYTES: usize = 8;
    const TAG: [u8; 4] = *b"RF64";
    fn encode(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl Scalar for f32 {
    const BYTES: usize = 4;
    const TAG: [u8; 4] = *b"RF32";
    fn encode(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl Scalar for u64 {
    const BYTES: usize = 8;
    const TAG: [u8; 4] = *b"IU64";
    fn encode(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl Scalar for u8 {
    const BYTES: usize = 1;
    const TAG: [u8; 4] = *b"BY08";
    fn encode(self, bytes: &mut [u8]) {
        bytes[0] = self;
    }
    fn decode(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

/// Header of every scratch file: `MCAR`, the element tag, the element count (little-endian).
const HEADER: usize = 16;
/// Bytes moved per positional call (a stack block, never heap).
const BLOCK: usize = 32 * 1024;
/// Archives of the process that may hold their file open at once.
pub const MAX_OPEN_HANDLES: usize = 96;

static NEXT: AtomicU64 = AtomicU64::new(0);
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
static OPEN_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// Scratch files whose descriptor is currently held open by an archive of this process.
pub fn open_handles() -> usize {
    OPEN_HANDLES.load(Ordering::Acquire)
}

fn io(error: std::io::Error) -> Error {
    Error::Invalid(format!("calculation scratch array: {error}"))
}

/// The scratch folder of one calculation: created by the first archive under
/// its configured parent or `std::env::temp_dir()` (a relative parent is refused), never under a
/// folder named `.mandacaru`, and removed when its last owner (the [`Context`] and
/// every archive) is gone.
#[derive(Debug, Default)]
pub struct ScratchDir {
    path: Mutex<Option<PathBuf>>,
    base: Option<PathBuf>,
}
impl ScratchDir {
    pub(crate) fn with_base(base: Option<PathBuf>) -> Self {
        Self {
            path: Mutex::new(None),
            base,
        }
    }
    /// The folder, created on first use.
    pub fn path(&self) -> Result<PathBuf> {
        let mut guard = self
            .path
            .lock()
            .map_err(|_| Error::Invalid("scratch folder lock was poisoned".into()))?;
        if let Some(path) = guard.as_ref() {
            return Ok(path.clone());
        }
        let base = self.base.clone().unwrap_or_else(std::env::temp_dir);
        if !base.is_absolute() {
            return Err(Error::Invalid(format!(
                "scratch arrays need an absolute temporary directory, not {}: set TMPDIR to one",
                base.display()
            )));
        }
        if base
            .components()
            .any(|part| matches!(part, Component::Normal(name) if name == ".mandacaru"))
        {
            return Err(Error::Invalid(format!(
                "scratch arrays are never written under a .mandacaru folder ({})",
                base.display()
            )));
        }
        std::fs::create_dir_all(&base).map_err(io)?;
        let path = loop {
            let path = base.join(format!(
                "mandacaru-scratch-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io(e)),
            }
        };
        *guard = Some(path.clone());
        Ok(path)
    }
    /// The folder if some archive has created it.
    pub fn existing(&self) -> Option<PathBuf> {
        self.path.lock().ok().and_then(|path| path.clone())
    }
}
impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Some(path) = self.path.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            // Archives remove their own files; a folder that is not empty is not ours to clear.
            let _ = std::fs::remove_dir(path);
        }
    }
}

/// A scratch file held open for the life of its archive; counted against [`MAX_OPEN_HANDLES`].
struct Held {
    file: File,
    #[cfg(not(unix))]
    cursor: Mutex<()>,
}
impl Held {
    fn acquire(file: File) -> Option<Self> {
        let mut open = OPEN_HANDLES.load(Ordering::Acquire);
        loop {
            if open >= MAX_OPEN_HANDLES {
                return None;
            }
            match OPEN_HANDLES.compare_exchange_weak(
                open,
                open + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Self {
                        file,
                        #[cfg(not(unix))]
                        cursor: Mutex::new(()),
                    })
                }
                Err(now) => open = now,
            }
        }
    }
}
impl Drop for Held {
    fn drop(&mut self) {
        OPEN_HANDLES.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(unix)]
fn read_at(file: &File, offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(bytes, offset)
}
#[cfg(not(unix))]
fn read_at(file: &File, offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(bytes)
}

#[cfg(unix)]
fn write_at(file: &File, offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(bytes, offset)
}
#[cfg(not(unix))]
fn write_at(file: &File, offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)
}

/// An immutable array of `T` on disk.
pub struct Archive<T> {
    _scratch: Arc<ScratchDir>,
    held: Option<Held>,
    path: PathBuf,
    length: usize,
    _scalar: PhantomData<T>,
    _memory: Reservation,
}
impl<T> Drop for Archive<T> {
    fn drop(&mut self) {
        self.held.take();
        let _ = std::fs::remove_file(&self.path);
    }
}
impl<T: Scalar> Archive<T> {
    pub fn create(
        ctx: &Context,
        length: usize,
        mut value: impl FnMut(usize) -> Result<T>,
    ) -> Result<Self> {
        if T::BYTES == 0 || T::BYTES > BLOCK {
            return Err(Error::Invalid("invalid scratch scalar size".into()));
        }
        product(&[length, T::BYTES])?
            .checked_add(HEADER)
            .ok_or(Error::Overflow)?;
        let memory = ctx.reserve(512)?;
        let scratch = ctx.scratch().clone();
        let directory = scratch.path()?;
        let (mut file, path) = loop {
            let path = directory.join(format!("array-{}.bin", NEXT.fetch_add(1, Ordering::Relaxed)));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => break (file, path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io(e)),
            }
        };
        // Install the owner before writing: partial and failed writes clean up.
        let mut out = Self {
            _scratch: scratch,
            held: None,
            path,
            length,
            _scalar: PhantomData,
            _memory: memory,
        };
        let mut header = [0u8; HEADER];
        header[..4].copy_from_slice(b"MCAR");
        header[4..8].copy_from_slice(&T::TAG);
        header[8..].copy_from_slice(&(length as u64).to_le_bytes());
        file.write_all(&header).map_err(io)?;
        let mut bytes = [0u8; BLOCK];
        let block = BLOCK / T::BYTES;
        for start in (0..length).step_by(block) {
            let count = (length - start).min(block);
            for i in 0..count {
                value(start + i)?.encode(&mut bytes[i * T::BYTES..(i + 1) * T::BYTES]);
            }
            file.write_all(&bytes[..count * T::BYTES]).map_err(io)?;
        }
        file.flush().map_err(io)?;
        // The writer's descriptor is kept when the process has budget for it (a hot array is
        // read many times) and closed otherwise.
        out.held = Held::acquire(file);
        ctx.note_archived(length * T::BYTES);
        Ok(out)
    }
    /// Create an empty archive of the given length, preallocating the required disk file.
    pub fn create_empty(ctx: &Context, length: usize) -> Result<Self> {
        if T::BYTES == 0 || T::BYTES > BLOCK {
            return Err(Error::Invalid("invalid scratch scalar size".into()));
        }
        let total_bytes = product(&[length, T::BYTES])?
            .checked_add(HEADER)
            .ok_or(Error::Overflow)?;
        let memory = ctx.reserve(512)?;
        let scratch = ctx.scratch().clone();
        let directory = scratch.path()?;
        let (mut file, path) = loop {
            let path = directory.join(format!("array-{}.bin", NEXT.fetch_add(1, Ordering::Relaxed)));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => break (file, path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io(e)),
            }
        };
        let mut out = Self {
            _scratch: scratch,
            held: None,
            path,
            length,
            _scalar: PhantomData,
            _memory: memory,
        };
        let mut header = [0u8; HEADER];
        header[..4].copy_from_slice(b"MCAR");
        header[4..8].copy_from_slice(&T::TAG);
        header[8..].copy_from_slice(&(length as u64).to_le_bytes());
        file.write_all(&header).map_err(io)?;
        file.set_len(total_bytes as u64).map_err(io)?;
        file.flush().map_err(io)?;
        out.held = Held::acquire(file);
        ctx.note_archived(length * T::BYTES);
        Ok(out)
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    /// Whether this archive holds its descriptor open (see [`MAX_OPEN_HANDLES`]).
    pub fn holds_descriptor(&self) -> bool {
        self.held.is_some()
    }
    fn read_bytes(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        match &self.held {
            #[cfg(unix)]
            Some(held) => read_at(&held.file, offset, bytes).map_err(io),
            #[cfg(not(unix))]
            Some(held) => {
                let _cursor = held.cursor.lock().map_err(|_| {
                    Error::Invalid("scratch array lock was poisoned".into())
                })?;
                read_at(&held.file, offset, bytes).map_err(io)
            }
            None => {
                let file = File::open(&self.path).map_err(io)?;
                read_at(&file, offset, bytes).map_err(io)
            }
        }
    }
    fn write_bytes(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        match &self.held {
            #[cfg(unix)]
            Some(held) => write_at(&held.file, offset, bytes).map_err(io),
            #[cfg(not(unix))]
            Some(held) => {
                let _cursor = held.cursor.lock().map_err(|_| {
                    Error::Invalid("scratch array lock was poisoned".into())
                })?;
                write_at(&held.file, offset, bytes).map_err(io)
            }
            None => {
                let file = OpenOptions::new()
                    .write(true)
                    .open(&self.path)
                    .map_err(io)?;
                write_at(&file, offset, bytes).map_err(io)
            }
        }
    }
    /// Positional write of `values` into elements `start..start + values.len()`.
    pub fn write_range(&self, start: usize, values: &[T]) -> Result<()> {
        if start
            .checked_add(values.len())
            .is_none_or(|end| end > self.length)
        {
            return Err(Error::Invalid("scratch write range is out of bounds".into()));
        }
        if values.is_empty() {
            return Ok(());
        }
        let mut offset = u64::try_from(
            product(&[start, T::BYTES])?
                .checked_add(HEADER)
                .ok_or(Error::Overflow)?,
        )
        .map_err(|_| Error::Overflow)?;
        let mut bytes = [0u8; BLOCK];
        let block = BLOCK / T::BYTES;
        for chunk in values.chunks(block) {
            let want = chunk.len() * T::BYTES;
            for (i, &v) in chunk.iter().enumerate() {
                v.encode(&mut bytes[i * T::BYTES..(i + 1) * T::BYTES]);
            }
            self.write_bytes(offset, &bytes[..want])?;
            offset += want as u64;
        }
        Ok(())
    }
    /// Flush any pending writes to disk.
    pub fn finish(&self) -> Result<()> {
        match &self.held {
            Some(held) => held.file.sync_data().map_err(io),
            None => Ok(()),
        }
    }
    /// Fills `output` with the elements `start..start + output.len()`, converted by `convert`.
    /// Allocates nothing: the bytes pass through a stack block.
    pub fn read_range_with<U>(
        &self,
        start: usize,
        output: &mut [U],
        convert: impl Fn(T) -> U,
    ) -> Result<()> {
        if start
            .checked_add(output.len())
            .is_none_or(|end| end > self.length)
        {
            return Err(Error::Invalid("scratch read range is out of bounds".into()));
        }
        let mut offset = u64::try_from(
            product(&[start, T::BYTES])?
                .checked_add(HEADER)
                .ok_or(Error::Overflow)?,
        )
        .map_err(|_| Error::Overflow)?;
        let mut bytes = [0u8; BLOCK];
        for values in output.chunks_mut(BLOCK / T::BYTES) {
            let want = values.len() * T::BYTES;
            self.read_bytes(offset, &mut bytes[..want])?;
            for (to, from) in values.iter_mut().zip(bytes[..want].chunks_exact(T::BYTES)) {
                *to = convert(T::decode(from));
            }
            offset += want as u64;
        }
        Ok(())
    }
    /// Fills `output` with the elements `start..start + output.len()`.
    pub fn read_range_into(&self, start: usize, output: &mut [T]) -> Result<()> {
        self.read_range_with(start, output, |v| v)
    }
    pub fn read_into<U>(&self, output: &mut [U], convert: impl Fn(T) -> U) -> Result<()> {
        if output.len() != self.length {
            return Err(Error::Invalid("scratch read dimensions disagree".into()));
        }
        self.read_range_with(0, output, convert)
    }
    pub fn read(&self, ctx: &Context) -> Result<Buffer<T>> {
        let mut values = ctx.allocate::<T>(self.length)?;
        self.read_into(values.as_mut_slice(), |v| v)?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The descriptor count is process-wide: the tests of this module take turns.
    static SERIAL: Mutex<()> = Mutex::new(());
    #[test]
    fn archive_reads_are_exact_bounded_and_removed_with_the_owner() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = Context::new(Some(1), 32 << 10).unwrap();
        let path;
        {
            let expected: Vec<_> = (0..1501)
                .map(|i| Complex64::new(i as f64 / 7., -(i as f64) / 3.))
                .collect();
            let archive = Archive::create(&ctx, expected.len(), |i| Ok(expected[i])).unwrap();
            path = archive.path.clone();
            assert_eq!(ctx.memory_used(), 512);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
            {
                let guard = ctx.reserve(10 << 10).unwrap();
                assert!(matches!(archive.read(&ctx), Err(Error::Budget { .. })));
                drop(guard);
            }
            let read = archive.read(&ctx).unwrap();
            assert_eq!(read.as_slice(), expected);
            drop(read);
            let mut partial = vec![Complex64::default(); 333];
            archive.read_range_into(107, &mut partial).unwrap();
            assert_eq!(partial, expected[107..440]);
            assert!(archive.read_range_into(1490, &mut partial).is_err());
            assert_eq!(ctx.memory_used(), 512);
            // A damaged scratch record must surface an I/O error, never zeros.
            OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(HEADER as u64 + 8)
                .unwrap();
            assert!(archive.read(&ctx).is_err());
        }
        assert!(!path.exists());
        assert_eq!(ctx.memory_used(), 0);
    }
    #[test]
    fn every_scalar_round_trips_with_its_own_tag() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let reals: Vec<f64> = (0..777).map(|i| (i as f64).sin() * 1e3).collect();
        let a = Archive::create(&ctx, reals.len(), |i| Ok(reals[i])).unwrap();
        let mut got = vec![0.; reals.len()];
        a.read_range_into(0, &mut got).unwrap();
        assert_eq!(got, reals);
        let singles: Vec<f32> = (0..777).map(|i| (i as f32).cos()).collect();
        let b = Archive::create(&ctx, singles.len(), |i| Ok(singles[i])).unwrap();
        let mut got = vec![0f32; singles.len()];
        b.read_range_into(5, &mut got[..700]).unwrap();
        assert_eq!(got[..700], singles[5..705]);
        let indices: Vec<u64> = (0..5000u64).map(|i| i * 0x9e37_79b9 + 3).collect();
        let c = Archive::create(&ctx, indices.len(), |i| Ok(indices[i])).unwrap();
        let mut got = vec![0u64; 123];
        c.read_range_into(4000, &mut got).unwrap();
        assert_eq!(got, indices[4000..4123]);
        let text: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
        let d = Archive::create(&ctx, text.len(), |i| Ok(text[i])).unwrap();
        let mut got = vec![0u8; 37_000];
        d.read_range_into(33_000, &mut got).unwrap();
        assert_eq!(got, text[33_000..70_000]);
        // The header names the element type: each tag is its own.
        let tags = [
            <Complex64 as Scalar>::TAG,
            <Complex32 as Scalar>::TAG,
            <f64 as Scalar>::TAG,
            <f32 as Scalar>::TAG,
            <u64 as Scalar>::TAG,
            <u8 as Scalar>::TAG,
        ];
        for (i, a) in tags.iter().enumerate() {
            assert!(tags[i + 1..].iter().all(|b| a != b));
        }
        let mut head = [0u8; HEADER];
        std::io::Read::read_exact(&mut File::open(&a.path).unwrap(), &mut head).unwrap();
        assert_eq!(&head[..8], b"MCARRF64");
        assert_eq!(u64::from_le_bytes(head[8..].try_into().unwrap()), 777);
    }
    #[test]
    fn descriptors_are_bounded_and_every_archive_still_reads() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let before = open_handles();
        {
            let arrays: Vec<_> = (0..300)
                .map(|i| Archive::create(&ctx, 1, |_| Ok(Complex64::new(i as f64, 0.))).unwrap())
                .collect();
            assert!(open_handles() <= MAX_OPEN_HANDLES);
            assert!(open_handles() >= before.min(MAX_OPEN_HANDLES));
            assert!(arrays.iter().any(|a| !a.holds_descriptor()));
            for (i, array) in arrays.iter().enumerate() {
                assert_eq!(
                    array.read(&ctx).unwrap().as_slice(),
                    [Complex64::new(i as f64, 0.)]
                );
            }
        }
        assert_eq!(open_handles(), before);
        assert_eq!(ctx.memory_used(), 0);
    }
    #[test]
    fn the_scratch_folder_is_under_the_temp_dir_and_leaves_with_its_owners() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        assert!(ctx.scratch().existing().is_none());
        let directory;
        {
            let a = Archive::create(&ctx, 3, |i| Ok(i as f64)).unwrap();
            directory = ctx.scratch().existing().unwrap();
            assert!(directory.starts_with(std::env::temp_dir()));
            assert!(directory.is_absolute());
            assert!(a.path.starts_with(&directory));
            assert!(directory.is_dir());
        }
        assert!(directory.is_dir());
        drop(ctx);
        assert!(!directory.exists());
    }
}
