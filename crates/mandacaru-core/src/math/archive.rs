//! Immutable calculation-owned scratch arrays with bounded, budgeted reads.
//! Scratch files use exclusive creation and are removed with their owner.
use crate::{product, Buffer, Context, Error, Reservation, Result};
use num_complex::{Complex32, Complex64};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};

pub trait Scalar: Copy + Default {
    const BYTES: usize;
    fn encode(self, bytes: &mut [u8]);
    fn decode(bytes: &[u8]) -> Self;
}
impl Scalar for Complex64 {
    const BYTES: usize = 16;
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
static NEXT: AtomicU64 = AtomicU64::new(0);
fn io(error: std::io::Error) -> Error {
    Error::Invalid(format!("calculation scratch array: {error}"))
}
pub struct Archive<T> {
    file: Mutex<Option<File>>,
    path: PathBuf,
    length: usize,
    _scalar: PhantomData<T>,
    _memory: Reservation,
}
impl<T> Drop for Archive<T> {
    fn drop(&mut self) {
        self.file
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let _ = std::fs::remove_file(&self.path);
    }
}
impl<T: Scalar> Archive<T> {
    pub fn create(
        ctx: &Context,
        length: usize,
        mut value: impl FnMut(usize) -> Result<T>,
    ) -> Result<Self> {
        if T::BYTES == 0 || T::BYTES > 8192 {
            return Err(Error::Invalid("invalid scratch scalar size".into()));
        }
        product(&[length, T::BYTES])?;
        let memory = ctx.reserve(512)?;
        let (file, path) = loop {
            let path = std::env::temp_dir().join(format!(
                "mandacaru-array-{}-{}.bin",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
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
        let out = Self {
            file: Mutex::new(Some(file)),
            path,
            length,
            _scalar: PhantomData,
            _memory: memory,
        };
        let mut guard = out
            .file
            .lock()
            .map_err(|_| Error::Invalid("scratch array lock was poisoned".into()))?;
        let file = guard.as_mut().unwrap();
        let mut bytes = [0u8; 8192];
        let block = bytes.len() / T::BYTES;
        for start in (0..length).step_by(block) {
            let count = (length - start).min(block);
            for i in 0..count {
                value(start + i)?.encode(&mut bytes[i * T::BYTES..(i + 1) * T::BYTES]);
            }
            file.write_all(&bytes[..count * T::BYTES]).map_err(io)?;
        }
        file.flush().map_err(io)?;
        guard.take(); // Retained scratch arrays do not retain an open descriptor.
        drop(guard);
        Ok(out)
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn read_into<U>(&self, output: &mut [U], convert: impl Fn(T) -> U) -> Result<()> {
        if output.len() != self.length {
            return Err(Error::Invalid("scratch read dimensions disagree".into()));
        }
        self.read_range_into(0, output, convert)
    }
    pub fn read_range_into<U>(
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
        let offset = u64::try_from(product(&[start, T::BYTES])?).map_err(|_| Error::Overflow)?;
        let mut file = File::open(&self.path).map_err(io)?;
        file.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut bytes = [0u8; 8192];
        for values in output.chunks_mut(bytes.len() / T::BYTES) {
            file.read_exact(&mut bytes[..values.len() * T::BYTES])
                .map_err(io)?;
            for (to, from) in values.iter_mut().zip(bytes.chunks_exact(T::BYTES)) {
                *to = convert(T::decode(from));
            }
        }
        Ok(())
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
    #[test]
    fn archive_reads_are_exact_bounded_and_removed_with_the_owner() {
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
            archive.read_range_into(107, &mut partial, |v| v).unwrap();
            assert_eq!(partial, expected[107..440]);
            assert!(archive.read_range_into(1490, &mut partial, |v| v).is_err());
            assert_eq!(ctx.memory_used(), 512);
            // A damaged scratch record must surface an I/O error, never zeros.
            assert!(archive.file.lock().unwrap().is_none());
            OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(8)
                .unwrap();
            assert!(archive.read(&ctx).is_err());
        }
        assert!(!path.exists());
        assert_eq!(ctx.memory_used(), 0);
    }
    #[test]
    fn retained_arrays_use_descriptors_only_during_io() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        {
            let arrays: Vec<_> = (0..300)
                .map(|i| Archive::create(&ctx, 1, |_| Ok(Complex64::new(i as f64, 0.))).unwrap())
                .collect();
            for (i, array) in arrays.iter().enumerate() {
                assert!(array.file.lock().unwrap().is_none());
                assert_eq!(
                    array.read(&ctx).unwrap().as_slice(),
                    [Complex64::new(i as f64, 0.)]
                );
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
}
