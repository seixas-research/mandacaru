//! Returning the pages the allocator holds free to the operating system.
//!
//! A calculation that has released its arrays still counts them in the resident set of the
//! process while the allocator keeps the freed pages for reuse; a scan of geometries then
//! carries the pages of one point into the next. [`release_freed_memory`] asks the allocator
//! to hand back what it holds free. It moves no live data and changes no result. The contract of
//! the two foreign calls is in `SAFETY.md`, "Returning freed heap pages".

#[cfg(target_os = "macos")]
mod system {
    extern "C" {
        pub fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
    }
    pub fn relieve() -> bool {
        // SAFETY: a null zone means every zone of the process and a zero goal means as much as
        // possible; the call takes no pointer from this program and returns a byte count.
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
        true
    }
}
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod system {
    extern "C" {
        pub fn malloc_trim(pad: usize) -> i32;
    }
    pub fn relieve() -> bool {
        // SAFETY: a zero pad keeps no slack at the top of the heap; the call takes no pointer
        // from this program and returns 1 when it released memory and 0 otherwise.
        unsafe { malloc_trim(0) };
        true
    }
}
#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
mod system {
    pub fn relieve() -> bool {
        false
    }
}

#[cfg(target_os = "macos")]
mod installed {
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    pub fn bytes() -> Option<u64> {
        let mut value: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        // SAFETY: `hw.memsize` is a 64-bit integer; `value` and `size` are valid for writes of
        // that size, and no new value is set (null, zero).
        let status = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut value as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && size == std::mem::size_of::<u64>() && value > 0).then_some(value)
    }
}
#[cfg(target_os = "linux")]
mod installed {
    pub fn bytes() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kib.checked_mul(1024)
    }
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod installed {
    pub fn bytes() -> Option<u64> {
        None
    }
}

/// The physical memory installed in the machine (`hw.memsize` on macOS, `MemTotal` of
/// `/proc/meminfo` on Linux), `None` where the platform does not say.
pub fn installed_memory_bytes() -> Option<u64> {
    installed::bytes()
}

/// Returns the pages the allocator holds free to the operating system, where the platform has a
/// call for it (macOS, glibc); `false` says it has none and nothing was done.
pub fn release_freed_memory() -> bool {
    system::relieve()
}

#[cfg(test)]
mod tests {
    use super::{installed_memory_bytes, release_freed_memory};

    #[test]
    fn the_installed_memory_is_known_on_the_supported_platforms() {
        let bytes = installed_memory_bytes();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(bytes.is_some_and(|b| b >= 1 << 28), "{bytes:?}");
        }
    }

    #[test]
    fn releasing_freed_memory_leaves_live_data_and_later_allocations_intact() {
        let keep: Vec<u64> = (0..2_000_000u64).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
        let discard: Vec<u64> = vec![7; 4_000_000];
        drop(discard);
        let supported = release_freed_memory();
        assert_eq!(supported, cfg!(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))));
        assert!(keep
            .iter()
            .enumerate()
            .all(|(i, v)| *v == (i as u64).wrapping_mul(0x9e37_79b9)));
        let again: Vec<u64> = (0..1_000_000u64).collect();
        assert_eq!(again[999_999], 999_999);
        assert!(release_freed_memory() == supported);
    }
}
