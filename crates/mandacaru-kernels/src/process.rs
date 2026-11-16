//! Whether a process exists: the liveness test of the scratch-folder sweep
//! (`mandacaru-core::math::archive::ScratchDir`), which removes the folders of runs that were
//! stopped without unwinding. The contract of the foreign call is in `SAFETY.md`, "Asking
//! whether a process exists".

#[cfg(unix)]
mod system {
    extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    /// `ESRCH`, "no such process", on macOS and Linux alike.
    const NO_SUCH_PROCESS: i32 = 3;

    pub fn alive(pid: u32) -> bool {
        // A pid that is not a positive `pid_t` names a process group or every process: never ask.
        let Ok(pid) = i32::try_from(pid) else {
            return true;
        };
        if pid <= 0 {
            return true;
        }
        // SAFETY: signal 0 delivers nothing; it only checks that `pid` exists and may be
        // signaled. The call takes no pointer and sets `errno` on failure, read at once below.
        let status = unsafe { kill(pid, 0) };
        status == 0 || std::io::Error::last_os_error().raw_os_error() != Some(NO_SUCH_PROCESS)
    }
}
#[cfg(not(unix))]
mod system {
    pub fn alive(_pid: u32) -> bool {
        true
    }
}

/// Whether the process `pid` exists. `true` unless the operating system says there is no such
/// process (a process of another user, which may not be signaled, exists); always `true` where
/// the platform has no such question, so nothing is ever taken for dead by default.
pub fn is_alive(pid: u32) -> bool {
    system::alive(pid)
}

#[cfg(test)]
mod tests {
    use super::is_alive;

    #[test]
    fn this_process_is_alive_and_a_reaped_child_is_not() {
        assert!(is_alive(std::process::id()));
        if cfg!(unix) {
            let mut child = std::process::Command::new("true").spawn().unwrap();
            let pid = child.id();
            child.wait().unwrap();
            assert!(!is_alive(pid), "a reaped child {pid} is reported alive");
        }
    }
}
