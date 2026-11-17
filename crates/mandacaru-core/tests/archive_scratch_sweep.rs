//! The scratch folders of runs stopped without unwinding (SIGTERM's default action, SIGKILL) are
//! swept by the next calculation (`math::archive::sweep_stale_folders`, called when a
//! `ScratchDir` creates its folder): only the exact `mandacaru-scratch-<pid>-<n>` names of
//! processes that no longer exist are removed.
#![cfg(unix)]
use mandacaru_core::math::archive::sweep_stale_folders;

#[test]
fn only_the_folders_of_dead_processes_are_swept() {
    let base = std::env::temp_dir().join(format!("archive-sweep-test-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    // A pid that existed and is gone: a child, reaped.
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap();
    let live = std::process::id();
    let folder = |name: String| {
        let path = base.join(name);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("payload.bin"), [1u8; 64]).unwrap();
        path
    };
    let stale = folder(format!("mandacaru-scratch-{dead}-0"));
    let own = folder(format!("mandacaru-scratch-{live}-7"));
    let parent = folder(format!(
        "mandacaru-scratch-{}-1",
        std::os::unix::process::parent_id()
    ));
    let near_misses = [
        folder(format!("mandacaru-scratch-{dead}-")),
        folder(format!("mandacaru-scratch-{dead}-1x")),
        folder(format!("mandacaru-scratch-{dead}")),
        folder(format!("other-scratch-{dead}-0")),
        folder(format!("mandacaru-scratch-{dead}-0-copy")),
    ];
    let removed = sweep_stale_folders(&base);
    println!("removed {removed:?}");
    assert_eq!(removed, vec![stale.clone()]);
    assert!(!stale.exists());
    for kept in [&own, &parent].into_iter().chain(&near_misses) {
        assert!(kept.join("payload.bin").exists(), "{kept:?} was touched");
    }
    std::fs::remove_dir_all(&base).unwrap();
}
