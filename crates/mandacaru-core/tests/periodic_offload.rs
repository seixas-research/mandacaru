//! The disk-offload threshold changes where a crystal keeps its bulk arrays, never its numbers:
//! the same saved crystal solved with every bulk array in memory and with every one on disk
//! gives the same energy, bands and density, and saves the same file.
use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::{Crystal, Options},
    Context,
};

fn fixture() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5/si_crystal.json")
        .to_str()
        .unwrap()
        .to_string()
}

#[test]
fn a_crystal_on_disk_solves_to_the_crystal_in_memory() {
    let path = fixture();
    let options = Options::default();
    // Every bulk array in memory: a threshold no array reaches (the memory limit is large).
    let ctx_memory = Context::with_offload(Some(1), 1 << 30, usize::MAX).unwrap();
    // Every bulk array on disk.
    let ctx_disk = Context::with_offload(Some(1), 1 << 30, 0).unwrap();
    let (memory, disk);
    {
        let in_memory = Crystal::load(&ctx_memory, &path).unwrap();
        assert_eq!(ctx_memory.archived_bytes(), 0);
        assert!(in_memory.points.iter().all(|p| !p.psi.archived()) && !in_memory.hats.archived());
        let on_disk = Crystal::load(&ctx_disk, &path).unwrap();
        assert!(on_disk.points.iter().all(|p| p.psi.archived()) && on_disk.hats.archived());
        let psi_bytes: usize = on_disk.points.iter().map(|p| p.psi.len() * 16).sum();
        assert!(ctx_disk.archived_bytes() >= psi_bytes, "psi must be on disk");
        // The resident reservation shrinks by what left for disk.
        assert!(ctx_disk.memory_used() + psi_bytes / 2 < ctx_memory.memory_used());
        memory = in_memory.solve(&options).unwrap();
        disk = on_disk.solve(&options).unwrap();
        // The saved snapshot carries the metadata text whichever side it was kept on.
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let (a, b) = (dir.join("offload_memory.json"), dir.join("offload_disk.json"));
        in_memory.save(a.to_str().unwrap()).unwrap();
        on_disk.save(b.to_str().unwrap()).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        std::fs::remove_file(a).unwrap();
        std::fs::remove_file(b).unwrap();
        assert!(memory.converged && disk.converged);
        assert!(
            (memory.free_energy - disk.free_energy).abs() < 1e-12,
            "{} against {}",
            memory.free_energy,
            disk.free_energy
        );
        assert_eq!(memory.history.len(), disk.history.len());
        assert_eq!(memory.eigenvalues, disk.eigenvalues);
        assert_eq!(memory.density.as_slice(), disk.density.as_slice());
        for (a, b) in memory.matrices.iter().zip(&disk.matrices) {
            assert_eq!(
                *a.matrix(&ctx_memory).unwrap(),
                *b.matrix(&ctx_disk).unwrap()
            );
        }
    }
    drop((memory, disk));
    assert_eq!(ctx_memory.memory_used(), 0);
    assert_eq!(ctx_disk.memory_used(), 0);
}

#[test]
fn the_scratch_folder_follows_the_context_and_leaves_with_it() {
    let path = fixture();
    let ctx = Context::with_offload(Some(1), 1 << 30, 0).unwrap();
    let directory;
    {
        let crystal = Crystal::load(&ctx, &path).unwrap();
        directory = ctx.scratch().existing().expect("psi went to disk");
        assert!(directory.starts_with(std::env::temp_dir()));
        assert!(std::fs::read_dir(&directory).unwrap().count() >= crystal.points.len());
    }
    // No archive is left; the folder goes with the last owner.
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
    drop(ctx);
    assert!(!directory.exists());
}
