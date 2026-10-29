use mandacaru_core::{
    execution,
    persistence::{self, Document, CHECKPOINT},
    quantum::{CachedProblem, Encoding, Operator, SolveOptions, C64},
    Context, Error,
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

/// Distinguishes directories made in the same clock tick of one process.
static SERIAL: AtomicUsize = AtomicUsize::new(0);

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "mandacaru-native-persistence-{}-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn context() -> Context {
    Context::new(Some(1), 64 << 20).unwrap()
}

#[test]
fn standalone_checkpoint_prepare_expectation_and_roundtrip() {
    let ctx = context();
    let dir = Directory::new();
    {
        let payload = json!({"format":CHECKPOINT,"version":2,"energy_unit":"Ha","n_qubits":1,"reference_qubits":[],"mapping":"jordan_wigner","preparation":"product","operators":[{"parameter":0.3,"generator":{"num_qubits":1,"terms":[["Y",0.,1.]]}}],"hamiltonian":{"num_qubits":1,"terms":[["Z",1.,0.]]},"metadata":{"note":"🌵"}});
        let doc = Document::parse(&ctx, &payload.to_string(), CHECKPOINT).unwrap();
        let state = doc.state(&ctx).unwrap();
        assert!((state.as_slice()[0].re - 0.3f64.cos()).abs() < 1e-13);
        assert!((doc.expectation(&ctx).unwrap() - 0.6f64.cos()).abs() < 1e-13);
        let path = dir.0.join("nested/state.json");
        doc.save(&path).unwrap();
        let loaded = Document::read(&ctx, &path, CHECKPOINT).unwrap();
        assert_eq!(loaded.payload(), doc.payload());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn atomic_failure_keeps_old_file_and_cleans_staging() {
    let dir = Directory::new();
    let path = dir.0.join("state.json");
    fs::write(&path, "previous").unwrap();
    let result = persistence::atomic_write(&path, |out| {
        out.write_all(b"partial").unwrap();
        Err(Error::Invalid("injected failure".into()))
    });
    assert!(result.is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), "previous");
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[test]
fn both_native_solvers_resume_their_owned_snapshots() {
    let ctx = context();
    {
        let h = Operator::new(
            &ctx,
            4,
            &[
                ("ZIII".into(), C64::new(0.3, 0.)),
                ("XXII".into(), C64::new(0.2, 0.)),
                ("YYII".into(), C64::new(0.2, 0.)),
            ],
        )
        .unwrap();
        let problem = CachedProblem::new(
            &ctx,
            Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
            h,
            true,
            false,
            true,
        )
        .unwrap();
        for adapt in [false, true] {
            let options = SolveOptions {
                adapt,
                ..Default::default()
            };
            let result = problem.solve(&options, None).unwrap();
            let doc = persistence::checkpoint(
                &ctx,
                &problem,
                &result.indices,
                &result.parameters,
                adapt,
                if adapt { "adapt-vqe" } else { "vqe" },
                json!({"complete":true}),
                result.energy,
            )
            .unwrap();
            let resumed = doc.resume(&problem, &options).unwrap();
            assert!(resumed.success);
            assert!((resumed.energy - result.energy).abs() < 1e-12);
            assert_eq!(resumed.indices, result.indices);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn quotas_reject_before_accounting_and_keep_failed_jobs_charged() {
    let ctx = context();
    {
        let mut ledger = execution::RunLedger::new(&ctx, Some(2), Some(100));
        let tickets = ledger.reserve(2, 100).unwrap();
        assert!(ledger.reserve(1, 1).is_err());
        assert_eq!(ledger.usage.attempted_jobs, 2);
        assert_eq!(ledger.usage.total_shots, 100);
        ledger.finish(tickets[0], false).unwrap();
        ledger.finish(tickets[1], true).unwrap();
        assert!(ledger.finish(tickets[0], true).is_err());
        assert_eq!(ledger.usage.failed_jobs, 1);
        assert_eq!(ledger.usage.completed_jobs, 1);
        assert!(ledger.reserve(1, 0).is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn grouping_counts_and_circuit_validation_without_sdks() {
    let ctx = context();
    let labels = vec!["IX".into(), "ZX".into(), "YI".into(), "II".into()];
    {
        let groups = execution::qwc(&ctx, &labels, None).unwrap();
        assert_eq!(
            groups.groups,
            vec![("YX".into(), vec![3, 0, 2]), ("ZX".into(), vec![1])]
        );
        let values =
            execution::expectations(&ctx, &labels, &[("00".into(), 3.), ("11".into(), 1.)])
                .unwrap();
        assert_eq!(values, vec![0.5, 1., 0.5, 1.]);
        let circuit = execution::compile(
            &ctx,
            2,
            &[0],
            &[vec![("XY".into(), 0., 0.5), ("YX".into(), 0., -0.5)]],
            &[0.4],
        )
        .unwrap();
        assert_eq!(circuit.gates.iter().filter(|g| g.name == "cx").count(), 4);
        assert!(execution::compile(
            &ctx,
            1,
            &[],
            &[vec![("X".into(), 0., 0.3), ("Z".into(), 0., 0.2)]],
            &[0.3]
        )
        .is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn native_band_writer_repeats_planes_without_padded_tensor() {
    let ctx = Context::new(Some(1), 8192).unwrap();
    let dir = Directory::new();
    mandacaru_core::volumetric::bxsf(
        &ctx,
        &dir.0.join("bands.bxsf"),
        [1, 1, 1, 1],
        &[0.75],
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        0.5,
        "test",
    )
    .unwrap();
    let content = fs::read_to_string(dir.0.join("bands.bxsf")).unwrap();
    assert_eq!(content.matches("0.750000").count(), 8);
    assert_eq!(ctx.memory_used(), 0);
}
