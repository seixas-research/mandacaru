use mandacaru_core::{
    execution,
    math::storage::Precision,
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
        let state = doc.state(&ctx, Precision::Float64).unwrap();
        assert!((state.complex128().unwrap()[0].re - 0.3f64.cos()).abs() < 1e-13);
        let narrow = doc.state(&ctx, Precision::Float32).unwrap();
        assert!((f64::from(narrow.complex64().unwrap()[0].re) - 0.3f64.cos()).abs() < 1e-7);
        assert!(narrow.complex128().is_err());
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

/// Hamiltonians stored as Parquet: the document layer on top of
/// `mandacaru_core::parquet` (the container itself is tested in `parquet.rs`,
/// `parquet_interop.rs` and `parquet_errors.rs`).
mod hamiltonian_parquet {
    use super::*;
    use mandacaru_core::parquet::{self, Codec, Column, Table, Utf8Column, WriteOptions};
    use mandacaru_core::persistence::HAMILTONIAN;
    use serde_json::Value;
    use std::collections::BTreeMap;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/parquet")
    }

    fn payload() -> Value {
        json!({
            "format": HAMILTONIAN,
            "version": 1,
            "num_qubits": 4,
            "mapping": "jordan_wigner",
            "num_particles": [1, 1],
            "n_spatial_orbitals": 2,
            "metadata": {"energy_unit": "Ha", "driver": "test", "note": "café 🌵"},
            "terms": [
                ["IIII", 0.75, 0.0],
                ["IIIZ", -0.125, 0.0],
                ["XXYY", 0.0625, -0.5],
                ["ZZII", 1.5, 2.0],
            ],
        })
    }

    fn document(ctx: &Context, payload: &Value) -> Document {
        Document::parse(ctx, &payload.to_string(), HAMILTONIAN).unwrap()
    }

    fn error_of(name: &str) -> String {
        let ctx = context();
        let error = Document::read(&ctx, &fixtures().join(name), HAMILTONIAN)
            .err()
            .unwrap_or_else(|| panic!("{name} should not load"));
        assert_eq!(ctx.memory_used(), 0, "{name}");
        error.to_string()
    }

    #[test]
    fn json_and_parquet_documents_are_equal_after_validate() {
        let ctx = context();
        let dir = Directory::new();
        {
            let original = document(&ctx, &payload());
            let json_path = dir.0.join("h.json");
            original.save(&json_path).unwrap();
            let from_json = Document::read(&ctx, &json_path, HAMILTONIAN).unwrap();
            for codec in [Codec::Zstd, Codec::Uncompressed] {
                let path = dir.0.join(format!("h-{}.parquet", codec.name()));
                original.save_parquet(&ctx, &path, codec).unwrap();
                assert!(parquet::is_parquet(&path).unwrap());
                let from_parquet = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
                assert_eq!(from_parquet.payload(), from_json.payload());
                assert_eq!(from_parquet.payload(), original.payload());
            }
            // The table view agrees with the file.
            let table = original.to_table(&ctx).unwrap();
            let again = Document::from_table(&ctx, &table, HAMILTONIAN, "'table'").unwrap();
            assert_eq!(again.payload(), original.payload());
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn the_format_is_found_by_content_not_by_extension() {
        let ctx = context();
        let dir = Directory::new();
        {
            let original = document(&ctx, &payload());
            // Parquet bytes under a .json name, JSON bytes under a .parquet name.
            let parquet_as_json = dir.0.join("really_parquet.json");
            let json_as_parquet = dir.0.join("really_json.parquet");
            original
                .save_parquet(&ctx, &parquet_as_json, Codec::Zstd)
                .unwrap();
            original.save(&json_as_parquet).unwrap();
            for path in [&parquet_as_json, &json_as_parquet] {
                let doc = Document::read(&ctx, path, HAMILTONIAN).unwrap();
                assert_eq!(doc.payload(), original.payload(), "{}", path.display());
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn writing_is_deterministic_sorted_and_stable_under_a_resave() {
        let ctx = context();
        let dir = Directory::new();
        {
            let mut shuffled = payload();
            shuffled["terms"].as_array_mut().unwrap().reverse();
            let sorted = document(&ctx, &payload());
            let reversed = document(&ctx, &shuffled);
            let (a, b, c) = (
                dir.0.join("a.parquet"),
                dir.0.join("b.parquet"),
                dir.0.join("c.parquet"),
            );
            sorted.save_parquet(&ctx, &a, Codec::Zstd).unwrap();
            sorted.save_parquet(&ctx, &b, Codec::Zstd).unwrap();
            reversed.save_parquet(&ctx, &c, Codec::Zstd).unwrap();
            assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
            assert_eq!(fs::read(&a).unwrap(), fs::read(&c).unwrap());
            // Read, save again: the bytes do not move.
            let back = Document::read(&ctx, &a, HAMILTONIAN).unwrap();
            let d = dir.0.join("d.parquet");
            back.save_parquet(&ctx, &d, Codec::Zstd).unwrap();
            assert_eq!(fs::read(&a).unwrap(), fs::read(&d).unwrap());
            // Rows come out in label order, I < X < Y < Z.
            let table = parquet::read_table(&ctx, &a).unwrap();
            let labels: Vec<&str> = table
                .column("pauli")
                .unwrap()
                .as_utf8()
                .unwrap()
                .iter()
                .collect();
            assert_eq!(labels, ["IIII", "IIIZ", "XXYY", "ZZII"]);
            assert_eq!(table.column_names(), ["pauli", "real", "imag"]);
            // The footer text is compact JSON.
            let footer = parquet::read_metadata(&ctx, &a).unwrap();
            assert_eq!(footer.metadata["mandacaru.num_particles"], "[1,1]");
            assert_eq!(footer.metadata["mandacaru.n_spatial_orbitals"], "2");
            assert_eq!(footer.metadata["mandacaru.version"], "1");
            assert_eq!(footer.created_by, parquet::CREATED_BY);
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn zero_terms_keep_the_register_width() {
        let ctx = context();
        let dir = Directory::new();
        {
            let mut empty = payload();
            empty["terms"] = json!([]);
            let doc = document(&ctx, &empty);
            for codec in [Codec::Zstd, Codec::Uncompressed] {
                let path = dir.0.join(format!("empty-{}.parquet", codec.name()));
                doc.save_parquet(&ctx, &path, codec).unwrap();
                let back = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
                assert_eq!(back.payload()["num_qubits"], json!(4));
                assert_eq!(back.payload()["terms"], json!([]));
                assert_eq!(back.payload(), doc.payload());
            }
            // Both foreign writers' empty files keep the width as well
            // (fastparquet writes no row group, pyarrow writes an empty one).
            for name in [
                "hamiltonian/fastparquet_empty.parquet",
                "hamiltonian/pyarrow_empty.parquet",
            ] {
                let back = Document::read(&ctx, &fixtures().join(name), HAMILTONIAN).unwrap();
                assert_eq!(back.payload()["terms"], json!([]), "{name}");
                assert!(back.payload()["num_qubits"].as_u64().unwrap() > 0, "{name}");
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    // ---- a Hamiltonian that does not satisfy the rules creates no file ----

    fn table(
        ctx: &Context,
        labels: &[&str],
        real: &[f64],
        imag: &[f64],
        kv: &[(&str, &str)],
    ) -> Table {
        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in kv {
            metadata.insert((*k).to_owned(), (*v).to_owned());
        }
        Table {
            columns: vec![
                (
                    "pauli".into(),
                    Column::Utf8(Utf8Column::from_strs(ctx, labels).unwrap()),
                ),
                ("real".into(), Column::Double(ctx.copy_slice(real).unwrap())),
                ("imag".into(), Column::Double(ctx.copy_slice(imag).unwrap())),
            ],
            metadata,
        }
    }

    const BASE_KV: [(&str, &str); 3] = [
        ("mandacaru.format", HAMILTONIAN),
        ("mandacaru.version", "1"),
        ("mandacaru.num_qubits", "2"),
    ];

    #[test]
    fn the_rules_of_a_hamiltonian_refuse_a_table_and_create_no_file() {
        let ctx = context();
        let dir = Directory::new();
        {
            let ok = table(&ctx, &["IX", "ZZ"], &[1.0, 2.0], &[0.0, 0.5], &BASE_KV);
            let doc = Document::from_table(&ctx, &ok, HAMILTONIAN, "'t'").unwrap();
            assert_eq!(
                doc.payload()["terms"],
                json!([["IX", 1.0, 0.0], ["ZZ", 2.0, 0.5]])
            );
            assert_eq!(doc.payload()["mapping"], json!("jordan_wigner"));
            assert_eq!(doc.payload()["metadata"], json!({}));

            let long = "I".repeat(51);
            let fifty_one = table(
                &ctx,
                &[long.as_str()],
                &[1.0],
                &[0.0],
                &[
                    ("mandacaru.format", HAMILTONIAN),
                    ("mandacaru.version", "1"),
                    ("mandacaru.num_qubits", "51"),
                ],
            );
            let cases: Vec<(&str, Table, &str)> = vec![
                ("51 qubits", fifty_one, "at most 50 qubits"),
                (
                    "NaN real",
                    table(&ctx, &["IX"], &[f64::NAN], &[0.0], &BASE_KV),
                    "non-finite",
                ),
                (
                    "infinite imag",
                    table(&ctx, &["IX"], &[1.0], &[f64::NEG_INFINITY], &BASE_KV),
                    "non-finite",
                ),
                (
                    "short label",
                    table(&ctx, &["IXZ"], &[1.0], &[0.0], &BASE_KV),
                    "Pauli labels must match the register",
                ),
                (
                    "bad letter",
                    table(&ctx, &["IQ"], &[1.0], &[0.0], &BASE_KV),
                    "invalid Pauli letter",
                ),
                (
                    "wrong energy unit",
                    table(
                        &ctx,
                        &["IX"],
                        &[1.0],
                        &[0.0],
                        &[
                            ("mandacaru.format", HAMILTONIAN),
                            ("mandacaru.version", "1"),
                            ("mandacaru.num_qubits", "2"),
                            ("mandacaru.metadata", "{\"energy_unit\":\"eV\"}"),
                        ],
                    ),
                    "must be in Ha",
                ),
                (
                    "missing format",
                    table(&ctx, &["IX"], &[1.0], &[0.0], &BASE_KV[1..]),
                    "is not a Mandacaru qubit-Hamiltonian Parquet file",
                ),
                (
                    "missing version",
                    table(&ctx, &["IX"], &[1.0], &[0.0], &[BASE_KV[0], BASE_KV[2]]),
                    "mandacaru.version",
                ),
                (
                    "missing qubits",
                    table(&ctx, &["IX"], &[1.0], &[0.0], &BASE_KV[..2]),
                    "mandacaru.num_qubits",
                ),
                (
                    "future version",
                    table(
                        &ctx,
                        &["IX"],
                        &[1.0],
                        &[0.0],
                        &[BASE_KV[0], ("mandacaru.version", "2"), BASE_KV[2]],
                    ),
                    "version 2",
                ),
                (
                    "version zero",
                    table(
                        &ctx,
                        &["IX"],
                        &[1.0],
                        &[0.0],
                        &[BASE_KV[0], ("mandacaru.version", "0"), BASE_KV[2]],
                    ),
                    "unsupported snapshot schema version",
                ),
            ];
            for (what, t, fragment) in &cases {
                let error = Document::from_table(&ctx, t, HAMILTONIAN, "'t'")
                    .err()
                    .unwrap_or_else(|| panic!("{what} was accepted"));
                assert!(error.to_string().contains(fragment), "{what}: {error}");
            }
            // The same two rules on the way in from JSON: nothing is written.
            let mut big = payload();
            big["num_qubits"] = json!(51);
            big["num_particles"] = json!(null);
            big["n_spatial_orbitals"] = json!(null);
            assert!(Document::parse(&ctx, &big.to_string(), HAMILTONIAN).is_err());
            assert!(Document::parse(&ctx, "{\"terms\":[[\"I\",NaN,0]]}", HAMILTONIAN).is_err());
            // A save that cannot happen leaves nothing behind.
            let blocked = dir.0.join("file").join("inner.parquet");
            fs::write(dir.0.join("file"), b"x").unwrap();
            let doc = document(&ctx, &payload());
            assert!(doc.save_parquet(&ctx, &blocked, Codec::Zstd).is_err());
            assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn a_failed_save_keeps_the_previous_parquet_file() {
        let big = context();
        let dir = Directory::new();
        let path = dir.0.join("h.parquet");
        let doc = document(&big, &payload());
        doc.save_parquet(&big, &path, Codec::Zstd).unwrap();
        let before = fs::read(&path).unwrap();
        // The document fits, the zstd encoder's working memory does not.
        let tight = Context::new(Some(1), 1 << 20).unwrap();
        let small = document(&tight, &payload());
        let error = small.save_parquet(&tight, &path, Codec::Zstd).unwrap_err();
        assert!(matches!(error, Error::Budget { .. }), "{error:?}");
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        drop(small);
        assert_eq!(tight.memory_used(), 0);
        drop(doc);
        assert_eq!(big.memory_used(), 0);
    }

    // ---- the frozen fixtures through Document::read ----

    #[test]
    fn the_hamiltonian_rules_run_on_the_negative_fixtures() {
        let expected = [
            (
                "hamiltonian_qubits_51",
                "Pauli-string files support at most 50 qubits",
            ),
            ("hamiltonian_bad_letter", "invalid Pauli letter"),
            (
                "hamiltonian_energy_unit_ev",
                "snapshot energies must be in Ha",
            ),
            (
                "hamiltonian_kv_version_0",
                "unsupported snapshot schema version",
            ),
            (
                "hamiltonian_kv_version_2",
                "uses Hamiltonian format version 2, but this build understands up to 1",
            ),
            (
                "hamiltonian_kv_unknown_mapping",
                "unsupported snapshot mapping",
            ),
            (
                "hamiltonian_kv_num_qubits_mismatch",
                "snapshot qubits disagree with mapping and orbitals",
            ),
            (
                "hamiltonian_wrong_pauli_length",
                "Pauli labels must match the register and coefficients must be finite",
            ),
            (
                "hamiltonian_kv_missing_version",
                "lacks the mandacaru.version entry",
            ),
            (
                "hamiltonian_kv_missing_num_qubits",
                "lacks the mandacaru.num_qubits entry",
            ),
            (
                "hamiltonian_kv_metadata_not_json",
                "unreadable mandacaru.metadata entry",
            ),
            ("hamiltonian_kv_bad_num_particles", "valid particle counts"),
            ("hamiltonian_nonfinite_inf", "non-finite coefficient"),
            ("null_in_pauli_pyarrow", "null"),
            ("null_in_real_pyarrow", "null"),
            ("null_in_real_fastparquet", "null"),
            ("nan_in_real_fastparquet", "null"),
        ];
        for (name, fragment) in expected {
            let message = error_of(&format!("negative/{name}.parquet"));
            assert!(message.contains(fragment), "{name}: {message}");
        }
        let message = error_of("negative/hamiltonian_missing_imag.parquet");
        assert!(
            message.ends_with("is missing the 'imag' column (found ['pauli', 'real'])"),
            "{message}"
        );
        assert!(message.starts_with('\''), "{message}");
        for name in [
            "hamiltonian_no_kv",
            "hamiltonian_kv_wrong_format",
            "hamiltonian_kv_missing_format",
        ] {
            let message = error_of(&format!("negative/{name}.parquet"));
            assert!(
                message.contains("is not a Mandacaru qubit-Hamiltonian Parquet file"),
                "{name}: {message}"
            );
        }
        // A column of the wrong type, and every other negative fixture, raise.
        let message = error_of("negative/column_double_pauli.parquet");
        assert!(message.contains("pauli"), "{message}");
    }

    #[test]
    fn a_checkpoint_never_comes_from_a_parquet_file() {
        let ctx = context();
        let error = Document::read(
            &ctx,
            &fixtures().join("hamiltonian/fastparquet_zstd.parquet"),
            CHECKPOINT,
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("qubit Hamiltonian is stored as Parquet"));
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn missing_optional_entries_take_their_defaults() {
        let ctx = context();
        let doc = Document::read(
            &ctx,
            &fixtures().join("hamiltonian/semantic_kv_defaults_only_required.parquet"),
            HAMILTONIAN,
        )
        .unwrap();
        let p = doc.payload();
        assert_eq!(p["mapping"], json!("jordan_wigner"));
        assert!(p["num_particles"].is_null() && p["n_spatial_orbitals"].is_null());
        assert_eq!(p["metadata"], json!({}));
        assert_eq!(p["version"], json!(1));
    }

    /// Python's `float.hex()` of a finite double.
    fn python_hex(x: f64) -> String {
        let bits = x.to_bits();
        let sign = if bits >> 63 == 1 { "-" } else { "" };
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = bits & ((1u64 << 52) - 1);
        if exponent == 0 && mantissa == 0 {
            format!("{sign}0x0.0p+0")
        } else if exponent == 0 {
            format!("{sign}0x0.{mantissa:013x}p-1022")
        } else {
            format!("{sign}0x1.{mantissa:013x}p{:+}", exponent - 1023)
        }
    }

    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut message = data.to_vec();
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in message.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, word) in block.chunks_exact(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                v = [
                    t1.wrapping_add(s0.wrapping_add(maj)),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for (slot, value) in h.iter_mut().zip(v) {
                *slot = slot.wrapping_add(value);
            }
        }
        h.iter().map(|word| format!("{word:08x}")).collect()
    }

    /// Terms in the shape the previous reader produced: duplicate labels summed
    /// in file order, sorted by label, as `(label, real.hex(), imag.hex())`.
    fn merged_terms(payload: &Value) -> Vec<(String, String, String)> {
        let mut sums: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for term in payload["terms"].as_array().unwrap() {
            let entry = sums
                .entry(term[0].as_str().unwrap().to_owned())
                .or_default();
            entry.0 += term[1].as_f64().unwrap();
            entry.1 += term[2].as_f64().unwrap();
        }
        sums.into_iter()
            .map(|(label, (re, im))| (label, python_hex(re), python_hex(im)))
            .collect()
    }

    #[test]
    fn every_frozen_hamiltonian_fixture_matches_the_previous_reader_bit_for_bit() {
        let frozen: Value =
            serde_json::from_slice(&fs::read(fixtures().join("old_api_results.json")).unwrap())
                .unwrap();
        let mut names: Vec<_> = frozen["fixtures"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        names.sort();
        let (mut compared, mut tolerant) = (0, 0);
        for name in names.iter().filter(|n| n.starts_with("hamiltonian/")) {
            let results = &frozen["fixtures"][name]["load_hamiltonian"];
            let ctx = context();
            let loaded = Document::read(&ctx, &fixtures().join(name), HAMILTONIAN);
            let expected = ["pyarrow", "fastparquet"]
                .iter()
                .map(|engine| &results[*engine])
                .find(|r| r["ok"] == json!(true));
            let Some(expected) = expected else {
                // Neither previous engine could load it (an unannotated binary
                // column, missing optional entries): this reader is more
                // tolerant, never different where both load.
                assert!(loaded.is_ok(), "{name}: {:?}", loaded.err());
                tolerant += 1;
                continue;
            };
            let expected = &expected["result"];
            let doc = loaded.unwrap_or_else(|e| panic!("{name}: {e}"));
            let p = doc.payload();
            for key in [
                "num_qubits",
                "mapping",
                "num_particles",
                "n_spatial_orbitals",
                "metadata",
            ] {
                assert_eq!(p[key], expected[key], "{name} {key}");
            }
            let terms = merged_terms(p);
            assert_eq!(
                terms.len() as u64,
                expected["n_terms"].as_u64().unwrap(),
                "{name}"
            );
            let lines: Vec<String> = terms
                .iter()
                .map(|(label, re, im)| format!("{label} {re} {im}"))
                .collect();
            assert_eq!(
                sha256_hex(lines.join("\n").as_bytes()),
                expected["terms_sha256"].as_str().unwrap(),
                "{name}"
            );
            if let Some(list) = expected["terms"].as_array() {
                for (want, got) in list.iter().zip(&terms) {
                    assert_eq!(want, &json!([got.0, got.1, got.2]), "{name}");
                }
            }
            drop(doc);
            assert_eq!(ctx.memory_used(), 0, "{name}");
            compared += 1;
        }
        assert!(
            compared >= 20,
            "only {compared} fixtures compared ({tolerant} tolerated)"
        );
    }

    // ---- size, scale and budgets ----

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A table of `rows` distinct `qubits`-letter labels (a bijection of the
    /// row number, so no duplicates) with pseudo-random real parts.
    fn big_table(ctx: &Context, rows: usize, qubits: usize) -> Table {
        let mask = (1u64 << (2 * qubits)) - 1;
        let letters = [b'I', b'X', b'Y', b'Z'];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let labels: Vec<String> = (0..rows as u64)
            .map(|i| {
                let code = (i.wrapping_mul(0x9E37_79B9_7F4A_7C15 | 1)) & mask;
                (0..qubits)
                    .map(|q| letters[((code >> (2 * q)) & 3) as usize] as char)
                    .collect()
            })
            .collect();
        let real: Vec<f64> = (0..rows)
            .map(|_| {
                (0..12)
                    .map(|_| (rng.next() >> 11) as f64 / (1u64 << 53) as f64)
                    .sum::<f64>()
                    - 6.0
            })
            .collect();
        let imag = vec![0.0; rows];
        let mut metadata = BTreeMap::new();
        metadata.insert("mandacaru.format".to_owned(), HAMILTONIAN.to_owned());
        metadata.insert("mandacaru.version".to_owned(), "1".to_owned());
        metadata.insert("mandacaru.num_qubits".to_owned(), qubits.to_string());
        Table {
            columns: vec![
                (
                    "pauli".into(),
                    Column::Utf8(Utf8Column::from_strs(ctx, &labels).unwrap()),
                ),
                (
                    "real".into(),
                    Column::Double(ctx.copy_slice(&real).unwrap()),
                ),
                (
                    "imag".into(),
                    Column::Double(ctx.copy_slice(&imag).unwrap()),
                ),
            ],
            metadata,
        }
    }

    #[test]
    fn a_large_hamiltonian_costs_less_than_28_bytes_a_term() {
        // The pin the previous format met: 4000 random 12-qubit terms.
        let ctx = context();
        let dir = Directory::new();
        {
            let table = big_table(&ctx, 4000, 12);
            let path = dir.0.join("big.parquet");
            parquet::write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
            let size = fs::metadata(&path).unwrap().len();
            eprintln!(
                "4000 terms on 12 qubits: {size} bytes, {:.2} per term",
                size as f64 / 4000.0
            );
            assert!(size < 28 * 4000, "{size} bytes");
            let doc = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
            assert_eq!(doc.payload()["terms"].as_array().unwrap().len(), 4000);
            // The Document writer sorts the rows, so the first save of an
            // unsorted table changes the bytes and every later save keeps them.
            let again = dir.0.join("again.parquet");
            let third = dir.0.join("third.parquet");
            doc.save_parquet(&ctx, &again, Codec::Zstd).unwrap();
            let back = Document::read(&ctx, &again, HAMILTONIAN).unwrap();
            back.save_parquet(&ctx, &third, Codec::Zstd).unwrap();
            assert_eq!(fs::read(&again).unwrap(), fs::read(&third).unwrap());
            let size = fs::metadata(&again).unwrap().len();
            assert!(size < 28 * 4000, "{size} bytes after sorting");
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn the_28_byte_pin_holds_when_the_last_page_holds_a_handful_of_rows() {
        // A row count of 65,536 k + 1..4 leaves a tail page whose zstd frame
        // is no smaller than its rows; that page must not make the chunk
        // (and, before the fix, the whole file) uncompressed.
        let ctx = context();
        let dir = Directory::new();
        for terms in [65_536usize, 65_537, 65_538, 65_540] {
            let table = big_table(&ctx, terms, 12);
            let path = dir.0.join(format!("tail-{terms}.parquet"));
            parquet::write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
            let size = fs::metadata(&path).unwrap().len();
            eprintln!(
                "{terms} terms on 12 qubits: {size} bytes, {:.2} per term",
                size as f64 / terms as f64
            );
            assert!(size < 28 * terms as u64, "{terms} terms: {size} bytes");
            let back = parquet::read_table(&ctx, &path).unwrap();
            assert_eq!(back.num_rows(), terms);
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn hundred_thousand_terms_and_more_than_one_row_group_round_trip() {
        let ctx = Context::new(Some(1), 1500 << 20).unwrap();
        let dir = Directory::new();
        {
            for (rows, qubits, groups) in [(100_000usize, 24usize, 1usize), (1_100_000, 24, 2)] {
                let path = dir.0.join(format!("{rows}.parquet"));
                {
                    let table = big_table(&ctx, rows, qubits);
                    parquet::write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
                }
                let footer = parquet::read_metadata(&ctx, &path).unwrap();
                assert_eq!(footer.num_rows as usize, rows);
                assert_eq!(footer.num_row_groups, groups, "{rows} terms");
                let doc = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
                assert_eq!(doc.payload()["terms"].as_array().unwrap().len(), rows);
                assert_eq!(doc.payload()["num_qubits"], json!(qubits));
                // Saving the document again gives a file with the same table.
                // (Rows are sorted on the way out, and the first file was
                // written unsorted, so compare through a second save.)
                let again = dir.0.join(format!("{rows}-again.parquet"));
                let third = dir.0.join(format!("{rows}-third.parquet"));
                doc.save_parquet(&ctx, &again, Codec::Zstd).unwrap();
                let back = Document::read(&ctx, &again, HAMILTONIAN).unwrap();
                back.save_parquet(&ctx, &third, Codec::Zstd).unwrap();
                assert_eq!(fs::read(&again).unwrap(), fs::read(&third).unwrap());
                assert_eq!(back.payload()["terms"].as_array().unwrap().len(), rows);
                drop((doc, back));
                assert_eq!(ctx.memory_used(), 0, "{rows} terms");
                fs::remove_file(&path).unwrap();
                fs::remove_file(&again).unwrap();
                fs::remove_file(&third).unwrap();
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn document_reads_either_succeed_or_hit_the_budget_and_leave_nothing() {
        let ctx = context();
        let dir = Directory::new();
        let path = dir.0.join("h.parquet");
        {
            let table = big_table(&ctx, 500, 8);
            parquet::write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
        }
        let (mut ok, mut refused) = (0, 0);
        let mut limit = 1usize << 8;
        while limit < 1 << 27 {
            let small = Context::new(Some(1), limit).unwrap();
            match Document::read(&small, &path, HAMILTONIAN) {
                Ok(doc) => {
                    ok += 1;
                    assert_eq!(doc.payload()["terms"].as_array().unwrap().len(), 500);
                }
                Err(Error::Budget { .. }) => refused += 1,
                Err(other) => panic!("limit {limit}: {other:?}"),
            }
            assert_eq!(small.memory_used(), 0, "limit {limit}");
            limit += limit / 3 + 1;
        }
        assert!(ok > 0 && refused > 0, "{ok} ok, {refused} refused");
        assert_eq!(ctx.memory_used(), 0);
    }
}

/// Coefficients with random magnitudes and exponents, as text, the way the
/// Python writer hands them over (`json.dumps` of a float is its shortest
/// round-trip form).
fn random_terms(count: usize, seed: u64) -> (Vec<String>, Vec<f64>, Vec<f64>) {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut float = |scale: f64| {
        let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        unit * scale
    };
    let mut labels = Vec::new();
    let (mut real, mut imag) = (Vec::new(), Vec::new());
    for i in 0..count {
        let label: String = (0..9)
            .map(|d| ['I', 'X', 'Y', 'Z'][(i >> (2 * d)) & 3])
            .collect();
        labels.push(label);
        real.push(float(10.0f64.powi((i % 7) as i32 - 3)));
        imag.push(float(3.0));
    }
    (labels, real, imag)
}

#[test]
fn a_hamiltonian_keeps_every_coefficient_bit_for_bit_through_json_and_parquet() {
    // serde_json without `float_roundtrip` parses some decimal strings one
    // unit in the last place away from the double they spell; about one
    // coefficient in six used to change on the way to the Parquet file.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let dir = Directory::new();
    let (labels, real, imag) = random_terms(3_000, 0x9E37_79B9_7F4A_7C15);
    let terms: Vec<_> = labels
        .iter()
        .zip(real.iter().zip(&imag))
        .map(|(label, (re, im))| json!([label, re, im]))
        .collect();
    let text = json!({"format": persistence::HAMILTONIAN, "version": 1, "num_qubits": 9,
                      "mapping": "jordan_wigner", "metadata": {"energy_unit": "Ha"},
                      "terms": terms})
    .to_string();
    let document = Document::parse(&ctx, &text, persistence::HAMILTONIAN).unwrap();
    let path = dir.0.join("h.parquet");
    document
        .save_parquet(&ctx, &path, mandacaru_core::parquet::Codec::Zstd)
        .unwrap();
    let table = mandacaru_core::parquet::read_table(&ctx, &path).unwrap();
    let pauli = table.column("pauli").unwrap().as_utf8().unwrap();
    let got_real = table.column("real").unwrap().as_f64().unwrap();
    let got_imag = table.column("imag").unwrap().as_f64().unwrap();
    let mut expected: Vec<(&str, f64, f64)> = labels
        .iter()
        .zip(real.iter().zip(&imag))
        .map(|(label, (re, im))| (label.as_str(), *re, *im))
        .collect();
    expected.sort_by(|a, b| a.0.cmp(b.0));
    assert_eq!(pauli.len(), expected.len());
    let mut changed = 0;
    for (i, (label, re, im)) in expected.iter().enumerate() {
        assert_eq!(pauli.get(i), *label);
        changed += usize::from(got_real[i].to_bits() != re.to_bits());
        changed += usize::from(got_imag[i].to_bits() != im.to_bits());
    }
    assert_eq!(changed, 0, "{changed} coefficients changed by an ulp");
}

fn random_model(ctx: &Context, m: usize, seed: u64) -> mandacaru_core::molecular::transform::Model {
    let mut state = seed;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let one: Vec<C64> = (0..m * m).map(|_| C64::new(random(), random())).collect();
    let two: Vec<C64> = (0..m * m * m * m)
        .map(|_| C64::new(random(), random()))
        .collect();
    mandacaru_core::molecular::transform::Model::new(
        ctx,
        m,
        [m / 2, m / 2],
        ctx.copy_slice(&one).unwrap(),
        ctx.copy_slice(&two).unwrap(),
        -1.25,
        false,
    )
    .unwrap()
}

#[test]
fn a_model_written_straight_to_parquet_equals_its_document_written_to_parquet() {
    use mandacaru_core::parquet::Codec;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let dir = Directory::new();
    let model = random_model(&ctx, 5, 0x1234_5678_9ABC_DEF1);
    let direct = dir.0.join("direct.parquet");
    let staged = dir.0.join("staged.parquet");
    model
        .save_parquet(&ctx, &direct, "jordan_wigner", Codec::Zstd)
        .unwrap();
    model
        .document(&ctx, "jordan_wigner")
        .unwrap()
        .save_parquet(&ctx, &staged, Codec::Zstd)
        .unwrap();
    assert_eq!(fs::read(&direct).unwrap(), fs::read(&staged).unwrap());
    // And it loads as a document equal to the model's own.
    let loaded = Document::read(&ctx, &direct, persistence::HAMILTONIAN).unwrap();
    let own = model.document(&ctx, "jordan_wigner").unwrap();
    let sorted = |document: &Document| {
        let mut terms = document.payload()["terms"].as_array().unwrap().clone();
        terms.sort_by(|a, b| a[0].as_str().cmp(&b[0].as_str()));
        terms
    };
    assert_eq!(sorted(&loaded), sorted(&own));
    assert_eq!(
        loaded.payload()["metadata"]["groups"],
        own.payload()["metadata"]["groups"]
    );
}

#[test]
fn a_model_with_hundreds_of_thousands_of_terms_writes_as_parquet_bit_exact() {
    // A dense random model over 15 spatial orbitals (30 qubits): hundreds of
    // thousands of Pauli terms. Parquet is written straight from the model's
    // operator; no JSON tree of the terms (twice over, with the groups) is
    // built, so the peak is the operator's own working budget plus the columns.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let dir = Directory::new();
    let m = 15usize;
    let model = random_model(&ctx, m, 0x2545_F491_4F6C_DD1D);
    let path = dir.0.join("model.parquet");
    let mut expected = {
        let operator = model.operator(&ctx, "jordan_wigner").unwrap();
        operator.labels()
    };
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    let operator_peak = ctx.memory_peak();
    let held = ctx.memory_used();
    model
        .save_parquet(
            &ctx,
            &path,
            "jordan_wigner",
            mandacaru_core::parquet::Codec::Zstd,
        )
        .unwrap();
    assert_eq!(
        ctx.memory_used(),
        held,
        "the save left a reservation behind"
    );
    assert!(
        ctx.memory_peak() <= operator_peak + (96 << 20),
        "building the operator peaks at {operator_peak} bytes, the save at {}",
        ctx.memory_peak()
    );
    let footer = mandacaru_core::parquet::read_metadata(&ctx, &path).unwrap();
    assert!(footer.num_rows > 200_000, "only {} terms", footer.num_rows);
    // The file holds exactly the operator's terms, bit for bit.
    let table = mandacaru_core::parquet::read_table(&ctx, &path).unwrap();
    let pauli = table.column("pauli").unwrap().as_utf8().unwrap();
    let real = table.column("real").unwrap().as_f64().unwrap();
    let imag = table.column("imag").unwrap().as_f64().unwrap();
    assert_eq!(pauli.len(), expected.len());
    for (i, (label, c)) in expected.iter().enumerate() {
        assert_eq!(pauli.get(i), label);
        assert_eq!(real[i].to_bits(), c.re.to_bits());
        assert_eq!(imag[i].to_bits(), c.im.to_bits());
    }
}
