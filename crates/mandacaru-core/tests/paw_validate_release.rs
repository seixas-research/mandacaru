//! The generator never releases a file that fails a gate: `paw::generate::build::release` takes a
//! record, writes it beside its name, reads it back, runs `paw::validate` and moves it into place
//! only when no row refuses it. A damaged record is handed to it directly, so the refusal is
//! tested without a search that would repair the damage.
mod paw_validate_support;

use mandacaru_core::paw::{
    diagnose::reference::AeLogs,
    generate::{
        build::{release, Error, Format, Release},
        reference::solve_reference,
    },
    record::PawDataset,
    validate::Status,
};
use paw_validate_support::{context, damaged, ghosted, good, request, status, table, Folder};
use serde_json::json;

/// The records handed in are already the library's stride; the write keeps every node.
fn how(recorded_defects: bool) -> Release {
    Release {
        format: Format::Parquet,
        stride: 1,
        phases: false,
        recorded_defects,
        spin_orbit_width: None,
    }
}

#[test]
fn a_record_that_passes_every_gate_is_released_with_its_report() {
    let ctx = context();
    let folder = Folder::new("release_ok");
    let path = folder.0.join("Li.parquet");
    let (recipe, _) = request("Li");
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let released = release(
        &ctx,
        &good(&ctx),
        &atom,
        &mut AeLogs::new(),
        &path,
        how(false),
    )
    .unwrap();
    assert!(released.report.passed());
    assert_eq!(status(&released.report, "G1", "reload"), Status::Pass);
    assert_eq!(folder.names(), vec!["Li.parquet".to_string()]);
    assert_eq!(
        PawDataset::read(&ctx, &path)
            .unwrap()
            .content_sha256(&ctx)
            .unwrap(),
        released.loaded.content_sha256(&ctx).unwrap()
    );
}

#[test]
fn a_record_that_fails_a_gate_is_never_written() {
    let ctx = context();
    let (recipe, _) = request("Li");
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let ds = good(&ctx);
    let damages: Vec<(&str, &str, PawDataset)> = vec![
        (
            "G9",
            "against partial_core_electrons",
            damaged(&ctx, &ds, |_, tables| {
                for value in table(tables, "smooth_core_density") {
                    *value *= 1.1;
                }
            }),
        ),
        (
            "G5",
            "core density",
            damaged(&ctx, &ds, |_, tables| {
                let core = table(tables, "core_density");
                let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
                core[7] += 1e-6 * peak;
            }),
        ),
        (
            "G14",
            "ionic tail",
            damaged(&ctx, &ds, |_, tables| {
                let v = table(tables, "v_local");
                let half = v.len() / 2;
                for x in v[half..].iter_mut() {
                    *x *= 1.01;
                }
            }),
        ),
        (
            "G6",
            "duality",
            damaged(&ctx, &ds, |payload, _| {
                payload["channels"]["0"]["duality_error"] = json!(1e-6);
            }),
        ),
    ];
    for (gate, check, broken) in damages {
        let folder = Folder::new(&format!(
            "release_{gate}_{}",
            check.replace([' ', '-'], "_")
        ));
        let path = folder.0.join("Li.parquet");
        let refused = release(&ctx, &broken, &atom, &mut AeLogs::new(), &path, how(false));
        let Err(Error::Gates(report)) = refused else {
            panic!("{gate} {check}: the damaged record must be refused, got {refused:?}");
        };
        assert!(!report.passed());
        assert_eq!(status(&report, gate, check), Status::Fail, "{gate} {check}");
        let text = Error::Gates(report).to_string();
        assert!(
            text.contains(gate) && text.contains("not written"),
            "{text}"
        );
        assert!(
            folder.names().is_empty(),
            "{gate} {check}: left behind {:?}",
            folder.names()
        );
    }
}

#[test]
fn a_ghost_is_refused_unless_the_file_records_it() {
    let ctx = context();
    let (recipe, _) = request("Li");
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let (ghost, _) = ghosted(&ctx, &good(&ctx));
    let folder = Folder::new("release_ghost");
    let path = folder.0.join("Li.parquet");
    let refused = release(&ctx, &ghost, &atom, &mut AeLogs::new(), &path, how(false));
    assert!(matches!(refused, Err(Error::Gates(_))), "{refused:?}");
    assert!(folder.names().is_empty());
    // `ghosts="flag"`: the same construction with its defect recorded in the file is written, the
    // rows that fail on it waived; a defect the record does not carry still refuses.
    let flagged = damaged(&ctx, &ghost, |payload, _| {
        payload["defects"]["ghosts"] = json!({"0": -1.0});
    });
    let refused = release(&ctx, &ghost, &atom, &mut AeLogs::new(), &path, how(true));
    assert!(
        matches!(refused, Err(Error::Gates(_))),
        "an unrecorded defect is not waived"
    );
    let released = release(&ctx, &flagged, &atom, &mut AeLogs::new(), &path, how(true)).unwrap();
    assert!(released.report.passed());
    assert_eq!(
        status(&released.report, "G2", "channels with projectors"),
        Status::Waived
    );
    assert_eq!(folder.names(), vec!["Li.parquet".to_string()]);
}
