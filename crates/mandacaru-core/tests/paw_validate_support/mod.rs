//! What the validation tests share: a generated lithium file (built once per test binary, with
//! its core states, a partial-core correction and a clean build report), and a way to damage a
//! record through its public parts (payload text and radial tables), so every deliberately broken
//! record goes through the reader's own refusals first.
#![allow(dead_code)]

use mandacaru_core::{
    paw::{
        generate::{
            build::{build, Format},
            request::{resolve, Overrides},
            search::{Mode, Options},
        },
        record::{Family, PawDataset},
        validate::{GateReport, Row, Status},
    },
    Context, Result,
};
use serde_json::Value;
use std::{path::PathBuf, sync::OnceLock};

/// A fresh folder under the system temporary directory, removed on drop.
pub struct Folder(pub PathBuf);

impl Folder {
    pub fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mandacaru_paw_validate_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn context() -> Context {
    Context::new(Some(2), 2 << 30).unwrap()
}

/// The recipe and search options of an LDA scalar-relativistic element.
pub fn request(symbol: &str) -> (mandacaru_core::paw::generate::recipe::Recipe, Options) {
    let (recipe, pinned) = resolve(
        symbol,
        Family::PawLcao,
        "lda",
        "scalar",
        Overrides::default(),
    )
    .unwrap();
    (
        recipe,
        Options {
            mode: Mode::Repair,
            pinned,
        },
    )
}

struct Fixture {
    bytes: Vec<u8>,
    report: GateReport,
}

static LITHIUM: OnceLock<Fixture> = OnceLock::new();

fn lithium() -> &'static Fixture {
    LITHIUM.get_or_init(|| {
        let ctx = context();
        let folder = Folder::new("fixture");
        let path = folder.0.join("Li.parquet");
        let (recipe, options) = request("Li");
        let built = build(&ctx, &recipe, &options, &path, Format::Parquet, 4, true)
            .expect("the lithium fixture builds and passes every gate");
        Fixture {
            bytes: std::fs::read(&path).unwrap(),
            report: built.report,
        }
    })
}

/// The gate report the build of the fixture produced.
pub fn build_report() -> &'static GateReport {
    &lithium().report
}

/// The fixture file read back: a known-good record.
pub fn good(ctx: &Context) -> PawDataset {
    let folder =
        Folder::new(&format!("read_{:?}", std::thread::current().id()).replace(['(', ')'], ""));
    let path = folder.0.join("Li.parquet");
    std::fs::write(&path, &lithium().bytes).unwrap();
    PawDataset::read(ctx, &path).unwrap()
}

/// `dataset` with its payload and tables edited by `edit`, read again through the file reader's
/// own checks.
pub fn damaged(
    ctx: &Context,
    dataset: &PawDataset,
    edit: impl FnOnce(&mut Value, &mut Vec<(String, Vec<f64>)>),
) -> PawDataset {
    try_damaged(ctx, dataset, edit).expect("the damaged record is still a readable record")
}

pub fn try_damaged(
    ctx: &Context,
    dataset: &PawDataset,
    edit: impl FnOnce(&mut Value, &mut Vec<(String, Vec<f64>)>),
) -> Result<PawDataset> {
    let parts = dataset.parts(ctx)?;
    let mut payload: Value = serde_json::from_str(&parts.text).unwrap();
    let mut tables: Vec<(String, Vec<f64>)> = parts
        .tables
        .iter()
        .map(|(name, values)| (name.clone(), values.to_vec()))
        .collect();
    edit(&mut payload, &mut tables);
    let columns = tables
        .into_iter()
        .map(|(name, values)| Ok((name, ctx.copy_slice(&values)?)))
        .collect::<Result<Vec<_>>>()?;
    PawDataset::from_parts(ctx, &payload.to_string(), columns)
}

pub fn table<'a>(tables: &'a mut [(String, Vec<f64>)], name: &str) -> &'a mut Vec<f64> {
    &mut tables
        .iter_mut()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no table {name}"))
        .1
}

/// The one row of `gate` whose check contains `part`.
pub fn row<'a>(report: &'a GateReport, gate: &str, part: &str) -> &'a Row {
    let found: Vec<_> = report
        .gate(gate)
        .into_iter()
        .filter(|r| r.check.contains(part))
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{gate} rows containing {part:?}: {:?}\nall rows:\n{}",
        found.iter().map(|r| &r.check).collect::<Vec<_>>(),
        report.lines().join("\n")
    );
    found[0]
}

pub fn status(report: &GateReport, gate: &str, part: &str) -> Status {
    row(report, gate, part).status
}

/// `exp(-(r / 0.45)^2)` wells (`depth > 0`) and bumps on the screened local potential, `depth`
/// Hartree deep.
pub fn with_well(ctx: &Context, dataset: &PawDataset, depth: f64) -> PawDataset {
    damaged(ctx, dataset, |_, tables| {
        let r = table(tables, "r").clone();
        for (v, r) in table(tables, "v_local_screened").iter_mut().zip(r) {
            *v -= depth * (-(r / 0.45).powi(2)).exp();
        }
    })
}

/// The first well of 4, 8, 16, ... Hartree that binds a ghost in `dataset`, with its depth.
pub fn ghosted(ctx: &Context, dataset: &PawDataset) -> (PawDataset, f64) {
    use mandacaru_core::paw::diagnose::{diagnose_record, reference::AeLogs};
    (1..8)
        .map(|k| 2f64.powi(k + 1))
        .find_map(|depth| {
            let ghosted = with_well(ctx, dataset, depth);
            let found = diagnose_record(ctx, &ghosted, None, &mut AeLogs::new()).ok()?;
            found.has_ghost().then_some((ghosted, depth))
        })
        .expect("some well binds a ghost in the s channel")
}
