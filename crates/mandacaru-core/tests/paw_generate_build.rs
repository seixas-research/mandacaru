//! `paw::generate::build`: generate, write, read back, check and move into place, the whole of
//! one element of `mandacaru-build`, with the record never leaving Rust.

use mandacaru_core::paw::generate::{
    build::{build, Error, Format},
    request::{resolve, Overrides},
    search::{Mode, Options},
};
use mandacaru_core::paw::record::{PawDataset, FORMAT_VERSION};
use mandacaru_core::Context;
use std::path::PathBuf;

/// A fresh folder under the system's temporary directory, removed on drop.
struct Folder(PathBuf);

impl Folder {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mandacaru_paw_generate_build_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn names(&self) -> Vec<String> {
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

fn request(symbol: &str) -> (mandacaru_core::paw::generate::recipe::Recipe, Options) {
    let (recipe, pinned) = resolve(
        symbol,
        mandacaru_core::paw::record::Family::PawLcao,
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

#[test]
fn hydrogen_is_written_read_back_and_checked_in_place() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let folder = Folder::new("h");
    let path = folder.0.join("H.parquet");
    let (recipe, options) = request("H");
    let built = build(&ctx, &recipe, &options, &path, Format::Parquet, 4, true).unwrap();
    assert_eq!(built.path, path);
    assert_eq!(
        folder.names(),
        vec!["H.parquet".to_string()],
        "nothing staged is left"
    );
    assert_eq!(built.loaded.format_version(), FORMAT_VERSION);
    assert_eq!(built.loaded.core_states.as_ref().map(Vec::len), Some(0));
    assert!(!built.channels.is_empty());
    for channel in &built.channels {
        assert!(!channel.ghost, "l={} holds a ghost", channel.l);
        assert!(channel.phase.is_some(), "l={} has no phase", channel.l);
        assert!(channel.lowest.is_finite() && channel.r_cut > 0.);
    }
    // The report is of the file: reading it again gives the record that was checked.
    let again = PawDataset::read(&ctx, &path).unwrap();
    assert_eq!(
        again.content_sha256(&ctx).unwrap(),
        built.loaded.content_sha256(&ctx).unwrap()
    );
}

#[test]
fn a_cored_element_reaches_the_file_with_its_core_states() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let folder = Folder::new("li");
    let path = folder.0.join("Li.parquet");
    let (recipe, options) = request("Li");
    let built = build(&ctx, &recipe, &options, &path, Format::Parquet, 4, false).unwrap();
    let back = PawDataset::read(&ctx, &path).unwrap();
    let states = back
        .core_states
        .as_ref()
        .expect("format 3 carries the core");
    assert_eq!(states.len(), 1);
    assert_eq!((states[0].n, states[0].l), (1, 0));
    assert_eq!(states[0].occupation, 2.);
    assert!(built.channels.iter().all(|c| c.phase.is_none()));
}

#[test]
fn the_build_checks_the_file_on_the_atom_the_search_used() {
    use mandacaru_core::paw::diagnose::{diagnose_record_with_unprojected, reference::AeLogs};
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let folder = Folder::new("li-diagnosis");
    let path = folder.0.join("Li.parquet");
    let (recipe, options) = request("Li");
    let built = build(&ctx, &recipe, &options, &path, Format::Parquet, 4, true).unwrap();
    // The reference atom is solved once per element. Checking the file on a second, independent
    // solve of the same atom gives the same answer to the last bit: the shared atom changes no check.
    let (again, missing) =
        diagnose_record_with_unprojected(&ctx, &built.loaded, &mut AeLogs::new()).unwrap();
    assert!(built.diagnosis.is_clean());
    assert_eq!(built.diagnosis.ghosts, again.ghosts);
    assert_eq!(built.diagnosis.phases, again.phases);
    assert_eq!(built.diagnosis.unprojected, again.unprojected);
    assert_eq!(built.diagnosis.missing, again.missing);
    assert_eq!(built.unprojected_missing, missing);
}

#[test]
fn nothing_is_left_beside_the_path_when_the_build_fails() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let folder = Folder::new("fail");
    // A stride that keeps a few dozen points of the radial grid: the file cannot reproduce the
    // generated levels (or cannot even be checked), and nothing may be released.
    let path = folder.0.join("H.parquet");
    let (recipe, options) = request("H");
    let result = build(&ctx, &recipe, &options, &path, Format::Parquet, 64, false);
    match &result {
        Err(Error::Reload(moved)) => assert!(!moved.is_empty()),
        Err(Error::Failed(error)) => println!("refused before the level check: {error}"),
        other => panic!("a 64-fold stride must not be released: {other:?}"),
    }
    assert!(
        folder.names().is_empty(),
        "left behind: {:?}",
        folder.names()
    );
    // A parent that is a file, not a folder: the write cannot happen, and only that file is left.
    let blocker = folder.0.join("blocker");
    std::fs::write(&blocker, b"not a folder").unwrap();
    let blocked = build(
        &ctx,
        &recipe,
        &options,
        &blocker.join("H.parquet"),
        Format::Parquet,
        4,
        false,
    );
    assert!(
        matches!(blocked, Err(Error::Failed(_))),
        "a write under a file must fail: {blocked:?}"
    );
    assert_eq!(folder.names(), vec!["blocker".to_string()]);
}
