//! A molecule over a format-2 cored PAW record (no core states): Kohn-Sham never needs the core
//! states, so it runs, and its energy and force are those of the same dataset in format 3 to
//! rounding (1e-12 Ha, 1e-10 Ha/Bohr); every route that needs the core-valence exchange refuses,
//! by name. The fixture is `lda-sr/O` of the library (format 3), written back without its core
//! states (format 2, the same content digest). Water in a 6 A box, grid_spacing 0.25 A (24 nodes
//! per axis), SZ, raw projectors. Needs `MANDACARU_PAW_PATH`.
mod paw_record_support;
mod paw_supercell_support;
mod setup_shells_support;

use mandacaru_core::{
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    parquet::Codec,
    paw::record::{PawDataset, WriteSpec},
    Context,
};
use paw_record_support::{dataset_path, TempDir};
use paw_supercell_support::{molecular_integrals, molecule_in_a_box};

const SIDE: f64 = 6. / 0.529_177_210_903;
const NODES: usize = 24;
const STORED: WriteSpec = WriteSpec {
    stride: 1,
    codec: Codec::Zstd,
};

/// A library root holding `lda-sr/H` as shipped and `lda-sr/O` as shipped (`strip` false) or
/// without its core states (`strip` true).
fn library_root(ctx: &Context, label: &str, strip: bool) -> TempDir {
    let dir = TempDir::new(label);
    std::fs::create_dir(dir.file("lda-sr")).unwrap();
    std::fs::copy(dataset_path("lda-sr", "H"), dir.file("lda-sr/H.parquet")).unwrap();
    let mut oxygen = PawDataset::read(ctx, &dataset_path("lda-sr", "O")).unwrap();
    assert!(
        oxygen.core_states.is_some(),
        "the fixture is a format-3 record"
    );
    if strip {
        oxygen.core_states = None;
        oxygen.core_scheme = None;
    }
    oxygen
        .write(ctx, &dir.file("lda-sr/O.parquet"), &STORED)
        .unwrap();
    dir
}

/// Water over the library at `root`: its Hamiltonian, and the converged LDA state and force.
fn water(ctx: &Context, root: &TempDir) -> MolecularIntegrals {
    std::env::set_var("MANDACARU_PAW_PATH", root.file(""));
    let (r, angle) = (1.809, 104.5f64.to_radians());
    let atoms = [
        ("O", [0., 0., 0.]),
        ("H", [r * (angle / 2.).sin(), 0., r * (angle / 2.).cos()]),
        ("H", [-r * (angle / 2.).sin(), 0., r * (angle / 2.).cos()]),
    ];
    molecular_integrals(ctx, &molecule_in_a_box(ctx, &atoms, SIDE, NODES))
}

fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-10,
        max_iterations: 200,
        ..Default::default()
    }
}

fn force(integrals: &MolecularIntegrals, state: &MeanField) -> Vec<f64> {
    integrals.gradient(state).unwrap().total.as_slice().to_vec()
}

#[test]
fn a_format_two_core_runs_kohn_sham_like_its_format_three_form_and_refuses_what_needs_its_core_states(
) {
    let ctx = Context::new(Some(3), 1800 << 20).unwrap();
    let original = std::env::var_os("MANDACARU_PAW_PATH");
    let (v3, v2) = (
        library_root(&ctx, "water-v3", false),
        library_root(&ctx, "water-v2", true),
    );
    let before = PawDataset::read(&ctx, &v3.file("lda-sr/O.parquet")).unwrap();
    let after = PawDataset::read(&ctx, &v2.file("lda-sr/O.parquet")).unwrap();
    assert_eq!((before.format_version(), after.format_version()), (3, 2));
    assert_eq!(
        before.content_sha256(&ctx).unwrap(),
        after.content_sha256(&ctx).unwrap()
    );
    drop((before, after));

    let three = water(&ctx, &v3);
    let two = water(&ctx, &v2);
    let state3 = three.kohn_sham("lda", [4, 4], false, &options()).unwrap();
    let state2 = two.kohn_sham("lda", [4, 4], false, &options()).unwrap();
    assert!(state3.converged && state2.converged);
    let (f3, f2) = (force(&three, &state3), force(&two, &state2));
    let de = (state3.electronic_energy - state2.electronic_energy).abs();
    let df = f3
        .iter()
        .zip(&f2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    println!(
        "water LDA: format 3 {:.15} format 2 {:.15}, |dE| {de:e} Ha, max |dF| {df:e} Ha/Bohr",
        state3.electronic_energy, state2.electronic_energy
    );
    // The same Hamiltonian: the one-body Kohn-Sham correction is built as the Kohn-Sham coupling
    // less the Hartree-Fock one (`onecenter::build`), where `core_exchange`, never built for a
    // format-2 core, cancels in exact arithmetic; the builds agree to the rounding of that
    // cancellation (measured 5e-14 Ha, 4e-13 Ha/Bohr).
    assert!(de < 1e-12 && df < 1e-10, "{de:e} {df:e}");

    // Hartree-Fock, a Hamiltonian over the frozen core and the screened hybrid refuse, by name.
    for method in ["rhf", "uhf"] {
        let error = two
            .mean_field(method, [4, 4], &options())
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("O: the dataset carries no core states"),
            "{method}: {error}"
        );
        assert!(
            three.mean_field(method, [4, 4], &options()).is_ok(),
            "{method}"
        );
    }
    let error = state2.model(&two).err().unwrap().to_string();
    assert!(
        error.contains("O: the dataset carries no core states"),
        "{error}"
    );
    assert!(state3.model(&three).is_ok());
    let error = state2
        .active_model(&two, r#"{"orbitals": 4}"#)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("O: the dataset carries no core states"), "{error}");
    let error = two
        .kohn_sham("hse06", [4, 4], false, &options())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("hse06 takes"), "{error}");

    match original {
        Some(value) => std::env::set_var("MANDACARU_PAW_PATH", value),
        None => std::env::remove_var("MANDACARU_PAW_PATH"),
    }
}
