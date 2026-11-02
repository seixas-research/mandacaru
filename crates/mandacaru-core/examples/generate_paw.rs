//! Generate a PAW-LCAO (or UPAW-LCAO) dataset without Python and print its headline numbers.
//!
//! `cargo run --release -p mandacaru-core --example generate_paw -- O lda [paw|upaw] [none|scalar] [output.parquet]`
use mandacaru_core::{
    models::relativity::Treatment,
    parquet::Codec,
    paw::{
        generate::{generate, recipe::Recipe},
        record::{PawDataset, WriteSpec},
    },
    Context,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let symbol = args.get(1).map_or("H", String::as_str);
    let functional = args.get(2).map_or("lda", String::as_str);
    let family = args.get(3).map_or("paw", String::as_str);
    let relativity = Treatment::parse(args.get(4).map_or("scalar", String::as_str))?;
    let recipe = match family {
        "paw" => Recipe::for_dataset(symbol, functional, relativity)?,
        "upaw" => Recipe::upaw(symbol, functional, relativity)?,
        other => return Err(format!("the family is paw or upaw, not {other:?}").into()),
    };
    let ctx = Context::new(Some(1), 1 << 30)?;
    let generated = generate(&ctx, &recipe)?;
    let dataset: &PawDataset = &generated.dataset;
    println!(
        "{}",
        serde_json::json!({
            "symbol": dataset.symbol,
            "family": dataset.family.name(),
            "xc": dataset.xc,
            "relativity": dataset.relativity,
            "valence_charge": dataset.valence_charge,
            "compensation_charge": dataset.compensation_charge,
            "one_center_energy_ha": dataset.one_center_energy,
            "atom_energy_ha": dataset.energies["atom_total"],
            "channels": dataset.channels.iter().map(|(l, c)| serde_json::json!({
                "l": l, "r_cut_bohr": c.r_cut, "duality_error": c.duality_error,
                "asymmetry": c.asymmetry, "overlap_minimum": c.overlap_minimum,
                "reference_energies_ha": c.reference_energies.data(),
            })).collect::<Vec<_>>(),
            "notes": generated.notes,
        })
    );
    if let Some(path) = args.get(5) {
        dataset.write(
            &ctx,
            std::path::Path::new(path),
            &WriteSpec {
                stride: 4,
                codec: Codec::Zstd,
            },
        )?;
    }
    Ok(())
}
