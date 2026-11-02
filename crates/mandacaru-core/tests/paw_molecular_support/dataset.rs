//! A dataset of the shipped library as the species JSON of the molecular PAW
//! input: the same fields the Python adapter (`integrations/paw_molecular.py`)
//! collects, read here through `paw::record`.
use super::table;
use mandacaru_core::{
    paw::{molecular::Arrays, record::PawDataset},
    Context,
};
use serde_json::{json, Value};
use std::path::Path;

/// The library file `<MANDACARU_PAW_PATH>/<set>/<element>.parquet`; a missing
/// library or file FAILS the test, it never skips (the `paw_record*` rule).
pub fn read(ctx: &Context, set: &str, element: &str) -> PawDataset {
    let root = std::env::var("MANDACARU_PAW_PATH")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .expect("MANDACARU_PAW_PATH is not set: point it at the mandacaru-paw checkout");
    let path = Path::new(&root)
        .join(set)
        .join(format!("{element}.parquet"));
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The species JSON of `dataset`; its tables are inserted into `arrays`.
pub fn species(ctx: &Context, arrays: &mut Arrays, dataset: &PawDataset) -> Value {
    let flat = |rows: &[mandacaru_core::Buffer<f64>]| -> Vec<f64> {
        rows.iter()
            .flat_map(|row| row.as_slice().iter().copied())
            .collect()
    };
    let channels: Vec<Value> = dataset
        .channels
        .iter()
        .map(|(l, channel)| {
            let wavevectors: Vec<Value> = channel
                .wavevectors
                .iter()
                .map(|q| table(ctx, arrays, q.data()))
                .collect();
            let coefficients: Vec<Value> = channel
                .wave_coefficients
                .iter()
                .map(|c| table(ctx, arrays, c.data()))
                .collect();
            json!({
                "l": l,
                "principal_n": channel.n,
                "radius": channel.r_cut,
                "occupations": channel.occupations.data(),
                "ae": table(ctx, arrays, &flat(&channel.ae_waves)),
                "projectors": table(ctx, arrays, &flat(&channel.projectors)),
                "raw_projectors": table(ctx, arrays, &flat(&channel.raw_projectors)),
                "coupling": table(ctx, arrays, channel.coupling.data()),
                "overlap": table(ctx, arrays, channel.overlap_correction.data()),
                "vanderbilt": table(ctx, arrays, channel.vanderbilt.data()),
                "wavevectors": wavevectors,
                "wave_coefficients": coefficients,
            })
        })
        .collect();
    let n = dataset.r.as_slice().len();
    let xc_core = if dataset.nlcc.applied {
        dataset.smooth_core_density.as_slice().to_vec()
    } else {
        vec![0.; n]
    };
    json!({
        "atomic_number": dataset.atomic_number,
        "spin_orbit": false,
        "frozen_subshells": dataset.frozen_subshells,
        "semicore_subshells": dataset.semicore_subshells,
        "r": table(ctx, arrays, dataset.r.as_slice()),
        "ionic": table(ctx, arrays, dataset.v_local.as_slice()),
        "core": table(ctx, arrays, dataset.core_density.as_slice()),
        "valence": table(ctx, arrays, dataset.valence_density.as_slice()),
        "xc_core": table(ctx, arrays, &xc_core),
        "charge": dataset.valence_charge,
        "local_radius": dataset.r_cut_local,
        "compensation_radius": dataset.compensation_radius,
        "functional": dataset.xc,
        "one_center_energy": dataset.one_center_energy,
        "relativistic_exchange": dataset.relativistic_exchange,
        "channels": channels,
    })
}
