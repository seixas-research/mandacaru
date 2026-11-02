//! The species tables of the typed record (`paw::species_tables`): the smooth
//! waves of the dual basis against the pseudo waves the record stores, and the basis invariance
//! of the frozen constant `X0 - V . D0` between the raw and the dual projector basis.
//!
//! Reads the shipped library: `MANDACARU_PAW_PATH`; a missing one fails the tests.
use mandacaru_core::{
    paw::{
        onecenter::Basis, pair_tables::layout, record::PawDataset, screened_atom::screened,
        species_tables::SpeciesTables,
    },
    Context,
};
use std::path::PathBuf;

fn read(ctx: &Context, set: &str, symbol: &str) -> PawDataset {
    let root = std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .expect("MANDACARU_PAW_PATH is unset");
    let path = PathBuf::from(root).join(format!("{set}/{symbol}.parquet"));
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn the_dual_smooth_waves_are_the_records_pseudo_waves_and_the_constant_is_basis_independent() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    for (set, symbol) in [("pbe-sr", "O"), ("lda-sr", "Si"), ("lda-sr", "H")] {
        let record = read(&ctx, set, symbol);
        let dual = SpeciesTables::from_record(&ctx, &record, Basis::Dual).unwrap();
        let view = dual.view();
        let mut worst = 0f64;
        for (channel, (_, stored)) in view.channels.iter().zip(&record.channels) {
            for (wave, pseudo) in channel.ps.iter().zip(&stored.pseudo_waves) {
                let scale = pseudo.as_slice().iter().fold(0f64, |a, v| a.max(v.abs()));
                for (k, (a, b)) in wave.iter().zip(pseudo.as_slice()).enumerate() {
                    if record.r.as_slice()[k] <= stored.r_cut {
                        worst = worst.max((a - b).abs() / scale);
                    }
                }
            }
        }
        println!(
            "{symbol}: Bessel-expanded smooth waves against the stored pseudo waves {worst:.2e}"
        );
        assert!(worst < 1e-6, "{symbol}");

        let omega = 0.11;
        let raw = SpeciesTables::from_record(&ctx, &record, Basis::Raw).unwrap();
        let (raw_view, dual_view) = (raw.view(), dual.view());
        let x_raw = screened(&ctx, &raw_view, &layout(&raw_view), omega).unwrap();
        let x_dual = screened(&ctx, &dual_view, &layout(&dual_view), omega).unwrap();
        println!(
            "{symbol}: X0 - V.D0 raw {:.10} dual {:.10}",
            x_raw.constant, x_dual.constant
        );
        assert!((x_raw.constant - x_dual.constant).abs() < 1e-8 * x_raw.constant.abs().max(1.));
    }
}
