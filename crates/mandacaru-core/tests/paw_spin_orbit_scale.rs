//! The spin-orbit calibration of the generator (`paw::generate::spin_orbit_scale`, gate G13): the
//! scale is derived from a measurement, scales the stored `coupling` and nothing else, closes a
//! linear deficit in one step and is idempotent; the multiplets of the isolated atom are grouped
//! by their degeneracy and the pair is the highest one that holds electrons. The measurement
//! itself (the engine's isolated-atom SCF) is gated on the library in `paw_validate`.
//! Needs `MANDACARU_PAW_PATH` for a Dirac record (`lda-fr/Si`).
use mandacaru_core::paw::{
    atom_scf::{clusters, Level},
    engine_spin_orbit::{assign, Splitting, MULTIPLET},
    generate::spin_orbit_scale::{calibrate, CLOSED},
    record::PawDataset,
};
use mandacaru_core::Context;
use std::collections::BTreeMap;

fn selenium(ctx: &Context) -> PawDataset {
    let root = std::env::var("MANDACARU_PAW_PATH")
        .expect("MANDACARU_PAW_PATH names the library with a pbe-fr set");
    let path = std::path::Path::new(&root)
        .join("pbe-fr")
        .join("Se.parquet");
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn silicon(ctx: &Context) -> PawDataset {
    let root = std::env::var("MANDACARU_PAW_PATH")
        .expect("MANDACARU_PAW_PATH names the library with an lda-fr set");
    let path = std::path::Path::new(&root)
        .join("lda-fr")
        .join("Si.parquet");
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// An engine whose splitting is `deficit` of the Dirac one at the record's original coupling and
/// proportional to the coupling it holds now.
fn engine(
    original: &BTreeMap<u32, f64>,
    deficit: f64,
) -> impl Fn(&PawDataset, Option<f64>) -> mandacaru_core::Result<Vec<Splitting>> + '_ {
    move |ds, _floor| {
        Ok(ds
            .spin_orbit
            .iter()
            .map(|(&l, so)| Splitting {
                l: l as usize,
                state: 0,
                engine: Some(deficit * 1e-3 * so.coupling.data()[0] / original[&l]),
                dirac: 1e-3,
                width: 0.02,
                fractional: false,
            })
            .collect())
    }
}

#[test]
fn a_linear_deficit_is_closed_by_the_inverse_factor_and_the_diagnostic_arrays_stay() {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut ds = silicon(&ctx);
    let original: BTreeMap<u32, f64> = ds
        .spin_orbit
        .iter()
        .map(|(&l, so)| (l, so.coupling.data()[0]))
        .collect();
    let screened: Vec<Vec<f64>> = ds
        .spin_orbit
        .values()
        .map(|so| so.coupling_screened.data().to_vec())
        .collect();
    let measure = engine(&original, 0.8);
    let scales = calibrate(&mut ds, &measure).unwrap();
    assert!(!scales.is_empty());
    for scale in &scales {
        assert!((scale.factor - 1.25).abs() < 1e-9, "{scale:?}");
        assert!((scale.before + 0.2).abs() < 1e-12, "{scale:?}");
        assert!(scale.after.abs() < CLOSED, "{scale:?}");
        let now = ds.spin_orbit[&(scale.l as u32)].coupling.data()[0];
        assert!((now / original[&(scale.l as u32)] - 1.25).abs() < 1e-9);
    }
    // Only the operator the engine reads moves; the branch construction keeps its arrays.
    for (so, before) in ds.spin_orbit.values().zip(&screened) {
        assert_eq!(so.coupling_screened.data(), before.as_slice());
    }
    // A measurement is idempotent: a record already scaled keeps its coupling (the same engine, 0.8 of the original coupling's Dirac splitting).
    let again = calibrate(&mut ds, engine(&original, 0.8)).unwrap();
    assert!(again.iter().all(|s| s.factor == 1.), "{again:?}");
}

#[test]
fn a_shell_the_atom_does_not_show_keeps_its_coupling() {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut ds = silicon(&ctx);
    let before: Vec<f64> = ds
        .spin_orbit
        .values()
        .map(|so| so.coupling.data()[0])
        .collect();
    let scales = calibrate(&mut ds, |d, _floor| {
        Ok(d.spin_orbit
            .keys()
            .map(|&l| Splitting {
                l: l as usize,
                state: 0,
                engine: None,
                dirac: 1e-3,
                width: 0.02,
                fractional: false,
            })
            .collect())
    })
    .unwrap();
    assert!(scales.iter().all(|s| s.factor == 1. && s.before.is_nan()));
    let after: Vec<f64> = ds
        .spin_orbit
        .values()
        .map(|so| so.coupling.data()[0])
        .collect();
    assert_eq!(before, after);
}

fn level(energy: f64, occupation: f64) -> Level {
    Level { energy, occupation }
}

#[test]
fn multiplets_group_by_the_box_splitting_and_the_pair_is_the_highest_occupied_one() {
    let ev = 1. / 27.211_386_245_988;
    // A semicore p pair at -3 / -2.5 eV, the valence one at -1 / -0.5 eV (its p3/2 quartet split
    // by 3 meV by the box and only a quarter occupied), and an empty second-zeta pair above.
    let mut levels = vec![level(-3. * ev, 1.), level(-3. * ev, 1.)];
    levels.extend([-2.5, -2.498, -2.499, -2.497].map(|e| level(e * ev, 1.)));
    levels.extend([level(-1. * ev, 1.), level(-1. * ev, 1.)]);
    levels.extend([-0.5, -0.498, -0.499, -0.497].map(|e| level(e * ev, 0.25)));
    levels.extend([level(0.5 * ev, 0.), level(0.5 * ev, 0.)]);
    levels.extend([1., 1., 1., 1.].map(|e| level(e * ev, 0.)));
    let groups = clusters(&levels, MULTIPLET);
    let counts: Vec<usize> = groups.iter().map(|g| g.count).collect();
    assert_eq!(counts, vec![2, 4, 2, 4, 2, 4]);
    // Both occupied p pairs, lowest first, each by its Dirac splitting (0.5 eV): the semicore
    // one and the valence one.
    let found: Vec<f64> = assign(&groups, 1, &[0.5 * ev, 0.5 * ev])
        .iter()
        .map(|x| x.expect("a pair").0 / ev)
        .collect();
    assert!(
        (found[0] - 0.5).abs() < 1e-9 && (found[1] - 0.5).abs() < 1e-9,
        "{found:?}"
    );
    // A d shell has no (4, 6) pair here.
    assert_eq!(assign(&groups, 2, &[0.5 * ev]), vec![None]);
    // The cubic box splits d5/2 into 4 + 2 by 136 meV and an s level sits between d3/2 and d5/2:
    // the upper multiplet is searched for (not taken as the next group) and read at its centroid.
    let mut d = vec![level(-4. * ev, 1.); 4];
    d.extend(vec![level(-3.9 * ev, 1.); 2]);
    d.extend(vec![level(-3.5 * ev, 1.); 4]);
    d.extend(vec![level(-3.364 * ev, 1.); 2]);
    let found = assign(&clusters(&d, MULTIPLET), 2, &[0.55 * ev])[0]
        .expect("the d pair")
        .0
        / ev;
    assert!(
        (found - (4. * 0.5 + 2. * 0.636) / 6.).abs() < 1e-9,
        "{found}"
    );
}

/// An engine whose splitting of the first reference of every channel is `scale^power` of the Dirac
/// one for the coupling it holds relative to `original` (a shell answering a change of the coupling
/// with less than its full weight, as the 5d of an open-4f atom does), the `l >= 2` channels marked
/// as fractionally filled.
fn sluggish<'a>(
    original: &'a BTreeMap<u32, f64>,
    start: f64,
    power: f64,
) -> impl Fn(&PawDataset, Option<f64>) -> mandacaru_core::Result<Vec<Splitting>> + 'a {
    move |ds, _floor| {
        Ok(ds
            .spin_orbit
            .iter()
            .map(|(&l, so)| Splitting {
                l: l as usize,
                state: 0,
                engine: Some(start * 1e-3 * (so.coupling.data()[0] / original[&l]).powf(power)),
                dirac: 1e-3,
                width: 0.02,
                fractional: l >= 2,
            })
            .collect())
    }
}

#[test]
fn a_shell_that_answers_with_half_the_weight_closes_by_the_measured_slope() {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut ds = selenium(&ctx);
    let original: BTreeMap<u32, f64> = ds
        .spin_orbit
        .iter()
        .map(|(&l, so)| (l, so.coupling.data()[0]))
        .collect();
    // 1.5 times the Dirac splitting at the coupling as built, engine ~ coupling^0.5: the factor
    // that closes it is (1/1.5)^2; at slope one the loop would creep towards it.
    let scales = calibrate(&mut ds, sluggish(&original, 1.5, 0.5)).unwrap();
    for scale in &scales {
        assert!(!scale.given_up && scale.after.abs() < CLOSED, "{scale:?}");
    }
}

#[test]
fn an_open_d_shell_that_does_not_move_is_given_up_with_its_coupling_as_built() {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut ds = selenium(&ctx);
    let before: Vec<f64> = ds.spin_orbit[&2].coupling.data().to_vec();
    // Only the d channel is measured, 1.4 times the Dirac splitting, and the coupling does nothing
    // to it (an open shell filled j by j): after the rounds the shell is given up.
    let scales = calibrate(&mut ds, |d, _floor| {
        Ok(vec![Splitting {
            l: 2,
            state: 0,
            engine: Some(1.4e-3 + 0. * d.spin_orbit.len() as f64),
            dirac: 1e-3,
            width: 0.02,
            fractional: true,
        }])
    })
    .unwrap();
    assert!(
        scales[0].given_up && scales[0].factor == 1.,
        "{:?}",
        scales[0]
    );
    let after = ds.spin_orbit[&2].coupling.data();
    for (x, y) in before.iter().zip(after) {
        assert!((x - y).abs() <= 1e-12 * x.abs().max(1.), "{x} {y}");
    }
    // The same shell filled whole is an error, not a shell to give up.
    let error = calibrate(&mut ds, |_, _| {
        Ok(vec![Splitting {
            l: 2,
            state: 0,
            engine: Some(1.4e-3),
            dirac: 1e-3,
            width: 0.02,
            fractional: false,
        }])
    })
    .unwrap_err();
    assert!(error.to_string().contains("not closed"), "{error}");
}

#[test]
fn an_open_shell_that_was_only_measured_at_another_width_is_given_up_without_a_factor_to_take_back()
{
    // Round 0: the p shell is 4 percent low (it is stepped) and the d shell inside the gate. Round 1: the SCF
    // does not converge at the first width (an error), the ladder answers, and the d shell is now out
    // of the gate: it is not stepped at another width, so no factor of it exists when it is given up.
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let mut ds = selenium(&ctx);
    let calls = std::cell::Cell::new(0);
    let scales = calibrate(&mut ds, |_, width| {
        let round = calls.get();
        calls.set(round + 1);
        let shell = |l: usize, engine: f64, fractional: bool| Splitting {
            l,
            state: 0,
            engine: Some(engine * 1e-3),
            dirac: 1e-3,
            width: 0.02,
            fractional,
        };
        match (round, width) {
            (0, _) => Ok(vec![shell(1, 0.96, false), shell(2, 1.0, true)]),
            (1, Some(_)) => Err(mandacaru_core::Error::Invalid(
                "the isolated atom did not converge (Se) at Fermi-Dirac widths".into(),
            )),
            (_, None) => Ok(vec![shell(1, 1.0, false), shell(2, 1.4, true)]),
            _ => Ok(vec![shell(1, 1.0, false), shell(2, 1.4, true)]),
        }
    })
    .unwrap();
    let d = scales.iter().find(|s| s.l == 2).unwrap();
    assert!(d.given_up && d.factor == 1., "{d:?}");
    assert!(scales.iter().find(|s| s.l == 1).unwrap().factor > 1.);
}

/// E6-1 benchmark/verdict check on shipped 6p elements (`pbe-fr/Bi`, `pbe-fr/Pb`):
/// `(2j + 1)`-weighted branch averages (`so.average_screened`, `so.overlap_average`) recover the
/// scalar `p` reference level and channel spectrum closely without needing separate j-averaged
/// projectors, while `spin_orbit_scale` calibrates `so.coupling` above `-2/(2l+1) * difference` to
/// compensate the relativistic `6p1/2` radial contraction against a spin-free metric.
#[test]
fn the_6p_j_average_of_shipped_bi_and_pb_recovers_the_scalar_channel() {
    use mandacaru_core::paw::generate::spin_orbit::{lowest_level, weights};
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let root = std::env::var("MANDACARU_PAW_PATH")
        .expect("MANDACARU_PAW_PATH names the library with a pbe-fr set");
    for symbol in ["Bi", "Pb"] {
        let path = std::path::Path::new(&root)
            .join("pbe-fr")
            .join(format!("{symbol}.parquet"));
        let ds = PawDataset::read(&ctx, &path).unwrap();
        let ch = &ds.channels[&1];
        let so = &ds.spin_orbit[&1];
        let [w_down, w_up] = weights(1);
        let count = so.reference_energies.shape()[1];
        let e = so.reference_energies.data();
        let split_lowest = e[count] - e[0];
        let split_valence = e[2 * count - 1] - e[count - 1];
        let mean_lowest = (w_down * e[0] + w_up * e[count]) / (w_down + w_up);
        let mean_valence = (w_down * e[count - 1] + w_up * e[2 * count - 1]) / (w_down + w_up);
        let scalar_lowest = ch.reference_energies.data()[0];
        let scalar_valence = ch.reference_energies.data()[count - 1];
        let avg_tables: Vec<&[f64]> = so.projectors.iter().map(|p| p.as_slice()).collect();
        let ch_tables: Vec<&[f64]> = ch.projectors.iter().map(|p| p.as_slice()).collect();
        let average_level = lowest_level(
            &ctx,
            ds.r.as_slice(),
            ds.v_local_screened.as_slice(),
            1,
            &avg_tables,
            so.average_screened.data(),
            so.overlap_average.data(),
        )
        .unwrap();
        let scalar_level = lowest_level(
            &ctx,
            ds.r.as_slice(),
            ds.v_local_screened.as_slice(),
            1,
            &ch_tables,
            ch.coupling_screened.data(),
            ch.overlap_correction.data(),
        )
        .unwrap();
        let ref_ratio = (mean_valence - scalar_valence).abs() / split_valence.abs();
        let lvl_ratio = (average_level - scalar_level).abs() / split_lowest.abs();
        eprintln!(
            "{symbol} p (count={count}): lowest centroid miss {:.3e} / split {split_lowest:.4} Ha, valence ref_ratio={ref_ratio:.3e}, lvl_ratio={lvl_ratio:.3e}",
            (mean_lowest - scalar_lowest).abs() / split_lowest.abs(),
        );
        // For Z=82..83 (6p), the Dirac vs Koelling-Harmon centroid shifts by ~10% of the valence
        // splitting while the smooth (2j+1)-averaged operator still recovers the scalar level
        // within a few percent, and `spin_orbit_scale` closes the engine splitting via `so.coupling`.
        assert!(ref_ratio < 0.15, "{symbol} 6p ref_ratio={ref_ratio}");
        assert!(lvl_ratio < 0.08, "{symbol} lowest p lvl_ratio={lvl_ratio}");
        // Bi 6p calibrated `so.coupling` was scaled above uncalibrated `so.coupling_screened` block.
        if symbol == "Bi" {
            let max_raw = so
                .coupling_screened
                .data()
                .iter()
                .fold(0f64, |m, x| m.max(x.abs()));
            let max_cal = so.coupling.data().iter().fold(0f64, |m, x| m.max(x.abs()));
            assert!(max_cal > max_raw);
        }
    }
}
