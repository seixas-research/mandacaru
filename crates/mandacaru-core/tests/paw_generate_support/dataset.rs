//! The generated dataset against the frozen one: every array of the freeze (pointwise on the
//! decimated points), the references against the inputs the Python assemble was called with, and
//! every scalar and diagnostic.
use super::compare::{
    gate_of, nan_max, number, relative, tail_allowance, Comparison, Scheme, GATE, SCALAR_GATE,
};
use super::{arrays, counterpart, Decimated};
use crate::paw_assemble_support::Frozen;
use mandacaru_core::paw::generate::Generated;
use serde_json::Value;

pub fn compare_dataset(
    f: &Frozen,
    data: &Decimated,
    generated: &Generated,
    scheme: Scheme,
) -> Comparison {
    let mut out = Comparison::default();
    let dataset = &generated.dataset;
    let ours = arrays(dataset);
    let pbe = dataset.xc == "pbe";
    for (key, spec) in f.doc["arrays"].as_object().unwrap() {
        let mine = ours
            .get(key)
            .unwrap_or_else(|| panic!("the generated dataset has no array {key}"));
        let expected = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .product::<usize>();
        assert_eq!(mine.len(), expected, "{key}: shape");
        let frozen = counterpart(f, data, key, "d_", spec);
        if key == "v_local" {
            let rho = ours["valence_density"]
                .iter()
                .zip(&ours["smooth_core_density"])
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let worst = frozen
                .values
                .iter()
                .enumerate()
                .map(|(k, b)| {
                    let at = (k + 1) * frozen.stride - 1;
                    (mine[at] - b).abs() / tail_allowance(scheme, pbe, rho[at], frozen.scale)
                })
                .fold(0., nan_max);
            out.push(format!("array.{key}"), worst * GATE, GATE);
        } else {
            out.push(
                format!("array.{key}"),
                frozen.deviation(mine),
                gate_of(f, key),
            );
        }
    }

    // The references and the screened local potential against the inputs the Python assemble
    // was called with (full resolution): the handoff between this generator and A2.
    let cases = f.cases("assemble_paw_channel");
    let mut seen = std::collections::BTreeSet::new();
    for (key, case) in cases {
        let pw = &case["inputs"]["pw"];
        let l = pw["l"].as_u64().unwrap() as u32;
        if !seen.insert(l) {
            continue;
        }
        let channel = &dataset.channels[&l];
        let scale = |v: &[f64]| v.iter().fold(0., |m: f64, x| m.max(x.abs())).max(1e-300);
        let worst = |a: &[f64], b: &[f64]| {
            assert_eq!(a.len(), b.len(), "{key}: length");
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0., nan_max)
                / scale(b)
        };
        let energies = pw["energies"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect::<Vec<_>>();
        out.push(
            format!("handoff.l{l}.energies"),
            worst(channel.reference_energies.data(), &energies),
            GATE,
        );
        out.push(
            format!("handoff.l{l}.r_cut"),
            (channel.r_cut - number(&pw["r_cut"])).abs(),
            1e-14,
        );
        for (i, spec) in pw["waves"].as_array().unwrap().iter().enumerate() {
            out.push(
                format!("handoff.l{l}.ae_wave_{i}"),
                worst(channel.ae_waves[i].as_slice(), &f.input(spec)),
                GATE,
            );
        }
        for (i, spec) in pw["coefficients"].as_array().unwrap().iter().enumerate() {
            out.push(
                format!("handoff.l{l}.coefficients_{i}"),
                worst(channel.wave_coefficients[i].data(), &f.input(spec)),
                1e-11,
            );
        }
        for (i, spec) in pw["wavevectors"].as_array().unwrap().iter().enumerate() {
            out.push(
                format!("handoff.l{l}.wavevectors_{i}"),
                worst(channel.wavevectors[i].data(), &f.input(spec)),
                GATE,
            );
        }
        let v_loc = f.input(&case["inputs"]["v_loc"]);
        out.push(
            format!("handoff.l{l}.v_loc"),
            worst(dataset.v_local_screened.as_slice(), &v_loc),
            GATE,
        );
        out.push(
            format!("handoff.l{l}.r_local"),
            (dataset.r_cut_local - number(&case["inputs"]["r_local"])).abs(),
            1e-14,
        );
    }

    let s = &f.doc["scalars"];
    for (name, mine) in [
        ("one_center_energy", dataset.one_center_energy),
        ("compensation_charge", dataset.compensation_charge),
        ("hartree_screening", dataset.hartree_screening),
        ("valence_charge", dataset.valence_charge),
    ] {
        out.push(
            format!("scalar.{name}"),
            relative(mine, number(&s[name])),
            SCALAR_GATE,
        );
    }
    for (name, mine) in [
        ("r_cut_local", dataset.r_cut_local),
        ("local_shift", dataset.local_shift),
        ("compensation_radius", dataset.compensation_radius),
        ("q_cut", dataset.q_cut),
        ("energy_offset", dataset.energy_offset),
        ("norm_deficit", dataset.norm_deficit),
    ] {
        out.push(
            format!("scalar.{name}"),
            (mine - number(&s[name])).abs(),
            1e-14,
        );
    }
    for (name, mine) in &dataset.energies {
        out.push(
            format!("energy.{name}"),
            relative(*mine, number(&s["energies"][name.as_str()])),
            SCALAR_GATE,
        );
    }
    assert_eq!(
        dataset.energies.len(),
        s["energies"].as_object().unwrap().len(),
        "energies keys"
    );
    out.push(
        "scalar.extra_l",
        (dataset.extra_l as f64 - number(&s["extra_l"])).abs(),
        0.5,
    );
    let listed = |v: &Value| -> Vec<Vec<i64>> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .collect()
    };
    out.push(
        "scalar.frozen_subshells",
        f64::from(dataset.frozen_subshells != listed(&s["frozen_subshells"])),
        0.5,
    );
    out.push(
        "scalar.semicore_subshells",
        f64::from(dataset.semicore_subshells != listed(&s["semicore_subshells"])),
        0.5,
    );
    out.push(
        "scalar.xc",
        f64::from(dataset.xc != s["xc"].as_str().unwrap()),
        0.5,
    );
    out.push(
        "scalar.relativity",
        f64::from(dataset.relativity != s["relativity"].as_str().unwrap()),
        0.5,
    );
    out.push(
        "scalar.relativistic_exchange",
        f64::from(dataset.relativistic_exchange != s["relativistic_exchange"].as_bool().unwrap()),
        0.5,
    );
    out.push(
        "scalar.family",
        f64::from(dataset.family.name() != s["family"].as_str().unwrap()),
        0.5,
    );
    out.push(
        "scalar.symbol",
        f64::from(dataset.symbol != s["symbol"].as_str().unwrap()),
        0.5,
    );

    let nlcc = &s["nlcc"];
    out.push(
        "nlcc.applied",
        f64::from(dataset.nlcc.applied != nlcc["applied"].as_bool().unwrap()),
        0.5,
    );
    for (name, mine) in [
        ("r_nlcc", dataset.nlcc.r_nlcc),
        ("core_electrons", dataset.nlcc.core_electrons),
        (
            "partial_core_electrons",
            dataset.nlcc.partial_core_electrons,
        ),
    ] {
        match (mine, nlcc.get(name).and_then(Value::as_f64)) {
            (Some(a), Some(b)) => out.push(format!("nlcc.{name}"), relative(a, b), SCALAR_GATE),
            (None, None) => {}
            _ => out.push(format!("nlcc.{name}"), f64::INFINITY, 0.5),
        }
    }

    let channels = s["channels"].as_object().unwrap();
    out.push(
        "channels.count",
        (dataset.channels.len() as f64 - channels.len() as f64).abs(),
        0.5,
    );
    for (l, frozen) in channels {
        let channel = &dataset.channels[&l.parse::<u32>().unwrap()];
        out.push(
            format!("channel.l{l}.n"),
            (f64::from(channel.n) - number(&frozen["n"])).abs(),
            0.5,
        );
        out.push(
            format!("channel.l{l}.r_cut"),
            (channel.r_cut - number(&frozen["r_cut"])).abs(),
            1e-14,
        );
        out.push(
            format!("channel.l{l}.occupation"),
            (channel.occupation - number(&frozen["occupation"])).abs(),
            1e-14,
        );
        out.push(
            format!("channel.l{l}.q_cut"),
            (channel.q_cut - number(&frozen["q_cut"])).abs(),
            1e-14,
        );
        out.push(
            format!("channel.l{l}.overlap_minimum"),
            relative(channel.overlap_minimum, number(&frozen["overlap_minimum"])),
            SCALAR_GATE,
        );
        out.push(
            format!("channel.l{l}.projectors"),
            (channel.projectors.len() as f64 - number(&frozen["n_projectors"])).abs(),
            0.5,
        );
        for (name, mine) in [
            ("duality_error", channel.duality_error),
            ("asymmetry", channel.asymmetry),
            ("consistency_error", channel.consistency_error),
        ] {
            // Round-off diagnostics move by their own size under one ulp: order of magnitude.
            let bound = 10. * number(&frozen[name]).abs() + 1e-12;
            out.push(format!("diagnostic.l{l}.{name}"), mine.abs() / bound, 1.);
        }
    }
    out
}
