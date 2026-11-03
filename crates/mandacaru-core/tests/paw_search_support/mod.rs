//! What the search tests share: the frozen outcome of the A0 `ghost_free` search (selected
//! attempt, label, `tried` list, the selected construction's scalars), a frozen dataset searched
//! through the legacy radial scheme of the freeze, and the comparison of the result with it.
#![allow(dead_code)]

use crate::paw_assemble_support::Frozen;
use crate::paw_generate_support::compare::{compare_atom, floor, lifted, relative, Scheme};
use crate::paw_generate_support::dataset::compare_dataset_with;
use crate::paw_generate_support::{base_recipe_of, Decimated};
use mandacaru_core::models::atom::Atom;
use mandacaru_core::paw::generate::energies::AtomSolver;
use mandacaru_core::paw::generate::recipe::Recipe;
use mandacaru_core::paw::generate::reference::solve_reference;
use mandacaru_core::paw::generate::search::{
    search_with_atom, Error, Mode, Options, Pinned, Searched, Tried, Why,
};
use mandacaru_core::Context;

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

/// What the Python search recorded for one dataset.
pub struct Outcome {
    pub index: usize,
    pub label: String,
    pub tried: Vec<String>,
    /// `attempts[selected].generated`, absent when the first construction was selected.
    pub generated: Option<serde_json::Value>,
}

pub fn outcome(f: &Frozen) -> Outcome {
    let search = &f.doc["ghost_free"];
    let index = search["selected_index"].as_u64().unwrap() as usize;
    let attempt = search["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["index"].as_u64() == Some(index as u64))
        .unwrap();
    Outcome {
        index,
        label: search["selected_label"].as_str().unwrap().to_string(),
        tried: search["tried"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect(),
        generated: attempt.get("generated").cloned(),
    }
}

/// A frozen dataset prepared for a search: the first construction's recipe and the solved atom.
pub struct Case {
    pub frozen: Frozen,
    pub recipe: Recipe,
    pub atom: Atom,
}

pub fn open(ctx: &Context, name: &str) -> Case {
    let frozen = Frozen::open(name);
    let recipe = base_recipe_of(&frozen);
    let atom = solve_reference(ctx, &recipe).unwrap();
    Case {
        frozen,
        recipe,
        atom,
    }
}

impl Case {
    /// The search of this dataset through the atom-solver scheme of the freeze.
    pub fn search(&self, ctx: &Context, mode: Mode) -> Result<Searched, Error> {
        search_with_atom(
            ctx,
            &self.recipe,
            &Options::new(&self.recipe, mode),
            &self.atom,
            &AtomSolver,
        )
    }

    pub fn search_pinned(
        &self,
        ctx: &Context,
        mode: Mode,
        pinned: Pinned,
    ) -> Result<Searched, Error> {
        search_with_atom(
            ctx,
            &self.recipe,
            &Options { mode, pinned },
            &self.atom,
            &AtomSolver,
        )
    }
}

/// The selected attempt of a search against the Python search: same index, same label, the same
/// failed attempts (`tried`, text for text), the selected construction's recorded parameters and
/// the whole record against the freeze. Panics with the name on a miss.
pub fn assert_same_selection(case: &Case, searched: &Searched, phase_tolerance: f64) {
    let name = &case.frozen.name;
    let frozen = outcome(&case.frozen);
    assert_eq!(searched.label, frozen.label, "{name}: selected label");
    assert_eq!(searched.index, frozen.index, "{name}: selected index");
    assert_same_tried(name, &searched.tried, &frozen.tried, phase_tolerance);
    if let Some(generated) = &frozen.generated {
        assert_selected_parameters(name, generated, searched);
    }
    let data = Decimated::open(name);
    let dataset = &searched.generated.dataset;
    let v_local_gate = if dataset.xc == "pbe" {
        lifted(floor(&case.frozen, "xc_energies", "tables"))
    } else {
        1e-12
    };
    compare_dataset_with(
        &case.frozen,
        &data,
        &searched.generated,
        Scheme::Legacy,
        v_local_gate,
    )
    .assert_within(&format!("{name} (selected attempt)"));
    compare_atom(&case.frozen, &case.atom).assert_within(&format!("{name} atom"));
}

/// The numbers of a text, in order, and the text with every number replaced by `#`.
fn numbers(text: &str) -> (Vec<f64>, String) {
    let (mut values, mut skeleton, mut token) = (Vec::new(), String::new(), String::new());
    let mut flush = |token: &mut String, skeleton: &mut String| {
        if let Ok(v) = token.parse::<f64>() {
            values.push(v);
            skeleton.push('#');
        } else {
            skeleton.push_str(token);
        }
        token.clear();
    };
    for c in text.chars() {
        if c.is_ascii_digit() || (c == '.' && !token.is_empty()) || (c == '-' && token.is_empty()) {
            token.push(c);
        } else {
            flush(&mut token, &mut skeleton);
            skeleton.push(c);
        }
    }
    flush(&mut token, &mut skeleton);
    (values, skeleton)
}

/// The failed attempts against the Python list, text for text, except that the numbers of an
/// entry may differ by `tolerance` when the rest of the text is the same (`0.` demands the same
/// text). Python reported a construction that could not be built as `label: ValueError`; the
/// Rust text carries the cause instead, so those entries are compared by label and kind.
pub fn assert_same_tried(name: &str, ours: &[Tried], frozen: &[String], tolerance: f64) {
    assert_eq!(
        ours.len(),
        frozen.len(),
        "{name}: number of failed attempts"
    );
    let mut worst = 0f64;
    for (got, want) in ours.iter().zip(frozen) {
        if let Some(label) = want.strip_suffix(": ValueError") {
            assert_eq!(
                got.label, label,
                "{name}: label of a construction that failed"
            );
            assert!(
                matches!(got.why, Why::NotBuilt(_)),
                "{name}: {got} should be a construction that could not be built"
            );
            continue;
        }
        let text = got.to_string();
        if text == *want {
            continue;
        }
        let ((a, shape_a), (b, shape_b)) = (numbers(&text), numbers(want));
        assert!(
            tolerance > 0. && shape_a == shape_b && a.len() == b.len(),
            "{name}: a failed attempt: {text:?} against {want:?}"
        );
        // Two values within `tolerance` print three decimals apart by at most one unit more.
        let entry = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        println!("{name}: {text:?} against {want:?}: {entry:.1e}");
        worst = worst.max(entry);
    }
    println!("{name}: worst numeric difference of the failed attempts {worst:.2e}");
    assert!(
        worst <= tolerance + 1.0e-3 + 1e-12,
        "{name}: the failed attempts differ by {worst:.2e} (gate {tolerance:.1e})"
    );
}

/// The recorded defects of a flagged search against the freeze: the same keys; the unprojected
/// phases, the s miss and the ghosts within `tight`, the constructed channels' phases within
/// `phase` (a sharp resonance moves with the last digits of the dataset).
pub fn assert_same_defects(f: &Frozen, searched: &Searched, tight: f64, phase: f64) {
    let name = &f.name;
    let want = &f.doc["scalars"]["defects"];
    let got = &searched.generated.dataset.defects;
    let keys = |v: &serde_json::Value, field: &str| -> Vec<u32> {
        v[field]
            .as_object()
            .map(|m| m.keys().map(|k| k.parse().unwrap()).collect())
            .unwrap_or_default()
    };
    assert_eq!(
        got.ghosts.keys().copied().collect::<Vec<_>>(),
        keys(want, "ghosts"),
        "{name}: ghost keys"
    );
    assert_eq!(
        got.phases.keys().copied().collect::<Vec<_>>(),
        keys(want, "phases"),
        "{name}: phase keys"
    );
    assert_eq!(
        got.unprojected.keys().copied().collect::<Vec<_>>(),
        keys(want, "unprojected"),
        "{name}: unprojected keys"
    );
    for (l, [near, far]) in &got.phases {
        let frozen = &want["phases"][l.to_string()];
        for (ours, theirs) in [(near, &frozen[0]), (far, &frozen[1])] {
            let d = (ours - theirs.as_f64().unwrap()).abs();
            println!("{name}: recorded phase l={l} {ours} against {theirs}: {d:.2e}");
            assert!(d <= phase, "{name}: phase l={l} {ours} against {theirs}");
        }
    }
    for (l, e) in &got.unprojected {
        let frozen = want["unprojected"][l.to_string()].as_f64().unwrap();
        assert!(
            (e - frozen).abs() <= tight,
            "{name}: unprojected l={l} {e} against {frozen}"
        );
    }
    for (l, e) in &got.ghosts {
        let frozen = want["ghosts"][l.to_string()].as_f64().unwrap();
        assert!((e - frozen).abs() <= tight, "{name}: ghost l={l}");
    }
    assert_eq!(
        got.s_miss.is_some(),
        !want["s_miss"].is_null(),
        "{name}: s_miss recorded"
    );
}

/// `attempts[selected].generated` of the freeze: the radii, the local radius, the shift, the
/// Bessel counts and the deficit exactly; the one-center energy and the compensation charge at
/// the relative gate of the support.
fn assert_selected_parameters(name: &str, generated: &serde_json::Value, searched: &Searched) {
    let dataset = &searched.generated.dataset;
    for (l, radius) in generated["r_cut"].as_object().unwrap() {
        let channel = &dataset.channels[&l.parse::<u32>().unwrap()];
        assert!(
            relative(channel.r_cut, radius.as_f64().unwrap()) < 1e-12,
            "{name}: r_cut of l={l}: {} against {radius}",
            channel.r_cut
        );
    }
    for (l, count) in generated["n_bessel"].as_object().unwrap() {
        let channel = &dataset.channels[&l.parse::<u32>().unwrap()];
        assert_eq!(
            channel.wavevectors[0].data().len() as u64,
            count.as_u64().unwrap(),
            "{name}: Bessel functions of l={l}"
        );
    }
    for (key, ours) in [
        ("r_cut_local", dataset.r_cut_local),
        ("local_shift", dataset.local_shift),
    ] {
        let frozen = generated[key].as_f64().unwrap();
        assert!(
            (ours - frozen).abs() <= 1e-12 * frozen.abs().max(1.),
            "{name}: {key} {ours} against {frozen}"
        );
    }
    for (key, ours) in [
        ("one_center_energy", dataset.one_center_energy),
        ("compensation_charge", dataset.compensation_charge),
    ] {
        let frozen = generated[key].as_f64().unwrap();
        assert!(
            relative(ours, frozen) < 1e-8,
            "{name}: {key} {ours} against {frozen}"
        );
    }
}
