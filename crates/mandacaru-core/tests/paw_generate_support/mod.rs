//! What the generator tests share: the recipe a frozen dataset was generated with, the arrays of
//! a generated record under the names the A0 freeze gave them, and the pointwise comparison.
//! The freeze reader itself is `paw_assemble_support` (A2).
#![allow(dead_code)]

pub mod compare;
pub mod dataset;

use crate::paw_assemble_support::{directory, Frozen, Npz};
use compare::nan_max;
use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::recipe::Recipe;
use mandacaru_core::paw::record::PawDataset;
use serde_json::Value;
use std::collections::BTreeMap;

/// The `GEN_POINTS` of the freeze script: a decimated array has `ceil(n / 2000)` stride.
const DECIMATED_POINTS: usize = 2000;

/// The light sample (every dataset up to 45,000 points) and the heavy one (Ce, Au, Pb, U).
pub const LIGHT: [&str; 14] = [
    "lda-sr_H",
    "lda-sr_C",
    "lda-sr_O",
    "lda-sr_Na",
    "lda-sr_Mg",
    "lda-sr_Si",
    "lda-sr_K",
    "lda-sr_Fe",
    "pbe-sr_O",
    "pbe-sr_Cu",
    "upaw_H",
    "upaw_O",
    "upaw_Al",
    "upaw_Cr",
];
pub const HEAVY: [&str; 5] = [
    "lda-sr_Ce",
    "pbe-sr_Ce",
    "lda-sr_Au",
    "lda-sr_Pb",
    "lda-sr_U",
];

/// The recipe the frozen dataset was generated with: its options, and the remedy of the ghost
/// search when the selected attempt was not the first construction.
pub fn recipe_of(f: &Frozen) -> Recipe {
    let doc = &f.doc;
    let (set, element) = (
        doc["set"].as_str().unwrap(),
        doc["element"].as_str().unwrap(),
    );
    let functional = doc["options"]["xc"].as_str().unwrap();
    assert_eq!(doc["options"]["relativity"], "scalar", "{set} {element}");
    let mut recipe = if set == "upaw" {
        Recipe::upaw(element, functional, Treatment::Scalar)
    } else {
        Recipe::for_dataset(element, functional, Treatment::Scalar)
    }
    .unwrap();
    let search = &doc["ghost_free"];
    let selected = search["selected_index"].as_u64().unwrap();
    let attempt = search["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["index"].as_u64() == Some(selected))
        .unwrap();
    if let Some(remedy) = attempt["remedy"].as_object() {
        for (key, value) in remedy {
            let number = value.as_f64().unwrap();
            match key.as_str() {
                "norm_deficit" => recipe.norm_deficit = number,
                "local_shift" => recipe.local_shift = number,
                "local_factor" => recipe.local_factor = number,
                other => panic!("remedy key {other} has no counterpart in the recipe"),
            }
        }
    }
    recipe
}

/// The decimated arrays of a frozen dataset (`d_<name>`, `a_<name>`).
pub struct Decimated {
    npz: Npz,
}

impl Decimated {
    pub fn open(name: &str) -> Self {
        let path = directory().join(format!("gen_{name}_data.npz"));
        let restore = format!(
            "python test/migration/freeze_paw_generation.py run gen:{}",
            name.replace('_', ":")
        );
        Self {
            npz: Npz::open(&path, &restore),
        }
    }
    pub fn array(&self, key: &str) -> Vec<f64> {
        self.npz.array(key)
    }
}

fn flat(m: &mandacaru_core::paw::record::Array) -> Vec<f64> {
    m.data().to_vec()
}

/// Every array of a record under the freeze's names.
pub fn arrays(dataset: &PawDataset) -> BTreeMap<String, Vec<f64>> {
    let mut out = BTreeMap::new();
    for (name, values) in dataset.tables() {
        out.insert(name, values.to_vec());
    }
    for (l, channel) in &dataset.channels {
        for (i, (q, c)) in channel
            .wavevectors
            .iter()
            .zip(&channel.wave_coefficients)
            .enumerate()
        {
            out.insert(format!("wavevectors_l{l}_{i}"), flat(q));
            out.insert(format!("wave_coefficients_l{l}_{i}"), flat(c));
        }
        for (name, array) in [
            ("coupling", &channel.coupling),
            ("coupling_screened", &channel.coupling_screened),
            ("overlap_correction", &channel.overlap_correction),
            ("norm_correction", &channel.norm_correction),
            ("kinetic_difference", &channel.kinetic_difference),
            ("potential_difference", &channel.potential_difference),
            ("vanderbilt", &channel.vanderbilt),
            ("reference_energies", &channel.reference_energies),
            ("occupations", &channel.occupations),
            ("residual_kinetic", &channel.residual_kinetic),
        ] {
            out.insert(format!("{name}_l{l}"), flat(array));
        }
    }
    out
}

/// The frozen counterpart of an array: whole (small arrays) or decimated, with its scale.
pub struct Counterpart {
    pub values: Vec<f64>,
    pub stride: usize,
    pub scale: f64,
}

pub fn counterpart(
    f: &Frozen,
    data: &Decimated,
    key: &str,
    prefix: &str,
    spec: &Value,
) -> Counterpart {
    if let Some(v) = spec.get("values") {
        let values = v
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        let scale = values.iter().fold(0., |m: f64, x| m.max(x.abs()));
        return Counterpart {
            values,
            stride: 1,
            scale,
        };
    }
    let n = spec["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .product::<usize>();
    let scale = [spec.get("max"), spec.get("abs_max")]
        .iter()
        .flatten()
        .filter_map(|x| x.as_f64())
        .fold(0., |m: f64, x| m.max(x.abs()));
    let _ = f;
    Counterpart {
        values: data.array(&format!("{prefix}{key}")),
        stride: n.div_ceil(DECIMATED_POINTS).max(1),
        scale,
    }
}

impl Counterpart {
    /// Worst `|ours - frozen|` over the frozen positions, relative to the frozen maximum, and the
    /// worst on positions where `keep(frozen_index)` holds.
    pub fn deviation(&self, ours: &[f64]) -> f64 {
        self.values
            .iter()
            .enumerate()
            .map(|(k, b)| (ours[(k + 1) * self.stride - 1] - b).abs())
            .fold(0., nan_max)
            / self.scale.max(1e-300)
    }
}
