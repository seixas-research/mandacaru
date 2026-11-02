//! A recorded `assemble_paw_channel` call rebuilt from the freeze, and the outputs of a channel
//! under the names the freeze gave them.
#![allow(dead_code)]

use super::Frozen;
use mandacaru_core::paw::generate::assemble::{assemble, Atomic, Partial, PawChannel};
use mandacaru_core::paw::pseudize::PseudoWaves;
use mandacaru_core::Context;
use nalgebra::DMatrix;
use serde_json::Value;
use std::collections::BTreeMap;

pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

pub struct Call {
    pub r: Vec<f64>,
    pub v_ae: Vec<f64>,
    pub v_loc: Vec<f64>,
    pub r_local: f64,
    pub l: usize,
    pub n: usize,
    pub energies: Vec<f64>,
    pub waves: Vec<f64>,
    pub smooth: PseudoWaves,
    pub q_cut: f64,
    pub occupations: Vec<f64>,
    pub strict: bool,
    pub condition_number: f64,
}

pub fn call(f: &Frozen, case: &Value) -> Call {
    let inp = &case["inputs"];
    let pw = &inp["pw"];
    let energies = reals(&pw["energies"]);
    let count = energies.len();
    let rows = |key: &str| -> Vec<Vec<f64>> {
        pw[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| f.input(s))
            .collect()
    };
    let wavevectors = rows("wavevectors");
    assert!(
        wavevectors.iter().all(|q| *q == wavevectors[0]),
        "the references of a channel share one Bessel basis"
    );
    let coefficients = rows("coefficients");
    let matrix = |key: &str| DMatrix::from_row_slice(count, count, &f.input(&pw[key]));
    Call {
        r: f.input(&inp["r"]),
        v_ae: f.input(&inp["v_ae"]),
        v_loc: f.input(&inp["v_loc"]),
        r_local: inp["r_local"].as_f64().unwrap(),
        l: pw["l"].as_u64().unwrap() as usize,
        n: inp["n"].as_u64().unwrap() as usize,
        waves: rows("waves").concat(),
        smooth: PseudoWaves {
            r_cut: pw["r_cut"].as_f64().unwrap(),
            wavevectors: wavevectors[0].clone(),
            coefficients: DMatrix::from_row_slice(
                count,
                coefficients[0].len(),
                &coefficients.concat(),
            ),
            residuals: reals(&pw["residual_kinetic"]),
            norms: matrix("norms"),
            achieved: matrix("achieved"),
            targets: matrix("targets"),
        },
        q_cut: pw["q_cut"].as_f64().unwrap(),
        occupations: reals(&inp["occupations"]),
        strict: inp["strict"].as_bool().unwrap(),
        condition_number: case["condition_number_vanderbilt"].as_f64().unwrap(),
        energies,
    }
}

impl Call {
    pub fn run(&self, ctx: &Context) -> PawChannel {
        self.try_run(ctx).unwrap()
    }
    pub fn try_run(&self, ctx: &Context) -> mandacaru_core::Result<PawChannel> {
        let atom = Atomic {
            r: &self.r,
            v_ae: &self.v_ae,
            v_loc: &self.v_loc,
            r_local: self.r_local,
        };
        let set = Partial {
            l: self.l,
            n: self.n,
            energies: &self.energies,
            waves: &self.waves,
            smooth: &self.smooth,
            q_cut: self.q_cut,
            occupations: &self.occupations,
        };
        assemble(ctx, &atom, &set, self.strict)
    }
}

fn flat(m: &DMatrix<f64>) -> Vec<f64> {
    m.transpose().as_slice().to_vec()
}

/// The outputs of a channel under the names the freeze recorded them with.
pub fn outputs(ch: &PawChannel) -> BTreeMap<String, Vec<f64>> {
    let count = ch.reference_energies.len();
    let n = ch.pseudo_waves.as_slice().len() / count;
    let mut out = BTreeMap::new();
    for i in 0..count {
        let row = |b: &[f64]| b[i * n..(i + 1) * n].to_vec();
        out.insert(format!("projector_{i}"), row(ch.projectors.as_slice()));
        out.insert(
            format!("raw_projector_{i}"),
            row(ch.raw_projectors.as_slice()),
        );
        out.insert(format!("pseudo_wave_{i}"), row(ch.pseudo_waves.as_slice()));
    }
    for (name, m) in [
        ("overlap_correction", &ch.overlap_correction),
        ("norm_correction", &ch.norm_correction),
        ("kinetic_difference", &ch.kinetic_difference),
        ("potential_difference", &ch.potential_difference),
        ("coupling_screened", &ch.coupling_screened),
        ("vanderbilt", &ch.vanderbilt),
    ] {
        out.insert(name.into(), flat(m));
    }
    for (name, v) in [
        ("duality_error", ch.duality_error),
        ("overlap_minimum", ch.overlap_minimum),
        ("asymmetry", ch.asymmetry),
        ("consistency_error", ch.consistency_error),
        ("norm_error", ch.norm_error()),
        ("eigenvalue", ch.eigenvalue()),
        ("r_cut", ch.r_cut),
        ("occupation", ch.occupation()),
        ("n", ch.n as f64),
    ] {
        out.insert(name.into(), vec![v]);
    }
    out.insert("reference_energies".into(), ch.reference_energies.clone());
    out
}
