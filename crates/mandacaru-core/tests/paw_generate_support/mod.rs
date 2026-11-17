//! What the generator tests share: the arrays of a generated record under the names the tables
//! and channels carry.
#![allow(dead_code)]

use mandacaru_core::paw::record::{PawDataset, SpinOrbit};
use std::collections::BTreeMap;

fn flat(m: &mandacaru_core::paw::record::Array) -> Vec<f64> {
    m.data().to_vec()
}

/// Every array of a record keyed by table name.
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
    out.extend(spin_orbit_arrays(&dataset.spin_orbit));
    out
}

/// The spin-orbit block of a record keyed by table name.
pub fn spin_orbit_arrays(spin_orbit: &BTreeMap<u32, SpinOrbit>) -> BTreeMap<String, Vec<f64>> {
    let mut out = BTreeMap::new();
    for (l, so) in spin_orbit {
        for (i, p) in so.projectors.iter().enumerate() {
            out.insert(format!("so_projector_l{l}_{i}"), p.as_slice().to_vec());
        }
        for (name, array) in [
            ("average", &so.average),
            ("average_screened", &so.average_screened),
            ("coupling", &so.coupling),
            ("coupling_screened", &so.coupling_screened),
            ("overlap_average", &so.overlap_average),
            ("overlap_coupling", &so.overlap_coupling),
            ("level_errors", &so.level_errors),
            ("reference_energies", &so.reference_energies),
        ] {
            out.insert(format!("so_{name}_l{l}"), flat(array));
        }
    }
    out
}
