//! The spin-orbit block against the freeze, item by item, and the pieces a control rebuilds a
//! block from.
use super::Case;
use crate::paw_generate_support::compare::{gate_of, Comparison};
use crate::paw_generate_support::{counterpart, spin_orbit_arrays, Decimated};
use mandacaru_core::paw::generate::shells::{partition, Shells};
use mandacaru_core::paw::generate::spin_orbit::Inputs;
use mandacaru_core::paw::generate::Generated;
use mandacaru_core::paw::record::SpinOrbit;
use std::collections::BTreeMap;

/// Every `so_*` array of the freeze against `spin_orbit`. A missing array reads as zeros: a
/// dataset without the term is compared as the zero coupling it amounts to.
pub fn spin_orbit_items(
    case: &Case,
    data: &Decimated,
    spin_orbit: &BTreeMap<u32, SpinOrbit>,
) -> Comparison {
    let ours = spin_orbit_arrays(spin_orbit);
    let mut out = Comparison::default();
    let mut seen = 0;
    for (key, spec) in case.frozen.doc["arrays"].as_object().unwrap() {
        if !key.starts_with("so_") {
            continue;
        }
        seen += 1;
        let frozen = counterpart(&case.frozen, data, key, "d_", spec);
        let length = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .product::<usize>();
        let mine = ours.get(key).cloned().unwrap_or_else(|| vec![0.; length]);
        assert_eq!(mine.len(), length, "{key}: shape");
        out.push(
            format!("array.{key}"),
            frozen.deviation(&mine),
            gate_of(&case.frozen, key),
        );
    }
    assert!(seen > 0, "the freeze has no spin-orbit block");
    assert!(
        ours.keys()
            .all(|k| case.frozen.doc["arrays"].get(k).is_some()),
        "the generated block has an array the freeze does not"
    );
    out
}

/// What the spin-orbit construction reads of a finished generation. The shells are returned
/// beside the inputs they are borrowed by.
pub struct Owned {
    pub shells: Shells,
    pub radii: BTreeMap<usize, f64>,
}

impl Owned {
    pub fn of(case: &Case, generated: &Generated) -> Self {
        Self {
            shells: partition(&case.atom, &case.recipe).unwrap(),
            radii: generated
                .dataset
                .channels
                .iter()
                .map(|(&l, c)| (l as usize, c.r_cut))
                .collect(),
        }
    }

    pub fn inputs<'a>(&'a self, case: &'a Case, generated: &'a Generated) -> Inputs<'a> {
        let dataset = &generated.dataset;
        Inputs {
            recipe: &case.recipe,
            atom: &case.atom,
            shells: &self.shells,
            radii: &self.radii,
            extra_l: dataset.extra_l as usize,
            anchor: generated.anchor,
            v_loc: dataset.v_local_screened.as_slice(),
            r_local: dataset.r_cut_local,
            hartree_screening: dataset.hartree_screening,
        }
    }
}
