//! The pieces a spin-orbit construction is rebuilt from.
use super::Case;
use mandacaru_core::paw::generate::shells::{partition, Shells};
use mandacaru_core::paw::generate::spin_orbit::Inputs;
use mandacaru_core::paw::generate::Generated;
use std::collections::BTreeMap;

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
