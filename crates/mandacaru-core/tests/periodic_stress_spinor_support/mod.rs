//! Shared by the `periodic_stress_spinor*` tests (AS-8): the spinor crystals of the strain freeze
//! and of the force freeze with their spin-orbit union projectors, the analytic stress of a
//! converged state, and the tensor helpers of the gates.
//!
//! The union list and the couplings are built from the `lda-fr` record exactly as the force tests
//! do ([`union_list`](crate::periodic_gradient_spinor_support::union_list), here from a `Data`).
//! A spinor stress needs both: the force provider ties `h^SO(k)` of this list to the matrix the SCF
//! crystal diagonalized with, so a wrong list or coupling raises.
#![allow(dead_code)]

pub mod altered;

use crate::{
    periodic_gradient_crystals_support::{inputs, Data},
    periodic_gradient_spinor_support::{union_list, Spinor},
};
use mandacaru_core::{
    paw::record::PawDataset,
    periodic::{
        gradient::State,
        stress_terms::{
            crystal::{crystal_strain, crystal_stress, CrystalStrain, SpinOrbitInputs},
            Stress, Tensor,
        },
        Options,
    },
};

pub type Grid = [[f64; 3]; 3];

/// A spinor crystal with its union projectors and couplings.
pub fn spinor_of(data: Data) -> Spinor {
    assert!(data.crystal.spinor, "{} is a spinor snapshot", data.case);
    let refs: Vec<&PawDataset> = data.datasets.iter().collect();
    let (projectors, couplings) = union_list(&data.crystal.context, &refs, &data.centers());
    Spinor {
        data,
        projectors,
        couplings,
    }
}

pub fn so_inputs(sp: &Spinor) -> SpinOrbitInputs<'_> {
    SpinOrbitInputs {
        projectors: &sp.projectors,
        couplings: &sp.couplings,
    }
}

/// The analytic stress of the converged `state` of `sp`.
pub fn stress_at(sp: &Spinor, state: &State, options: &Options) -> Stress {
    let cores = sp.data.cores();
    crystal_stress(
        &sp.data.crystal.context,
        inputs(&sp.data, &cores),
        sp.data.layout(),
        sp.data.operators(),
        &sp.data.crystal,
        options,
        state,
        Some(so_inputs(sp)),
    )
    .unwrap()
}

/// [`stress_at`] on a `Data` (the spinor list is built on the way): the form the run helpers of
/// `periodic_stress_crystals_support` take.
pub fn stress_of_data(data: &Data, state: &State, options: &Options) -> Stress {
    let refs: Vec<&PawDataset> = data.datasets.iter().collect();
    let (projectors, couplings) = union_list(&data.crystal.context, &refs, &data.centers());
    let cores = data.cores();
    crystal_stress(
        &data.crystal.context,
        inputs(data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        options,
        state,
        Some(SpinOrbitInputs {
            projectors: &projectors,
            couplings: &couplings,
        }),
    )
    .unwrap()
}

/// The strain provider and the frozen state of `sp` (what the controls damage). `cores` must
/// outlive the result.
pub fn strain_of<'a>(
    sp: &'a Spinor,
    cores: &'a [Option<mandacaru_core::periodic::setup::provider::Core<'a>>],
    state: &State,
    options: &Options,
) -> CrystalStrain<'a> {
    crystal_strain(
        &sp.data.crystal.context,
        inputs(&sp.data, cores),
        sp.data.layout(),
        sp.data.operators(),
        &sp.data.crystal,
        options,
        state,
        Some(so_inputs(sp)),
    )
    .unwrap()
}

/// The peak entry of a tensor.
pub fn peak(t: &Tensor<f64>) -> f64 {
    t.iter().flatten().fold(0f64, |m, x| m.max(x.abs()))
}

/// The largest entry-wise difference of two tensors (a NaN is infinite).
pub fn miss(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
    let mut worst = 0f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (a[i][j] - b[i][j]).abs();
            if d.is_nan() {
                return f64::INFINITY;
            }
            worst = worst.max(d);
        }
    }
    worst
}

pub fn sum(a: &Tensor<f64>, b: &Tensor<f64>, sign: f64) -> Grid {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + sign * b[i][j]))
}
