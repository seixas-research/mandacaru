//! Standalone geometry -> PBE -> stationary nuclear gradient, without Python.
use mandacaru_core::{
    math::SpatialGrid, molecular::MolecularIntegrals, Context, Geometry, LengthUnit,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(Some(1), 64 * 1024 * 1024)?;
    let geometry = Geometry::new(
        vec![1, 1],
        vec![[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [false; 3],
        LengthUnit::Angstrom,
    )?;
    let grid = SpatialGrid::from_geometry(&geometry, [0.35 / 0.529177210903; 3], false, false)?;
    let integrals = MolecularIntegrals::new(
        &ctx,
        geometry,
        grid,
        r#"{"name":"HAO"}"#,
        "spectral",
        None,
        Default::default(),
    )?;
    let state = integrals.kohn_sham("pbe", [1, 1], false, &Default::default())?;
    let gradient = integrals.gradient(&state)?;
    println!(
        "PBE energy: {:.12} Ha; converged={}",
        state.electronic_energy + integrals.nuclear,
        state.converged
    );
    println!(
        "Nuclear gradient (Ha/Bohr): {:?}",
        gradient.total.as_slice()
    );
    Ok(())
}
