//! A complete geometry -> integrals -> RHF -> VQE calculation without Python.
use mandacaru_core::{
    math::SpatialGrid,
    molecular::{MolecularIntegrals, ScfOptions},
    quantum::SolveOptions,
    Context, Geometry, LengthUnit,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(Some(1), 512 * 1024 * 1024)?;
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
    let reference = integrals.mean_field("rhf", [1, 1], &ScfOptions::default())?;
    let problem =
        reference
            .model(&integrals)
            .unwrap()
            .problem(&ctx, "jordan_wigner", true, false, true)?;
    let vqe = problem.solve(&SolveOptions::default(), None)?;
    println!(
        "{}",
        serde_json::json!({"rhf_ha":reference.electronic_energy+integrals.nuclear,"rhf_converged":reference.converged,"vqe_ha":vqe.energy,"vqe_converged":vqe.success,"overlap":integrals.overlap.as_slice(),"one_body":integrals.one.as_slice(),"eri":integrals.eri()?.dense(&ctx)?.as_slice(),"memory_peak_bytes":ctx.memory_peak()})
    );
    if !reference.converged || !vqe.success {
        std::process::exit(1);
    }
    Ok(())
}
