//! cargo run -p mandacaru-core --example solve_crystal -- CRYSTAL.json [width_Ha]
use mandacaru_core::{
    periodic::{Crystal, Options},
    Context,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: solve_crystal CRYSTAL.json [width_Ha]")?;
    let ctx = Context::new(Some(1), 512 * 1024 * 1024)?;
    let crystal = Crystal::load(&ctx, &path)?;
    let options = Options {
        width: std::env::args()
            .nth(2)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(0.01 / 27.211386245988),
        ..Default::default()
    };
    let state = crystal.solve(&options)?;
    println!(
        "{}",
        serde_json::json!({"energy_ha":state.energy,"free_energy_ha":state.free_energy,"extrapolated_energy_ha":state.extrapolated_energy,"converged":state.converged,"iterations":state.history.len(),"fermi_level_ha":state.fermi,"eigenvalues_ha":state.eigenvalues,"memory_peak_bytes":ctx.memory_peak()})
    );
    if !state.converged {
        std::process::exit(1);
    }
    Ok(())
}
