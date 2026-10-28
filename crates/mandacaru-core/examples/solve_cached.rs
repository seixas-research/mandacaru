//! Run with `cargo run -p mandacaru-core --example solve_cached -- CACHE [adapt]`.
use mandacaru_core::{
    quantum::{CachedProblem, SolveOptions},
    Context,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: solve_cached CACHE [adapt]")?;
    let context = Context::new(Some(1), 512 * 1024 * 1024)?;
    let problem = CachedProblem::load(&context, std::path::Path::new(&path), true, false, true)?;
    let options = SolveOptions {
        adapt: std::env::args().nth(2).as_deref() == Some("adapt"),
        ..Default::default()
    };
    let result = problem.solve(&options, None)?;
    println!(
        "{}",
        serde_json::json!({"energy_ha":result.energy,"reference_ha":result.reference,"success":result.success,"message":result.message,"parameters":result.parameters,"operators":result.labels,"optimizer_steps":result.optimizer_steps,"num_evaluations":result.history.len(),"dimension":problem.basis.len(),"memory_peak_bytes":context.memory_peak(),"elapsed_seconds":result.elapsed_seconds})
    );
    if !result.success {
        std::process::exit(1);
    }
    Ok(())
}
