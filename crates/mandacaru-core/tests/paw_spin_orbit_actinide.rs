//! An actinide through the Dirac chain. The freeze holds no actinide `fr` dataset (the sample is
//! Na, Fe, Ce, Au, Pb and Si), so there is no oracle to match: this checks what does not need
//! one, on uranium (5f, 6d, 7s with the 6s and 6p semicore, 138,000 points): the Dirac atom
//! converges, every `j` is recovered from the stored pair, each branch pair is ordered as the
//! `L.S` coupling says, the level errors are finite, and the generation stays inside a gigabyte.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_spin_orbit_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::energies::AtomSolver;
use mandacaru_core::paw::generate::generate_with_atom;
use mandacaru_core::paw::generate::recipe::Recipe;
use mandacaru_core::paw::generate::reference::solve_reference;
use mandacaru_core::paw::generate::shells::partition;
use mandacaru_core::paw::generate::spin_orbit::{branches, recover, Inputs};
use mandacaru_core::paw::record::Array;
use nalgebra::DMatrix;
use paw_spin_orbit_support::context;

fn matrix(a: &Array) -> DMatrix<f64> {
    DMatrix::from_row_slice(a.shape()[0], a.shape()[1], a.data())
}

#[test]
#[ignore = "minutes (no freeze, a 138,000-point Dirac atom): cargo test --release -p mandacaru-core --test paw_spin_orbit_actinide -- --ignored --nocapture"]
fn uranium_has_a_consistent_spin_orbit_block() {
    let ctx = context();
    let recipe = Recipe::for_dataset("U", "lda", Treatment::Dirac).unwrap();
    let atom = solve_reference(&ctx, &recipe).unwrap();
    assert!(atom.converged);
    let generated = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let dataset = &generated.dataset;
    let shells = partition(&atom, &recipe).unwrap();
    let radii = dataset
        .channels
        .iter()
        .map(|(&l, c)| (l as usize, c.r_cut))
        .collect();
    let inputs = Inputs {
        recipe: &recipe,
        atom: &atom,
        shells: &shells,
        radii: &radii,
        extra_l: dataset.extra_l as usize,
        anchor: generated.anchor,
        v_loc: dataset.v_local_screened.as_slice(),
        r_local: dataset.r_cut_local,
        hartree_screening: dataset.hartree_screening,
    };
    println!(
        "U: {} iterations, {} spin-orbit channels, context peak {} B",
        atom.iterations,
        dataset.spin_orbit.len(),
        ctx.memory_peak()
    );
    assert!(dataset.spin_orbit.keys().copied().eq(1..=3));
    for (&l, so) in &dataset.spin_orbit {
        let l = l as usize;
        let [down, up] = branches(&ctx, &inputs, l).unwrap();
        let count = down.projectors.len();
        let (average, coupling) = (matrix(&so.average), matrix(&so.coupling));
        for (is_up, expected) in [(false, &down.ionic), (true, &up.ionic)] {
            let back = recover(l, &average, &coupling, is_up, count);
            let scale = expected.amax().max(1e-300);
            assert!((back - expected).amax() / scale < 1e-14, "l={l}");
        }
        let energies = so.reference_energies.data();
        println!(
            "U l={l}: j splitting of the first reference {:+.6e} Ha, level errors {:?}",
            energies[count] - energies[0],
            so.level_errors.data()
        );
        // An empty channel anchored at one scattering energy for both branches splits by zero.
        assert!(
            energies[count] >= energies[0],
            "l={l}: j = l+1/2 lies at or above j = l-1/2"
        );
        assert!(so.level_errors.data().iter().all(|e| e.is_finite()));
    }
    assert!(
        ctx.memory_peak() < 1 << 30,
        "the context peak stays under a gigabyte"
    );
}
