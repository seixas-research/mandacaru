//! AS-6 with the gradient correction on a REAL state: the skew LiH cell with PBE (the `pbe-sr`
//! datasets). The X-3 term (the strain of the spectral gradient, `-dv sum flux_b d_a n_s`) takes
//! the production gradients and flux of `solver::collinear_xc`, the arrays the xc potential is
//! made of, and the partial core sees the spin-averaged potential. The reconverged PBE free-energy
//! difference is the oracle. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_support::{
    gates::{bites, Tensor},
    run::{check, equation_of_state, weak_terms},
};

#[test]
fn the_stress_of_a_converged_pbe_state() {
    let (data, stress, want) = check("lih_pbe", true);
    equation_of_state(&data.manifest, &stress);
    // The gradient term is live on this crystal: it is not an empty term of the sum, and its
    // removal misses the oracle (the stress of a PBE crystal with the LDA terms alone).
    let gradient = &stress.breakdown.xc_gradient;
    let peak = gradient.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
    assert!(peak > 1e-4, "the gradient term is empty: {peak:e}");
    let without: Tensor =
        std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - gradient[a][b]));
    let bite = bites(&without, &want);
    println!("  control without the gradient term: bites {bite:.2e} gates");
    assert!(bite > 100.);
    let weak = weak_terms(&stress, &want);
    assert!(
        weak.is_empty(),
        "terms the oracle cannot discriminate: {weak:?}"
    );
}
