//! B2 on DFT metals (`periodic::berry::hall::intrinsic`), checked by relations, not values: a
//! metal's intrinsic Hall conductivity on the affordable meshes is not converged (vera,
//! 2026-11-11: Ni 4^3 is 94% merged plaquettes, Fe 5^3 breaks the cubic symmetry by
//! sigma_yz / sigma_xy = 0.38), so these tests pin what must hold at any mesh.
//! - Non-trivial zeros: a noncentrosymmetric unpolarized metal (BeLi) and a collinear spin-polarized
//!   one (BeB with the B atom displaced) have layers of either sign that cancel.
//! - The intrinsic formula on an insulator is the Chern result; the stride-2 estimate and the
//!   warnings are wired.
//! - Magnetic spin-orbit metals (slow): Ni fcc and Fe bcc, sigma odd in the moment, unchanged by a
//!   left-handed relabeling, along the moment (Ni), with nu_1 = nu_2 (Fe).
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::periodic::berry::hall::{self, Hall};
use nalgebra::{Matrix3, Vector3};

/// The largest component of a tensor.
fn largest(sigma: &[[f64; 3]; 3]) -> f64 {
    sigma.iter().flatten().fold(0f64, |m, v| m.max(v.abs()))
}

/// Every plane family and channel: layers of either sign, each sum zero; sigma zero.
fn cancels(hall: &Hall) {
    let mut biggest = 0f64;
    for c in 0..3 {
        for (s, layers) in hall.layers[c].iter().enumerate() {
            let sum: f64 = layers.iter().sum();
            assert!(
                sum.abs() < 1e-10,
                "family {c}, channel {s}: {layers:?} sums to {sum:e}"
            );
            biggest = layers.iter().fold(biggest, |m, v| m.max(v.abs()));
        }
    }
    assert!(
        biggest > 1e-3,
        "the layers are not zero to begin with: {biggest:e}"
    );
    assert!(largest(&hall.sigma) < 1e-10, "{:?}", hall.sigma);
}

/// BeLi in a skew cell (Angstrom rows (6.2, 0, 0), (0.7, 5.8, 0), (0.5, 0.4, 6.4)): no inversion.
fn beli() -> Cell {
    let lattice = angstrom_cell([[6.2, 0., 0.], [0.7, 5.8, 0.], [0.5, 0.4, 6.4]]);
    Cell {
        library: "lda-sr",
        symbols: vec!["Be", "Li"],
        grid: grid_of(&lattice, 0.3),
        lattice,
        fractional: vec![[0.1, 0.15, 0.2], [0.55, 0.5, 0.45]],
        moments: vec![0.; 2],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [3, 3, 3],
        hubbard: None,
        width: ev(0.02),
    }
}

/// Zincblende BeB (fcc a = 4.0 Angstrom) with the B atom displaced to (0.21, 0.27, 0.31),
/// spin-polarized from moments 0.5: a collinear magnetic metal without inversion.
fn beb_displaced() -> Cell {
    let lattice = angstrom_cell([[0., 2., 2.], [2., 0., 2.], [2., 2., 0.]]);
    Cell {
        library: "lda-sr",
        symbols: vec!["Be", "B"],
        grid: grid_of(&lattice, 0.3),
        lattice,
        fractional: vec![[0.; 3], [0.21, 0.27, 0.31]],
        moments: vec![0.5, 0.5],
        spin: true,
        spinor: false,
        functional: "lda",
        mesh: [4, 4, 4],
        hubbard: None,
        width: ev(0.02),
    }
}

#[test]
#[ignore = "slow: 14 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_metals -- --ignored an_unpolarized_noncentrosymmetric_metal_cancels_layers_of_either_sign"]
fn an_unpolarized_noncentrosymmetric_metal_cancels_layers_of_either_sign() {
    let solved = solve(&beli());
    let hall = hall::anomalous_hall(&solved.sampler(), [4, 4, 4]).unwrap();
    assert!(hall.metallic);
    cancels(&hall);
}

#[test]
#[ignore = "slow: 75 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_metals -- --ignored a_collinear_spin_polarized_metal_cancels_in_every_channel_without_spin_orbit"]
fn a_collinear_spin_polarized_metal_cancels_in_every_channel_without_spin_orbit() {
    let solved = solve(&beb_displaced());
    assert_eq!(solved.crystal.spins, 2);
    assert!(
        solved.state.magnetization.abs() > 0.05,
        "{}",
        solved.state.magnetization
    );
    let hall = hall::anomalous_hall(&solved.sampler(), [4, 4, 4]).unwrap();
    assert!(hall.metallic && hall.layers[0].len() == 2);
    cancels(&hall);
}

#[test]
fn the_intrinsic_formula_on_an_insulator_is_the_chern_result() {
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let quantized = hall::anomalous_hall(&sampler, [2, 2, 2]).unwrap();
    let weighted = hall::intrinsic(&sampler, [2, 2, 2]).unwrap();
    assert!(!quantized.metallic && weighted.metallic);
    for c in 0..3 {
        for (a, b) in quantized.layers[c][0].iter().zip(&weighted.layers[c][0]) {
            assert!((a - b).abs() < 1e-9, "{c}: {a} against {b}");
        }
    }
    assert_eq!(weighted.merged, 0);
    // The SCF level (from a 2 x 1 x 1 SCF mesh) need not hold the electrons on the Berry mesh;
    // the re-solved one does, and the layers above are B1's at it.
    eprintln!(
        "LiH: the SCF Fermi level would misplace {:.3e} electrons on the 2^3 Berry mesh",
        weighted.scf_electron_error
    );
    assert!(weighted.fermi.is_finite());
}

/// One H per 3 Bohr along x: a half-filled band, a metal (centrosymmetric and real, so its zero
/// is trivial: this test checks the wiring of the estimate and the warnings, not physics).
fn hydrogen_chain() -> Cell {
    let mut cell = h2_chain("lda-sr", "lda");
    cell.symbols = vec!["H"];
    cell.lattice = Matrix3::from_diagonal(&Vector3::new(3., 7., 7.));
    cell.grid = [9, 20, 20];
    cell.fractional = vec![[0.5, 0.5, 0.5]];
    cell.moments = vec![0.];
    cell.mesh = [6, 1, 1];
    cell.width = 1e-2;
    cell
}

#[test]
#[ignore = "slow: 58 s (measured 2026-11-11); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_metals -- --ignored the_stride_two_estimate_starts_at_eight_points_and_a_coarser_mesh_says_so"]
fn the_stride_two_estimate_starts_at_eight_points_and_a_coarser_mesh_says_so() {
    let solved = solve(&hydrogen_chain());
    let sampler = solved.sampler();
    let estimated = hall::anomalous_hall(&sampler, [8, 8, 8]).unwrap();
    assert!(estimated.sigma_coarse.is_some() && estimated.warnings.is_empty());
    assert_eq!(estimated.plaquettes, 3 * 8 * 64);
    // The Fermi level re-solved on the Berry mesh holds the cell's electron.
    assert!((estimated.fermi - solved.state.fermi).abs() < 0.05);
    for size in [[4, 4, 4], [3, 2, 2]] {
        let hall = hall::anomalous_hall(&sampler, size).unwrap();
        assert!(hall.sigma_coarse.is_none(), "{size:?}");
        assert_eq!(hall.warnings.len(), 1, "{size:?}: {:?}", hall.warnings);
        assert!(
            hall.warnings[0].contains("at least 8"),
            "{}",
            hall.warnings[0]
        );
    }
}

/// Ni fcc (a = 3.52 Angstrom) on lda-fr, a spinor crystal with its moment along z.
fn nickel(moment: f64) -> Cell {
    let a = 3.52;
    let lattice = angstrom_cell([
        [0., a / 2., a / 2.],
        [a / 2., 0., a / 2.],
        [a / 2., a / 2., 0.],
    ]);
    Cell {
        library: "lda-fr",
        symbols: vec!["Ni"],
        grid: grid_of(&lattice, 0.2),
        lattice,
        fractional: vec![[0.; 3]],
        moments: vec![moment],
        spin: false,
        spinor: true,
        functional: "lda",
        mesh: [4, 4, 4],
        hubbard: None,
        width: ev(0.01),
    }
}

/// Fe bcc (a = 2.87 Angstrom) on lda-fr, a spinor crystal with its moment along z.
fn iron(moment: f64) -> Cell {
    let a = 2.87;
    let h = a / 2.;
    let lattice = angstrom_cell([[-h, h, h], [h, -h, h], [h, h, -h]]);
    Cell {
        library: "lda-fr",
        symbols: vec!["Fe"],
        grid: grid_of(&lattice, 0.2),
        lattice,
        fractional: vec![[0.; 3]],
        moments: vec![moment],
        spin: false,
        spinor: true,
        functional: "lda",
        mesh: [5, 5, 5],
        hubbard: None,
        width: ev(0.02),
    }
}

fn report(label: &str, hall: &Hall) {
    let s = &hall.sigma;
    eprintln!(
        "{label}: sigma_xy {:.6e} sigma_yz {:.6e} sigma_zx {:.6e} (e^2/(hbar a0)), numbers {:?}, \
         merged {}/{}, max flux {:.3}, Fermi shift of the SCF level {:.3e} electrons, warnings {}",
        s[0][1],
        s[1][2],
        s[2][0],
        hall.numbers,
        hall.merged,
        hall.plaquettes,
        hall.max_flux,
        hall.scf_electron_error,
        hall.warnings.len()
    );
}

/// `sigma(-M) = -sigma(M)` component by component, to 1e-5 of the largest component: two
/// separately converged SCF runs (tolerance 1e-10 Ha) agree to about 4e-7 here (Ni 4^3), while a
/// defect would break the relation at order one.
fn odd(plus: &Hall, minus: &Hall) {
    let scale = largest(&plus.sigma);
    assert!(
        scale > 1e-8,
        "a magnetic spin-orbit metal has a Hall conductivity: {scale:e}"
    );
    for a in 0..3 {
        for b in 0..3 {
            let sum = plus.sigma[a][b] + minus.sigma[a][b];
            assert!(sum.abs() < 1e-5 * scale, "({a}, {b}): {sum:e} of {scale:e}");
        }
    }
}

#[test]
#[ignore = "slow: 295 s (measured 2026-11-11, 162 MB); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_metals -- --ignored nickel --nocapture"]
fn nickel_conductivity_is_odd_in_the_moment_unchanged_by_handedness_and_along_the_moment() {
    let mesh = [4, 4, 4];
    let up = solve(&nickel(0.6));
    let plus = hall::anomalous_hall(&up.sampler(), mesh).unwrap();
    report("Ni +M", &plus);
    let minus = hall::anomalous_hall(&solve(&nickel(-0.6)).sampler(), mesh).unwrap();
    report("Ni -M", &minus);
    odd(&plus, &minus);
    let left = solve(&swapped(&nickel(0.6)));
    assert!(left.setup.lattice.determinant() < 0.);
    let relabeled = hall::anomalous_hall(&left.sampler(), mesh).unwrap();
    report("Ni +M, a_1 <-> a_2", &relabeled);
    let scale = largest(&plus.sigma);
    for a in 0..3 {
        for b in 0..3 {
            let d = plus.sigma[a][b] - relabeled.sigma[a][b];
            assert!(d.abs() < 1e-5 * scale, "({a}, {b}): {d:e} of {scale:e}");
        }
    }
    // M along z: sigma_xy carries the tensor; on 4^3 the cubic symmetry is kept to 1e-3.
    assert!(plus.sigma[1][2].abs() < 1e-3 * plus.sigma[0][1].abs());
    assert!(plus.sigma[2][0].abs() < 1e-3 * plus.sigma[0][1].abs());
    // Not converged, and it says so: most plaquettes merge band groups.
    assert!(
        plus.warnings.iter().any(|w| w.contains("plaquettes")),
        "{:?}",
        plus.warnings
    );
}

#[test]
#[ignore = "slow: 370 s (measured 2026-11-11, 210 MB); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test berry_metals -- --ignored iron --nocapture"]
fn iron_conductivity_is_odd_in_the_moment_with_equal_in_plane_numbers() {
    let mesh = [5, 5, 5];
    let plus = hall::anomalous_hall(&solve(&iron(2.)).sampler(), mesh).unwrap();
    report("Fe +M", &plus);
    let minus = hall::anomalous_hall(&solve(&iron(-2.)).sampler(), mesh).unwrap();
    report("Fe -M", &minus);
    odd(&plus, &minus);
    // a_1 and a_2 are images of each other under the operations that keep M along z.
    let nu = plus.numbers;
    assert!(
        (nu[0] - nu[1]).abs() < 1e-8 * nu[0].abs().max(1e-12),
        "{nu:?}"
    );
    // The convergence metric (vera): with M along z the tensor must be sigma_xy alone; on 5^3
    // sigma_yz / sigma_xy is far from zero. Reported, not asserted.
    eprintln!(
        "Fe 5^3: sigma_yz / sigma_xy = {:.3}",
        plus.sigma[1][2] / plus.sigma[0][1]
    );
}
