//! The explicit derivatives of the lattice sums (`periodic::setup::gradients`): the frozen
//! Python gradients of the ion pair sum and of the short-range ion compensation (Si and Al
//! cells with atom 1 displaced; gate `100 x` the measured floor of each array), their central
//! differences of the Rust sums, the translation sum rules, and the dense `U`, `U_ion`, `E_ion`
//! and mesh contractions (E1-E3), which have no Python twin, against central differences of the
//! Rust sums they differentiate. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_support;

use mandacaru_core::{
    math::reciprocal::alias_table,
    periodic::setup::{
        electrostatics::{couplings, hats, ion_charge},
        gradients::{
            dense_gradients, hat_overlaps, ion_constants_gradient, ion_constants_strain,
            ion_overlaps, short_range_ion_compensation_gradient,
        },
        ions::{ion_constants, short_range_ion_compensation},
        shapes::{Shapes, COMPENSATION_CUTOFF},
    },
};
use num_complex::Complex64;
use setup_gradients_support::*;

const STEP: f64 = 1e-4;

#[test]
fn the_ion_pair_gradient_equals_the_frozen_python_and_the_difference_of_the_sum() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let cell = case.cell();
        let got = ion_constants_gradient(&cell.sites()).unwrap();
        let flat: Vec<f64> = got.iter().flatten().copied().collect();
        let want = case.quantity("ion_constants_gradient").re;
        let miss = relative_real(&flat, &want);
        let gate = case.gate("ion_constants_gradient");
        // The sum it differentiates, as the Rust twin computes it (and the frozen value).
        let sum = ion_constants(&cell.sites()).unwrap();
        assert!((sum - case.quantity("ion_constants").re[0]).abs() < 1e-12 * sum.abs());
        let mut fd = 0.0f64;
        for atom in 0..got.len() {
            for axis in 0..3 {
                let plus = ion_constants(&cell.moved(atom, axis, STEP).sites()).unwrap();
                let minus = ion_constants(&cell.moved(atom, axis, -STEP).sites()).unwrap();
                fd = fd.max((got[atom][axis] - (plus - minus) / (2. * STEP)).abs());
            }
        }
        let scale = flat.iter().fold(0f64, |m, x| m.max(x.abs()));
        let sum_rule: f64 = (0..3)
            .map(|k| got.iter().map(|g| g[k]).sum::<f64>().abs())
            .fold(0., f64::max);
        println!("{name}: frozen {miss:.2e} (gate {gate:.1e}), central difference {fd:.2e} of max {scale:.2e}, sum rule {sum_rule:.1e}");
        assert!(miss <= gate && sum_rule < 1e-13);
        // The existing gate (absolute 1e-8) is vacuous on a small array: also relative.
        assert!(fd < 1e-8 && fd <= 1e-5 * scale.max(1e-300) + 1e-14);
        if name != "al" {
            assert!(scale > 1e-6, "the displaced cell must carry a force");
            // Controls: the opposite sign and a 1 % wrong width each miss by orders.
            let negated: Vec<f64> = flat.iter().map(|x| -x).collect();
            assert!(relative_real(&negated, &want) > 100. * gate);
            let mut wrong = cell.clone();
            wrong.sigma *= 1.01;
            let other: Vec<f64> = ion_constants_gradient(&wrong.sites())
                .unwrap()
                .iter()
                .flatten()
                .copied()
                .collect();
            assert!(relative_real(&other, &want) > 100. * gate);
        } else {
            assert!(scale == 0., "one atom: translation invariance");
        }
    }
}

#[test]
fn the_ion_compensation_gradient_equals_the_frozen_python_and_the_difference_of_the_sum() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let cell = case.cell();
        let potentials = case.potentials(&ctx);
        let got = short_range_ion_compensation_gradient(&ctx, &cell.sites(), &potentials).unwrap();
        // (channel, atom, axis)
        let flat: Vec<Complex64> = got.iter().flatten().flatten().copied().collect();
        let quantity = case.quantity("short_range_ion_compensation_gradient");
        let want = quantity.complex();
        let miss = relative(&flat, &want);
        let gate = case.gate("short_range_ion_compensation_gradient");
        let scale = flat.iter().fold(0f64, |m, x| m.max(x.norm()));
        // Translation sum rule over the atoms, per channel and axis.
        let sum_rule = got
            .iter()
            .flat_map(|channel| {
                (0..3).map(move |k| channel.iter().map(|a| a[k]).sum::<Complex64>().norm())
            })
            .fold(0., f64::max);
        // The sums it differentiates, central difference of the Rust sum.
        let sites = cell.sites();
        let base = short_range_ion_compensation(&ctx, &sites, &potentials).unwrap();
        let want_sum = case.quantity("short_range_ion_compensation").complex();
        assert!(relative(&base, &want_sum) <= case.gate("short_range_ion_compensation"));
        let mut fd = 0.0f64;
        let probes: Vec<(usize, usize)> = if cell.atoms.len() > 1 {
            vec![(0, 0), (1, 2)]
        } else {
            vec![(0, 1)]
        };
        for (atom, axis) in probes {
            let plus = short_range_ion_compensation(
                &ctx,
                &cell.moved(atom, axis, STEP).sites(),
                &potentials,
            )
            .unwrap();
            let minus = short_range_ion_compensation(
                &ctx,
                &cell.moved(atom, axis, -STEP).sites(),
                &potentials,
            )
            .unwrap();
            for c in 0..got.len() {
                let numeric = (plus[c] - minus[c]) / (2. * STEP);
                fd = fd.max((got[c][atom][axis] - numeric).norm());
            }
        }
        println!("{name}: frozen {miss:.2e} (gate {gate:.1e}), central difference {fd:.2e} of max {scale:.2e}, sum rule {sum_rule:.1e}");
        assert!(miss <= gate && sum_rule < 1e-12 && fd < 1e-7);
        if scale > 0. {
            assert!(
                fd <= 1e-5 * scale + 1e-14,
                "relative gate for the small Al arrays"
            );
            // Controls: the opposite sign, and a gradient without the partner's recoil.
            let negated: Vec<Complex64> = flat.iter().map(|x| -x).collect();
            assert!(relative(&negated, &want) > 1.);
            let owner_only: Vec<Complex64> = got
                .iter()
                .enumerate()
                .flat_map(|(c, channel)| {
                    let owner = cell.channels[c].0;
                    channel.iter().enumerate().flat_map(move |(a, v)| {
                        v.iter()
                            .map(move |x| if a == owner { *x } else { Complex64::default() })
                    })
                })
                .collect();
            assert!(relative(&owner_only, &want) > 0.1);
        } else {
            assert!(name == "al" && sum_rule == 0.);
        }
    }
}

#[test]
fn the_dense_gradients_are_the_difference_of_the_dense_sums() {
    let si = setup_support::Silicon::load();
    let ctx = context();
    let cell = Cell {
        atoms: si.atoms.clone(),
        channels: si.channels.clone(),
        lattice: si.lattice,
        sigma: si.sigma,
    };
    // Displace atom 1 so that nothing is symmetric, as the Python fixture does.
    let cell = cell.moved(1, 0, 0.05).moved(1, 1, 0.02).moved(1, 2, -0.03);
    let sites = cell.sites();
    let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
    let gradients = dense_gradients(&ctx, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
    let rows = cell.channels.len();
    let (mut worst_u, mut worst_ion, mut worst_e) = (0.0f64, 0.0f64, 0.0f64);
    let (mut scale_u, mut scale_ion, mut scale_e) = (0.0f64, 0.0f64, 0.0f64);
    let mut flipped_u = 0.0f64;
    for atom in 0..cell.atoms.len() {
        for axis in 0..3 {
            let moved = |delta: f64| {
                let c = cell.moved(atom, axis, delta);
                let s = Shapes::new(&ctx, &c.sites(), 20.).unwrap();
                couplings(&ctx, &s, &c.sites(), COMPENSATION_CUTOFF).unwrap()
            };
            let (plus, minus) = (moved(STEP), moved(-STEP));
            let got = gradients.get(atom, axis);
            for i in 0..rows * rows {
                let numeric = (plus.u.as_slice()[i] - minus.u.as_slice()[i]) / (2. * STEP);
                worst_u = worst_u.max((got.u[i] - numeric).norm());
                flipped_u = flipped_u.max((-got.u[i] - numeric).norm());
                scale_u = scale_u.max(got.u[i].norm());
            }
            for a in 0..rows {
                let numeric = (plus.u_ion.as_slice()[a] - minus.u_ion.as_slice()[a]) / (2. * STEP);
                worst_ion = worst_ion.max((got.u_ion[a] - numeric).norm());
                scale_ion = scale_ion.max(got.u_ion[a].norm());
            }
            let numeric = (plus.e_ion - minus.e_ion) / (2. * STEP);
            worst_e = worst_e.max((got.e_ion - numeric).abs());
            scale_e = scale_e.max(got.e_ion.abs());
        }
    }
    println!("dense gradients against central differences (relative to the array maximum): U {:.2e} (max {scale_u:.2e}), U_ion {:.2e} (max {scale_ion:.2e}), E_ion {:.2e} (max {scale_e:.2e})", worst_u / scale_u, worst_ion / scale_ion, worst_e / scale_e);
    assert!(
        scale_u > 1e-3 && scale_ion > 1e-3 && scale_e > 1e-3,
        "non-vacuous"
    );
    assert!(worst_u < 1e-7 * scale_u && worst_ion < 1e-7 * scale_ion && worst_e < 1e-7 * scale_e);
    // Translation: moving every atom together changes none of the sums.
    for axis in 0..3 {
        let total_e: f64 = (0..cell.atoms.len())
            .map(|a| gradients.get(a, axis).e_ion)
            .sum();
        assert!(total_e.abs() < 1e-12 * scale_e);
        for a in 0..rows {
            let ion: Complex64 = (0..cell.atoms.len())
                .map(|n| gradients.get(n, axis).u_ion[a])
                .sum();
            assert!(ion.norm() < 1e-12 * scale_ion);
            for b in 0..rows {
                let u: Complex64 = (0..cell.atoms.len())
                    .map(|n| gradients.get(n, axis).u[a * rows + b])
                    .sum();
                assert!(u.norm() < 1e-12 * scale_u);
            }
        }
    }
    // Control: the opposite sign misses the central difference by twice the array.
    assert!(flipped_u > 100. * 1e-7 * scale_u);
}

fn weights(n: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..n).map(|_| Complex64::new(next(), next())).collect()
}

#[test]
fn the_mesh_contractions_are_the_difference_of_the_hats_and_ions_on_the_grid() {
    let si = setup_support::Silicon::load();
    let ctx = context();
    let cell = Cell {
        atoms: si.atoms.clone(),
        channels: si.channels.clone(),
        lattice: si.lattice,
        sigma: si.sigma,
    }
    .moved(1, 0, 0.05)
    .moved(1, 2, -0.03);
    let sites = cell.sites();
    let q_max =
        mandacaru_core::periodic::setup::electrostatics::max_wavenumber(&ctx, &si.grid).unwrap();
    let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
    let n = 1000;
    // The alias star is nonempty on this grid, so the alias-resolved derivative is tested.
    assert!(!alias_table(&ctx, &si.grid).unwrap().is_empty());
    let w = weights(n, 7);
    let got_hats = hat_overlaps(&ctx, &si.grid, &shapes, &sites, &w).unwrap();
    let got_ions = ion_overlaps(&ctx, &si.grid, &sites, &w).unwrap();
    let contract =
        |values: &[Complex64]| -> Complex64 { values.iter().zip(&w).map(|(v, w)| v * w).sum() };
    let (mut worst_h, mut scale_h, mut worst_i, mut scale_i) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut flipped = 0.0f64;
    for atom in 0..cell.atoms.len() {
        for axis in 0..3 {
            let moved = |delta: f64| {
                let c = cell.moved(atom, axis, delta);
                let hat_values = hats(&ctx, &si.grid, &shapes, &c.sites()).unwrap();
                let rows: Vec<Complex64> = (0..cell.channels.len())
                    .map(|a| contract(&hat_values.as_slice()[a * n..(a + 1) * n]))
                    .collect();
                let ion = contract(ion_charge(&ctx, &si.grid, &c.sites()).unwrap().as_slice());
                (rows, ion)
            };
            let ((hp, ip), (hm, im)) = (moved(STEP), moved(-STEP));
            for a in 0..cell.channels.len() {
                let numeric = (hp[a] - hm[a]) / (2. * STEP);
                let analytic = if cell.channels[a].0 == atom {
                    got_hats[a][axis]
                } else {
                    Complex64::default()
                };
                worst_h = worst_h.max((analytic - numeric).norm());
                flipped = flipped.max((-analytic - numeric).norm());
                scale_h = scale_h.max(analytic.norm());
            }
            let numeric = (ip - im) / (2. * STEP);
            worst_i = worst_i.max((got_ions[atom][axis] - numeric).norm());
            scale_i = scale_i.max(got_ions[atom][axis].norm());
        }
    }
    println!("mesh contractions against central differences (relative): hats {:.2e} (max {scale_h:.2e}), ions {:.2e} (max {scale_i:.2e})", worst_h / scale_h, worst_i / scale_i);
    assert!(scale_h > 1e-3 && scale_i > 1e-3);
    assert!(worst_h < 1e-7 * scale_h && worst_i < 1e-7 * scale_i);
    // Control: the opposite sign misses the central difference by twice the array.
    assert!(flipped > 100. * 1e-7 * scale_h);
}

#[test]
fn the_ion_pair_strain_derivative_is_the_difference_of_the_sum_under_a_strain() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let cell = case.cell();
        let got = ion_constants_strain(&cell.sites()).unwrap();
        let fd_step = 1e-5;
        let (mut worst, mut scale, mut flipped) = (0.0f64, 0.0f64, 0.0f64);
        for a in 0..3 {
            for b in 0..3 {
                let sum = |delta: f64| {
                    let mut eps = [[0.; 3]; 3];
                    eps[a][b] = delta;
                    ion_constants(&cell.strained(eps).sites()).unwrap()
                };
                let numeric = (sum(fd_step) - sum(-fd_step)) / (2. * fd_step);
                worst = worst.max((got[a][b] - numeric).abs());
                flipped = flipped.max((-got[a][b] - numeric).abs());
                scale = scale.max(got[a][b].abs());
                assert!((got[a][b] - got[b][a]).abs() < 1e-13 * scale.max(1.));
            }
        }
        println!("{name}: strain derivative {worst:.2e} of max {scale:.2e}");
        assert!(scale > 1e-5 && worst < 1e-7, "{name}");
        // Control: the opposite sign misses the difference by twice the array.
        assert!(flipped > 100. * 1e-7, "{name}");
    }
}

#[test]
fn the_local_potential_slope_equals_the_frozen_python_and_the_difference_of_the_value() {
    for name in ["si", "al2"] {
        let case = Case::load(name);
        let ctx = context();
        let potential = &case.potentials(&ctx)[0];
        let radius = Array::read(&case.inputs()["local_potential_radius_bohr"]).re;
        let slope: Vec<f64> = radius.iter().map(|r| potential.slope(*r)).collect();
        let value: Vec<f64> = radius.iter().map(|r| potential.value(*r)).collect();
        let miss = relative_real(&slope, &case.quantity("local_potential_derivative").re);
        let gate = case.gate("local_potential_derivative");
        let value_miss = relative_real(&value, &case.quantity("local_potential").re);
        let step = 1e-6;
        let fd = radius
            .iter()
            .zip(&slope)
            .map(|(r, s)| {
                let numeric = (potential.value(r + step) - potential.value(r - step)) / (2. * step);
                (s - numeric).abs()
            })
            .fold(0., f64::max);
        println!("{name}: slope {miss:.2e} (gate {gate:.1e}), value {value_miss:.2e}, central difference {fd:.2e}");
        // The existing test's gate (2e-6) holds away from the table's first node, where the
        // potential is flat below and the slope jumps.
        assert!(miss <= gate && value_miss <= 1e-12 && fd < 2e-6);
        // Control: the value in place of the slope.
        assert!(
            relative_real(&value, &case.quantity("local_potential_derivative").re) > 1e3 * gate
        );
    }
}
