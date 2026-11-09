//! The row-center derivative of the basis (`periodic::setup::basis_gradient`): `radial_slope`,
//! `BasisGradient.evaluate` and the short-range cross block, against the frozen Python (Si and
//! Al, SZ, `lda-sr`; gate `max(1e-12, 100 x` the measured floor of each array`)`), against
//! central differences of the functions they differentiate, and against the translation
//! identity `d M / d c = -(G + G^+)` of the sphere matrix. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_shells_support;

use mandacaru_core::{
    basis::table,
    math::angular,
    periodic::setup::{
        basis_gradient::{basis_gradient, radial_slope, short_range_cross, subset, ShortRange},
        shells::{Shell, Shells},
    },
};
use nalgebra::Vector3;
use num_complex::Complex64;
use setup_gradients_support::*;
use std::sync::Arc;

fn radial_of(case: &Case, table_index: usize) -> (Vec<f64>, Vec<f64>) {
    let t = &case.inputs()["basis_radial_tables"][table_index];
    (Array::read(&t["r"]).re, Array::read(&t["values"]).re)
}

#[test]
fn the_radial_slope_equals_the_frozen_python_and_the_difference_of_the_function() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let nodes = Array::read(&case.inputs()["radial_slope_r_bohr"]).re;
        let want = case.quantity("radial_slope");
        let functions = case.inputs()["basis_functions"].as_array().unwrap();
        let mut got = Vec::new();
        let mut fd = 0.0f64;
        for f in functions {
            let (l, index) = (
                f["l"].as_u64().unwrap() as usize,
                f["table"].as_u64().unwrap() as usize,
            );
            let (r, values) = radial_of(&case, index);
            let slope = radial_slope(&ctx, &r, &values, l, &nodes).unwrap();
            // Away from the nodes of the table the central difference of the function.
            let fd_step = 1e-6;
            let plus = table::evaluate(
                &ctx,
                &r,
                &values,
                l,
                &nodes.iter().map(|x| x + fd_step).collect::<Vec<_>>(),
                false,
            )
            .unwrap();
            let minus = table::evaluate(
                &ctx,
                &r,
                &values,
                l,
                &nodes.iter().map(|x| x - fd_step).collect::<Vec<_>>(),
                false,
            )
            .unwrap();
            for (j, x) in nodes.iter().enumerate() {
                let near_node =
                    r.iter().any(|b| (b - x).abs() < 2. * fd_step) || *x < 2. * fd_step + r[0];
                if !near_node && *x < r[r.len() - 1] - 2. * fd_step {
                    fd = fd.max((slope[j] - (plus[j] - minus[j]) / (2. * fd_step)).abs());
                }
            }
            got.extend(slope);
        }
        let miss = relative_real(&got, &want.re);
        let gate = case.gate("radial_slope");
        println!("{name}: radial slope {miss:.2e} (gate {gate:.1e}), central difference {fd:.2e}");
        assert!(miss <= gate && fd < 1e-8);
        // Control: the value instead of the slope.
        let (r, values) = radial_of(&case, 0);
        let value = table::evaluate(&ctx, &r, &values, 0, &nodes, false).unwrap();
        let first = &want.re[..nodes.len()];
        assert!(relative_real(&value, first) > 1e3 * gate);
    }
}

fn single(shell: &Shell, m: i32) -> Shell {
    Shell {
        ms: vec![m],
        rows: vec![0],
        ..shell.clone()
    }
}

#[test]
fn the_basis_gradient_equals_the_frozen_python_and_the_difference_of_the_function() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let points = triples(&Array::read(&case.inputs()["basis_gradient_points_bohr"]).re);
        let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            points.iter().map(|p| p[0]).collect(),
            points.iter().map(|p| p[1]).collect(),
            points.iter().map(|p| p[2]).collect(),
        );
        let (shells, functions) = case.basis(&ctx);
        let want = case.quantity("BasisGradient.evaluate").complex();
        let (nf, np) = (functions.len(), points.len());
        let mut got = vec![Complex64::default(); want.len()];
        let mut fd = 0.0f64;
        for (f, &(_, l, m, center)) in functions.iter().enumerate() {
            let shell = shells
                .shells()
                .iter()
                .find(|s| s.rows.contains(&f))
                .map(|s| single(s, m))
                .unwrap();
            for axis in 0..3 {
                let values = basis_gradient(&ctx, &shell, m, axis, [&x, &y, &z]).unwrap();
                got[(f * 3 + axis) * np..(f * 3 + axis + 1) * np].copy_from_slice(&values);
                // Central difference of R(r) Y_lm(x - c), written here from the table.
                let fd_step = 1e-5;
                let (r, tab) = {
                    let table_index = case.inputs()["basis_functions"][f]["table"]
                        .as_u64()
                        .unwrap() as usize;
                    radial_of(&case, table_index)
                };
                let value = |p: [f64; 3]| -> Complex64 {
                    let d = Vector3::from(p) - Vector3::from(center);
                    let rho = d.norm();
                    let radial = table::evaluate(&ctx, &r, &tab, l, &[rho], false).unwrap()[0];
                    angular::complex(l, m, [d[0] / rho, d[1] / rho, d[2] / rho]) * radial
                };
                for (j, p) in points.iter().enumerate() {
                    let away = (Vector3::from(*p) - Vector3::from(center)).norm();
                    if away < 1e-3 || away > r[r.len() - 1] - 1e-3 {
                        continue;
                    }
                    let (mut up, mut down) = (*p, *p);
                    up[axis] += fd_step;
                    down[axis] -= fd_step;
                    let numeric = (value(up) - value(down)) / (2. * fd_step);
                    fd = fd.max((values[j] - numeric).norm());
                }
            }
        }
        let miss = relative(&got, &want);
        let gate = case.gate("BasisGradient.evaluate");
        println!("{name}: basis gradient {miss:.2e} (gate {gate:.1e}), central difference {fd:.2e} of {:.2e}", want.iter().fold(0f64, |a, v| a.max(v.norm())));
        assert!(miss <= gate && fd < 1e-8, "{nf} functions");
        // Control: the axis order reversed.
        let swapped: Vec<Complex64> = (0..nf)
            .flat_map(|f| {
                (0..3)
                    .rev()
                    .flat_map(move |a| (0..np).map(move |j| (f, a, j)))
            })
            .map(|(f, a, j)| got[(f * 3 + a) * np + j])
            .collect();
        assert!(relative(&swapped, &want) > 1e3 * gate);
    }
}

fn kpoints(case: &Case) -> Vec<[f64; 3]> {
    triples(&Array::read(&case.inputs()["short_range_cross"]["kpoints_bohr_inv"]).re)
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test setup_basis_gradient -- --ignored the_short_range_cross_blocks_equal_the_frozen_python"]
fn the_short_range_cross_blocks_equal_the_frozen_python() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let cell = case.cell();
        let potentials = case.potentials(&ctx);
        let (shells, functions) = case.basis(&ctx);
        let m = functions.len();
        let k = kpoints(&case);
        let rows: Vec<usize> = case.inputs()["short_range_cross"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let cols: Vec<usize> = case.inputs()["short_range_cross"]["cols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        for atom in 0..cell.atoms.len().min(2) {
            let sphere = ShortRange {
                center: cell.atoms[atom].center,
                potential: &potentials[atom],
                sigma: cell.sigma,
            };
            // even rows against odd columns
            let bra = subset(&shells, &rows).unwrap();
            let ket = subset(&shells, &cols).unwrap();
            let cross =
                short_range_cross(&ctx, &cell.lattice, &k, &sphere, &bra, None, &ket).unwrap();
            let key = format!("short_range_cross.even_odd.atom{atom}");
            let miss = relative(cross.as_slice(), &case.quantity(&key).complex());
            let gate = case.gate(&key);
            println!("{name} atom {atom}: even/odd block {miss:.2e} (gate {gate:.1e})");
            assert!(miss <= gate);
            // the derivative bra against the whole basis: (k, axis * M + mu, nu)
            let mut got = vec![Complex64::default(); k.len() * 3 * m * m];
            for axis in 0..3 {
                let block = short_range_cross(
                    &ctx,
                    &cell.lattice,
                    &k,
                    &sphere,
                    &shells,
                    Some(axis),
                    &shells,
                )
                .unwrap();
                for ik in 0..k.len() {
                    for mu in 0..m {
                        for nu in 0..m {
                            got[(ik * 3 * m + axis * m + mu) * m + nu] =
                                block.as_slice()[(ik * m + mu) * m + nu];
                        }
                    }
                }
            }
            let key = format!("short_range_cross.gradient_bra.atom{atom}");
            let miss = relative(&got, &case.quantity(&key).complex());
            let gate = case.gate(&key);
            println!("{name} atom {atom}: gradient bra {miss:.2e} (gate {gate:.1e})");
            assert!(miss <= gate);
            // Control: the plain bra (values) in place of the derivative.
            let plain = short_range_cross(&ctx, &cell.lattice, &k, &sphere, &shells, None, &shells)
                .unwrap();
            let tiled: Vec<Complex64> = (0..k.len())
                .flat_map(|ik| (0..3).flat_map(move |_| (0..m * m).map(move |i| (ik, i))))
                .map(|(ik, i)| plain.as_slice()[ik * m * m + i])
                .collect();
            assert!(relative(&tiled, &case.quantity(&key).complex()) > 1e3 * gate);
        }
    }
}

#[test]
fn the_cross_block_obeys_the_translation_identity_of_a_rigid_shift() {
    for name in ["si", "al2"] {
        let case = Case::load(name);
        let ctx = context();
        let cell = case.cell();
        let potentials = case.potentials(&ctx);
        let (shells, _) = case.basis(&ctx);
        let k = kpoints(&case);
        let m = shells.functions();
        let sphere = ShortRange {
            center: cell.atoms[0].center,
            potential: &potentials[0],
            sigma: cell.sigma,
        };
        // Every basis center moved by `delta`, the sphere held: d M / d c = -(G + G^+).
        let shifted = |axis: usize, delta: f64| -> Shells {
            Shells::new(
                shells
                    .shells()
                    .iter()
                    .map(|s| {
                        let mut s = s.clone();
                        s.center[axis] += delta;
                        s.radial = Arc::clone(&s.radial);
                        s
                    })
                    .collect(),
            )
            .unwrap()
        };
        let fd_step = 1e-4;
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for axis in 0..3 {
            let g = short_range_cross(
                &ctx,
                &cell.lattice,
                &k,
                &sphere,
                &shells,
                Some(axis),
                &shells,
            )
            .unwrap();
            let (up, down) = (shifted(axis, fd_step), shifted(axis, -fd_step));
            let plus = short_range_cross(&ctx, &cell.lattice, &k, &sphere, &up, None, &up).unwrap();
            let minus =
                short_range_cross(&ctx, &cell.lattice, &k, &sphere, &down, None, &down).unwrap();
            for ik in 0..k.len() {
                for mu in 0..m {
                    for nu in 0..m {
                        let analytic = -(g.as_slice()[(ik * m + mu) * m + nu]
                            + g.as_slice()[(ik * m + nu) * m + mu].conj());
                        let numeric = (plus.as_slice()[(ik * m + mu) * m + nu]
                            - minus.as_slice()[(ik * m + mu) * m + nu])
                            / (2. * fd_step);
                        worst = worst.max((analytic - numeric).norm());
                        scale = scale.max(analytic.norm());
                    }
                }
            }
        }
        println!(
            "{name}: translation identity {:.2e} of max {scale:.2e}",
            worst / scale
        );
        assert!(scale > 1e-2 && worst < 1e-7 * scale);
    }
}
