//! The periodic setup's electrostatics against the frozen Si crystal
//! (`si_crystal.json`, T1 = 1e-10 of the array maximum) and against checks that
//! need no frozen number: the shape transforms against an independent
//! quadrature and the small-q law, the translation law, Hermiticity and
//! positivity of `U`, the tail of the cutoff, and the blocked dense sum under a
//! small memory limit.
mod setup_support;

use mandacaru_core::{
    periodic::setup::{
        electrostatics::{coulomb, couplings, hats, ion_charge, kernel, max_wavenumber},
        shapes::{Shapes, Sites, COMPENSATION_CUTOFF},
    },
    Context, Error,
};
use nalgebra::DMatrix;
use num_complex::Complex64;
use setup_support::*;
use std::f64::consts::PI;

const T1: f64 = 1e-10;

fn sites(si: &Silicon) -> Sites<'_> {
    Sites {
        atoms: &si.atoms,
        channels: &si.channels,
        lattice: si.lattice,
        sigma: si.sigma,
    }
}

#[test]
fn the_grid_arrays_equal_the_frozen_crystal() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let sites = sites(&si);
    {
        let q_max = max_wavenumber(&ctx, &si.grid).unwrap();
        let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
        println!(
            "Si 10^3 grid: largest wave number {q_max:.4}, shape tables to {}",
            shapes.top()
        );
        assert!(q_max > 9. && q_max < 10. && shapes.top() == 120.);
        let k = kernel(&ctx, &si.grid).unwrap();
        let want = reals(&si.json["kernel"]);
        let kernel_error = relative(k.as_slice(), &want);
        let ion = ion_charge(&ctx, &si.grid, &sites).unwrap();
        let ion_error = relative(ion.as_slice(), &complexes(&si.json["ion"]));
        let h = hats(&ctx, &si.grid, &shapes, &sites).unwrap();
        // A vector beyond the tables raises instead of reading a clamped value.
        let mut out = vec![Complex64::default(); si.channels.len()];
        let far = [[shapes.top() + 1., 0., 0.]];
        assert!(shapes.transforms(&sites, &far, &mut out, 1).is_err());
        let hats_want = complexes(&si.json["hats"]);
        assert_eq!(h.as_slice().len(), si.channels.len() * 1000);
        let hats_error = relative(h.as_slice(), &hats_want);
        println!(
            "Si 10^3 grid: kernel {kernel_error:.2e}, ion {ion_error:.2e}, hats {hats_error:.2e} (relative to the array maximum)"
        );
        assert!(kernel_error < T1 && ion_error < T1 && hats_error < T1);
        // Control: the fcc grid cuts the alias star, so the box value alone is not
        // the frozen kernel and the alias mean is what the test gates.
        let b = mandacaru_core::math::reciprocal::reciprocal_cell(&si.grid).unwrap();
        let plain: Vec<f64> = (0..1000)
            .map(|at| {
                coulomb(mandacaru_core::math::reciprocal::box_vector(
                    &b, [10; 3], at,
                ))
            })
            .collect();
        let control = relative(&plain, &want);
        println!("control, the box value without the alias mean: kernel error {control:.2e}");
        assert!(control > 1e-3);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_dense_sums_equal_the_frozen_crystal_and_obey_their_identities() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let sites = sites(&si);
    {
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let c = couplings(&ctx, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
        let (u_error, uion_error) = (
            relative(c.u.as_slice(), &complexes(&si.json["u"])),
            relative(c.u_ion.as_slice(), &complexes(&si.json["uion"])),
        );
        let eion_want = si.json["eion"].as_f64().unwrap();
        let eion_error = (c.e_ion - eion_want).abs() / eion_want.abs();
        println!("Si dense sums: u {u_error:.2e}, uion {uion_error:.2e}, eion {eion_error:.2e}");
        assert!(u_error < T1 && uion_error < T1 && eion_error < T1);

        // U is Hermitian (by construction) and positive semidefinite: it is a
        // Gram matrix of the shapes under the Coulomb kernel.
        let n = si.channels.len();
        let matrix = DMatrix::from_fn(n, n, |a, b| c.u.as_slice()[a * n + b]);
        let eigenvalues = matrix.clone().symmetric_eigen().eigenvalues;
        let lowest = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "U: lowest eigenvalue {lowest:.3e}, highest {:.3e}",
            eigenvalues.max()
        );
        assert!((&matrix - matrix.adjoint()).norm() < 1e-13 * matrix.norm());
        assert!(lowest > -1e-10);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// A trapezoid-free reference: Gauss-Legendre on `[0, r_g]` of `g_L j_L(qr) r^2`.
fn quadrature(l: usize, radius: f64, q: f64) -> f64 {
    use mandacaru_core::math::radial::gauss_legendre;
    use mandacaru_core::paw::multipole::shape_function;
    use mandacaru_kernels::two_center::spherical_bessel;
    let rule = gauss_legendre(400).unwrap();
    let r: Vec<f64> = rule.iter().map(|(x, _)| 0.5 * radius * (x + 1.)).collect();
    let g = shape_function(&r, radius, l).unwrap();
    rule.iter()
        .zip(r.iter().zip(&g))
        .map(|((_, w), (r, g))| 0.5 * radius * w * g * spherical_bessel(l as i32, q * r) * r * r)
        .sum()
}

#[test]
fn the_shape_tables_obey_the_moment_law_and_an_independent_quadrature() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let sites = sites(&si);
    {
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let radius = si.atoms[0].compensation_radius;
        let (mut worst, mut between) = (0.0f64, 0.0f64);
        for l in 0..3usize {
            let table = shapes.table(radius, l).unwrap();
            assert_eq!(table.q.len(), 12001);
            assert_eq!(table.q[0], 0.);
            // F_L(q) -> q^L / (2L+1)!! with the unit L-th moment of the shape.
            let double: f64 = (1..=l).map(|k| (2 * k + 1) as f64).product();
            let q: f64 = 0.05;
            let law = q.powi(l as i32) / double;
            let got = table.at(q);
            let miss = (got - law).abs() / law.abs().max(1e-300);
            println!("L = {l}: F(0.05) = {got:.10e} against q^L/(2L+1)!! = {law:.10e} (relative {miss:.2e})");
            assert!(miss < 2e-3 * (l as f64 + 1.));
            let peak = (0..table.q.len())
                .map(|i| table.f[i].abs())
                .fold(0., f64::max);
            // On the nodes: the trapezoid rule of the table against Gauss-Legendre.
            for q in [0.0, 0.37, 1.9, 6.0, 15.0, 40.0, 119.0] {
                worst = worst.max((table.at(q) - quadrature(l, radius, q)).abs() / peak);
            }
            // Between the nodes: the linear interpolation the oracle also uses.
            for q in [0.375, 1.937, 6.0043, 15.0011, 40.0307, 118.9937] {
                between = between.max((table.at(q) - quadrature(l, radius, q)).abs() / peak);
            }
        }
        println!("shape tables against 400-point Gauss-Legendre (relative to each table's peak): on the nodes {worst:.2e}, between them {between:.2e}");
        assert!(worst < 1e-9 && between < 1e-4);
        // The monopole shape carries unit charge: its transform at G = 0 is
        // 4 pi times F_0(0) times Y_00 = sqrt(4 pi); every L > 0 channel vanishes.
        let mut out = vec![Complex64::default(); si.channels.len()];
        shapes.transforms(&sites, &[[0.; 3]], &mut out, 1).unwrap();
        for (value, &(atom, l, _)) in out.iter().zip(&si.channels) {
            // The phase of the atom is one at G = 0.
            let _ = atom;
            if l == 0 {
                assert!((value - Complex64::new((4. * PI).sqrt(), 0.)).norm() < 1e-9);
            } else {
                assert!(value.norm() < 1e-12);
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_global_translation_phases_the_shapes_and_leaves_the_sums_alone() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let t = [0.713, -1.291, 0.402];
    let moved: Vec<_> = si
        .atoms
        .iter()
        .map(|a| {
            let mut a = *a;
            for k in 0..3 {
                a.center[k] += t[k];
            }
            a
        })
        .collect();
    let (before, after) = (
        sites(&si),
        Sites {
            atoms: &moved,
            channels: &si.channels,
            lattice: si.lattice,
            sigma: si.sigma,
        },
    );
    {
        let shapes = Shapes::new(&ctx, &before, 20.).unwrap();
        let g = [[0.37, -1.2, 2.9], [3.1, 0.2, -0.6], [0., 0., 1.7]];
        let n = si.channels.len();
        let (mut a, mut b) = (
            vec![Complex64::default(); n * 3],
            vec![Complex64::default(); n * 3],
        );
        shapes.transforms(&before, &g, &mut a, 3).unwrap();
        shapes.transforms(&after, &g, &mut b, 3).unwrap();
        let mut worst = 0.0f64;
        for row in 0..n {
            for (v, gv) in g.iter().enumerate() {
                let angle = -(gv[0] * t[0] + gv[1] * t[1] + gv[2] * t[2]);
                let phase = Complex64::new(angle.cos(), angle.sin());
                worst = worst.max((b[row * 3 + v] - a[row * 3 + v] * phase).norm());
            }
        }
        println!("translation law of the channel transforms: {worst:.2e}");
        assert!(worst < 1e-13);
        let (c0, c1) = (
            couplings(&ctx, &shapes, &before, COMPENSATION_CUTOFF).unwrap(),
            couplings(&ctx, &shapes, &after, COMPENSATION_CUTOFF).unwrap(),
        );
        let du = relative(c1.u.as_slice(), c0.u.as_slice());
        let dion = relative(c1.u_ion.as_slice(), c0.u_ion.as_slice());
        println!(
            "global translation: dU {du:.2e}, dU_ion {dion:.2e}, dE_ion {:.2e}",
            (c1.e_ion - c0.e_ion).abs()
        );
        assert!(du < 1e-12 && dion < 1e-12 && (c1.e_ion - c0.e_ion).abs() < 1e-12);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_on_site_tail_completes_the_cutoff() {
    let si = Silicon::load();
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let sites = sites(&si);
    {
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let n = si.channels.len();
        let wide = couplings(&ctx, &shapes, &sites, 15.).unwrap();
        let narrow = couplings(&ctx, &shapes, &sites, 8.).unwrap();
        // With the tail beyond each cutoff the two sums agree far better than
        // without it: the tail is what the cutoff leaves out.
        let radius = si.atoms[0].compensation_radius;
        let mut checked = 0;
        for a in 0..n {
            let l = si.channels[a].1;
            let (t8, t15) = (
                shapes.tail(radius, l, 8.).unwrap(),
                shapes.tail(radius, l, 15.).unwrap(),
            );
            let with_tail = (wide.u.as_slice()[a * n + a] - narrow.u.as_slice()[a * n + a]).norm();
            // Remove the tails: what is left is the dense shell 8 < |G| <= 15 alone.
            let shell =
                (wide.u.as_slice()[a * n + a] - t15 - narrow.u.as_slice()[a * n + a] + t8).norm();
            println!("channel {a} (L = {l}): tail beyond 8 {t8:.4e}, beyond 15 {t15:.2e}; |U15 - U8| with tails {with_tail:.2e}, shell sum alone {shell:.2e}");
            // The shell alone misses the continuum tail by its whole size; with the
            // tail the cutoffs agree to the discreteness of the lattice sum.
            checked += 1;
            assert!(shell > 0.5 * t8, "channel {a}");
            assert!(with_tail < 0.25 * shell, "channel {a}");
        }
        assert_eq!(checked, n);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_small_memory_limit_blocks_the_sums_or_raises_budget_cleanly() {
    let si = Silicon::load();
    let sites = sites(&si);
    let big = Context::new(Some(1), 256 << 20).unwrap();
    let reference = {
        let shapes = Shapes::new(&big, &sites, 20.).unwrap();
        let c = couplings(&big, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
        let h = hats(&big, &si.grid, &shapes, &sites).unwrap();
        (
            c.u.as_slice().to_vec(),
            c.u_ion.as_slice().to_vec(),
            c.e_ion,
            h.as_slice().to_vec(),
        )
    };
    println!("256 MiB context: peak {} bytes", big.memory_peak());
    // 4 MiB: the transient budget is limit/16 = 256 KiB, a few hundred columns
    // per block, many blocks; the answer is the same.
    let small = Context::new(Some(1), 4 << 20).unwrap();
    {
        let shapes = Shapes::new(&small, &sites, 20.).unwrap();
        let c = couplings(&small, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
        let h = hats(&small, &si.grid, &shapes, &sites).unwrap();
        let du = relative(c.u.as_slice(), &reference.0);
        let dh = relative(h.as_slice(), &reference.3);
        println!(
            "4 MiB context: peak {} bytes, dU {du:.2e}, dhats {dh:.2e}, dE_ion {:.2e}",
            small.memory_peak(),
            (c.e_ion - reference.2).abs()
        );
        assert!(du < 1e-12 && dh < 1e-14 && (c.e_ion - reference.2).abs() < 1e-12);
        assert!(small.memory_peak() <= 4 << 20);
    }
    assert_eq!(small.memory_used(), 0);
    // A few kilobytes cannot hold even one Bessel row of the tables.
    let tiny = Context::new(Some(1), 4096).unwrap();
    {
        match Shapes::new(&tiny, &sites, 20.) {
            Err(Error::Budget { .. }) => {}
            Err(other) => panic!("expected Budget, got {other}"),
            Ok(_) => panic!("a 4 KiB context built the shape tables"),
        }
    }
    assert_eq!(tiny.memory_used(), 0);
    // Tables fit, the dense block does not: raises Budget, leaves nothing reserved.
    let narrow = Context::new(Some(1), 700 << 10).unwrap();
    {
        let outcome = Shapes::new(&narrow, &sites, 20.).and_then(|shapes| {
            let _hold = narrow.reserve(narrow.memory_limit() - narrow.memory_used() - 2000)?;
            couplings(&narrow, &shapes, &sites, COMPENSATION_CUTOFF).map(|_| ())
        });
        assert!(matches!(outcome, Err(Error::Budget { .. })), "{outcome:?}");
    }
    assert_eq!(narrow.memory_used(), 0);
}

/// Eight Al atoms in the 2 x 1 x 1 conventional fcc cell (132.9 A^3, the cell of
/// the plan's memory gate), 72 channels, the shipped Al table, grid_spacing = 0.25 A.
#[test]
fn eight_aluminum_atoms_stay_inside_the_transient_budget() {
    use mandacaru_core::periodic::setup::shapes::{Atom, Transient};
    use mandacaru_core::{
        math::SpatialGrid, paw::local_split::grid_spacing, units::BOHR_TO_ANGSTROM,
    };
    let shipped = Shipped::read("lda-sr", "Al");
    let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
    let a = 4.05 * to_bohr;
    let shape = [32usize, 16, 16];
    let step = [2. * a / 32., 0., 0., 0., a / 16., 0., 0., 0., a / 16.];
    let grid = SpatialGrid::new(shape, [0.; 3], step, true).unwrap();
    let lattice = nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(2. * a, a, a));
    let mut atoms = Vec::new();
    for cell in 0..2 {
        for site in [[0., 0., 0.], [0.5, 0.5, 0.], [0.5, 0., 0.5], [0., 0.5, 0.5]] {
            atoms.push(Atom {
                center: [(site[0] + cell as f64) * a, site[1] * a, site[2] * a],
                charge: shipped.charge,
                compensation_radius: shipped.compensation_radius,
            });
        }
    }
    let channels: Vec<(usize, usize, i32)> = (0..8)
        .flat_map(|atom| {
            (0..3usize).flat_map(move |l| (-(l as i32)..=l as i32).map(move |m| (atom, l, m)))
        })
        .collect();
    assert_eq!(channels.len(), 72);
    let sites = Sites {
        atoms: &atoms,
        channels: &channels,
        lattice,
        sigma: 1.4 * grid_spacing(&step),
    };
    let ctx = Context::new(Some(1), 1800 << 20).unwrap();
    let transient = Transient::of(&ctx);
    {
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let tables = ctx.memory_used();
        let c = couplings(&ctx, &shapes, &sites, COMPENSATION_CUTOFF).unwrap();
        let h = hats(&ctx, &grid, &shapes, &sites).unwrap();
        let outputs = ctx.memory_used() - tables;
        let transient_peak = ctx.memory_peak() - tables - outputs;
        println!(
            "Al8 (72 channels, {} grid points): tables {:.2} MiB, outputs (U, U_ion, hats) {:.2} MiB, context peak {:.2} MiB, transient above them {:.2} MiB of the {:.1} MiB budget; E_ion {:.6}",
            grid.index.points(),
            tables as f64 / 1048576.,
            outputs as f64 / 1048576.,
            ctx.memory_peak() as f64 / 1048576.,
            transient_peak as f64 / 1048576.,
            (transient.dense_bytes + transient.radial_bytes) as f64 / 1048576.,
            c.e_ion
        );
        assert_eq!(h.as_slice().len(), 72 * grid.index.points());
        assert!(transient.dense_bytes + transient.radial_bytes <= 20 << 20);
        assert!(transient_peak <= transient.dense_bytes + transient.radial_bytes);
    }
    assert_eq!(ctx.memory_used(), 0);
}
