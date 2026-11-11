//! The density gradient of the periodic meta-GGA against the product rule on the orbitals.
//!
//! A one-orbital density has `tau = tau_W = |grad rho|^2 / (8 rho)` at every point, so r2SCAN's
//! `alpha` vanishes. The crystal's kinetic density comes from the orbitals' spectral gradients,
//! its density gradient from the spectral derivative of the sampled density, whose band is twice
//! the orbitals' and so aliases on the orbital grid. This measures what the second choice costs
//! against the product rule `grad rho = 2 Re sum_ab D_ab (grad psi_a) psi_b^*` (the molecular
//! r2SCAN takes it since 2026-10-19): on H2 in a box (one orbital: the product-rule `tau_W` equals
//! `tau` exactly) and on H2O (a partial core, compact 2p orbitals), at Gamma, at two grids.

mod setup_shells_support;

use mandacaru_core::math::{fft::Fft3, storage::MatrixSource, SpatialGrid};
use mandacaru_core::models::xc;
use mandacaru_core::periodic::{
    fields::bloch_gradients,
    setup::parse::{create_setup, AtomSpec, SetupInput},
    solver::Options,
};
use mandacaru_core::quantum::C64;
use nalgebra::Matrix3;
use setup_shells_support::*;
use std::sync::Arc;

struct System {
    label: &'static str,
    species: &'static [&'static str],
    /// (species index, Cartesian position in Bohr from the box center).
    atoms: &'static [(usize, [f64; 3])],
    cell: [f64; 3],
}

const H2: System = System {
    label: "H2",
    species: &["H"],
    atoms: &[(0, [0., 0., -0.7]), (0, [0., 0., 0.7])],
    cell: [8., 8., 9.],
};
const WATER: System = System {
    label: "H2O",
    species: &["O", "H"],
    atoms: &[
        (0, [0., 0., 0.22]),
        (1, [0., 1.43, -0.89]),
        (1, [0., -1.43, -0.89]),
    ],
    cell: [9., 9., 9.],
};

fn input(system: &System, shape: [usize; 3], files: Vec<(String, String)>) -> SetupInput {
    let c = system.cell;
    SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: Matrix3::from_row_slice(&[c[0], 0., 0., 0., c[1], 0., 0., 0., c[2]]),
        grid_shape: shape,
        atoms: system
            .atoms
            .iter()
            .map(|&(species, r)| AtomSpec {
                species,
                symbol: system.species[species].into(),
                fractional: [0.5 + r[0] / c[0], 0.5 + r[1] / c[1], 0.5 + r[2] / c[2]],
                moment: 0.,
            })
            .collect(),
        kpoints_frac: vec![[0., 0., 0.]],
        weights: vec![1.],
        functional: "r2scan".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: false,
        symmetry: false,
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    }
}

/// `field` (periodic samples on `shape`) Fourier-interpolated onto twice the points per axis:
/// exact for a band-limited field, so the product of two such orbitals is resolved there without
/// aliasing. The coarse points are every second fine point.
fn refine(ctx: &mandacaru_core::Context, field: &[C64], shape: [usize; 3]) -> Vec<C64> {
    let fine = [2 * shape[0], 2 * shape[1], 2 * shape[2]];
    if field.iter().all(|v| *v == C64::default()) {
        return vec![C64::default(); fine[0] * fine[1] * fine[2]];
    }
    let mut coefficients = field.to_vec();
    Fft3::new(ctx, shape)
        .unwrap()
        .transform(&mut coefficients, false)
        .unwrap();
    let mut out = vec![C64::default(); fine[0] * fine[1] * fine[2]];
    let wrap = |i: usize, n: usize, m: usize| if 2 * i <= n { i } else { m + i - n };
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let target = (wrap(i, shape[0], fine[0]) * fine[1] + wrap(j, shape[1], fine[1]))
                    * fine[2]
                    + wrap(k, shape[2], fine[2]);
                out[target] = coefficients[(i * shape[1] + j) * shape[2] + k];
            }
        }
    }
    Fft3::new(ctx, fine)
        .unwrap()
        .transform(&mut out, true)
        .unwrap();
    // The FFT pair's normalization, from the largest coarse value and its fine twin.
    let (index, _) = field
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
        .unwrap();
    let (i, j, k) = (
        index / (shape[1] * shape[2]),
        (index / shape[2]) % shape[1],
        index % shape[2],
    );
    let twin = ((2 * i) * fine[1] + 2 * j) * fine[2] + 2 * k;
    let scale = field[index] / out[twin];
    for v in &mut out {
        *v *= scale;
    }
    out
}

/// The r2SCAN energy of `(rho, |grad rho|^2, tau)` samples (spin-unpolarized), Hartree.
fn r2scan_energy(rho: &[f64], sigma: &[f64], tau: &[f64], dv: f64) -> f64 {
    let mut energy = 0.;
    for k in 0..rho.len() {
        if rho[k] < 1e-12 {
            continue;
        }
        let (half, quarter) = (0.5 * rho[k], 0.25 * sigma[k]);
        energy += xc::point_partials(
            "r2scan",
            [
                half,
                half,
                quarter,
                quarter,
                quarter,
                0.5 * tau[k],
                0.5 * tau[k],
            ],
            false,
        )
        .unwrap()
        .0 * dv;
    }
    energy
}

/// The density, `|grad rho|^2` by the product rule and `tau` of `D` on orbital samples `psi`
/// (`m` rows) and their gradients, plus the core and its gradient and kinetic density.
fn fields(
    d: &nalgebra::DMatrix<C64>,
    psi: &[C64],
    gradients: &[C64],
    m: usize,
    n: usize,
    core: &[f64],
    core_gradient: &[C64],
    core_tau: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut rho, mut sigma, mut tau) = (vec![0.; n], vec![0.; n], vec![0.; n]);
    for k in 0..n {
        let mut g = [0f64; 3];
        for a in 0..m {
            for b in 0..m {
                let dab = d[(a, b)];
                rho[k] += (dab * psi[a * n + k] * psi[b * n + k].conj()).re;
                for axis in 0..3 {
                    let ga = gradients[(axis * m + a) * n + k];
                    let gb = gradients[(axis * m + b) * n + k];
                    g[axis] += 2. * (dab * ga * psi[b * n + k].conj()).re;
                    tau[k] += 0.5 * (dab * ga * gb.conj()).re;
                }
            }
        }
        rho[k] = (rho[k] + core[k]).max(0.);
        sigma[k] = (0..3)
            .map(|a| (g[a] + core_gradient[a * n + k].re).powi(2))
            .sum();
        // An interpolated core kinetic density can ring below zero where it vanishes.
        tau[k] = tau[k].max(0.) + core_tau[k].max(0.);
    }
    (rho, sigma, tau)
}

/// Prints and returns `(1 - int tau_W / int tau` with the spectral gradient where the density
/// is above 1e-3, the same with the product rule, `E_xc[spectral] - E_xc[product]` in Hartree)`.
fn measure(system: &System, shape: [usize; 3]) -> (f64, f64, f64) {
    use mandacaru_core::paw::digest::sha256_file;
    let ctx = context();
    let root =
        std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join("pbe-sr");
    let files = system
        .species
        .iter()
        .map(|s| {
            let path = root.join(format!("{s}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                sha256_file(&path).unwrap(),
            )
        })
        .collect();
    let datasets = system
        .species
        .iter()
        .map(|s| Arc::new(dataset(&ctx, "pbe-sr", s)))
        .collect();
    let crystal = create_setup(&ctx, input(system, shape, files), datasets)
        .unwrap()
        .build(&ctx)
        .unwrap();
    let options = Options {
        tolerance: 1e-10,
        density_tolerance: 1e-8,
        ..Options::default()
    };
    let result = crystal.solve(&options).unwrap();
    assert!(
        result.converged,
        "{}: the SCF did not converge",
        system.label
    );
    let (n, m, grid) = (crystal.grid.index.points(), crystal.m, &crystal.grid);
    let point = &crystal.points[0];
    let mut scratch = vec![C64::default(); m * n];
    let psi = point.psi.slice(0, &mut scratch).unwrap().to_vec();
    let gradients = bloch_gradients(&ctx, grid, point.k, &psi, m).unwrap();
    let d = result.matrices[0].matrix(&ctx).unwrap().clone();
    let density = &result.density.as_slice()[..n];
    let solver_tau = &result.kinetic_density.as_ref().unwrap().as_slice()[..n];
    // The valence density and kinetic density rebuilt from D and the samples (both spins of the
    // restricted channel): they must be the solver's, which fixes the conventions.
    let mut product = vec![[0f64; 3]; n];
    let (mut rho_miss, mut tau_miss) = (0f64, 0f64);
    for k in 0..n {
        let (mut rho, mut tau) = (0., 0.);
        for a in 0..m {
            for b in 0..m {
                let dab = d[(a, b)];
                rho += (dab * psi[a * n + k] * psi[b * n + k].conj()).re;
                for axis in 0..3 {
                    let ga = gradients.as_slice()[(axis * m + a) * n + k];
                    let gb = gradients.as_slice()[(axis * m + b) * n + k];
                    product[k][axis] += 2. * (dab * ga * psi[b * n + k].conj()).re;
                    tau += 0.5 * (dab * ga * gb.conj()).re;
                }
            }
        }
        rho_miss = rho_miss.max((rho - density[k]).abs());
        tau_miss = tau_miss.max((tau - solver_tau[k]).abs());
    }
    assert!(
        rho_miss < 1e-12 && tau_miss < 1e-12,
        "{}: conventions ({rho_miss:e}, {tau_miss:e})",
        system.label
    );
    // What the XC sees: valence plus the smooth partial core (its own spectral gradient on both
    // sides, the core being a sampled field).
    let core = crystal.core.as_slice();
    let total: Vec<C64> = (0..n).map(|k| C64::new(density[k] + core[k], 0.)).collect();
    let core_field: Vec<C64> = core.iter().map(|&v| C64::new(v, 0.)).collect();
    let spectral = bloch_gradients(&ctx, grid, [0.; 3], &total, 1).unwrap();
    let core_gradient = bloch_gradients(&ctx, grid, [0.; 3], &core_field, 1).unwrap();
    let core_tau = crystal
        .core_tau
        .as_ref()
        .map(|t| t.as_slice().to_vec())
        .unwrap_or(vec![0.; n]);
    let dv = grid.index.dv();
    let (mut tw_spectral, mut tw_product, mut t_sum) = (0., 0., 0.);
    let (mut e_spectral, mut e_product) = (0., 0.);
    for k in 0..n {
        let rho = (density[k] + core[k]).max(0.);
        if rho < 1e-12 {
            continue;
        }
        let s2: f64 = (0..3)
            .map(|a| spectral.as_slice()[a * n + k].re.powi(2))
            .sum();
        let p2: f64 = (0..3)
            .map(|a| (product[k][a] + core_gradient.as_slice()[a * n + k].re).powi(2))
            .sum();
        let t = solver_tau[k].max(0.) + core_tau[k];
        if rho > 1e-3 {
            tw_spectral += s2 / (8. * rho) * dv;
            tw_product += p2 / (8. * rho) * dv;
            t_sum += t * dv;
        }
        let xc = |sigma: f64| {
            let half = 0.5 * rho;
            let quarter = 0.25 * sigma;
            xc::point_partials(
                "r2scan",
                [half, half, quarter, quarter, quarter, 0.5 * t, 0.5 * t],
                false,
            )
            .unwrap()
            .0
        };
        e_spectral += xc(s2) * dv;
        e_product += xc(p2) * dv;
    }
    // The same orbitals, the same D, on the grid of twice the points: the continuum functional
    // of this state (the product of band-limited orbitals no longer aliases).
    let shape = grid.index.shape();
    let fine_shape = [2 * shape[0], 2 * shape[1], 2 * shape[2]];
    let fine_n = fine_shape.iter().product::<usize>();
    let mut step = grid.step_values();
    for v in &mut step {
        *v *= 0.5;
    }
    let fine_grid = SpatialGrid::new(fine_shape, grid.origin, step, true).unwrap();
    let fine_psi: Vec<C64> = (0..m)
        .flat_map(|a| refine(&ctx, &psi[a * n..(a + 1) * n], shape))
        .collect();
    let fine_gradients = bloch_gradients(&ctx, &fine_grid, point.k, &fine_psi, m).unwrap();
    let fine_core: Vec<f64> = refine(&ctx, &core_field, shape)
        .iter()
        .map(|v| v.re)
        .collect();
    let fine_core_field: Vec<C64> = fine_core.iter().map(|&v| C64::new(v, 0.)).collect();
    let fine_core_gradient =
        bloch_gradients(&ctx, &fine_grid, [0.; 3], &fine_core_field, 1).unwrap();
    let fine_core_tau: Vec<f64> = refine(
        &ctx,
        &core_tau
            .iter()
            .map(|&v| C64::new(v, 0.))
            .collect::<Vec<_>>(),
        shape,
    )
    .iter()
    .map(|v| v.re)
    .collect();
    let (rho_f, sigma_f, tau_f) = fields(
        &d,
        &fine_psi,
        fine_gradients.as_slice(),
        m,
        fine_n,
        &fine_core,
        fine_core_gradient.as_slice(),
        &fine_core_tau,
    );
    let charge = |rho: &[f64], dv: f64| rho.iter().sum::<f64>() * dv;
    let (rho_c, sigma_c, tau_c) = fields(
        &d,
        &psi,
        gradients.as_slice(),
        m,
        n,
        core,
        core_gradient.as_slice(),
        &core_tau,
    );
    let reference = r2scan_energy(&rho_f, &sigma_f, &tau_f, fine_grid.index.dv());
    let coarse_product = r2scan_energy(&rho_c, &sigma_c, &tau_c, dv);
    eprintln!(
        "{} {shape:?}: charge coarse {:.10} fine {:.10}; E_xc reference (2x grid) {reference:.8} Ha; coarse spectral {:+.3e}, coarse product {:+.3e} from it",
        system.label,
        charge(&rho_c, dv),
        charge(&rho_f, fine_grid.index.dv()),
        e_spectral - reference,
        coarse_product - reference
    );
    let (gap_spectral, gap_product) = (1. - tw_spectral / t_sum, 1. - tw_product / t_sum);
    let delta = e_spectral - e_product;
    eprintln!(
        "{} {shape:?} (h {:.3} Bohr): free energy {:.10} Ha; 1 - int tau_W / int tau: spectral {gap_spectral:+.3e}, product {gap_product:+.3e}; E_xc spectral - product {delta:+.3e} Ha",
        system.label,
        system.cell[0] / shape[0] as f64,
        result.free_energy
    );
    (gap_spectral, gap_product, delta)
}

#[test]
fn refinement_reproduces_a_band_limited_field() {
    let ctx = context();
    let shape = [6usize, 8, 10];
    let n: usize = shape.iter().product();
    let value = |x: f64, y: f64, z: f64| {
        let t = std::f64::consts::TAU;
        C64::new(
            (t * x).cos() + 0.3 * (2. * t * y).sin() + 0.1 * (t * (x + z)).cos(),
            0.2 * (t * z).sin(),
        )
    };
    let field: Vec<C64> = (0..n)
        .map(|i| {
            let (a, b, c) = (
                i / (shape[1] * shape[2]),
                (i / shape[2]) % shape[1],
                i % shape[2],
            );
            value(
                a as f64 / shape[0] as f64,
                b as f64 / shape[1] as f64,
                c as f64 / shape[2] as f64,
            )
        })
        .collect();
    let fine = refine(&ctx, &field, shape);
    let f = [2 * shape[0], 2 * shape[1], 2 * shape[2]];
    let miss = (0..fine.len())
        .map(|i| {
            let (a, b, c) = (i / (f[1] * f[2]), (i / f[2]) % f[1], i % f[2]);
            (fine[i]
                - value(
                    a as f64 / f[0] as f64,
                    b as f64 / f[1] as f64,
                    c as f64 / f[2] as f64,
                ))
            .norm()
        })
        .fold(0f64, f64::max);
    eprintln!(
        "refinement miss {miss:.2e}, |fine| {:.3e}",
        fine.iter().map(|v| v.norm_sqr()).sum::<f64>()
    );
    assert!(miss < 1e-12);
}

/// Measured 2026-11-10 (row 1b, `dev_plan/NOT_YET_AVAILABLE_LOG.md`): the spectral gradient keeps
/// the one-orbital identity of H2 to 1.1e-5 of the integrated tau at h 0.400 Bohr and 2.3e-7 at
/// 0.286, its r2SCAN energy 7.6e-6 and 2.1e-7 Ha from that of the same orbitals on a twice finer
/// grid. On water the spectral and product-rule energies differ by -3.1e-4 Ha (h 0.375) and
/// -9.2e-5 Ha (0.281), and neither is consistently nearer that reference (spectral -2.3e-4 /
/// +4.4e-5, product +8.2e-5 / +1.4e-4): a discretization difference below the grid error of the
/// free energy (5.8 mHa between the two grids), not the molecular defect of 2026-10-19 (a stencil
/// `tau_W` that made `alpha` of one orbital nonzero). The gates keep the one-orbital identity.
#[test]
#[ignore = "slow: 12 s (measured 2026-11-10); run: cargo test --profile release-test -p mandacaru-core --test periodic_meta_gradient -- --ignored --nocapture"]
fn the_spectral_density_gradient_keeps_the_one_orbital_identity() {
    for (shape, limit) in [([20, 20, 22], 2e-5), ([28, 28, 32], 1e-6)] {
        let (spectral, product, delta) = measure(&H2, shape);
        // One orbital: the product-rule tau_W is tau itself, and the spectral one stays near it.
        assert!(product.abs() < 1e-12);
        assert!(
            spectral.abs() < limit,
            "{shape:?}: 1 - int tau_W / int tau = {spectral:e}"
        );
        assert!(
            delta.abs() < 1e-5,
            "{shape:?}: E_xc spectral - product = {delta:e} Ha"
        );
    }
    for shape in [[24, 24, 24], [32, 32, 32]] {
        let (_, _, delta) = measure(&WATER, shape);
        assert!(
            delta.abs() < 5e-4,
            "{shape:?}: E_xc spectral - product = {delta:e} Ha"
        );
    }
}
