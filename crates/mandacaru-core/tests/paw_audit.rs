//! Quantitative PAW formulation audits (`dev_plan/PAW_PLAN.md` §2):
//!
//! 1. Rigid-shift egg-box audit of the molecular grid `ws` cross-term (`paw/molecular/mod.rs`)
//!    across molecules (`H2`, `O2`, `H2O`, `CO`) and grid spacings (`0.50`, `0.35`, `0.25` Bohr),
//!    decomposing the total egg-box ripple, the compensation-pairs (`ws`) ripple at both
//!    reconverged and fixed densities, and the term-by-term analytic net force sum.
//! 2. Periodic beyond-multipole augmentation remainder evaluation (`periodic/setup/remainder.rs` +
//!    `onsite_remainder`) on converged periodic KS SCF crystals (`Ni` FCC, `Cu` FCC, `Fe` BCC,
//!    `Si` diamond), quantifying the omitted second-order one-center Hartree remainder and the
//!    reciprocal-space grid-vs-remainder cross term for `3d` / heavy shells where `r_g < r_c`.
mod paw_molecular_support;

use mandacaru_core::{
    math::{
        fft::Fft3,
        reciprocal::{alias_table, box_vector, reciprocal_cell},
        storage::MatrixSource,
    },
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    paw::{molecular::Arrays, onecenter::Basis as OneCenterBasis, record::PawDataset},
    periodic::{
        setup::{
            create_setup, parse_input_json,
            remainder::{onsite_remainder, projector_columns, Remainder, REMAINDER_TABLE_MAX},
            shapes::Shapes,
        },
        Crystal, Options as PeriodicOptions,
    },
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;
use paw_molecular_support::{analytic_shells, dataset};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

const TO_EV_A: f64 = 27.211386245988 / 0.529177210903;
const HA_TO_MEV: f64 = 27211.386245988;

fn paw_root() -> PathBuf {
    std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .expect("MANDACARU_PAW_PATH is unset")
}

fn build_molecule(
    ctx: &Context,
    set: &str,
    elements: &[&str],
    numbers: &[u8],
    positions: &[[f64; 3]],
    shells: &[usize],
    cell: f64,
    spacing: f64,
) -> MolecularIntegrals {
    let mut arrays = Arrays::new();
    let mut species = Vec::new();
    let mut atoms = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for element in elements {
        let index = seen.iter().position(|s| s == element).unwrap_or_else(|| {
            let data = dataset::read(ctx, set, element);
            species.push(dataset::species(ctx, &mut arrays, &data));
            seen.push(element);
            seen.len() - 1
        });
        atoms.push(index);
    }
    paw_molecular_support::molecule(
        ctx,
        numbers,
        positions,
        cell,
        spacing,
        (species, atoms, arrays),
        "raw",
        &analytic_shells(shells),
    )
}

fn scf_options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-10,
        max_iterations: 400,
        ..Default::default()
    }
}

fn solve_ks(integrals: &MolecularIntegrals, particles: [usize; 2]) -> MeanField {
    let state = integrals
        .kohn_sham(
            "pbe",
            particles,
            particles[0] != particles[1],
            &scf_options(),
        )
        .unwrap();
    assert!(state.converged, "molecular SCF did not converge");
    state
}

fn total_density(state: &MeanField) -> DMatrix<f64> {
    let m = state.densities[0].nrows();
    if state.densities.len() == 1 {
        state.densities[0].map(|z| 2. * z.re)
    } else {
        let mut out = DMatrix::zeros(m, m);
        for d in &state.densities {
            out += d.map(|z| z.re);
        }
        out
    }
}

fn pair_energy(integrals: &MolecularIntegrals, d_tot: &DMatrix<f64>) -> f64 {
    let paw = integrals
        .corrections
        .as_ref()
        .unwrap()
        .dft
        .as_ref()
        .unwrap();
    paw.pairs
        .factors
        .iter()
        .zip(&paw.pairs.weights)
        .map(|(factor, weight)| {
            let m = d_tot.component_mul(factor).sum();
            0.5 * weight * m * m
        })
        .sum()
}

fn translate_geometry(base: &MolecularIntegrals, delta: [f64; 3]) -> Geometry {
    let moved: Vec<[f64; 3]> = base
        .geometry
        .positions()
        .iter()
        .map(|p| [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]])
        .collect();
    Geometry::new(
        base.geometry.numbers().to_vec(),
        moved,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap()
}

fn span(values: &[f64]) -> f64 {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

fn net_by_term(
    integrals: &MolecularIntegrals,
    state: &MeanField,
) -> (Vec<(String, [f64; 3])>, [f64; 3]) {
    let g = integrals.gradient(state).unwrap();
    let natoms = integrals.geometry.numbers().len();
    let mut terms = Vec::new();
    for (name, buf) in &g.terms {
        let s = buf.as_slice();
        let mut net = [0.; 3];
        for a in 0..natoms {
            for k in 0..3 {
                net[k] += s[3 * a + k];
            }
        }
        terms.push((name.to_string(), net));
    }
    let mut total_net = [0.; 3];
    let tot = g.total.as_slice();
    for a in 0..natoms {
        for k in 0..3 {
            total_net[k] += tot[3 * a + k];
        }
    }
    (terms, total_net)
}

fn run_ws_audit_for_case(
    ctx: &Context,
    label: &str,
    elements: &[&str],
    numbers: &[u8],
    positions: &[[f64; 3]],
    shells: &[usize],
    particles: [usize; 2],
    cell: f64,
    spacings: &[f64],
) -> Vec<(f64, f64, f64)> {
    let offsets = [0.0, 0.25, 0.50, 0.75];
    let mut results = Vec::new();
    for &h in spacings {
        let base = build_molecule(ctx, "pbe-sr", elements, numbers, positions, shells, cell, h);
        let s0 = base.corrections.as_ref().unwrap().overlap.clone();
        let base_state = solve_ks(&base, particles);
        let d0 = total_density(&base_state);
        let ao_d0 = &base.lowdin * &d0 * &base.lowdin;

        let mut e_tot = Vec::new();
        let mut e_pairs_scf = Vec::new();
        let mut e_ws_fixed_d = Vec::new();
        let mut max_overlap_drift = 0.0f64;

        for &frac in &offsets {
            let shift = [frac * h, 0., 0.];
            let shifted_holder;
            let shifted: &MolecularIntegrals = if frac == 0. {
                &base
            } else {
                shifted_holder = base.at_geometry(translate_geometry(&base, shift)).unwrap();
                &shifted_holder
            };
            let s_diff = (&shifted.corrections.as_ref().unwrap().overlap - &s0)
                .abs()
                .max();
            max_overlap_drift = max_overlap_drift.max(s_diff);
            let state_holder;
            let state: &MeanField = if frac == 0. {
                &base_state
            } else {
                state_holder = solve_ks(shifted, particles);
                &state_holder
            };
            e_tot.push(state.free_energy() + shifted.nuclear);
            e_pairs_scf.push(pair_energy(shifted, &total_density(state)));
            let s_inv_half = shifted.lowdin.clone().try_inverse().unwrap();
            let d_fixed_ortho = &s_inv_half * &ao_d0 * &s_inv_half;
            e_ws_fixed_d.push(pair_energy(shifted, &d_fixed_ortho));
        }

        let tot_span = span(&e_tot);
        let pairs_span = span(&e_pairs_scf);
        let ws_span = span(&e_ws_fixed_d);

        // Gradient decomposition at frac = 0.25 (off grid symmetry) + FD check on ws slope.
        let probe_shift = [0.25 * h, 0., 0.];
        let probe = base
            .at_geometry(translate_geometry(&base, probe_shift))
            .unwrap();
        let probe_state = solve_ks(&probe, particles);
        let (terms, total_net) = net_by_term(&probe, &probe_state);
        let mut comp_center_x = 0.;
        let mut comp_basis_x = 0.;
        let mut ws_net_x = 0.;
        let mut mesh_net_x = 0.;
        let mut free_net_x = 0.;
        for (name, net) in &terms {
            if name == "pairs_coulomb/compensation_center" {
                comp_center_x += net[0];
                ws_net_x += net[0];
            } else if name == "pairs_coulomb/compensation_basis" {
                comp_basis_x += net[0];
                ws_net_x += net[0];
            } else if name.ends_with("/basis")
                || name.starts_with("hartree")
                || name.starts_with("xc")
                || name.starts_with("external")
            {
                mesh_net_x += net[0];
            } else {
                free_net_x += net[0];
            }
        }
        let ao_d = {
            let low = &probe.lowdin;
            low * total_density(&probe_state) * low
        };
        let fd_ws = |delta: f64| {
            let geom = translate_geometry(&base, [0.25 * h + delta, 0., 0.]);
            let shifted = base.at_geometry(geom).unwrap();
            let s_inv_half = shifted.lowdin.clone().try_inverse().unwrap();
            let d_ortho = &s_inv_half * &ao_d * &s_inv_half;
            pair_energy(&shifted, &d_ortho)
        };
        let step = 1e-3;
        let c1 = (fd_ws(step) - fd_ws(-step)) / (2. * step);
        let c2 = (fd_ws(0.5 * step) - fd_ws(-0.5 * step)) / step;
        let d_ws_dt = (4. * c2 - c1) / 3.;
        let ws_force_err = (ws_net_x - d_ws_dt).abs() * TO_EV_A;

        println!(
            "{:<14} h={h:.2} Bohr | dE_tot={:.3e} Ha ({:.3e} meV) | dE_pairs={:.3e} Ha | dE_ws(fixed-D)={:.3e} Ha ({:.3e} meV) | |dS|={max_overlap_drift:.1e} | netF_x[tot={:.2e}, ws={:.2e} (ctr={:.2e}, bas={:.2e}), d_ws_dt={:.2e}, mesh={:.2e}, free={:.1e}] Ha/B | ws_slope_err={ws_force_err:.2e} eV/A",
            label,
            tot_span,
            tot_span * HA_TO_MEV,
            pairs_span,
            ws_span,
            ws_span * HA_TO_MEV,
            total_net[0],
            ws_net_x,
            comp_center_x,
            comp_basis_x,
            d_ws_dt,
            mesh_net_x,
            free_net_x,
        );

        assert!(
            max_overlap_drift < 1e-12,
            "overlap shifted by {max_overlap_drift:.2e}"
        );
        assert!(
            free_net_x.abs() < 1e-10,
            "{label} h={h}: mesh-free net force {free_net_x:.2e}"
        );
        assert!(
            ws_force_err < 1e-5,
            "{label} h={h}: ws force vs +dE_ws/dt mismatch {ws_force_err:.2e} eV/A"
        );
        results.push((h, ws_span, tot_span));
    }
    results
}

#[test]
fn rigid_shift_egg_box_audit_of_molecular_ws_cross_term() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let h2 = run_ws_audit_for_case(
        &ctx,
        "H2",
        &["H", "H"],
        &[1, 1],
        &[[0.07, 0.11, -0.71], [-0.03, -0.05, 0.72]],
        &[0, 1],
        [1, 1],
        12.,
        &[0.50, 0.35, 0.25],
    );
    assert!(h2[2].1 < h2[0].1 * 0.05 && h2[2].1 < 2e-5, "H2: {h2:?}");
    let h2o = run_ws_audit_for_case(
        &ctx,
        "H2O",
        &["O", "H", "H"],
        &[8, 1, 1],
        &[
            [0.11, 0.07, 0.12],
            [1.55, 0.04, -0.93],
            [-1.43, -0.05, -0.97],
        ],
        &[0, 1, 2],
        [4, 4],
        12.,
        &[0.50, 0.35],
    );
    assert!(
        h2o[1].1 < h2o[0].1 && h2o[0].1 < 5e-3,
        "H2O ws ripple did not shrink: {h2o:?}"
    );
}

#[test]
#[ignore = "slow (~180 s): full O2, CO, and H2O h=0.25 ws egg-box spacing scans"]
fn rigid_shift_egg_box_audit_o2_and_co() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let spacings = [0.50, 0.35, 0.25];
    let h2o = run_ws_audit_for_case(
        &ctx,
        "H2O",
        &["O", "H", "H"],
        &[8, 1, 1],
        &[
            [0.11, 0.07, 0.12],
            [1.55, 0.04, -0.93],
            [-1.43, -0.05, -0.97],
        ],
        &[0, 1, 2],
        [4, 4],
        12.,
        &spacings,
    );
    assert!(
        h2o[2].1 < h2o[1].1 && h2o[1].1 < h2o[0].1 && h2o[2].1 < h2o[0].1 * 0.2 && h2o[2].1 < 5e-4,
        "H2O: {h2o:?}"
    );
    let o2 = run_ws_audit_for_case(
        &ctx,
        "O2(3Sigma_g-)",
        &["O", "O"],
        &[8, 8],
        &[[0.04, -0.05, -1.15], [0.01, 0.03, 1.14]],
        &[0, 1, 2],
        [7, 5],
        12.,
        &spacings,
    );
    assert!(o2[2].1 < o2[0].1 * 0.05 && o2[2].1 < 5e-4, "O2: {o2:?}");
    let co = run_ws_audit_for_case(
        &ctx,
        "CO",
        &["C", "O"],
        &[6, 8],
        &[[0.05, -0.07, -1.06], [-0.02, 0.04, 1.07]],
        &[0, 1, 2],
        [5, 5],
        12.,
        &spacings,
    );
    assert!(co[2].1 < co[0].1 * 0.05 && co[2].1 < 1e-4, "CO: {co:?}");
}

fn load_dataset(ctx: &Context, set: &str, symbol: &str) -> Arc<PawDataset> {
    let path = paw_root().join(format!("{set}/{symbol}.parquet"));
    Arc::new(PawDataset::read(ctx, &path).unwrap())
}

fn periodic_reference_density_raw(
    ds: &PawDataset,
    projectors: &[mandacaru_core::paw::onecenter::Projector],
) -> DMatrix<f64> {
    let n = projectors.len();
    let mut ref_mat = DMatrix::zeros(n, n);
    // Group projectors by (l, m) and compute B^T * diag(occ / (2l+1)) * B.
    let mut groups: BTreeMap<(usize, i32), Vec<(usize, usize)>> = BTreeMap::new();
    for (col, p) in projectors.iter().enumerate() {
        groups.entry((p.l, p.m)).or_default().push((col, p.index));
    }
    for ((l, _m), cols) in groups {
        let ch = &ds.channels[&(l as u32)];
        let w = ch.ae_waves.len();
        let b = DMatrix::from_row_slice(w, w, ch.vanderbilt.data());
        let diag = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            w,
            ch.occupations
                .data()
                .iter()
                .map(|&o| o / (2 * l + 1) as f64),
        ));
        let block = b.transpose() * diag * &b;
        for &(c1, i1) in &cols {
            for &(c2, i2) in &cols {
                ref_mat[(c1, c2)] = block[(i1, i2)];
            }
        }
    }
    ref_mat
}

fn reciprocal_lattice_sum(
    c: &Crystal,
    rho_smooth_r: &[f64],
    q: &[C64],
    d_proj_atoms: &[Vec<DMatrix<C64>>],
    datasets: &[Arc<PawDataset>],
    atoms_species: &[usize],
    centers: &[[f64; 3]],
    projectors_all: &[mandacaru_core::periodic::setup::spheres::Projector],
) -> (f64, f64, f64, f64, f64, usize, usize, f64) {
    let ctx = &c.context;
    let n = c.grid.index.points();
    let volume = n as f64 * c.grid.index.dv();
    let mut fft = Fft3::new(ctx, c.grid.index.shape()).unwrap();

    // Total smooth electron density on the real-space grid (sum over spin channels).
    let mut rho_tot_r = vec![0.0; n];
    for s in 0..c.spins {
        for k in 0..n {
            rho_tot_r[k] += rho_smooth_r[s * n + k];
        }
    }
    // Forward FFT with phase factor * dv matching periodic::solver::reciprocal.
    let mut rho_g_buf = vec![C64::default(); n];
    for (dst, &val) in rho_g_buf.iter_mut().zip(&rho_tot_r) {
        *dst = C64::new(val, 0.);
    }
    fft.transform(&mut rho_g_buf, false).unwrap();
    for (k, v) in rho_g_buf.iter_mut().enumerate() {
        *v *= c.phase.as_slice()[k] * c.grid.index.dv();
    }

    // Smooth + hats + ion in G-space.
    let mut hats_g = vec![C64::default(); n];
    let mut staged = vec![C64::default(); n];
    for (a, &charge) in q.iter().enumerate().take(c.channels) {
        let slice = c.hats.slice(a * n, &mut staged).unwrap();
        for k in 0..n {
            hats_g[k] += charge * slice[k];
        }
    }

    let b = reciprocal_cell(&c.grid).unwrap();
    let shape = c.grid.index.shape();
    let lattice = {
        let s = c.grid.step;
        let sh = shape;
        nalgebra::Matrix3::from_columns(&[
            s.column(0) * sh[0] as f64,
            s.column(1) * sh[1] as f64,
            s.column(2) * sh[2] as f64,
        ])
    };
    let sites_atoms: Vec<mandacaru_core::periodic::setup::shapes::Atom> = centers
        .iter()
        .enumerate()
        .map(
            |(a, &center)| mandacaru_core::periodic::setup::shapes::Atom {
                center,
                charge: datasets[atoms_species[a]].valence_charge,
                compensation_radius: datasets[atoms_species[a]].compensation_radius,
            },
        )
        .collect();
    let mut channels = Vec::new();
    for (atom, &s) in atoms_species.iter().enumerate() {
        let lmax = datasets[s].channels.keys().copied().max().unwrap_or(0) as usize;
        for big_l in 0..=2 * lmax {
            for m in -(big_l as i32)..=big_l as i32 {
                channels.push((atom, big_l, m));
            }
        }
    }
    let sites = mandacaru_core::periodic::setup::shapes::Sites {
        atoms: &sites_atoms,
        channels: &channels,
        lattice,
        sigma: 0.4,
    };
    let shapes = Shapes::new(ctx, &sites, REMAINDER_TABLE_MAX).unwrap();

    // Box wavevectors and alias cutoff check.
    let box_g: Vec<[f64; 3]> = (0..n).map(|k| box_vector(&b, shape, k)).collect();
    let norms: Vec<f64> = box_g
        .iter()
        .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt())
        .collect();
    let max_g = norms.iter().copied().fold(0.0f64, f64::max);
    let _aliases = alias_table(ctx, &c.grid).unwrap();
    let keep_idx: Vec<usize> = norms
        .iter()
        .enumerate()
        .filter(|(_, norm)| **norm <= REMAINDER_TABLE_MAX)
        .map(|(k, _)| k)
        .collect();
    let keep_g: Vec<[f64; 3]> = keep_idx.iter().map(|&k| box_g[k]).collect();

    let owners: Vec<usize> = projectors_all.iter().map(|p| p.atom).collect();
    let mut rem_g_tot = vec![C64::default(); keep_g.len()];
    let mut rem_g_delta = vec![C64::default(); keep_g.len()];
    let mut per_atom_rem_tot = Vec::with_capacity(atoms_species.len());

    for (atom, &s) in atoms_species.iter().enumerate() {
        let cols = projector_columns(&owners, atom);
        let oc_projs: Vec<mandacaru_core::paw::onecenter::Projector> = cols
            .iter()
            .map(|&col| mandacaru_core::paw::onecenter::Projector {
                l: projectors_all[col].l,
                m: projectors_all[col].m,
                index: projectors_all[col].index,
                real: false,
            })
            .collect();
        let rem = Remainder::new(
            ctx,
            &datasets[s],
            &shapes,
            OneCenterBasis::Raw,
            &oc_projs,
            centers[atom],
        )
        .unwrap();
        let m = oc_projs.len();
        let mut table = vec![C64::default(); m * m * keep_g.len()];
        rem.transforms(&keep_g, &mut table).unwrap();

        let d_tot = &d_proj_atoms[0][atom];
        let d_del = &d_proj_atoms[1][atom];
        let mut atom_rem_tot = vec![C64::default(); keep_g.len()];
        for i in 0..m {
            for j in 0..m {
                // Note solver.rs:1066 uses projected[(i, j)] * block[(j, i)].
                let w_tot = d_tot[(i, j)];
                let w_del = d_del[(i, j)];
                let slice = &table[(j * m + i) * keep_g.len()..][..keep_g.len()];
                for (idx, &eg) in slice.iter().enumerate() {
                    atom_rem_tot[idx] += w_tot * eg;
                    rem_g_tot[idx] += w_tot * eg;
                    rem_g_delta[idx] += w_del * eg;
                }
            }
        }
        per_atom_rem_tot.push(atom_rem_tot);
    }

    let mut e_cross_tot = 0.0;
    let mut e_cross_delta = 0.0;
    let mut e_cross_delta_elec = 0.0;
    let mut e_self_delta_g = 0.0;
    let mut e_inter_rem = 0.0;
    for (v, &k) in keep_idx.iter().enumerate() {
        let ker = c.kernel.as_slice()[k];
        let elec_source = rho_g_buf[k] + hats_g[k];
        let pot_source = elec_source + c.ion.as_slice()[k];
        e_cross_tot += ker * (pot_source.conj() * rem_g_tot[v]).re;
        e_cross_delta += ker * (pot_source.conj() * rem_g_delta[v]).re;
        e_cross_delta_elec += ker * (elec_source.conj() * rem_g_delta[v]).re;
        e_self_delta_g += 0.5 * ker * rem_g_delta[v].norm_sqr();
        if per_atom_rem_tot.len() > 1 {
            let sum_self: f64 = per_atom_rem_tot.iter().map(|a| a[v].norm_sqr()).sum();
            e_inter_rem += 0.5 * ker * (rem_g_tot[v].norm_sqr() - sum_self);
        }
    }
    e_cross_tot /= volume;
    e_cross_delta /= volume;
    e_cross_delta_elec /= volume;
    e_self_delta_g /= volume;
    e_inter_rem /= volume;

    (
        e_cross_tot,
        e_cross_delta,
        e_cross_delta_elec,
        e_self_delta_g,
        e_inter_rem,
        keep_idx.len(),
        n,
        max_g,
    )
}

struct RemainderMetrics {
    scf_energy: f64,
    onsite_2nd: f64,
    cross_delta_elec: f64,
    inter_rem: f64,
    self_delta_g: f64,
}

struct CrystalCaseSpec {
    label: &'static str,
    symbol: &'static str,
    lattice_bohr: [[f64; 3]; 3],
    atoms_frac: &'static [[f64; 3]],
    moments: &'static [f64],
    grid: [usize; 3],
    kmesh: [usize; 3],
    spin: bool,
}

fn run_periodic_remainder_cases(
    ctx: &Context,
    cases: &[CrystalCaseSpec],
) -> BTreeMap<&'static str, RemainderMetrics> {
    let mut by_label: BTreeMap<&'static str, RemainderMetrics> = BTreeMap::new();

    for case in cases {
        let ds = load_dataset(&ctx, "pbe-sr", case.symbol);
        let r_g = ds.compensation_radius;
        let r_c_max = ds.channels.values().map(|c| c.r_cut).fold(0.0f64, f64::max);

        let atoms_json: Vec<Value> = case
            .atoms_frac
            .iter()
            .zip(case.moments)
            .map(|(f, &m)| {
                json!({
                    "species": 0,
                    "symbol": case.symbol,
                    "fractional": f,
                    "moment": m,
                })
            })
            .collect();
        let flat_lat: Vec<f64> = case
            .lattice_bohr
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let raw = json!({
            "lattice": flat_lat,
            "grid_shape": case.grid,
            "atoms": atoms_json,
            "mesh": {"size": case.kmesh, "gamma": true},
            "functional": "pbe",
            "basis_size": "DZP",
            "symmetry": false,
            "spin": case.spin,
            "projector_basis": "raw",
        });
        let setup = create_setup(&ctx, parse_input_json(&raw).unwrap(), vec![ds.clone()]).unwrap();
        let projectors_all = setup.projectors.clone();
        let centers = setup.centers.clone();
        let atoms_species = setup.atoms_species.clone();
        let crystal = setup.build(&ctx).unwrap();

        let opts = PeriodicOptions {
            tolerance: 1e-8,
            max_iterations: 120,
            width: 0.005,
            ..Default::default()
        };
        let res = crystal.solve_with(&opts, None, None).unwrap();
        assert!(res.converged, "{} failed to converge", case.label);

        // Compute converged projector density matrix D_tot = sum_{s,k} w_k C_k^dagger D_{sk} C_k.
        let p = crystal.p;
        let nk = crystal.points.len();
        let mut projected = DMatrix::<C64>::zeros(p, p);
        for s in 0..crystal.spins {
            for (k, point) in crystal.points.iter().enumerate() {
                let d = res.matrices[s * nk + k].matrix(&ctx).unwrap();
                projected += point.projections.adjoint()
                    * &*d
                    * &point.projections
                    * C64::new(point.weight, 0.);
            }
        }

        let owners: Vec<usize> = projectors_all.iter().map(|pr| pr.atom).collect();
        let natoms = centers.len();
        let mut d_tot_atoms = Vec::with_capacity(natoms);
        let mut d_del_atoms = Vec::with_capacity(natoms);
        let mut e_onsite_2nd = 0.0;
        let mut e_onsite_full_minus_lin = 0.0;
        let mut max_delta_d = 0.0f64;

        for atom in 0..natoms {
            let cols = projector_columns(&owners, atom);
            let m = cols.len();
            let oc_projs: Vec<mandacaru_core::paw::onecenter::Projector> = cols
                .iter()
                .map(|&col| mandacaru_core::paw::onecenter::Projector {
                    l: projectors_all[col].l,
                    m: projectors_all[col].m,
                    index: projectors_all[col].index,
                    real: false,
                })
                .collect();
            let d_tot = DMatrix::from_fn(m, m, |i, j| projected[(cols[i], cols[j])]);
            let d_ref = periodic_reference_density_raw(&ds, &oc_projs);
            let d_del = DMatrix::from_fn(m, m, |i, j| d_tot[(i, j)] - C64::new(d_ref[(i, j)], 0.));
            for v in d_del.iter() {
                max_delta_d = max_delta_d.max(v.norm());
            }

            // Evaluate exact one-center Coulomb remainder DeltaW in Raw basis.
            let delta_w = onsite_remainder(&ctx, &ds, OneCenterBasis::Raw, &oc_projs).unwrap();
            let mut sum_2nd = C64::default();
            let mut sum_quad_tot = C64::default();
            let mut sum_quad_ref = C64::default();
            let mut sum_lin_ref = C64::default();
            for a in 0..m {
                for b in 0..m {
                    for c in 0..m {
                        for d in 0..m {
                            let w_abcd = delta_w.data[((a * m + b) * m + c) * m + d];
                            // (a,b) in delta_w pairs with d_tot[(b,a)] per pairs.rs:749.
                            let dab_del = d_del[(b, a)];
                            let dcd_del = d_del[(d, c)];
                            sum_2nd += dab_del * w_abcd * dcd_del;

                            let dab_t = d_tot[(b, a)];
                            let dcd_t = d_tot[(d, c)];
                            let dab_0 = C64::new(d_ref[(b, a)], 0.);
                            let dcd_0 = C64::new(d_ref[(d, c)], 0.);
                            sum_quad_tot += dab_t * w_abcd * dcd_t;
                            sum_quad_ref += dab_0 * w_abcd * dcd_0;
                            sum_lin_ref += dab_del * w_abcd * dcd_0;
                        }
                    }
                }
            }
            e_onsite_2nd += 0.5 * sum_2nd.re;
            // Exact quadratic DeltaW[D] minus linearized DeltaW[D0] + (D-D0) . grad DeltaW[D0].
            e_onsite_full_minus_lin +=
                0.5 * sum_quad_tot.re - (0.5 * sum_quad_ref.re + sum_lin_ref.re);

            d_tot_atoms.push(d_tot);
            d_del_atoms.push(d_del);
        }

        let (
            e_cross_tot,
            e_cross_delta,
            e_cross_delta_elec,
            e_self_delta_g,
            e_inter_rem,
            kept_g,
            total_g,
            max_g,
        ) = reciprocal_lattice_sum(
            &crystal,
            res.density.as_slice(),
            res.moments.as_slice(),
            &[d_tot_atoms, d_del_atoms],
            std::slice::from_ref(&ds),
            &atoms_species,
            &centers,
            &projectors_all,
        );

        println!(
            "{:<11} r_g={r_g:.3} r_c={r_c_max:.3} | max|dD|={max_delta_d:.3e} | \
             dE_onsite^(2)={:.3e} Ha/at ({:.2} meV/at) | dE_cross_elec(dD)={:.3e} Ha | dE_inter_rem={:.3e} Ha | dE_rem_self_G(dD)={:.3e} Ha | dE_cross(tot)={:.2e} | G<={REMAINDER_TABLE_MAX:.1}: {kept_g}/{total_g} (|G|_max={max_g:.2})",
            case.label,
            e_onsite_2nd / natoms as f64,
            (e_onsite_2nd / natoms as f64) * HA_TO_MEV,
            e_cross_delta_elec,
            e_inter_rem,
            e_self_delta_g,
            e_cross_tot,
        );
        let _ = e_cross_delta;
        by_label.insert(
            case.label,
            RemainderMetrics {
                scf_energy: res.free_energy,
                onsite_2nd: e_onsite_2nd / natoms as f64,
                cross_delta_elec: e_cross_delta_elec / natoms as f64,
                inter_rem: e_inter_rem / natoms as f64,
                self_delta_g: e_self_delta_g / natoms as f64,
            },
        );

        assert!(
            (e_onsite_2nd - e_onsite_full_minus_lin).abs() < 1e-5,
            "{}: onsite quadratic identity diff {:.2e}",
            case.label,
            (e_onsite_2nd - e_onsite_full_minus_lin).abs()
        );
        assert!(res.energy.is_finite());
    }
    by_label
}

#[test]
fn periodic_beyond_multipole_remainder_3d_metals_audit() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let a_cu = 3.615 / 0.529177210903;
    let a_fe = 2.866 / 0.529177210903;
    let a_si = 5.431 / 0.529177210903;
    let cases = [
        CrystalCaseSpec {
            label: "Cu_FCC(eq)",
            symbol: "Cu",
            lattice_bohr: [
                [0., 0.5 * a_cu, 0.5 * a_cu],
                [0.5 * a_cu, 0., 0.5 * a_cu],
                [0.5 * a_cu, 0.5 * a_cu, 0.],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[0.],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: false,
        },
        CrystalCaseSpec {
            label: "Fe_BCC(FM)",
            symbol: "Fe",
            lattice_bohr: [
                [-0.5 * a_fe, 0.5 * a_fe, 0.5 * a_fe],
                [0.5 * a_fe, -0.5 * a_fe, 0.5 * a_fe],
                [0.5 * a_fe, 0.5 * a_fe, -0.5 * a_fe],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[2.2],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: true,
        },
        CrystalCaseSpec {
            label: "Si_diamond",
            symbol: "Si",
            lattice_bohr: [
                [0., 0.5 * a_si, 0.5 * a_si],
                [0.5 * a_si, 0., 0.5 * a_si],
                [0.5 * a_si, 0.5 * a_si, 0.],
            ],
            atoms_frac: &[[0., 0., 0.], [0.25, 0.25, 0.25]],
            moments: &[0., 0.],
            grid: [14, 14, 14],
            kmesh: [4, 4, 4],
            spin: false,
        },
    ];
    let by_label = run_periodic_remainder_cases(&ctx, &cases);
    // Inter-atomic remainder is negligible when augmentation spheres do not overlap.
    assert!(by_label["Si_diamond"].inter_rem.abs() < 1e-5);
}

#[test]
#[ignore = "slow (~175 s): full 7-case Ni FM/NM and compressed Cu_2at(-10%) beyond-multipole remainder scan"]
fn periodic_beyond_multipole_remainder_full_scan() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let a_ni = 3.524 / 0.529177210903;
    let a_cu = 3.615 / 0.529177210903;
    let a_fe = 2.866 / 0.529177210903;
    let a_si = 5.431 / 0.529177210903;
    let cases = [
        CrystalCaseSpec {
            label: "Ni_FCC(FM)",
            symbol: "Ni",
            lattice_bohr: [
                [0., 0.5 * a_ni, 0.5 * a_ni],
                [0.5 * a_ni, 0., 0.5 * a_ni],
                [0.5 * a_ni, 0.5 * a_ni, 0.],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[0.6],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: true,
        },
        CrystalCaseSpec {
            label: "Ni_FCC(NM)",
            symbol: "Ni",
            lattice_bohr: [
                [0., 0.5 * a_ni, 0.5 * a_ni],
                [0.5 * a_ni, 0., 0.5 * a_ni],
                [0.5 * a_ni, 0.5 * a_ni, 0.],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[0.],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: false,
        },
        CrystalCaseSpec {
            label: "Cu_FCC(eq)",
            symbol: "Cu",
            lattice_bohr: [
                [0., 0.5 * a_cu, 0.5 * a_cu],
                [0.5 * a_cu, 0., 0.5 * a_cu],
                [0.5 * a_cu, 0.5 * a_cu, 0.],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[0.],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: false,
        },
        CrystalCaseSpec {
            label: "Cu_2at(eq)",
            symbol: "Cu",
            lattice_bohr: [
                [0., 0.5 * a_cu, 0.5 * a_cu],
                [a_cu, 0., 0.5 * a_cu],
                [a_cu, 0.5 * a_cu, 0.],
            ],
            atoms_frac: &[[0., 0., 0.], [0.5, 0., 0.]],
            moments: &[0., 0.],
            grid: [24, 12, 12],
            kmesh: [2, 4, 4],
            spin: false,
        },
        CrystalCaseSpec {
            label: "Cu_2at(-10%)",
            symbol: "Cu",
            lattice_bohr: [
                [0., 0.45 * a_cu, 0.45 * a_cu],
                [0.9 * a_cu, 0., 0.45 * a_cu],
                [0.9 * a_cu, 0.45 * a_cu, 0.],
            ],
            atoms_frac: &[[0., 0., 0.], [0.5, 0., 0.]],
            moments: &[0., 0.],
            grid: [24, 12, 12],
            kmesh: [2, 4, 4],
            spin: false,
        },
        CrystalCaseSpec {
            label: "Fe_BCC(FM)",
            symbol: "Fe",
            lattice_bohr: [
                [-0.5 * a_fe, 0.5 * a_fe, 0.5 * a_fe],
                [0.5 * a_fe, -0.5 * a_fe, 0.5 * a_fe],
                [0.5 * a_fe, 0.5 * a_fe, -0.5 * a_fe],
            ],
            atoms_frac: &[[0., 0., 0.]],
            moments: &[2.2],
            grid: [12, 12, 12],
            kmesh: [4, 4, 4],
            spin: true,
        },
        CrystalCaseSpec {
            label: "Si_diamond",
            symbol: "Si",
            lattice_bohr: [
                [0., 0.5 * a_si, 0.5 * a_si],
                [0.5 * a_si, 0., 0.5 * a_si],
                [0.5 * a_si, 0.5 * a_si, 0.],
            ],
            atoms_frac: &[[0., 0., 0.], [0.25, 0.25, 0.25]],
            moments: &[0., 0.],
            grid: [14, 14, 14],
            kmesh: [4, 4, 4],
            spin: false,
        },
    ];
    let by_label = run_periodic_remainder_cases(&ctx, &cases);
    let ni_fm = &by_label["Ni_FCC(FM)"];
    let ni_nm = &by_label["Ni_FCC(NM)"];
    println!(
        "Ni FM-NM diffs: dE_scf={:.3e} Ha ({:.2} meV), d(dE_onsite^2)={:.3e} Ha ({:.2} meV), d(dE_cross_elec)={:.3e} Ha ({:.2} meV), d(dE_self_G)={:.3e} Ha ({:.2} meV)",
        ni_fm.scf_energy - ni_nm.scf_energy,
        (ni_fm.scf_energy - ni_nm.scf_energy) * HA_TO_MEV,
        ni_fm.onsite_2nd - ni_nm.onsite_2nd,
        (ni_fm.onsite_2nd - ni_nm.onsite_2nd) * HA_TO_MEV,
        ni_fm.cross_delta_elec - ni_nm.cross_delta_elec,
        (ni_fm.cross_delta_elec - ni_nm.cross_delta_elec) * HA_TO_MEV,
        ni_fm.self_delta_g - ni_nm.self_delta_g,
        (ni_fm.self_delta_g - ni_nm.self_delta_g) * HA_TO_MEV,
    );
    let cu_eq = &by_label["Cu_2at(eq)"];
    let cu_cmp = &by_label["Cu_2at(-10%)"];
    println!(
        "Cu 2at (-10% - eq) diffs: dE_scf/at={:.3e} Ha, d(dE_onsite^2)={:.3e} Ha ({:.2} meV/at), dE_inter_rem(eq)={:.3e} Ha/at, dE_inter_rem(-10%)={:.3e} Ha/at",
        0.5 * (cu_cmp.scf_energy - cu_eq.scf_energy),
        cu_cmp.onsite_2nd - cu_eq.onsite_2nd,
        (cu_cmp.onsite_2nd - cu_eq.onsite_2nd) * HA_TO_MEV,
        cu_eq.inter_rem,
        cu_cmp.inter_rem,
    );
}
