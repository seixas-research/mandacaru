//! The crystal stress on a REAL converged state of the skew LiH cell (LDA, 1 k-point): the frozen
//! energy rebuilt from the setup pieces at the unstrained cell equals the SCF energy
//! (`scf::state_energy`, the exactness gate: it fails if a piece of the oracle's energy, hence
//! of the stress, is missing), and the analytic stress built from the production output fields
//! (`scf::output_fields`, the re-diagonalized bands) equals the central difference, in all nine
//! entries, of the sum of the frozen pieces rebuilt at the strained cell. Needs
//! `MANDACARU_PAW_PATH`.
//!
//! The frozen pieces are those of the plan's section 9 rung 3: `D`, `W`, the potentials, the
//! moments and the node density are held fixed and the setup is rebuilt at `x -> (1 + eps) x`
//! with the pins held; a SCF reconverged on each strained cell (rungs 4-5) needs a strained
//! snapshot and is AS-6.
mod periodic_gradient_crystals_support;
mod periodic_stress_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_support;

use mandacaru_core::periodic::{
    gradient::scf,
    setup::{
        pulay::{kinetic_form, projection, Mesh},
        shells::{DenseRows, Sampling},
    },
    stress_terms::stress,
};
use num_complex::Complex64;
use periodic_stress_support::{
    gates::{gate, Tensor},
    lagrangian, oracle, xc, World,
};

const DENSE_GATE: f64 = 1e-7;
const SCF_TOLERANCE: f64 = 1e-13;
const SCF_DENSITY_TOLERANCE: f64 = 1e-11;
const STEP: f64 = 1e-5;
/// The step of the nonlinear difference: the kinks of the shape tables are crossed by a number of
/// frequencies that grows with the step, and nothing is held at zero there.
const FREE_STEP: f64 = 2.5e-6;
const FREE_STEPS: [f64; 4] = [2.5e-6, 5e-6, 1e-5, 2e-5];
const DENSE_STEP: f64 = 2e-5;
/// Rebuilt-versus-SCF energy: the quantity differs only by roundoff of two assemblies.
const ENERGY_GATE: f64 = 1e-10;

#[test]
fn the_stress_of_a_converged_lih_state() {
    let mut data = periodic_gradient_crystals_support::Data::load("lih");
    // The oracle tolerances of the freeze are 1e-11 Ha and 1e-9 in the density; the stress of a
    // loosely converged state is wrong at first order in the residual, and the nonlinear
    // difference below sees it (measured: a miss of 1.1e-7 independent of the step at 1e-9).
    data.options.tolerance = SCF_TOLERANCE;
    data.options.density_tolerance = SCF_DENSITY_TOLERANCE;
    let mut w = World::new("lih", &[0], "lda");
    // The dense sums are pinned to the integer triples no table node straddles under the strains
    // of the differences below (AS-0 pin 4). The SCF must extremize the SAME functional, or its
    // Hamiltonian is not the derivative of the energy that is differenced: the first run summed
    // the full ball in the SCF and the pinned list in F, and F was not stationary at the SCF
    // state (5.5e-8 Ha in an orbital rotation) and missed the stress by 1.1e-7 independently of
    // the step. So the crystal's U, U_ion and E_ion are rebuilt over the pinned list.
    let (list, vectors) = oracle::pinned_dense(&w, DENSE_STEP);
    {
        let frame = periodic_stress_support::Frame::new(&w, &[[0.; 3]; 3]);
        let sums = strain_support::pinned::sums(
            &w.data.shapes,
            &frame.sites(&w),
            &list,
            mandacaru_core::periodic::setup::shapes::COMPENSATION_CUTOFF,
            true,
        );
        data.crystal.u.as_mut_slice().copy_from_slice(&sums.u);
        data.crystal
            .uion
            .as_mut_slice()
            .copy_from_slice(&sums.u_ion);
        data.crystal.eion = sums.e_ion;
    }
    let result = data.solve();
    let state = data.state(&result);
    let crystal = &data.crystal;
    let ctx = &crystal.context;
    w.reference = crystal.reference;
    let (n, m) = (crystal.grid.index.points(), crystal.m);
    let degeneracy = crystal.degeneracy();
    // The input density matrix, its node density and channel moments, from the setup pieces.
    let bands = &state.bands[0][0];
    let density = bands.density();
    let mesh = Mesh::new(ctx, &crystal.grid).unwrap();
    let k = w.kpoints[0];
    let mut psi = vec![Complex64::new(0., 0.); m * n];
    data.shells
        .values(
            ctx,
            &Sampling {
                lattice: &data.lattice,
                region: mesh.region(),
                kpoints: &[k],
            },
            mesh.points(),
            &mut DenseRows::new(&mut psi, m, n),
        )
        .unwrap();
    let rho: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = Complex64::new(0., 0.);
            for mu in 0..m {
                for nu in 0..m {
                    s += psi[mu * n + i].conj() * density[(nu, mu)] * psi[nu * n + i];
                }
            }
            degeneracy * w.weights[0] * s.re
        })
        .collect();
    let c = projection(ctx, &data.shells, &data.lattice, k, &data.projectors).unwrap();
    let r = c.adjoint() * &density * &c;
    let q: Vec<Complex64> = data
        .blocks
        .iter()
        .map(|b| {
            let mut t = Complex64::new(0., 0.);
            for i in 0..b.nrows() {
                for j in 0..b.ncols() {
                    t += b[(i, j)] * r[(j, i)];
                }
            }
            t * (degeneracy * w.weights[0])
        })
        .collect();
    // Exactness: the pieces of the oracle at eps = 0 against the SCF energy of the same state.
    w.rho = rho;
    w.q = q.clone();
    w.bands = vec![bands.clone()];
    let zero = [[0.; 3]; 3];
    let kinetic = kinetic_form(ctx, &crystal.grid, &mesh, k, (&psi, m), (&psi, m)).unwrap();
    let short = periodic_gradient_crystals_support::short_range_total(&data, k);
    let fixed = kinetic + short + &c * &data.ion_operator * c.adjoint();
    let band: f64 = degeneracy
        * w.weights[0]
        * (0..m)
            .flat_map(|i| (0..m).map(move |j| (i, j)))
            .map(|(i, j)| (density[(i, j)] * fixed[(j, i)]).re)
            .sum::<f64>();
    let hartree = oracle::hartree_energy(&w, [&zero; 4]);
    let dense = oracle::dense_energy(&w, &zero, &list);
    let linear = oracle::linear_energy(&w, &zero);
    let constants = oracle::constants_energy(&w, &zero);
    let xc_energy = xc::evaluate(
        ctx,
        &crystal.grid,
        &xc::wave(ctx, &crystal.grid),
        crystal.grid.index.dv(),
        "lda",
        crystal.relativistic,
        &w.rho,
        &w.core(&crystal.grid, &w.centers()),
    )
    .energy;
    let pieces = band + hartree + dense + linear + constants + crystal.constant + xc_energy;
    println!(
        "pieces: band {band:.9}, hartree {hartree:.9}, dense {dense:.9}, linear {linear:.9}, ion constants {constants:.9}, constant {:.9}, xc {xc_energy:.9}",
        crystal.constant
    );
    println!("SCF terms {:?}", result.terms);
    {
        let frame = periodic_stress_support::Frame::new(&w, &[[0.; 3]; 3]);
        let sums = strain_support::pinned::sums(
            &data.shapes,
            &frame.sites(&w),
            &list,
            mandacaru_core::periodic::setup::shapes::COMPENSATION_CUTOFF,
            true,
        );
        let gap = |a: &[Complex64], b: &[Complex64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).norm())
                .fold(0., f64::max)
        };
        let (gu, gi) = (
            gap(&sums.u, crystal.u.as_slice()),
            gap(&sums.u_ion, crystal.uion.as_slice()),
        );
        println!(
            "u gap {gu:.2e}, uion gap {gi:.2e}, e_ion {:.12} vs snapshot {:.12}",
            sums.e_ion, crystal.eion
        );
        assert!(gu < 1e-11 && gi < 1e-11 && (sums.e_ion - crystal.eion).abs() < 1e-11);
        let c = w.q.len();
        let mut quad = Complex64::new(0., 0.);
        for a in 0..c {
            let uq: Complex64 = (0..c)
                .map(|b| crystal.u.as_slice()[a * c + b] * w.q[b])
                .sum();
            quad += 0.5 * w.q[a].conj() * uq + w.q[a].conj() * crystal.uion.as_slice()[a];
        }
        let lin: f64 = (0..c)
            .map(|a| (w.q[a] * crystal.linear.as_slice()[a]).re)
            .sum();
        println!(
            "snapshot dense {:.9}, linear {:.9}; mine dense {:.9}, linear {linear:.9}",
            quad.re + crystal.eion,
            lin,
            dense
        );
        // The Hartree self term from the snapshot's own arrays.
        let nn = crystal.grid.index.points();
        let rho_g = w.reciprocal(w.grid());
        let hats = crystal.hats.read_range(ctx, 0, c * nn).unwrap();
        let volume = nn as f64 * crystal.grid.index.dv();
        let mut e = 0.;
        for j in 0..nn {
            let charge: Complex64 = (0..c).map(|a| w.q[a] * hats[a * nn + j]).sum();
            e += crystal.kernel.as_slice()[j]
                * (0.5 * rho_g[j].norm_sqr()
                    + (rho_g[j].conj() * (charge + crystal.ion.as_slice()[j])).re);
        }
        println!(
            "snapshot-array Hartree {:.9}, mine {hartree:.9}",
            e / volume
        );
        assert!(
            (e / volume - hartree).abs() < 1e-11,
            "the Hartree piece is not the SCF's"
        );
    }
    let drho = w
        .rho
        .iter()
        .zip(result.density.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let dq = q
        .iter()
        .zip(result.moments.as_slice())
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    println!("node density against the SCF's: {drho:.2e}, moments: {dq:.2e}");
    assert!(
        drho < 1e-12 && dq < 1e-12,
        "the density or the moments are not the SCF's"
    );
    let densities = scf::densities(crystal, &state.bands).unwrap();
    let scf_energy = scf::state_energy(crystal, &densities).unwrap();
    println!("rebuilt energy {pieces:.12} Ha, SCF energy {scf_energy:.12} Ha");
    assert!(
        (pieces - scf_energy).abs() < ENERGY_GATE * scf_energy.abs().max(1.),
        "the frozen energy rebuilt from the setup pieces misses the SCF energy by {:.3e} Ha: a piece is missing",
        pieces - scf_energy
    );
    // The production fields of the output density and the bands re-diagonalized with them.
    let output = scf::output_fields(crystal, &densities, &q).unwrap();
    let rediagonalized = scf::rediagonalize(crystal, &output, &data.options, &[]).unwrap();
    w.bands = vec![rediagonalized[0][0].clone()];
    w.local = output.local.as_slice()[..n].to_vec();
    w.moments = output.moments.as_slice().to_vec();
    w.taint(STEP);
    let mut fields = w.fields();
    let tie = fields
        .xc
        .potential
        .iter()
        .zip(output.xc.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    println!("test-side xc potential against the production one: {tie:.2e}");
    assert!(
        tie < 1e-9,
        "the oracle's xc evaluation is not the production one: {tie:.2e}"
    );
    fields.xc.potential = output.xc.as_slice().to_vec();
    let got = w.with_setup(Some(vectors), |setup, frozen| {
        let frozen = mandacaru_core::periodic::gradient::Frozen {
            state: frozen.state.clone(),
            fields: fields.clone(),
            q: frozen.q.clone(),
            hubbard: None,
        };
        stress(setup, &frozen).unwrap()
    });
    // The oracle sum over every piece, one entry at a time.
    let mut total: Tensor = [[0.; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let matrix: f64 = oracle::central(a, b, STEP, |e| oracle::matrix_energies(&w, e, None))
                .iter()
                .sum();
            let hartree: f64 = (0..4)
                .map(|i| {
                    oracle::central(a, b, STEP, |e| {
                        let mut s = [&zero; 4];
                        s[i] = e;
                        [oracle::hartree_energy(&w, s)]
                    })[0]
                })
                .sum();
            let dense = oracle::central(a, b, DENSE_STEP, |e| {
                [oracle::dense_energy(&w, e, &list) + oracle::linear_energy(&w, e)]
            })[0];
            let constants = oracle::central(a, b, STEP, |e| [oracle::constants_energy(&w, e)])[0];
            let xc: f64 = oracle::central(a, b, STEP, |e| {
                oracle::xc_energies(&w, e, &fields.xc.potential)
            })
            .iter()
            .sum();
            total[a][b] = matrix + hartree + dense + constants + xc;
        }
    }
    gate(
        "converged LiH stress, sum of every term",
        &got.gradient,
        &total,
        DENSE_GATE,
    );
    println!(
        "  pressure {:.6e} Ha/Bohr^3, antisymmetric {:?}",
        got.pressure(),
        got.antisymmetric
    );

    // The nonlinear rung 3: the same state against F(eps) = E[S(eps)^{-1/2} y; eps] itself, with
    // the Lowdin carry, the density and the moments re-formed at each strained cell. No frequency
    // is held at zero here (the fields are the production ones), so the analytic route is rebuilt
    // without the mask.
    w.taint = vec![false; n];
    let plain = {
        let mut fields = w.fields();
        fields.xc.potential = output.xc.as_slice().to_vec();
        fields
    };
    let analytic = w.with_setup(Some(vectors), |setup, frozen| {
        let frozen = mandacaru_core::periodic::gradient::Frozen {
            state: frozen.state.clone(),
            fields: plain.clone(),
            q: frozen.q.clone(),
            hubbard: None,
        };
        stress(setup, &frozen).unwrap()
    });
    let y = lagrangian::carried_coefficients(&w);
    let f0 = lagrangian::energy(&mut w, &zero, &y, &list);
    println!(
        "F(0) {f0:.12} Ha against the SCF energy {scf_energy:.12} Ha ({:.2e})",
        f0 - scf_energy
    );
    assert!(
        (f0 - scf_energy).abs() < 1e-7,
        "the strained functional at eps = 0 is not the SCF energy"
    );
    // The central difference of the whole functional has a roundoff floor (the carried state is a
    // chain of an eigendecomposition and sums of 1 Ha, noise about 1e-15 over the step) that falls
    // with the step and a table-kink floor that grows with it; the sweep reports both and the
    // gate is the best step.
    // Stationarity of F in the orbitals: rotating an occupied orbital into a virtual one must not
    // change F at first order (the Lowdin-carry identity the analytic route rests on needs
    // H C = S C eps for the operator that is the exact derivative of this energy).
    {
        let occupied = w.bands[0].occupations.iter().filter(|f| **f > 1e-8).count();
        let t = 1e-5;
        let mut worst = 0f64;
        for i in 0..occupied {
            for a in occupied..(occupied + 4).min(y.ncols()) {
                let rotated = |angle: f64| {
                    let mut r = y.clone();
                    for row in 0..y.nrows() {
                        let (u, v) = (y[(row, i)], y[(row, a)]);
                        r[(row, i)] = u * angle.cos() + v * angle.sin();
                        r[(row, a)] = v * angle.cos() - u * angle.sin();
                    }
                    r
                };
                let (plus, minus) = (rotated(t), rotated(-t));
                let g = (lagrangian::energy(&mut w, &zero, &plus, &list)
                    - lagrangian::energy(&mut w, &zero, &minus, &list))
                    / (2. * t);
                worst = worst.max(g.abs());
            }
        }
        println!(
            "  stationarity: worst dF/d(rotation) {worst:.2e} Ha over {occupied} occupied orbitals"
        );
    }
    let mut best = f64::INFINITY;
    let mut chosen = [[0.; 3]; 3];
    for step in FREE_STEPS {
        let mut free = [[0.; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let side = |w: &mut World, delta: f64| {
                    lagrangian::energy(w, &periodic_stress_support::entry(a, b, delta), &y, &list)
                };
                let (plus, minus) = (side(&mut w, step), side(&mut w, -step));
                free[a][b] = (plus - minus) / (2. * step);
            }
        }
        let m = periodic_stress_support::gates::miss(&analytic.gradient, &free);
        println!(
            "  step {step:.1e}: miss {m:.2e} of the peak {:.3e}",
            periodic_stress_support::gates::peak(&free)
        );
        if m < best {
            best = m;
            chosen = free;
        }
    }
    gate(
        "converged LiH stress against the nonlinear F(eps), best step",
        &analytic.gradient,
        &chosen,
        DENSE_GATE,
    );
    println!(
        "  pressure of that functional {:.6e} Ha/Bohr^3",
        analytic.pressure()
    );
}
