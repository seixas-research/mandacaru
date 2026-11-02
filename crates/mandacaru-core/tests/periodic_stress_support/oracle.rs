//! The oracles: every piece of the frozen energy rebuilt at the strained cell `x -> (1 + eps) x`
//! (lattice, atoms, grid nodes and origin, basis and projector centers, Cartesian k at fixed
//! fractional k) with the pins held (node counts, the Gaussian width, the shape tables and the
//! radial tables of the REFERENCE cell, the lattice operations, rigid quadrature spheres), the
//! frozen numbers `D`, `W`, `v_i`, `m_a`, `q_a`, `rho_i` held fixed. Written from the setup's
//! public builders and nothing of the strain code.
#![allow(dead_code)]

use super::{matrix3, xc, Eps, Frame, World};
use crate::{
    setup_pulay_support::{form, trace},
    strain_samples_support::Carried,
    strain_support::{pinned, straddles},
};
use mandacaru_core::periodic::setup::{
    basis_gradient::{short_range_cross, ShortRange},
    electrostatics::{hats, ion_charge, kernel},
    ions::{ion_constants, linear},
    pulay::{kinetic_form, projection, Matrix, Mesh},
    shapes::COMPENSATION_CUTOFF,
    shells::{DenseRows, Sampling},
};
use num_complex::Complex64;

/// `[overlap, kinetic, local, short_range, projection]` energies of the frozen state at the
/// strained cell. The local term keeps the REFERENCE `dv` (the node values `rho_i` are fixed: the
/// Jacobian of the charge is the Hartree term's); the overlap and the kinetic matrix carry their
/// own.
pub fn matrix_energies(w: &World, eps: &Eps, hubbard: Option<&Matrix>) -> [f64; 5] {
    let ctx = w.ctx();
    let data = &w.data;
    let frame = Frame::new(w, eps);
    let mesh = Mesh::new(ctx, &frame.grid).unwrap();
    let (n, m) = (mesh.len(), data.crystal.m);
    let dv0 = w.grid().index.dv();
    let mut operator = data.ion_operator.clone();
    for (block, q) in data.blocks.iter().zip(&w.moments) {
        operator += block * *q;
    }
    if let Some(o) = hubbard {
        operator += o;
    }
    let mut out = [0.; 5];
    for (i, k) in w.kpoints.iter().enumerate() {
        let carried = Carried::new(
            &data.lattice,
            &data.shells,
            &data.projectors,
            &matrix3(eps),
            *k,
        );
        let mut psi = vec![Complex64::new(0., 0.); m * n];
        carried
            .shells
            .values(
                ctx,
                &Sampling {
                    lattice: &carried.lattice,
                    region: mesh.region(),
                    kpoints: &[carried.k],
                },
                mesh.points(),
                &mut DenseRows::new(&mut psi, m, n),
            )
            .unwrap();
        let weight = 2.0 * w.weights[i];
        let (density, weighted) = (w.bands[i].density(), w.bands[i].weighted(w.reference));
        let overlap = form(&psi, &psi, n, mesh.dv, None);
        let local = form(&psi, &psi, n, dv0, Some(&w.local));
        let kinetic =
            kinetic_form(ctx, &frame.grid, &mesh, carried.k, (&psi, m), (&psi, m)).unwrap();
        let mut short = Matrix::zeros(m, m);
        for (a, atom) in frame.atoms.iter().enumerate() {
            let sphere = ShortRange {
                center: atom.center,
                potential: &data.potentials[a],
                sigma: data.sigma,
            };
            let block = short_range_cross(
                ctx,
                &carried.lattice,
                &[carried.k],
                &sphere,
                &carried.shells,
                None,
                &carried.shells,
            )
            .unwrap();
            short += Matrix::from_row_slice(m, m, block.as_slice());
        }
        let c = projection(
            ctx,
            &carried.shells,
            &carried.lattice,
            carried.k,
            &carried.projectors,
        )
        .unwrap();
        let cm = &c * &operator * c.adjoint();
        let cq = &c * &data.augmentation * c.adjoint();
        out[0] -= weight * trace(&weighted, &overlap);
        out[1] += weight * trace(&density, &kinetic);
        out[2] += weight * trace(&density, &local);
        out[3] += weight * trace(&density, &short);
        out[4] += weight * (trace(&density, &cm) - trace(&weighted, &cq));
    }
    out
}

/// `E_G` with each ingredient strained by its own `eps`: `[dv and Omega, kernel, hats, ions]`.
/// The node values `rho_i` and the moments `q` are fixed, `rho_G` is rebuilt on the grid of the
/// first strain (its own `dv` and phase).
pub fn hartree_energy(w: &World, strains: [&Eps; 4]) -> f64 {
    let ctx = w.ctx();
    let data = &w.data;
    let n = w.grid().index.points();
    let jacobian = Frame::new(w, strains[0]);
    let volume = n as f64 * jacobian.grid.index.dv();
    let rho = w.reciprocal(&jacobian.grid);
    let k = kernel(ctx, &Frame::new(w, strains[1]).grid).unwrap();
    let hat_frame = Frame::new(w, strains[2]);
    let table = hats(ctx, &hat_frame.grid, &data.shapes, &hat_frame.sites(w)).unwrap();
    let ion_frame = Frame::new(w, strains[3]);
    let ion = ion_charge(ctx, &ion_frame.grid, &ion_frame.sites(w)).unwrap();
    let mut energy = 0.;
    for j in 0..n {
        let compensation: Complex64 =
            w.q.iter()
                .enumerate()
                .map(|(a, q)| q * table.as_slice()[a * n + j])
                .sum();
        energy += k.as_slice()[j]
            * (0.5 * rho[j].norm_sqr() + (rho[j].conj() * (compensation + ion.as_slice()[j])).re);
    }
    energy / volume
}

/// `Re (q^+ U q / 2 + q^+ u_ion) + E_ion` over the pinned list at the strained cell (the on-site
/// tail included), the oracle of E-5.
pub fn dense_energy(w: &World, eps: &Eps, list: &[[i64; 3]]) -> f64 {
    let frame = Frame::new(w, eps);
    let sums = pinned::sums(
        &w.data.shapes,
        &frame.sites(w),
        list,
        COMPENSATION_CUTOFF,
        true,
    );
    let c = w.q.len();
    let mut total = Complex64::new(0., 0.);
    for a in 0..c {
        let uq: Complex64 = (0..c).map(|b| sums.u[a * c + b] * w.q[b]).sum();
        total += 0.5 * w.q[a].conj() * uq + w.q[a].conj() * sums.u_ion[a];
    }
    total.re + sums.e_ion
}

/// `Re (q . linear)`, `linear = short_range - onsite`, at the strained cell (the on-site
/// long-range integral is a function of the pinned radius, charge and width: it has no strain
/// derivative, but it is in the energy).
pub fn linear_energy(w: &World, eps: &Eps) -> f64 {
    let frame = Frame::new(w, eps);
    let linear = linear(w.ctx(), &frame.sites(w), &w.data.potentials).unwrap();
    w.q.iter().zip(&linear).map(|(q, l)| (q * l).re).sum()
}

/// The point-ion constant at the strained cell.
pub fn constants_energy(w: &World, eps: &Eps) -> f64 {
    ion_constants(&Frame::new(w, eps).sites(w)).unwrap()
}

/// `[X-1, X-3, X-4]`: `E_xc` of the fixed node values with only the `dv` strained, only the wave
/// vectors strained, and the linearization `dv sum vbar_i core_i` with only the core carried (the
/// energy itself would be differenced at the roundoff of its 1 Ha sum; `vbar` being the
/// functional derivative of `E_xc` is the separate check of `periodic_stress_xc.rs`).
pub fn xc_energies(w: &World, eps: &Eps, vbar: &[f64]) -> [f64; 3] {
    let ctx = w.ctx();
    let grid = w.grid();
    let frame = Frame::new(w, eps);
    let wave0 = xc::wave(ctx, grid);
    let core0 = w.core(grid, &w.centers());
    let run = |dv: f64, wave: &[f64]| {
        xc::evaluate(
            ctx,
            grid,
            wave,
            dv,
            w.functional,
            w.data.crystal.relativistic,
            &w.rho,
            &core0,
        )
        .energy
    };
    let core = w.core(&frame.grid, &frame.centers());
    [
        run(frame.grid.index.dv(), &wave0),
        run(grid.index.dv(), &xc::wave(ctx, &frame.grid)),
        grid.index.dv() * vbar.iter().zip(&core).map(|(v, c)| v * c).sum::<f64>(),
    ]
}

/// The central difference of a vector-valued oracle in the strain entry `(a, b)`.
pub fn central<const N: usize>(
    a: usize,
    b: usize,
    step: f64,
    mut f: impl FnMut(&Eps) -> [f64; N],
) -> [f64; N] {
    let plus = f(&super::entry(a, b, step));
    let minus = f(&super::entry(a, b, -step));
    std::array::from_fn(|i| (plus[i] - minus[i]) / (2. * step))
}

/// The kinetic symbol term of the stress, `deg sum_k w_k sum_G (-Q_a Q_b / Omega)(G) <u|D|u>(G)`,
/// formed with the alias MEAN of the products (`bloch_outer`, index 0) and with the product of
/// the mean vectors (`BlochSymbols::vectors`, index 1: the wrong symbol of the plan's control).
pub fn symbol_terms(w: &World) -> [[[f64; 3]; 3]; 2] {
    use mandacaru_core::math::{fft::Fft3, reciprocal};
    let ctx = w.ctx();
    let data = &w.data;
    let grid = w.grid();
    let mesh = Mesh::new(ctx, grid).unwrap();
    let (n, m) = (mesh.len(), data.crystal.m);
    let mut fft = Fft3::new(ctx, grid.index.shape()).unwrap();
    let mut out = [[[0.; 3]; 3]; 2];
    for (i, k) in w.kpoints.iter().enumerate() {
        let mut psi = vec![Complex64::new(0., 0.); m * n];
        data.shells
            .values(
                ctx,
                &Sampling {
                    lattice: &data.lattice,
                    region: mesh.region(),
                    kpoints: &[*k],
                },
                mesh.points(),
                &mut DenseRows::new(&mut psi, m, n),
            )
            .unwrap();
        let [x, y, z] = mesh.points();
        let mut u = vec![Complex64::new(0., 0.); m * n];
        for mu in 0..m {
            for j in 0..n {
                let angle = -(k[0] * x[j] + k[1] * y[j] + k[2] * z[j]);
                u[mu * n + j] = psi[mu * n + j] * Complex64::new(angle.cos(), angle.sin());
            }
            fft.transform(&mut u[mu * n..(mu + 1) * n], false).unwrap();
            for v in &mut u[mu * n..(mu + 1) * n] {
                *v *= mesh.dv;
            }
        }
        let d = w.bands[i].density();
        let expectation: Vec<f64> = (0..n)
            .map(|g| {
                let mut s = Complex64::new(0., 0.);
                for mu in 0..m {
                    for nu in 0..m {
                        s += d[(nu, mu)] * u[mu * n + g].conj() * u[nu * n + g];
                    }
                }
                s.re
            })
            .collect();
        let symbols = reciprocal::bloch(ctx, grid, *k).unwrap();
        let outer = reciprocal::bloch_outer(ctx, grid, *k).unwrap();
        let weight = 2.0 * w.weights[i];
        for a in 0..3 {
            for b in 0..3 {
                let row = super::voigt_row(a, b);
                let (mut right, mut wrong) = (0., 0.);
                for g in 0..n {
                    right += -outer.as_slice()[row * n + g] * expectation[g];
                    wrong += -symbols.vectors.as_slice()[a * n + g]
                        * symbols.vectors.as_slice()[b * n + g]
                        * expectation[g];
                }
                out[0][a][b] += weight * right / mesh.volume;
                out[1][a][b] += weight * wrong / mesh.volume;
            }
        }
    }
    out
}

/// The reference vectors of the pinned list that no strain entry `+-h` makes straddle a node of
/// the piecewise-linear shape tables (a whole-list difference has a kink floor far above the
/// gate; the analytic sum is fed the same list).
pub fn without_straddles(w: &World, list: &[[i64; 3]], step: f64) -> Vec<[i64; 3]> {
    let norm = |g: [f64; 3]| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let mut keep = vec![true; list.len()];
    for p in 0..3 {
        for q in 0..3 {
            let side = |delta: f64| {
                let frame = Frame::new(w, &super::entry(p, q, delta));
                pinned::vectors(&frame.sites(w), list)
            };
            let (plus, minus) = (side(step), side(-step));
            for (i, flag) in keep.iter_mut().enumerate() {
                *flag &= !straddles(norm(plus[i]), norm(minus[i]));
            }
        }
    }
    list.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(n, _)| *n)
        .collect()
}

/// The pinned dense list as Cartesian vectors at the reference cell, and its length against the ball's.
pub fn pinned_dense(w: &World, step: f64) -> (Vec<[i64; 3]>, Vec<[f64; 3]>) {
    let frame = Frame::new(w, &[[0.; 3]; 3]);
    let reference = pinned::list(&frame.sites(w), COMPENSATION_CUTOFF);
    let list = without_straddles(w, &reference, step);
    println!(
        "  dense list: {} of {} vectors kept",
        list.len(),
        reference.len()
    );
    let vectors = pinned::vectors(&frame.sites(w), &list);
    (list, vectors)
}

/// `d E_xc / d eps_ab` through the wave vectors of the spectral gradient alone, the node values
/// fixed: `dv sum_i (f_i(+h) - f_i(-h)) / 2h`, formed node by node.
pub fn xc_wave_derivative(w: &World, a: usize, b: usize, step: f64) -> f64 {
    let ctx = w.ctx();
    let grid = w.grid();
    let core0 = w.core(grid, &w.centers());
    let side = |delta: f64| {
        let frame = Frame::new(w, &super::entry(a, b, delta));
        xc::evaluate(
            ctx,
            grid,
            &xc::wave(ctx, &frame.grid),
            grid.index.dv(),
            w.functional,
            w.data.crystal.relativistic,
            &w.rho,
            &core0,
        )
        .nodes
    };
    let (plus, minus) = (side(step), side(-step));
    grid.index.dv() * plus.iter().zip(&minus).map(|(p, m)| p - m).sum::<f64>() / (2. * step)
}

/// `deg sum_k w_k Re tr[D T]` at the strained cell: the kinetic piece alone (the oracle of the
/// symbol control, which needs no sphere).
pub fn kinetic_energy(w: &World, eps: &Eps) -> f64 {
    let ctx = w.ctx();
    let data = &w.data;
    let frame = Frame::new(w, eps);
    let mesh = Mesh::new(ctx, &frame.grid).unwrap();
    let (n, m) = (mesh.len(), data.crystal.m);
    let mut total = 0.;
    for (i, k) in w.kpoints.iter().enumerate() {
        let carried = Carried::new(
            &data.lattice,
            &data.shells,
            &data.projectors,
            &matrix3(eps),
            *k,
        );
        let mut psi = vec![Complex64::new(0., 0.); m * n];
        carried
            .shells
            .values(
                ctx,
                &Sampling {
                    lattice: &carried.lattice,
                    region: mesh.region(),
                    kpoints: &[carried.k],
                },
                mesh.points(),
                &mut DenseRows::new(&mut psi, m, n),
            )
            .unwrap();
        let kinetic =
            kinetic_form(ctx, &frame.grid, &mesh, carried.k, (&psi, m), (&psi, m)).unwrap();
        total += 2.0 * w.weights[i] * trace(&w.bands[i].density(), &kinetic);
    }
    total
}
