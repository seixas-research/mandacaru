//! The Rust setup as a geometry provider (`periodic::setup::provider`): the matrix terms of the
//! crystal force on a frozen state equal the derivative of the frozen energy with every atom's
//! functions, projectors and quadrature sphere moved, the explicit pieces are the lattice-sum
//! derivatives of the setup, the layout is validated, and what the provider cannot build
//! (`fields`, `bands`) raises. Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;

use mandacaru_core::{
    math::spline::Cubic,
    paw::fields::place,
    periodic::{
        gradient::{pulay::contract, Bands, Derivatives, Frozen, Layout, Request, State, Term},
        setup::{
            electrostatics::max_wavenumber,
            gradients::{hat_overlaps, ion_constants_gradient, ion_overlaps},
            provider::{Core, Fields, Inputs, Operators, SetupDerivatives},
            pulay::{kinetic_form, mesh_dot, projection, Matrix},
            shapes::Shapes,
        },
    },
    quantum::C64,
    Context, Error,
};
use num_complex::Complex64;
use setup_gradients_support::context;
use setup_pulay_support::*;

const GATE: f64 = 1e-8;
const H: f64 = 2e-4;
const REFERENCE: f64 = 0.37;

/// Deterministic numbers in `[-1, 1]`.
fn seq(i: usize, salt: usize) -> f64 {
    (((i * 7919 + salt * 104729 + 13) as f64) * 0.618033988749895).sin()
}

fn hermitian(n: usize, salt: usize) -> Matrix {
    let a = Matrix::from_fn(n, n, |i, j| {
        Complex64::new(seq(i * n + j, salt), seq(i * n + j, salt + 50))
    });
    (&a + a.adjoint()) * Complex64::new(0.5, 0.)
}

struct World {
    fx: Fixture,
    kpoints: Vec<[f64; 3]>,
    weights: Vec<f64>,
    blocks: Vec<Matrix>,
    augmentation: Matrix,
    ion_operator: Matrix,
    bands: Vec<Bands>,
    local: Vec<f64>,
    moments: Vec<C64>,
}

impl World {
    fn new(ctx: &Context, name: &'static str) -> Self {
        let fx = Fixture::new(ctx, name);
        let (m, p) = (fx.functions(), fx.projectors.len());
        let channels = fx.case.cell().channels.len();
        let nb = 3;
        let bands = (0..2)
            .map(|k| Bands {
                coefficients: Matrix::from_fn(m, nb, |i, j| {
                    Complex64::new(seq(i * nb + j, 3 + k), seq(i * nb + j, 5 + k)) * 0.5
                }),
                eigenvalues: vec![REFERENCE - 0.3, REFERENCE + 0.1, REFERENCE + 0.4],
                occupations: vec![1.0, 0.8, 0.3],
            })
            .collect();
        let n = fx.grid.index.points();
        Self {
            kpoints: vec![KPOINT, [-0.09, 0.121, 0.2]],
            weights: vec![0.6, 0.4],
            blocks: (0..channels).map(|a| hermitian(p, 10 + a)).collect(),
            augmentation: hermitian(p, 30),
            ion_operator: hermitian(p, 31),
            bands,
            local: rough_potential(n),
            moments: (0..channels)
                .map(|a| Complex64::new(seq(a, 40) * 0.3, 0.))
                .collect(),
            fx,
        }
    }
    fn layout(&self) -> Layout {
        Layout {
            atoms: self.fx.atoms(),
            m: self.fx.functions(),
            p: self.fx.projectors.len(),
            spins: 1,
            degeneracy: 2.0,
            weights: self.weights.clone(),
            row_owner: self.fx.owner.clone(),
            projector_owner: self.fx.projectors.iter().map(|p| p.atom).collect(),
            reference: REFERENCE,
            request: Request {
                functional: "lda".to_string(),
                spinor: false,
                single_precision: false,
            },
        }
    }
    fn operators(&self) -> Operators {
        Operators {
            blocks: self.blocks.clone(),
            augmentation: self.augmentation.clone(),
            ion_operator: self.ion_operator.clone(),
        }
    }
    fn frozen(&self, fields: Fields) -> Frozen<Fields> {
        Frozen {
            state: State {
                bands: vec![self.bands.clone()],
            },
            fields,
            q: self.moments.clone(),
            hubbard: None,
        }
    }
}

/// The five matrix-term energies `sum_k deg w_k Re tr[...]` of the frozen state at the geometry
/// with `atom` moved by `d` along `axis`: overlap (`-W S`), kinetic, local, short-range and
/// projection (`D C H_C C^+ - W C q C^+`), written with the setup's forms and nothing of the
/// provider.
fn energies(ctx: &Context, w: &World, atom: usize, axis: usize, d: f64) -> [f64; 5] {
    let fx = &w.fx;
    let shells = fx.moved_basis(atom, axis, d);
    let projectors = fx.moved_projectors(atom, axis, d);
    let mut centers = fx.centers.clone();
    centers[atom][axis] += d;
    let mesh = fx.mesh(ctx);
    let n = mesh.len();
    let m = fx.functions();
    let mut out = [0.; 5];
    for (k, kpoint) in w.kpoints.iter().enumerate() {
        let weight = 2.0 * w.weights[k];
        let (density, weighted) = (w.bands[k].density(), w.bands[k].weighted(REFERENCE));
        let psi = bloch_values(ctx, fx, &shells, &mesh, *kpoint);
        let overlap = form(&psi, &psi, n, mesh.dv, None);
        let local = form(&psi, &psi, n, mesh.dv, Some(&w.local));
        let kinetic = kinetic_form(ctx, &fx.grid, &mesh, *kpoint, (&psi, m), (&psi, m)).unwrap();
        let short = short_range_total(fx, ctx, &shells, &centers, *kpoint);
        let c = projection(ctx, &shells, &fx.lattice, *kpoint, &projectors).unwrap();
        let mut operator = w.ion_operator.clone();
        for (block, q) in w.blocks.iter().zip(&w.moments) {
            operator += block * *q;
        }
        let cm = &c * &operator * c.adjoint();
        let cq = &c * &w.augmentation * c.adjoint();
        out[0] -= weight * trace(&weighted, &overlap);
        out[1] += weight * trace(&density, &kinetic);
        out[2] += weight * trace(&density, &local);
        out[3] += weight * trace(&density, &short);
        out[4] += weight * (trace(&density, &cm) - trace(&weighted, &cq));
    }
    out
}

fn fields_of(w: &World, mesh_len: usize) -> Fields {
    Fields::new(
        vec![w.local.clone()],
        vec![0.; mesh_len],
        vec![Complex64::new(0., 0.); mesh_len],
        w.moments.clone(),
    )
}

/// The pairs `(atom, axis)` whose frozen energy is differenced (each costs four rebuilds of every
/// matrix at both k-points): the contraction itself runs over all of them.
const SI_MOVES: [(usize, usize); 3] = [(0, 0), (1, 1), (1, 2)];
const AL2_MOVES: [(usize, usize); 3] = [(0, 2), (1, 0), (1, 2)];

#[test]
fn the_matrix_terms_of_the_force_equal_the_derivative_of_the_frozen_energy_si() {
    matrix_terms("si", &SI_MOVES);
}

#[test]
fn the_matrix_terms_of_the_force_equal_the_derivative_of_the_frozen_energy_al2() {
    matrix_terms("al2", &AL2_MOVES);
}

fn matrix_terms(name: &'static str, moves: &[(usize, usize)]) {
    let ctx = context();
    let w = World::new(&ctx, name);
    let cell = w.fx.case.cell();
    let sites = cell.sites();
    let q_max = max_wavenumber(&ctx, &w.fx.grid).unwrap();
    let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
    let no_cores = vec![None; w.fx.atoms()];
    let inputs = Inputs {
        grid: &w.fx.grid,
        shells: &w.fx.shells,
        projectors: &w.fx.projectors,
        kpoints: &w.kpoints,
        sites,
        shapes: &shapes,
        potentials: &w.fx.potentials,
        cores: &no_cores,
        dense_cutoff: mandacaru_core::periodic::setup::shapes::COMPENSATION_CUTOFF,
    };
    let provider = SetupDerivatives::new(&ctx, inputs, w.layout(), w.operators()).unwrap();
    let mesh_len = w.fx.grid.index.points();
    let frozen = w.frozen(fields_of(&w, mesh_len));
    let got = contract(&provider, &frozen).unwrap();
    let mut worst = [0f64; 5];
    for &(atom, axis) in moves {
        let mut table = [[0f64; 5]; 4];
        let mut step = |slot: usize, d: f64| table[slot] = energies(&ctx, &w, atom, axis, d);
        step(0, H);
        step(1, -H);
        step(2, 2. * H);
        step(3, -2. * H);
        let fd: Vec<f64> = (0..5)
            .map(|t| {
                let one = (table[0][t] - table[1][t]) / (2. * H);
                let two = (table[2][t] - table[3][t]) / (4. * H);
                (4. * one - two) / 3.
            })
            .collect();
        let mine = [
            got.overlap[atom][axis],
            got.kinetic[atom][axis],
            got.local[atom][axis],
            got.short_range[atom][axis],
            got.rows[atom][axis] + got.columns[atom][axis],
        ];
        for t in 0..5 {
            let scale = fd[t].abs().max(mine[t].abs()).max(1e-3);
            let error = (fd[t] - mine[t]).abs() / scale;
            worst[t] = worst[t].max(error);
            assert!(
                error < GATE,
                "{name} atom {atom} axis {axis} term {t}: {error:.2e} (fd {:.6e}, mine {:.6e})",
                fd[t],
                mine[t]
            );
            // Controls: every term is far from zero for some atom and axis, and the
            // opposite sign of any of them is not the answer.
            if fd[t].abs() > 1e-3 {
                assert!((fd[t] + mine[t]).abs() / scale > 100. * GATE);
            }
        }
        // The projector-moving share is not zero: dropping it misses the projection
        // energy.
        if fd[4].abs() > 1e-3 {
            let rows_only = got.rows[atom][axis];
            assert!((fd[4] - rows_only).abs() / fd[4].abs() > 100. * GATE);
        }
    }
    // Translation: the matrix terms sum to zero over the atoms, term by term (the rows and
    // the columns of the projections together).
    for t in 0..5 {
        for axis in 0..3 {
            let sum: f64 = (0..w.fx.atoms())
                .map(|a| match t {
                    0 => got.overlap[a][axis],
                    1 => got.kinetic[a][axis],
                    2 => got.local[a][axis],
                    3 => got.short_range[a][axis],
                    _ => got.rows[a][axis] + got.columns[a][axis],
                })
                .sum();
            // The mesh terms carry the egg-box ripple; they are reported, not gated at 1e-12.
            if t < 3 {
                println!("{name}: term {t} axis {axis} sum over atoms {sum:.2e}");
            } else {
                assert!(sum.abs() < 1e-9, "{name} term {t} axis {axis}: {sum:.2e}");
            }
        }
    }
    println!("{name}: matrix terms against the frozen-energy difference {worst:?}");
}

#[test]
fn the_explicit_pieces_are_the_lattice_sum_derivatives_of_the_setup() {
    let ctx = context();
    let w = World::new(&ctx, "si");
    let cell = w.fx.case.cell();
    let sites = cell.sites();
    let q_max = max_wavenumber(&ctx, &w.fx.grid).unwrap();
    let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
    // A smooth core on atom 0 only.
    let r: Vec<f64> = (0..400).map(|i| 0.01 + 0.015 * i as f64).collect();
    let values: Vec<f64> = r
        .iter()
        .map(|x| (1. + 0.3 * x) * (-x * x / 0.8).exp())
        .collect();
    let cores = vec![
        Some(Core {
            radius: &r,
            values: &values,
        }),
        None,
    ];
    let inputs = Inputs {
        grid: &w.fx.grid,
        shells: &w.fx.shells,
        projectors: &w.fx.projectors,
        kpoints: &w.kpoints,
        sites,
        shapes: &shapes,
        potentials: &w.fx.potentials,
        cores: &cores,
        dense_cutoff: mandacaru_core::periodic::setup::shapes::COMPENSATION_CUTOFF,
    };
    let provider = SetupDerivatives::new(&ctx, inputs, w.layout(), w.operators()).unwrap();
    let n = w.fx.grid.index.points();
    let weights: Vec<Complex64> = (0..n)
        .map(|j| Complex64::new(seq(j, 60), seq(j, 61)) * 0.1)
        .collect();
    let xc: Vec<f64> = (0..n).map(|j| seq(j, 62)).collect();
    let fields = Fields::new(
        vec![w.local.clone()],
        xc.clone(),
        weights.clone(),
        w.moments.clone(),
    );
    let hats = hat_overlaps(&ctx, &w.fx.grid, &shapes, &sites, &weights).unwrap();
    let ions = ion_overlaps(&ctx, &w.fx.grid, &sites, &weights).unwrap();
    let constants = ion_constants_gradient(&sites).unwrap();
    let channels = sites.channels.len();
    for atom in 0..w.fx.atoms() {
        for axis in 0..3 {
            let got = provider.hat_overlap(&fields, atom, axis).unwrap();
            assert_eq!(got.len(), channels);
            for (a, channel) in sites.channels.iter().enumerate() {
                let want = if channel.0 == atom {
                    hats[a][axis]
                } else {
                    Complex64::new(0., 0.)
                };
                assert_eq!(got[a], want);
            }
            assert_eq!(
                provider.ion_overlap(&fields, atom, axis).unwrap(),
                ions[atom][axis].re
            );
            assert_eq!(
                provider.constant(atom, axis).unwrap(),
                constants[atom][axis]
            );
            let dense = provider.dense(atom, axis).unwrap();
            assert_eq!(dense.u.shape(), (channels, channels));
            assert_eq!((dense.uion.len(), dense.linear.len()), (channels, channels));
            // E5: the core of atom 0 against the xc potential, by a central difference of the
            // placement; atom 1 has none.
            let core = provider.core_overlap(&fields, atom, axis).unwrap();
            if atom == 1 {
                assert_eq!(core, 0.);
                continue;
            }
            let profile = Cubic::new(
                &ctx,
                ctx.copy_slice(&r).unwrap(),
                ctx.copy_slice(&values).unwrap(),
            )
            .unwrap();
            let mesh = w.fx.mesh(&ctx);
            let energy = |d: f64| {
                let mut c = w.fx.centers[0];
                c[axis] += d;
                let rho = place(&ctx, &w.fx.grid, &profile, &[c]).unwrap();
                Matrix::from_element(
                    1,
                    1,
                    Complex64::new(mesh_dot(&mesh, &xc, rho.as_slice()).unwrap(), 0.),
                )
            };
            let fd = central(H, &mut |d| energy(d))[(0, 0)].re;
            assert!(
                (core - fd).abs() < GATE * fd.abs().max(1e-3),
                "axis {axis}: {core} vs {fd}"
            );
        }
    }
    // The terms with no meaning without a density raise instead of returning something.
    assert!(matches!(
        provider.fields(
            &State {
                bands: vec![w.bands.clone()]
            },
            &w.moments
        ),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        provider.bands(0, 0, &fields, None),
        Err(Error::Unsupported(_))
    ));
    // A wrong k-point or axis raises.
    assert!(provider.projection(2).is_err());
    assert!(provider.dense(0, 3).is_err());
    assert!(provider.cross(0, 0, Term::Overlap, &fields).is_ok());
}

#[test]
fn an_inconsistent_layout_is_refused() {
    let ctx = context();
    let w = World::new(&ctx, "si");
    let cell = w.fx.case.cell();
    let sites = cell.sites();
    let q_max = max_wavenumber(&ctx, &w.fx.grid).unwrap();
    let shapes = Shapes::new(&ctx, &sites, q_max).unwrap();
    let cores = vec![None; w.fx.atoms()];
    let build = |layout: Layout, operators: Operators| {
        let inputs = Inputs {
            grid: &w.fx.grid,
            shells: &w.fx.shells,
            projectors: &w.fx.projectors,
            kpoints: &w.kpoints,
            sites,
            shapes: &shapes,
            potentials: &w.fx.potentials,
            cores: &cores,
            dense_cutoff: mandacaru_core::periodic::setup::shapes::COMPENSATION_CUTOFF,
        };
        SetupDerivatives::new(&ctx, inputs, layout, operators)
    };
    assert!(build(w.layout(), w.operators()).is_ok());
    let mut layout = w.layout();
    layout.p += 1;
    assert!(build(layout, w.operators()).is_err());
    let mut layout = w.layout();
    layout.weights.pop();
    assert!(build(layout, w.operators()).is_err());
    let mut layout = w.layout();
    layout.projector_owner.reverse();
    assert!(build(layout, w.operators()).is_err());
    let mut operators = w.operators();
    operators.blocks.pop();
    assert!(build(w.layout(), operators).is_err());
}
