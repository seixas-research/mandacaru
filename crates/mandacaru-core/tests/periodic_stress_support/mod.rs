//! Shared by the `periodic_stress*` tests: a frozen state built by hand on a frozen case (Si, Al2,
//! the skew LiH cell of `periodic_gradients`), the strain provider over its setup pieces, and the
//! oracles (`oracle`: every energy piece rebuilt at the strained cell with the pins held).
//!
//! The state is hand-built on purpose: two k-points (a third on request), fractional occupations,
//! a rough local potential, node densities with a gradient, channel moments of mixed phase. Every
//! piece of the frozen "Lagrangian" is then an independent fixed number, and the analytic strain
//! contraction is compared with the central difference of THAT function rebuilt at `x -> (1 + eps) x`.
#![allow(dead_code)]

pub mod gates;
pub mod lagrangian;
pub mod oracle;
pub mod xc;

use crate::periodic_gradient_crystals_support::Data;
use mandacaru_core::{
    math::{fft::Fft3, spline::Cubic, SpatialGrid},
    paw::fields::place,
    periodic::{
        gradient::{Bands, Frozen, Layout, Request, State},
        setup::{
            electrostatics::kernel,
            fields::phase,
            provider::Inputs,
            shapes::COMPENSATION_CUTOFF,
            strain_provider::{StrainFields, StrainSetup, XcFields},
        },
    },
    quantum::C64,
    Context,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;

pub type Eps = [[f64; 3]; 3];

/// The constant the solver adds to its eigenvalues (an arbitrary number: `W` must remove it).
pub const REFERENCE: f64 = 0.37;

/// An irrational k-point (Cartesian, Bohr^-1) inside the first zone of every fixture: its Bloch
/// phases are complex on every translation.
pub const GENERIC_K: [f64; 3] = [0.173205, -0.141421, 0.0866025];

/// The Gamma point.
pub const GAMMA: [f64; 3] = [0.; 3];

/// Deterministic numbers in `[-1, 1]`.
pub fn seq(i: usize, salt: usize) -> f64 {
    (((i * 7919 + salt * 104729 + 13) as f64) * 0.618033988749895).sin()
}

/// The Voigt row (xx, yy, zz, yz, xz, xy) of the symmetric pair `(a, b)`.
pub fn voigt_row(a: usize, b: usize) -> usize {
    [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        .iter()
        .position(|&(p, q)| (p, q) == (a, b) || (q, p) == (a, b))
        .unwrap()
}

/// One complex number per channel such that `sum_a x_a B_a` is Hermitian, as every physical set of
/// channel moments is: a channel whose block is Hermitian takes a real number, the others come in
/// pairs `B_b = s B_a^+` (`s = +-1`, the complex harmonics `M` and `-M`) and take `x_b = s conj(x_a)`.
/// The stress contraction (like the force engine's) assumes a Hermitian `D_ion + sum_a m_a B_a`.
pub fn paired(
    blocks: &[mandacaru_core::periodic::setup::pulay::Matrix],
    salt: usize,
    scale: f64,
) -> Vec<C64> {
    let c = blocks.len();
    let mut out = vec![Complex64::new(0., 0.); c];
    let mut done = vec![false; c];
    let tolerance = |a: usize| 1e-9 * blocks[a].norm().max(1.);
    for a in 0..c {
        if done[a] {
            continue;
        }
        done[a] = true;
        if (&blocks[a] - blocks[a].adjoint()).norm() < tolerance(a) {
            out[a] = Complex64::new(seq(a, salt) * scale, 0.);
            continue;
        }
        let partner = (0..c).filter(|&b| !done[b]).find_map(|b| {
            [1., -1.].into_iter().find_map(|s| {
                ((&blocks[b] - blocks[a].adjoint() * Complex64::new(s, 0.)).norm() < tolerance(a))
                    .then_some((b, s))
            })
        });
        let (b, s) = partner.expect("a non-Hermitian channel block has its adjoint partner");
        done[b] = true;
        out[a] = Complex64::new(seq(a, salt) * scale, seq(a, salt + 1) * scale);
        out[b] = out[a].conj() * s;
    }
    out
}

pub fn entry(a: usize, b: usize, e: f64) -> Eps {
    let mut eps = [[0.; 3]; 3];
    eps[a][b] = e;
    eps
}

pub fn matrix3(eps: &Eps) -> Matrix3<f64> {
    Matrix3::from_fn(|a, b| eps[a][b])
}

pub struct World {
    pub data: Data,
    pub functional: &'static str,
    /// The constant the eigenvalues of the bands are shifted by.
    pub reference: f64,
    pub kpoints: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub bands: Vec<Bands>,
    /// The potential the basis sees (a fixed rough field).
    pub local: Vec<f64>,
    /// The channel potentials `m_a`.
    pub moments: Vec<C64>,
    /// The channel charges `q_a` of the hats.
    pub q: Vec<C64>,
    /// The valence node densities.
    pub rho: Vec<f64>,
    /// The reciprocal nodes whose `rho_G` is held at zero: the frequencies a table node of the
    /// piecewise-linear shape tables straddles under a strain step (see [`World::taint`]).
    pub taint: Vec<bool>,
}

impl World {
    /// `case` at the k-points `ks` (indices into the snapshot's mesh), `functional` `lda` or `pbe`.
    pub fn new(case: &'static str, ks: &[usize], functional: &'static str) -> Self {
        let data = Data::load(case);
        let (m, n) = (data.crystal.m, data.crystal.grid.index.points());
        let nb = m.min(4);
        let share = 1. / ks.len() as f64;
        let bands = ks
            .iter()
            .enumerate()
            .map(|(i, _)| Bands {
                coefficients: nalgebra::DMatrix::from_fn(m, nb, |r, c| {
                    Complex64::new(seq(r * nb + c, 3 + i), seq(r * nb + c, 5 + i)) * 0.5
                }),
                eigenvalues: (0..nb).map(|j| REFERENCE - 0.3 + 0.2 * j as f64).collect(),
                occupations: [1.0, 0.8, 0.3, 0.1][..nb].to_vec(),
            })
            .collect();
        let q = paired(&data.blocks, 20, 0.4);
        let moments = paired(&data.blocks, 40, 0.3);
        let local = (0..n)
            .map(|j| 0.4 + (0.013 * j as f64).sin() + 0.5 * (0.0071 * j as f64).cos())
            .collect();
        let rho = (0..n)
            .map(|j| 0.03 * (1.8 + (0.013 * j as f64).sin() + 0.5 * (0.0071 * j as f64).cos()))
            .collect();
        Self {
            kpoints: ks.iter().map(|&k| data.kpoints[k]).collect(),
            weights: ks.iter().map(|_| share).collect(),
            bands,
            local,
            moments,
            q,
            rho,
            taint: vec![false; n],
            functional,
            reference: REFERENCE,
            data,
        }
    }

    /// The same state at other k-points (Cartesian, Bohr^-1): the setup pieces do not depend on
    /// the snapshot's mesh, only the bands' count must match.
    pub fn at(mut self, kpoints: &[[f64; 3]]) -> Self {
        assert_eq!(
            kpoints.len(),
            self.bands.len(),
            "one k-point per set of bands"
        );
        self.kpoints = kpoints.to_vec();
        self
    }

    pub fn ctx(&self) -> &Context {
        &self.data.crystal.context
    }

    pub fn grid(&self) -> &SpatialGrid {
        &self.data.crystal.grid
    }

    pub fn layout(&self) -> Layout {
        Layout {
            atoms: self.data.atoms.len(),
            m: self.data.crystal.m,
            p: self.data.crystal.p,
            spins: 1,
            degeneracy: 2.0,
            weights: self.weights.clone(),
            row_owner: self.data.owner.clone(),
            projector_owner: self.data.projectors.iter().map(|p| p.atom).collect(),
            reference: self.reference,
            request: Request {
                functional: self.functional.to_string(),
                spinor: false,
                single_precision: false,
            },
        }
    }

    /// The smooth core nodes of the atoms at `centers` on `grid` (the dataset tables, the
    /// placement the crystal builds its `core` with).
    pub fn core(&self, grid: &SpatialGrid, centers: &[[f64; 3]]) -> Vec<f64> {
        let ctx = self.ctx();
        let mut out = vec![0.; grid.index.points()];
        for (core, center) in self.data.cores().iter().zip(centers) {
            let Some(core) = core else { continue };
            let profile = Cubic::new(
                ctx,
                ctx.copy_slice(core.radius).unwrap(),
                ctx.copy_slice(core.values).unwrap(),
            )
            .unwrap();
            let placed = place(ctx, grid, &profile, &[*center]).unwrap();
            for (to, from) in out.iter_mut().zip(placed.as_slice()) {
                *to += from;
            }
        }
        out
    }

    pub fn centers(&self) -> Vec<[f64; 3]> {
        self.data.centers()
    }

    /// `rho_G = dv sum_i rho_i e^{-iG.r_i}` of the valence node values on `grid` (`phase` is the
    /// grid's own), the quantity the energy's Hartree term reads.
    pub fn reciprocal(&self, grid: &SpatialGrid) -> Vec<Complex64> {
        let ctx = self.ctx();
        let mut fft = Fft3::new(ctx, grid.index.shape()).unwrap();
        let mut work: Vec<Complex64> = self.rho.iter().map(|&v| Complex64::new(v, 0.)).collect();
        fft.transform(&mut work, false).unwrap();
        let phase = phase(ctx, grid).unwrap();
        let dv = grid.index.dv();
        for (v, p) in work.iter_mut().zip(phase.as_slice()) {
            *v *= p * dv;
        }
        for (v, held) in work.iter_mut().zip(&self.taint) {
            if *held {
                *v = Complex64::new(0., 0.);
            }
        }
        work
    }

    /// Holds `rho_G` at zero on every frequency whose box vector or any alias straddles a node of
    /// the shape tables (0.01 Bohr^-1 apart, the interpolant is piecewise linear) between the
    /// grids strained by `+step` and `-step` in any entry. A central difference of the hats over such
    /// a frequency carries the kink of the table, not the derivative the analytic term has: the
    /// functional of the remaining frequencies is the oracle and the analytic route alike.
    pub fn taint(&mut self, step: f64) {
        let mut flags = vec![false; self.rho.len()];
        for a in 0..3 {
            for b in 0..3 {
                let plus = Frame::new(self, &entry(a, b, step));
                let minus = Frame::new(self, &entry(a, b, -step));
                let held = crate::strain_support::tainted(self.ctx(), &plus.grid, &minus.grid);
                for (f, t) in flags.iter_mut().zip(held) {
                    *f |= t;
                }
            }
        }
        println!(
            "  {} of {} reciprocal nodes straddle a shape-table node",
            flags.iter().filter(|f| **f).count(),
            flags.len()
        );
        self.taint = flags;
    }

    /// The frozen fields at the unstrained cell.
    pub fn fields(&self) -> StrainFields {
        let ctx = self.ctx();
        let grid = self.grid();
        let n = grid.index.points();
        let density = self.reciprocal(grid);
        let k = kernel(ctx, grid).unwrap();
        let volume = n as f64 * grid.index.dv();
        let weights = density
            .iter()
            .zip(k.as_slice())
            .map(|(r, k)| r.conj() * (k / volume))
            .collect();
        let core = self.core(grid, &self.centers());
        let xc = xc::evaluate(
            ctx,
            grid,
            &xc::wave(ctx, grid),
            grid.index.dv(),
            self.functional,
            self.data.crystal.relativistic,
            &self.rho,
            &core,
        );
        StrainFields {
            local: vec![self.local.clone()],
            moments: self.moments.clone(),
            weights,
            density,
            xc: XcFields {
                energy: xc.energy,
                potential: xc.potential,
                gradients: xc.gradients,
                flux: xc.flux,
                nonlocal: None,
            },
            kinetic: None,
        }
    }

    pub fn frozen(&self, fields: StrainFields) -> Frozen<StrainFields> {
        Frozen {
            state: State {
                bands: vec![self.bands.clone()],
            },
            fields,
            q: self.q.clone(),
            hubbard: None,
        }
    }

    /// Runs `f` with the strain provider over the setup pieces of the case and the frozen state.
    /// `dense` pins the vectors of the dense sums (the ball of the reference cell without it).
    pub fn with_setup<R>(
        &self,
        dense: Option<Vec<[f64; 3]>>,
        f: impl FnOnce(&StrainSetup, &Frozen<StrainFields>) -> R,
    ) -> R {
        self.with_layout(dense, |_| {}, f)
    }

    /// The same with the layout (the request, the spin count) edited first: the refusals.
    pub fn with_layout<R>(
        &self,
        dense: Option<Vec<[f64; 3]>>,
        edit: impl FnOnce(&mut Layout),
        f: impl FnOnce(&StrainSetup, &Frozen<StrainFields>) -> R,
    ) -> R {
        let data = &self.data;
        let cores = data.cores();
        let inputs = Inputs {
            grid: self.grid(),
            shells: &data.shells,
            projectors: &data.projectors,
            kpoints: &self.kpoints,
            sites: data.sites(),
            shapes: &data.shapes,
            potentials: &data.potentials,
            cores: &cores,
            dense_cutoff: COMPENSATION_CUTOFF,
        };
        let mut layout = self.layout();
        edit(&mut layout);
        let mut setup = StrainSetup::new(self.ctx(), inputs, layout, data.operators()).unwrap();
        if let Some(vectors) = dense {
            setup = setup.with_dense_vectors(vectors);
        }
        let frozen = self.frozen(self.fields());
        f(&setup, &frozen)
    }
}

/// The cell, atoms, grid and k-points carried by `x -> (1 + eps) x` (fractional k held).
pub struct Frame {
    pub map: Matrix3<f64>,
    pub lattice: Matrix3<f64>,
    pub grid: SpatialGrid,
    pub atoms: Vec<mandacaru_core::periodic::setup::shapes::Atom>,
}

impl Frame {
    pub fn new(world: &World, eps: &Eps) -> Self {
        let map = Matrix3::identity() + matrix3(eps);
        let data = &world.data;
        let reference = world.grid();
        let step = map * reference.step;
        let mut rows = [0.; 9];
        for (i, v) in rows.iter_mut().enumerate() {
            *v = step[(i / 3, i % 3)];
        }
        let grid = SpatialGrid::new(
            reference.index.shape(),
            (map * Vector3::from(reference.origin)).into(),
            rows,
            reference.periodic,
        )
        .unwrap();
        let atoms = data
            .atoms
            .iter()
            .map(|a| {
                let mut a = a.clone();
                a.center = (map * Vector3::from(a.center)).into();
                a
            })
            .collect();
        Self {
            map,
            lattice: map * data.lattice,
            grid,
            atoms,
        }
    }

    pub fn centers(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|a| a.center).collect()
    }

    pub fn sites<'a>(
        &'a self,
        world: &'a World,
    ) -> mandacaru_core::periodic::setup::shapes::Sites<'a> {
        mandacaru_core::periodic::setup::shapes::Sites {
            atoms: &self.atoms,
            channels: &world.data.channels,
            lattice: self.lattice,
            sigma: world.data.sigma,
        }
    }
}
