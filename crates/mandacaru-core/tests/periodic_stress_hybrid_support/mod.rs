//! Shared by the `periodic_stress_hybrid*` tests (AS-9): the exchange strain of a hybrid crystal
//! at FIXED density matrices, against central differences of the production energy entry on a
//! crystal strained by `x -> (1 + eps) x`.
//!
//! The rig is a frozen hybrid case (`reference/periodic_gradients/<case>`: the crystal, the Rust
//! setup of the same structure) with hand-built density matrices (no SCF: Hermitian, positive
//! semidefinite, rank 4, different at every k-point). The strained crystal is the loaded one with
//! its strain-dependent members rewritten from the setup's builders, in three groups that can be
//! strained apart (their derivatives add, the energy being differentiated at one point):
//!
//! * `cell`: the grid (step and origin), the Cartesian k-points (fractional held), the atom
//!   centers and the lattice of the augmentation (`Augmentation::new` refuses a lattice that is
//!   not the grid's cell: the pin of the plan's correction 5 comes for free);
//! * `samples`: the Bloch samples, resampled at the strained nodes from the strained shells;
//! * `projections`: `C(k)` from the strained shells and projectors (rigid spheres).
//!
//! `cell` carries H-1, H-2 and H-3, `samples` H-4 and `projections` H-5 and H-6.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{inputs, snapshot_of, Data};

pub mod wrong;
use mandacaru_core::{
    math::{storage::ComplexSamples, SpatialGrid},
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{exchange_operators_with, ExxDivTreatment, Screening},
        setup::{
            pulay::{projection, Mesh},
            shapes::{Atom, Sites},
            shells::{DenseRows, Sampling, Shells},
            spheres::Projector,
        },
        stress_terms::hybrid::{exchange_strain, Production, Sources, Tangents, Terms},
        Crystal,
    },
    quantum::C64,
};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

pub type Eps = [[f64; 3]; 3];
pub type Tensor = [[f64; 3]; 3];

/// Which strain-dependent members of the crystal follow the strain.
#[derive(Clone, Copy)]
pub struct Parts {
    pub cell: bool,
    pub samples: bool,
    pub projections: bool,
}

pub const ALL: Parts = Parts {
    cell: true,
    samples: true,
    projections: true,
};
pub const CELL: Parts = Parts {
    cell: true,
    samples: false,
    projections: false,
};
pub const SAMPLES: Parts = Parts {
    cell: false,
    samples: true,
    projections: false,
};
pub const PROJECTIONS: Parts = Parts {
    cell: false,
    samples: false,
    projections: true,
};

pub fn entry(a: usize, b: usize, e: f64) -> Eps {
    let mut eps = [[0.; 3]; 3];
    eps[a][b] = e;
    eps
}

pub fn seq(i: usize, salt: usize) -> f64 {
    (((i * 7919 + salt * 104729 + 13) as f64) * 0.618033988749895).sin()
}

/// A Hermitian positive semidefinite density matrix of rank `min(m / 2, 4)`.
pub fn density(m: usize, k: usize) -> DMatrix<C64> {
    let rank = (m / 2).clamp(1, 4);
    let mut d = DMatrix::<C64>::zeros(m, m);
    for (j, f) in [1.8, 1.2, 0.8, 0.4].iter().take(rank).enumerate() {
        let x = DVector::<C64>::from_fn(m, |i, _| {
            C64::new(seq(i * 7 + j, k + 3), seq(i * 3 + j + 1, 2 * k + 5))
        });
        let norm = x.norm_squared();
        d += (&x * x.adjoint()) * C64::new(f / norm, 0.);
    }
    d
}

/// A basis, its projectors, the lattice and the k-point carried by `eps`: the centers move with
/// the cell and the fractional k-point is held.
pub struct Carried {
    pub lattice: Matrix3<f64>,
    pub shells: Shells,
    pub projectors: Vec<Projector>,
    pub k: [f64; 3],
}

impl Carried {
    pub fn new(
        lattice: &Matrix3<f64>,
        shells: &Shells,
        projectors: &[Projector],
        eps: &Matrix3<f64>,
        k: [f64; 3],
    ) -> Self {
        let map = Matrix3::identity() + eps;
        let at = |c: [f64; 3]| -> [f64; 3] { (map * Vector3::from(c)).into() };
        let moved = shells
            .shells()
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.center = at(s.center);
                s
            })
            .collect();
        let projectors = projectors
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.center = at(p.center);
                p
            })
            .collect();
        Self {
            lattice: map * lattice,
            shells: Shells::new(moved).unwrap(),
            projectors,
            k: (map.transpose().try_inverse().unwrap() * Vector3::from(k)).into(),
        }
    }
}

pub struct Rig {
    pub data: Data,
    /// The crystal the strained energies are evaluated on (rewritten for every strain).
    pub work: Crystal,
    pub d: Vec<DMatrix<C64>>,
    pub screening: Screening,
    pub exxdiv: ExxDivTreatment,
}

impl Rig {
    pub fn new(case: &'static str, exxdiv: ExxDivTreatment) -> Self {
        let data = Data::load(case);
        let snapshot = snapshot_of(case, &data.manifest);
        let work = Crystal::from_json(&data.crystal.context, &snapshot.to_string()).unwrap();
        let m = data.crystal.m;
        let d = (0..data.crystal.points.len())
            .map(|k| density(m, k))
            .collect();
        let screening = Screening::of(&data.crystal).unwrap();
        Self {
            data,
            work,
            d,
            screening,
            exxdiv,
        }
    }

    /// The analytic exchange strain at the unstrained cell with `sources`.
    pub fn analytic(&self, sources: &dyn Sources) -> Terms {
        let cores = self.data.cores();
        let inp = inputs(&self.data, &cores);
        let tangents = Tangents::new(&self.data.crystal.context, &inp).unwrap();
        exchange_strain(&self.data.crystal, &self.d, self.exxdiv, &tangents, sources).unwrap()
    }

    /// The grid carried by `map`.
    fn grid(&self, map: &Matrix3<f64>) -> SpatialGrid {
        let reference = &self.data.crystal.grid;
        let step = map * reference.step;
        let mut rows = [0.; 9];
        for (i, v) in rows.iter_mut().enumerate() {
            *v = step[(i / 3, i % 3)];
        }
        SpatialGrid::new(
            reference.index.shape(),
            (map * Vector3::from(reference.origin)).into(),
            rows,
            reference.periodic,
        )
        .unwrap()
    }

    /// Rewrites `work` for the strain `eps`: the members named by `parts` follow it, the others
    /// keep the reference values.
    fn set(&mut self, eps: &Eps, parts: &Parts) {
        let data = &self.data;
        let ctx = &data.crystal.context;
        let eps = Matrix3::from_fn(|a, b| eps[a][b]);
        let map = Matrix3::identity() + eps;
        let (m, n) = (data.crystal.m, data.crystal.grid.index.points());
        let strained = self.grid(&map);
        let mesh = Mesh::new(ctx, &strained).unwrap();
        let reference = self.grid(&Matrix3::identity());
        for (index, point) in self.work.points.iter_mut().enumerate() {
            let base = &data.crystal.points[index];
            let carried = Carried::new(&data.lattice, &data.shells, &data.projectors, &eps, base.k);
            point.k = if parts.cell { carried.k } else { base.k };
            point.projections = if parts.projections {
                projection(
                    ctx,
                    &carried.shells,
                    &carried.lattice,
                    carried.k,
                    &carried.projectors,
                )
                .unwrap()
            } else {
                base.projections.clone()
            };
            point.psi = if parts.samples {
                let mut psi = vec![C64::default(); m * n];
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
                ComplexSamples::Float64(ctx.copy_slice(&psi).unwrap())
            } else {
                ComplexSamples::Float64(ctx.copy_slice(&base.psi.read(ctx).unwrap()).unwrap())
            };
        }
        self.work.grid = if parts.cell { strained } else { reference };
    }

    /// Runs `f` on the crystal strained by `eps` in the `parts` and its augmentation (the sites
    /// follow the cell part).
    pub fn strained<R>(
        &mut self,
        eps: &Eps,
        parts: &Parts,
        f: impl FnOnce(&Crystal, &Augmentation) -> R,
    ) -> R {
        self.set(eps, parts);
        let data = &self.data;
        let map = if parts.cell {
            Matrix3::identity() + Matrix3::from_fn(|a, b| eps[a][b])
        } else {
            Matrix3::identity()
        };
        let atoms: Vec<Atom> = data
            .atoms
            .iter()
            .map(|a| {
                let mut a = *a;
                a.center = (map * Vector3::from(a.center)).into();
                a
            })
            .collect();
        let sites = Sites {
            atoms: &atoms,
            channels: &data.channels,
            lattice: map * data.lattice,
            sigma: data.sigma,
        };
        let augmentation = Augmentation::new(&self.work, sites, &data.shapes)
            .unwrap()
            .with_one_center(data.crystal.hybrid.as_ref().unwrap().one_center());
        f(&self.work, &augmentation)
    }

    /// The production exchange energy (`exchange_operators_with` with the one-center model) of
    /// the crystal strained by `eps` in the `parts`.
    pub fn energy(&mut self, eps: &Eps, parts: &Parts) -> f64 {
        let (d, screening, exxdiv) = (&self.d.clone(), self.screening, self.exxdiv);
        self.strained(eps, parts, |work, augmentation| {
            exchange_operators_with(work, d, screening, exxdiv, Some(augmentation))
                .unwrap()
                .energy
        })
    }

    /// `dE/d eps_ab` by Richardson central differences (steps `step` and `2 step`), and its floor:
    /// the distance to the same extrapolation at `step / 2`.
    pub fn difference(&mut self, a: usize, b: usize, parts: &Parts, step: f64) -> (f64, f64) {
        let mut central = |h: f64| {
            (self.energy(&entry(a, b, h), parts) - self.energy(&entry(a, b, -h), parts)) / (2. * h)
        };
        let (c1, c2, c_half) = (central(step), central(2. * step), central(step / 2.));
        let rich = c1 + (c1 - c2) / 3.;
        let rich_half = c_half + (c_half - c1) / 3.;
        (rich, (rich - rich_half).abs())
    }

    /// All nine entries: the values, and the worst floor.
    pub fn tensor(&mut self, parts: &Parts, step: f64) -> (Tensor, f64) {
        let mut out = [[0.; 3]; 3];
        let mut floor = 0f64;
        for a in 0..3 {
            for b in 0..3 {
                let (value, f) = self.difference(a, b, parts, step);
                out[a][b] = value;
                floor = floor.max(f);
            }
        }
        (out, floor)
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}

pub fn sum(parts: &[&Tensor]) -> Tensor {
    parts.iter().fold([[0.; 3]; 3], |t, p| add(&t, p))
}

pub fn transposed(t: &Tensor) -> Tensor {
    std::array::from_fn(|i| std::array::from_fn(|j| t[j][i]))
}

/// The worst entry of `got - want` (NaN counts as infinite).
pub fn miss(got: &Tensor, want: &Tensor) -> f64 {
    let mut worst = 0f64;
    for a in 0..3 {
        for b in 0..3 {
            let d = (got[a][b] - want[a][b]).abs();
            if !d.is_finite() {
                return f64::INFINITY;
            }
            worst = worst.max(d);
        }
    }
    worst
}

pub fn peak(t: &Tensor) -> f64 {
    t.iter().flatten().fold(0f64, |m, x| m.max(x.abs()))
}

/// The plan's gate (Hartree) and the multiple of the measured step floor that replaces it when
/// the floor is larger.
pub const GATE: f64 = 1e-8;
pub const FLOORS: f64 = 100.;

pub fn gate(floor: f64) -> f64 {
    GATE.max(FLOORS * floor)
}

pub fn report(name: &str, got: &Tensor, want: &Tensor, floor: f64) {
    let g = gate(floor);
    let m = miss(got, want);
    println!(
        "  {name}: peak {:.3e}, miss {m:.2e} Ha, floor {floor:.1e}, gate {g:.1e} ({:.2} x)",
        peak(want),
        m / g
    );
    assert!(m < g, "{name}: miss {m:e} against {g:e}");
    assert!(
        peak(want) > 100. * g,
        "{name}: the derivative {:e} is not above its gate",
        peak(want)
    );
}

/// The exchange strain of the hand-built state of `case` against the four differences (the cell,
/// the samples, the projections, all three), every entry.
pub fn check(case: &'static str, exxdiv: ExxDivTreatment, step: f64) {
    check_rig(Rig::new(case, exxdiv), case, step);
}

/// [`check`] of a rig already in hand (a spinor rig of `periodic_stress_spinor_hybrid_support`).
pub fn check_rig(mut rig: Rig, case: &str, step: f64) {
    let terms = rig.analytic(&Production);
    let t = &terms;
    println!(
        "{case} {}: pair energy {:+.9e} Ha",
        rig.exxdiv.as_str(),
        t.energy
    );
    for (name, part) in t.parts() {
        println!("  {name}: peak {:.3e}", peak(part));
    }
    let (cell, floor) = rig.tensor(&CELL, step);
    report(
        "cell (H-1 + H-2 + H-3)",
        &sum(&[&t.jacobian, &t.kernel, &t.shape_volume, &t.shape_form]),
        &cell,
        floor,
    );
    let (samples, floor) = rig.tensor(&SAMPLES, step);
    report(
        "samples (H-4: bra + ket)",
        &sum(&[&t.bra, &t.ket]),
        &samples,
        floor,
    );
    let (projections, floor) = rig.tensor(&PROJECTIONS, step);
    report(
        "projections (H-5 + H-6)",
        &sum(&[&t.compensation, &t.sphere]),
        &projections,
        floor,
    );
    let (all, floor) = rig.tensor(&ALL, step);
    report("all", &t.total(), &all, floor);
}

/// The kernel controls on the cell group alone (the kernel and the head are cell members): the
/// derivatives with a wrong kernel strain must miss the central difference by more than 100 gates
/// where the right one meets it. On a k mesh every pair has its own `k_diff`, the Nyquist faces
/// of the alias averages are live and the head weight is the pair's own.
pub fn controls_kernel(case: &'static str, exxdiv: ExxDivTreatment, step: f64) {
    use wrong::{MeanVector, NoHead, ZeroPair};
    let mut rig = Rig::new(case, exxdiv);
    let (cell, floor) = rig.tensor(&CELL, step);
    let g = gate(floor);
    let cell_of = |t: &Terms| sum(&[&t.jacobian, &t.kernel, &t.shape_volume, &t.shape_form]);
    let right = rig.analytic(&Production);
    assert!(
        miss(&cell_of(&right), &cell) < g,
        "the right kernel meets the gate"
    );
    let no_head = NoHead::default();
    let mut variants: Vec<(&str, &dyn Sources)> =
        vec![("the mean vector", &MeanVector), ("k_diff = 0", &ZeroPair)];
    if exxdiv == ExxDivTreatment::GygiBaldereschi {
        variants.push(("the head dropped", &no_head));
    }
    for (name, sources) in variants {
        let bite = miss(&cell_of(&rig.analytic(sources)), &cell) / g;
        println!("  control {name}: bites {bite:.2e} gates");
        assert!(bite > 100., "{name}: bites only {bite:.2}");
    }
    if exxdiv == ExxDivTreatment::GygiBaldereschi {
        // Only the pairs with `k_diff = 0` (one per k-point) have a head entry.
        assert!(no_head.heads.get() > 0, "no head entry in any pair");
        println!("  the head control removed {} entries", no_head.heads.get());
    }
}
