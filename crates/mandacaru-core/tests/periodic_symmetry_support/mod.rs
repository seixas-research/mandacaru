//! Shared by the symmetry-reduced-mesh tests (`periodic_hubbard_symmetry.rs`,
//! `periodic_magnetic_symmetry.rs`): the crystals, their whole-mesh references and the gates.
#![allow(dead_code)]
use mandacaru_core::{
    paw::record::PawDataset,
    periodic::{
        setup::{
            create_setup, parse_input_json,
            symmetry::{Geometry, Symmetry},
        },
        Crystal, Options, PeriodicResult,
    },
    Context,
};
use nalgebra::Matrix3;
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};

pub const HARTREE_EV: f64 = 27.211386245988;
pub const ANGSTROM: f64 = 1.8897261246257702;
/// `(occupations, energy in eV, density tolerance of the SCF)`. The first is for the cells where
/// the quadrature is exactly invariant under the operations; a mesh halved by time reversal
/// alone leaves the density unsymmetrized, which keeps its residual above 5e-9.
pub const EXACT: (f64, f64, f64) = (1e-10, 1e-8, 5e-10);
/// Above the symmetry floor of the cubic operations (8e-7 and 5.4e-6 eV measured for NiO, plain
/// DFT included), three orders below the error of a wrong convention.
pub const CUBIC: (f64, f64, f64) = (3e-6, 2e-5, 2e-8);

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

pub fn datasets(ctx: &Context, symbols: &[&str]) -> Vec<Arc<PawDataset>> {
    let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
    symbols
        .iter()
        .map(|s| {
            let path = PathBuf::from(&root).join(format!("lda-sr/{s}.parquet"));
            Arc::new(PawDataset::read(ctx, &path).unwrap())
        })
        .collect()
}

/// Fractional coordinates of the Monkhorst-Pack points of `spectrum::mesh`.
pub fn full_mesh(size: [usize; 3], gamma: bool) -> Vec<Value> {
    let mut points = vec![];
    let coordinate = |a: usize, i: usize| {
        (2. * i as f64 + 1. - size[a] as f64) / (2. * size[a] as f64)
            + if gamma && size[a] % 2 == 0 {
                0.5 / size[a] as f64
            } else {
                0.
            }
    };
    let weight = 1. / (size[0] * size[1] * size[2]) as f64;
    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                points.push(
                    json!({"fractional": [coordinate(0, i), coordinate(1, j), coordinate(2, k)],
                                   "weight": weight}),
                );
            }
        }
    }
    points
}

/// A crystal description in Bohr; `lattice` rows are written as columns of the cell.
pub struct Cell {
    pub lattice: [f64; 9],
    pub grid: [usize; 3],
    pub atoms: Vec<(&'static str, usize, [f64; 3])>,
    pub symbols: Vec<&'static str>,
    /// One initial moment per atom (empty: none).
    pub moments: Vec<f64>,
}

impl Cell {
    pub fn request(&self, hubbard: Value, spin: bool) -> Value {
        json!({"lattice": self.lattice, "grid_shape": self.grid,
               "atoms": self.atoms.iter().enumerate().map(|(i, (s, species, f))|
                   json!({"species": species, "symbol": s, "fractional": f,
                              "moment": self.moments.get(i).copied().unwrap_or(0.)})).collect::<Vec<_>>(),
               "hubbard_u": hubbard, "spin": spin, "functional": "lda", "basis_size": "SZ"})
    }
    /// The reduced run (`symmetry` on, the mesh reduced by the setup) and the whole-mesh
    /// reference (`symmetry` off, every point listed).
    pub fn crystals(&self, hubbard: Value, spin: bool, size: [usize; 3]) -> (Crystal, Crystal) {
        self.crystals_with(hubbard, spin, size, true)
    }
    /// [`Cell::crystals`] with the symmetry of the reduced run chosen: `false` keeps only the
    /// time-reversal halving of the mesh the setup makes for a non-magnetic crystal.
    pub fn crystals_with(
        &self,
        hubbard: Value,
        spin: bool,
        size: [usize; 3],
        symmetry: bool,
    ) -> (Crystal, Crystal) {
        let ctx = context();
        let build = |raw: Value| {
            let setup = create_setup(
                &ctx,
                parse_input_json(&raw).unwrap(),
                datasets(&ctx, &self.symbols),
            )
            .unwrap();
            setup.build(&ctx).unwrap()
        };
        let mut reduced = self.request(hubbard.clone(), spin);
        reduced["mesh"] = json!({"size": size, "gamma": true});
        reduced["symmetry"] = json!(symmetry);
        let mut whole = self.request(hubbard, spin);
        whole["kpoints"] = Value::Array(full_mesh(size, true));
        whole["symmetry"] = json!(false);
        (build(reduced), build(whole))
    }
}

pub fn nickel_fcc() -> Cell {
    let h = 3.52 * ANGSTROM / 2.;
    Cell {
        lattice: [0., h, h, h, 0., h, h, h, 0.],
        grid: [10, 10, 10],
        atoms: vec![("Ni", 0, [0., 0., 0.])],
        symbols: vec!["Ni"],
        moments: vec![],
    }
}

/// Rocksalt NiO in its two-atom primitive cell (non-magnetic: the group is the full cubic one).
pub fn nickel_oxide() -> Cell {
    let h = 4.17 * ANGSTROM / 2.;
    Cell {
        lattice: [0., h, h, h, 0., h, h, h, 0.],
        grid: [10, 10, 10],
        atoms: vec![("Ni", 0, [0., 0., 0.]), ("O", 1, [0.5, 0.5, 0.5])],
        symbols: vec!["Ni", "O"],
        moments: vec![],
    }
}

/// Rocksalt NiO in the four-atom rhombohedral cell of the type-II antiferromagnet (two Ni on
/// alternate (111) layers, two O), here WITHOUT moments: the operations of the cubic crystal
/// then swap the two Ni atoms, by a translation that is not a lattice vector of this cell.
pub fn nickel_oxide_antiferromagnetic_cell() -> Cell {
    let a = 4.17 * ANGSTROM;
    let vectors = [[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]].map(|v| v.map(|x| x * a / 2.));
    // Columns of the cell, as `Cell::lattice` writes them row by row.
    let cell = Matrix3::from_fn(|i, j| vectors[j][i]);
    let lattice: [f64; 9] = std::array::from_fn(|k| cell[(k / 3, k % 3)]);
    let inverse = cell.try_inverse().unwrap();
    let fractional = |cartesian: [f64; 3]| {
        let f = inverse * nalgebra::Vector3::from(cartesian);
        [f[0], f[1], f[2]]
    };
    let second_nickel = [0., a / 2., a / 2.];
    let oxygen = [a / 2., 0., 0.];
    let moved = |p: [f64; 3]| {
        [
            p[0] + second_nickel[0],
            p[1] + second_nickel[1],
            p[2] + second_nickel[2],
        ]
    };
    Cell {
        lattice,
        grid: [14, 14, 14],
        atoms: vec![
            ("Ni", 0, fractional([0.; 3])),
            ("Ni", 0, fractional(second_nickel)),
            ("O", 1, fractional(oxygen)),
            ("O", 1, fractional(moved(oxygen))),
        ],
        symbols: vec!["Ni", "O"],
        moments: vec![],
    }
}

/// Hexagonal close-packed Ni: the screw axis swaps the two atoms, the site group is D3h.
pub fn nickel_hcp() -> Cell {
    let (a, c) = (2.49 * ANGSTROM, 4.07 * ANGSTROM);
    Cell {
        lattice: [a, -a / 2., 0., 0., a * 3f64.sqrt() / 2., 0., 0., 0., c],
        grid: [8, 8, 12],
        atoms: vec![
            ("Ni", 0, [1. / 3., 2. / 3., 0.25]),
            ("Ni", 0, [2. / 3., 1. / 3., 0.75]),
        ],
        symbols: vec!["Ni"],
        moments: vec![],
    }
}

/// Three Ni atoms around a threefold axis (a 3-cycle of the group `3m`): the operations permute
/// the Hubbard atoms cyclically and the site group is only a mirror.
pub fn nickel_triangle() -> Cell {
    let (a, c) = (5.0 * ANGSTROM, 3.9 * ANGSTROM);
    let x = 0.28;
    Cell {
        lattice: [a, -a / 2., 0., 0., a * 3f64.sqrt() / 2., 0., 0., 0., c],
        grid: [12, 12, 10],
        atoms: vec![
            ("Ni", 0, [x, 0., 0.]),
            ("Ni", 0, [0., x, 0.]),
            ("Ni", 0, [-x, -x, 0.]),
        ],
        symbols: vec!["Ni"],
        moments: vec![],
    }
}

pub fn options(density_tolerance: f64) -> Options {
    Options {
        width: 0.05 / HARTREE_EV,
        density_tolerance,
        tolerance: 1e-10,
        max_iterations: 200,
        ..Default::default()
    }
}

pub fn solve(c: &Crystal, density_tolerance: f64) -> PeriodicResult {
    let state = c.solve(&options(density_tolerance)).unwrap();
    if !state.converged {
        for it in state.history.iter().rev().take(6).rev() {
            eprintln!(
                "iteration {}: free = {:.12}, residual = {:.3e}",
                it.iteration, it.free_energy, it.residual
            );
        }
        panic!("the SCF did not converge on {} k-points", c.points.len());
    }
    state
}

pub fn occupation_error(a: &PeriodicResult, b: &PeriodicResult) -> f64 {
    let (a, b) = (a.hubbard.as_ref().unwrap(), b.hubbard.as_ref().unwrap());
    a.occupations
        .iter()
        .zip(&b.occupations)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).norm()))
        .fold(0., f64::max)
}

pub fn dudarev(u: f64) -> Value {
    json!({"Ni": {"l": "d", "U": u}})
}

/// The reduced mesh against the whole one: occupations, energy; returns the k-point counts.
pub fn compare(
    cell: &Cell,
    hubbard: Value,
    spin: bool,
    size: [usize; 3],
    gates: (f64, f64, f64),
) -> (usize, usize) {
    compare_with(cell, hubbard, spin, size, true, gates)
}

pub fn compare_with(
    cell: &Cell,
    hubbard: Value,
    spin: bool,
    size: [usize; 3],
    symmetry: bool,
    (occupation_gate, energy_gate, density_tolerance): (f64, f64, f64),
) -> (usize, usize) {
    let (reduced, whole) = cell.crystals_with(hubbard, spin, size, symmetry);
    let (a, b) = (
        solve(&reduced, density_tolerance),
        solve(&whole, density_tolerance),
    );
    let occupations = occupation_error(&a, &b);
    let energy = (a.energy - b.energy).abs() * HARTREE_EV;
    let u = (a.hubbard_energy.unwrap() - b.hubbard_energy.unwrap()).abs() * HARTREE_EV;
    eprintln!(
        "{} k-points against {}: |dn| = {occupations:.2e}, |dE| = {energy:.2e} eV, |dE_U| = {u:.2e} eV, \
         E = {:.10} Ha",
        reduced.points.len(),
        whole.points.len(),
        a.energy
    );
    assert!(
        occupations < occupation_gate,
        "occupations differ by {occupations:.2e}"
    );
    assert!(energy < energy_gate, "energies differ by {energy:.2e} eV");
    // The +U term alone is not at the floor of the plain-DFT energy.
    assert!(u < energy_gate.min(2e-6), "E_U differs by {u:.2e} eV");
    assert!(
        a.hubbard_energy.unwrap() * HARTREE_EV > 1e-3,
        "the +U term must not be negligible"
    );
    (reduced.points.len(), whole.points.len())
}

/// The group the space-group search finds for `cell` (Ni and O atoms).
pub fn group_of(cell: &Cell) -> mandacaru_core::periodic::point_group::PointGroup {
    let lattice = Matrix3::from_row_slice(&cell.lattice);
    let fractional: Vec<[f64; 3]> = cell.atoms.iter().map(|a| a.2).collect();
    let species: Vec<u32> = cell
        .atoms
        .iter()
        .map(|a| if a.0 == "O" { 8 } else { 28 })
        .collect();
    let geometry = Geometry {
        lattice,
        fractional: &fractional,
        species: &species,
        labels: &species,
        grid: cell.grid,
        symprec: 1e-4,
        max_l: 4,
    };
    let symmetry = Symmetry::new(&geometry).unwrap();
    symmetry.to_point_group(fractional.len()).unwrap()
}
