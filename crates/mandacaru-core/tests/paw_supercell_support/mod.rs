//! Shared by the PAW-LCAO supercell tests (`molecular_paw_supercell`, `periodic_variational_forces`):
//! the primitive cells, the crystal setup of `copies` cells, the raw-AO view of a Hamiltonian, and a
//! molecule in a box with its molecular PAW-LCAO Hamiltonian (also `molecular_stability`).
#![allow(dead_code)]

use mandacaru_core::{
    molecular::MolecularIntegrals,
    periodic::setup::{
        parse::{create_setup, AtomSpec, SetupInput},
        PeriodicSetup,
    },
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use std::sync::Arc;

/// A primitive cell: lattice (columns, Bohr), grid, atoms `(symbol, fractional)`.
pub struct Primitive {
    pub lattice: Matrix3<f64>,
    pub shape: [usize; 3],
    pub atoms: Vec<(&'static str, [f64; 3])>,
}

/// LiH in a skew cell.
pub fn lithium_hydride_cell() -> Primitive {
    Primitive {
        lattice: Matrix3::new(7., 0.8, 0.4, 0., 6.5, 0.6, 0., 0., 7.5),
        shape: [20, 19, 22],
        atoms: vec![("Li", [0.1, 0.2, 0.15]), ("H", [0.1 + 3. / 7., 0.2, 0.15])],
    }
}

/// An H2 chain in a skew cell: one molecule (bond 1.4 Bohr, along `a1`) per 5.6 Bohr.
pub fn hydrogen_chain_cell() -> Primitive {
    Primitive {
        lattice: Matrix3::new(5.6, 0.3, 0.2, 0., 7., 0.4, 0., 0., 7.),
        shape: [16, 20, 20],
        atoms: vec![("H", [0.1, 0.2, 0.15]), ("H", [0.1 + 1.4 / 5.6, 0.22, 0.15])],
    }
}

/// Silicon (fcc, 10.26 Bohr), two atoms in the primitive cell, a grid of about 0.2 A.
pub fn silicon_cell() -> Primitive {
    let h = 10.26 / 2.;
    Primitive {
        lattice: Matrix3::new(0., h, h, h, 0., h, h, h, 0.),
        shape: [20, 20, 20],
        atoms: vec![("Si", [0.; 3]), ("Si", [0.25; 3])],
    }
}

/// An N2 chain in a skew cell (p functions in the basis): one molecule (bond 2.07 Bohr, along
/// `a1`) per 5.6 Bohr.
pub fn nitrogen_chain_cell() -> Primitive {
    Primitive {
        lattice: Matrix3::new(5.6, 0.3, 0.2, 0., 7., 0.4, 0., 0., 7.),
        shape: [16, 20, 20],
        atoms: vec![("N", [0.1, 0.2, 0.15]), ("N", [0.1 + 2.07 / 5.6, 0.23, 0.16])],
    }
}

/// `copies` cells of `cell` along `a1` as one cell (atoms cell-major), at Gamma, on the same grid
/// spacing, with the filter cutoff and the ion width of the primitive cell pinned, every atom
/// moved by `shift` (Bohr). The setup names its dataset files (the pair engine reads them).
pub fn cells_of(
    ctx: &Context,
    library: &str,
    spinor: bool,
    cell: &Primitive,
    copies: usize,
    shift: [f64; 3],
) -> PeriodicSetup {
    cells_with_basis(ctx, library, spinor, cell, copies, shift, "SZ")
}

/// [`cells_of`] with the basis size `basis_size` (`"SZ"`, `"DZP"`, ...).
pub fn cells_with_basis(
    ctx: &Context,
    library: &str,
    spinor: bool,
    cell: &Primitive,
    copies: usize,
    shift: [f64; 3],
    basis_size: &str,
) -> PeriodicSetup {
    let primitive = cell.lattice;
    let shape = cell.shape;
    let spacing = (0..3).map(|a| primitive.column(a).norm() / shape[a] as f64).fold(0f64, f64::max);
    let mut lattice = primitive;
    lattice.set_column(0, &(primitive.column(0) * copies as f64));
    let inverse = lattice.try_inverse().unwrap();
    let offset = inverse * nalgebra::Vector3::from(shift);
    let mut species: Vec<&str> = Vec::new();
    for (symbol, _) in &cell.atoms {
        if !species.contains(symbol) {
            species.push(symbol);
        }
    }
    let mut atoms = Vec::new();
    for copy in 0..copies {
        for (symbol, x) in &cell.atoms {
            atoms.push(AtomSpec {
                species: species.iter().position(|s| s == symbol).unwrap(),
                symbol: (*symbol).into(),
                fractional: [
                    (x[0] + copy as f64) / copies as f64 + offset[0],
                    x[1] + offset[1],
                    x[2] + offset[2],
                ],
                moment: 0.,
            });
        }
    }
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(library);
    let files = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            let digest = mandacaru_core::paw::digest::dataset_content_sha256(ctx, &path).unwrap();
            (path.to_string_lossy().into_owned(), digest)
        })
        .collect();
    let datasets = species
        .iter()
        .map(|symbol| Arc::new(crate::setup_shells_support::dataset(ctx, library, symbol)))
        .collect();
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice,
        grid_shape: [shape[0] * copies, shape[1], shape[2]],
        atoms,
        kpoints_frac: vec![[0.; 3]],
        weights: vec![1.],
        functional: "lda".into(),
        precision: "float64".into(),
        basis_size: basis_size.into(),
        spinor,
        spin: false,
        symmetry: false,
        filter_cutoff: Some(std::f64::consts::PI / spacing),
        sigma: Some(1.4 * spacing),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    };
    create_setup(ctx, input, datasets).unwrap()
}

/// The overlap, the one-body operator (from the Lowdin frame) and every two-body integral
/// `(ab|cd)` (chemists' order, `[((a m + b) m + c) m + d]`) of `integrals` in its raw atomic
/// orbitals.
pub fn raw_atomic_orbitals(integrals: &MolecularIntegrals) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let m = integrals.m();
    let square = |b: &mandacaru_core::Buffer<f64>| DMatrix::from_row_slice(m, m, b.as_slice());
    let y = integrals.lowdin.clone().try_inverse().unwrap();
    let h = y.transpose() * square(&integrals.one) * &y;
    let eri = integrals.eri().unwrap();
    let mut t = vec![0f64; m * m * m * m];
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    t[((p * m + q) * m + r) * m + s] = eri.value(p, r, q, s);
                }
            }
        }
    }
    // Four one-index transforms with Y: (ab|cd) = sum Y_pa Y_qb Y_rc Y_sd (pq|rs).
    for axis in 0..4 {
        let mut out = vec![0f64; m * m * m * m];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut idx = [i / (m * m * m), (i / (m * m)) % m, (i / m) % m, i % m];
            let target = idx[axis];
            for p in 0..m {
                idx[axis] = p;
                *slot += y[(p, target)] * t[((idx[0] * m + idx[1]) * m + idx[2]) * m + idx[3]];
            }
        }
        t = out;
    }
    (square(&integrals.overlap), h, t)
}

/// The closed-shell energy (Hartree, constants included) of the raw-AO density `p` (both spins)
/// on the raw-AO operators `h`, `eri` (chemists' order) and the constant `constant`.
pub fn closed_shell_energy(h: &DMatrix<f64>, eri: &[f64], constant: f64, p: &DMatrix<f64>) -> f64 {
    let m = h.nrows();
    let mut value = h.dot(p) + constant;
    for ab in 0..m * m {
        for cd in 0..m * m {
            let (a, b, c, d) = (ab / m, ab % m, cd / m, cd % m);
            value += eri[ab * m * m + cd] * (0.5 * p[(a, b)] * p[(c, d)] - 0.25 * p[(a, c)] * p[(b, d)]);
        }
    }
    value
}

/// The PAW-LCAO supercell of `mesh` on `setup` with its primitive atom `atom` moved, with all its
/// images, by `shift` (Bohr, Cartesian): its Hamiltonian, Cholesky tolerance 1e-12.
pub fn displaced(ctx: &Context, setup: &PeriodicSetup, mesh: [usize; 3], atom: usize, shift: [f64; 3]) -> MolecularIntegrals {
    let mut fractional = setup.fractional().unwrap();
    let inverse = setup.lattice.try_inverse().unwrap();
    let step = inverse * nalgebra::Vector3::from(shift);
    for a in 0..3 {
        fractional[atom][a] += step[a];
    }
    let moved = setup.with_positions(ctx, &fractional).unwrap();
    MolecularIntegrals::paw_supercell(ctx, Arc::new(moved), mesh, Some(1e-12)).unwrap()
}

/// A molecule `atoms` (Bohr, centered) in a cubic box of `side` Bohr on a grid of `nodes` per
/// axis: the PAW-LCAO supercell setup at Gamma (all three axes periodic), with the filter cutoff
/// and the ion width pinned to the grid spacing as in `cells_of`.
pub fn molecule_in_a_box(ctx: &Context, atoms: &[(&'static str, [f64; 3])], side: f64, nodes: usize) -> PeriodicSetup {
    let cell = Primitive {
        lattice: Matrix3::from_diagonal_element(side),
        shape: [nodes; 3],
        atoms: atoms
            .iter()
            .map(|(symbol, x)| (*symbol, std::array::from_fn(|a| 0.5 + x[a] / side)))
            .collect(),
    };
    cells_of(ctx, "lda-sr", false, &cell, 1, [0.; 3])
}

/// The molecular PAW-LCAO Hamiltonian of the molecule of `setup`, on its own grid (isolated
/// Poisson), the same datasets and the same radial tables of its basis.
pub fn molecular_integrals(ctx: &Context, setup: &PeriodicSetup) -> MolecularIntegrals {
    let description = setup.description.as_ref().unwrap();
    let mut arrays = std::collections::BTreeMap::new();
    for (t, table) in description.tables.iter().enumerate() {
        arrays.insert(format!("r{t}"), ctx.copy_slice(&table.r).unwrap());
        arrays.insert(format!("v{t}"), ctx.copy_slice(&table.values).unwrap());
    }
    let basis: Vec<serde_json::Value> = (0..description.function_atom.len())
        .map(|mu| {
            let t = description.function_table[mu];
            serde_json::json!({
                "atom": description.function_atom[mu],
                "l": description.tables[t].l,
                "m": description.function_m[mu],
                "r": format!("r{t}"),
                "values": format!("v{t}"),
            })
        })
        .collect();
    let species: Vec<serde_json::Value> = description
        .datasets
        .iter()
        .map(|(file, _)| serde_json::json!({"file": file}))
        .collect();
    let metadata = serde_json::json!({
        "format": "molecular-paw-input",
        "version": 1,
        "length_unit": "bohr",
        "energy_unit": "Ha",
        "projector_basis": "raw",
        "atoms": (0..setup.centers.len()).collect::<Vec<_>>(),
        "species": species,
        "basis": basis,
    });
    let numbers = setup.atoms_species.iter().map(|&s| setup.datasets[s].atomic_number as u8).collect();
    let lattice: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|x| setup.lattice[(x, a)]));
    let geometry = mandacaru_core::Geometry::new(
        numbers,
        setup.centers.clone(),
        lattice,
        [false; 3],
        mandacaru_core::LengthUnit::Bohr,
    )
    .unwrap();
    let shape = setup.grid.index.shape();
    let grid = mandacaru_core::math::SpatialGrid::new(shape, setup.grid.origin, setup.grid.step_values(), false).unwrap();
    let integrals = mandacaru_core::paw::molecular::build(
        ctx,
        geometry,
        grid,
        &metadata.to_string(),
        arrays,
        "spectral",
        1e-10,
        mandacaru_core::math::storage::Precision::Float64,
    )
    .unwrap();
    integrals
}
