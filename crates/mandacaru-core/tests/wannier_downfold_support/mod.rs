//! Shared by `wannier_downfold` and `wannier_screening`: an H2 molecule in a chain cell, localized
//! on a full mesh by the production path. Needs `MANDACARU_PAW_PATH`.
#![allow(dead_code)]
use crate::setup_shells_support::dataset;
use mandacaru_core::{
    paw::digest::sha256_file,
    periodic::{
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options as Scf,
        wannier::{
            downfold::{CoulombBox, Request},
            model, sampling,
            setup::{Setup, Wannier},
            Options,
        },
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use serde_json::json;
use std::sync::Arc;

/// H2 (1.4 Bohr bond along x) in a 6 x 8 x 8 Bohr cell, a mesh of `size` cells along x.
pub fn chain(size: [usize; 3]) -> (Context, Wannier) {
    chain_with(size, None)
}

/// [`chain`], two spin channels when `moments` (the starting moments of the two atoms) is given.
pub fn chain_with(size: [usize; 3], moments: Option<[f64; 2]>) -> (Context, Wannier) {
    chain_on(size, moments, 2)
}

/// [`chain_with`] on a context of `threads` workers.
pub fn chain_on(size: [usize; 3], moments: Option<[f64; 2]>, threads: usize) -> (Context, Wannier) {
    let ctx = Context::new(Some(threads), 2 << 30).unwrap();
    let root =
        std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join("lda-sr");
    let path = root.join("H.parquet");
    let files = vec![(
        path.to_string_lossy().into_owned(),
        sha256_file(&path).unwrap(),
    )];
    let cell = [6., 8., 8.];
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: Matrix3::from_diagonal(&nalgebra::Vector3::from(cell)),
        grid_shape: [15, 20, 20],
        atoms: [-0.7, 0.7]
            .iter()
            .enumerate()
            .map(|(i, dx)| AtomSpec {
                species: 0,
                symbol: "H".into(),
                fractional: [0.5 + dx / cell[0], 0.5, 0.5],
                moment: moments.map_or(0., |m| m[i]),
            })
            .collect(),
        kpoints_frac: vec![[0., 0., 0.]],
        weights: vec![1.],
        functional: "lda".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: moments.is_some(),
        symmetry: false,
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    };
    let datasets = vec![Arc::new(dataset(&ctx, "lda-sr", "H"))];
    let setup = Arc::new(create_setup(&ctx, input, datasets).unwrap());
    let crystal = Arc::new(setup.build(&ctx).unwrap());
    let state = Arc::new(
        crystal
            .solve(&Scf {
                tolerance: 1e-9,
                density_tolerance: 1e-7,
                ..Scf::default()
            })
            .unwrap(),
    );
    assert!(state.converged);
    let description = crystal.description.as_ref().unwrap();
    let request = json!({
        "size": size,
        "trials": [
            {"center": [2.3, 4., 4.], "orbital": "s", "label": "H0"},
            {"center": [3.7, 4., 4.], "orbital": "s", "label": "H1"},
        ],
        "centers": description.centers,
        "channel_labels": description.channels.iter()
            .map(|c| json!([c.0, c.1, c.2])).collect::<Vec<_>>(),
    });
    let mut wannier = Setup::from_json(crystal.clone(), state, &request.to_string()).unwrap();
    let count = size.iter().product::<usize>();
    let fractional: Vec<[f64; 3]> = (0..count)
        .map(|k| sampling::mesh_point(size, k).unwrap())
        .collect();
    let weights = vec![1. / count as f64; count];
    for point in setup.matrices_at(&ctx, &fractional, &weights).unwrap() {
        wannier.add_point(point).unwrap();
    }
    let wannier = wannier
        .with_provider(Some(setup.clone()))
        .finish(&Options::default())
        .unwrap();
    assert!(wannier.channels[0].localized.converged);
    (ctx, wannier)
}

pub fn request(functions: Option<Vec<(usize, [i64; 3])>>) -> Request {
    Request {
        functions,
        num_particles: None,
        screening: None,
        cutoff: None,
        bands: None,
        coulomb: CoulombBox::Supercell,
        workers: None,
    }
}

/// The same functions with a phase `exp(i theta_n)` on each: a different gauge of one model, in
/// which the Hamiltonian and density blocks are complex.
pub fn rephase(wannier: &mut Wannier, theta: [f64; 2]) {
    let ctx = wannier.crystal.context.clone();
    let size = wannier.size;
    let count = size.iter().product::<usize>();
    let (smearing, fermi, width) = (
        wannier.state.options.smearing,
        wannier.state.fermi,
        wannier.state.options.width,
    );
    for channel in &mut wannier.channels {
        let mut hk = vec![C64::default(); count * 4];
        let mut rho = vec![C64::default(); count * 4];
        for (k, gauge) in channel.localized.gauge.iter_mut().enumerate() {
            for n in 0..2 {
                let phase = C64::from_polar(1., theta[n]);
                for m in 0..gauge.nrows() {
                    gauge[(m, n)] *= phase;
                }
            }
            let energies = channel.frames[k].energies.as_slice();
            let h = gauge.adjoint()
                * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    energies.len(),
                    energies.iter().map(|e| C64::new(*e, 0.)),
                ))
                * &*gauge;
            let d = gauge.adjoint()
                * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    energies.len(),
                    energies
                        .iter()
                        .map(|e| C64::new(2. * smearing.occupation((e - fermi) / width), 0.)),
                ))
                * &*gauge;
            for i in 0..2 {
                for j in 0..2 {
                    hk[(k * 2 + i) * 2 + j] = h[(i, j)];
                    rho[(k * 2 + i) * 2 + j] = d[(i, j)];
                }
            }
        }
        channel.hamiltonian_bvk = model::fourier_blocks(&ctx, size, 2, &hk).unwrap();
        channel.density_bvk = model::fourier_blocks(&ctx, size, 2, &rho).unwrap();
    }
}

/// The chain's Born-von Karman supercell (`size` cells along the chain) as a Gamma-point crystal
/// with the PRIMITIVE cell's Kohn-Sham potential tiled onto its grid: the same Hamiltonian as the
/// mesh of `wannier` (a non-self-consistent state), so its Gamma states are the mesh's states.
pub fn supercell(
    wannier: &Wannier,
    size: usize,
) -> (
    mandacaru_core::periodic::Crystal,
    mandacaru_core::periodic::PeriodicResult,
) {
    let ctx = wannier.crystal.context.clone();
    let root =
        std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join("lda-sr");
    let path = root.join("H.parquet");
    let files = vec![(
        path.to_string_lossy().into_owned(),
        sha256_file(&path).unwrap(),
    )];
    let cell = [6. * size as f64, 8., 8.];
    let mut atoms = vec![];
    for j in 0..size {
        for dx in [-0.7, 0.7] {
            atoms.push(AtomSpec {
                species: 0,
                symbol: "H".into(),
                fractional: [(3. + dx + 6. * j as f64) / cell[0], 0.5, 0.5],
                moment: 0.,
            });
        }
    }
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: Matrix3::from_diagonal(&nalgebra::Vector3::from(cell)),
        grid_shape: [15 * size, 20, 20],
        atoms,
        kpoints_frac: vec![[0., 0., 0.]],
        weights: vec![1.],
        functional: "lda".into(),
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
    };
    let datasets = vec![Arc::new(dataset(&ctx, "lda-sr", "H"))];
    let crystal = create_setup(&ctx, input, datasets)
        .unwrap()
        .build(&ctx)
        .unwrap();
    let mut state = crystal
        .solve(&Scf {
            tolerance: 1e-9,
            density_tolerance: 1e-7,
            ..Scf::default()
        })
        .unwrap();
    // the primitive potential, tiled: V_s(x, y, z) = V_p(x mod 15, y, z)
    let (p, q) = (
        wannier.crystal.grid.index.points(),
        crystal.grid.index.points(),
    );
    let primitive = &wannier.state.potential.as_slice()[..p];
    let shape = crystal.grid.index.shape();
    let tiled = state.potential.as_mut_slice();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                tiled[(x * shape[1] + y) * shape[2] + z] = primitive[((x % 15) * 20 + y) * 20 + z];
            }
        }
    }
    assert_eq!(tiled.len(), q);
    (crystal, state)
}
