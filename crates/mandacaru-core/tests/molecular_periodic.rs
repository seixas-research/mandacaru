//! The Born-von Karman supercell Hamiltonian (`molecular::periodic`): commensurate grid,
//! image-summed orbitals, lattice-summed electrostatics and the unfolding onto the primitive cell.
use mandacaru_core::{
    math::storage::Precision,
    molecular::{
        periodic::{madelung_constant, mean_field_bands, Supercell},
        MolecularIntegrals, ScfOptions,
    },
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;

const BOHR: f64 = 0.529177210903;

/// One H per primitive cell, `a` Angstrom along x, 6 Angstrom of vacuum transversely.
fn chain(a: f64) -> Geometry {
    Geometry::new(
        vec![1],
        vec![[0.; 3]],
        [[a, 0., 0.], [0., 6., 0.], [0., 0., 6.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap()
}

fn supercell(ctx: &Context, primitive: &Geometry, mesh: [usize; 3], h: f64) -> MolecularIntegrals {
    MolecularIntegrals::supercell(
        ctx,
        primitive,
        mesh,
        r#"{"name":"HAO"}"#,
        h / BOHR,
        None,
        Precision::Float64,
    )
    .unwrap()
}

#[test]
fn the_madelung_constant_of_the_simple_cubic_lattice_is_the_tabulated_one() {
    // v_M = -alpha / L with alpha = 2.837297479 for a unit charge in a neutralizing background.
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    for length in [3.0, 7.5] {
        let cell = [[length, 0., 0.], [0., length, 0.], [0., 0., length]];
        let v = madelung_constant(&ctx, &cell).unwrap();
        assert!((v * length + 2.837297479).abs() < 1e-8, "{v}");
    }
}

#[test]
fn the_supercell_is_cell_major_and_its_grid_is_commensurate() {
    let primitive = chain(2.0);
    let (supercell, geometry) = Supercell::new(&primitive, [3, 1, 1]).unwrap();
    assert_eq!(supercell.cells, vec![[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    let grid = supercell.grid(&geometry, 0.35 / BOHR).unwrap();
    // round(2.0 / 0.35) = 6 nodes per primitive cell.
    assert_eq!(grid.index.shape()[0], 18);
    assert!(grid.periodic);
    let k = supercell.kpoints();
    assert_eq!(k.len(), 3);
    assert!((k[1][0] - 1. / 3.).abs() < 1e-15 && (k[2][0] + 1. / 3.).abs() < 1e-15);
    // Refusals: no periodic axis, a mesh along a vacuum axis.
    let molecule = Geometry::new(
        vec![1],
        vec![[0.; 3]],
        [[2., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    assert!(Supercell::new(&molecule, [1, 1, 1]).is_err());
    assert!(Supercell::new(&primitive, [2, 2, 1]).is_err());
}

#[test]
fn the_hamiltonian_keeps_the_primitive_translation() {
    // Orbital samples, one-body matrix and Coulomb integrals are invariant under one primitive
    // translation: the grid is commensurate and the images are summed.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let integrals = supercell(&ctx, &chain(2.0), [2, 1, 1], 0.35);
    let m = integrals.m();
    assert_eq!(m, 2);
    let shape = integrals.grid.index.shape();
    let g = integrals.grid.index.points();
    let shift = shape[0] / 2;
    let mut row0 = vec![0.; g];
    let mut row1 = vec![0.; g];
    integrals.samples.widen_range(0, &mut row0).unwrap();
    integrals.samples.widen_range(g, &mut row1).unwrap();
    let plane = shape[1] * shape[2];
    let mut spread = 0f64;
    for k in 0..g {
        let moved = (k + shift * plane) % g;
        spread = spread.max((row0[k] - row1[moved]).abs());
    }
    assert!(spread < 1e-13, "image-summed samples differ by {spread:e}");
    let one = integrals.one.as_slice();
    assert!(
        (one[0] - one[3]).abs() < 1e-12,
        "on-site spread {:e}",
        one[0] - one[3]
    );
    let eri = integrals.eri().unwrap();
    let swap = [1, 0];
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                for s in 0..2 {
                    let a = eri.value(p, q, r, s);
                    let b = eri.value(swap[p], swap[q], swap[r], swap[s]);
                    assert!(
                        (a - b).abs() < 1e-10,
                        "ERI translation {p}{q}{r}{s}: {a} {b}"
                    );
                }
            }
        }
    }
    // The constant holds the Madelung term for the neutral cell, and a charged cell is refused.
    let v = madelung_constant(&ctx, integrals.geometry.cell()).unwrap();
    assert!((integrals.madelung - 0.5 * 2. * v).abs() < 1e-14);
    assert!(integrals.particles(1, 0., None).is_err());
    assert_eq!(integrals.particles(0, 0., None).unwrap(), [1, 1]);
}

#[test]
fn a_two_cell_mesh_is_the_one_cell_problem_of_the_doubled_cell() {
    // mesh (2,1,1) of a = 2 A and mesh (1,1,1) of a = 4 A with two atoms are the same crystal on
    // the same grid (0.25 A divides both cells), so their RHF energies agree to round-off.
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let mesh = supercell(&ctx, &chain(2.0), [2, 1, 1], 0.25);
    let doubled = Geometry::new(
        vec![1, 1],
        vec![[0.; 3], [2.0, 0., 0.]],
        [[4.0, 0., 0.], [0., 6., 0.], [0., 0., 6.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let single = supercell(&ctx, &doubled, [1, 1, 1], 0.25);
    assert_eq!(mesh.grid.index.shape(), single.grid.index.shape());
    let options = ScfOptions::default();
    let a = mesh.mean_field("rhf", [1, 1], &options).unwrap();
    let b = single.mean_field("rhf", [1, 1], &options).unwrap();
    let ea = a.electronic_energy + mesh.nuclear;
    let eb = b.electronic_energy + single.nuclear;
    assert!((ea - eb).abs() < 1e-9, "{ea} {eb}");
}

#[test]
fn unfolded_bands_at_the_mesh_points_are_the_supercell_spectrum() {
    // Born-von Karman identity: the supercell RHF eigenvalues are the union of the unfolded
    // bands over the mesh points. An H2 chain is a closed-shell insulator, so the restricted
    // reference keeps the translation symmetry.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let primitive = Geometry::new(
        vec![1, 1],
        vec![[0.; 3], [0.74, 0., 0.]],
        [[2.0, 0., 0.], [0., 6., 0.], [0., 0., 6.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let integrals = MolecularIntegrals::supercell(
        &ctx,
        &primitive,
        [3, 1, 1],
        r#"{"name":"HAO","virtual_orbitals":1}"#,
        0.4 / BOHR,
        None,
        Precision::Float64,
    )
    .unwrap();
    // A tight SCF: the reference's translation symmetry is only as good as its convergence.
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let reference = integrals.mean_field("rhf", [3, 3], &options).unwrap();
    let supercell = integrals.supercell.clone().unwrap();
    let bands = mean_field_bands(&integrals, &reference, &supercell.kpoints()).unwrap();
    let np = bands.n_primitive;
    assert_eq!(bands.spins, 1);
    let mut unfolded = bands.eigenvalues.clone();
    unfolded.sort_by(f64::total_cmp);
    let mut direct = reference.energies[0].clone();
    direct.sort_by(f64::total_cmp);
    assert_eq!(unfolded.len(), direct.len());
    for (a, b) in unfolded.iter().zip(&direct) {
        // Measured: 3e-9 Ha on a 5.2 Ha virtual; the 2s functions make the overlap
        // ill-conditioned and S^{1/2}, S(k)^{-1/2} amplify its round-off.
        assert!((a - b).abs() < 1e-8, "unfolded {a} vs supercell {b}");
    }
    // Every band's Lowdin weights sum to one.
    for chunk in bands.weights.chunks(np) {
        assert!((chunk.iter().sum::<f64>() - 1.).abs() < 1e-10);
    }
}

#[test]
fn unfolding_a_nearest_neighbor_chain_gives_the_analytic_band_everywhere() {
    // A synthetic tight-binding chain of 8 cells, one orbital each: F_{R,R+-1} = t, S = 1.
    // Its hopping is shorter than half the supercell, so the interpolated band is exactly
    // eps(k) = e0 + 2 t cos(2 pi k) at every k, on and off the mesh.
    let primitive = Geometry::new(
        vec![1],
        vec![[0.; 3]],
        [[1.0, 0., 0.], [0., 10., 0.], [0., 0., 10.]],
        [true, false, false],
        LengthUnit::Bohr,
    )
    .unwrap();
    let n = 8;
    let (supercell, geometry) = Supercell::new(&primitive, [n, 1, 1]).unwrap();
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let basis = mandacaru_core::basis::Basis::build(&ctx, &geometry, r#"{"name":"HAO"}"#).unwrap();
    let cells = supercell.orbital_cells(&basis).unwrap();
    let centers: Vec<_> = basis.orbitals.iter().map(|o| o.center).collect();
    let (e0, t) = (-0.3, -0.1);
    let overlap = DMatrix::<f64>::identity(n, n);
    let fock = DMatrix::from_fn(n, n, |i, j| {
        let d = (i as i64 - j as i64).rem_euclid(n as i64);
        if d == 0 {
            e0
        } else if d == 1 || d == n as i64 - 1 {
            t
        } else {
            0.
        }
    });
    let k: Vec<[f64; 3]> = (0..17)
        .map(|i| [-0.5 + i as f64 / 16. + 0.013, 0., 0.])
        .collect();
    let bands = supercell
        .unfold(&cells, &centers, &overlap, &[fock], &k)
        .unwrap();
    for (point, value) in k.iter().zip(&bands.eigenvalues) {
        let exact = e0 + 2. * t * (2. * std::f64::consts::PI * point[0]).cos();
        assert!(
            (value - exact).abs() < 1e-13,
            "k = {}: {value} vs {exact}",
            point[0]
        );
    }
    // The Bloch operator of a single-orbital chain is the normalized plane wave.
    let orthonormal = DMatrix::<C64>::identity(n, n);
    let w = supercell
        .bloch_operators(&cells, &orthonormal, &[[0.25, 0., 0.]])
        .unwrap();
    let norm: f64 = w[0].iter().map(|v| v.norm_sqr()).sum();
    assert!((norm - 1.).abs() < 1e-14);
}

#[test]
fn a_metallic_chain_reaches_the_rhf_minimum_not_the_level_shifted_saddle() {
    // Four cells of the a = 2 A chain: the k = +-1/4 pair is degenerate at the Fermi level. The
    // level-shifted SCF used to stop on a symmetric, non-aufbau stationary point (-9.321676 eV per
    // cell, the occupied orbital 0.03 Ha above an empty one) and call it converged; the minimum
    // mixes the pair and lies 0.46 eV per cell lower (-9.781445 eV, an independent dense RHF from
    // the same integrals, 2026-11-09). The aufbau gate restarts from the saddle with the pair
    // mixed. Re-pinned 2026-11-10 for the band-limited supercell Hamiltonian (orbitals truncated
    // at pi / (2h), G-space ion potential): -9.781445 -> -11.114980 eV per cell at h = 0.35 A,
    // still the aufbau minimum (occupied below empty, checked below).
    const HARTREE_EV: f64 = 27.211386245988;
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    // vera's cell: 10 A of vacuum transversely.
    let primitive = Geometry::new(
        vec![1],
        vec![[0.; 3]],
        [[2.0, 0., 0.], [0., 10., 0.], [0., 0., 10.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let integrals = supercell(&ctx, &primitive, [4, 1, 1], 0.35);
    let reference = integrals
        .mean_field("rhf", [2, 2], &ScfOptions::default())
        .unwrap();
    let per_cell = (reference.electronic_energy + integrals.nuclear) / 4. * HARTREE_EV;
    assert!(reference.converged);
    let levels = &reference.energies[0];
    assert!(levels[1] < levels[2], "not aufbau: {levels:?}");
    assert!(
        (per_cell + 11.114980).abs() < 1e-4,
        "{per_cell} eV per cell"
    );
}

/// RHF energy per cell (Hartree) of the H2 chain (a = 3 A, d = 0.74 A, 8 A vacuum, 2 cells) with
/// every atom shifted by `shift` Angstrom along the chain.
fn h2_chain_energy(ctx: &Context, h: f64, shift: f64) -> f64 {
    let primitive = Geometry::new(
        vec![1, 1],
        vec![[shift, 0., 0.], [0.74 + shift, 0., 0.]],
        [[3.0, 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [true, false, false],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let integrals = supercell(ctx, &primitive, [2, 1, 1], h);
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let rhf = integrals.mean_field("rhf", [2, 2], &options).unwrap();
    (rhf.electronic_energy + integrals.nuclear) / 2.
}

#[test]
fn the_energy_does_not_depend_on_where_the_crystal_sits_on_the_grid() {
    // Band-limited orbitals and a G-space ion potential: a rigid translation by a fraction of a
    // grid step changes only phases. Before (point-sampled cusp, softened -Z/r), the same chain
    // moved by 6 meV per cell at 0.001 A and by up to 0.65 eV at 0.1 A (grid_spacing 0.2).
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    for h in [0.25, 0.15] {
        let reference = h2_chain_energy(&ctx, h, 0.);
        for fraction in [0.001 / h, 0.25, 0.5, 0.731] {
            let moved = h2_chain_energy(&ctx, h, fraction * h);
            assert!(
                (moved - reference).abs() < 1e-9,
                "h = {h}, shift {:.4} A: {:e} Ha",
                fraction * h,
                moved - reference
            );
        }
    }
}

#[test]
fn the_energy_converges_monotonically_with_the_grid() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let energies: Vec<f64> = [0.3, 0.25, 0.2, 0.15]
        .iter()
        .map(|&h| h2_chain_energy(&ctx, h, 0.))
        .collect();
    for pair in energies.windows(2) {
        assert!(pair[1] < pair[0], "not monotone: {energies:?}");
    }
    // Successive differences shrink as the band limit grows.
    let steps: Vec<f64> = energies.windows(2).map(|p| p[0] - p[1]).collect();
    assert!(steps[2] < steps[0], "not converging: {energies:?}");
}

#[test]
fn a_band_limited_orbital_peaks_at_its_center() {
    // The Fourier synthesis puts each orbital where its atom is (a sign error would mirror it).
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let primitive = Geometry::new(
        vec![1],
        vec![[1.2, 2.0, 2.5]],
        [[4.0, 0., 0.], [0., 5., 0.], [0., 0., 6.]],
        [true, true, true],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let integrals = supercell(&ctx, &primitive, [1, 1, 1], 0.2);
    let g = integrals.grid.index.points();
    let mut row = vec![0.; g];
    integrals.samples.widen_range(0, &mut row).unwrap();
    let peak = (0..g).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
    let point = integrals.grid.point(peak);
    let center = integrals.basis.orbitals[0].center;
    let distance = (0..3)
        .map(|j| (point[j] - center[j]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(distance < 0.2 / BOHR, "peak {point:?} vs center {center:?}");
}
