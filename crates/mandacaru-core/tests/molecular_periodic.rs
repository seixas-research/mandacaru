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
    // The MP2 shift is v_M of the supercell's lattice: the doubled cell's, as the energy's constant.
    let v = mesh.exchange_madelung().unwrap();
    assert!((v - single.exchange_madelung().unwrap()).abs() < 1e-12);
    assert!(
        (v - 2. * mesh.madelung / mesh.neutral_electrons() as f64).abs() < 1e-12,
        "{v}"
    );
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

/// `a + c / L + b / L^3` by least squares on `(L, y)` rows: `(a, c, b)`.
fn inverse_power_fit(rows: &[(f64, f64)]) -> (f64, f64, f64) {
    let design = DMatrix::from_fn(rows.len(), 3, |i, j| rows[i].0.powi(-[0, 1, 3][j]));
    let values = nalgebra::DVector::from_iterator(rows.len(), rows.iter().map(|r| r.1));
    let normal = design.transpose() * &design;
    let x = normal.lu().solve(&(design.transpose() * values)).unwrap();
    (x[0], x[1], x[2])
}

/// The HAO supercell's MP2 denominators take the exchange Madelung shift (vera's ruling
/// 2026-11-15, 2e; self-referenced: the isolated HAO Hamiltonian on the same grid is another
/// Hamiltonian at finite grid_spacing, band-limited orbitals and a Fourier ion potential against
/// sampled cusps, so it is not the supercell's limit). H2 in cubic periodic boxes of side
/// `n x grid_spacing` (0.3 A, `n` even from 24 to 60, ten boxes, one offset), mesh 1, L in Bohr:
/// (1) the premise: the raw RHF HOMO fitted to `a + c / L + b / L^3` carries `c = -v_M L =
/// 2.837297` (the Madelung term of the exchange's dropped `G = 0`) to 3 %; (2) the
/// implementation: the 1/L coefficient of the HOMO's MP2 natural occupation with the shift
/// (`natural_orbitals` with `exchange_madelung()`) is below 10 % of the unshifted one's (no
/// shift: 100 %, half the shift about 50 %, a shift on the virtuals too: 100 %); (3) the
/// supercell is unchanged under a rigid shift of half a grid step, to 1e-7 Ha.
#[test]
#[ignore = "slow: about two minutes; run: cargo test --profile release-test -p mandacaru-core --test molecular_periodic -- --ignored the_hao_supercell_mp2 --nocapture"]
fn the_hao_supercell_mp2_denominators_take_the_exchange_madelung_shift() {
    use mandacaru_core::molecular::correlation::natural_orbitals;
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let spacing = 0.3;
    let options = ScfOptions {
        tolerance: 1e-11,
        ..ScfOptions::default()
    };
    let h2 = |side: f64, offset: f64| {
        let cell = [[side, 0., 0.], [0., side, 0.], [0., 0., side]];
        let center = side / 2. + offset;
        let positions = vec![
            [center - 0.37, center, center],
            [center + 0.37, center, center],
        ];
        Geometry::new(vec![1, 1], positions, cell, [true; 3], LengthUnit::Angstrom).unwrap()
    };
    let (mut levels, mut raw, mut shifted) = (Vec::new(), Vec::new(), Vec::new());
    for nodes in (24..=60).step_by(4) {
        let side = nodes as f64 * spacing;
        let crystal = supercell(&ctx, &h2(side, 0.), [1, 1, 1], spacing);
        assert_eq!(
            crystal.grid.index.shape(),
            [nodes; 3],
            "the box is n x grid_spacing"
        );
        let rhf = crystal.mean_field("rhf", [1, 1], &options).unwrap();
        assert!(rhf.converged);
        let model = rhf.model(&crystal).unwrap();
        let v = crystal.exchange_madelung().unwrap();
        let occupation = |shift: f64| {
            natural_orbitals(
                &ctx,
                model.m,
                model.one.as_slice(),
                model.two.as_slice(),
                1,
                shift,
            )
            .unwrap()
            .occupied_occupations
            .as_slice()[0]
        };
        let (o0, o1) = (occupation(0.), occupation(v));
        let l = side / BOHR;
        println!("H2 HAO n {nodes} (L {side:.1} A): v_M {v:.6} Ha (-2.837297 / L {:.6}); HOMO {:.8} Ha; HOMO occupation no shift {o0:.8}, shifted {o1:.8}", -2.837297479 / l, rhf.energies[0][0]);
        levels.push((l, rhf.energies[0][0]));
        raw.push((l, o0));
        shifted.push((l, o1));
    }
    let (_, c_level, b_level) = inverse_power_fit(&levels);
    let (_, c_raw, _) = inverse_power_fit(&raw);
    let (_, c_shifted, _) = inverse_power_fit(&shifted);
    println!(
        "H2 HAO: HOMO 1/L coefficient {c_level:.4} (2.837297, {:+.2} %), L^-3 {b_level:.3} Ha Bohr^3; occupation 1/L coefficient no shift {c_raw:.4e}, shifted {c_shifted:.4e} ({:.1} % of it)",
        100. * (c_level / 2.837297479 - 1.),
        100. * (c_shifted / c_raw).abs()
    );
    // The rigid half-step shift of the smallest box.
    let side = 24. * spacing;
    let energy = |offset: f64| {
        let crystal = supercell(&ctx, &h2(side, offset), [1, 1, 1], spacing);
        let rhf = crystal.mean_field("rhf", [1, 1], &options).unwrap();
        assert!(rhf.converged);
        rhf.electronic_energy + crystal.nuclear
    };
    let moved = (energy(0.5 * spacing) - energy(0.)).abs();
    println!("H2 HAO: RHF energy change under a half-step rigid shift {moved:.2e} Ha");
    assert!(
        (c_level - 2.837297479).abs() < 0.03 * 2.837297479,
        "{c_level}"
    );
    assert!(
        c_shifted.abs() < 0.1 * c_raw.abs(),
        "{c_shifted:e} {c_raw:e}"
    );
    assert!(moved < 1e-7, "{moved:e}");
}

/// MG-REST item 9 (measurement): the box scatter of the HAO supercell's one-body integrals. H2 in
/// cubic boxes of side `n x grid_spacing` (`n` from 24 to 60 step 4, at 0.3 and 0.2 A, one offset):
/// the 1s kinetic and overlap diagonals, which an isolated molecule's box leaves constant once the
/// tails fit, against their mean over the boxes of 12 A and more, and the RHF HOMO's residual from
/// `a + c / L + b / L^3`. Alongside, the lattice sum the band limit performs on a model (a Slater
/// 1s, `zeta = 1.2`, on the same boxes): `sum_G |phi(G)|^2 w(G)^2 / Omega` and its kinetic
/// counterpart with the sharp cutoff `w = 1` below `k_c` (`molecular::periodic::sample_images`)
/// and with cosine-squared tapers from `0.9`, `0.8`, `0.7` and `0.5 k_c`, each against its own limit (the
/// radial integral). Measured 2026-11-15 (DECISIONS, MG-REST item 9): the kinetic diagonal
/// departs from its mean by 3.1e-4 Ha at 0.3 A and 8.2e-5 at 0.2 A; on the model the sharp edge
/// scatters the kinetic sum by 1.6e-3 and 4.3e-4, a taper from 0.9 k_c by 1.5e-4 and 1.7e-5 but
/// lowers the kinetic limit by 2.1e-2 and 9.0e-3 Ha (from 0.7 k_c: 6.9e-5, 4.4e-6; 7.5e-2,
/// 3.5e-2): the taper's bias exceeds the scatter it removes, so the basis keeps its sharp band
/// limit. Gates the measured floor of the kinetic
/// scatter (1e-3 and 2e-4 Ha).
#[test]
fn the_box_scatter_of_the_hao_supercell_is_measured_against_its_band_limit() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let options = ScfOptions {
        tolerance: 1e-11,
        ..ScfOptions::default()
    };
    for spacing in [0.3, 0.2] {
        let (mut kinetic, mut overlap, mut levels) = (Vec::new(), Vec::new(), Vec::new());
        let nodes: Vec<usize> = if spacing > 0.25 {
            (24..=60).step_by(4).collect()
        } else {
            (36..=84).step_by(6).collect()
        };
        for &n in &nodes {
            let side = n as f64 * spacing;
            let cell = [[side, 0., 0.], [0., side, 0.], [0., 0., side]];
            let c = side / 2.;
            let geometry = Geometry::new(
                vec![1, 1],
                vec![[c - 0.37, c, c], [c + 0.37, c, c]],
                cell,
                [true; 3],
                LengthUnit::Angstrom,
            )
            .unwrap();
            let crystal = supercell(&ctx, &geometry, [1, 1, 1], spacing);
            let m = crystal.m();
            let rhf = crystal.mean_field("rhf", [1, 1], &options).unwrap();
            assert!(rhf.converged);
            let l = side / BOHR;
            let (t, s) = (crystal.kinetic.as_slice()[0], crystal.overlap.as_slice()[0]);
            println!("HAO H2 {spacing} A n {n} (L {side:.1} A): <1s|T|1s> {t:.9} <1s|1s> {s:.9} HOMO {:.8} Ha ({m} orbitals)", rhf.energies[0][0]);
            kinetic.push((side, t));
            overlap.push((side, s));
            levels.push((l, rhf.energies[0][0]));
        }
        let spread = |rows: &[(f64, f64)]| {
            let tail: Vec<f64> = rows.iter().filter(|r| r.0 >= 12.).map(|r| r.1).collect();
            let mean = tail.iter().sum::<f64>() / tail.len() as f64;
            tail.iter().fold(0f64, |w, v| w.max((v - mean).abs()))
        };
        let (a, c, b) = inverse_power_fit(&levels);
        let residual = levels.iter().fold(0f64, |w, &(l, e)| {
            w.max((e - a - c / l - b / l.powi(3)).abs())
        });
        println!("HAO H2 {spacing} A: boxes of 12 A and more, largest departure from the mean: kinetic {:.2e} Ha, overlap {:.2e}; HOMO residual of a + c/L + b/L^3 {residual:.2e} Ha", spread(&kinetic), spread(&overlap));
        assert!(
            spread(&kinetic) < if spacing > 0.25 { 1e-3 } else { 2e-4 },
            "{spacing}: {:e}",
            spread(&kinetic)
        );
        // The model lattice sum on the same boxes.
        let zeta: f64 = 1.2;
        let phi2 =
            |k: f64| 64. * std::f64::consts::PI * zeta.powi(5) / (zeta * zeta + k * k).powi(4);
        let grid_spacing = spacing / BOHR;
        let cutoff = std::f64::consts::PI / (2. * grid_spacing);
        for start in [1.0, 0.9, 0.8, 0.7, 0.5] {
            let w = |k: f64| {
                if k >= cutoff {
                    0.
                } else if k <= start * cutoff || start >= 1. {
                    1.
                } else {
                    (0.5 * std::f64::consts::PI * (k / cutoff - start) / (1. - start))
                        .cos()
                        .powi(2)
                }
            };
            // The limit: (1 / 2 pi^2) int k^2 |phi|^2 w^2 dk (norm) and with k^2 / 2 (kinetic).
            let steps = 200_000;
            let dk = cutoff / steps as f64;
            let (mut norm_limit, mut kinetic_limit) = (0., 0.);
            for i in 0..steps {
                let k = (i as f64 + 0.5) * dk;
                let f = k * k * phi2(k) * w(k).powi(2) * dk / (2. * std::f64::consts::PI.powi(2));
                norm_limit += f;
                kinetic_limit += 0.5 * k * k * f;
            }
            let (mut norms, mut kinetics) = (Vec::new(), Vec::new());
            for &n in &nodes {
                let l = n as f64 * grid_spacing;
                let g = 2. * std::f64::consts::PI / l;
                let reach = (cutoff / g).ceil() as i64;
                let (mut norm, mut kin) = (0., 0.);
                for x in -reach..=reach {
                    for y in -reach..=reach {
                        for z in -reach..=reach {
                            let k = g * ((x * x + y * y + z * z) as f64).sqrt();
                            let weight = w(k).powi(2) * phi2(k) / l.powi(3);
                            norm += weight;
                            kin += 0.5 * k * k * weight;
                        }
                    }
                }
                norms.push((n as f64 * spacing, norm - norm_limit));
                kinetics.push((n as f64 * spacing, kin - kinetic_limit));
            }
            let largest = |rows: &[(f64, f64)]| {
                rows.iter()
                    .filter(|r| r.0 >= 12.)
                    .fold(0f64, |w, r| w.max(r.1.abs()))
            };
            println!(
                "model 1s {spacing} A taper from {start} k_c: norm limit {norm_limit:.6} (sharp limit loses {:.2e}), kinetic limit {kinetic_limit:.6}; lattice sum minus limit on boxes of 12 A and more: norm {:.2e}, kinetic {:.2e} Ha",
                1. - norm_limit, largest(&norms), largest(&kinetics)
            );
        }
    }
}
