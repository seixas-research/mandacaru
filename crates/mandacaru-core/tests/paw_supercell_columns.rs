//! The streamed Cholesky columns of the PAW-LCAO supercell's two-body source against the stored
//! pair densities (`BLOCH_VQE_PLAN.md` 4.5): `BasisCoulomb` forms each column from the `n`
//! functions of the supercell, `StoredColumns` reads it from the `n^2` held transforms of their
//! products ([`mandacaru_core::periodic::wannier::downfold::pairs`]). The diagonal, every column
//! and the Cholesky-factored integrals agree to rounding. The factors the Hamiltonian stores per
//! translation orbit (`cholesky_by_translation_orbits`) reproduce every pair integral to the
//! residual they report, against the exact columns. Needs `MANDACARU_PAW_PATH`.
mod paw_supercell_support;
mod setup_shells_support;

use paw_supercell_support::{
    cells_of, cells_with_basis, hydrogen_chain_cell, lithium_hydride_cell, nitrogen_chain_cell,
    silicon_cell, Primitive,
};

use mandacaru_core::{
    molecular::{
        integrals::{cholesky_from_columns, Eri, PairColumns},
        periodic::paw::PawSupercell,
        MolecularIntegrals,
    },
    periodic::wannier::downfold::basis::{BasisCoulomb, StoredColumns},
    Context,
};
use std::sync::Arc;

/// The largest `|streamed - stored|` over the diagonal, the columns of every `stride`-th pivot
/// and (with `factored`) every Cholesky-factored integral, and the largest diagonal entry. The
/// stored columns cost a pass over every pair's compensation charge, so a larger cell samples.
fn compare(
    name: &str,
    cell: &Primitive,
    mesh: [usize; 3],
    stride: usize,
    factored: bool,
) -> (f64, f64) {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let setup = Arc::new(cells_of(&ctx, "lda-sr", false, cell, 1, [0.; 3]));
    let integrals = MolecularIntegrals::paw_supercell(&ctx, setup.clone(), mesh, None).unwrap();
    let cells = PawSupercell::new(&ctx, &setup, mesh).unwrap();
    let args = (
        &cells.crystal,
        mesh,
        &cells.points[..],
        &cells.harmonics,
        &integrals.lowdin,
    );
    let streamed = BasisCoulomb::new(args.0, args.1, args.2, args.3, args.4).unwrap();
    let stored = StoredColumns::new(args.0, args.1, args.2, args.3, args.4).unwrap();
    let m = streamed.m();
    let pairs = m * (m + 1) / 2;
    let (a, b) = (streamed.diagonal().unwrap(), stored.diagonal().unwrap());
    let largest = b.iter().fold(0f64, |x, v| x.max(v.abs()));
    let mut columns = a
        .iter()
        .zip(&b)
        .fold(0f64, |x, (u, v)| x.max((u - v).abs()));
    let diagonal = columns;
    for pivot in (0..pairs).step_by(stride) {
        let (a, b) = (
            streamed.column(pivot).unwrap(),
            stored.column(pivot).unwrap(),
        );
        columns = columns.max(
            a.iter()
                .zip(&b)
                .fold(0f64, |x, (u, v)| x.max((u - v).abs())),
        );
    }
    let mut energies = 0f64;
    let mut ranks = (None, None);
    if factored {
        // The stored columns are factored with the streamed pivots: with free pivoting the
        // translation-degenerate diagonals of a mesh could let rounding pick different pivots.
        let fast = cholesky_from_columns(&ctx, &streamed, 1e-10, None).unwrap();
        let slow = cholesky_from_columns(&ctx, &stored, 1e-10, fast.pivots()).unwrap();
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        energies =
                            energies.max((fast.value(p, q, r, s) - slow.value(p, q, r, s)).abs());
                    }
                }
            }
        }
        ranks = (
            fast.report().cholesky.map(|c| c.0),
            slow.report().cholesky.map(|c| c.0),
        );
    }
    println!(
        "{name} {mesh:?}: m {m}, largest (pr|pr) {largest:.3e}; max |streamed - stored| diagonal {diagonal:.2e}, \
         columns {columns:.2e}, Cholesky integrals {energies:.2e} (ranks {:?}); metric - stored {:.2e}; \
         departures {:?} {:?}",
        ranks,
        (&streamed.metric - &stored.metric).norm(),
        streamed.departures(),
        stored.departures()
    );
    (columns.max(energies), largest)
}

/// Agreement to rounding of Hartree-scale integrals.
const TOLERANCE: f64 = 1e-11;

#[test]
fn the_streamed_columns_are_the_stored_ones_for_a_hydrogen_chain() {
    let (gap, scale) = compare("H2 chain", &hydrogen_chain_cell(), [2, 1, 1], 1, true);
    assert!(gap < TOLERANCE * scale.max(1.), "{gap:e}");
}

#[test]
fn the_streamed_columns_are_the_stored_ones_for_lithium_hydride() {
    let (gap, scale) = compare("LiH", &lithium_hydride_cell(), [2, 1, 1], 1, true);
    assert!(gap < TOLERANCE * scale.max(1.), "{gap:e}");
}

/// p functions: the compensation charges of every `L` and the self-energy tails (every eleventh
/// column; all 136 columns and the factored integrals agree to 3e-15 and 6e-15 Ha, measured).
#[test]
fn the_streamed_columns_are_the_stored_ones_for_a_nitrogen_chain() {
    let (gap, scale) = compare("N2 chain", &nitrogen_chain_cell(), [2, 1, 1], 11, false);
    assert!(gap < TOLERANCE * scale.max(1.), "{gap:e}");
}

/// The Hamiltonian's factors, stored once per translation orbit of pivots, against the exact
/// pair matrix (every `stride`-th column read from the streamed source): `|V_ab - sum_P L^P_a
/// L^P_b| <= sqrt(R_aa R_bb)` for the positive semidefinite residual `R`, so every entry misses
/// by at most the reported largest residual diagonal (plus the translation defect of the grid,
/// 3e-13 measured on Si SZ (2, 2, 1)). Also: the stored vectors are about the rank over the
/// cells; the plain pivoted Cholesky's rank at the same tolerance is within 5 %; the replay of
/// the representative pivots and kept counts at the same geometry gives the same integrals.
/// Returns (worst miss over the reported residual, rank, plain rank, stored, cells, orbits
/// whose block dropped eigenvalues).
fn orbits(
    name: &str,
    cell: &Primitive,
    basis: &str,
    mesh: [usize; 3],
    tolerance: f64,
    stride: usize,
) -> (f64, usize, usize, usize, usize, usize) {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let setup = Arc::new(cells_with_basis(
        &ctx, "lda-sr", false, cell, 1, [0.; 3], basis,
    ));
    let integrals =
        MolecularIntegrals::paw_supercell(&ctx, setup.clone(), mesh, Some(tolerance)).unwrap();
    let eri = integrals.eri().unwrap();
    let Eri::Augmented { base, .. } = eri.as_ref() else {
        panic!("a PAW supercell is augmented")
    };
    let Eri::Translated {
        stored,
        residual,
        kept,
        ..
    } = base.as_ref()
    else {
        panic!("stored per orbit")
    };
    let (stored, residual) = (*stored, residual.unwrap());
    let rank = base.report().cholesky.unwrap().0;
    let factors = base.factor_count().unwrap();
    let truncated = kept.len()
        - kept
            .iter()
            .zip(0..)
            .filter(|&(&k, o)| k == base_orbit_size(base, o))
            .count();
    let cells = mesh.iter().product::<usize>();
    let source = integrals.paw_supercell.as_ref().unwrap();
    let m = integrals.m();
    let pairs = m * (m + 1) / 2;
    let index: Vec<(usize, usize)> = (0..m).flat_map(|p| (0..=p).map(move |r| (p, r))).collect();
    let mut miss = 0f64;
    for b in (0..pairs).step_by(stride) {
        let exact = source.columns.column(b).unwrap();
        let (q, s) = index[b];
        for (a, &(p, r)) in index.iter().enumerate() {
            miss = miss.max((exact[a] - base.value(p, q, r, s)).abs());
        }
    }
    let plain = cholesky_from_columns(&ctx, source.columns.as_ref(), tolerance, None).unwrap();
    let plain_rank = plain.factor_count().unwrap();
    let mut replayed = 0f64;
    let geometry = integrals.geometry.clone();
    let again = integrals.at_geometry_with_fixed_pivots(geometry).unwrap();
    let again = again.eri().unwrap();
    for &(p, r) in index.iter().step_by(stride) {
        for &(q, s) in &index {
            replayed = replayed.max((again.value(p, q, r, s) - eri.value(p, q, r, s)).abs());
        }
    }
    println!(
        "{name} {basis} {mesh:?} at {tolerance:e}: m {m}, pairs {pairs}, rank {rank} (plain pivoted {plain_rank}), \
         {factors} factors, stored {stored} over {cells} cells, {truncated} orbits truncated; reported residual \
         {residual:.3e}, worst |V - LL| {miss:.3e} (every {stride}th column); fixed-pivot replay {replayed:.2e}"
    );
    assert!(replayed < 1e-12, "replay {replayed:e}");
    (miss - residual, rank, plain_rank, stored, cells, truncated)
}

/// How many factors stored vector `o` of `eri` carries (the size of its pivot's orbit).
fn base_orbit_size(eri: &Eri, o: usize) -> usize {
    let Eri::Translated { columns, .. } = eri else {
        unreachable!()
    };
    columns.iter().filter(|&&(v, _)| v == o).count()
}

/// The translation defect of the grid, an upper bound measured with margin (3.3e-13).
const DEFECT: f64 = 1e-12;

#[test]
fn the_factors_stored_per_orbit_give_every_pair_integral_of_lithium_hydride() {
    let (over, rank, plain, stored, cells, _) =
        orbits("LiH", &lithium_hydride_cell(), "SZ", [3, 1, 1], 1e-10, 1);
    assert!(over < DEFECT, "{over:e}");
    assert!(
        rank * 20 < plain * 21 && plain * 20 < rank * 21,
        "rank {rank} plain {plain}"
    );
    assert!(stored * cells < rank * 2, "stored {stored}");
}

#[test]
fn the_factors_stored_per_orbit_give_every_pair_integral_of_silicon_four_cells() {
    let (over, rank, plain, stored, cells, _) =
        orbits("Si", &silicon_cell(), "SZ", [2, 2, 1], 1e-10, 7);
    assert!(over < DEFECT, "{over:e}");
    assert!(
        rank * 20 < plain * 21 && plain * 20 < rank * 21,
        "rank {rank} plain {plain}"
    );
    assert!(stored * cells < rank * 2, "stored {stored}");
}

/// Si DZP (2, 1, 1) at 1e-8, where blocks drop eigenvalues (the pair matrix is 79 % of full rank
/// there): the same bound holds with the truncated orbits.
#[test]
fn the_factors_stored_per_orbit_give_every_pair_integral_of_silicon_dzp_when_blocks_truncate() {
    let (over, rank, plain, stored, cells, truncated) =
        orbits("Si", &silicon_cell(), "DZP", [2, 1, 1], 1e-8, 13);
    assert!(truncated > 0);
    assert!(over < DEFECT, "{over:e}");
    assert!(
        rank * 20 < plain * 21 && plain * 20 < rank * 21,
        "rank {rank} plain {plain}"
    );
    assert!(stored * cells < rank * 2, "stored {stored}");
}
