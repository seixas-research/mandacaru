//! The j-resolved spin-orbit term of a Dirac PAW-LCAO dataset. Atomic units.
//!
//! A Dirac dataset is **scalar-relativistic plus a spin-orbit term**, not a j-resolved
//! augmentation sphere. The PAW-LCAO overlap operator is `1 + sum |p~> q <p~|`, so a j-dependent
//! `q` would give the metric of the generalized eigenproblem an `L.S` structure and every consumer
//! of the overlap would have to learn about spin. The channels of the dataset therefore stay
//! scalar (the j average, which is what the Koelling-Harmon equation solves) and the spin-orbit
//! coupling is a one-center difference in the Hamiltonian, built the way `D` and `q` are.
//!
//! For every channel `l >= 1` (empty `extra_l` channels included) two **unitary** branches, one
//! per `j = l -/+ 1/2` (`kappa = l`, `-(l+1)`), are pseudized in the same screened local
//! potential: references from the Dirac atom with the branch's own `kappa`, Bessel expansions with
//! no norm deficit, projectors and one-center matrices as for a scalar channel. Any j-dependent
//! separable operator is two terms, `V_j = V_avg + V_SO L.S`, because `L.S` is `l/2` on
//! `j = l + 1/2` and `-(l+1)/2` on `j = l - 1/2`. Both are built on the union of the two branches'
//! projectors: the average with the `(2j+1)` weights and the difference scaled by `2/(2l+1)`, so
//! each `j` is recovered exactly ([`recover`]). The branches are unitary so that their overlap
//! corrections stay of order 1e-4 and the metric stays spin-free.
//!
//! The one-center exchange of the hybrid path reads the scalar partial waves while this term
//! carries j-resolved radial functions: a model error of the design, not of this port.
use super::{
    assemble::{assemble, Atomic, Partial},
    recipe::{Cutoffs, Recipe},
    reference::{branch_equation, reference_waves_with},
    shells::Shells,
};
use crate::{
    models::atom::Atom,
    paw::{
        invalid,
        pseudize::{optimize, Equation, NormCondition, PseudoInput},
        pseudize_tables::resample,
        record::{Array, SpinOrbit},
        spectrum::{self, Channel},
    },
    product, Buffer, Context, Result,
};
use nalgebra::DMatrix;
use std::collections::BTreeMap;

/// Spacing (Bohr) of the uniform grid the level of a branch is solved on.
pub const LEVEL_SPACING: f64 = 0.004;
/// Extent (Bohr) of that grid; the nodes are `k * LEVEL_SPACING` for `k` in `1..EXTENT / SPACING`.
pub const LEVEL_EXTENT: f64 = 22.0;

/// What the spin-orbit construction needs of the generation that has just made the scalar
/// channels.
pub struct Inputs<'a> {
    pub recipe: &'a Recipe,
    pub atom: &'a Atom,
    pub shells: &'a Shells,
    /// The augmentation radius of every channel, extra ones included.
    pub radii: &'a BTreeMap<usize, f64>,
    pub extra_l: usize,
    /// The first reference energy the empty channels ended on (see `Generated::anchor`).
    pub anchor: Option<f64>,
    /// The screened local potential and its radius.
    pub v_loc: &'a [f64],
    pub r_local: f64,
    /// `int v_H[n~ + n^] g`, the Hartree screening the ionic couplings are unscreened with.
    pub hartree_screening: f64,
}

/// One `j` branch of a channel.
#[derive(Debug)]
pub struct Branch {
    /// The dual projectors, one radial table per reference.
    pub projectors: Vec<Buffer<f64>>,
    /// `D^scr`, the screened coupling.
    pub screened: DMatrix<f64>,
    /// `D^scr - q int v_H g`, the unscreened (ionic) coupling.
    pub ionic: DMatrix<f64>,
    /// `q`, the overlap correction.
    pub overlap: DMatrix<f64>,
    /// The energies of the references.
    pub energies: Vec<f64>,
    /// The branch's lowest generalized level less its first reference energy (Hartree).
    pub error: f64,
}

/// The `kappa` of the two branches of `l`: `j = l - 1/2`, then `j = l + 1/2`.
pub fn kappas(l: usize) -> [i32; 2] {
    [l as i32, -(l as i32 + 1)]
}

/// `(2j + 1)` of the two branches: the weights of the average.
pub fn weights(l: usize) -> [f64; 2] {
    [2. * l as f64, 2. * l as f64 + 2.]
}

/// The recipe whose cutoffs are the radii the scalar construction ended on (a branch pseudizes
/// in the sphere of its channel).
fn pinned(input: &Inputs<'_>) -> Recipe {
    let mut recipe = input.recipe.clone();
    recipe.cutoffs = Cutoffs::PerChannel(input.radii.clone());
    recipe
}

fn rows(m: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)]);
        }
    }
    out
}

/// The lowest level of the generalized radial problem of channel `l`: `T_l + v~ +
/// sum |p~> D <p~|` against `1 + sum |p~> q <p~|` for the dual `projectors` (radial tables on `r`)
/// and the row-major `rank x rank` matrices `coupling` and `overlap`, on a uniform grid of
/// [`LEVEL_SPACING`], one grid, no extrapolation.
pub fn lowest_level(
    ctx: &Context,
    r: &[f64],
    v_loc: &[f64],
    l: usize,
    projectors: &[&[f64]],
    coupling: &[f64],
    overlap: &[f64],
) -> Result<f64> {
    let count = (LEVEL_EXTENT / LEVEL_SPACING) as usize;
    let grid = (1..count)
        .map(|k| k as f64 * LEVEL_SPACING)
        .collect::<Vec<_>>();
    let centrifugal = (l * (l + 1)) as f64;
    let v_grid = resample(ctx, r, v_loc, &grid)?;
    let potential = grid
        .iter()
        .zip(&v_grid)
        .map(|(x, v)| v + centrifugal / (2.0 * x * x))
        .collect::<Vec<_>>();
    let mut rows_u = Vec::with_capacity(projectors.len() * grid.len());
    for table in projectors {
        let on_grid = resample(ctx, r, table, &grid)?;
        rows_u.extend(on_grid.iter().zip(&grid).map(|(p, x)| p * x));
    }
    Ok(spectrum::eigenpair(
        ctx,
        Channel {
            radii: &grid,
            potential: &potential,
            projectors: &rows_u,
            coupling,
            overlap,
            rank: projectors.len(),
        },
        0,
        false,
    )?
    .energy)
}

/// The branch of channel `l` whose radial equation, for every channel, is `equations(l)`: the
/// references of the Dirac atom, their unitary Bessel expansions and the channel assembled in the
/// screened local potential. The production branches are [`channels`]; a caller passing other
/// equations builds a different construction (the controls of the tests do).
pub fn branch(
    ctx: &Context,
    input: &Inputs<'_>,
    l: usize,
    equations: &dyn Fn(usize) -> Equation,
) -> Result<Branch> {
    let recipe = pinned(input);
    let (references, _notes) = reference_waves_with(
        ctx,
        &recipe,
        input.atom,
        input.shells,
        &BTreeMap::new(),
        input.extra_l,
        input.anchor,
        equations,
    )?;
    let reference = references
        .get(&l)
        .ok_or_else(|| invalid(format!("the spin-orbit channel l={l} has no references")))?;
    let r = input.atom.r.as_slice();
    let v_ae = input.atom.effective.as_slice();
    let smooth = optimize(
        ctx,
        &PseudoInput {
            r,
            v_ae,
            l,
            waves: &reference.waves,
            energies: &reference.energies,
            r_cut: reference.r_cut,
            q_cut: recipe.q_cut,
            n_bessel: recipe.n_bessel,
            norm: NormCondition::Factor(1.),
            equation: equations(l),
        },
    )?;
    let channel = assemble(
        ctx,
        &Atomic {
            r,
            v_ae,
            v_loc: input.v_loc,
            r_local: input.r_local,
        },
        &Partial {
            l,
            n: reference.n(),
            energies: &reference.energies,
            waves: &reference.waves,
            smooth: &smooth,
            q_cut: recipe.q_cut,
            occupations: &[],
        },
        false,
    )?;
    let n = r.len();
    let count = channel.reference_energies.len();
    let tables = (0..count)
        .map(|k| &channel.projectors.as_slice()[k * n..(k + 1) * n])
        .collect::<Vec<_>>();
    let level = lowest_level(
        ctx,
        r,
        input.v_loc,
        l,
        &tables,
        &rows(&channel.coupling_screened),
        &rows(&channel.overlap_correction),
    )?;
    let projectors = (0..count)
        .map(|k| ctx.copy_slice(&channel.projectors.as_slice()[k * n..(k + 1) * n]))
        .collect::<Result<Vec<_>>>()?;
    Ok(Branch {
        projectors,
        ionic: &channel.coupling_screened - &channel.overlap_correction * input.hartree_screening,
        screened: channel.coupling_screened.clone(),
        overlap: channel.overlap_correction.clone(),
        energies: reference.energies.clone(),
        error: level - reference.energies[0],
    })
}

/// The two branches of `l` with the Dirac equation of the atom: `j = l - 1/2`, `j = l + 1/2`.
pub fn branches(ctx: &Context, input: &Inputs<'_>, l: usize) -> Result<[Branch; 2]> {
    let one = |kappa: i32| {
        branch(ctx, input, l, &|ll| {
            // Every other channel of the evaluation sits on its `j = l + 1/2` branch; only the
            // channel `l` is read.
            branch_equation(input.recipe, if ll == l { kappa } else { -(ll as i32 + 1) })
        })
    };
    let [down, up] = kappas(l);
    Ok([one(down)?, one(up)?])
}

/// `blockdiag(a, b)`.
fn block_diag(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(a.nrows() + b.nrows(), a.ncols() + b.ncols());
    out.view_mut((0, 0), a.shape()).copy_from(a);
    out.view_mut((a.nrows(), a.ncols()), b.shape()).copy_from(b);
    out
}

/// The `(2j + 1)`-weighted average of the two blocks with the weights of `weights`.
fn average(weights: [f64; 2], down: &DMatrix<f64>, up: &DMatrix<f64>) -> DMatrix<f64> {
    let total = weights[0] + weights[1];
    block_diag(&(down * (weights[0] / total)), &(up * (weights[1] / total)))
}

/// The `L.S` difference: `-2/(2l+1)` on the `j = l - 1/2` block, `+2/(2l+1)` on the other.
fn difference(l: usize, down: &DMatrix<f64>, up: &DMatrix<f64>) -> DMatrix<f64> {
    let factor = 2. / (2 * l + 1) as f64;
    block_diag(&(down * -factor), &(up * factor))
}

fn array(ctx: &Context, shape: Vec<usize>, values: &[f64]) -> Result<Array> {
    let mut data = ctx.allocate::<f64>(product(&shape)?)?;
    data.as_mut_slice().copy_from_slice(values);
    Ok(Array { shape, data })
}

fn matrix(ctx: &Context, m: &DMatrix<f64>) -> Result<Array> {
    array(ctx, vec![m.nrows(), m.ncols()], &rows(m))
}

/// The record channel from the two branches, the average taken with `weights`
/// (`weights(l)` in production).
pub fn combine(
    ctx: &Context,
    l: usize,
    [down, up]: [Branch; 2],
    weights: [f64; 2],
) -> Result<SpinOrbit> {
    let energies = down
        .energies
        .iter()
        .chain(&up.energies)
        .copied()
        .collect::<Vec<_>>();
    let (screened_avg, ionic_avg, overlap_avg) = (
        average(weights, &down.screened, &up.screened),
        average(weights, &down.ionic, &up.ionic),
        average(weights, &down.overlap, &up.overlap),
    );
    let (screened_ls, ionic_ls, overlap_ls) = (
        difference(l, &down.screened, &up.screened),
        difference(l, &down.ionic, &up.ionic),
        difference(l, &down.overlap, &up.overlap),
    );
    let errors = [down.error, up.error];
    let mut projectors = down.projectors;
    projectors.extend(up.projectors);
    Ok(SpinOrbit {
        projectors,
        average: matrix(ctx, &ionic_avg)?,
        average_screened: matrix(ctx, &screened_avg)?,
        coupling: matrix(ctx, &ionic_ls)?,
        coupling_screened: matrix(ctx, &screened_ls)?,
        overlap_average: matrix(ctx, &overlap_avg)?,
        overlap_coupling: matrix(ctx, &overlap_ls)?,
        reference_energies: array(ctx, vec![2, energies.len() / 2], &energies)?,
        level_errors: array(ctx, vec![2], &errors)?,
    })
}

/// The matrix of one branch recovered from the stored pair `(average, coupling)` of a record
/// channel: `V_j = V_avg + V_SO (L.S)_j` with `L.S = -(l+1)/2` on `j = l - 1/2` (`up == false`)
/// and `l/2` on `j = l + 1/2`, read on the diagonal block of that branch's `count` projectors
/// (the first `count` for the lower `j`, the last `count` for the upper).
pub fn recover(
    l: usize,
    average: &DMatrix<f64>,
    coupling: &DMatrix<f64>,
    up: bool,
    count: usize,
) -> DMatrix<f64> {
    let ls = if up {
        l as f64 / 2.
    } else {
        -(l as f64 + 1.) / 2.
    };
    let at = if up { average.nrows() - count } else { 0 };
    let both = average + coupling * ls;
    both.view((at, at), (count, count)).into_owned()
}

/// The spin-orbit channels of a Dirac dataset: every `l >= 1` of the scalar construction.
pub fn channels(ctx: &Context, input: &Inputs<'_>) -> Result<BTreeMap<u32, SpinOrbit>> {
    let mut out = BTreeMap::new();
    for &l in input.radii.keys().filter(|&&l| l >= 1) {
        out.insert(
            l as u32,
            combine(ctx, l, branches(ctx, input, l)?, weights(l))?,
        );
    }
    Ok(out)
}
