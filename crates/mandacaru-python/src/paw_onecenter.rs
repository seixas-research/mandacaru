//! Bindings of the PAW one-center electron-electron terms and the local-potential
//! range separation. Arrays are copied in and returned owned; the work is in
//! `mandacaru_core::paw::{onecenter*, local_split}`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    paw::{
        local_split,
        onecenter::{self, Basis, Channel, Couplings, Projector, Sphere},
        onecenter_hybrid::{
            self as hybrid, ChannelData, ChannelStack, ExactAtom, Factors, Hybrid, HybridAtom,
            Terms,
        },
        onecenter_radial as radial,
        onecenter_terms::{self as terms, CoreOrbital, Occupied, ReferenceAtom},
    },
    product, Context, Reservation,
};
use nalgebra::DMatrix;
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// One channel's arrays, owned.
struct Owned {
    l: usize,
    waves: usize,
    ae: Vec<f64>,
    ps: Vec<f64>,
    vanderbilt: Vec<f64>,
    r_cut: f64,
    occupations: Vec<f64>,
    projector_radius: f64,
}

impl Owned {
    fn channel(&self) -> Channel<'_> {
        Channel {
            l: self.l,
            waves: self.waves,
            ae: &self.ae,
            ps: &self.ps,
            vanderbilt: &self.vanderbilt,
            r_cut: self.r_cut,
        }
    }
    fn occupied(&self) -> Occupied<'_> {
        Occupied {
            channel: self.channel(),
            occupations: &self.occupations,
            projector_radius: self.projector_radius,
        }
    }
}

/// The per-channel arrays of a dataset: `ls[i]` is the momentum of `ae[i]`
/// (`waves x grid`), `ps[i]`, `vanderbilt[i]` (`waves x waves`).
struct Channels<'a, 'py> {
    ls: &'a [usize],
    ae: &'a [PyReadonlyArray2<'py, f64>],
    ps: &'a [PyReadonlyArray2<'py, f64>],
    vanderbilt: &'a [PyReadonlyArray2<'py, f64>],
    r_cuts: &'a [f64],
    occupations: &'a [PyReadonlyArray1<'py, f64>],
    projector_radii: &'a [f64],
}

fn own_channels(
    context: &Context,
    channels: &Channels<'_, '_>,
    keep: &mut Vec<Reservation>,
) -> PyResult<Vec<Owned>> {
    let count = channels.ls.len();
    if [
        channels.ae.len(),
        channels.ps.len(),
        channels.vanderbilt.len(),
        channels.r_cuts.len(),
    ]
    .iter()
    .any(|n| *n != count)
    {
        return Err(invalid("the channel arrays differ in length"));
    }
    let mut out = Vec::new();
    for i in 0..count {
        let (ae, a) = copy(context, &channels.ae[i])?;
        let (ps, b) = copy(context, &channels.ps[i])?;
        let (vanderbilt, c) = copy(context, &channels.vanderbilt[i])?;
        keep.extend([a, b, c]);
        let occupations = match channels.occupations.get(i) {
            Some(o) => {
                let (o, d) = copy(context, o)?;
                keep.push(d);
                o
            }
            None => Vec::new(),
        };
        out.push(Owned {
            l: channels.ls[i],
            waves: channels.ae[i].shape()[0],
            ae,
            ps,
            vanderbilt,
            r_cut: channels.r_cuts[i],
            occupations,
            projector_radius: channels.projector_radii.get(i).copied().unwrap_or(0.),
        });
    }
    Ok(out)
}

fn projectors(rows: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<Projector>> {
    if rows.shape()[1] != 4 {
        return Err(invalid("projectors are rows of l, m, index, real"));
    }
    let data = rows.as_slice()?;
    data.chunks(4)
        .map(|p| {
            if p[0] < 0 || p[2] < 0 {
                return Err(invalid("a projector has a negative l or index"));
            }
            Ok(Projector {
                l: p[0] as usize,
                m: p[1] as i32,
                index: p[2] as usize,
                real: p[3] != 0,
            })
        })
        .collect()
}

fn basis(name: &str) -> PyResult<Basis> {
    Basis::parse(name).map_err(error)
}

#[pyfunction]
fn paw_radial_coulomb(
    context: PyRef<'_, PyContext>,
    r: PyReadonlyArray1<'_, f64>,
    rho_1: PyReadonlyArray1<'_, f64>,
    rho_2: PyReadonlyArray1<'_, f64>,
    l: usize,
) -> PyResult<f64> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (rho_1, _b) = copy(&context.inner, &rho_1)?;
    let (rho_2, _c) = copy(&context.inner, &rho_2)?;
    radial::radial_coulomb(&r, &rho_1, &rho_2, l).map_err(error)
}

#[pyfunction]
#[pyo3(signature=(context, r1, r2, levels, omega, points=None))]
fn paw_long_range_kernel<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    levels: Vec<usize>,
    omega: f64,
    points: Option<usize>,
) -> PyResult<Array<'py, f64>> {
    let (r1, _a) = copy(&context.inner, &r1)?;
    let (r2, _b) = copy(&context.inner, &r2)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| radial::long_range_radial_kernel(ctx, &r1, &r2, &levels, omega, points))
        .map_err(error)?;
    array(
        py,
        &[levels.len(), r1.len(), r2.len()],
        out.into_iter().flatten().collect(),
    )
}

#[pyfunction]
fn paw_long_range_shape<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    r_g: f64,
    l: usize,
    omega: f64,
) -> PyResult<Array<'py, f64>> {
    let (radius, _a) = copy(&context.inner, &radius)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| radial::long_range_shape_potential(ctx, &radius, r_g, l, omega))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

/// `keys` rows: `L, la, ma, lb, mb, lc, mc, ld, md`.
#[pyfunction]
fn paw_angular_coupling<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray2<'py, i64>,
    real: bool,
) -> PyResult<Array<'py, Complex64>> {
    if keys.shape()[1] != 9 || keys.as_slice()?.iter().any(|k| k.unsigned_abs() > 64) {
        return Err(invalid(
            "coupling keys are rows of L, la, ma, lb, mb, lc, mc, ld, md",
        ));
    }
    let keys = keys.as_slice()?.to_vec();
    let n = keys.len() / 9;
    let out = py
        .detach(|| {
            let mut couplings = Couplings::new()?;
            Ok::<_, mandacaru_core::Error>(
                keys.chunks(9)
                    .map(|k| {
                        if [0, 1, 3, 5, 7].iter().any(|&i| k[i] < 0) {
                            return Complex64::new(0., 0.);
                        }
                        let p = |i: usize| Projector {
                            l: k[i] as usize,
                            m: k[i + 1] as i32,
                            index: 0,
                            real,
                        };
                        couplings.angular(k[0] as usize, &p(1), &p(3), &p(5), &p(7))
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .map_err(error)?;
    array(py, &[n], out)
}

#[pyfunction]
fn paw_three_j_squared(a: usize, b: usize, c: usize) -> PyResult<f64> {
    radial::three_j_squared(a, b, c).map_err(error)
}

fn tensor_array<'py>(py: Python<'py>, tensor: onecenter::Tensor) -> PyResult<Bound<'py, PyAny>> {
    let n = tensor.n;
    if tensor.real {
        Ok(array(
            py,
            &[n, n, n, n],
            tensor.data.iter().map(|v| v.re).collect(),
        )?
        .into_any())
    } else {
        Ok(array(py, &[n, n, n, n], tensor.data)?.into_any())
    }
}

/// `Delta W[a, b, c, d]` (`omega = 0`: the bare Coulomb kernel); `symmetrize` returns
/// the half sum with the exchange of the two electrons. Real for real projectors.
#[pyfunction]
fn paw_one_center_tensor<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    ls: Vec<usize>,
    ae: Vec<PyReadonlyArray2<'py, f64>>,
    ps: Vec<PyReadonlyArray2<'py, f64>>,
    vanderbilt: Vec<PyReadonlyArray2<'py, f64>>,
    r_cuts: Vec<f64>,
    compensation_radius: f64,
    basis_name: &str,
    layout: PyReadonlyArray2<'py, i64>,
    omega: f64,
    symmetrize: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let basis = basis(basis_name)?;
    let ctx = &context.inner;
    let (r, _r) = copy(ctx, &r)?;
    let mut keep = Vec::new();
    let channels = Channels {
        ls: &ls,
        ae: &ae,
        ps: &ps,
        vanderbilt: &vanderbilt,
        r_cuts: &r_cuts,
        occupations: &[],
        projector_radii: &[],
    };
    let owned = own_channels(ctx, &channels, &mut keep)?;
    let layout = projectors(&layout)?;
    let tensor = py
        .detach(|| {
            let sphere = Sphere {
                r: &r,
                channels: owned.iter().map(Owned::channel).collect(),
                compensation_radius,
            };
            let tensor = onecenter::one_center_tensor(ctx, &sphere, basis, &layout, omega)?;
            Ok::<_, mandacaru_core::Error>(if symmetrize {
                tensor.exchange_symmetrized()
            } else {
                tensor
            })
        })
        .map_err(error)?;
    tensor_array(py, tensor)
}

fn blocks_out<'py>(
    py: Python<'py>,
    ls: Vec<usize>,
    matrices: Vec<DMatrix<f64>>,
) -> PyResult<Vec<(usize, Array<'py, f64>)>> {
    ls.into_iter()
        .zip(matrices)
        .map(|(l, m)| {
            let (rows, cols) = (m.nrows(), m.ncols());
            let data = (0..rows)
                .flat_map(|i| (0..cols).map(move |j| (i, j)))
                .map(|(i, j)| m[(i, j)])
                .collect();
            Ok((l, array(py, &[rows, cols], data)?))
        })
        .collect()
}

/// `(X0 - V.D0, [(l, V_l)])` of the reference atom's short-range semilocal exchange.
#[pyfunction]
fn paw_frozen_exchange<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    ls: Vec<usize>,
    ae: Vec<PyReadonlyArray2<'py, f64>>,
    ps: Vec<PyReadonlyArray2<'py, f64>>,
    vanderbilt: Vec<PyReadonlyArray2<'py, f64>>,
    r_cuts: Vec<f64>,
    occupations: Vec<PyReadonlyArray1<'py, f64>>,
    projector_radii: Vec<f64>,
    omega: f64,
    basis_name: &str,
) -> PyResult<(f64, Vec<(usize, Array<'py, f64>)>)> {
    let basis = basis(basis_name)?;
    let ctx = &context.inner;
    let (r, _r) = copy(ctx, &r)?;
    let mut keep = Vec::new();
    let channels = Channels {
        ls: &ls,
        ae: &ae,
        ps: &ps,
        vanderbilt: &vanderbilt,
        r_cuts: &r_cuts,
        occupations: &occupations,
        projector_radii: &projector_radii,
    };
    let owned = own_channels(ctx, &channels, &mut keep)?;
    let (constant, blocks) = py
        .detach(|| {
            let occupied = owned.iter().map(Owned::occupied).collect::<Vec<_>>();
            terms::frozen_exchange_terms(ctx, &r, &occupied, omega, basis)
        })
        .map_err(error)?;
    let ls = blocks.iter().map(|b| b.l).collect();
    let matrices = blocks.into_iter().map(|b| b.matrix).collect();
    Ok((constant, blocks_out(py, ls, matrices)?))
}

/// `[(n_electron-free) radial functions]`, one row per core subshell, and the density
/// difference (electrons) to the stored core.
#[pyfunction]
#[pyo3(signature=(context, r, atom_grid, ls, occupations, orbitals, stored))]
fn paw_core_orbitals<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    atom_grid: Option<PyReadonlyArray1<'py, f64>>,
    ls: Vec<usize>,
    occupations: Vec<f64>,
    orbitals: PyReadonlyArray2<'py, f64>,
    stored: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Array<'py, f64>, f64)> {
    let ctx = &context.inner;
    let (r, _a) = copy(ctx, &r)?;
    let (stored, _b) = copy(ctx, &stored)?;
    let grid = match atom_grid {
        Some(g) => Some(copy(ctx, &g)?),
        None => None,
    };
    let (flat, _c) = copy(ctx, &orbitals)?;
    let width = orbitals.shape()[1];
    if ls.len() != occupations.len() || ls.len() != orbitals.shape()[0] {
        return Err(invalid("the core subshell lists differ in length"));
    }
    let subshells = ls
        .iter()
        .zip(&occupations)
        .zip(flat.chunks(width.max(1)))
        .map(|((l, o), u)| (*l, *o, u.to_vec()))
        .collect::<Vec<_>>();
    let (radial, error_electrons) = py
        .detach(|| {
            terms::core_orbital_tables(
                &r,
                grid.as_ref().map(|g| g.0.as_slice()),
                &subshells,
                &stored,
            )
        })
        .map_err(error)?;
    let rows = radial.len();
    Ok((
        array(py, &[rows, r.len()], radial.into_iter().flatten().collect())?,
        error_electrons,
    ))
}

type ExactOut<'py> = Vec<(usize, Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)>;

/// `(l, V, K, D0)` per channel: the frozen exchange-correlation, the frozen core's
/// exchange and the reference density matrix, in the projector basis.
#[pyfunction]
fn paw_exact_terms<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    ls: Vec<usize>,
    ae: Vec<PyReadonlyArray2<'py, f64>>,
    ps: Vec<PyReadonlyArray2<'py, f64>>,
    vanderbilt: Vec<PyReadonlyArray2<'py, f64>>,
    r_cuts: Vec<f64>,
    occupations: Vec<PyReadonlyArray1<'py, f64>>,
    projector_radii: Vec<f64>,
    core_density: PyReadonlyArray1<'py, f64>,
    valence_density: PyReadonlyArray1<'py, f64>,
    smooth_core: PyReadonlyArray1<'py, f64>,
    functional: &str,
    relativistic: bool,
    core_ls: Vec<usize>,
    core_occupations: Vec<f64>,
    core_radial: PyReadonlyArray2<'py, f64>,
    basis_name: &str,
) -> PyResult<ExactOut<'py>> {
    let basis = basis(basis_name)?;
    let ctx = &context.inner;
    let (r, _r) = copy(ctx, &r)?;
    let (core_density, _a) = copy(ctx, &core_density)?;
    let (valence_density, _b) = copy(ctx, &valence_density)?;
    let (smooth_core, _c) = copy(ctx, &smooth_core)?;
    let (flat, _d) = copy(ctx, &core_radial)?;
    if core_ls.len() != core_occupations.len() || core_ls.len() != core_radial.shape()[0] {
        return Err(invalid("the core subshell lists differ in length"));
    }
    let width = core_radial.shape()[1].max(1);
    let core = core_ls
        .iter()
        .zip(&core_occupations)
        .zip(flat.chunks(width))
        .map(|((l, o), radial)| CoreOrbital {
            l: *l,
            occupation: *o,
            radial: radial.to_vec(),
        })
        .collect::<Vec<_>>();
    let mut keep = Vec::new();
    let channels = Channels {
        ls: &ls,
        ae: &ae,
        ps: &ps,
        vanderbilt: &vanderbilt,
        r_cuts: &r_cuts,
        occupations: &occupations,
        projector_radii: &projector_radii,
    };
    let owned = own_channels(ctx, &channels, &mut keep)?;
    let out = py
        .detach(|| {
            let atom = ReferenceAtom {
                r: &r,
                channels: owned.iter().map(Owned::occupied).collect(),
                core_density: &core_density,
                valence_density: &valence_density,
                smooth_core: &smooth_core,
                functional,
                relativistic,
                core: &core,
            };
            terms::exact_one_center_terms(ctx, &atom, basis)
        })
        .map_err(error)?;
    let flatten = |m: &DMatrix<f64>| -> Vec<f64> {
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| m[(i, j)])
            .collect()
    };
    out.into_iter()
        .map(|(l, v, k, d0)| {
            let shape = [v.nrows(), v.ncols()];
            Ok((
                l,
                array(py, &shape, flatten(&v))?,
                array(py, &shape, flatten(&k))?,
                array(py, &shape, flatten(&d0))?,
            ))
        })
        .collect()
}

fn dmatrix(rows: usize, cols: usize, data: &[f64]) -> DMatrix<f64> {
    DMatrix::from_fn(rows, cols, |i, j| data[i * cols + j])
}

fn blocks_in(
    ctx: &Context,
    ls: &[usize],
    blocks: &[PyReadonlyArray2<'_, f64>],
    keep: &mut Vec<Reservation>,
) -> PyResult<Vec<(usize, DMatrix<f64>)>> {
    if ls.len() != blocks.len() {
        return Err(invalid(
            "the channel blocks differ in count from their momenta",
        ));
    }
    let mut out = Vec::new();
    for (l, block) in ls.iter().zip(blocks) {
        let (data, hold) = copy(ctx, block)?;
        keep.push(hold);
        out.push((*l, dmatrix(block.shape()[0], block.shape()[1], &data)));
    }
    Ok(out)
}

fn layout_rows(rows: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<(usize, i32, usize)>> {
    if rows.shape()[1] != 3 {
        return Err(invalid("a layout is rows of l, m, index"));
    }
    rows.as_slice()?
        .chunks(3)
        .map(|p| {
            if p[0] < 0 || p[2] < 0 {
                return Err(invalid("a projector has a negative l or index"));
            }
            Ok((p[0] as usize, p[1] as i32, p[2] as usize))
        })
        .collect()
}

fn expand(
    layout: &[(usize, i32, usize)],
    blocks: &[(usize, DMatrix<f64>)],
) -> PyResult<DMatrix<f64>> {
    let refs = blocks.iter().map(|(l, m)| (*l, m)).collect::<Vec<_>>();
    hybrid::expand_blocks(layout, &refs).map_err(error)
}

fn matrix_array<'py>(py: Python<'py>, m: &DMatrix<f64>) -> PyResult<Array<'py, f64>> {
    array(
        py,
        &[m.nrows(), m.ncols()],
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| m[(i, j)])
            .collect(),
    )
}

type ExactAtomOut<'py> = (
    Array<'py, f64>,
    Array<'py, f64>,
    Array<'py, f64>,
    Array<'py, f64>,
    Array<'py, f64>,
    Array<'py, f64>,
);

/// `(V, K, D0, coupling, values, factors)` of one sphere: the layout-expanded one-body
/// blocks, the coupling `-V - Delta W . D0 + K`, and the low-rank factors of the
/// pair-pair matrix of the real exchange tensor `tensor` (`omega = 0`).
#[pyfunction]
fn paw_exact_atom<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    layout: PyReadonlyArray2<'py, i64>,
    tensor: PyReadonlyArrayDyn<'py, f64>,
    ls: Vec<usize>,
    v_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    k_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    d0_blocks: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<ExactAtomOut<'py>> {
    let ctx = &context.inner;
    let layout = layout_rows(&layout)?;
    let n = layout.len();
    let (tensor, _a) = copy(ctx, &tensor)?;
    let mut keep = Vec::new();
    // The three layout-expanded `n x n` blocks the core takes over.
    let blocks = product(&[3, n, n, std::mem::size_of::<f64>()]).map_err(error)?;
    let _expanded = ctx.reserve(blocks).map_err(error)?;
    let v = expand(&layout, &blocks_in(ctx, &ls, &v_blocks, &mut keep)?)?;
    let k = expand(&layout, &blocks_in(ctx, &ls, &k_blocks, &mut keep)?)?;
    let d0 = expand(&layout, &blocks_in(ctx, &ls, &d0_blocks, &mut keep)?)?;
    let atom: ExactAtom = py
        .detach(|| hybrid::exact_atom(ctx, n, &tensor, v, k, d0))
        .map_err(error)?;
    let rank = atom.values.len();
    // The returned arrays are collected beside the atom: `v`, `k`, `d0` and the
    // coupling (`n^2` each), the values, and the `rank` factors.
    let outputs = product(&[4, n, n])
        .and_then(|matrices| {
            let factors = product(&[rank, n, n])?;
            matrices
                .checked_add(factors)
                .and_then(|sum| sum.checked_add(rank))
                .ok_or(mandacaru_core::Error::Overflow)
        })
        .and_then(|doubles| product(&[doubles, std::mem::size_of::<f64>()]))
        .map_err(error)?;
    let _outputs = ctx.reserve(outputs).map_err(error)?;
    let factors = atom
        .factors
        .iter()
        .flat_map(|f| (0..n).flat_map(move |i| (0..n).map(move |j| f[(i, j)])))
        .collect::<Vec<_>>();
    Ok((
        matrix_array(py, &atom.v)?,
        matrix_array(py, &atom.k)?,
        matrix_array(py, &atom.d0)?,
        matrix_array(py, &atom.coupling)?,
        array(py, &[rank], atom.values.clone())?,
        array(py, &[rank, n, n], factors)?,
    ))
}

/// The low-rank factors of each sphere as matrices. Each factor array is charged to
/// `ctx` (into `keep`) before it is rebuilt, one copy.
fn factor_set(
    ctx: &Context,
    members: &[Vec<usize>],
    factors: &[PyReadonlyArrayDyn<'_, f64>],
    keep: &mut Vec<Reservation>,
) -> PyResult<Vec<Vec<DMatrix<f64>>>> {
    if members.len() != factors.len() {
        return Err(invalid("each sphere needs its members and its factors"));
    }
    members
        .iter()
        .zip(factors)
        .map(|(m, f)| {
            let shape = f.shape();
            if shape.len() != 3 || shape[1] != m.len() || shape[2] != m.len() {
                return Err(invalid(
                    "factors are (k, n, n) for the n members of a sphere",
                ));
            }
            let data = f.as_slice()?;
            keep.push(
                ctx.reserve(product(&[data.len(), std::mem::size_of::<f64>()]).map_err(error)?)
                    .map_err(error)?,
            );
            Ok(data
                .chunks((m.len() * m.len()).max(1))
                .map(|u| dmatrix(m.len(), m.len(), u))
                .collect())
        })
        .collect()
}

fn complex_matrix<'py>(
    ctx: &Context,
    input: &PyReadonlyArray2<'py, Complex64>,
    keep: &mut Vec<Reservation>,
) -> PyResult<Vec<Complex64>> {
    let (data, hold) = copy(ctx, input)?;
    keep.push(hold);
    Ok(data)
}

/// The stacked channels as one `(k, M, M)` array, moved out of the buffer: no copy.
fn stacked<'py>(py: Python<'py>, channels: ChannelStack) -> PyResult<Bound<'py, PyAny>> {
    let shape = [channels.count, channels.rows, channels.rows];
    match channels.data {
        ChannelData::Real(buffer) => {
            let (data, _held) = buffer.into_parts();
            Ok(array(py, &shape, data)?.into_any())
        }
        ChannelData::Complex(buffer) => {
            let (data, _held) = buffer.into_parts();
            Ok(array(py, &shape, data)?.into_any())
        }
    }
}

/// `C_A U^k C_A^T` of every factor of every sphere, stacked `(k, M, M)`; real when
/// `real` is set (real projections).
#[pyfunction]
fn paw_exact_channels<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    c: PyReadonlyArray2<'py, Complex64>,
    members: Vec<Vec<usize>>,
    factors: Vec<PyReadonlyArrayDyn<'py, f64>>,
    real: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let ctx = &context.inner;
    let (rows, width) = (c.shape()[0], c.shape()[1]);
    let mut keep = Vec::new();
    let c = complex_matrix(ctx, &c, &mut keep)?;
    if members.iter().flatten().any(|m| *m >= width) {
        return Err(invalid("a member column exceeds the projections"));
    }
    let owned = factor_set(ctx, &members, &factors, &mut keep)?;
    let out = py
        .detach(|| {
            let atoms = members
                .iter()
                .zip(&owned)
                .map(|(m, f)| Factors {
                    members: m,
                    factors: f,
                })
                .collect::<Vec<_>>();
            hybrid::exact_channels(ctx, &c, rows, width, &atoms, real)
        })
        .map_err(error)?;
    stacked(py, out)
}

/// `dC_A U^k C_A^T + C_A U^k dC_A^T` of every factor, stacked `(k, M, M)`.
#[pyfunction]
fn paw_exact_channel_derivatives<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    c: PyReadonlyArray2<'py, Complex64>,
    dc: PyReadonlyArray2<'py, Complex64>,
    members: Vec<Vec<usize>>,
    factors: Vec<PyReadonlyArrayDyn<'py, f64>>,
    real: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let ctx = &context.inner;
    let (rows, width) = (c.shape()[0], c.shape()[1]);
    if dc.shape() != c.shape() {
        return Err(invalid("dC must have the shape of C"));
    }
    let mut keep = Vec::new();
    let c = complex_matrix(ctx, &c, &mut keep)?;
    let dc = complex_matrix(ctx, &dc, &mut keep)?;
    if members.iter().flatten().any(|m| *m >= width) {
        return Err(invalid("a member column exceeds the projections"));
    }
    let owned = factor_set(ctx, &members, &factors, &mut keep)?;
    let out = py
        .detach(|| {
            let atoms = members
                .iter()
                .zip(&owned)
                .map(|(m, f)| Factors {
                    members: m,
                    factors: f,
                })
                .collect::<Vec<_>>();
            hybrid::exact_channel_derivatives(ctx, &c, &dc, rows, width, &atoms, real)
        })
        .map_err(error)?;
    stacked(py, out)
}

/// The one-center terms of a screened hybrid, held natively between evaluations.
#[pyclass(name = "OneCenterHybrid", module = "mandacaru._native", frozen)]
pub struct PyOneCenterHybrid {
    inner: Hybrid,
    context: Context,
    real_tensors: bool,
    /// The budget of the `n^4` tensors and blocks the object keeps, for its lifetime.
    _held: Vec<Reservation>,
}

type TermsOut<'py> = (f64, f64, f64, Vec<Bound<'py, PyAny>>);

/// The operators as arrays: complex ones are moved out of their buffers; real ones need
/// one real copy per operator, charged to `ctx` first.
fn terms_out<'py>(
    ctx: &Context,
    py: Python<'py>,
    terms: Terms,
    real: bool,
) -> PyResult<TermsOut<'py>> {
    let dim = terms.dim;
    let operators = terms
        .operators
        .into_iter()
        .map(|o| {
            if real {
                let mut part = ctx.allocate::<f64>(o.as_slice().len()).map_err(error)?;
                for (slot, v) in part.as_mut_slice().iter_mut().zip(o.as_slice()) {
                    *slot = v.re;
                }
                let (data, _held) = part.into_parts();
                Ok(array(py, &[dim, dim], data)?.into_any())
            } else {
                let (data, _held) = o.into_parts();
                Ok(array(py, &[dim, dim], data)?.into_any())
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        terms.energy,
        terms.exact_exchange,
        terms.semilocal,
        operators,
    ))
}

#[pymethods]
impl PyOneCenterHybrid {
    /// Per sphere: its member columns, its complex exchange tensor (`n^4`), its
    /// projector layout (rows `l, m, index`) and the frozen potential blocks
    /// `{l: V_l}` (`block_ls`, `blocks`); `frozen` lists the spheres' frozen
    /// constants `X0 - V.D0`.
    #[new]
    fn new(
        context: PyRef<'_, PyContext>,
        members: Vec<Vec<usize>>,
        tensors: Vec<PyReadonlyArrayDyn<'_, Complex64>>,
        layouts: Vec<PyReadonlyArray2<'_, i64>>,
        block_ls: Vec<Vec<usize>>,
        blocks: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
        projectors: usize,
        fraction: f64,
        frozen: Vec<f64>,
        real_tensors: bool,
    ) -> PyResult<Self> {
        let ctx = context.inner.clone();
        let count = members.len();
        if [
            tensors.len(),
            layouts.len(),
            block_ls.len(),
            blocks.len(),
            frozen.len(),
        ]
        .iter()
        .any(|n| *n != count)
        {
            return Err(invalid(
                "each sphere needs members, a tensor, a layout, blocks and a constant",
            ));
        }
        let mut keep = Vec::new();
        let mut atoms = Vec::new();
        for i in 0..count {
            let (tensor, hold) = copy(&ctx, &tensors[i])?;
            keep.push(hold);
            let layout = layout_rows(&layouts[i])?;
            if layout.len() != members[i].len() {
                return Err(invalid("a sphere's layout and members differ in length"));
            }
            let v = expand(
                &layout,
                &blocks_in(&ctx, &block_ls[i], &blocks[i], &mut keep)?,
            )?;
            atoms.push(HybridAtom {
                v,
                members: members[i].clone(),
                tensor,
            });
        }
        let inner = Hybrid::new(atoms, projectors, fraction, frozen.iter().sum()).map_err(error)?;
        Ok(Self {
            inner,
            context: ctx,
            real_tensors,
            _held: keep,
        })
    }

    #[getter]
    fn constant(&self) -> f64 {
        self.inner.constant()
    }

    /// One density (closed shell) or two (spin channels); `real` states that every
    /// density is real, so real tensors give real operators.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        densities: Vec<PyReadonlyArray2<'py, Complex64>>,
        real: bool,
    ) -> PyResult<TermsOut<'py>> {
        let mut keep = Vec::new();
        let spins = densities
            .iter()
            .map(|d| complex_matrix(&self.context, d, &mut keep))
            .collect::<PyResult<Vec<_>>>()?;
        let result = py
            .detach(|| {
                let refs = spins.iter().map(|d| d.as_slice()).collect::<Vec<_>>();
                self.inner.evaluate(&self.context, &refs)
            })
            .map_err(error)?;
        terms_out(&self.context, py, result, real && self.real_tensors)
    }

    /// The four spin blocks `uu, ud, du, dd`; one complex `2P x 2P` operator.
    fn evaluate_spinor<'py>(
        &self,
        py: Python<'py>,
        blocks: Vec<PyReadonlyArray2<'py, Complex64>>,
    ) -> PyResult<TermsOut<'py>> {
        if blocks.len() != 4 {
            return Err(invalid("the spinor density has four spin blocks"));
        }
        let mut keep = Vec::new();
        let d = blocks
            .iter()
            .map(|b| complex_matrix(&self.context, b, &mut keep))
            .collect::<PyResult<Vec<_>>>()?;
        let result = py
            .detach(|| {
                self.inner
                    .evaluate_spinor(&self.context, [[&d[0], &d[1]], [&d[2], &d[3]]])
            })
            .map_err(error)?;
        terms_out(&self.context, py, result, false)
    }
}

// --- local_split -----------------------------------------------------------

#[pyfunction]
fn paw_long_range_potential<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    charge: f64,
    sigma: f64,
    radius: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (radius, _a) = copy(&context.inner, &radius)?;
    let out = local_split::long_range_potential(charge, sigma, &radius);
    array(py, &[out.len()], out)
}

/// `sampled - v^lr` (or the derivative's): `sampled` is the dataset's local potential
/// (or its derivative) at `radius`.
#[pyfunction]
fn paw_short_range_potential<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    sampled: PyReadonlyArray1<'py, f64>,
    charge: f64,
    sigma: f64,
    radius: PyReadonlyArray1<'py, f64>,
    derivative: bool,
) -> PyResult<Array<'py, f64>> {
    let (sampled, _a) = copy(&context.inner, &sampled)?;
    let (radius, _b) = copy(&context.inner, &radius)?;
    let out = if derivative {
        local_split::short_range_potential_derivative(&sampled, charge, sigma, &radius)
    } else {
        local_split::short_range_potential(&sampled, charge, sigma, &radius)
    }
    .map_err(error)?;
    array(py, &[out.len()], out)
}

/// Most nodes `paw_linspace` returns: a scan grid, not a table (16 Mi nodes, 128 MiB).
const LINSPACE_MAX: usize = 1 << 24;

/// `np.linspace(start, stop, num)`, the memory reserved from the context first.
#[pyfunction]
fn paw_linspace<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    start: f64,
    stop: f64,
    num: usize,
) -> PyResult<Array<'py, f64>> {
    if num > LINSPACE_MAX {
        return Err(invalid(&format!(
            "the number of nodes must be at most {LINSPACE_MAX}, got {num}"
        )));
    }
    let _memory = context
        .inner
        .reserve(num * std::mem::size_of::<f64>())
        .map_err(error)?;
    let out = local_split::linspace(start, stop, num);
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_tail_radius(
    context: PyRef<'_, PyContext>,
    radius: PyReadonlyArray1<'_, f64>,
    sampled: PyReadonlyArray1<'_, f64>,
    charge: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let (radius, _a) = copy(&context.inner, &radius)?;
    let (sampled, _b) = copy(&context.inner, &sampled)?;
    local_split::tail_radius(&radius, &sampled, charge, tolerance).map_err(error)
}

#[pyfunction]
fn paw_short_range_radius(sigma: f64, sigmas: f64, cutoff: f64, tail: f64) -> f64 {
    local_split::short_range_radius(sigma, sigmas, cutoff, tail)
}

#[pyfunction]
fn paw_split_width(step: PyReadonlyArray2<'_, f64>, factor: f64) -> PyResult<f64> {
    if step.shape() != [3, 3] {
        return Err(invalid("the grid step is a 3 x 3 matrix"));
    }
    let mut matrix = [0.; 9];
    matrix.copy_from_slice(step.as_slice()?);
    Ok(factor * local_split::grid_spacing(&matrix))
}

#[pyfunction]
fn paw_radial_rule<'py>(
    py: Python<'py>,
    radius: f64,
    panel: f64,
    points: usize,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    let (r, w) = local_split::radial_rule(radius, panel, points).map_err(error)?;
    Ok((array(py, &[r.len()], r)?, array(py, &[w.len()], w)?))
}

#[pyfunction]
fn paw_angular_rule<'py>(
    py: Python<'py>,
    polar: usize,
    azimuthal: usize,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    let (directions, weights) = local_split::angular_rule(polar, azimuthal).map_err(error)?;
    let n = weights.len();
    Ok((
        array(py, &[3, n], directions.into_iter().flatten().collect())?,
        array(py, &[n], weights)?,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_radial_coulomb, m)?)?;
    m.add_function(wrap_pyfunction!(paw_long_range_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(paw_long_range_shape, m)?)?;
    m.add_function(wrap_pyfunction!(paw_angular_coupling, m)?)?;
    m.add_function(wrap_pyfunction!(paw_three_j_squared, m)?)?;
    m.add_function(wrap_pyfunction!(paw_one_center_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(paw_frozen_exchange, m)?)?;
    m.add_function(wrap_pyfunction!(paw_core_orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(paw_exact_terms, m)?)?;
    m.add_function(wrap_pyfunction!(paw_exact_atom, m)?)?;
    m.add_function(wrap_pyfunction!(paw_exact_channels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_exact_channel_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(paw_long_range_potential, m)?)?;
    m.add_function(wrap_pyfunction!(paw_short_range_potential, m)?)?;
    m.add_function(wrap_pyfunction!(paw_linspace, m)?)?;
    m.add_function(wrap_pyfunction!(paw_tail_radius, m)?)?;
    m.add_function(wrap_pyfunction!(paw_short_range_radius, m)?)?;
    m.add_function(wrap_pyfunction!(paw_split_width, m)?)?;
    m.add_function(wrap_pyfunction!(paw_radial_rule, m)?)?;
    m.add_function(wrap_pyfunction!(paw_angular_rule, m)?)?;
    m.add_class::<PyOneCenterHybrid>()?;
    Ok(())
}
