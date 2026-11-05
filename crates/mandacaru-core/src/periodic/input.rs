//! Validated portable density-independent crystal data. No Python dependency.
use super::invalid;
use crate::math::{
    archive::Archive,
    storage::{ComplexSamples, Precision},
};
use crate::{
    math::{linalg::eigen, SpatialGrid},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector};
use serde_json::Value;
use std::collections::BTreeMap;

pub enum InputArray {
    Real(Buffer<f64>),
    Complex(Buffer<C64>),
    Indices(Buffer<u64>),
    Samples(ComplexSamples),
}
#[derive(Default)]
pub struct InputArrays(pub BTreeMap<String, InputArray>);
impl InputArrays {
    fn samples(
        &mut self,
        ctx: &Context,
        v: &Value,
        n: usize,
        precision: Precision,
    ) -> Result<ComplexSamples> {
        if let Some(key) = v.as_str() {
            let samples = match self.0.remove(key) {
                Some(InputArray::Samples(samples)) => samples,
                Some(InputArray::Complex(values)) => ComplexSamples::Float64(values),
                _ => {
                    return Err(invalid(format!(
                        "missing or invalid Bloch sample array {key}"
                    )))
                }
            };
            if samples.len() != n || !samples.finite() {
                return Err(invalid("Bloch samples have invalid dimensions or values"));
            }
            return samples.into_precision(ctx, precision);
        }
        ComplexSamples::Float64(complex(ctx, v, n, "Bloch samples")?).into_precision(ctx, precision)
    }
    fn real(&mut self, ctx: &Context, v: &Value, n: usize, name: &str) -> Result<Buffer<f64>> {
        if let Some(key) = v.as_str() {
            if let Some(InputArray::Real(out)) = self.0.remove(key) {
                if out.as_slice().len() == n && out.as_slice().iter().all(|x| x.is_finite()) {
                    return Ok(out);
                }
            }
            return Err(invalid(format!(
                "{name}: missing or invalid real array {key}"
            )));
        }
        real(ctx, v, n, name)
    }
    fn complex(&mut self, ctx: &Context, v: &Value, n: usize, name: &str) -> Result<Buffer<C64>> {
        if let Some(key) = v.as_str() {
            if let Some(InputArray::Complex(out)) = self.0.remove(key) {
                if out.as_slice().len() == n
                    && out
                        .as_slice()
                        .iter()
                        .all(|x| x.re.is_finite() && x.im.is_finite())
                {
                    return Ok(out);
                }
            }
            return Err(invalid(format!(
                "{name}: missing or invalid complex array {key}"
            )));
        }
        complex(ctx, v, n, name)
    }
}
/// The crystal snapshot's metadata text, kept for `Crystal::save`: in memory, or in the scratch
/// folder when it reaches the context's offload threshold, so a long SCF carries no copy of it.
enum Source {
    Resident(String),
    Archived(Archive<u8>),
}
impl Source {
    fn keep(ctx: &Context, text: &str) -> Result<Self> {
        if ctx.offloads(text.len()) {
            let bytes = text.as_bytes();
            Ok(Self::Archived(Archive::create(ctx, bytes.len(), |i| {
                Ok(bytes[i])
            })?))
        } else {
            Ok(Self::Resident(text.to_string()))
        }
    }
    fn resident_bytes(&self) -> usize {
        match self {
            Self::Resident(text) => text.len(),
            Self::Archived(_) => 0,
        }
    }
    fn text(&self, ctx: &Context) -> Result<String> {
        match self {
            Self::Resident(text) => Ok(text.clone()),
            Self::Archived(bytes) => {
                let _reading = ctx.reserve(bytes.len())?;
                let mut text = vec![0u8; bytes.len()];
                bytes.read_range_into(0, &mut text)?;
                String::from_utf8(text).map_err(|_| invalid("the crystal metadata text is damaged"))
            }
        }
    }
}
pub struct KPoint {
    pub k: [f64; 3],
    pub fractional: [f64; 3],
    pub weight: f64,
    pub psi: ComplexSamples,
    pub overlap: DMatrix<C64>,
    pub fixed: DMatrix<C64>,
    pub projections: DMatrix<C64>,
    pub lowdin: DMatrix<C64>,
    pub spin_orbit: Option<DMatrix<C64>>,
}
pub struct Crystal {
    pub context: Context,
    pub grid: SpatialGrid,
    pub m: usize,
    pub p: usize,
    pub channels: usize,
    pub points: Vec<KPoint>,
    pub blocks: Buffer<C64>,
    pub core: Buffer<f64>,
    pub core_tau: Option<Buffer<f64>>,
    pub precision: Precision,
    pub initial: Buffer<f64>,
    pub initial_m: Buffer<f64>,
    pub initial_q: Buffer<C64>,
    pub kernel: Buffer<f64>,
    pub xc_wave: Buffer<f64>,
    pub phase: Buffer<C64>,
    pub kerker: Buffer<f64>,
    pub ion: Buffer<C64>,
    pub hats: ComplexSamples,
    pub u: Buffer<C64>,
    pub uion: Buffer<C64>,
    pub linear: Buffer<C64>,
    pub eion: f64,
    pub ion_constant: f64,
    pub constant: f64,
    pub reference: f64,
    pub electrons: f64,
    pub spins: usize,
    pub spinor: bool,
    pub time_reversal: bool,
    pub functional: String,
    pub relativistic: bool,
    pub permutations: Vec<Vec<usize>>,
    pub qsym: Buffer<C64>,
    /// DFT+U shells, when the snapshot carries a `hubbard` object.
    pub hubbard: Option<super::hubbard::Spec>,
    /// The PAW spheres of a hybrid crystal (its `sites` block); `None` for a semilocal one.
    pub hybrid: Option<super::hybrid_setup::Hybrid>,
    /// The `setup` block (radial tables, function and projector lists, symmetry operations) the
    /// derivative engines rebuild the geometry from; `None` for a snapshot written without it.
    pub description: Option<super::description::Description>,
    /// Why the setup cannot be described (a `setup` block carrying `unavailable`), if so.
    pub description_unavailable: Option<String>,
    source: Source,
    _memory: Reservation,
}
pub(super) fn number(v: &Value, name: &str) -> Result<f64> {
    v.as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("{name} must be finite")))
}
pub(super) fn count(v: &Value, name: &str) -> Result<usize> {
    v.as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid(format!("{name} must be a nonnegative integer")))
}
pub(super) fn real(ctx: &Context, v: &Value, n: usize, name: &str) -> Result<Buffer<f64>> {
    let values = v
        .as_array()
        .filter(|v| v.len() == n)
        .ok_or_else(|| invalid(format!("{name} must contain {n} values")))?;
    let mut out = ctx.allocate(n)?;
    for (a, b) in out.as_mut_slice().iter_mut().zip(values) {
        *a = number(b, name)?;
    }
    Ok(out)
}
pub(super) fn complex(ctx: &Context, v: &Value, n: usize, name: &str) -> Result<Buffer<C64>> {
    let values = v
        .as_array()
        .filter(|v| v.len() == n)
        .ok_or_else(|| invalid(format!("{name} must contain {n} complex pairs")))?;
    let mut out = ctx.allocate(n)?;
    for (a, b) in out.as_mut_slice().iter_mut().zip(values) {
        let pair = b
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or_else(|| invalid(format!("{name} needs [real,imaginary] pairs")))?;
        *a = C64::new(number(&pair[0], name)?, number(&pair[1], name)?);
    }
    Ok(out)
}
fn vector(v: &Value, name: &str) -> Result<[f64; 3]> {
    let a = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| invalid(format!("invalid {name}")))?;
    Ok([
        number(&a[0], name)?,
        number(&a[1], name)?,
        number(&a[2], name)?,
    ])
}
fn hermitian(a: &DMatrix<C64>, name: &str) -> Result<()> {
    let scale = a.iter().map(|v| v.norm()).fold(1f64, f64::max);
    if (a - a.adjoint()).iter().any(|v| v.norm() > 1e-10 * scale) {
        return Err(invalid(format!("{name} must be Hermitian")));
    }
    Ok(())
}
pub fn kpoint(
    ctx: &Context,
    value: &Value,
    m: usize,
    p: usize,
    n: usize,
    precision: Precision,
) -> Result<KPoint> {
    kpoint_arrays(ctx, value, m, p, n, &mut InputArrays::default(), precision)
}
pub fn kpoint_arrays(
    ctx: &Context,
    value: &Value,
    m: usize,
    p: usize,
    n: usize,
    arrays: &mut InputArrays,
    precision: Precision,
) -> Result<KPoint> {
    let psi = arrays.samples(ctx, &value["psi"], product(&[m, n])?, precision)?;
    let overlap = arrays.complex(ctx, &value["overlap"], product(&[m, m])?, "overlap")?;
    let overlap = DMatrix::from_row_slice(m, m, overlap.as_slice());
    hermitian(&overlap, "overlap")?;
    let fixed = arrays.complex(ctx, &value["fixed"], product(&[m, m])?, "fixed Hamiltonian")?;
    let fixed = DMatrix::from_row_slice(m, m, fixed.as_slice());
    hermitian(&fixed, "fixed Hamiltonian")?;
    let projections = arrays.complex(
        ctx,
        &value["projections"],
        product(&[m, p])?,
        "projector overlaps",
    )?;
    let projections = DMatrix::from_row_slice(m, p, projections.as_slice());
    let (s, c) = eigen(overlap.clone())?;
    if s[0] <= 1e-10 {
        return Err(invalid("periodic overlap is singular or indefinite"));
    }
    let lowdin =
        &c * DMatrix::from_diagonal(&DVector::from_iterator(
            m,
            s.iter().map(|s| C64::new(1. / s.sqrt(), 0.)),
        )) * c.adjoint();
    let spin_orbit = if value["spin_orbit"].is_null() {
        None
    } else {
        let values = arrays.complex(
            ctx,
            &value["spin_orbit"],
            product(&[2, m, 2, m])?,
            "spin-orbit Hamiltonian",
        )?;
        let matrix = DMatrix::from_row_slice(2 * m, 2 * m, values.as_slice());
        hermitian(&matrix, "spin-orbit Hamiltonian")?;
        Some(matrix)
    };
    Ok(KPoint {
        k: vector(&value["k"], "Cartesian k-point")?,
        fractional: vector(&value["fractional"], "fractional k-point")?,
        weight: number(&value["weight"], "k-point weight")?,
        psi,
        overlap,
        fixed,
        projections,
        lowdin,
        spin_orbit,
    })
}
/// What `serde_json` can take to parse `source` into a [`Value`] tree, read off the text.
///
/// Every array element and object member is one node, and each separator or opening bracket
/// ends or starts one, so the node count is at most the count of `,`, `[` and `{` plus the
/// root. A node is charged three `Value`s: the array holding it doubles its capacity, and for
/// a moment the old and the new buffer both exist. An object member also holds a key and a
/// tree entry, charged [`MEMBER_BYTES`] on top (counted by `:`), and the strings hold at most
/// the text itself. A separator inside a string raises the bound and never lowers it.
///
/// This is a bound from the structure of the text, not a multiple of its length. The densest
/// possible text (one-digit numbers) costs 48 bytes per text byte here, more than the flat 32 it
/// replaces, so a pathological file is charged more and refused sooner; a real crystal file of
/// 19-character numbers costs about 9 (NiO 348 MB for 39.5 MB of text, against 1263 MB), which
/// is what the counting-allocator test of `tests/periodic_load_memory.rs` pins against the heap.
fn tree_bytes(source: &str) -> Result<usize> {
    const MEMBER_BYTES: usize = 192;
    let (mut nodes, mut members) = (1usize, 0usize);
    for byte in source.bytes() {
        match byte {
            b',' | b'[' | b'{' => nodes += 1,
            b':' => members += 1,
            _ => {}
        }
    }
    product(&[nodes, 3 * std::mem::size_of::<Value>()])?
        .checked_add(product(&[members, MEMBER_BYTES])?)
        .and_then(|bytes| bytes.checked_add(source.len()))
        .ok_or(crate::Error::Overflow)
}

/// The periodic exact exchange sums over `Crystal::points`, so it is exact only on the whole mesh:
/// equal weights, a point set closed under `k -> -k`, and no symmetry table beside several points.
/// An irreducible wedge or a time-reversal-halved mesh would give a wrong energy and wrong bands
/// with no error, so it is refused. One Gamma point is complete whatever the table says.
fn hybrid_needs_the_full_mesh(points: &[KPoint], permutations: usize) -> Result<()> {
    const TOLERANCE: f64 = 1e-8;
    let first = points.first().map_or(0., |p| p.weight);
    let uniform = points.iter().all(|p| (p.weight - first).abs() < 1e-10);
    let closed = points.iter().all(|p| {
        points.iter().any(|q| {
            (0..3).all(|i| {
                let sum = p.fractional[i] + q.fractional[i];
                (sum - sum.round()).abs() < TOLERANCE
            })
        })
    });
    if uniform && closed && (permutations == 0 || points.len() == 1) {
        return Ok(());
    }
    Err(crate::Error::Unsupported(
        "a hybrid functional needs the full k mesh: its exact exchange sums over the listed \
         k-points, and these are a symmetry or time-reversal reduction of the mesh. Use \
         symmetry=False together with a Gamma-centered mesh of one or two points per direction \
         (all of its points are time-reversal invariant, so nothing is halved); on any other \
         mesh use symmetry=False with unrestricted=True and nonzero initial magnetic moments \
         (only a spin-polarized run keeps the whole mesh; a spin-orbit crystal has no such \
         option)"
            .into(),
    ))
}

/// Components to assemble a [`Crystal`] directly from Rust setup engines.
pub struct CrystalComponents {
    pub grid: SpatialGrid,
    pub m: usize,
    pub p: usize,
    pub channels: usize,
    pub points: Vec<KPoint>,
    pub blocks: Buffer<C64>,
    pub core: Buffer<f64>,
    pub core_tau: Option<Buffer<f64>>,
    pub precision: Precision,
    pub initial: Buffer<f64>,
    pub initial_m: Buffer<f64>,
    pub initial_q: Buffer<C64>,
    pub kernel: Buffer<f64>,
    pub xc_wave: Buffer<f64>,
    pub phase: Buffer<C64>,
    pub kerker: Buffer<f64>,
    pub ion: Buffer<C64>,
    pub hats: ComplexSamples,
    pub u: Buffer<C64>,
    pub uion: Buffer<C64>,
    pub linear: Buffer<C64>,
    pub eion: f64,
    pub ion_constant: f64,
    pub constant: f64,
    pub reference: f64,
    pub electrons: f64,
    pub spins: usize,
    pub spinor: bool,
    pub time_reversal: bool,
    pub functional: String,
    pub relativistic: bool,
    pub permutations: Vec<Vec<usize>>,
    pub qsym: Buffer<C64>,
    pub hubbard: Option<super::hubbard::Spec>,
    pub hybrid: Option<super::hybrid_setup::Hybrid>,
    pub description: Option<super::description::Description>,
    pub description_unavailable: Option<String>,
}

impl Crystal {
    pub(crate) fn from_components(ctx: &Context, c: CrystalComponents) -> Result<Self> {
        let labels = c.description.as_ref()
            .ok_or_else(|| invalid("native crystal needs channel labels"))?.channels.clone();
        let step: Vec<f64> = (0..3).flat_map(|i| (0..3).map(move |j| c.grid.step[(i, j)])).collect();
        let points: Vec<Value> = c.points.iter().map(|p| serde_json::json!({
            "k": p.k, "fractional": p.fractional, "weight": p.weight,
        })).collect();
        let metadata = serde_json::json!({
            "schema_version": 1, "representation": "periodic-paw-snapshot",
            "energy_unit": "Ha", "length_unit": "bohr",
            "grid": {"shape": c.grid.index.shape(), "origin": c.grid.origin, "step": step},
            "m": c.m, "p": c.p, "channels": c.channels, "channel_labels": labels,
            "kpoints": points, "eion": c.eion, "ion_constant": c.ion_constant,
            "constant": c.constant, "reference": c.reference, "electrons": c.electrons,
            "spins": c.spins, "spinor": c.spinor, "time_reversal": c.time_reversal,
            "functional": c.functional, "precision": c.precision.name(),
            "relativistic": c.relativistic,
            "setup": {"unavailable": c.description_unavailable.as_deref().unwrap_or(
                "native setup description serialization is not yet available")},
        }).to_string();
        let source = Source::keep(ctx, &metadata)?;
        let memory = ctx.reserve(4096usize.checked_add(source.resident_bytes()).ok_or(crate::Error::Overflow)?)?;
        Ok(Self {
            context: ctx.clone(),
            grid: c.grid,
            m: c.m,
            p: c.p,
            channels: c.channels,
            points: c.points,
            blocks: c.blocks,
            core: c.core,
            core_tau: c.core_tau,
            precision: c.precision,
            initial: c.initial,
            initial_m: c.initial_m,
            initial_q: c.initial_q,
            kernel: c.kernel,
            xc_wave: c.xc_wave,
            phase: c.phase,
            kerker: c.kerker,
            ion: c.ion,
            hats: c.hats,
            u: c.u,
            uion: c.uion,
            linear: c.linear,
            eion: c.eion,
            ion_constant: c.ion_constant,
            constant: c.constant,
            reference: c.reference,
            electrons: c.electrons,
            spins: c.spins,
            spinor: c.spinor,
            time_reversal: c.time_reversal,
            functional: c.functional,
            relativistic: c.relativistic,
            permutations: c.permutations,
            qsym: c.qsym,
            hubbard: c.hubbard,
            hybrid: c.hybrid,
            description: c.description,
            description_unavailable: c.description_unavailable,
            source,
            _memory: memory,
        })
    }

    pub fn from_json(ctx: &Context, source: &str) -> Result<Self> {
        Self::from_arrays(ctx, source, InputArrays::default())
    }
    pub fn from_arrays(ctx: &Context, source: &str, mut arrays: InputArrays) -> Result<Self> {
        let _parse = ctx.reserve(tree_bytes(source)?)?;
        let v: Value = serde_json::from_str(source)
            .map_err(|e| invalid(format!("invalid crystal JSON: {e}")))?;
        if v["schema_version"] != 1
            || v["representation"] != "periodic-paw-snapshot"
            || v["energy_unit"] != "Ha"
            || v["length_unit"] != "bohr"
        {
            return Err(invalid(
                "crystal cache needs schema 1, periodic-paw-snapshot, Ha and bohr",
            ));
        }
        let shape = v["grid"]["shape"]
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or_else(|| invalid("grid needs three dimensions"))?;
        let shape = [
            count(&shape[0], "grid shape")?,
            count(&shape[1], "grid shape")?,
            count(&shape[2], "grid shape")?,
        ];
        let step = arrays.real(ctx, &v["grid"]["step"], 9, "grid step")?;
        let grid = SpatialGrid::new(
            shape,
            vector(&v["grid"]["origin"], "grid origin")?,
            step.as_slice().try_into().unwrap(),
            true,
        )?;
        let n = grid.index.points();
        let precision = Precision::parse(v["precision"].as_str().unwrap_or("float64"))?;
        let m = count(&v["m"], "basis dimension")?;
        let p = count(&v["p"], "projector dimension")?;
        let channels = count(&v["channels"], "moment count")?;
        if m == 0 {
            return Err(invalid("periodic basis must be nonempty"));
        }
        let raw = v["kpoints"]
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("crystal needs k-points"))?;
        let permutation_count = if v["permutations"].is_string() {
            count(&v["permutation_count"], "permutation count")?
        } else {
            v["permutations"]
                .as_array()
                .ok_or_else(|| invalid("symmetry permutations must be supplied"))?
                .len()
        };
        let matrices = product(&[
            raw.len(),
            m,
            m.checked_mul(3)
                .and_then(|m| m.checked_add(p))
                .ok_or(crate::Error::Overflow)?,
            16,
        ])?;
        let kept = Source::keep(ctx, source)?;
        let memory = ctx.reserve(
            matrices
                .checked_add(product(&[permutation_count, n, 8])?)
                .and_then(|n| n.checked_add(kept.resident_bytes()))
                .ok_or(crate::Error::Overflow)?,
        )?;
        let _matrix_work = ctx.reserve(product(&[m, m, 16, 12])?)?;
        // The Bloch orbital stacks of all k-points go to the scratch folder together when their
        // total reaches the offload threshold (a stack the caller already archived stays as is).
        let archive_psi = ctx.offloads(product(&[
            raw.len(),
            m,
            n,
            if precision == Precision::Float32 { 8 } else { 16 },
        ])?);
        let points = raw
            .iter()
            .map(|v| {
                let mut point = kpoint_arrays(ctx, v, m, p, n, &mut arrays, precision)?;
                if archive_psi && !point.psi.archived() {
                    point.psi = std::mem::replace(
                        &mut point.psi,
                        ComplexSamples::Float64(ctx.allocate(0)?),
                    )
                    .into_archived(ctx)?;
                }
                Ok(point)
            })
            .collect::<Result<Vec<_>>>()?;
        if points.iter().any(|p| p.weight <= 0.)
            || (points.iter().map(|p| p.weight).sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err(invalid("k-point weights must be positive and sum to one"));
        }
        let permutations: Vec<Vec<usize>> = if let Some(key) = v["permutations"].as_str() {
            let Some(InputArray::Indices(indices)) = arrays.0.remove(key) else {
                return Err(invalid("missing symmetry index array"));
            };
            if indices.as_slice().len() != product(&[permutation_count, n])? {
                return Err(invalid("symmetry index array has wrong length"));
            }
            indices
                .as_slice()
                .chunks(n)
                .map(|row| {
                    row.iter()
                        .map(|&x| {
                            usize::try_from(x).map_err(|_| invalid("symmetry index overflow"))
                        })
                        .collect()
                })
                .collect::<Result<_>>()?
        } else {
            v["permutations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|perm| {
                    perm.as_array()
                        .filter(|p| p.len() == n)
                        .ok_or_else(|| invalid("symmetry permutation has wrong length"))?
                        .iter()
                        .map(|i| count(i, "symmetry index"))
                        .collect()
                })
                .collect::<Result<_>>()?
        };
        for perm in &permutations {
            let _scratch = ctx.reserve(n)?;
            let mut seen = vec![false; n];
            for &i in perm {
                if i >= n || seen[i] {
                    return Err(invalid("symmetry indices must form a permutation"));
                }
                seen[i] = true;
            }
        }
        let blocks = arrays.complex(
            ctx,
            &v["blocks"],
            product(&[channels, p, p])?,
            "moment blocks",
        )?;
        let labels = v["channel_labels"]
            .as_array()
            .filter(|a| a.len() == channels)
            .ok_or_else(|| invalid("moment channel labels are required"))?;
        let labels = labels
            .iter()
            .map(|v| {
                let a = v
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .ok_or_else(|| invalid("invalid moment label"))?;
                let atom = count(&a[0], "moment atom")?;
                let l = count(&a[1], "moment degree")?;
                let m = a[2]
                    .as_i64()
                    .filter(|m| l <= 16 && m.unsigned_abs() <= l as u64)
                    .ok_or_else(|| invalid("invalid moment order"))?;
                Ok((atom, l, m))
            })
            .collect::<Result<Vec<_>>>()?;
        for (i, &(atom, l, m_order)) in labels.iter().enumerate() {
            if labels[..i].contains(&(atom, l, m_order)) {
                return Err(invalid("duplicate moment label"));
            }
            let j = labels
                .iter()
                .position(|v| *v == (atom, l, -m_order))
                .ok_or_else(|| invalid("moment channels need conjugate partners"))?;
            let a = DMatrix::from_row_slice(p, p, &blocks.as_slice()[i * p * p..(i + 1) * p * p]);
            let b = DMatrix::from_row_slice(p, p, &blocks.as_slice()[j * p * p..(j + 1) * p * p]);
            let phase = if m_order.abs() % 2 == 0 { 1. } else { -1. };
            if (a.adjoint() * C64::new(phase, 0.) - b)
                .iter()
                .any(|v| v.norm() > 1e-9)
            {
                return Err(invalid(
                    "moment blocks violate spherical-harmonic conjugation",
                ));
            }
        }
        let u = arrays.complex(
            ctx,
            &v["u"],
            product(&[channels, channels])?,
            "compact Coulomb",
        )?;
        hermitian(
            &DMatrix::from_row_slice(channels, channels, u.as_slice()),
            "compact Coulomb",
        )?;
        let kernel = arrays.real(ctx, &v["kernel"], n, "Coulomb kernel")?;
        if kernel.as_slice()[0].abs() > 1e-14 || kernel.as_slice().iter().any(|v| *v < 0.) {
            return Err(invalid(
                "Coulomb kernel must be nonnegative with G=0 dropped",
            ));
        }
        let electrons = number(&v["electrons"], "electrons")?;
        let spins = count(&v["spins"], "spin channels")?;
        let spinor = v["spinor"].as_bool().unwrap_or(false);
        let time_reversal = v["time_reversal"].as_bool().unwrap_or(false);
        if electrons < 0. || electrons > 2. * m as f64 + 1e-10 || ![1, 2].contains(&spins) {
            return Err(invalid("invalid periodic electron count or spin channels"));
        }
        if time_reversal && !spinor {
            return Err(invalid(
                "spinor time-reversal constraint requires a spinor crystal",
            ));
        }
        if spinor && (spins != 1 || !permutations.is_empty()) {
            return Err(invalid(
                "spinor crystals need one spinor block without scalar symmetry averaging",
            ));
        }
        if points
            .iter()
            .any(|point| point.spin_orbit.is_some() != spinor)
        {
            return Err(invalid(
                "spin-orbit matrices must be supplied at every k-point of a spinor crystal",
            ));
        }
        let definition = crate::models::functional(
            v["functional"]
                .as_str()
                .ok_or_else(|| invalid("missing functional"))?,
        )?;
        if definition.screened_exchange.is_some() && !v["hubbard"].is_null() {
            return Err(invalid("DFT+U is not available with hybrid functionals"));
        }
        if definition.screened_exchange.is_some() {
            hybrid_needs_the_full_mesh(&points, permutations.len())?;
        }
        let hubbard = super::hubbard::Spec::parse(
            &v["hubbard"],
            p,
            spins,
            spinor,
            precision,
            permutations.len(),
        )?;
        let functional = definition.name.to_string();
        let hybrid = match definition.screened_exchange {
            Some(range) => Some(super::hybrid_setup::Hybrid::parse(
                ctx,
                &grid,
                p,
                &labels
                    .iter()
                    .map(|&(atom, l, order)| (atom, l, order as i32))
                    .collect::<Vec<_>>(),
                blocks.as_slice(),
                &v["sites"],
                range,
                spinor,
                time_reversal,
            )?),
            None => None,
        };
        // The block serves the derivative engines only: a block that cannot be used (an analytic basis,
        // or one the snapshot contradicts) leaves the energy intact and is refused, with its cause, when
        // a force or a stress asks for it. Only an invalid block is set aside; a budget error is not.
        let mut description_unavailable = v["setup"]["unavailable"].as_str().map(str::to_string);
        let description = if v["setup"].is_null() || description_unavailable.is_some() {
            None
        } else {
            let cell =
                nalgebra::Matrix3::from_fn(|i, j| grid.index.shape()[j] as f64 * grid.step[(i, j)]);
            match super::description::Description::parse(
                ctx,
                &v,
                m,
                p,
                &cell,
                grid.index.shape(),
                &permutations,
            ) {
                Ok(d) => Some(d),
                Err(crate::Error::Invalid(why)) => {
                    description_unavailable = Some(format!("its 'setup' block is refused: {why}"));
                    None
                }
                Err(other) => return Err(other),
            }
        };
        let core_tau = if definition.requires_tau {
            let values = arrays.real(ctx, &v["core_tau"], n, "core kinetic density")?;
            if values.as_slice().iter().any(|v| *v < 0.) {
                return Err(invalid("core kinetic density must be nonnegative"));
            }
            Some(values)
        } else {
            None
        };
        let result = Self {
            context: ctx.clone(),
            grid,
            m,
            p,
            channels,
            points,
            blocks,
            u,
            kernel,
            electrons,
            spins,
            spinor,
            time_reversal,
            functional,
            core: arrays.real(ctx, &v["core"], n, "core density")?,
            core_tau,
            precision,
            initial: arrays.real(ctx, &v["initial"], n, "initial density")?,
            initial_m: arrays.real(
                ctx,
                &v["initial_m"],
                product(&[if spinor { 3 } else { 1 }, n])?,
                "initial magnetization",
            )?,
            initial_q: arrays.complex(ctx, &v["initial_q"], channels, "initial moments")?,
            xc_wave: arrays.real(
                ctx,
                &v["xc_wave"],
                product(&[3, n])?,
                "XC derivative symbols",
            )?,
            phase: arrays.complex(ctx, &v["phase"], n, "Fourier phase")?,
            kerker: arrays.real(ctx, &v["kerker"], n, "Kerker factors")?,
            ion: arrays.complex(ctx, &v["ion"], n, "ionic reciprocal density")?,
            hats: {
                let hats = arrays.samples(
                    ctx,
                    &v["hats"],
                    product(&[channels, n])?,
                    Precision::Float64,
                )?;
                if ctx.offloads(product(&[channels, n, 16])?) {
                    hats.into_archived(ctx)?
                } else {
                    hats
                }
            },
            uion: arrays.complex(ctx, &v["uion"], channels, "compact ion couplings")?,
            linear: arrays.complex(ctx, &v["linear"], channels, "short-range moment couplings")?,
            eion: number(&v["eion"], "ionic energy")?,
            ion_constant: number(&v["ion_constant"], "ionic constant")?,
            constant: number(&v["constant"], "constant energy")?,
            reference: number(&v["reference"], "eigenvalue reference")?,
            relativistic: v["relativistic"]
                .as_bool()
                .ok_or_else(|| invalid("relativistic must be boolean"))?,
            permutations,
            qsym: arrays.complex(
                ctx,
                &v["qsym"],
                product(&[channels, channels])?,
                "moment symmetrization",
            )?,
            hubbard,
            hybrid,
            description,
            description_unavailable,
            source: kept,
            _memory: memory,
        };
        if !arrays.0.is_empty() {
            return Err(invalid("unreferenced arrays in crystal construction"));
        }
        if result.time_reversal && result.initial_m.as_slice().iter().any(|v| *v != 0.) {
            return Err(invalid(
                "time-reversal spinor SCF requires zero initial magnetization",
            ));
        }
        Ok(result)
    }
    pub fn load(ctx: &Context, path: &str) -> Result<Self> {
        let n = std::fs::metadata(path)
            .map_err(|e| invalid(format!("crystal input: {e}")))?
            .len();
        let n = usize::try_from(n).map_err(|_| crate::Error::Overflow)?;
        let _memory = ctx.reserve(n)?;
        let text =
            std::fs::read_to_string(path).map_err(|e| invalid(format!("crystal input: {e}")))?;
        Self::from_json(ctx, &text)
    }
    pub fn save(&self, path: &str) -> Result<()> {
        let mut value: Value = serde_json::from_str(&self.source.text(&self.context)?)
            .map_err(|e| invalid(e.to_string()))?;
        value["functional"] = serde_json::json!(self.functional);
        value["precision"] = serde_json::json!(self.precision.name());
        value["spinor"] = serde_json::json!(self.spinor);
        value["time_reversal"] = serde_json::json!(self.time_reversal);
        if let Some(core) = &self.core_tau {
            value["core_tau"] = serde_json::json!(core.as_slice());
        }
        for (key, buffer) in [
            ("blocks", &self.blocks),
            ("initial_q", &self.initial_q),
            ("phase", &self.phase),
            ("ion", &self.ion),
            ("u", &self.u),
            ("uion", &self.uion),
            ("linear", &self.linear),
            ("qsym", &self.qsym),
        ] {
            value[key] = serde_json::json!(buffer
                .as_slice()
                .iter()
                .map(|x| [x.re, x.im])
                .collect::<Vec<_>>());
        }
        value["hats"] = serde_json::json!(self
            .hats
            .read(&self.context)?
            .iter()
            .map(|v| [v.re, v.im])
            .collect::<Vec<_>>());
        for (key, buffer) in [
            ("core", &self.core),
            ("initial", &self.initial),
            ("initial_m", &self.initial_m),
            ("kernel", &self.kernel),
            ("xc_wave", &self.xc_wave),
            ("kerker", &self.kerker),
        ] {
            value[key] = serde_json::json!(buffer.as_slice());
        }
        value["permutations"] = serde_json::json!(self.permutations);
        if let Some(hubbard) = &self.hubbard {
            value["hubbard"] = hubbard.to_json();
        }
        for (row, point) in value["kpoints"]
            .as_array_mut()
            .ok_or_else(|| invalid("missing kpoint metadata"))?
            .iter_mut()
            .zip(&self.points)
        {
            row["psi"] = serde_json::json!(point
                .psi
                .read(&self.context)?
                .iter()
                .map(|x| [x.re, x.im])
                .collect::<Vec<_>>());
            if let Some(matrix) = &point.spin_orbit {
                row["spin_orbit"] = serde_json::json!((0..matrix.nrows())
                    .flat_map(|i| (0..matrix.ncols())
                        .map(move |j| [matrix[(i, j)].re, matrix[(i, j)].im]))
                    .collect::<Vec<_>>());
            }
            for (key, matrix) in [
                ("overlap", &point.overlap),
                ("fixed", &point.fixed),
                ("projections", &point.projections),
            ] {
                row[key] = serde_json::json!((0..matrix.nrows())
                    .flat_map(|i| (0..matrix.ncols())
                        .map(move |j| [matrix[(i, j)].re, matrix[(i, j)].im]))
                    .collect::<Vec<_>>());
            }
        }
        crate::persistence::atomic_write(std::path::Path::new(path), |writer| {
            serde_json::to_writer(writer, &value)
                .map_err(|e| invalid(format!("crystal output: {e}")))
        })
    }

    /// Number of bands in each independently diagonalized spin block.
    pub fn bands(&self) -> usize {
        self.m * if self.spinor { 2 } else { 1 }
    }
    /// Charge and three magnetization components for a spinor density.
    pub fn field_channels(&self) -> usize {
        if self.spinor && !self.time_reversal {
            4
        } else {
            self.spins
        }
    }
    pub fn degeneracy(&self) -> f64 {
        if self.spins == 1 && !self.spinor {
            2.
        } else {
            1.
        }
    }
}
