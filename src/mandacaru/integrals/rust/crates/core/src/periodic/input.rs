//! Validated portable density-independent crystal data. No Python dependency.
use super::invalid;
use crate::{
    molecular::{scf::eigen, SpatialGrid},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::{DMatrix, DVector};
use serde_json::Value;
pub struct KPoint {
    pub k: [f64; 3],
    pub fractional: [f64; 3],
    pub weight: f64,
    pub psi: Buffer<C64>,
    pub overlap: DMatrix<C64>,
    pub fixed: DMatrix<C64>,
    pub projections: DMatrix<C64>,
    pub lowdin: DMatrix<C64>,
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
    pub initial: Buffer<f64>,
    pub initial_m: Buffer<f64>,
    pub initial_q: Buffer<C64>,
    pub kernel: Buffer<f64>,
    pub xc_wave: Buffer<f64>,
    pub phase: Buffer<C64>,
    pub kerker: Buffer<f64>,
    pub ion: Buffer<C64>,
    pub hats: Buffer<C64>,
    pub u: Buffer<C64>,
    pub uion: Buffer<C64>,
    pub linear: Buffer<C64>,
    pub eion: f64,
    pub ion_constant: f64,
    pub constant: f64,
    pub reference: f64,
    pub electrons: f64,
    pub spins: usize,
    pub functional: String,
    pub relativistic: bool,
    pub permutations: Vec<Vec<usize>>,
    pub qsym: Buffer<C64>,
    source: String,
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
pub fn kpoint(ctx: &Context, value: &Value, m: usize, p: usize, n: usize) -> Result<KPoint> {
    let psi = complex(ctx, &value["psi"], product(&[m, n])?, "Bloch samples")?;
    let overlap = complex(ctx, &value["overlap"], product(&[m, m])?, "overlap")?;
    let overlap = DMatrix::from_row_slice(m, m, overlap.as_slice());
    hermitian(&overlap, "overlap")?;
    let fixed = complex(ctx, &value["fixed"], product(&[m, m])?, "fixed Hamiltonian")?;
    let fixed = DMatrix::from_row_slice(m, m, fixed.as_slice());
    hermitian(&fixed, "fixed Hamiltonian")?;
    let projections = complex(
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
    Ok(KPoint {
        k: vector(&value["k"], "Cartesian k-point")?,
        fractional: vector(&value["fractional"], "fractional k-point")?,
        weight: number(&value["weight"], "k-point weight")?,
        psi,
        overlap,
        fixed,
        projections,
        lowdin,
    })
}
impl Crystal {
    pub fn from_json(ctx: &Context, source: &str) -> Result<Self> {
        let _parse = ctx.reserve(product(&[source.len(), 32])?)?;
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
        let step = real(ctx, &v["grid"]["step"], 9, "grid step")?;
        let grid = SpatialGrid::new(
            shape,
            vector(&v["grid"]["origin"], "grid origin")?,
            step.as_slice().try_into().unwrap(),
            true,
        )?;
        let n = grid.index.points();
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
        let permutations_raw = v["permutations"]
            .as_array()
            .ok_or_else(|| invalid("invalid symmetry permutations"))?;
        let matrices = product(&[
            raw.len(),
            m,
            m.checked_add(p).ok_or(crate::Error::Overflow)?,
            16,
            12,
        ])?;
        let memory = ctx.reserve(
            matrices
                .checked_add(product(&[permutations_raw.len(), n, 8])?)
                .and_then(|n| n.checked_add(source.len()))
                .ok_or(crate::Error::Overflow)?,
        )?;
        let points = raw
            .iter()
            .map(|v| kpoint(ctx, v, m, p, n))
            .collect::<Result<Vec<_>>>()?;
        if points.iter().any(|p| p.weight <= 0.)
            || (points.iter().map(|p| p.weight).sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err(invalid("k-point weights must be positive and sum to one"));
        }
        let mut permutations = vec![];
        for perm in permutations_raw {
            let perm = perm
                .as_array()
                .filter(|p| p.len() == n)
                .ok_or_else(|| invalid("symmetry permutation has wrong length"))?;
            let perm = perm
                .iter()
                .map(|v| count(v, "symmetry index"))
                .collect::<Result<Vec<_>>>()?;
            let _scratch = ctx.reserve(n)?;
            let mut seen = vec![false; n];
            for &i in &perm {
                if i >= n || seen[i] {
                    return Err(invalid("symmetry indices must form a permutation"));
                }
                seen[i] = true;
            }
            permutations.push(perm);
        }
        let blocks = complex(
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
        let u = complex(
            ctx,
            &v["u"],
            product(&[channels, channels])?,
            "compact Coulomb",
        )?;
        hermitian(
            &DMatrix::from_row_slice(channels, channels, u.as_slice()),
            "compact Coulomb",
        )?;
        let kernel = real(ctx, &v["kernel"], n, "Coulomb kernel")?;
        if kernel.as_slice()[0].abs() > 1e-14 || kernel.as_slice().iter().any(|v| *v < 0.) {
            return Err(invalid(
                "Coulomb kernel must be nonnegative with G=0 dropped",
            ));
        }
        let electrons = number(&v["electrons"], "electrons")?;
        let spins = count(&v["spins"], "spin channels")?;
        if electrons < 0. || electrons > 2. * m as f64 + 1e-10 || ![1, 2].contains(&spins) {
            return Err(invalid("invalid periodic electron count or spin channels"));
        }
        let functional = crate::molecular::xc::functional(
            v["functional"]
                .as_str()
                .ok_or_else(|| invalid("missing functional"))?,
        )?
        .to_string();
        Ok(Self {
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
            functional,
            core: real(ctx, &v["core"], n, "core density")?,
            initial: real(ctx, &v["initial"], n, "initial density")?,
            initial_m: real(ctx, &v["initial_m"], n, "initial magnetization")?,
            initial_q: complex(ctx, &v["initial_q"], channels, "initial moments")?,
            xc_wave: real(
                ctx,
                &v["xc_wave"],
                product(&[3, n])?,
                "XC derivative symbols",
            )?,
            phase: complex(ctx, &v["phase"], n, "Fourier phase")?,
            kerker: real(ctx, &v["kerker"], n, "Kerker factors")?,
            ion: complex(ctx, &v["ion"], n, "ionic reciprocal density")?,
            hats: complex(
                ctx,
                &v["hats"],
                product(&[channels, n])?,
                "compensation form factors",
            )?,
            uion: complex(ctx, &v["uion"], channels, "compact ion couplings")?,
            linear: complex(ctx, &v["linear"], channels, "short-range moment couplings")?,
            eion: number(&v["eion"], "ionic energy")?,
            ion_constant: number(&v["ion_constant"], "ionic constant")?,
            constant: number(&v["constant"], "constant energy")?,
            reference: number(&v["reference"], "eigenvalue reference")?,
            relativistic: v["relativistic"]
                .as_bool()
                .ok_or_else(|| invalid("relativistic must be boolean"))?,
            permutations,
            qsym: complex(
                ctx,
                &v["qsym"],
                product(&[channels, channels])?,
                "moment symmetrization",
            )?,
            source: source.to_string(),
            _memory: memory,
        })
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
        std::fs::write(path, &self.source).map_err(|e| invalid(format!("crystal output: {e}")))
    }
}
