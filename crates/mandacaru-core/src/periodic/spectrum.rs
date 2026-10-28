//! Metallic occupations, reciprocal meshes and spectral observables.
use super::invalid;
use crate::{finite, product, Buffer, Context, Reservation, Result};
use std::{collections::HashMap, f64::consts::PI};
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smearing {
    FermiDirac,
    Gaussian,
    MethfesselPaxton,
}
impl Smearing {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "fermi-dirac" | "fd" | "fermi" => Ok(Self::FermiDirac),
            "gaussian" => Ok(Self::Gaussian),
            "methfessel-paxton" | "mp" => Ok(Self::MethfesselPaxton),
            _ => Err(invalid("unknown smearing method")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::FermiDirac => "fermi-dirac",
            Self::Gaussian => "gaussian",
            Self::MethfesselPaxton => "methfessel-paxton",
        }
    }
    pub fn occupation(self, x: f64) -> f64 {
        match self {
            Self::FermiDirac => 0.5 * (1. - (0.5 * x).tanh()),
            Self::Gaussian => 0.5 * libm::erfc(x),
            Self::MethfesselPaxton => 0.5 * libm::erfc(x) - x * (-x * x).exp() / (2. * PI.sqrt()),
        }
    }
    pub fn entropy(self, x: f64) -> f64 {
        match self {
            Self::FermiDirac => {
                let f = self.occupation(x);
                let term = |x: f64| if x > 0. { x * x.ln() } else { 0. };
                -term(f) - term(1. - f)
            }
            Self::Gaussian => (-x * x).exp() / (2. * PI.sqrt()),
            Self::MethfesselPaxton => -(2. * x * x - 1.) * (-x * x).exp() / (4. * PI.sqrt()),
        }
    }
}
// Pairwise eight-lane reduction avoids moving a saturated insulating Fermi
// bracket merely by changing the nesting of k-point and band summations.
fn occupation_sum(mut values: impl Iterator<Item = f64>, length: usize) -> f64 {
    if length < 8 {
        return values.sum();
    }
    let mut lanes = std::array::from_fn::<_, 8, _>(|_| values.next().unwrap());
    for _ in 0..(length / 8 - 1) {
        for lane in &mut lanes {
            *lane += values.next().unwrap();
        }
    }
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    values.fold(sum, |a, b| a + b)
}
pub fn fermi(
    eps: &[Vec<f64>],
    weights: &[f64],
    electrons: f64,
    method: Smearing,
    width: f64,
    degeneracy: f64,
) -> Result<f64> {
    if eps.is_empty()
        || eps.len() != weights.len()
        || eps
            .iter()
            .any(|v| v.is_empty() || v.iter().any(|x| !x.is_finite()))
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
        || !width.is_finite()
        || width <= 0.
        || !electrons.is_finite()
        || electrons < 0.
        || !degeneracy.is_finite()
        || degeneracy <= 0.
    {
        return Err(invalid(
            "invalid Fermi-level dimensions, energies, electron count or smearing width",
        ));
    }
    let capacity = degeneracy
        * eps
            .iter()
            .zip(weights)
            .map(|(e, w)| e.len() as f64 * w)
            .sum::<f64>();
    if electrons > capacity + 1e-9 || capacity <= 0. {
        return Err(invalid("electron count exceeds the band capacity"));
    }
    let mut lo = eps.iter().flatten().copied().fold(f64::INFINITY, f64::min) - 60. * width;
    let mut hi = eps
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        + 60. * width;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let length = eps.iter().map(Vec::len).sum();
        let count = degeneracy
            * occupation_sum(
                eps.iter().zip(weights).flat_map(|(e, w)| {
                    e.iter()
                        .map(move |e| w * method.occupation((e - mid) / width))
                }),
                length,
            );
        if count < electrons {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}
pub struct Mesh {
    pub full: Vec<[f64; 3]>,
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub mapping: Vec<usize>,
    _memory: Reservation,
}
fn key(v: [f64; 3]) -> [i64; 3] {
    v.map(|x| (((x - x.floor()) * 1e10).round() as i64).rem_euclid(10_000_000_000))
}
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
pub fn mesh(
    ctx: &Context,
    size: [usize; 3],
    gamma: bool,
    operations: &[[i32; 9]],
    time_reversal: bool,
) -> Result<Mesh> {
    if size.contains(&0) {
        return Err(invalid("k-mesh dimensions must be positive"));
    }
    for operation in operations {
        let a = nalgebra::Matrix3::from_row_slice(&operation.map(|v| v as f64));
        if (a.determinant().abs() - 1.).abs() > 1e-6 {
            return Err(invalid(
                "k-point rotations must be integer unimodular matrices",
            ));
        }
    }
    let n = product(&size)?;
    let memory = ctx.reserve(product(&[n, 192])?)?;
    let mut full = Vec::new();
    full.try_reserve_exact(n)
        .map_err(|_| crate::Error::Allocation)?;
    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                let indices = [i, j, k];
                full.push(std::array::from_fn(|a| {
                    (2. * indices[a] as f64 + 1. - size[a] as f64) / (2. * size[a] as f64)
                        + if gamma && size[a] % 2 == 0 {
                            0.5 / size[a] as f64
                        } else {
                            0.
                        }
                }));
            }
        }
    }
    let lookup: HashMap<_, _> = full.iter().enumerate().map(|(i, p)| (key(*p), i)).collect();
    let mut parent: Vec<_> = (0..n).collect();
    for (i, p) in full.iter().enumerate() {
        for operation in operations {
            let transformed =
                std::array::from_fn(|a| (0..3).map(|b| p[b] * operation[3 * b + a] as f64).sum());
            let &j = lookup.get(&key(transformed)).ok_or_else(|| {
                invalid("a supplied symmetry operation does not preserve the k-mesh")
            })?;
            let a = find(&mut parent, i);
            let b = find(&mut parent, j);
            parent[a.max(b)] = a.min(b);
        }
        if time_reversal {
            let &j = lookup
                .get(&key(p.map(|x| -x)))
                .ok_or_else(|| invalid("k-mesh is not time-reversal invariant"))?;
            let a = find(&mut parent, i);
            let b = find(&mut parent, j);
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut representatives = HashMap::new();
    let mut points = vec![];
    let mut counts = vec![];
    let mut mapping = vec![];
    for i in 0..n {
        let root = find(&mut parent, i);
        let next = points.len();
        let index = *representatives.entry(root).or_insert_with(|| {
            points.push(full[root]);
            counts.push(0usize);
            next
        });
        counts[index] += 1;
        mapping.push(index);
    }
    let weights = counts.iter().map(|v| *v as f64 / n as f64).collect();
    Ok(Mesh {
        full,
        points,
        weights,
        mapping,
        _memory: memory,
    })
}
pub fn broaden(
    ctx: &Context,
    energies: &[f64],
    eps: &[Vec<f64>],
    weights: &[f64],
    width: f64,
    degeneracy: f64,
    projections: Option<(&[f64], usize)>,
) -> Result<Buffer<f64>> {
    finite(energies, "DOS energy grid")?;
    if !width.is_finite()
        || width <= 0.
        || !degeneracy.is_finite()
        || degeneracy <= 0.
        || eps.is_empty()
        || eps.len() != weights.len()
    {
        return Err(invalid("invalid DOS dimensions or width"));
    }
    let bands = eps[0].len();
    if bands == 0
        || eps
            .iter()
            .any(|e| e.len() != bands || e.iter().any(|v| !v.is_finite()))
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
    {
        return Err(invalid("invalid DOS bands or weights"));
    }
    let (projection, columns) = projections.unwrap_or((&[], 1));
    if columns == 0
        || (projections.is_some() && projection.len() != product(&[eps.len(), columns, bands])?)
        || projection.iter().any(|p| !p.is_finite() || *p < 0.)
    {
        return Err(invalid("invalid projected DOS weights"));
    }
    let mut out = ctx.allocate::<f64>(product(&[energies.len(), columns])?)?;
    for (k, levels) in eps.iter().enumerate() {
        for (b, &level) in levels.iter().enumerate() {
            for (i, &energy) in energies.iter().enumerate() {
                let peak = degeneracy * weights[k] / (width * (2. * PI).sqrt())
                    * (-0.5 * ((energy - level) / width).powi(2)).exp();
                for column in 0..columns {
                    out.as_mut_slice()[i * columns + column] += peak
                        * if projections.is_some() {
                            projection[(k * columns + column) * bands + b]
                        } else {
                            1.
                        };
                }
            }
        }
    }
    Ok(out)
}
