//! Analytic HAO and Mandacaru Gaussian recipes. Unit Slater fits are immutable
//! generated data; scaling, shell selection and sampling are native operations.
use super::{invalid, SpatialGrid};
use crate::{product, Buffer, Context, Geometry, Reservation, Result};
use serde_json::Value;
use std::{collections::BTreeMap, f64::consts::PI};

const AUFBAU: [(usize, usize); 19] = [
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
];
fn configuration(z: u8) -> Vec<(usize, usize, usize)> {
    let mut left = z as usize;
    let mut out = Vec::new();
    for (n, l) in AUFBAU {
        if left == 0 {
            break;
        }
        let occupied = left.min(2 * (2 * l + 1));
        left -= occupied;
        out.push((n, l, occupied));
    }
    out.sort_unstable();
    out
}
fn slater(z: u8, n: usize, l: usize) -> Result<f64> {
    let mut groups = BTreeMap::new();
    for (nn, ll, occupied) in configuration(z) {
        *groups.entry((nn, ll.saturating_sub(1))).or_insert(0usize) += occupied;
    }
    let target = (n, l.saturating_sub(1));
    if !groups.contains_key(&target) {
        return Err(invalid(
            "Slater scaling requires an occupied neutral-atom subshell",
        ));
    }
    let mut screening = 0.;
    for (group, occupied) in groups {
        if group == target {
            screening += (if n == 1 { 0.30 } else { 0.35 }) * (occupied - 1) as f64;
        } else if group < target {
            screening += occupied as f64
                * if target.1 == 0 && group.0 == n - 1 {
                    0.85
                } else {
                    1.
                };
        }
    }
    let n_star = match n {
        1 => 1.,
        2 => 2.,
        3 => 3.,
        4 => 3.7,
        5 => 4.,
        6 => 4.2,
        _ => {
            return Err(invalid(
                "Gaussian Slater recipes support principal shells 1..6",
            ))
        }
    };
    Ok((z as f64 - screening) / n_star)
}
#[derive(Clone, Debug)]
pub enum Radial {
    Hydrogenic {
        n: usize,
        z: f64,
        norm: f64,
    },
    Gaussian {
        exponents: Vec<f64>,
        weighted: Vec<f64>,
    },
}
#[derive(Clone, Debug)]
pub struct Orbital {
    pub atom: usize,
    pub l: usize,
    pub m: i32,
    pub center: [f64; 3],
    pub radial: Radial,
}
pub struct Basis {
    pub orbitals: Vec<Orbital>,
    _memory: Vec<Reservation>,
}
fn integers(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or_else(|| invalid("basis recipe needs integer lists"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid("basis recipe needs nonnegative integers"))
        })
        .collect()
}
fn pairs(value: &Value) -> Result<Vec<(usize, usize)>> {
    value
        .as_array()
        .ok_or_else(|| invalid("basis recipe needs angular/count pairs"))?
        .iter()
        .map(|v| {
            let a = integers(v)?;
            if a.len() != 2 || a[0] > 8 || a[1] > 20 {
                return Err(invalid("invalid angular/count pair"));
            }
            Ok((a[0], a[1]))
        })
        .collect()
}
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1., |a, n| a * n as f64)
}
fn gaussian(l: usize, exponents: Vec<f64>, coefficients: Vec<f64>) -> Result<Radial> {
    if l > 8
        || exponents.len() != coefficients.len()
        || exponents.is_empty()
        || exponents.iter().any(|a| !a.is_finite() || *a <= 0.)
        || coefficients.iter().any(|c| !c.is_finite())
    {
        return Err(invalid("invalid Gaussian contraction"));
    }
    let power = l as f64 + 1.5;
    let gamma = libm::tgamma(power);
    let mut weighted: Vec<_> = exponents
        .iter()
        .zip(coefficients)
        .map(|(a, c)| c * (2. * (2. * a).powf(power) / gamma).sqrt())
        .collect();
    let mut overlap = 0.;
    for (a, wa) in exponents.iter().zip(&weighted) {
        for (b, wb) in exponents.iter().zip(&weighted) {
            overlap += wa * wb * gamma / (2. * (a + b).powf(power));
        }
    }
    if !overlap.is_finite() || overlap <= 0. {
        return Err(invalid("Gaussian contraction has zero or invalid norm"));
    }
    for w in &mut weighted {
        *w /= overlap.sqrt();
    }
    Ok(Radial::Gaussian {
        exponents,
        weighted,
    })
}
type Shell = (usize, Vec<f64>, Vec<f64>);
fn pattern(fits: &Value, z: u8, n: usize, l: usize, pattern: &[usize]) -> Result<Vec<Shell>> {
    let count: usize = pattern.iter().sum();
    if pattern.contains(&0) || count > 10 {
        return Err(invalid(
            "Gaussian contraction patterns must contain 1..10 total primitives",
        ));
    }
    let key = format!("{n},{l},{count}");
    let fit = &fits["fits"][&key];
    if fit.is_null() {
        return Err(invalid(format!(
            "unit Slater fit {key} is unavailable in the native basis data"
        )));
    }
    let zeta = slater(z, n, l)?;
    let exponents: Vec<_> = fit["exponents"]
        .as_array()
        .ok_or_else(|| invalid("invalid native fit data"))?
        .iter()
        .map(|v| v.as_f64().unwrap() * zeta * zeta)
        .collect();
    let coefficients: Vec<_> = fit["coefficients"]
        .as_array()
        .ok_or_else(|| invalid("invalid native fit data"))?
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let mut start = 0;
    let mut shells = Vec::new();
    for &width in pattern {
        let stop = start + width;
        shells.push((
            l,
            exponents[start..stop].to_vec(),
            if width == 1 {
                vec![1.]
            } else {
                coefficients[start..stop].to_vec()
            },
        ));
        start = stop;
    }
    Ok(shells)
}
fn gaussian_shells(fits: &Value, z: u8, recipe: &Value) -> Result<Vec<Shell>> {
    let config = configuration(z);
    let valence = config.iter().map(|a| a.0).max().unwrap();
    let core = integers(&recipe["core"])?;
    let valence_pattern = integers(&recipe["valence"])?;
    let mut shells = Vec::new();
    let mut outer = Vec::new();
    for (n, l, _) in config {
        if n < valence {
            shells.extend(pattern(fits, z, n, l, &core)?);
        } else {
            outer.extend(pattern(fits, z, n, l, &valence_pattern)?);
        }
    }
    shells.extend(outer.iter().cloned());
    if z >= 3 {
        for (l, count) in pairs(&recipe["tight"])? {
            let tightest = outer
                .iter()
                .filter(|s| s.0 == l)
                .flat_map(|s| s.1.iter())
                .copied()
                .reduce(f64::max);
            if let Some(tightest) = tightest {
                for k in 0..count {
                    shells.push((l, vec![tightest * 3f64.powi(k as i32 + 1)], vec![1.]));
                }
            }
        }
    }
    let min_z = recipe["polarization_min_z"].as_u64().unwrap_or(3);
    let pol = if z as u64 >= min_z {
        pairs(&recipe["polarization"])?
    } else if z <= 2 {
        pairs(&recipe["polarization_h"])?
    } else {
        vec![]
    };
    for (l, count) in pol {
        let factor = if z <= 2 {
            match l {
                1 => 0.9,
                2 => 1.2,
                3 => 1.5,
                4 => 1.8,
                5 => 2.1,
                _ => 0.2 * l as f64,
            }
        } else {
            match l {
                2 => 0.4,
                3 => 0.6,
                4 => 0.8,
                5 => 1.,
                6 => 1.2,
                _ => 0.2 * l as f64,
            }
        };
        let base = factor * slater(z, valence, 0)?.powi(2);
        for k in 0..count {
            shells.push((
                l,
                vec![base * 3f64.powf((count - 1) as f64 / 2. - k as f64)],
                vec![1.],
            ));
        }
    }
    let mut diffuse = if recipe["diffuse_present"].as_bool().unwrap_or(false) {
        let max_l = recipe["diffuse_l_max"].as_u64().unwrap_or(99) as usize;
        shells
            .iter()
            .map(|s| s.0)
            .filter(|l| *l <= max_l)
            .collect::<Vec<_>>()
    } else {
        integers(&recipe[if z >= 3 { "diffuse" } else { "diffuse_h" }])?
    };
    diffuse.sort_unstable();
    diffuse.dedup();
    for l in diffuse {
        let minimum = |angular: usize| {
            shells
                .iter()
                .filter(|s| s.0 == angular)
                .flat_map(|s| s.1.iter())
                .copied()
                .reduce(f64::min)
        };
        let exponent = minimum(l)
            .or_else(|| minimum(0))
            .ok_or_else(|| invalid("diffuse function needs a parent shell"))?
            / 3.5;
        shells.push((l, vec![exponent], vec![1.]));
    }
    Ok(shells)
}
impl Basis {
    pub fn build(ctx: &Context, geometry: &Geometry, spec: &str) -> Result<Self> {
        if spec.len() > 1024 * 1024 {
            return Err(invalid("basis specification exceeds the 1 MiB input limit"));
        }
        let _input = ctx.reserve(product(&[spec.len(), 32])?)?;
        let spec: Value = serde_json::from_str(spec)
            .map_err(|e| invalid(format!("invalid basis specification: {e}")))?;
        let name = spec["name"]
            .as_str()
            .ok_or_else(|| invalid("basis needs a name"))?;
        let mut orbitals = Vec::new();
        let data = include_str!("../../data/slater_fits.json");
        let _fit_memory = ctx.reserve(if name == "HAO" {
            0
        } else {
            product(&[data.len(), 24])?
        })?;
        let fits: Value = if name == "HAO" {
            Value::Null
        } else {
            serde_json::from_str(data).map_err(|e| invalid(e.to_string()))?
        };
        let mut guards = Vec::new();
        for (atom, (&z, &center)) in geometry
            .numbers()
            .iter()
            .zip(geometry.positions())
            .enumerate()
        {
            let atom_spec = if name == "per_element" {
                spec["elements"]
                    .get(z.to_string())
                    .or_else(|| spec["elements"].get("*"))
                    .ok_or_else(|| {
                        invalid(format!("basis has no specification for atomic number {z}"))
                    })?
            } else {
                &spec
            };
            let name = atom_spec["name"]
                .as_str()
                .ok_or_else(|| invalid("per-element basis needs a name"))?;
            let spec = atom_spec;
            if name == "HAO" {
                let count = match spec.get("virtual_orbitals") {
                    None => 0,
                    Some(v) => v
                        .as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| {
                            invalid("HAO virtual_orbitals must be a nonnegative integer")
                        })?,
                };
                let occupied = configuration(z);
                let mut shells: Vec<_> = occupied.iter().map(|s| (s.0, s.1)).collect();
                let empty: Vec<_> = AUFBAU
                    .iter()
                    .filter(|s| !shells.contains(s))
                    .copied()
                    .collect();
                if count > empty.len() {
                    return Err(invalid("HAO virtual shell count exceeds the filling table"));
                }
                shells.extend(&empty[..count]);
                for (n, l) in shells {
                    for m in -(l as i32)..=l as i32 {
                        guards.push(ctx.reserve(2048)?);
                        let norm = ((2. * z as f64 / n as f64).powi(3) * factorial(n - l - 1)
                            / (2. * n as f64 * factorial(n + l)))
                        .sqrt();
                        orbitals.push(Orbital {
                            atom,
                            l,
                            m,
                            center,
                            radial: Radial::Hydrogenic {
                                n,
                                z: z as f64,
                                norm,
                            },
                        });
                    }
                }
            } else if name == "gaussian" {
                for (l, exponents, coefficients) in gaussian_shells(&fits, z, &spec["recipe"])? {
                    let radial = gaussian(l, exponents, coefficients)?;
                    for m in -(l as i32)..=l as i32 {
                        guards.push(ctx.reserve(2048)?);
                        orbitals.push(Orbital {
                            atom,
                            l,
                            m,
                            center,
                            radial: radial.clone(),
                        });
                    }
                }
            } else {
                return Err(invalid(
                    "native molecular bases are HAO or a Gaussian recipe",
                ));
            }
        }
        Ok(Self {
            orbitals,
            _memory: guards,
        })
    }
    pub fn sample(&self, ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
        let g = grid.index.points();
        let mut out = ctx.allocate(product(&[self.orbitals.len(), g])?)?;
        for (a, orbital) in self.orbitals.iter().enumerate() {
            for k in 0..g {
                out.as_mut_slice()[a * g + k] = orbital.evaluate(grid.point(k));
            }
        }
        Ok(out)
    }
}
impl Orbital {
    pub fn exact_kinetic(&self) -> f64 {
        match &self.radial {
            Radial::Hydrogenic { n, z, .. } => z * z / (2. * (*n as f64).powi(2)),
            Radial::Gaussian {
                exponents,
                weighted,
            } => {
                let l = self.l as f64;
                let mut total = 0.;
                for (a, da) in exponents.iter().zip(weighted) {
                    for (b, db) in exponents.iter().zip(weighted) {
                        let integral = |power: f64| {
                            libm::tgamma((power + 1.) / 2.) / (2. * (a + b).powf((power + 1.) / 2.))
                        };
                        total += 0.5
                            * da
                            * db
                            * (l * (2. * l + 1.) * integral(2. * l)
                                - 2. * l * (a + b) * integral(2. * l + 2.)
                                + 4. * a * b * integral(2. * l + 4.));
                    }
                }
                total
            }
        }
    }
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        let [x, y, z] = std::array::from_fn(|j| point[j] - self.center[j]);
        let r = (x * x + y * y + z * z).sqrt().max(1e-15);
        if !r.is_finite() || r > 1e100 {
            return 0.;
        }
        let radial = match &self.radial {
            Radial::Gaussian {
                exponents,
                weighted,
            } => {
                exponents
                    .iter()
                    .zip(weighted)
                    .map(|(a, d)| d * (-a * r * r).exp())
                    .sum::<f64>()
                    * r.powi(self.l as i32)
            }
            Radial::Hydrogenic { n, z, norm } => {
                let rho = 2. * z * r / (*n as f64);
                let order = n - self.l - 1;
                let alpha = 2. * self.l as f64 + 1.;
                let mut previous = 1.;
                let mut value = if order == 0 { 1. } else { 1. + alpha - rho };
                for k in 2..=order {
                    let next = ((2. * k as f64 - 1. + alpha - rho) * value
                        - (k as f64 - 1. + alpha) * previous)
                        / k as f64;
                    previous = value;
                    value = next;
                }
                norm * (-rho / 2.).exp() * rho.powi(self.l as i32) * value
            }
        };
        radial * real_harmonic(self.l, self.m, (z / r).clamp(-1., 1.).acos(), y.atan2(x))
    }
}
pub fn real_harmonic(l: usize, m: i32, theta: f64, phi: f64) -> f64 {
    let order = m.unsigned_abs() as usize;
    let cos_t = theta.cos();
    let mut legendre = 1. / (4. * PI).sqrt();
    for k in 1..=order {
        legendre *= -((2 * k + 1) as f64 / (2 * k) as f64).sqrt() * theta.sin();
    }
    if l > order {
        let mut previous = legendre;
        legendre *= ((2 * order + 3) as f64).sqrt() * cos_t;
        for degree in order + 2..=l {
            let d = degree as f64;
            let o = order as f64;
            let a = ((4. * d * d - 1.) / (d * d - o * o)).sqrt();
            let b = (((d - 1.).powi(2) - o * o) / (4. * (d - 1.).powi(2) - 1.)).sqrt();
            let next = a * (cos_t * legendre - b * previous);
            previous = legendre;
            legendre = next;
        }
    }
    if m == 0 {
        legendre
    } else {
        2f64.sqrt()
            * if order % 2 == 0 { 1. } else { -1. }
            * legendre
            * if m > 0 {
                (order as f64 * phi).cos()
            } else {
                (order as f64 * phi).sin()
            }
    }
}
