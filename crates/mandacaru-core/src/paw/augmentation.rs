//! Atom-centered PAW transition densities. Raw projector tables are decoded
//! once; radial reconstruction and reciprocal form factors are native.
use super::{
    invalid,
    molecular::{
        data::{self, Arrays},
        onecenter::shape,
    },
};
use crate::{
    math::{angular, radial::validate, spline::simpson},
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};
use nalgebra::DMatrix;
use serde_json::Value;
use std::f64::consts::PI;

struct Channel {
    l: usize,
    radius: f64,
    ae: Vec<Vec<f64>>,
    ps: Vec<Vec<f64>>,
}
pub struct Projector {
    pub column: usize,
    pub channel: usize,
    pub radial: usize,
    pub m: i32,
}
pub struct Level {
    pub l: usize,
    pub m: i32,
    /// Integral conj(Y_LM) conj(Y_a) Y_b on local projector pairs.
    pub angular: Buffer<f64>,
    pub moments: Buffer<f64>,
}
pub struct Atom {
    pub center: [f64; 3],
    pub radius: f64,
    pub compensation_radius: f64,
    pub projectors: Vec<Projector>,
    pub levels: Vec<Level>,
    radial: Buffer<f64>,
    ae: Vec<Vec<f64>>,
    ps: Vec<Vec<f64>>,
    _memory: Reservation,
}
pub struct Augmentation {
    pub atoms: Vec<Atom>,
    pub projectors: usize,
}

impl Augmentation {
    pub fn from_arrays(
        ctx: &Context,
        source: &str,
        mut arrays: Arrays,
        projectors: usize,
    ) -> Result<Self> {
        let _parse = ctx.reserve(product(&[source.len(), 32])?)?;
        let metadata: Value = serde_json::from_str(source).map_err(|e| invalid(e.to_string()))?;
        let entries = metadata["atoms"]
            .as_array()
            .ok_or_else(|| invalid("PAW augmentation needs atom tables"))?;
        let atoms = entries
            .iter()
            .map(|v| Atom::parse(ctx, v, &mut arrays))
            .collect::<Result<Vec<_>>>()?;
        let columns = atoms
            .iter()
            .flat_map(|a| a.projectors.iter().map(|p| p.column))
            .collect::<Vec<_>>();
        if columns.len() != projectors
            || columns.iter().any(|p| *p >= projectors)
            || columns
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != projectors
            || !arrays.is_empty()
        {
            return Err(invalid(
                "PAW augmentation tables must cover each projector column exactly once",
            ));
        }
        Ok(Self { atoms, projectors })
    }
}
impl Atom {
    fn parse(ctx: &Context, v: &Value, arrays: &mut Arrays) -> Result<Self> {
        let positions = v["center"]
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("PAW atom center needs three Bohr coordinates"))?;
        let center = [
            data::number(&positions[0])?,
            data::number(&positions[1])?,
            data::number(&positions[2])?,
        ];
        let compensation_radius = data::number(&v["compensation_radius"])?;
        if compensation_radius <= 0. {
            return Err(invalid("PAW compensation radius must be positive"));
        }
        let radial = data::table(arrays, &v["r"])?;
        validate(radial.as_slice(), radial.as_slice())?;
        let n = radial.as_slice().len();
        let entries = v["channels"]
            .as_array()
            .ok_or_else(|| invalid("PAW augmentation needs radial channels"))?;
        let mut channels = vec![];
        let total = entries
            .iter()
            .map(|c| c["wavevectors"].as_array().map_or(0, Vec::len))
            .sum::<usize>();
        let memory = ctx.reserve(product(&[total, n, 8, 4])?)?;
        let raw = v["projector_basis"] == "raw";
        if !raw && v["projector_basis"] != "dual" {
            return Err(invalid(
                "PAW augmentation projector basis must be raw or dual",
            ));
        }
        for entry in entries {
            let l = data::count(&entry["l"])?;
            let radius = data::number(&entry["radius"])?;
            if l > 8 || radius <= 0. || radius > radial.as_slice()[n - 1] {
                return Err(invalid("invalid PAW augmentation radial channel"));
            }
            let qkeys = entry["wavevectors"]
                .as_array()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| invalid("PAW channel needs Bessel wavevectors"))?;
            let ckeys = entry["wave_coefficients"]
                .as_array()
                .filter(|v| v.len() == qkeys.len())
                .ok_or_else(|| invalid("PAW channel Bessel coefficients disagree"))?;
            let refs = qkeys.len();
            let values = data::table(arrays, &entry["ae"])?;
            let transform = data::table(arrays, &entry["vanderbilt"])?;
            if values.as_slice().len() != product(&[refs, n])?
                || transform.as_slice().len() != product(&[refs, refs])?
            {
                return Err(invalid("invalid PAW augmentation partial-wave dimensions"));
            }
            let transform = if raw {
                DMatrix::from_row_slice(refs, refs, transform.as_slice())
                    .try_inverse()
                    .ok_or_else(|| invalid("singular PAW Vanderbilt transform"))?
            } else {
                DMatrix::identity(refs, refs)
            };
            let ae = values
                .as_slice()
                .chunks_exact(n)
                .map(|r| r.to_vec())
                .collect::<Vec<_>>();
            let mut ps = vec![vec![0.; n]; refs];
            for i in 0..refs {
                let q = data::table(arrays, &qkeys[i])?;
                let coefficients = data::table(arrays, &ckeys[i])?;
                if q.as_slice().len() != coefficients.as_slice().len() {
                    return Err(invalid("PAW Bessel expansion dimensions disagree"));
                }
                let bessel =
                    crate::kernels::bessel(ctx, l as i32, radial.as_slice(), q.as_slice())?;
                for k in 0..n {
                    ps[i][k] = if radial.as_slice()[k] <= radius {
                        (0..q.as_slice().len())
                            .map(|j| {
                                coefficients.as_slice()[j]
                                    * bessel.as_slice()[k * q.as_slice().len() + j]
                            })
                            .sum()
                    } else {
                        ae[i][k]
                    };
                }
            }
            let change = |waves: &[Vec<f64>]| {
                (0..refs)
                    .map(|i| {
                        (0..n)
                            .map(|k| (0..refs).map(|j| transform[(i, j)] * waves[j][k]).sum())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            channels.push(Channel {
                l,
                radius,
                ae: change(&ae),
                ps: change(&ps),
            });
        }
        let entries = v["projectors"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("PAW augmentation atom needs projector descriptors"))?;
        let projectors = entries.iter().map(|p| {
            let channel = data::count(&p["channel"])?;
            let radial = data::count(&p["radial"])?;
            let column = data::count(&p["column"])?;
            let m = p["m"].as_i64().and_then(|m| i32::try_from(m).ok()).ok_or_else(|| invalid("invalid PAW magnetic quantum number"))?;
            if p["real"]!=false || channel>=channels.len() || radial>=channels[channel].ae.len() || m.unsigned_abs() as usize>channels[channel].l { return Err(invalid("periodic PAW augmentation needs complex spherical projectors with valid radial indices")); }
            Ok(Projector {column,channel,radial,m})
        }).collect::<Result<Vec<_>>>()?;
        let p = projectors.len();
        let lmax = channels.iter().map(|c| c.l).max().unwrap_or(0) * 2;
        let mut levels = vec![];
        for l in 0..=lmax {
            for m in -(l as i32)..=l as i32 {
                let mut angular = ctx.allocate::<f64>(product(&[p, p])?)?;
                let mut moments = ctx.allocate::<f64>(product(&[p, p])?)?;
                for (a, pa) in projectors.iter().enumerate() {
                    for (b, pb) in projectors.iter().enumerate() {
                        let ca = &channels[pa.channel];
                        let cb = &channels[pb.channel];
                        let coupling = angular::gaunt([l, ca.l, cb.l], [m, pa.m, pb.m])?;
                        if coupling.abs() < 1e-12 {
                            continue;
                        }
                        angular.as_mut_slice()[a * p + b] = coupling;
                        let inside = radial
                            .as_slice()
                            .partition_point(|r| *r <= ca.radius.max(cb.radius));
                        let integrand = (0..inside)
                            .map(|k| {
                                (ca.ae[pa.radial][k] * cb.ae[pb.radial][k]
                                    - ca.ps[pa.radial][k] * cb.ps[pb.radial][k])
                                    * radial.as_slice()[k].powi((l + 2) as i32)
                            })
                            .collect::<Vec<_>>();
                        moments.as_mut_slice()[a * p + b] =
                            simpson(&radial.as_slice()[..inside], &integrand)?;
                    }
                }
                if angular.as_slice().iter().any(|v| *v != 0.) {
                    levels.push(Level {
                        l,
                        m,
                        angular,
                        moments,
                    });
                }
            }
        }
        let radius = channels
            .iter()
            .map(|c| c.radius)
            .fold(compensation_radius, f64::max);
        // Retain only the radial function selected by each local projector.
        let ae = projectors
            .iter()
            .map(|p| channels[p.channel].ae[p.radial].clone())
            .collect();
        let ps = projectors
            .iter()
            .map(|p| channels[p.channel].ps[p.radial].clone())
            .collect();
        let mut memory = memory;
        // Magnetic copies share their radial data conceptually; until packed
        // radials are introduced every retained copy is explicitly charged.
        let required = product(&[p, n, 8, 2])?;
        if required > memory.bytes() {
            memory = ctx.reserve(required)?;
        }
        Ok(Self {
            center,
            radius,
            compensation_radius,
            projectors,
            levels,
            radial,
            ae,
            ps,
            _memory: memory,
        })
    }
    /// Compensation charge Fourier transforms in (level,vector) C order.
    pub fn compensation(&self, ctx: &Context, vectors: &[[f64; 3]]) -> Result<Buffer<C64>> {
        self.transforms(ctx, vectors, false)
    }
    /// Complete all-electron minus smooth projector-pair Fourier transform.
    /// Compensation plus its remainder equals this quantity without a low-q
    /// or multipole approximation.
    pub fn difference(&self, ctx: &Context, vectors: &[[f64; 3]]) -> Result<Buffer<C64>> {
        self.transforms(ctx, vectors, true)
    }
    fn transforms(
        &self,
        ctx: &Context,
        vectors: &[[f64; 3]],
        difference: bool,
    ) -> Result<Buffer<C64>> {
        if vectors.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("PAW form-factor vectors must be finite"));
        }
        let p = self.projectors.len();
        let rows = if difference {
            product(&[p, p])?
        } else {
            self.levels.len()
        };
        let mut out = ctx.allocate::<C64>(product(&[rows, vectors.len()])?)?;
        let r = self.radial.as_slice();
        let _work = ctx.reserve(product(&[r.len(), 8, 3])?)?;
        for (g, vector) in vectors.iter().enumerate() {
            let q = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
            let phase =
                C64::from_polar(1., -(0..3).map(|a| vector[a] * self.center[a]).sum::<f64>());
            for (index, level) in self.levels.iter().enumerate() {
                let angular = 4.
                    * PI
                    * C64::new(0., -1.).powu(level.l as u32)
                    * angular::complex(level.l, level.m, *vector)
                    * phase;
                if angular.norm_sqr() < 1e-30 {
                    continue;
                }
                if difference {
                    for a in 0..p {
                        for b in 0..p {
                            let coupling = level.angular.as_slice()[a * p + b];
                            if coupling == 0. {
                                continue;
                            }
                            let density = (0..r.len())
                                .map(|k| {
                                    (self.ae[a][k] * self.ae[b][k] - self.ps[a][k] * self.ps[b][k])
                                        * r[k]
                                        * r[k]
                                        * mandacaru_kernels::two_center::spherical_bessel(
                                            level.l as i32,
                                            q * r[k],
                                        )
                                })
                                .collect::<Vec<_>>();
                            out.as_mut_slice()[(a * p + b) * vectors.len() + g] +=
                                angular * coupling * simpson(r, &density)?;
                        }
                    }
                } else {
                    let radial: Vec<_> = (0..801)
                        .map(|i| self.compensation_radius * i as f64 / 800.)
                        .collect();
                    let values: Vec<_> = radial
                        .iter()
                        .map(|r| {
                            r * r
                                * shape(*r, self.compensation_radius, level.l)
                                * mandacaru_kernels::two_center::spherical_bessel(
                                    level.l as i32,
                                    q * r,
                                )
                        })
                        .collect();
                    out.as_mut_slice()[index * vectors.len() + g] =
                        angular * simpson(&radial, &values)?;
                }
            }
        }
        Ok(out)
    }
}
