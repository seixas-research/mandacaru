//! Native PAW dataset persistence and potential evaluation, independent of Python.
use super::{channel::Channel, invalid, smooth::SmoothWaves};
use crate::{product, Buffer, Context, Result};
use nalgebra::DMatrix;
use serde_json::{json, Value};
use std::path::Path;

pub struct DatasetChannel {
    pub l: usize,
    pub n: Vec<usize>,
    pub energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub radius: f64,
    pub ae: Buffer<f64>,
    pub smooth: SmoothWaves,
    pub channel: Channel,
    pub ionic: nalgebra::DMatrix<f64>,
}
pub struct Dataset {
    pub context: Context,
    pub atomic_number: u8,
    pub functional: String,
    pub r: Buffer<f64>,
    pub reference_energy: f64,
    pub valence_charge: f64,
    pub local_radius: f64,
    pub compensation_radius: f64,
    pub compensation_charge: f64,
    pub hartree_screening: f64,
    pub one_center_energy: f64,
    pub ionic: Buffer<f64>,
    pub screened: Buffer<f64>,
    pub valence_density: Buffer<f64>,
    pub core_density: Buffer<f64>,
    pub smooth_core: Buffer<f64>,
    pub channels: Vec<DatasetChannel>,
}

fn matrix_values(matrix: &DMatrix<f64>) -> Vec<f64> {
    (0..matrix.nrows())
        .flat_map(|i| (0..matrix.ncols()).map(move |j| matrix[(i, j)]))
        .collect()
}
impl Dataset {
    pub fn document(&self) -> Result<Value> {
        let channels=self.channels.iter().map(|ch|json!({
            "l":ch.l,"shells":ch.n,"energies":ch.energies,"occupations":ch.occupations,"radius":ch.radius,
            "ae":ch.ae.as_slice(),"pseudo":ch.channel.pseudo.as_slice(),"projectors":ch.channel.projectors.as_slice(),
            "overlap":matrix_values(&ch.channel.overlap),"screened_coupling":matrix_values(&ch.channel.screened),
            "ionic_coupling":matrix_values(&ch.ionic),"kinetic":matrix_values(&ch.channel.kinetic),"potential":matrix_values(&ch.channel.potential),
            "wavevectors":ch.smooth.wavevectors,"coefficients":ch.smooth.coefficients.as_slice(),"norms":matrix_values(&ch.smooth.norms),
            "residuals":ch.smooth.residuals,"duality_error":ch.channel.duality_error,"asymmetry":ch.channel.asymmetry,
        })).collect::<Vec<_>>();
        Ok(
            json!({"format":"mandacaru-paw","version":1,"length_unit":"bohr","energy_unit":"Ha",
            "atomic_number":self.atomic_number,"functional":self.functional,"relativity":"none",
            "reference_energy":self.reference_energy,"valence_charge":self.valence_charge,
            "local_radius":self.local_radius,"compensation_radius":self.compensation_radius,"compensation_charge":self.compensation_charge,
            "hartree_screening":self.hartree_screening,"one_center_energy":self.one_center_energy,
            "r":self.r.as_slice(),"ionic":self.ionic.as_slice(),"screened":self.screened.as_slice(),
            "valence_density":self.valence_density.as_slice(),"core_density":self.core_density.as_slice(),"smooth_core":self.smooth_core.as_slice(),
            "channels":channels}),
        )
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let count = product(&[
            self.r.as_slice().len(),
            self.channels
                .len()
                .checked_mul(5)
                .and_then(|x| x.checked_add(10))
                .ok_or(crate::Error::Overflow)?,
            64,
        ])?;
        let _memory = self.context.reserve(count)?;
        let document = self.document()?;
        crate::persistence::atomic_write(path, |file| {
            serde_json::to_writer(file, &document).map_err(|e| invalid(e.to_string()))
        })
    }
    pub fn local_values(&self, radii: &[f64]) -> Result<(Buffer<f64>, Buffer<f64>)> {
        if radii.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid(
                "PAW coordinates must be finite nonnegative Bohr radii",
            ));
        }
        let mut values = self.context.allocate::<f64>(radii.len())?;
        let mut slopes = self.context.allocate::<f64>(radii.len())?;
        let mesh = self.r.as_slice();
        let data = self.ionic.as_slice();
        for (i, &r) in radii.iter().enumerate() {
            if r >= mesh[mesh.len() - 1] {
                values.as_mut_slice()[i] = -self.valence_charge / r;
                slopes.as_mut_slice()[i] = self.valence_charge / (r * r);
                continue;
            }
            let first = mesh
                .partition_point(|x| *x < r)
                .saturating_sub(2)
                .min(mesh.len() - 4);
            let mut value = 0.;
            let mut slope = 0.;
            for a in first..first + 4 {
                let denominator = (first..first + 4)
                    .filter(|&b| b != a)
                    .map(|b| mesh[a] - mesh[b])
                    .product::<f64>();
                value += data[a] / denominator
                    * (first..first + 4)
                        .filter(|&b| b != a)
                        .map(|b| r - mesh[b])
                        .product::<f64>();
                for b in first..first + 4 {
                    if b != a {
                        slope += data[a] / denominator
                            * (first..first + 4)
                                .filter(|&c| c != a && c != b)
                                .map(|c| r - mesh[c])
                                .product::<f64>();
                    }
                }
            }
            values.as_mut_slice()[i] = value;
            slopes.as_mut_slice()[i] = if r == 0. { 0. } else { slope };
        }
        Ok((values, slopes))
    }
    pub fn projector_values(&self, l: usize, radii: &[f64]) -> Result<Buffer<f64>> {
        let channel = self
            .channels
            .iter()
            .find(|ch| ch.l == l)
            .ok_or_else(|| invalid("PAW angular channel is absent"))?;
        if radii.iter().any(|r| !r.is_finite() || *r < 0.) {
            return Err(invalid("projector radii must be finite and nonnegative"));
        }
        let mut out = self
            .context
            .allocate::<f64>(product(&[channel.energies.len(), radii.len()])?)?;
        for i in 0..channel.energies.len() {
            for (k, &r) in radii.iter().enumerate() {
                out.as_mut_slice()[i * radii.len() + k] =
                    if r > channel.radius.max(self.local_radius) {
                        0.
                    } else {
                        crate::math::radial::interpolate(
                            self.r.as_slice(),
                            &channel.channel.projectors.as_slice()
                                [i * self.r.as_slice().len()..(i + 1) * self.r.as_slice().len()],
                            r,
                        )
                    };
            }
        }
        Ok(out)
    }
    pub fn load(ctx: &Context, path: &Path) -> Result<Self> {
        let bytes = std::fs::metadata(path)
            .map_err(|e| invalid(format!("PAW input: {e}")))?
            .len();
        let size = usize::try_from(bytes).map_err(|_| crate::Error::Overflow)?;
        let _memory = ctx.reserve(product(&[size, 32])?)?;
        let source = std::fs::read(path).map_err(|e| invalid(format!("PAW input: {e}")))?;
        let value: Value =
            serde_json::from_slice(&source).map_err(|e| invalid(format!("PAW input: {e}")))?;
        Self::from_document(ctx, &value)
    }
    pub fn from_document(ctx: &Context, v: &Value) -> Result<Self> {
        if v["format"] != "mandacaru-paw"
            || v["version"] != 1
            || v["energy_unit"] != "Ha"
            || v["length_unit"] != "bohr"
            || v["relativity"] != "none"
        {
            return Err(invalid(
                "unsupported PAW schema, units or relativistic model",
            ));
        }
        let number = v["atomic_number"]
            .as_u64()
            .filter(|z| *z > 0 && *z <= 118)
            .ok_or_else(|| invalid("invalid PAW atomic number"))? as u8;
        let functional = v["functional"]
            .as_str()
            .filter(|s| ["lda", "pbe"].contains(s))
            .ok_or_else(|| invalid("unsupported PAW functional"))?
            .to_owned();
        let real = |value: &Value, name: &str| {
            value[name]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid(format!("invalid PAW {name}")))
        };
        let array = |value: &Value, name: &str, expected: Option<usize>| -> Result<Buffer<f64>> {
            let values = value[name]
                .as_array()
                .ok_or_else(|| invalid(format!("missing PAW array {name}")))?;
            if expected.is_some_and(|n| n != values.len()) {
                return Err(invalid(format!("incorrect PAW {name} dimensions")));
            }
            let mut out = ctx.allocate::<f64>(values.len())?;
            for (i, val) in values.iter().enumerate() {
                out.as_mut_slice()[i] = val
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| invalid(format!("invalid value in PAW {name}")))?;
            }
            Ok(out)
        };
        let r = array(v, "r", None)?;
        let n = r.as_slice().len();
        let ionic = array(v, "ionic", Some(n))?;
        crate::math::radial::validate(r.as_slice(), ionic.as_slice())?;
        let screened = array(v, "screened", Some(n))?;
        let valence_density = array(v, "valence_density", Some(n))?;
        let core_density = array(v, "core_density", Some(n))?;
        let smooth_core = array(v, "smooth_core", Some(n))?;
        if valence_density
            .as_slice()
            .iter()
            .chain(core_density.as_slice())
            .chain(smooth_core.as_slice())
            .any(|x| *x < 0.)
        {
            return Err(invalid("negative PAW reference density"));
        }
        let valence_charge = real(v, "valence_charge")?;
        let local_radius = real(v, "local_radius")?;
        let compensation_radius = real(v, "compensation_radius")?;
        if valence_charge <= 0.
            || valence_charge > number as f64
            || local_radius <= 0.
            || compensation_radius <= 0.
            || local_radius >= r.as_slice()[n - 1]
            || compensation_radius >= r.as_slice()[n - 1]
        {
            return Err(invalid("invalid PAW charge or radius"));
        }
        let raw = v["channels"]
            .as_array()
            .filter(|v| !v.is_empty() && v.len() <= 9)
            .ok_or_else(|| invalid("PAW needs angular channels"))?;
        let mut channels = vec![];
        for ch in raw {
            let l = ch["l"]
                .as_u64()
                .filter(|l| *l <= 8)
                .ok_or_else(|| invalid("invalid PAW angular momentum"))?
                as usize;
            if channels.iter().any(|c: &DatasetChannel| c.l == l) {
                return Err(invalid("duplicate PAW angular channel"));
            }
            let energies = array(ch, "energies", None)?;
            let count = energies.as_slice().len();
            if count == 0 || count > 4 {
                return Err(invalid("PAW channel needs one to four references"));
            }
            let occupations = array(ch, "occupations", Some(count))?;
            if occupations
                .as_slice()
                .iter()
                .any(|x| *x < 0. || *x > 2. * (2 * l + 1) as f64)
            {
                return Err(invalid("invalid PAW occupations"));
            }
            let mm = product(&[count, count])?;
            let matrix = |name| -> Result<DMatrix<f64>> {
                let data = array(ch, name, Some(mm))?;
                let matrix = DMatrix::from_row_slice(count, count, data.as_slice());
                if (&matrix - matrix.transpose()).amax() > 1e-8 {
                    return Err(invalid(format!("PAW {name} is not symmetric")));
                }
                Ok(matrix)
            };
            let overlap = matrix("overlap")?;
            let screened = matrix("screened_coupling")?;
            let ionic = matrix("ionic_coupling")?;
            let kinetic = matrix("kinetic")?;
            let potential = matrix("potential")?;
            let ae = array(ch, "ae", Some(product(&[count, n])?))?;
            let pseudo = array(ch, "pseudo", Some(product(&[count, n])?))?;
            let projectors = array(ch, "projectors", Some(product(&[count, n])?))?;
            let qs = array(ch, "wavevectors", None)?;
            if qs.as_slice().len() < 5
                || qs.as_slice().len() > 32
                || qs.as_slice().iter().any(|q| *q <= 0.)
            {
                return Err(invalid("invalid PAW Bessel wave vectors"));
            }
            let coefficients = array(
                ch,
                "coefficients",
                Some(product(&[count, qs.as_slice().len()])?),
            )?;
            let norms = matrix("norms")?;
            let residuals = array(ch, "residuals", Some(count))?;
            let shells = ch["shells"]
                .as_array()
                .filter(|s| s.len() <= count)
                .ok_or_else(|| invalid("invalid PAW shell labels"))?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .filter(|n| *n > l as u64 && *n < 16)
                        .map(|n| n as usize)
                        .ok_or_else(|| invalid("invalid PAW principal quantum number"))
                })
                .collect::<Result<Vec<_>>>()?;
            let radius = real(ch, "radius")?;
            let duality_error = real(ch, "duality_error")?;
            let asymmetry = real(ch, "asymmetry")?;
            if radius <= r.as_slice()[0]
                || radius >= r.as_slice()[n - 1]
                || duality_error > 1e-8
                || asymmetry > 5e-5
            {
                return Err(invalid("invalid PAW channel radius or consistency report"));
            }
            channels.push(DatasetChannel {
                l,
                n: shells,
                energies: energies.as_slice().to_vec(),
                occupations: occupations.as_slice().to_vec(),
                radius,
                ae,
                smooth: SmoothWaves {
                    wavevectors: qs.as_slice().to_vec(),
                    coefficients,
                    norms,
                    residuals: residuals.as_slice().to_vec(),
                },
                channel: Channel {
                    pseudo,
                    projectors,
                    overlap,
                    screened,
                    kinetic,
                    potential,
                    duality_error,
                    asymmetry,
                },
                ionic,
            });
        }
        if (channels.iter().flat_map(|ch| &ch.occupations).sum::<f64>() - valence_charge).abs()
            > 1e-8
        {
            return Err(invalid(
                "PAW reference occupations do not conserve the valence charge",
            ));
        }
        Ok(Self {
            context: ctx.clone(),
            atomic_number: number,
            functional,
            r,
            reference_energy: real(v, "reference_energy")?,
            valence_charge,
            local_radius,
            compensation_radius,
            compensation_charge: real(v, "compensation_charge")?,
            hartree_screening: real(v, "hartree_screening")?,
            one_center_energy: real(v, "one_center_energy")?,
            ionic,
            screened,
            valence_density,
            core_density,
            smooth_core,
            channels,
        })
    }
}
