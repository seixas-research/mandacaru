//! Wannier downfolding: the many-body problem of a fragment of the localized functions.
//!
//! The fragment's one-body block and density matrix come from the Kohn-Sham Hamiltonian and
//! density in the Wannier gauge; its interaction is the Coulomb interaction of the functions'
//! pair densities on the Born-von Karman supercell of the k mesh, truncated at half its shortest
//! lattice vector (Spencer and Alavi 2008), with the PAW compensation charges and one-center
//! remainders of every pair ([`pairs`]). The model's one-body part is `h - V^dc`, the Kohn-Sham
//! block less the Hartree and exchange potential of the fragment's own electrons
//! ([`problem`]). Atomic units; the result is in Hartree.
//!
//! With `screening` the interaction is the static constrained or full RPA screened one
//! ([`screened`]). With `coulomb="isolated"` the bare interaction is that of the fragment alone in a
//! doubled supercell ([`isolated`]); the screening is the supercell's either way. Not available,
//! and refused by name: a spinor crystal.
pub mod coulomb;
pub mod functions;
mod isolated;
pub mod pairs;
pub mod problem;
mod screened;

use self::{
    coulomb::Cell,
    functions::supercell_functions,
    isolated::padded as padded_fragment,
    pairs::{coulomb as bare_integrals, Mode, Pairs},
    problem::{assemble, Matrix, Problem},
    screened::{correction, Rest},
};
use super::setup::Wannier;
use crate::{
    periodic::{invalid, response, screening::transitions::Remainders},
    quantum::C64,
    Error, Result,
};
use nalgebra::Vector3;

/// How the interaction is screened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Screening {
    /// Constrained RPA: every transition but those inside the Bloch subspace of the fragment.
    Crpa,
    /// Full RPA: every transition screens.
    Rpa,
}
impl Screening {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "crpa" => Ok(Self::Crpa),
            "rpa" => Ok(Self::Rpa),
            other => Err(invalid(format!(
                "screening must be None, 'crpa' or 'rpa'; got {other:?}"
            ))),
        }
    }
}

/// The box the interaction is evaluated in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoulombBox {
    /// The Born-von Karman supercell, truncated at half its shortest lattice vector.
    Supercell,
    /// The fragment alone in a supercell twice as large.
    Isolated,
}
impl CoulombBox {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "supercell" => Ok(Self::Supercell),
            "isolated" => Ok(Self::Isolated),
            other => Err(invalid(format!(
                "coulomb must be 'supercell' or 'isolated'; got {other:?}"
            ))),
        }
    }
}

/// What a downfolding asks for.
pub struct Request {
    /// The fragment's modes `(function, R)`; `None`: every function of the home cell.
    pub functions: Option<Vec<(usize, [i64; 3])>>,
    /// `(up, down)` electrons; `None`: the Kohn-Sham count.
    pub num_particles: Option<[usize; 2]>,
    pub screening: Option<Screening>,
    /// The vectors the screened interaction keeps, `|q + G| <= cutoff` (Bohr^-1), and the number
    /// of lowest Kohn-Sham states that screen; `None`: the defaults. Only with a screening.
    pub cutoff: Option<f64>,
    pub bands: Option<usize>,
    pub coulomb: CoulombBox,
    /// Workers of the calculator's pool the screening's transition loop uses; `None`: all. The
    /// numbers do not depend on it.
    pub workers: Option<usize>,
}

/// How many rms radii past its outermost centers a fragment's charge is taken to reach.
pub const FRAGMENT_REACH: f64 = 1.5;

pub struct Downfolded {
    pub problem: Problem,
    /// Notes the caller should show (a fragment reaching past the interaction's truncation).
    pub warnings: Vec<String>,
    /// Truncation radius of the interaction (Bohr).
    pub radius: f64,
    /// Per spin channel, `<w_p|S|w_q>` of the fragment from the grid overlap and the overlap
    /// correction, and the zeroth Fourier component of its pair densities: both the identity (in
    /// the isolated box, up to the tail of a function beyond its Wigner-Seitz cell).
    pub norms: Vec<(Matrix, Matrix)>,
}

/// The static polarizability at one mesh point.
pub struct Response {
    /// `chi_{GG'}(q)` (Bohr^-3 Hartree^-1) on the vectors `vectors = q + G`.
    pub chi: Matrix,
    pub vectors: Vec<[f64; 3]>,
    /// Integer cell coordinates of `G`.
    pub g: Vec<[i64; 3]>,
}

impl Wannier {
    /// The static Kohn-Sham polarizability of the mesh point `q` (integer mesh coordinates): full
    /// RPA, or with `Screening::Crpa` without the transitions inside the Bloch subspace of the
    /// functions `targets`. An audit seam over the screening's building block; the screened
    /// interaction is [`Wannier::downfold`]'s.
    pub fn polarizability(
        &self,
        q: [i64; 3],
        screening: Screening,
        targets: &[usize],
        cutoff: Option<f64>,
        bands: Option<usize>,
        workers: Option<usize>,
    ) -> Result<Response> {
        let crystal = &*self.crystal;
        let setup = response::setup::Setup::new(crystal)?;
        let ctx = &crystal.context;
        let remainders = Remainders::new(ctx, &setup)?;
        let cell = Cell::of(crystal, self.size)?;
        let mut rest = Rest::new(
            self,
            &setup,
            &remainders,
            &cell,
            targets,
            screening,
            cutoff,
            bands,
        )?;
        rest.medium.workers = workers;
        let (chi, vectors, g) = rest.polarizability(q)?;
        Ok(Response { chi, vectors, g })
    }

    /// `rho_ab(Q = 0)` of every pair of the first `bands` Kohn-Sham states of mesh point `k` in
    /// spin channel `spin`, with the PAW augmentation: `<a|S|b>`, the identity for orthonormal
    /// states. An audit seam of the transition densities.
    pub fn transition_norms(&self, spin: usize, k: usize, bands: Option<usize>) -> Result<Matrix> {
        let crystal = &*self.crystal;
        if spin >= self.channels.len() || k >= self.channels[spin].frames.len() {
            return Err(invalid("a spin channel or mesh point that does not exist"));
        }
        let setup = response::setup::Setup::new(crystal)?;
        let ctx = &crystal.context;
        let remainders = Remainders::new(ctx, &setup)?;
        let cell = Cell::of(crystal, self.size)?;
        let rest = Rest::new(
            self,
            &setup,
            &remainders,
            &cell,
            &[],
            Screening::Rpa,
            None,
            bands,
        )?;
        rest.norms(spin, k)
    }
}

impl Wannier {
    pub fn downfold(&self, request: &Request) -> Result<Downfolded> {
        if request.screening.is_none() && (request.cutoff.is_some() || request.bands.is_some()) {
            return Err(invalid(
                "screening_cutoff and screening_bands need a screening ('crpa' or 'rpa')",
            ));
        }
        if request.cutoff.is_some_and(|c| !(c.is_finite() && c > 0.)) || request.bands == Some(0) {
            return Err(invalid(
                "screening_cutoff must be positive and screening_bands at least 1",
            ));
        }
        let crystal = &*self.crystal;
        if crystal.spinor {
            return Err(Error::Unsupported(
                "the downfolded problem of spinor Wannier functions is not built: its two-body \
                 terms and spin-orbital layout are those of a spin-orbit Hamiltonian"
                    .into(),
            ));
        }
        let description = response::description(crystal)?;
        if description.projector_basis != "raw" {
            return Err(Error::Unsupported(format!(
                "Wannier downfolding reads the raw projector basis, the crystal was built with {:?}",
                description.projector_basis
            )));
        }
        let setup = response::setup::Setup::new(crystal)?;
        let count = self.trials.len();
        let modes = request
            .functions
            .clone()
            .unwrap_or_else(|| (0..count).map(|n| (n, [0; 3])).collect());
        if modes.is_empty() {
            return Err(invalid("a fragment names at least one function"));
        }
        if modes.iter().any(|(n, _)| *n >= count) {
            return Err(invalid(format!(
                "the fragment names a function outside the {count} that were localized"
            )));
        }
        for (i, a) in modes.iter().enumerate() {
            if modes[..i].contains(a) {
                return Err(invalid("a fragment lists each function once"));
            }
        }
        let mut ids: Vec<usize> = modes.iter().map(|(n, _)| *n).collect();
        ids.sort_unstable();
        ids.dedup();
        let located: Vec<Mode> = modes
            .iter()
            .map(|(n, r)| Mode {
                at: ids.iter().position(|i| i == n).unwrap(),
                cell: *r,
            })
            .collect();
        let cell = Cell::of(crystal, self.size)?;
        let ctx = &crystal.context;
        let spins = self.channels.len();
        let isolated = request.coulomb == CoulombBox::Isolated;
        let center = fragment_center(self, &modes);
        // The pair densities on the supercell (the screening and the sum rules read these) and,
        // for the isolated box, the fragment alone in the doubled box (the bare integrals).
        let mut pairs = Vec::with_capacity(spins);
        let mut alone: Vec<Pairs> = vec![];
        let mut doubled: Option<Cell> = None;
        for channel in &self.channels {
            let functions = supercell_functions(crystal, channel, &cell, &ids)?;
            pairs.push(Pairs::new(crystal, &setup, &cell, &functions, &located)?);
            if isolated {
                let (big, padded, identity) =
                    padded_fragment(crystal, &setup, &cell, &functions, &located, center)?;
                alone.push(Pairs::new(crystal, &setup, &big, &padded, &identity)?);
                doubled = Some(big);
            }
        }
        let (box_cell, box_pairs): (&Cell, &Vec<Pairs>) = match &doubled {
            Some(big) => (big, &alone),
            None => (&cell, &pairs),
        };
        // The kernel of the bare integrals is truncated at half the box's shortest vector; the
        // screened correction is always the supercell's (its own radius for the head).
        let radius = box_cell.half_shortest;
        let norms = box_pairs
            .iter()
            .map(|p| p.norms(&setup, box_cell))
            .collect::<Result<Vec<_>>>()?;
        let screened = match request.screening {
            Some(_) => {
                let remainders = Remainders::new(ctx, &setup)?;
                let mut rest = Rest::new(
                    self,
                    &setup,
                    &remainders,
                    &cell,
                    &ids,
                    request.screening.unwrap(),
                    request.cutoff,
                    request.bands,
                )?;
                rest.medium.workers = request.workers;
                let screened = rest.interaction(cell.half_shortest)?;
                drop(rest);
                Some((remainders, screened))
            }
            None => None,
        };
        let mut chemist = Vec::with_capacity(spins);
        for s in 0..spins {
            let mut row = Vec::with_capacity(spins);
            for t in 0..spins {
                let mut integrals = bare_integrals(
                    ctx,
                    &setup,
                    box_cell,
                    radius,
                    &box_pairs[s],
                    &box_pairs[t],
                    false,
                )?;
                if let Some((remainders, screened)) = &screened {
                    let extra =
                        correction(&setup, remainders, &cell, &pairs[s], &pairs[t], screened)?;
                    for (v, e) in integrals.values.iter_mut().zip(&extra.values) {
                        *v += e;
                    }
                }
                row.push(integrals);
            }
            chemist.push(row);
        }
        let n = modes.len();
        let size = self.size;
        let block = |table: &[C64], a: &(usize, [i64; 3]), b: &(usize, [i64; 3])| {
            let t: [usize; 3] =
                std::array::from_fn(|i| (b.1[i] - a.1[i]).rem_euclid(size[i] as i64) as usize);
            let r = (t[0] * size[1] + t[1]) * size[2] + t[2];
            table[(r * count + a.0) * count + b.0]
        };
        let mut kohn_sham = vec![];
        let mut gamma = vec![];
        for channel in &self.channels {
            let h = Matrix::from_fn(n, n, |i, j| {
                block(channel.hamiltonian_bvk.as_slice(), &modes[i], &modes[j])
            });
            // gamma_pq = <c_p^+ c_q> = <w_q|D|w_p>
            let rho = Matrix::from_fn(n, n, |i, j| {
                block(channel.density_bvk.as_slice(), &modes[i], &modes[j])
            });
            kohn_sham.push((&h + h.adjoint()) * C64::new(0.5, 0.));
            gamma.push(rho.transpose());
        }
        let problem = assemble(
            modes.clone(),
            &chemist,
            kohn_sham,
            gamma,
            request.num_particles,
            screened.as_ref().map(|(_, s)| s.head),
        )?;
        let mut warnings = vec![];
        for channel in &self.channels {
            let extent = fragment_extent(self, channel, &modes);
            if extent > radius {
                warnings.push(format!(
                    "the fragment spans about {:.2} A ({FRAGMENT_REACH} rms radii past its outer \
                     centers), beyond the interaction's truncation radius {:.2} A: its functions \
                     meet their own periodic copies; use a denser k mesh or a smaller fragment",
                    extent * crate::units::BOHR_TO_ANGSTROM,
                    radius * crate::units::BOHR_TO_ANGSTROM,
                ));
                break;
            }
        }
        Ok(Downfolded {
            problem,
            warnings,
            radius,
            norms,
        })
    }
}

impl Wannier {
    /// The Coulomb integrals `(pq|rs)` (chemists', Hartree, `[((p*n+q)*n+r)*n+s]`) of the smooth
    /// pseudo pair densities of the fragment `modes` in spin channel `spin` alone: the grid part
    /// of the bare interaction, without the compensation charges, the self-energy tails and the
    /// one-center remainders. An audit seam: the full integrals are [`Wannier::downfold`]'s.
    pub fn grid_integrals(&self, spin: usize, modes: &[(usize, [i64; 3])]) -> Result<Vec<C64>> {
        let crystal = &*self.crystal;
        let channel = self
            .channels
            .get(spin)
            .ok_or_else(|| invalid("a spin channel that does not exist"))?;
        let setup = response::setup::Setup::new(crystal)?;
        let mut ids: Vec<usize> = modes.iter().map(|(n, _)| *n).collect();
        ids.sort_unstable();
        ids.dedup();
        let located: Vec<Mode> = modes
            .iter()
            .map(|(n, r)| Mode {
                at: ids.iter().position(|i| i == n).unwrap(),
                cell: *r,
            })
            .collect();
        let cell = Cell::of(crystal, self.size)?;
        let functions = supercell_functions(crystal, channel, &cell, &ids)?;
        let pairs = Pairs::new(crystal, &setup, &cell, &functions, &located)?;
        Ok(bare_integrals(
            &crystal.context,
            &setup,
            &cell,
            cell.half_shortest,
            &pairs,
            &pairs,
            true,
        )?
        .values)
    }
}

/// The mean of the fragment's function centers (Bohr), the first spin channel's.
fn fragment_center(wannier: &Wannier, modes: &[(usize, [i64; 3])]) -> [f64; 3] {
    let lattice = super::sampling::lattice(&wannier.crystal.grid);
    let centers = wannier.channels[0].localized.spread.centers.as_slice();
    let mut sum = Vector3::zeros();
    for (n, r) in modes {
        sum += Vector3::from(centers[*n])
            + lattice * Vector3::new(r[0] as f64, r[1] as f64, r[2] as f64);
    }
    (sum / modes.len() as f64).into()
}

/// The fragment's diameter (Bohr): the largest distance between two of its functions' centers
/// plus [`FRAGMENT_REACH`] root-mean-square radii of each.
fn fragment_extent(
    wannier: &Wannier,
    channel: &super::model::Channel,
    modes: &[(usize, [i64; 3])],
) -> f64 {
    let lattice = super::sampling::lattice(&wannier.crystal.grid);
    let spread = &channel.localized.spread;
    let point = |(n, r): &(usize, [i64; 3])| {
        Vector3::from(spread.centers.as_slice()[*n])
            + lattice * Vector3::new(r[0] as f64, r[1] as f64, r[2] as f64)
    };
    let radius = |n: usize| {
        let c = spread.centers.as_slice()[n];
        (spread.second_moments.as_slice()[n] - c.iter().map(|v| v * v).sum::<f64>())
            .max(0.)
            .sqrt()
    };
    let mut extent = 0f64;
    for a in modes {
        for b in modes {
            extent = extent
                .max((point(a) - point(b)).norm() + FRAGMENT_REACH * (radius(a.0) + radius(b.0)));
        }
    }
    extent
}
