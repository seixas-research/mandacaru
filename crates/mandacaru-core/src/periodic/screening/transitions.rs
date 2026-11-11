//! Transition densities `rho_ab(q + G) = int_cell psi_{ak}^*(r) psi_{b,k+q}(r) e^{-i(q+G).r} dr` of
//! pseudo Bloch states, with the PAW augmentation of the pair:
//!
//! ```text
//! rho_ab(Q) = smooth(Q) + sum_c Q^c_ab g_c(Q) + sum_A sum_{ij in A} <psi_a|p_i> <p_j|psi_b> E^A_ij(Q),
//! Q^c_ab = sum_ij <psi_a|p_i> B_c[i,j] <p_j|psi_b>
//! ```
//!
//! the smooth product on the cell grid, the compensation charge of every multipole channel `c`
//! (shape `g_c`, moment blocks `B_c`) and the remainder the compensation leaves of the
//! augmentation density of each atom's projector pairs (`E^A`, [`Remainder`]). Atomic units.
use crate::{
    math::{fft::Fft3, SpatialGrid},
    periodic::{
        invalid,
        response::setup::Setup,
        setup::remainder::{projector_columns, Remainder},
        Crystal,
    },
    product,
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// The remainder tables of every atom with projectors, built once per setup.
pub struct Remainders {
    atoms: Vec<(Vec<usize>, Remainder)>,
}

impl Remainders {
    pub(in crate::periodic) fn new(ctx: &Context, setup: &Setup) -> Result<Self> {
        let owners = setup.projector_atoms();
        let labels = setup.projector_labels();
        let sites = setup.sites();
        let mut atoms = vec![];
        for atom in 0..sites.atoms.len() {
            let columns = projector_columns(&owners, atom);
            if columns.is_empty() {
                continue;
            }
            let projectors: Vec<crate::paw::onecenter::Projector> = columns
                .iter()
                .map(|&j| crate::paw::onecenter::Projector {
                    l: labels[j].0,
                    m: labels[j].1,
                    index: labels[j].2,
                    real: false,
                })
                .collect();
            let remainder = Remainder::new(
                ctx,
                setup.dataset(atom),
                setup.shapes(),
                crate::paw::onecenter::Basis::Raw,
                &projectors,
                sites.atoms[atom].center,
            )?;
            atoms.push((columns, remainder));
        }
        Ok(Self { atoms })
    }
    /// For each atom with projectors: its columns and `E_ab(Q)` at `vectors`,
    /// `[(a * m + b) * vectors.len() + v]`.
    pub fn tables(&self, vectors: &[[f64; 3]]) -> Result<Vec<(&[usize], Vec<C64>)>> {
        self.atoms
            .iter()
            .map(|(columns, remainder)| {
                let m = columns.len();
                let mut table = vec![C64::default(); m * m * vectors.len()];
                remainder.transforms(vectors, &mut table)?;
                Ok((columns.as_slice(), table))
            })
            .collect()
    }
}

/// The augmentation of the pair densities at one set of wave vectors.
pub struct Augmentation<'a> {
    blocks: &'a [C64],
    p: usize,
    count: usize,
    /// The projector columns each multipole channel's atom owns.
    own: Vec<Vec<usize>>,
    /// `shape[c * count + v]`: the compensation shape of channel `c` at `Q_v`.
    shape: Vec<C64>,
    /// Per atom: its projector columns and `E_ab(Q_v)`.
    remainders: Vec<(&'a [usize], Vec<C64>)>,
}

impl<'a> Augmentation<'a> {
    pub(in crate::periodic) fn new(
        crystal: &'a Crystal,
        setup: &'a Setup,
        remainders: &'a Remainders,
        vectors: &[[f64; 3]],
    ) -> Result<Self> {
        let channels = crystal.channels;
        let mut shape = vec![C64::default(); channels * vectors.len()];
        setup
            .shapes()
            .transforms(&setup.sites(), vectors, &mut shape, vectors.len())?;
        let owners = setup.projector_atoms();
        let own = setup
            .sites()
            .channels
            .iter()
            .map(|&(atom, _, _)| projector_columns(&owners, atom))
            .collect();
        Ok(Self {
            blocks: crystal.blocks.as_slice(),
            p: crystal.p,
            count: vectors.len(),
            own,
            shape,
            remainders: remainders.tables(vectors)?,
        })
    }
    /// Adds the augmentation of the pair `(a, b)` with projector overlaps `pa = <psi_a|p>` and
    /// `pb = <psi_b|p>` to `out` (the smooth part is already there).
    pub fn add(&self, pa: &[C64], pb: &[C64], out: &mut [C64]) {
        let (p, count) = (self.p, self.count);
        for (c, own) in self.own.iter().enumerate() {
            let mut charge = C64::default();
            for &i in own {
                for &j in own {
                    charge += pa[i] * self.blocks[(c * p + i) * p + j] * pb[j].conj();
                }
            }
            if charge.norm_sqr() == 0. {
                continue;
            }
            let row = &self.shape[c * count..(c + 1) * count];
            for (o, s) in out.iter_mut().zip(row) {
                *o += charge * s;
            }
        }
        for (columns, table) in &self.remainders {
            let m = columns.len();
            for (a, &i) in columns.iter().enumerate() {
                for (b, &j) in columns.iter().enumerate() {
                    let pair = pa[i] * pb[j].conj();
                    if pair.norm_sqr() == 0. {
                        continue;
                    }
                    let row = &table[(a * m + b) * count..(a * m + b + 1) * count];
                    for (o, e) in out.iter_mut().zip(row) {
                        *o += pair * e;
                    }
                }
            }
        }
    }
}

/// What one worker needs to transform a product: its own FFT plan and work array.
pub struct Scratch {
    fft: Fft3,
    work: Buffer<C64>,
}

/// The smooth transition densities at the wave vectors `q + G` of a cell grid. Shared read-only
/// by the workers, each of which owns a [`Scratch`].
pub struct Engine {
    shape: [usize; 3],
    points: usize,
    q_phase: Vec<C64>,
    index: Vec<usize>,
    to_g: Vec<C64>,
    pub vectors: Vec<[f64; 3]>,
}

impl Engine {
    /// `g`: integer reciprocal-lattice coordinates of the cell; `q`: Cartesian.
    pub fn new(ctx: &Context, grid: &SpatialGrid, q: [f64; 3], g: &[[i64; 3]]) -> Result<Self> {
        let shape = grid.index.shape();
        let points = grid.index.points();
        let cell: Matrix3<f64> =
            grid.step * Matrix3::from_diagonal(&Vector3::from(shape.map(|n| n as f64)));
        let reciprocal = 2.
            * PI
            * cell
                .try_inverse()
                .ok_or_else(|| invalid("singular cell"))?
                .transpose();
        let mut index = Vec::with_capacity(g.len());
        let mut to_g = Vec::with_capacity(g.len());
        let mut vectors = Vec::with_capacity(g.len());
        for v in g {
            if (0..3).any(|a| 2 * v[a].unsigned_abs() as usize >= shape[a]) {
                return Err(invalid(
                    "a screened reciprocal vector reaches the grid's Nyquist wave number: lower \
                     the screening cutoff or refine the grid",
                ));
            }
            let wrapped: [usize; 3] =
                std::array::from_fn(|a| v[a].rem_euclid(shape[a] as i64) as usize);
            index.push((wrapped[0] * shape[1] + wrapped[1]) * shape[2] + wrapped[2]);
            let k = reciprocal * Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64);
            to_g.push(C64::from_polar(
                grid.index.dv(),
                -(k[0] * grid.origin[0] + k[1] * grid.origin[1] + k[2] * grid.origin[2]),
            ));
            vectors.push([q[0] + k[0], q[1] + k[1], q[2] + k[2]]);
        }
        let q_phase = (0..points)
            .map(|x| {
                let r = grid.point(x);
                C64::from_polar(1., -(q[0] * r[0] + q[1] * r[1] + q[2] * r[2]))
            })
            .collect();
        let _ = ctx;
        Ok(Self {
            shape,
            points,
            q_phase,
            index,
            to_g,
            vectors,
        })
    }
    /// One worker's scratch.
    pub fn scratch(&self, ctx: &Context) -> Result<Scratch> {
        Ok(Scratch {
            fft: Fft3::new(ctx, self.shape)?,
            work: ctx.allocate(product(&[self.points])?)?,
        })
    }
    /// `out[g] = dv e^{-i r0.G} DFT[psi_a^* psi_b e^{-i q.r}](G)` for the `G` of the engine.
    pub fn smooth(
        &self,
        scratch: &mut Scratch,
        a: &[C64],
        b: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        if a.len() != self.points || b.len() != self.points || out.len() != self.index.len() {
            return Err(invalid("a transition density has the wrong size"));
        }
        let work = scratch.work.as_mut_slice();
        for (((w, x), y), phase) in work.iter_mut().zip(a).zip(b).zip(&self.q_phase) {
            *w = x.conj() * y * phase;
        }
        scratch.fft.transform(work, false)?;
        for ((o, &i), s) in out.iter_mut().zip(&self.index).zip(&self.to_g) {
            *o = work[i] * s;
        }
        Ok(())
    }
}
