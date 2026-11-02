//! The Rust setup as a geometry provider of the crystal force
//! ([`crate::periodic::gradient::Derivatives`]).
//!
//! Everything that depends on the atoms and not on the converged state is implemented here from
//! the finished pieces of the setup: the row-center blocks of the matrix terms
//! ([`super::pulay`]), the projections and their center derivative, and the explicit lattice
//! sums ([`super::gradients`], [`super::dense`]). Derivatives are `dE/dtau` of the named
//! quantity, one k-point at a time (the derivative samples of the last k-point are kept for
//! the four terms that read them and dropped for the next).
//!
//! Two methods are NOT here and raise `Unsupported`: [`fields`](Derivatives::fields) and
//! [`bands`](Derivatives::bands) need the SCF's own routines (`solver.rs`'s potentials of a
//! density, its energy and its diagonalization), which stay private until the `solver.rs`
//! holder exposes them; until then a caller builds the [`Frozen`](crate::periodic::gradient::
//! Frozen) state itself and contracts it with [`contract`](crate::periodic::gradient::contract).
use super::{
    basis_gradient::ShortRange,
    gradients::{
        dense_gradients, hat_overlaps, ion_constants_gradient, ion_overlaps,
        short_range_ion_compensation_gradient, DenseGradients,
    },
    ions::LocalPotential,
    pulay::{
        core_gradient, grid_cross, kinetic_cross, mesh_dot, projection, projection_gradient,
        short_range_gradients, KSamples, Matrix, Mesh,
    },
    shapes::{Shapes, Sites},
    shells::Shells,
    spheres::Projector,
};
use crate::{
    math::SpatialGrid,
    periodic::{
        gradient::{Bands, Dense, Derivatives, Layout, State, Term},
        invalid,
    },
    quantum::C64,
    Context, Error, Result,
};
use std::cell::{OnceCell, RefCell};

/// One atom's smooth core density: the radial table the placement samples.
#[derive(Clone, Copy)]
pub struct Core<'a> {
    pub radius: &'a [f64],
    pub values: &'a [f64],
}

/// The plain inputs of the geometry (borrowed; nothing here is a dataset).
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub grid: &'a SpatialGrid,
    pub shells: &'a Shells,
    pub projectors: &'a [Projector],
    /// Cartesian k-points (Bohr^-1), in the order of `Layout::weights`.
    pub kpoints: &'a [[f64; 3]],
    pub sites: Sites<'a>,
    pub shapes: &'a Shapes,
    /// One local potential per atom (its short-range sphere).
    pub potentials: &'a [LocalPotential],
    /// One smooth core per atom, `None` without.
    pub cores: &'a [Option<Core<'a>>],
    /// The reciprocal cutoff of the dense sums (as of `electrostatics::couplings`).
    pub dense_cutoff: f64,
}

/// The state-independent operators the contraction needs: the channel blocks `B_a`, `q` and
/// `D_ion` (all `p x p`).
pub struct Operators {
    pub blocks: Vec<Matrix>,
    pub augmentation: Matrix,
    pub ion_operator: Matrix,
}

/// The fields of one density as the geometry reads them. `local[s]` is the potential the basis
/// sees in spin channel `s` on the mesh, `xc` the exchange-correlation potential against which
/// the partial core is differentiated (ONE array: the partial core is spin-unpolarized, so it
/// sees the spin average `(v_up + v_dn) / 2` of a polarized run, never `v_up` alone; no sum rule
/// catches the factor), `weights` the reciprocal weights `K rho^* / V` of the
/// hats and ions (one per node), `moments` the channel potentials `m_a`.
pub struct Fields {
    pub local: Vec<Vec<f64>>,
    pub xc: Vec<f64>,
    pub weights: Vec<C64>,
    pub moments: Vec<C64>,
    hats: OnceCell<Vec<[C64; 3]>>,
    ions: OnceCell<Vec<[C64; 3]>>,
}

impl Fields {
    pub fn new(local: Vec<Vec<f64>>, xc: Vec<f64>, weights: Vec<C64>, moments: Vec<C64>) -> Self {
        Self {
            local,
            xc,
            weights,
            moments,
            hats: OnceCell::new(),
            ions: OnceCell::new(),
        }
    }
}

/// The geometry provider.
pub struct SetupDerivatives<'a> {
    ctx: &'a Context,
    inputs: Inputs<'a>,
    layout: Layout,
    operators: Operators,
    mesh: Mesh,
    samples: RefCell<Option<KSamples>>,
    spheres: RefCell<Option<(usize, Vec<[Matrix; 3]>)>>,
    dense: OnceCell<DenseGradients>,
    linear: OnceCell<Vec<Vec<[C64; 3]>>>,
    constants: OnceCell<Vec<[f64; 3]>>,
}

impl<'a> SetupDerivatives<'a> {
    pub fn new(
        ctx: &'a Context,
        inputs: Inputs<'a>,
        layout: Layout,
        operators: Operators,
    ) -> Result<Self> {
        let atoms = inputs.sites.atoms.len();
        let p = inputs.projectors.len();
        let consistent = layout.atoms == atoms
            && layout.m == inputs.shells.functions()
            && layout.p == p
            && layout.weights.len() == inputs.kpoints.len()
            && inputs.potentials.len() == atoms
            && inputs.cores.len() == atoms
            && operators.blocks.len() == inputs.sites.channels.len()
            && inputs.projectors.iter().all(|q| q.atom < atoms)
            && layout
                .projector_owner
                .iter()
                .zip(inputs.projectors)
                .all(|(owner, q)| *owner == q.atom);
        if !consistent {
            return Err(invalid(
                "crystal gradient: the layout, the operators and the setup disagree",
            ));
        }
        Ok(Self {
            ctx,
            mesh: Mesh::new(ctx, inputs.grid)?,
            inputs,
            layout,
            operators,
            samples: RefCell::new(None),
            spheres: RefCell::new(None),
            dense: OnceCell::new(),
            linear: OnceCell::new(),
            constants: OnceCell::new(),
        })
    }

    fn kpoint(&self, k: usize) -> Result<[f64; 3]> {
        self.inputs
            .kpoints
            .get(k)
            .copied()
            .ok_or_else(|| invalid("crystal gradient: k-point out of range"))
    }

    /// Runs `f` on the derivative samples of k-point `k`, forming them once for the four terms.
    fn with_samples<T>(&self, k: usize, f: impl FnOnce(&KSamples) -> Result<T>) -> Result<T> {
        let kpoint = self.kpoint(k)?;
        let mut cache = self.samples.borrow_mut();
        if cache.as_ref().is_none_or(|s| s.k != kpoint) {
            *cache = None; // release the previous k-point before the next is reserved
            *cache = Some(KSamples::new(
                self.ctx,
                self.inputs.shells,
                &self.inputs.sites.lattice,
                &self.mesh,
                kpoint,
            )?);
        }
        f(cache.as_ref().expect("formed above"))
    }

    /// `G_A` of every sphere at k-point `k`, shared by the rows' share and the sphere's share.
    fn with_spheres<T>(&self, k: usize, f: impl FnOnce(&[[Matrix; 3]]) -> T) -> Result<T> {
        let kpoint = self.kpoint(k)?;
        let mut cache = self.spheres.borrow_mut();
        if cache.as_ref().is_none_or(|(held, _)| *held != k) {
            let spheres: Vec<ShortRange> = self
                .inputs
                .sites
                .atoms
                .iter()
                .zip(self.inputs.potentials)
                .map(|(atom, potential)| ShortRange {
                    center: atom.center,
                    potential,
                    sigma: self.inputs.sites.sigma,
                })
                .collect();
            let blocks = short_range_gradients(
                self.ctx,
                self.inputs.shells,
                &self.inputs.sites.lattice,
                kpoint,
                &spheres,
            )?;
            *cache = Some((k, blocks));
        }
        Ok(f(&cache.as_ref().expect("formed above").1))
    }

    fn hats<'f>(&self, fields: &'f Fields) -> Result<&'f [[C64; 3]]> {
        if fields.weights.len() != self.mesh.len() {
            return Err(invalid("crystal gradient: one reciprocal weight per node"));
        }
        if fields.hats.get().is_none() {
            let value = hat_overlaps(
                self.ctx,
                self.inputs.grid,
                self.inputs.shapes,
                &self.inputs.sites,
                &fields.weights,
            )?;
            let _ = fields.hats.set(value);
        }
        Ok(fields.hats.get().expect("set above"))
    }

    fn ions<'f>(&self, fields: &'f Fields) -> Result<&'f [[C64; 3]]> {
        if fields.weights.len() != self.mesh.len() {
            return Err(invalid("crystal gradient: one reciprocal weight per node"));
        }
        if fields.ions.get().is_none() {
            let value = ion_overlaps(
                self.ctx,
                self.inputs.grid,
                &self.inputs.sites,
                &fields.weights,
            )?;
            let _ = fields.ions.set(value);
        }
        Ok(fields.ions.get().expect("set above"))
    }

    fn dense_gradients(&self) -> Result<&DenseGradients> {
        if self.dense.get().is_none() {
            let value = dense_gradients(
                self.ctx,
                self.inputs.shapes,
                &self.inputs.sites,
                self.inputs.dense_cutoff,
            )?;
            let _ = self.dense.set(value);
        }
        Ok(self.dense.get().expect("set above"))
    }
}

fn axis_of(axis: usize) -> Result<usize> {
    if axis < 3 {
        Ok(axis)
    } else {
        Err(invalid("a gradient axis is 0, 1 or 2"))
    }
}

impl Derivatives for SetupDerivatives<'_> {
    type Fields = Fields;

    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn blocks(&self) -> &[Matrix] {
        &self.operators.blocks
    }
    fn augmentation(&self) -> &Matrix {
        &self.operators.augmentation
    }
    fn ion_operator(&self) -> &Matrix {
        &self.operators.ion_operator
    }
    fn projection(&self, k: usize) -> Result<Matrix> {
        projection(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            self.inputs.projectors,
        )
    }
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        projection_gradient(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            self.inputs.projectors,
        )
    }
    fn cross(&self, spin: usize, k: usize, term: Term, fields: &Fields) -> Result<[Matrix; 3]> {
        match term {
            Term::Overlap => self.with_samples(k, |s| grid_cross(self.ctx, &self.mesh, s, None)),
            Term::Kinetic => self.with_samples(k, |s| {
                kinetic_cross(self.ctx, self.inputs.grid, &self.mesh, s)
            }),
            Term::Local => {
                let v = fields
                    .local
                    .get(spin)
                    .ok_or_else(|| invalid("crystal gradient: one local potential per spin"))?;
                self.with_samples(k, |s| grid_cross(self.ctx, &self.mesh, s, Some(v)))
            }
            // The rows' share of every sphere: the sum over the spheres of `G_A`.
            Term::ShortRange => self.with_spheres(k, |blocks| {
                let m = self.layout.m;
                std::array::from_fn(|axis| {
                    blocks
                        .iter()
                        .fold(Matrix::zeros(m, m), |sum, g| sum + &g[axis])
                })
            }),
        }
    }
    fn sphere_cross(&self, _spin: usize, k: usize, _fields: &Fields) -> Result<Vec<[Matrix; 3]>> {
        self.with_spheres(k, |blocks| blocks.to_vec())
    }

    fn fields(&self, _state: &State, _q: &[C64]) -> Result<Fields> {
        Err(Error::Unsupported(
            "the potentials of a density need the SCF's own routines (solver.rs keeps them \
             private): build the fields and the frozen state outside and contract them"
                .into(),
        ))
    }
    fn moments<'f>(&self, fields: &'f Fields) -> &'f [C64] {
        &fields.moments
    }
    fn bands(
        &self,
        _spin: usize,
        _k: usize,
        _fields: &Fields,
        _hubbard: Option<&Matrix>,
    ) -> Result<Bands> {
        Err(Error::Unsupported(
            "the diagonalization of the crystal Hamiltonian is the SCF's (solver.rs keeps it \
             private): build the frozen state outside and contract it"
                .into(),
        ))
    }

    fn hat_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<Vec<C64>> {
        let axis = axis_of(axis)?;
        let all = self.hats(fields)?;
        // A channel moves with its owner alone.
        Ok(self
            .inputs
            .sites
            .channels
            .iter()
            .zip(all)
            .map(|(channel, d)| {
                if channel.0 == atom {
                    d[axis]
                } else {
                    C64::new(0., 0.)
                }
            })
            .collect())
    }
    fn ion_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        let axis = axis_of(axis)?;
        self.ions(fields)?
            .get(atom)
            .map(|d| d[axis].re)
            .ok_or_else(|| invalid("crystal gradient: atom out of range"))
    }
    fn ion_energy(&self, atom: usize, axis: usize) -> Result<f64> {
        let axis = axis_of(axis)?;
        if atom >= self.layout.atoms {
            return Err(invalid("crystal gradient: atom out of range"));
        }
        Ok(self.dense_gradients()?.get(atom, axis).e_ion)
    }
    fn dense(&self, atom: usize, axis: usize) -> Result<Dense> {
        let axis = axis_of(axis)?;
        if atom >= self.layout.atoms {
            return Err(invalid("crystal gradient: atom out of range"));
        }
        let channels = self.inputs.sites.channels.len();
        let item = self.dense_gradients()?.get(atom, axis);
        if self.linear.get().is_none() {
            let value = short_range_ion_compensation_gradient(
                self.ctx,
                &self.inputs.sites,
                self.inputs.potentials,
            )?;
            let _ = self.linear.set(value);
        }
        let linear = self.linear.get().expect("set above");
        Ok(Dense {
            u: Matrix::from_row_slice(channels, channels, &item.u),
            uion: item.u_ion.clone(),
            linear: linear.iter().map(|row| row[atom][axis]).collect(),
        })
    }
    fn constant(&self, atom: usize, axis: usize) -> Result<f64> {
        let axis = axis_of(axis)?;
        if self.constants.get().is_none() {
            let value = ion_constants_gradient(&self.inputs.sites)?;
            let _ = self.constants.set(value);
        }
        self.constants
            .get()
            .expect("set above")
            .get(atom)
            .map(|d| d[axis])
            .ok_or_else(|| invalid("crystal gradient: atom out of range"))
    }
    fn core_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        let axis = axis_of(axis)?;
        let Some(core) = self
            .inputs
            .cores
            .get(atom)
            .ok_or_else(|| invalid("crystal gradient: atom out of range"))?
        else {
            return Ok(0.);
        };
        let slope = core_gradient(
            self.ctx,
            self.inputs.grid,
            core.radius,
            core.values,
            self.inputs.sites.atoms[atom].center,
            axis,
        )?;
        mesh_dot(&self.mesh, &fields.xc, slope.as_slice())
    }
}
