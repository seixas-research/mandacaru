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
//! [`fields`](Derivatives::fields) and [`bands`](Derivatives::bands) exist once the geometry is
//! tied to the SCF crystal whose state is differentiated ([`SetupDerivatives::with_scf`]): they
//! are the potentials of the output density, the re-diagonalization with them and the Fermi level
//! over the whole mesh, from [`crate::periodic::gradient::scf`]. Without that crystal they raise
//! `Unsupported`, and a caller builds the [`Frozen`](crate::periodic::gradient::Frozen) state
//! itself and contracts it with [`contract`](crate::periodic::gradient::contract).
//!
//! A hybrid crystal also gets its exact-exchange terms (`hybrid`, a child module): the fields
//! re-diagonalize with `-a K[D_out]`, and [`exchange_gradient`](Derivatives::exchange_gradient)
//! returns the explicit derivative of the screened exchange, the basis functions moving.
//!
//! A spinor crystal (`Layout::request.spinor`, described by the SCALAR `M` functions and `P`
//! projectors) is presented to the engine in the doubled frame of
//! [`gradient::spinor`](crate::periodic::gradient::spinor): `2M`, `2P`, `I2 x (.)` operators and
//! projections, the Pauli sum of the local term. Its spin-orbit projectors and couplings come
//! with [`SetupDerivatives::with_spin_orbit`] (`spinor`, a child module).
mod hybrid;
mod spinor;

use super::{
    basis_gradient::ShortRange,
    electrostatics::ion_charge,
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
        gradient::{
            scf::{self, OutputFields},
            spinor::{doubled, doubled_layout},
            Bands, Dense, Derivatives, Layout, SpinOrbit, State, Term,
        },
        hybrid_gradient, invalid,
        point_group::PointGroup,
        Crystal, Options,
    },
    quantum::C64,
    Context, Error, Reservation, Result,
};
use std::cell::{OnceCell, RefCell};

/// Gates of the geometry tie of [`SetupDerivatives::with_scf`]: the change of basis between the
/// setup's lattice and the grid's cell is an integer matrix to `CELL_TOLERANCE`, and the ion table
/// of a Rust setup agrees with the Python-built one to 3.4e-16 relative (measured on Si, Al2 and
/// LiH, `periodic_gradient_crystals_setup.rs`) against `ION_TOLERANCE`.
const CELL_TOLERANCE: f64 = 1e-9;
const ION_TOLERANCE: f64 = 1e-11;

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
    /// The re-diagonalized bands of the density, when the fields come from a real SCF crystal
    /// ([`SetupDerivatives::with_scf`]): the Fermi level needs the whole mesh, so
    /// [`Derivatives::bands`] returns these.
    bands: Vec<Vec<Bands>>,
    hats: OnceCell<Vec<[C64; 3]>>,
    ions: OnceCell<Vec<[C64; 3]>>,
    _held: Option<Reservation>,
}

impl Fields {
    pub fn new(local: Vec<Vec<f64>>, xc: Vec<f64>, weights: Vec<C64>, moments: Vec<C64>) -> Self {
        Self {
            local,
            xc,
            weights,
            moments,
            bands: Vec::new(),
            hats: OnceCell::new(),
            ions: OnceCell::new(),
            _held: None,
        }
    }
}

/// The geometry provider.
pub struct SetupDerivatives<'a> {
    ctx: &'a Context,
    inputs: Inputs<'a>,
    /// The layout the engine reads: the scalar one, or its doubled frame for a spinor.
    layout: Layout,
    /// The number of basis functions of one spin block (`layout.m` is `2M` for a spinor).
    functions: usize,
    operators: Operators,
    spin_orbit: Option<spinor::SpinOrbitData<'a>>,
    mesh: Mesh,
    samples: RefCell<Option<KSamples>>,
    spheres: RefCell<Option<(usize, Vec<[Matrix; 3]>)>>,
    dense: OnceCell<DenseGradients>,
    linear: OnceCell<Vec<Vec<[C64; 3]>>>,
    constants: OnceCell<Vec<[f64; 3]>>,
    scf: Option<(&'a Crystal, Options)>,
    /// The operations of a symmetry-reduced mesh (from the SCF crystal), averaged over by the engine.
    point_group: Option<PointGroup>,
}

impl<'a> SetupDerivatives<'a> {
    pub fn new(
        ctx: &'a Context,
        inputs: Inputs<'a>,
        layout: Layout,
        operators: Operators,
    ) -> Result<Self> {
        if layout.spins != 1 && layout.spins != 2 {
            return Err(invalid("crystal gradient: one or two spin channels"));
        }
        let atoms = inputs.sites.atoms.len();
        let p = inputs.projectors.len();
        let functions = layout.m;
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
        let (layout, operators) = if layout.request.spinor {
            let operators = Operators {
                blocks: operators.blocks.iter().map(doubled).collect(),
                augmentation: doubled(&operators.augmentation),
                ion_operator: doubled(&operators.ion_operator),
            };
            (doubled_layout(&layout)?, operators)
        } else {
            (layout, operators)
        };
        Ok(Self {
            ctx,
            mesh: Mesh::new(ctx, inputs.grid)?,
            inputs,
            layout,
            functions,
            operators,
            spin_orbit: None,
            samples: RefCell::new(None),
            spheres: RefCell::new(None),
            dense: OnceCell::new(),
            linear: OnceCell::new(),
            constants: OnceCell::new(),
            scf: None,
            point_group: None,
        })
    }

    /// Ties the geometry to the SCF crystal whose state is differentiated, so that
    /// [`Derivatives::fields`] and [`Derivatives::bands`] exist: the layout, the mesh, the
    /// k-points, the dimensions, the cell and the atoms (through the crystal's ion table) must be
    /// the crystal's, and the crystal one this engine differentiates ([`scf::refuse`]). `options` are those of the SCF (smearing and width
    /// define the occupations).
    pub fn with_scf(mut self, crystal: &'a Crystal, options: Options) -> Result<Self> {
        scf::refuse(crystal)?;
        let l = &self.layout;
        let agree = self.functions == crystal.m
            && self.inputs.projectors.len() == crystal.p
            && l.request.spinor == crystal.spinor
            && l.spins == crystal.spins
            && l.degeneracy == crystal.degeneracy()
            && l.reference == crystal.reference
            && l.weights.len() == crystal.points.len()
            && self.inputs.sites.channels.len() == crystal.channels
            && self.mesh.len() == crystal.grid.index.points()
            && l.weights
                .iter()
                .zip(&crystal.points)
                .all(|(w, p)| *w == p.weight)
            && self
                .inputs
                .kpoints
                .iter()
                .zip(&crystal.points)
                .all(|(k, p)| (0..3).all(|a| (k[a] - p.k[a]).abs() < 1e-12))
            && l.request.functional == crystal.functional;
        if !agree {
            return Err(invalid(
                "crystal gradient: the geometry and the SCF crystal disagree (sizes, k-points, \
                 weights, reference, degeneracy or functional)",
            ));
        }
        self.same_cell_and_ions(crystal)?;
        self.same_spheres(crystal)?;
        self.point_group = crystal
            .description
            .as_ref()
            .and_then(|d| d.point_group.clone());
        self.scf = Some((crystal, options));
        self.same_spin_orbit()?;
        Ok(self)
    }

    /// Replaces the operations the engine averages over (those of the SCF crystal, set by
    /// [`Self::with_scf`]); `None` leaves the k-sum of a reduced mesh unaveraged. For the controls
    /// of the symmetrizer: a caller of the engine takes the crystal's own.
    #[doc(hidden)]
    pub fn set_point_group(&mut self, group: Option<PointGroup>) {
        self.point_group = group;
    }

    /// The setup's lattice and atom centers are those the crystal's grid and ion charges were
    /// built from. Both matrices hold the cell vectors as COLUMNS: the grid's are `shape[j]`
    /// steps long, and they must generate the setup's lattice (the two bases differ by an integer
    /// unimodular matrix: a primitive cell and its transpose can be one lattice, as the circulant fcc cell
    /// is, and then the energy is the same). The Gaussian ions of the setup's atoms must
    /// reproduce the crystal's ion table, which carries every center, charge and width,
    /// translations included. A transposed lattice or a moved atom would otherwise
    /// differentiate another energy without any other symptom.
    fn same_cell_and_ions(&self, crystal: &Crystal) -> Result<()> {
        let shape = crystal.grid.index.shape();
        let lattice = &self.inputs.sites.lattice;
        let cell = nalgebra::Matrix3::from_fn(|i, j| shape[j] as f64 * crystal.grid.step[(i, j)]);
        let same = lattice.try_inverse().is_some_and(|inverse| {
            let change = inverse * cell;
            let integer = change
                .iter()
                .all(|x| (x - x.round()).abs() < CELL_TOLERANCE);
            integer && (change.determinant().abs() - 1.).abs() < CELL_TOLERANCE
        });
        if !same {
            return Err(invalid(
                "crystal gradient: the setup's lattice does not generate the SCF crystal's cell \
                 (a transposed or rescaled lattice?)",
            ));
        }
        let ions = ion_charge(self.ctx, &crystal.grid, &self.inputs.sites)?;
        let (gap, scale) = ions
            .as_slice()
            .iter()
            .zip(crystal.ion.as_slice())
            .fold((0f64, 0f64), |(gap, scale), (a, b)| {
                (gap.max((a - b).norm()), scale.max(b.norm()))
            });
        if gap > ION_TOLERANCE * scale {
            return Err(invalid(
                "crystal gradient: the setup's atoms do not reproduce the SCF crystal's ion \
                 charges (centers, charges or ion width differ)",
            ));
        }
        Ok(())
    }

    /// The row-center derivative of one matrix term in the SCALAR frame (`M x M`).
    fn scalar_cross(
        &self,
        spin: usize,
        k: usize,
        term: Term,
        fields: &Fields,
    ) -> Result<[Matrix; 3]> {
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
                let m = self.functions;
                std::array::from_fn(|axis| {
                    blocks
                        .iter()
                        .fold(Matrix::zeros(m, m), |sum, g| sum + &g[axis])
                })
            }),
        }
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

    fn point_group(&self) -> Option<&PointGroup> {
        self.point_group.as_ref()
    }

    fn symmetrize_moments(&self, q: &mut [C64]) -> Result<()> {
        let Some((crystal, _)) = &self.scf else {
            return Ok(());
        };
        let channels = crystal.channels;
        if q.len() != channels {
            return Err(invalid(format!(
                "the channel moments have {} entries, the SCF crystal's moment symmetrization \
                 acts on {channels}",
                q.len()
            )));
        }
        let raw = q.to_vec();
        for (a, out) in q.iter_mut().enumerate() {
            *out = (0..channels)
                .map(|b| crystal.qsym.as_slice()[a * channels + b] * raw[b])
                .sum();
        }
        Ok(())
    }

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
        let c = projection(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            self.inputs.projectors,
        )?;
        Ok(self.frame(c))
    }
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        let g = projection_gradient(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            self.inputs.projectors,
        )?;
        Ok(g.map(|g| self.frame(g)))
    }
    fn cross(&self, spin: usize, k: usize, term: Term, fields: &Fields) -> Result<[Matrix; 3]> {
        if self.layout.request.spinor && term == Term::Local {
            return self.spinor_local(k, fields);
        }
        Ok(self
            .scalar_cross(spin, k, term, fields)?
            .map(|g| self.frame(g)))
    }
    fn sphere_cross(&self, _spin: usize, k: usize, _fields: &Fields) -> Result<Vec<[Matrix; 3]>> {
        self.with_spheres(k, |blocks| {
            blocks
                .iter()
                .map(|g| g.clone().map(|g| self.frame(g)))
                .collect()
        })
    }

    fn fields(&self, state: &State, q: &[C64]) -> Result<Fields> {
        let Some((crystal, options)) = &self.scf else {
            return Err(Error::Unsupported(
                "the potentials of a density need the SCF crystal: tie the geometry to it with \
                 with_scf, or build the frozen state outside and contract it"
                    .into(),
            ));
        };
        // What the fields keep for the whole contraction, reserved before it is formed: the copies
        // of the potentials, weights and moments, and the re-diagonalized bands (`m x m`
        // coefficients, eigenvalues and occupations for every spin and k-point).
        let (n, m) = (self.mesh.len(), self.layout.m);
        let mesh_bytes =
            crate::product(&[crystal.field_channels() * n + n + 2 * n + 2 * q.len(), 8])?;
        let band_bytes = crate::product(&[
            self.layout.spins * self.layout.weights.len(),
            m * m * 16 + 2 * m * 8,
        ])?;
        let held = self.ctx.reserve(mesh_bytes + band_bytes)?;
        // A hybrid holds the exchange matrices (`m x m` per spin and k-point) while it diagonalizes.
        let exchange_bytes = if crystal.screened_exchange()?.is_some() {
            band_bytes
        } else {
            0
        };
        let _transient = self.ctx.reserve(exchange_bytes)?;
        let densities = scf::densities(crystal, &state.bands)?;
        let output = scf::output_fields(crystal, &densities, q)?;
        let exchange = scf::exchange_operators(crystal, &densities, options.exxdiv_treatment)?;
        let bands = scf::rediagonalize(crystal, &output, options, &exchange)?;
        let OutputFields {
            local,
            moments,
            xc,
            weights,
        } = output;
        let mut fields = Fields::new(
            local.as_slice().chunks(n).map(<[f64]>::to_vec).collect(),
            xc.as_slice().to_vec(),
            weights.as_slice().to_vec(),
            moments.as_slice().to_vec(),
        );
        fields.bands = bands;
        fields._held = Some(held);
        Ok(fields)
    }
    fn moments<'f>(&self, fields: &'f Fields) -> &'f [C64] {
        &fields.moments
    }
    fn bands(
        &self,
        spin: usize,
        k: usize,
        fields: &Fields,
        hubbard: Option<&Matrix>,
    ) -> Result<Bands> {
        if hubbard.is_some() {
            return Err(Error::Unsupported(
                "crystal forces are not available for a DFT+U crystal".into(),
            ));
        }
        fields
            .bands
            .get(spin)
            .and_then(|s| s.get(k))
            .cloned()
            .ok_or_else(|| {
                Error::Unsupported(
                    "the bands of a density come from the SCF crystal (with_scf); fields built \
                     outside carry none"
                        .into(),
                )
            })
    }

    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        self.spin_orbit_data().map(|so| &so.operator)
    }
    fn spin_orbit_projection(&self, k: usize) -> Result<Matrix> {
        let so = self.spin_orbit_data()?;
        projection(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            so.projectors,
        )
    }
    fn spin_orbit_projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        let so = self.spin_orbit_data()?;
        projection_gradient(
            self.ctx,
            self.inputs.shells,
            &self.inputs.sites.lattice,
            self.kpoint(k)?,
            so.projectors,
        )
    }

    fn exchange_gradient(&self, state: &State) -> Result<hybrid_gradient::Gradient> {
        self.exchange_terms(state)
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
