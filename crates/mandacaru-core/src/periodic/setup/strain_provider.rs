//! The Rust setup as a strain provider of the crystal stress
//! ([`crate::periodic::stress_terms::Strain`]).
//!
//! Everything that depends on the cell and not on the converged state is implemented here from the
//! finished strain pieces of the setup: the weighted Bloch tangents ([`super::strain_samples`])
//! and the grid forms built on them ([`super::strain_pulay`]), the sphere derivatives
//! ([`super::strain_spheres`]), the grid arrays ([`super::strain_grid`]) and the dense sums
//! ([`super::strain_sums`]). The inputs are the force provider's ([`super::provider::Inputs`],
//! [`super::provider::Operators`]); the output-density data a strain term reads beyond the
//! potentials ([`StrainFields`]: the reciprocal density, the xc energy and, for a GGA, the
//! gradients and the flux) are handed in by the caller, who owns the state (a real SCF crystal
//! fills them in AS-6; a test builds them from a density it holds).
//!
//! A spinor crystal (`Layout::request.spinor`, described by the SCALAR `M` functions and `P`
//! projectors) is presented in the doubled frame of the force engine (`spinor`, a child module):
//! `2M`, `2P`, `I2 x (.)` operators, projections and matrix tangents, the Pauli sum of the local
//! term's channel tangents, and the spin-orbit union-projector term of
//! [`StrainSetup::with_spin_orbit`].
//!
//! The dense sums run over the ball `|G| <= dense_cutoff` of the REFERENCE cell, or over a
//! caller-supplied list of reciprocal vectors ([`StrainSetup::with_dense_vectors`], the pinned
//! integer triples of the plan's section 2 at the reference cell): the derivative of a
//! re-enumerated list would add a jump that is not a derivative.
mod spinor;

use super::{
    basis_gradient::ShortRange,
    electrostatics::{hats, ion_charge},
    gradients::ion_constants_strain,
    provider::{Inputs, Operators},
    pulay::{mesh_dot, projection, Matrix, Mesh},
    shapes::visit_lattice,
    strain_grid::{core_strain, hats_strain, ions_strain, kernel_strain, VOIGT},
    strain_pulay::{bloch_values, grid_strain, kinetic_strain, Tensor as Matrices},
    strain_samples::StrainSamples,
    strain_spheres::{linear_strain, projection_strain, short_range_strain},
    strain_sums::CouplingsStrainSum,
};
use crate::{
    models,
    periodic::{
        gradient::{Layout, SpinOrbit, Term},
        invalid,
        point_group::PointGroup,
        stress_terms::{
            electrostatic::{self, Grid, Hartree},
            xc, Dense, Strain, Tensor,
        },
    },
    quantum::C64,
    Context, Result,
};

/// The output-density data of the xc strain terms.
#[derive(Clone, Debug)]
pub struct XcFields {
    /// `E_xc[rho + rho_core]` of the output density (Hartree).
    pub energy: f64,
    /// The spin-averaged xc potential `(v_up + v_dn) / 2` per node (what the partial core sees).
    pub potential: Vec<f64>,
    /// `partial_c n_s` for the two rotated spin channels, `(6, n)` with row `3 s + c`; `None` for
    /// a local functional.
    pub gradients: Option<Vec<f64>>,
    /// `d f / d (partial_c n_s)`, the same layout; `None` for a local functional.
    pub flux: Option<Vec<f64>>,
}

/// The fields of one output density as the strain terms read them.
#[derive(Clone, Debug)]
pub struct StrainFields {
    /// The potential the basis sees in spin channel `s` (`v_ee + v_xc`) per node.
    pub local: Vec<Vec<f64>>,
    /// The channel potentials `m_a`.
    pub moments: Vec<C64>,
    /// `K rho_G^* / Omega`.
    pub weights: Vec<C64>,
    /// `rho_G = dv sum_i rho_i e^{-iG.r_i}` (the spin sum), per node of the reciprocal set.
    pub density: Vec<C64>,
    pub xc: XcFields,
}

/// The strain provider.
pub struct StrainSetup<'a> {
    ctx: &'a Context,
    inputs: Inputs<'a>,
    /// The layout the contraction reads: the scalar one, or its doubled frame for a spinor.
    layout: Layout,
    /// The number of basis functions of one spin block (`layout.m` is `2M` for a spinor).
    functions: usize,
    operators: Operators,
    spin_orbit: Option<spinor::SpinOrbitData<'a>>,
    mesh: Mesh,
    dense_vectors: Option<Vec<[f64; 3]>>,
    /// The operations of a symmetry-reduced mesh, averaged over by the contraction.
    point_group: Option<PointGroup>,
}

impl<'a> StrainSetup<'a> {
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
            && operators.augmentation.shape() == (p, p)
            && operators.ion_operator.shape() == (p, p)
            && operators.blocks.iter().all(|b| b.shape() == (p, p))
            && inputs.projectors.iter().all(|q| q.atom < atoms);
        if !consistent {
            return Err(invalid(
                "crystal stress: the layout, the operators and the setup disagree",
            ));
        }
        let functions = layout.m;
        let (layout, operators) = spinor::frame_of(layout, operators)?;
        Ok(Self {
            ctx,
            mesh: Mesh::new(ctx, inputs.grid)?,
            inputs,
            layout,
            functions,
            operators,
            spin_orbit: None,
            dense_vectors: None,
            point_group: None,
        })
    }

    /// Averages the stress over the operations a symmetry-reduced mesh was built with (CF-6).
    pub fn with_point_group(mut self, group: Option<PointGroup>) -> Self {
        self.point_group = group;
        self
    }

    /// The same, in place (the controls of the symmetrizer replace the crystal's own group).
    #[doc(hidden)]
    pub fn set_point_group(&mut self, group: Option<PointGroup>) {
        self.point_group = group;
    }

    /// Sums the dense sums over these reciprocal vectors (Cartesian, at the reference cell)
    /// instead of the ball of the reference cell.
    pub fn with_dense_vectors(mut self, vectors: Vec<[f64; 3]>) -> Self {
        self.dense_vectors = Some(vectors);
        self
    }

    fn kpoint(&self, k: usize) -> Result<[f64; 3]> {
        self.inputs
            .kpoints
            .get(k)
            .copied()
            .ok_or_else(|| invalid("crystal stress: k-point out of range"))
    }

    fn lattice(&self) -> &nalgebra::Matrix3<f64> {
        &self.inputs.sites.lattice
    }

    fn check_fields(&self, fields: &StrainFields) -> Result<()> {
        let n = self.mesh.len();
        let channels = if self.layout.request.spinor {
            matches!(fields.local.len(), 1 | 4)
        } else {
            fields.local.len() == self.layout.spins
        };
        let fine = channels
            && fields.local.iter().all(|v| v.len() == n)
            && fields.weights.len() == n
            && fields.density.len() == n
            && fields.xc.potential.len() == n
            && fields.moments.len() == self.inputs.sites.channels.len();
        if fine {
            Ok(())
        } else {
            Err(invalid(
                "crystal stress: the fields need one value per mesh node and one moment per channel",
            ))
        }
    }

    /// The matrix tangent of one term in the SCALAR frame (`M x M`) for the potential of channel
    /// `spin`.
    fn scalar_strain(
        &self,
        spin: usize,
        k: usize,
        term: Term,
        fields: &StrainFields,
    ) -> Result<Matrices> {
        let (ctx, shells, kpoint) = (self.ctx, self.inputs.shells, self.kpoint(k)?);
        let strain = || StrainSamples::new(shells, self.lattice(), &self.mesh, kpoint);
        let values = || bloch_values(ctx, shells, self.lattice(), &self.mesh, kpoint);
        let tensor: Matrices = match term {
            Term::Overlap => {
                grid_strain(ctx, &self.mesh, &strain(), values()?.as_slice(), None, true)?
            }
            Term::Kinetic => kinetic_strain(
                ctx,
                self.inputs.grid,
                &self.mesh,
                &strain(),
                values()?.as_slice(),
                kpoint,
            )?,
            Term::Local => {
                let v = fields
                    .local
                    .get(spin)
                    .ok_or_else(|| invalid("crystal stress: one local potential per spin"))?;
                grid_strain(
                    ctx,
                    &self.mesh,
                    &strain(),
                    values()?.as_slice(),
                    Some(v),
                    false,
                )?
            }
            Term::ShortRange => {
                let sites = &self.inputs.sites;
                let spheres: Vec<ShortRange> = sites
                    .atoms
                    .iter()
                    .zip(self.inputs.potentials)
                    .map(|(atom, potential)| ShortRange {
                        center: atom.center,
                        potential,
                        sigma: sites.sigma,
                    })
                    .collect();
                let per_sphere = short_range_strain(ctx, shells, self.lattice(), kpoint, &spheres)?;
                let m = self.functions;
                let mut total: Matrices = collect(|_, _| Matrix::zeros(m, m));
                for sphere in &per_sphere {
                    for a in 0..3 {
                        for b in 0..3 {
                            total[a][b] += &sphere[a][b];
                        }
                    }
                }
                total
            }
        };
        Ok(tensor)
    }
}

fn collect<T>(f: impl Fn(usize, usize) -> T) -> Tensor<T> {
    std::array::from_fn(|a| std::array::from_fn(|b| f(a, b)))
}

impl Strain for StrainSetup<'_> {
    type Fields = StrainFields;

    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn point_group(&self) -> Option<&PointGroup> {
        self.point_group.as_ref()
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
    fn volume(&self) -> f64 {
        self.mesh.volume
    }

    fn projection(&self, k: usize) -> Result<Matrix> {
        Ok(self.frame(projection(
            self.ctx,
            self.inputs.shells,
            self.lattice(),
            self.kpoint(k)?,
            self.inputs.projectors,
        )?))
    }

    fn projection_strain(&self, k: usize) -> Result<Tensor<Matrix>> {
        Ok(self.frame_tensor(projection_strain(
            self.ctx,
            self.inputs.shells,
            self.lattice(),
            self.kpoint(k)?,
            self.inputs.projectors,
        )?))
    }

    fn matrix_strain(
        &self,
        spin: usize,
        k: usize,
        term: Term,
        fields: &StrainFields,
    ) -> Result<Tensor<Matrix>> {
        self.check_fields(fields)?;
        if self.layout.request.spinor && term == Term::Local {
            return self.spinor_local(k, fields);
        }
        Ok(self.frame_tensor(self.scalar_strain(spin, k, term, fields)?))
    }

    fn moments<'f>(&self, fields: &'f StrainFields) -> &'f [C64] {
        &fields.moments
    }

    fn hartree(&self, fields: &StrainFields, q: &[C64]) -> Result<Hartree> {
        self.check_fields(fields)?;
        let (ctx, grid, sites) = (self.ctx, self.inputs.grid, &self.inputs.sites);
        let (shapes, n) = (self.inputs.shapes, self.mesh.len());
        if q.len() != sites.channels.len() {
            return Err(invalid("crystal stress: one moment per channel"));
        }
        let table = hats(ctx, grid, shapes, sites)?;
        let ion = ion_charge(ctx, grid, sites)?;
        let mut charge = ctx.allocate::<C64>(n)?;
        for (j, to) in charge.as_mut_slice().iter_mut().enumerate() {
            *to = ion.as_slice()[j]
                + q.iter()
                    .enumerate()
                    .map(|(a, qa)| qa * table.as_slice()[a * n + j])
                    .sum::<C64>();
        }
        electrostatic::hartree(&Grid {
            volume: self.mesh.volume,
            density: &fields.density,
            weights: &fields.weights,
            charge,
            kernel: kernel_strain(ctx, grid)?,
            hats: hats_strain(ctx, grid, shapes, sites, q)?,
            ions: ions_strain(ctx, grid, sites)?,
        })
    }

    fn dense(&self) -> Result<Dense> {
        let (ctx, sites, shapes) = (self.ctx, &self.inputs.sites, self.inputs.shapes);
        let sums = match &self.dense_vectors {
            Some(vectors) => {
                let mut sum = CouplingsStrainSum::new(ctx, shapes, sites)?;
                for chunk in vectors.chunks(sum.block()) {
                    sum.add(chunk)?;
                }
                sum.finish()
            }
            None => {
                // The ball of the reference cell, enumerated as `couplings` does.
                let mut sum = CouplingsStrainSum::new(ctx, shapes, sites)?;
                let block = sum.block();
                visit_lattice(
                    &sites.reciprocal()?,
                    self.inputs.dense_cutoff,
                    true,
                    block,
                    &mut |vectors| sum.add(vectors),
                )?;
                sum.finish()
            }
        };
        let c = sites.channels.len();
        let linear = linear_strain(ctx, sites, self.inputs.potentials)?;
        Ok(Dense {
            u: collect(|a, b| Matrix::from_row_slice(c, c, sums.u(a, b))),
            u_ion: collect(|a, b| sums.u_ion(a, b).to_vec()),
            linear: collect(|a, b| linear.iter().map(|channel| channel[a][b]).collect()),
            e_ion: sums.e_ion,
        })
    }

    fn constants(&self) -> Result<Tensor<f64>> {
        ion_constants_strain(&self.inputs.sites)
    }

    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        Ok(&self.spin_orbit_data()?.operator)
    }

    fn spin_orbit_projection(&self, k: usize) -> Result<Matrix> {
        projection(
            self.ctx,
            self.inputs.shells,
            self.lattice(),
            self.kpoint(k)?,
            self.spin_orbit_data()?.projectors,
        )
    }

    fn spin_orbit_strain(&self, k: usize) -> Result<Tensor<Matrix>> {
        projection_strain(
            self.ctx,
            self.inputs.shells,
            self.lattice(),
            self.kpoint(k)?,
            self.spin_orbit_data()?.projectors,
        )
    }

    fn xc(&self, fields: &StrainFields) -> Result<xc::Terms> {
        self.check_fields(fields)?;
        // A gradient functional without its gradients and flux would give a silently zero term.
        let gga = matches!(
            models::functional(&self.layout.request.functional)?.semilocal,
            "pbe" | "hse06"
        );
        if gga != fields.xc.gradients.is_some() || gga != fields.xc.flux.is_some() {
            return Err(invalid(
                "crystal stress: a gradient functional needs the gradients and the flux of its \
                 rotated spin channels, a local one has none",
            ));
        }
        let (ctx, grid) = (self.ctx, self.inputs.grid);
        let n = self.mesh.len();
        let mut core = [[0.; 3]; 3];
        for (atom, table) in self.inputs.cores.iter().enumerate() {
            let Some(table) = table else { continue };
            let center = self.inputs.sites.atoms[atom].center;
            for &(a, b) in &VOIGT {
                let slope = core_strain(ctx, grid, table.radius, table.values, &[center], (a, b))?;
                let t = mesh_dot(&self.mesh, &fields.xc.potential, slope.as_slice())?;
                core[a][b] += t;
                if a != b {
                    core[b][a] += t;
                }
            }
        }
        Ok(xc::Terms {
            jacobian: xc::jacobian(fields.xc.energy),
            gradient: xc::gradient(
                self.mesh.dv,
                n,
                fields.xc.gradients.as_deref(),
                fields.xc.flux.as_deref(),
            )?,
            core,
        })
    }
}
