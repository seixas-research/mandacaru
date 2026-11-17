//! The Kohn-Sham states of one k-point as the static response reads them: energies, the weight of
//! each state in the correlated (target) subspace, the pseudo Bloch functions on the cell grid and
//! their projector overlaps. A view: the samples belong to a Wannier frame (or to a test).
use crate::{
    math::storage::{ComplexRead, ComplexSamples},
    periodic::invalid,
    quantum::C64,
    Context, Result,
};
use nalgebra::DMatrix;

pub struct Bloch<'a> {
    /// The k-point in the reciprocal lattice's coordinates.
    pub fractional: [f64; 3],
    /// The first `energies.len()` states, ascending, in Hartree on the SCF's Fermi scale.
    pub energies: &'a [f64],
    /// Each state's weight `p_a = sum_{n in T} |<psi_a|S|phi_n>|^2` in the target subspace.
    pub target: Vec<f64>,
    /// State `a` is `samples[a * n .. (a + 1) * n]`: the Bloch function (not its periodic part)
    /// at the grid points of the cell. A spinor's (`components == 2`) samples are
    /// `[component][state][point]` over exactly the `energies.len()` states.
    pub samples: &'a ComplexSamples,
    /// Rows `<psi_a|p_i>` over the crystal's projectors (a spinor's: `(component, state)`);
    /// `None` for states without a PAW augmentation.
    pub projections: Option<&'a DMatrix<C64>>,
    /// Grid points of the cell.
    pub points: usize,
    /// 1, or 2 for spinors (the transition densities are charge densities, summed over both).
    pub components: usize,
}

impl<'a> Bloch<'a> {
    pub fn check(&self) -> Result<()> {
        let bands = self.energies.len();
        let rows = self.components * bands;
        if bands == 0
            || !(1..=2).contains(&self.components)
            || self.target.len() != bands
            || self.samples.len() < rows * self.points
            || (self.components == 2 && self.samples.len() != rows * self.points)
            || self.projections.is_some_and(|p| p.nrows() < rows)
            || self.energies.iter().any(|e| !e.is_finite())
            || self
                .target
                .iter()
                .any(|p| !(-1e-10..=1. + 1e-10).contains(p))
        {
            return Err(invalid(
                "the Kohn-Sham states of a k-point are inconsistent",
            ));
        }
        Ok(())
    }
    pub fn bands(&self) -> usize {
        self.energies.len()
    }
    /// The grid values of state `a` (its first component).
    pub fn state(&self, ctx: &Context, a: usize) -> Result<ComplexRead<'_>> {
        self.samples.read_range(ctx, a * self.points, self.points)
    }
    /// `<psi_a|p_i>` of state `a` (its first component); empty without a PAW augmentation.
    pub fn projection(&self, a: usize) -> Vec<C64> {
        self.component_projection(a, 0)
    }
    /// `<psi_a^c|p_i>` of component `c` of state `a`; empty without a PAW augmentation.
    pub fn component_projection(&self, a: usize, c: usize) -> Vec<C64> {
        let row = c * self.bands() + a;
        self.projections
            .map(|p| (0..p.ncols()).map(|i| p[(row, i)]).collect())
            .unwrap_or_default()
    }
}
