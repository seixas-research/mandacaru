//! The matrix terms of the stress at one k-point, contracted with the density before any matrix
//! of the strain is formed (see [`super::super::strain_pulay`]): the overlap `-W dS`, the grid
//! local potential, the alias-averaged kinetic matrix, the short-range spheres and a meta-GGA's
//! kinetic-density operator, in the scalar frame.
//!
//! A spinor request arrives in the doubled frame: the spin-diagonal terms contract the spin
//! trace `H = M_aa + M_bb`, and a Pauli-channel potential `v^c` contracts the component of the
//! density it multiplies (`sigma_c x dV^c`, `tr[M (sigma_c x dV)] = tr[M^c dV]`):
//!
//! ```text
//! M^0 = M_aa + M_bb,  M^1 = M_ba + M_ab,  M^2 = i (M_ab - M_ba),  M^3 = M_aa - M_bb,
//! ```
//!
//! the transpose of the blocks [`pauli_sum`](crate::periodic::gradient::spinor::pauli_sum)
//! scatters.
use super::{StrainFields, StrainSetup};
use crate::{
    math::gemm::complex_product,
    periodic::{
        gradient::Term,
        invalid,
        setup::{
            basis_gradient::ShortRange,
            pulay::{block_len, block_samples, kinetic_setup, KineticSetup, Matrix},
            shells::Sampling,
            strain_grid::VOIGT,
            strain_pulay::{bloch_values, dotc, hermitian, tau_strain},
            strain_samples::{weighted_derivatives_along, StrainSamples},
        },
        stress_terms::{Direction, SpinMatrices},
    },
    product,
    quantum::C64,
    Result,
};
use nalgebra::DMatrixView;
use num_complex::Complex64;

/// What one term contracts: its form and the scalar-frame matrix against it.
#[derive(Clone, Copy)]
enum Form<'f> {
    Overlap,
    Local(&'f [f64]),
    Kinetic,
    ShortRange,
    Tau(&'f [f64]),
}

struct Item<'f> {
    spin: usize,
    term: usize,
    form: Form<'f>,
    matrix: Matrix,
}

fn trace_re(left: &Matrix, right: &Matrix) -> f64 {
    let (n, m) = left.shape();
    let mut sum = C64::new(0., 0.);
    for j in 0..m {
        for i in 0..n {
            sum += left[(i, j)] * right[(j, i)];
        }
    }
    sum.re
}

impl StrainSetup<'_> {
    /// The scalar-frame `(potential channel, matrix)` pairs a term contracts for one spin channel.
    fn components<'f>(
        &self,
        term: Term,
        spin: usize,
        fields: &'f StrainFields,
        matrices: &[SpinMatrices],
    ) -> Result<Vec<(Option<&'f [f64]>, Matrix)>> {
        let m = self.functions;
        let source = matrices
            .get(spin)
            .ok_or_else(|| invalid("crystal stress: one density per spin channel"))?;
        let full = if term == Term::Overlap {
            &source.weighted
        } else {
            &source.density
        };
        let potential: Option<&'f [Vec<f64>]> = match term {
            Term::Local => Some(&fields.local),
            Term::KineticDensity => Some(
                fields
                    .kinetic
                    .as_ref()
                    .map(|(blocks, _)| blocks.as_slice())
                    .ok_or_else(|| {
                        invalid("crystal stress: the meta-GGA term needs the fields' v_tau")
                    })?,
            ),
            _ => None,
        };
        if !self.layout.request.spinor {
            if full.shape() != (m, m) {
                return Err(invalid("crystal stress: the density is not m x m"));
            }
            let channel = match potential {
                Some(blocks) => Some(
                    blocks
                        .get(spin)
                        .map(|v| v.as_slice())
                        .ok_or_else(|| invalid("crystal stress: one potential per spin"))?,
                ),
                None => None,
            };
            return Ok(vec![(channel, full.clone())]);
        }
        if full.shape() != (2 * m, 2 * m) {
            return Err(invalid("crystal stress: the spinor density is not 2m x 2m"));
        }
        let block = |r: usize, c: usize| full.view((r * m, c * m), (m, m)).into_owned();
        let (aa, bb, ab, ba) = (block(0, 0), block(1, 1), block(0, 1), block(1, 0));
        let i = C64::i();
        let parts = vec![&aa + &bb, &ba + &ab, (&ab - &ba) * i, &aa - &bb];
        Ok(match potential {
            None => vec![(None, parts[0].clone())],
            Some(blocks) if blocks.len() == 1 => {
                vec![(Some(blocks[0].as_slice()), parts[0].clone())]
            }
            Some(blocks) if blocks.len() == 4 => blocks
                .iter()
                .zip(parts)
                .map(|(v, h)| (Some(v.as_slice()), h))
                .collect(),
            Some(_) => {
                return Err(invalid(
                    "crystal stress: one (time-reversal) or four Pauli channels of the potential",
                ))
            }
        })
    }

    /// `[spin][term][direction]` of `Re tr[M_t dM_t(E)]`; see [`super::Strain::matrix_strain`].
    pub(super) fn matrix_terms(
        &self,
        k: usize,
        fields: &StrainFields,
        matrices: &[SpinMatrices],
        terms: &[Term],
        directions: &[Direction],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        self.check_fields(fields)?;
        if matrices.len() != self.layout.spins {
            return Err(invalid("crystal stress: one density per spin channel"));
        }
        let (ctx, shells, kpoint) = (self.ctx, self.inputs.shells, self.kpoint(k)?);
        let mut items: Vec<Item> = Vec::new();
        for spin in 0..self.layout.spins {
            for (index, &term) in terms.iter().enumerate() {
                for (channel, matrix) in self.components(term, spin, fields, matrices)? {
                    let form = match (term, channel) {
                        (Term::Overlap, _) => Form::Overlap,
                        (Term::Kinetic, _) => Form::Kinetic,
                        (Term::ShortRange, _) => Form::ShortRange,
                        (Term::Local, Some(v)) => Form::Local(v),
                        (Term::KineticDensity, Some(v)) => Form::Tau(v),
                        _ => return Err(invalid("crystal stress: a term without its potential")),
                    };
                    items.push(Item {
                        spin,
                        term: index,
                        form,
                        matrix: hermitian(&matrix),
                    });
                }
            }
        }
        let mut out = vec![vec![vec![0.; directions.len()]; terms.len()]; self.layout.spins];
        let strain = StrainSamples::new(shells, self.lattice(), &self.mesh, kpoint);
        // The grid forms and the kinetic forms are streamed over blocks of mesh points: the Bloch
        // sums and the tangent of each direction exist for one block at a time, and the kinetic
        // operator is applied through its real-space ket (one array of the mesh and the basis).
        let grid_items: Vec<usize> = (0..items.len())
            .filter(|&i| {
                matches!(
                    items[i].form,
                    Form::Overlap | Form::Local(_) | Form::Kinetic
                )
            })
            .collect();
        if !grid_items.is_empty() {
            let (n, m) = (self.mesh.len(), self.functions);
            let kinetic_wanted = grid_items
                .iter()
                .any(|&i| matches!(items[i].form, Form::Kinetic));
            let kinetic = if kinetic_wanted {
                Some(kinetic_setup(
                    ctx,
                    self.inputs.grid,
                    &self.mesh,
                    shells,
                    self.lattice(),
                    kpoint,
                    true,
                )?)
            } else {
                None
            };
            // The tangent-independent parts: `tr(E) Re tr[H F]` of the forms with their own `dv`,
            // and the kinetic symbol terms.
            let mut base = vec![0.; items.len()];
            let mut symbol = vec![vec![0.; directions.len()]; items.len()];
            if let Some(KineticSetup {
                moments: Some((t, symbols)),
                ..
            }) = &kinetic
            {
                for &at in &grid_items {
                    if !matches!(items[at].form, Form::Kinetic) {
                        continue;
                    }
                    let h = &items[at].matrix;
                    base[at] = trace_re(h, t);
                    for (d, e) in directions.iter().enumerate() {
                        for (a, row) in e.iter().enumerate() {
                            for (b, &weight) in row.iter().enumerate() {
                                if weight != 0. {
                                    let pair = VOIGT
                                        .iter()
                                        .position(|&(p, q)| (p, q) == (a, b) || (q, p) == (a, b))
                                        .expect("a pair of axes is one of the six");
                                    symbol[at][d] -= weight * trace_re(h, &symbols[pair]);
                                }
                            }
                        }
                    }
                }
            }
            let block = block_len(ctx, m, n, grid_items.len() + 6);
            let mut start = 0;
            while start < n {
                let range = start..(start + block).min(n);
                let nb = range.len();
                let (values, _) = block_samples(
                    ctx,
                    shells,
                    self.lattice(),
                    &self.mesh,
                    kpoint,
                    &range,
                    false,
                )?;
                let _held = ctx.reserve(product(&[m, nb, 16, grid_items.len() + 2])?)?;
                let chi = DMatrixView::from_slice(values.as_slice(), nb, m);
                // `Y = ket H` for each form on this block, `(point, function)`.
                let mut ys: Vec<Matrix> = Vec::with_capacity(grid_items.len());
                for &at in &grid_items {
                    let h = &items[at].matrix;
                    let y = match items[at].form {
                        Form::Kinetic => {
                            let ket = &kinetic.as_ref().expect("formed above").ket;
                            complex_product(
                                ket.view((range.start, 0), (nb, m)),
                                false,
                                h.as_view(),
                            )?
                        }
                        Form::Overlap | Form::Local(_) => {
                            let mut y = complex_product(chi, false, h.as_view())?;
                            for column in y.column_iter_mut() {
                                for (x, value) in column.into_iter().enumerate() {
                                    let potential = match items[at].form {
                                        Form::Local(v) => v[range.start + x],
                                        _ => 1.,
                                    };
                                    *value *= self.mesh.dv * potential;
                                }
                            }
                            // The `dv` form's own Jacobian base `Re tr[H F]`.
                            if matches!(items[at].form, Form::Overlap) {
                                base[at] += dotc(values.as_slice(), y.as_slice()).re;
                            }
                            y
                        }
                        _ => unreachable!("a grid item"),
                    };
                    ys.push(y);
                }
                for (d, e) in directions.iter().enumerate() {
                    let tangent = ctx.allocate::<Complex64>(product(&[m, nb])?)?;
                    let mut tangent = tangent;
                    let all = self.mesh.points();
                    weighted_derivatives_along(
                        ctx,
                        shells,
                        &Sampling {
                            lattice: self.lattice(),
                            region: self.mesh.region(),
                            kpoints: &[kpoint],
                        },
                        [
                            &all[0][range.clone()],
                            &all[1][range.clone()],
                            &all[2][range.clone()],
                        ],
                        e,
                        tangent.as_mut_slice(),
                    )?;
                    for (&at, y) in grid_items.iter().zip(&ys) {
                        out[items[at].spin][items[at].term][d] +=
                            2. * dotc(tangent.as_slice(), y.as_slice()).re;
                    }
                }
                start = range.end;
            }
            for &at in &grid_items {
                for (d, e) in directions.iter().enumerate() {
                    let trace = e[0][0] + e[1][1] + e[2][2];
                    let jacobian = match items[at].form {
                        Form::Local(_) => 0.,
                        _ => trace * base[at],
                    };
                    out[items[at].spin][items[at].term][d] += jacobian + symbol[at][d];
                }
            }
        }
        // The short-range spheres: one sweep for every matrix and direction.
        let sphere_items: Vec<usize> = (0..items.len())
            .filter(|&i| matches!(items[i].form, Form::ShortRange))
            .collect();
        if !sphere_items.is_empty() {
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
            let held: Vec<Matrix> = sphere_items
                .iter()
                .map(|&i| items[i].matrix.clone())
                .collect();
            let tables = self
                .projections
                .short_range(ctx, shells, self.lattice(), &spheres)?;
            let mut values = vec![vec![0.; directions.len()]; held.len()];
            for table in tables {
                for (to, from) in values
                    .iter_mut()
                    .zip(table.strain(kpoint, directions, &held))
                {
                    to.iter_mut().zip(from).for_each(|(t, f)| *t += f);
                }
            }
            for (&at, per_direction) in sphere_items.iter().zip(values) {
                for (d, v) in per_direction.into_iter().enumerate() {
                    out[items[at].spin][items[at].term][d] += v;
                }
            }
        }
        // A meta-GGA's kinetic-density operator, through its matrices.
        for item in items.iter().filter(|i| matches!(i.form, Form::Tau(_))) {
            let Form::Tau(potential) = item.form else {
                continue;
            };
            let values = bloch_values(ctx, shells, self.lattice(), &self.mesh, kpoint)?;
            let tangents = tau_strain(
                ctx,
                self.inputs.grid,
                &self.mesh,
                &strain,
                values.as_slice(),
                kpoint,
                potential,
                directions,
            )?;
            for (d, dm) in tangents.iter().enumerate() {
                out[item.spin][item.term][d] += trace_re(&item.matrix, dm);
            }
        }
        Ok(out)
    }
}
