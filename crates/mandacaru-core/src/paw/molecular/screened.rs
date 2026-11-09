//! The PAW pieces of a screened hybrid on a molecule: the augmentation of the
//! pair densities under `erfc(omega R)/R`, the one-center exchange of the
//! spheres and the frozen short-range semilocal exchange they replace.
//!
//! A pair density of the PAW-LCAO basis is the smooth one on the grid plus the
//! compensation charges `Q g_L Y_LM` plus the one-center remainder `Delta W`.
//! The grid part is screened by the Poisson solver of the hybrid. Here the
//! other two are screened the same way: the shape potential, and with it the
//! grid matrices `W` and the compensation-compensation matrix `U`, is the bare
//! one minus the `erf` part, and `Delta W` is built with the screened radial
//! kernel (see [`crate::paw::screened_atom::screened`]). The semilocal short-range exchange
//! of the reference atom's one-center densities, linearized at its occupations,
//! is the one-body matrix `C V C^T` and the constant `X0 - V . D0`; the hybrid
//! adds `-fraction` times both.
use super::{onecenter::view, *};
use crate::paw::screened_atom::{screened as species_terms, ScreenedAtom};
use crate::{math::storage::with_rows, paw::onecenter_radial::long_range_shape_potential};

/// What a screened hybrid adds to the Kohn-Sham model of a PAW-LCAO molecule,
/// in the Loewdin-orthonormal frame of the integrals it was built from.
pub struct Screened {
    /// The factors of the short-range augmentation: the compensation charges
    /// under `erfc` and the screened one-center tensor. They extend the grid's
    /// screened Cholesky factors as an `Eri::Augmented` correction.
    pub pairs: Arc<PairCorrection>,
    /// How many of `pairs.factors` come first from the one-center tensor
    /// `Delta W^SR`; the rest are the compensation charges.
    pub one_center_factors: usize,
    /// `C V C^T`: the frozen short-range semilocal exchange potential in the basis.
    pub one_body: DMatrix<f64>,
    /// `X0 - V . D0`, summed over the atoms.
    pub constant: f64,
    /// What an open shell adds to the spheres' semilocal exchange.
    pub spin: SpinTerms,
}

/// The spin dependence of the spheres' semilocal short-range exchange: each
/// atom's species tables and its projection on the Loewdin-orthonormal basis.
pub struct SpinTerms {
    atoms: Arc<Vec<ScreenedAtom>>,
    /// `(species, L^T C)` per atom, with `C` the projector coefficients of the basis.
    spheres: Vec<(usize, DMatrix<f64>)>,
    _memory: crate::Reservation,
}
impl SpinTerms {
    /// The polarization energy of the spheres (see
    /// [`crate::paw::screened_polarization`]) and its Fock matrices `C V_s C^T`
    /// for the spin densities `up` and `down` of the orthonormal frame.
    pub fn polarization(
        &self,
        ctx: &Context,
        up: &DMatrix<f64>,
        down: &DMatrix<f64>,
    ) -> Result<(f64, [DMatrix<f64>; 2])> {
        let m = up.nrows();
        let mut energy = 0.;
        let mut fock = [DMatrix::zeros(m, m), DMatrix::zeros(m, m)];
        for (species, c) in &self.spheres {
            let terms = self.atoms[*species].polarization.evaluate(
                ctx,
                &(c.transpose() * up * c),
                &(c.transpose() * down * c),
            )?;
            energy += terms.energy;
            fock[0] += c * terms.up * c.transpose();
            fock[1] += c * terms.down * c.transpose();
        }
        Ok((energy, fock))
    }

    /// The non-collinear polarization energy of the spheres (see
    /// [`crate::paw::screened_polarization`]'s `vector`) and its channel Fock matrices
    /// `C O^a C^T` for the four real Pauli components `pauli` (`D_0 = D_uu + D_dd`, ...) of a
    /// spinor density in the orthonormal frame. A collinear density gives [`Self::polarization`]
    /// with `O^0 +- O^3` its spin matrices.
    pub fn vector_polarization(
        &self,
        ctx: &Context,
        pauli: &[DMatrix<f64>; 4],
    ) -> Result<(f64, [DMatrix<f64>; 4])> {
        let m = pauli[0].nrows();
        let mut energy = 0.;
        let mut fock: [DMatrix<f64>; 4] = std::array::from_fn(|_| DMatrix::zeros(m, m));
        for (species, c) in &self.spheres {
            let projected: [DMatrix<f64>; 4] = std::array::from_fn(|a| c.transpose() * &pauli[a] * c);
            let terms = self.atoms[*species].polarization.evaluate_vector(ctx, &projected)?;
            energy += terms.energy;
            for (f, o) in fock.iter_mut().zip(&terms.operators) {
                *f += c * o * c.transpose();
            }
        }
        Ok((energy, fock))
    }

    /// The spheres' short-range semilocal exchange (all-electron minus smooth,
    /// spherically averaged) at the spin densities `up` and `down`, summed over
    /// the atoms: `(exact, frozen)`, where `exact` evaluates the functional on the
    /// current density and `frozen` is the linearization `X0 + V . (D - D0)` the
    /// hybrid actually uses. Their difference is the frozen model's error.
    pub fn semilocal(
        &self,
        ctx: &Context,
        up: &DMatrix<f64>,
        down: &DMatrix<f64>,
    ) -> Result<(f64, f64)> {
        let (mut exact, mut frozen) = (0., 0.);
        for (species, c) in &self.spheres {
            let atom = &self.atoms[*species];
            let (p_up, p_down) = (c.transpose() * up * c, c.transpose() * down * c);
            exact += atom.polarization.evaluate(ctx, &p_up, &p_down)?.unpolarized;
            frozen += atom.constant + (&atom.v * (&p_up + &p_down)).trace();
        }
        Ok((exact, frozen))
    }
}

/// The shape potential of the compensation multipole `(l)` of radius `radius` at
/// `distances`, under `erfc(omega r)/r`: the bare potential minus the `erf` part.
fn short_range_potential(
    ctx: &Context,
    distances: &[f64],
    radius: f64,
    l: usize,
    omega: f64,
) -> Result<Vec<f64>> {
    let long = long_range_shape_potential(ctx, distances, radius, l, omega)?;
    Ok(distances
        .iter()
        .zip(long)
        .map(|(r, long)| potential(*r, radius, l) - long)
        .collect())
}

impl Setup {
    /// The species-level screened terms at `omega`, built once per value.
    pub(super) fn screened_atoms(&self, ctx: &Context, omega: f64) -> Result<Arc<Vec<ScreenedAtom>>> {
        let mut cache = self
            .screened
            .lock()
            .map_err(|_| invalid("the screened PAW cache is poisoned"))?;
        if let Some((_, atoms)) = cache.iter().find(|(value, _)| *value == omega) {
            return Ok(atoms.clone());
        }
        let atoms = Arc::new(
            self.species
                .iter()
                .zip(&self.species_terms)
                .map(|(species, terms)| {
                    species_terms(ctx, &view(species), &terms.projectors, omega)
                })
                .collect::<Result<Vec<_>>>()?,
        );
        cache.push((omega, atoms.clone()));
        Ok(atoms)
    }

    /// The screened-hybrid terms of this setup at the geometry of `integrals`.
    pub fn screened_exchange(
        &self,
        integrals: &MolecularIntegrals,
        omega: f64,
    ) -> Result<Screened> {
        let ctx = &integrals.context;
        let (geometry, grid, basis) = (&integrals.geometry, &integrals.grid, &integrals.basis);
        let (species, atoms, terms) = (&self.species, &self.atoms, &self.species_terms);
        let m = basis.orbitals.len();
        let g = grid.index.points();
        let screened = self.screened_atoms(ctx, omega)?;
        let c = atoms
            .iter()
            .enumerate()
            .map(|(a, &s)| projection(ctx, basis, &species[s], &terms[s], geometry.positions()[a]))
            .collect::<Result<Vec<_>>>()?;
        let mut multipoles = Vec::new();
        for (a, &s) in atoms.iter().enumerate() {
            for ch in &terms[s].multipoles {
                multipoles.push(GlobalMultipole {
                    atom: a,
                    l: ch.l,
                    m: ch.m,
                    q: &c[a] * &ch.block * c[a].transpose(),
                });
            }
        }
        let nch = multipoles.len();
        let rank: usize = atoms.iter().map(|&s| screened[s].factors.len()).sum();
        let memory = ctx.reserve(
            product(&[rank + 2 * nch + 4, m, m, 8])?
                .checked_add(product(&[rank + 2 * nch, 64])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let _workspace = ctx.reserve(
            product(&[nch, m, m, 8, 5])?
                .checked_add(product(&[nch, nch, 8, 12])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let mut factors = Vec::new();
        let mut weights = Vec::new();
        let mut one_body = DMatrix::zeros(m, m);
        let mut constant = 0.;
        for (a, &s) in atoms.iter().enumerate() {
            let atom = &screened[s];
            for (factor, &weight) in atom.factors.iter().zip(&atom.weights) {
                factors.push(&c[a] * factor * c[a].transpose());
                weights.push(weight);
            }
            one_body += &c[a] * &atom.v * c[a].transpose();
            constant += atom.constant;
        }
        let one_center_factors = factors.len();
        // W: the short-range shape potentials on the grid, projected on the pairs.
        let mut ws = vec![DMatrix::zeros(m, m); nch];
        {
            let mut distances = ctx.allocate::<f64>(g)?;
            let mut offsets = vec![[0.; 3]; g];
            with_rows!(integrals.samples.rows(), |stored| {
                let mut projector = if m >= 16 {
                    Some(crate::math::projection::RealProjector::new(
                        ctx, stored, m, g,
                    )?)
                } else {
                    None
                };
                let mut field = ctx.allocate::<f64>(g)?;
                for (a, ch) in multipoles.iter().enumerate() {
                    let sp = &species[atoms[ch.atom]];
                    let center = geometry.positions()[ch.atom];
                    for (k, offset) in offsets.iter_mut().enumerate() {
                        *offset = std::array::from_fn(|i| grid.point(k)[i] - center[i]);
                        distances.as_mut_slice()[k] = distance(*offset, [0.; 3]);
                    }
                    let values = short_range_potential(
                        ctx,
                        distances.as_slice(),
                        sp.compensation_radius,
                        ch.l,
                        omega,
                    )?;
                    for k in 0..g {
                        let weight = values[k] * harmonic(ch.l, ch.m, offsets[k]) * grid.index.dv();
                        if projector.is_some() {
                            field.as_mut_slice()[k] = weight;
                        } else {
                            for i in 0..m {
                                for j in 0..m {
                                    ws[a][(i, j)] += weight
                                        * stored[i * g + k].widen()
                                        * stored[j * g + k].widen();
                                }
                            }
                        }
                    }
                    if let Some(projector) = projector.as_mut() {
                        ws[a]
                            .as_mut_slice()
                            .copy_from_slice(projector.project(field.as_slice())?);
                    }
                }
            });
        }
        // U: each shape's short-range potential integrated over the partner shapes.
        let mut electrostatic = PointAccumulator::new(ctx, nch, nch)?;
        for (b, &s) in atoms.iter().enumerate() {
            let sp = &species[s];
            let mut points = Vec::new();
            sphere(
                sp.compensation_radius,
                0.,
                50,
                24,
                48,
                |offset, r, weight| {
                    points.push((offset, r, weight));
                    Ok(())
                },
            )?;
            let _scratch = ctx.reserve(product(&[points.len(), nch + 2, 8])?)?;
            let mut potentials = Vec::with_capacity(nch);
            for ci in &multipoles {
                let center = geometry.positions()[ci.atom];
                let deltas: Vec<[f64; 3]> = points
                    .iter()
                    .map(|(offset, _, _)| {
                        std::array::from_fn(|k| geometry.positions()[b][k] + offset[k] - center[k])
                    })
                    .collect();
                let radii: Vec<f64> = deltas.iter().map(|d| distance(*d, [0.; 3])).collect();
                let values = short_range_potential(
                    ctx,
                    &radii,
                    species[atoms[ci.atom]].compensation_radius,
                    ci.l,
                    omega,
                )?;
                potentials.push(
                    values
                        .iter()
                        .zip(&deltas)
                        .map(|(v, d)| v * harmonic(ci.l, ci.m, *d))
                        .collect::<Vec<_>>(),
                );
            }
            let mut left = vec![0.; nch];
            let mut densities = vec![0.; nch];
            for (k, (offset, r, weight)) in points.iter().enumerate() {
                for (slot, values) in left.iter_mut().zip(&potentials) {
                    *slot = values[k];
                }
                densities.fill(0.);
                for (j, cj) in multipoles.iter().enumerate().filter(|(_, ch)| ch.atom == b) {
                    densities[j] = shape(*r, sp.compensation_radius, cj.l)
                        * harmonic(cj.l, cj.m, *offset)
                        * weight;
                }
                electrostatic.push(&left, &densities)?;
            }
        }
        let values = electrostatic.finish()?;
        let u = DMatrix::from_column_slice(nch, nch, values.as_slice());
        drop(values);
        if nch > 0 {
            let coupled = DMatrix::from_fn(2 * nch, 2 * nch, |a, b| {
                if a < nch && b < nch {
                    0.5 * (u[(a, b)] + u[(b, a)])
                } else if a.abs_diff(b) == nch {
                    1.
                } else {
                    0.
                }
            });
            let (values, vectors) = real_eigen(coupled)?;
            for (k, weight) in values.into_iter().enumerate() {
                if weight == 0. {
                    continue;
                }
                let mut factor = DMatrix::zeros(m, m);
                for a in 0..nch {
                    factor += &multipoles[a].q * vectors[(a, k)] + &ws[a] * vectors[(a + nch, k)];
                }
                factors.push(factor);
                weights.push(weight);
            }
        }
        // The Loewdin-orthonormal frame, as the integrals' own corrections.
        let lowdin = &integrals.lowdin;
        let spin_memory = ctx.reserve(
            atoms
                .iter()
                .map(|&s| product(&[m, terms[s].projectors.len(), 8, 2]))
                .sum::<Result<usize>>()?,
        )?;
        let spheres = atoms
            .iter()
            .zip(&c)
            .map(|(&s, c)| (s, lowdin.transpose() * c))
            .collect();
        for factor in &mut factors {
            *factor = lowdin.transpose() * &*factor * lowdin;
        }
        Ok(Screened {
            pairs: Arc::new(PairCorrection {
                factors,
                weights,
                memory,
            }),
            one_center_factors,
            one_body: lowdin.transpose() * one_body * lowdin,
            constant,
            spin: SpinTerms {
                atoms: screened,
                spheres,
                _memory: spin_memory,
            },
        })
    }
}
