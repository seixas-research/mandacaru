//! The PAW terms of the nuclear gradient of a general fixed density: a one-body density `P`, an
//! energy-weighted density `W` (the overlap derivative `-tr[W dS]`) and a two-body density
//! written as Coulomb forms `1/2 lambda_k B(X_k, X_k)` of symmetric atomic-orbital matrices
//! `X_k = Phi x_k Phi^T`, all in the raw (non-orthogonal) basis and in the Hartree-Fock model.
//! `B` is the Coulomb bilinear form of the model's two-body integrals: the grid Hartree term,
//! the compensation moments `q.w + 1/2 q U q` and the one-center factors. Its exchange lives in
//! the two-body density, so no exchange term exists here. A Hartree-Fock determinant is the
//! special case `P = D`, `W = sum f e a a^T`, forms of `D (x) D - exchange`
//! (`tests/variational_forces.rs`).
//!
//! The derivative is linear in every density, so each term is one contraction over all forms:
//! the projection matrices `Gamma = 2 sum_k lambda_k X_k C M_k`, the electrostatic weights
//! `sum_k lambda_k q_k q_k^T / 2` and the compensation-center weights `sum_k lambda_k q_k rho_k`.
//! The grid parts (the basis functions under the Hartree and compensation potentials) are
//! contracted by the caller, which receives each form's compensation potential.
use super::{
    compensation::{self, Channel},
    projections, spheres, Contribution, Occupied,
};
use crate::paw::molecular::Setup;
use crate::{molecular::MolecularIntegrals, product, Buffer, Result};
use nalgebra::{DMatrix, DVector};

fn sym(m: &DMatrix<f64>) -> DMatrix<f64> {
    (m + m.transpose()) * 0.5
}

/// The two-body density as Coulomb forms over one orbital frame.
pub struct Forms<'a> {
    /// `Phi`, `m x r`: the orbitals the forms are written in, raw-AO coefficients.
    pub frame: &'a DMatrix<f64>,
    /// `(lambda_k, x_k)`, `x_k` symmetric `r x r`.
    pub terms: &'a [(f64, DMatrix<f64>)],
    /// `rho_k` on the grid, `terms.len() x n`.
    pub rho: &'a [f64],
}

/// The PAW contributions and, per form, the compensation potential `sum_A q_k[A] phi_A` on the
/// grid (`terms.len() x n`), which the caller adds to the Hartree potential of `rho_k`.
pub struct DensityTerms {
    pub contributions: Vec<Contribution>,
    pub compensation: Buffer<f64>,
}

impl Setup {
    /// The PAW part of the fixed-density gradient. `one` are the pseudo-orbitals of `P`
    /// (`weight` their occupations) and `rho_one` its grid density; `weighted` is `W`.
    pub fn density_terms(
        &self,
        integrals: &MolecularIntegrals,
        one: &[Occupied],
        rho_one: &[f64],
        weighted: &DMatrix<f64>,
        forms: &Forms<'_>,
    ) -> Result<DensityTerms> {
        let ctx = &integrals.context;
        let n = integrals.grid.index.points();
        let m = integrals.m();
        let dv = integrals.grid.index.dv();
        let nforms = forms.terms.len();
        let r = forms.frame.ncols();
        if rho_one.len() != n
            || forms.rho.len() != nforms * n
            || forms.frame.nrows() != m
            || weighted.shape() != (m, m)
            || forms
                .terms
                .iter()
                .any(|(l, x)| x.shape() != (r, r) || !l.is_finite())
            || one.iter().any(|o| o.ao.len() != m)
        {
            return Err(crate::Error::Invalid(
                "the fixed densities must match the basis, the grid and the form frame".into(),
            ));
        }
        let channels: Vec<Channel<'_>> = compensation::channels(self);
        let nch = channels.len();
        let atoms = integrals.geometry.numbers().len();
        let _memory = ctx.reserve(product(&[nch.max(1), nforms.max(1), 8, 4])?)?;
        let one_matrix = one.iter().fold(DMatrix::<f64>::zeros(m, m), |acc, o| {
            let a = DVector::from_column_slice(&o.ao);
            acc + &a * a.transpose() * o.weight
        });
        let (matrices, _) = projections::project(self, integrals, &[])?;
        // The compensation fields (nch x n), the grid moments w_k and the charges q_k.
        let mut fields = ctx.allocate::<f64>(product(&[nch.max(1), n])?)?;
        for (a, channel) in channels.iter().enumerate() {
            compensation::grid_field(
                integrals,
                channel,
                &mut fields.as_mut_slice()[a * n..(a + 1) * n],
            );
        }
        let cached = integrals
            .corrections
            .as_ref()
            .and_then(|c| Some((c.electrostatic.as_ref()?, c.ionic.as_deref()?)));
        let electrostatics = compensation::moments(
            self,
            integrals,
            &channels,
            cached,
            &[],
            &[],
            &[],
            rho_one,
            false,
        )?;
        let usym = &electrostatics.usym;
        let ionic = &electrostatics.ionic;
        let forms_ao: Vec<DMatrix<f64>> = forms
            .terms
            .iter()
            .map(|(_, x)| forms.frame * sym(x) * forms.frame.transpose())
            .collect();
        let charge = |x: &DMatrix<f64>, a: usize| {
            let c = &matrices[channels[a].atom];
            (sym(channels[a].block).component_mul(&(c.transpose() * x * c))).sum()
        };
        let mut q = vec![vec![0.; nch]; nforms];
        let mut w = vec![vec![0.; nch]; nforms];
        for k in 0..nforms {
            let rho = &forms.rho[k * n..(k + 1) * n];
            for a in 0..nch {
                q[k][a] = charge(&forms_ao[k], a);
                w[k][a] = dv
                    * rho
                        .iter()
                        .zip(&fields.as_slice()[a * n..(a + 1) * n])
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
            }
        }
        let q_one: Vec<f64> = (0..nch).map(|a| charge(&one_matrix, a)).collect();
        // The projection matrices of every term, per atom (m x p).
        let samples = projections::derivative_samples(self, integrals)?;
        let unit: Vec<Occupied> = (0..m)
            .map(|i| Occupied {
                spin: 0,
                weight: 1.,
                energy: 0.,
                w_weight: 0.,
                ao: (0..m).map(|j| if i == j { 1. } else { 0. }).collect(),
            })
            .collect();
        let rows = |gamma: &[DMatrix<f64>]| -> Vec<Vec<DVector<f64>>> {
            gamma
                .iter()
                .map(|g| (0..m).map(|i| g.row(i).transpose()).collect())
                .collect()
        };
        let (mut potential, mut overlap, mut coulomb) = (Vec::new(), Vec::new(), Vec::new());
        for (atom, &s) in self.atoms.iter().enumerate() {
            let terms = &self.species_terms[s];
            let c = &matrices[atom];
            let own: Vec<usize> = (0..nch).filter(|&a| channels[a].atom == atom).collect();
            let mut hp = sym(&terms.coupling);
            for &a in &own {
                hp += sym(channels[a].block) * ionic[a];
            }
            potential.push(&one_matrix * c * hp * 2.);
            overlap.push(weighted * c * sym(&terms.overlap) * -2.);
            let mut gamma = DMatrix::<f64>::zeros(m, c.ncols());
            for k in 0..nforms {
                let (lambda, _) = forms.terms[k];
                let projected = c.transpose() * &forms_ao[k] * c;
                let mut coupling = DMatrix::<f64>::zeros(c.ncols(), c.ncols());
                for &a in &own {
                    let coefficient =
                        w[k][a] + (0..nch).map(|b| usym[(a, b)] * q[k][b]).sum::<f64>();
                    coupling += sym(channels[a].block) * coefficient;
                }
                for (f, &weight) in terms.factors.iter().zip(&terms.weights) {
                    let moment = (f.component_mul(&projected)).sum();
                    coupling += sym(f) * (weight * moment);
                }
                gamma += &forms_ao[k] * c * coupling * (2. * lambda);
            }
            coulomb.push(gamma);
        }
        let mut out = vec![
            Contribution {
                name: "projector_potential/projection",
                gradient: projections::force(integrals, &unit, &samples, &rows(&potential)),
            },
            Contribution {
                name: "pulay_projector/projection",
                gradient: projections::force(integrals, &unit, &samples, &rows(&overlap)),
            },
            Contribution {
                name: "two_body/projection",
                gradient: projections::force(integrals, &unit, &samples, &rows(&coulomb)),
            },
            Contribution {
                name: "projector_potential/local_sphere",
                gradient: spheres::local_force(self, integrals, one)?,
            },
            Contribution {
                name: "projector_potential/ionic",
                gradient: compensation::ionic_force(self, integrals, &channels, &q_one)?,
            },
            Contribution {
                name: "external/center",
                gradient: spheres::external_force(self, integrals, rho_one),
            },
        ];
        let weights = DMatrix::from_fn(nch, nch, |a, b| {
            (0..nforms)
                .map(|k| 0.5 * forms.terms[k].0 * q[k][a] * q[k][b])
                .sum::<f64>()
        });
        out.push(Contribution {
            name: "two_body/electrostatic",
            gradient: compensation::electrostatic_force(self, integrals, &channels, &weights)?,
        });
        let mut center = ctx.allocate::<f64>(product(&[nch.max(1), n])?)?;
        for k in 0..nforms {
            let (lambda, rho) = (forms.terms[k].0, &forms.rho[k * n..(k + 1) * n]);
            for a in 0..nch {
                let scale = lambda * q[k][a];
                for (o, x) in center.as_mut_slice()[a * n..(a + 1) * n]
                    .iter_mut()
                    .zip(rho)
                {
                    *o += scale * x;
                }
            }
        }
        let center_slice = center.as_slice();
        out.push(Contribution {
            name: "two_body/compensation_center",
            gradient: compensation::field_force(integrals, &channels, |a, k| {
                center_slice[a * n + k]
            })?,
        });
        drop(center);
        let mut compensation_potential = ctx.allocate::<f64>(product(&[nforms.max(1), n])?)?;
        for k in 0..nforms {
            let target = &mut compensation_potential.as_mut_slice()[k * n..(k + 1) * n];
            for a in 0..nch {
                let scale = q[k][a];
                for (o, f) in target
                    .iter_mut()
                    .zip(&fields.as_slice()[a * n..(a + 1) * n])
                {
                    *o += scale * f;
                }
            }
        }
        debug_assert!(out.iter().all(|c| c.gradient.len() == 3 * atoms));
        Ok(DensityTerms {
            contributions: out,
            compensation: compensation_potential,
        })
    }
}
