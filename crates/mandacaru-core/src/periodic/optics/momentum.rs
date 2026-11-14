//! The momentum matrix of a crystal's Bloch states in its static LCAO basis, and the
//! diamagnetic band sum it gives (`TDDFT_PLAN.md`, section 13.5.2 and 13.5.3).
//!
//! The velocity-gauge Hamiltonian at fixed basis is `H(k) + A . P(k) + (A^2 / 2) S(k)`, so `P` is
//! the exact derivative of the SCF's own Hamiltonian along a uniform vector potential:
//!
//! - **grid part:** the SCF kinetic matrix is spectral, `T = sum_G conj(u(G)) <q^2>/2 u(G)` with
//!   the alias means of `reciprocal::bloch`; its derivative is
//!   `P_grid = sum_G conj(u(G)) <q> u(G)`, with the same transform and normalization
//!   (`setup::pulay::periodic_parts`);
//! - **one-center part:** `T^dagger p T = p~ + sum |p~_i> Delta p_ij <p~_j|`,
//!   `Delta p = -i (<phi_i|grad|phi_j> - <phi~_i|grad|phi~_j>)`, from the radial-angular routine
//!   the molecule uses (`paw::momentum`), rotated from real to the crystal's complex harmonics
//!   and moved to the SCF's own projector basis (`setup::blocks::to_raw_basis` for raw
//!   projectors). No projector gauge phase appears: the projectors are fixed functions.
//!
//! `P` is Hermitian by construction up to the quadrature; it is Hermitized and the anti-Hermitian
//! residual reported. In the band basis the diamagnetic sum of one spin channel is
//! `N_dia,s = sum_k w_k sum_{v occupied, c empty} 2 abs(e . P_cv)^2 / (e_c - e_v)`; the crystal's
//! is the band degeneracy times the sum over channels. For a frozen-core PAW dataset it exceeds
//! the valence count even in a complete basis, so the basis measure is the distance from the
//! converged-basis `N_dia` of the same dataset, never from `N`.
use super::super::{
    berry::{Bands, Metal},
    bloch_mesh::{MeshIndex, Sampler},
    invalid,
    setup::{blocks::to_raw_basis, pulay::periodic_parts, pulay::Mesh, PeriodicSetup},
    wannier::sampling::Frame,
};
use crate::{
    math::{
        angular::{cartesian, complex},
        fft::Fft3,
        reciprocal,
    },
    paw::{
        momentum::{one_center_gradient, SphereWaves},
        pair_tables::{angular_rule, Projector as Wave},
    },
    quantum::C64,
    Context, Result,
};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// The largest anti-Hermitian part of `P` (relative to its largest element) a crystal accepts.
pub const HERMITIAN_TOLERANCE: f64 = 1e-6;

/// `Delta p` (three `P x P` matrices, Hermitian) in the order and the projector basis of the
/// crystal's projectors.
pub fn one_center_momentum(ctx: &Context, setup: &PeriodicSetup) -> Result<[Matrix; 3]> {
    if setup.spinor {
        return Err(crate::Error::Unsupported(
            "the momentum matrix of a spinor crystal is not implemented".into(),
        ));
    }
    let p = setup.projectors.len();
    let mut out = [
        Matrix::zeros(p, p),
        Matrix::zeros(p, p),
        Matrix::zeros(p, p),
    ];
    let rule = angular_rule(24, 48)?;
    let atoms = setup.atoms_species.len();
    let mut datasets = Vec::with_capacity(atoms);
    for atom in 0..atoms {
        let dataset = setup.datasets[setup.atoms_species[atom]].clone();
        let columns: Vec<usize> = (0..p)
            .filter(|&i| setup.projectors[i].atom == atom)
            .collect();
        // The atom's channels in order of l, each with its partial waves on the dataset grid.
        let knots = dataset.r.as_slice();
        let ls: Vec<u32> = dataset.channels.keys().copied().collect();
        let mut tables = Vec::with_capacity(ls.len());
        for l in &ls {
            let channel = &dataset.channels[l];
            let ae: Vec<Vec<f64>> = channel
                .ae_waves
                .iter()
                .map(|w| w.as_slice().to_vec())
                .collect();
            let ps: Vec<Vec<f64>> = channel
                .pseudo_waves
                .iter()
                .map(|w| w.as_slice().to_vec())
                .collect();
            if ae.len() != ps.len() {
                return Err(invalid("a PAW channel has unequal partial-wave counts"));
            }
            tables.push((*l as usize, channel.r_cut, ae, ps));
        }
        let spheres: Vec<SphereWaves<'_>> = tables
            .iter()
            .map(|(_, radius, ae, ps)| SphereWaves {
                radius: *radius,
                ae,
                ps,
            })
            .collect();
        // The real-harmonic projector list: channel, then m, then the radial index.
        let mut waves = Vec::new();
        for (channel, (l, _, ae, _)) in tables.iter().enumerate() {
            for m in -(*l as i32)..=*l as i32 {
                for radial in 0..ae.len() {
                    waves.push(Wave {
                        channel,
                        radial,
                        l: *l,
                        m,
                    });
                }
            }
        }
        let real = one_center_gradient(ctx, knots, &spheres, &waves)?;
        let position = |l: usize, m: i32, radial: usize| -> Result<usize> {
            waves
                .iter()
                .position(|w| w.l == l && w.m == m && w.radial == radial)
                .ok_or_else(|| invalid("a crystal projector has no partial wave"))
        };
        // <Y^c_a|O|Y^c_b> = sum_rs conj(U_ar) U_bs <S_r|O|S_s>, U_ar = int Y^c_a S_r dOmega (the
        // complex harmonics in the real ones, by the quadrature that is exact at these degrees).
        let rotation = |l: usize, mc: i32, mr: i32| -> C64 {
            rule.iter()
                .map(|(direction, weight)| {
                    complex(l, mc, *direction) * (cartesian(l, mr, *direction) * weight)
                })
                .sum()
        };
        for &a in &columns {
            let pa = &setup.projectors[a];
            for &b in &columns {
                let pb = &setup.projectors[b];
                if pa.l.abs_diff(pb.l) != 1 {
                    continue;
                }
                for (axis, block) in out.iter_mut().enumerate() {
                    let mut value = C64::new(0., 0.);
                    for r in -(pa.l as i32)..=pa.l as i32 {
                        let ua = rotation(pa.l, pa.m, r);
                        if ua.norm() < 1e-14 {
                            continue;
                        }
                        let i = position(pa.l, r, pa.index)?;
                        for s in -(pb.l as i32)..=pb.l as i32 {
                            let ub = rotation(pb.l, pb.m, s);
                            if ub.norm() < 1e-14 {
                                continue;
                            }
                            let j = position(pb.l, s, pb.index)?;
                            value += ua.conj() * ub * real[axis][(i, j)];
                        }
                    }
                    // Delta p = -i G.
                    block[(a, b)] = value * C64::new(0., -1.);
                }
            }
        }
        datasets.push(dataset);
    }
    if !setup.dual {
        to_raw_basis(out.iter_mut(), &setup.projectors, &datasets)?;
    }
    Ok(out)
}

/// The momentum of the bands of one frame: `bands x bands` matrices along the three Cartesian
/// axes, Hermitized, and the anti-Hermitian residual before (relative to the largest element).
pub struct BandMomenta {
    pub matrices: [Matrix; 3],
    pub residual: f64,
}

/// [`BandMomenta`] of `frame` (every band it holds, one component) at the Cartesian `k`.
pub fn band_momenta(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: &Mesh,
    frame: &Frame,
    k: [f64; 3],
    one_center: &[Matrix; 3],
) -> Result<BandMomenta> {
    if frame.components != 1 {
        return Err(crate::Error::Unsupported(
            "the momentum matrix of a spinor crystal is not implemented".into(),
        ));
    }
    let n = mesh.len();
    let bands = frame.projections.nrows();
    let samples = frame
        .samples
        .read_range(ctx, 0, crate::product(&[bands, n])?)?;
    let mut fft = Fft3::new(ctx, setup.grid.index.shape())?;
    let (parts, _held) = periodic_parts(ctx, &mut fft, mesh, k, &samples, bands)?;
    let symbols = reciprocal::bloch(ctx, &setup.grid, k)?;
    let mut matrices = [
        Matrix::zeros(bands, bands),
        Matrix::zeros(bands, bands),
        Matrix::zeros(bands, bands),
    ];
    let mut largest = 0f64;
    let mut defect = 0f64;
    for (axis, out) in matrices.iter_mut().enumerate() {
        let weights = &symbols.vectors.as_slice()[axis * n..(axis + 1) * n];
        let mut weighted = parts.clone();
        for mut column in weighted.column_iter_mut() {
            for (value, w) in column.iter_mut().zip(weights) {
                *value *= *w / mesh.volume;
            }
        }
        let grid = parts.adjoint() * weighted;
        let sphere = &frame.projections * &one_center[axis] * frame.projections.adjoint();
        let total = grid + sphere;
        largest = largest.max(total.iter().map(|v| v.norm()).fold(0., f64::max));
        defect = defect.max(
            (&total - total.adjoint())
                .iter()
                .map(|v| v.norm())
                .fold(0., f64::max),
        );
        *out = (&total + total.adjoint()) * C64::new(0.5, 0.);
    }
    let residual = if largest > 0. { defect / largest } else { 0. };
    if residual > HERMITIAN_TOLERANCE {
        return Err(crate::Error::Kernel(format!(
            "the momentum matrix is not Hermitian: anti-Hermitian part {residual:.3e} of its \
             largest element (bound {HERMITIAN_TOLERANCE:.0e})"
        )));
    }
    Ok(BandMomenta { matrices, residual })
}

/// The diamagnetic band sum of a crystal along a direction (module documentation).
#[derive(Clone, Debug)]
pub struct Diamagnetic {
    /// `N_dia,s` of each spin channel.
    pub channels: Vec<f64>,
    /// The band degeneracy times the sum over channels.
    pub total: f64,
    /// The valence electrons of the cell.
    pub electrons: f64,
    /// The largest anti-Hermitian residual of `P` over the mesh.
    pub residual: f64,
}

/// Refuses a mesh that is not closed under `k -> -k` (to 1e-10 in fractional coordinates): its
/// ground state would carry a current.
pub fn check_inversion_closed(mesh: &MeshIndex) -> Result<()> {
    let wrap = |x: f64| x - x.round();
    for i in 0..mesh.points() {
        let k = mesh.point(i);
        let closed = (0..mesh.points()).any(|j| {
            let q = mesh.point(j);
            (0..3).all(|a| wrap(k[a] + q[a]).abs() < 1e-10)
        });
        if !closed {
            return Err(invalid(format!(
                "the k mesh is not closed under k -> -k (no partner of {k:?}); its ground state \
                 would carry a current"
            )));
        }
    }
    Ok(())
}

/// Every occupied-to-empty transition of the insulator `sampler` holds, on `mesh` (whole, closed
/// under `-k`, uniform weights), along the unit `direction`: per transition its spin channel, its
/// gap `e_c - e_v` and its weight `w_k abs(e . P_cv)^2` (no band degeneracy).
#[derive(Clone, Debug)]
pub struct Transitions {
    pub channels: Vec<usize>,
    pub gaps: Vec<f64>,
    pub weights: Vec<f64>,
    /// Electrons one band of a channel holds (2 unpolarized, 1 otherwise).
    pub degeneracy: f64,
    pub electrons: f64,
    /// The cell volume, Bohr^3.
    pub volume: f64,
    /// The largest anti-Hermitian residual of `P` over the mesh.
    pub residual: f64,
}

/// [`Transitions`] of `sampler` on `mesh` along `direction`.
pub fn transitions(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    direction: [f64; 3],
) -> Result<Transitions> {
    check_inversion_closed(mesh)?;
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0. {
        return Err(invalid("the direction must be a finite nonzero vector"));
    }
    let e = direction.map(|v| v / norm);
    let crystal = sampler.crystal;
    let ctx = &crystal.context;
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined)?;
    let one_center = one_center_momentum(ctx, sampler.setup)?;
    let grid_mesh = Mesh::new(ctx, &sampler.setup.grid)?;
    let all: Vec<Vec<usize>> = (0..sampler.channels())
        .map(|_| (0..crystal.bands()).collect())
        .collect();
    let weight = 1. / mesh.points() as f64;
    let mut out = Transitions {
        channels: vec![],
        gaps: vec![],
        weights: vec![],
        degeneracy: sampler.degeneracy(),
        electrons: crystal.electrons,
        volume: sampler.lattice.determinant().abs(),
        residual: 0.,
    };
    let mut point = vec![0usize; sampler.channels()];
    sampler.frames(mesh, &all, |s, frame| {
        let fractional = mesh.point(point[s]);
        point[s] += 1;
        let k: [f64; 3] = (sampler.reciprocal * nalgebra::Vector3::from(fractional)).into();
        let momenta = band_momenta(ctx, sampler.setup, &grid_mesh, frame, k, &one_center)?;
        out.residual = out.residual.max(momenta.residual);
        let along = &momenta.matrices[0] * C64::new(e[0], 0.)
            + &momenta.matrices[1] * C64::new(e[1], 0.)
            + &momenta.matrices[2] * C64::new(e[2], 0.);
        let (lo, hi) = manifold.ranges[s];
        let energies = frame.energies.as_slice();
        for v in lo..hi {
            for c in hi..energies.len() {
                let gap = energies[c] - energies[v];
                if gap <= 0. {
                    return Err(invalid("an empty band lies below an occupied one"));
                }
                out.channels.push(s);
                out.gaps.push(gap);
                out.weights.push(weight * along[(c, v)].norm_sqr());
            }
        }
        Ok(())
    })?;
    Ok(out)
}

impl Transitions {
    /// The diamagnetic band sum (module documentation).
    pub fn diamagnetic(&self, channels: usize) -> Diamagnetic {
        let mut per = vec![0f64; channels];
        for ((s, gap), w) in self.channels.iter().zip(&self.gaps).zip(&self.weights) {
            per[*s] += 2. * w / gap;
        }
        Diamagnetic {
            total: self.degeneracy * per.iter().sum::<f64>(),
            channels: per,
            electrons: self.electrons,
            residual: self.residual,
        }
    }

    /// The independent-particle `epsilon_2(omega) = (4 pi^2 / (Omega omega^2)) g sum w
    /// delta(omega - gap)` on `frequencies` (Hartree, positive), each line a unit-area Gaussian of
    /// standard deviation `width` (`models::spectrum::line_spectrum`). Its first moment is the
    /// band sum: `int omega epsilon_2 d omega = 2 pi^2 N_dia / Omega`.
    pub fn independent_epsilon2(
        &self,
        ctx: &Context,
        frequencies: &[f64],
        width: f64,
    ) -> Result<Vec<f64>> {
        if frequencies.iter().any(|w| !(w.is_finite() && *w > 0.)) {
            return Err(invalid("epsilon_2 needs positive finite frequencies"));
        }
        let strengths: Vec<f64> = self.weights.iter().map(|w| self.degeneracy * w).collect();
        let lines = crate::models::spectrum::line_spectrum(
            ctx,
            &self.gaps,
            &strengths,
            frequencies,
            width,
            crate::models::spectrum::LineShape::Gaussian,
        )?;
        let prefactor = 4. * std::f64::consts::PI.powi(2) / self.volume;
        Ok(lines
            .as_slice()
            .iter()
            .zip(frequencies)
            .map(|(s, w)| prefactor * s / (w * w))
            .collect())
    }
}

/// The diamagnetic band sum of the insulator `sampler` holds, along the unit `direction`, on
/// `mesh` (whole, closed under `-k`, uniform weights).
pub fn diamagnetic_sum(
    sampler: &Sampler<'_>,
    mesh: &MeshIndex,
    direction: [f64; 3],
) -> Result<Diamagnetic> {
    Ok(transitions(sampler, mesh, direction)?.diamagnetic(sampler.channels()))
}
