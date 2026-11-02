//! The oracle of the dense reciprocal sums under a strain: `couplings` summed over a PINNED list
//! of integer triples (`G = B n`, the list of the reference cell), so that the strained sum
//! differs from the reference only by smooth dependences. `periodic::setup::electrostatics::couplings`
//! re-enumerates `|G| <= cutoff` at whatever cell it is given, which is the negative control.
use mandacaru_core::periodic::setup::{
    electrostatics::{coulomb, ion_transform},
    shapes::{visit_lattice, Shapes, Sites},
};
use nalgebra::Vector3;
use num_complex::Complex64;
use std::f64::consts::PI;

/// The integer triples `n` of `visit_lattice(reciprocal, cutoff, skip_zero)`.
pub fn list(sites: &Sites, cutoff: f64) -> Vec<[i64; 3]> {
    let lattice = sites.lattice;
    let mut out = Vec::new();
    visit_lattice(
        &sites.reciprocal().unwrap(),
        cutoff,
        true,
        4096,
        &mut |vectors| {
            for g in vectors {
                let n = lattice.transpose() * Vector3::from(*g) / (2. * PI);
                out.push([0, 1, 2].map(|a| n[a].round() as i64));
                assert!(
                    (n - n.map(f64::round)).norm() < 1e-8,
                    "a lattice vector off the grid"
                );
            }
            Ok(())
        },
    )
    .unwrap();
    out
}

/// `(U, U_ion, E_ion)` of `couplings` over the triples `list` at the cell of `sites`.
pub struct Sums {
    pub u: Vec<Complex64>,
    pub u_ion: Vec<Complex64>,
    pub e_ion: f64,
}

/// `with_tail` adds the on-site tail beyond `cutoff` to the diagonal of `U`, as `couplings` does.
pub fn sums(
    shapes: &Shapes,
    sites: &Sites,
    list: &[[i64; 3]],
    cutoff: f64,
    with_tail: bool,
) -> Sums {
    let rows = sites.channels.len();
    let b = sites.reciprocal().unwrap();
    let mut u = vec![Complex64::default(); rows * rows];
    let mut u_ion = vec![Complex64::default(); rows];
    let mut e_ion = 0.;
    for chunk in list.chunks(2048) {
        let vectors: Vec<[f64; 3]> = chunk
            .iter()
            .map(|n| (b * Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64)).into())
            .collect();
        let m = vectors.len();
        let mut f = vec![Complex64::default(); rows * m];
        shapes.transforms(sites, &vectors, &mut f, m).unwrap();
        for (g, v) in vectors.iter().enumerate() {
            let (k, ion) = (coulomb(*v), ion_transform(sites, *v));
            e_ion += 0.5 * k * ion.norm_sqr();
            for a in 0..rows {
                let weighted = f[a * m + g].conj() * k;
                for c in 0..rows {
                    u[a * rows + c] += weighted * f[c * m + g];
                }
                u_ion[a] += weighted * ion;
            }
        }
    }
    let volume = sites.volume();
    for value in u.iter_mut().chain(u_ion.iter_mut()) {
        *value /= volume;
    }
    for a in (0..rows).filter(|_| with_tail) {
        let (atom, l, _) = sites.channels[a];
        u[a * rows + a] += shapes
            .tail(sites.atoms[atom].compensation_radius, l, cutoff)
            .unwrap();
    }
    Sums {
        u,
        u_ion,
        e_ion: e_ion / volume,
    }
}

/// `(sums(plus) - sums(minus)) / (2 h)` formed vector by vector: each term of the pinned list is
/// differenced on its own before the sum, so the round-off of the difference follows the terms
/// and not their (much larger) total. That is what allows the step of the shape tables' kink
/// test (`1e-8`: a handful of vectors straddle a node of the 0.01 grid) to be used.
pub fn derivative(shapes: &Shapes, plus: &Sites, minus: &Sites, list: &[[i64; 3]], h: f64) -> Sums {
    let rows = plus.channels.len();
    let mut out = Sums {
        u: vec![Complex64::default(); rows * rows],
        u_ion: vec![Complex64::default(); rows],
        e_ion: 0.,
    };
    for (sites, sign) in [(plus, 1.), (minus, -1.)] {
        let b = sites.reciprocal().unwrap();
        let volume = sites.volume();
        for chunk in list.chunks(2048) {
            let vectors: Vec<[f64; 3]> = chunk
                .iter()
                .map(|n| (b * Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64)).into())
                .collect();
            let m = vectors.len();
            let mut f = vec![Complex64::default(); rows * m];
            shapes.transforms(sites, &vectors, &mut f, m).unwrap();
            for (g, v) in vectors.iter().enumerate() {
                let (k, ion) = (coulomb(*v), ion_transform(sites, *v));
                out.e_ion += sign * 0.5 * k * ion.norm_sqr() / volume;
                for a in 0..rows {
                    let weighted = f[a * m + g].conj() * (k / volume);
                    for c in 0..rows {
                        out.u[a * rows + c] += weighted * f[c * m + g] * sign;
                    }
                    out.u_ion[a] += weighted * ion * sign;
                }
            }
        }
    }
    // The on-site tail is a function of the pinned cutoff radius: no strain dependence.
    let scale = 1. / (2. * h);
    for value in out.u.iter_mut().chain(out.u_ion.iter_mut()) {
        *value *= scale;
    }
    out.e_ion *= scale;
    out
}

/// The vectors `B n` of the triples at the lattice of `sites`.
pub fn vectors(sites: &Sites, list: &[[i64; 3]]) -> Vec<[f64; 3]> {
    let b = sites.reciprocal().unwrap();
    list.iter()
        .map(|n| (b * Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64)).into())
        .collect()
}
