//! The Wannier-center flow of a crystal's plane and the Z2 indices built on it: strings along
//! `G_a` at a fixed `k_c`, the transverse coordinate `k_b` bisected where the centers move too far
//! ([`topology::adaptive`]), every string from the direct DFT links of [`strings`].
//!
//! A time-reversal-invariant plane (`k_c` 0 or 1/2) of a time-reversal-invariant crystal is flown
//! over `k_b in [0, 1/2]` only (the centers at `-k_b` are those at `k_b`): its Z2 index is the
//! parity of the crossings ([`topology::z2`]). Any other plane is flown over the whole period and
//! carries its Chern number as the winding ([`topology::winding`]).
//!
//! A spin-degenerate crystal (scalar-relativistic, one unpolarized channel) holds every Wannier
//! center twice, once per spin, so every crossing is counted twice and its Z2 is zero: the index is
//! reported as the zero of that doubled manifold. A collinear spin-polarized crystal has two
//! channels and no time reversal in either, and raises.
use super::{
    invalid, plane_axes, strings,
    topology::{self, Adaptive, Flow, Z2},
    Bands, Manifold, Metal, Sampler,
};
use crate::{Error, Result};

/// Halvings of a transverse interval before a flow gives up.
pub const DEPTH: u32 = 6;

/// The flow of one plane.
#[derive(Clone, Debug)]
pub struct PlaneFlow {
    /// The normal `c` of the plane (`G_c`) and the offset `k_c` (fractional).
    pub normal: usize,
    pub offset: f64,
    /// The string direction `a` and the transverse direction `b`.
    pub direction: usize,
    pub transverse: usize,
    /// Points of each string.
    pub length: usize,
    /// The transverse samples (ascending) and the sorted centers of each: `centers[sample][n]`.
    pub flow: Flow,
    /// The flow stops at `k_b = 1/2` (a time-reversal-invariant plane); else it is a whole period.
    pub half: bool,
    /// The Chern number of a whole-period flow, oriented along `G_a x G_b` of the cyclic in-plane
    /// axes (`plane_axes`), whichever way the strings run.
    pub winding: Option<i64>,
    /// The Z2 index of a half flow.
    pub z2: Option<Z2>,
    /// The centers are the spin-doubled set (every center appears twice).
    pub spin_degenerate: bool,
    /// The half-open band range of the manifold.
    pub bands: (usize, usize),
    /// The smallest singular value of every link used.
    pub minimum: f64,
    /// Strings computed (each one `length` frames).
    pub strings: usize,
}

/// The crystal is time-reversal invariant: a spinor with a time-reversal-invariant density and no
/// magnetic moment, or an unpolarized scalar-relativistic one (spin degenerate by construction).
pub fn time_reversal_invariant(c: &crate::periodic::Crystal) -> bool {
    if c.spinor {
        c.time_reversal
    } else {
        c.spins == 1
    }
}

fn trim(offset: f64) -> bool {
    [0., 0.5].iter().any(|t| (offset - t).abs() < 1e-12)
}

/// The sorted centers of one string and the smallest singular value of its links.
fn string(
    sampler: &Sampler,
    manifold: &Manifold,
    direction: usize,
    length: usize,
    origin: [f64; 3],
) -> Result<(Vec<f64>, f64)> {
    let mut size = [1; 3];
    size[direction] = length;
    let lattice = sampler.lattice(size, origin, manifold)?;
    let set = strings::strings(sampler, &lattice, direction)?;
    Ok((set.centers[0][0].clone(), set.minimum))
}

/// The flow of the plane normal to `G_normal` at `k_normal = offset`: strings of `size[direction]`
/// points along `G_direction`, and `size[transverse]` equal transverse intervals over the period
/// (half of them, rounded up, over the half period of a time-reversal-invariant plane), each
/// bisected to a depth of [`DEPTH`].
pub fn plane_flow(
    sampler: &Sampler,
    normal: usize,
    direction: usize,
    size: [usize; 3],
    offset: f64,
    bands: Bands,
) -> Result<PlaneFlow> {
    let axes = plane_axes(normal)?;
    let transverse = if direction == axes[0] {
        axes[1]
    } else if direction == axes[1] {
        axes[0]
    } else {
        return Err(invalid(format!(
            "a flow in the plane normal to G_{normal} has its strings along G_{} or G_{}; got \
             direction {direction}",
            axes[0], axes[1]
        )));
    };
    let length = size[direction];
    let period = size[transverse];
    if length < 3 {
        return Err(invalid(format!(
            "the strings of a Wannier-center flow need at least 3 points; got {length}"
        )));
    }
    if period < 2 {
        return Err(invalid(format!(
            "a Wannier-center flow needs at least 2 transverse mesh points; the mesh is {size:?}"
        )));
    }
    if !offset.is_finite() || !(0. ..1.).contains(&offset) {
        return Err(invalid(
            "the offset of a plane is a fractional coordinate in [0, 1)",
        ));
    }
    let c = sampler.crystal;
    if c.spins != 1 {
        return Err(Error::Unsupported(
            "the Wannier-center flow of a collinear spin-polarized crystal (two channels) is not \
             implemented: run the spin-degenerate or the spinor calculation"
                .into(),
        ));
    }
    let manifold = sampler.manifold(bands, Metal::Undefined)?;
    let half = time_reversal_invariant(c) && trim(offset) && matches!(bands, Bands::Occupied);
    let end = if half { 0.5 } else { 1. };
    let intervals = if half { period.div_ceil(2) } else { period };
    let label = format!(
        "the plane k_{normal} = {offset} (strings along G_{direction}, k_{transverse} from 0 to {end})"
    );
    let mut origin = [0.; 3];
    origin[normal] = offset;
    let mut minimum = f64::INFINITY;
    let mut count = 0;
    let flow = topology::adaptive(
        &label,
        end,
        Adaptive {
            intervals,
            depth: DEPTH,
        },
        |t| {
            origin[transverse] = t;
            let (centers, smallest) = string(sampler, &manifold, direction, length, origin)?;
            minimum = minimum.min(smallest);
            count += 1;
            Ok(centers)
        },
    )?;
    let spin_degenerate = !c.spinor;
    let (winding, z2) = if half {
        // A spin-degenerate set is its own Kramers doubling: duplicate it.
        let flow = if spin_degenerate {
            Flow {
                k: flow.k.clone(),
                centers: flow
                    .centers
                    .iter()
                    .map(|set| {
                        let mut both: Vec<f64> = set.iter().chain(set).copied().collect();
                        both.sort_by(f64::total_cmp);
                        both
                    })
                    .collect(),
            }
        } else {
            flow.clone()
        };
        (None, Some(topology::z2(&flow)?))
    } else {
        // `winding` is the Chern number of the plane oriented (string, transverse); the plane is
        // oriented along `G_a x G_b` of the cyclic axes, as `get_chern_number`.
        let sign = if direction == axes[0] { 1 } else { -1 };
        (Some(sign * topology::winding(&flow)?), None)
    };
    Ok(PlaneFlow {
        normal,
        offset,
        direction,
        transverse,
        length,
        flow,
        half,
        winding,
        z2,
        spin_degenerate,
        bands: manifold.ranges[0],
        minimum,
        strings: count,
    })
}

/// The Z2 indices of a crystal: the six time-reversal-invariant planes and what they give.
#[derive(Clone, Debug)]
pub struct Indices {
    /// `planes[c][0]` the flow of `k_c = 0`, `planes[c][1]` that of `k_c = 1/2`.
    pub planes: Vec<[PlaneFlow; 2]>,
    /// `nu_0`.
    pub strong: u8,
    /// `nu_1, nu_2, nu_3`: the plane `k_j = 1/2`.
    pub weak: [u8; 3],
}

/// `(nu_0; nu_1 nu_2 nu_3)` from the flows of the six planes. `nu_j` is the Z2 index of the plane
/// `k_j = 1/2` and `nu_0` that of `k_j = 0` plus it, mod 2, which must agree for the three `j`.
/// Strings run along `G_{(c+1) mod 3}` of the mesh `size`.
pub fn indices(sampler: &Sampler, size: [usize; 3]) -> Result<Indices> {
    if !time_reversal_invariant(sampler.crystal) {
        return Err(Error::Unsupported(
            "the Z2 indices need a time-reversal-invariant crystal; a collinear spin-polarized or \
             magnetic spinor crystal has none"
                .into(),
        ));
    }
    let mut planes = Vec::with_capacity(3);
    for c in 0..3 {
        let direction = plane_axes(c)?[0];
        let flows = [0., 0.5]
            .map(|offset| plane_flow(sampler, c, direction, size, offset, Bands::Occupied));
        let [a, b] = flows;
        planes.push([a?, b?]);
    }
    let parity = |flow: &PlaneFlow| flow.z2.as_ref().map_or(0, |z| z.parity);
    let strong: Vec<u8> = planes
        .iter()
        .map(|p| (parity(&p[0]) + parity(&p[1])) % 2)
        .collect();
    if strong.iter().any(|s| *s != strong[0]) {
        return Err(invalid(format!(
            "the strong index nu_0 differs between the three plane pairs ({strong:?}): the planes \
             k_c = 0 and k_c = 1/2 must give the same parity for every c; refine the strings"
        )));
    }
    let weak = [
        parity(&planes[0][1]),
        parity(&planes[1][1]),
        parity(&planes[2][1]),
    ];
    Ok(Indices {
        planes,
        strong: strong[0],
        weak,
    })
}
