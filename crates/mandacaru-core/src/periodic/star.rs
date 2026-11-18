//! The whole k mesh a space-group-reduced mesh stands for, as the exact exchange needs it (SH-5b, SH-5c).
//!
//! A hybrid's exchange sums over the occupied states of EVERY mesh point, but a mesh reduced by the
//! operations `{(W_s, t_s)}` (`x -> W x + t` in fractional coordinates) lists one point `k` per
//! star. The states of the other points are those of `k`, moved by an operation `s` (and, for a
//! point reached through `-k`, by time reversal):
//!
//! ```text
//! psi_{R_s k}(r)  = psi_k(R_s^-1 (r - t_s))            u_{R_s k}[node(n)] = e^{-i k . R_s^-1 t_s} u_k[n]
//! <p_{b,i,l,m} | psi_{R_s k}> = e^{-i k . R'} sum_{m'} conj(T^l_{m' m}) <p_{a,i,l,m'} | psi_k>
//! ```
//!
//! `u` is the periodic part on the grid and `node(n)` the node the operation sends `n` to (the
//! permutation the SCF symmetrizes the density with); `b = s(a)` is the atom the operation sends `a`
//! to, `R' = R_s^T (R_s tau_a + t_s - tau_b)` the (lattice) vector the image sphere sits away by, and
//! `T^l` the complex-harmonic rotation of the operation (`setup::symmetry::rotation_tables`,
//! `Y_lm(R u) = sum_m' Y_lm'(u) T_{m' m}`). A time-reversed image is the conjugate of the rotated
//! one (`Augmentation::reversed` for the projections). The states of a star are those of its listed
//! point rotated: nothing of the basis at `R_s k` is needed.
//!
//! A spinor state (a time-reversal-symmetric spin-orbit crystal) also turns its spin: `psi'_s =
//! sum_t U_st psi_t` with `U` the SU(2) matrix of the operation's proper part (an improper operation
//! acts on the spin as `-R` does, the inversion trivially), on the grid and on the projections of
//! each component; the time-reversed image is `(up, down) -> (down^*, -up^*)` of the rotated one.
//!
//! The listed weights are the sizes of the stars over the mesh: every point of a star weighs the
//! same, so the weight of an image is `w / |star|`, and the weight per whole-mesh point must be the
//! same for every listed point (a mesh that is not the group's reduction is refused).
use super::{invalid, screened_augmentation::Augmentation, Crystal};
use crate::{periodic::setup::symmetry::rotation_tables, quantum::C64, Buffer, Result};
use nalgebra::{DMatrix, Matrix3, Vector3};

type Matrix = DMatrix<C64>;

/// Two fractional coordinates are the same point of the mesh to this distance, modulo 1.
const SAME_POINT: f64 = 1e-7;

/// One point of a star: reached from the listed point by the operation `op` (an index into the
/// crystal's permutations), then by time reversal when `reversed`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Image {
    pub op: usize,
    pub reversed: bool,
    /// Its Cartesian wave vector `R_op k` (`-R_op k` when reversed): a representative, not
    /// reduced to the first zone.
    pub k: [f64; 3],
}

/// The images of every listed point, and what it takes to rotate a state to them.
pub(super) struct Reduction {
    /// Per listed point, its images (the identity first).
    pub stars: Vec<Vec<Image>>,
    cartesian: Vec<Matrix3<f64>>,
    /// `t_s` in Cartesian coordinates (Bohr).
    translations: Vec<Vector3<f64>>,
    atom_maps: Vec<Vec<usize>>,
    /// Per operation and atom, the lattice vector `R_s tau_a + t_s - tau_b` (Bohr, exactly a
    /// lattice vector).
    shifts: Vec<Vec<Vector3<f64>>>,
    /// Per operation, `T^l` for `l = 0..=max_l`.
    tables: Vec<Vec<Matrix>>,
    /// Per operation, the SU(2) matrix of its rotation (`psi'_s = sum_t U_st psi_t`).
    spin: Vec<[[C64; 2]; 2]>,
    /// Row of `(atom, l, m, radial)` in the crystal's projector order.
    projectors: Vec<(usize, usize, i32, usize)>,
}

/// Whether the mesh of `c` stands for more points than it lists through space-group operations
/// (more than the identity: a trivial group is the time-reversal halving alone).
pub(super) fn reduced(c: &Crystal) -> bool {
    c.permutations.len() > 1
}

impl Reduction {
    pub(super) fn new(c: &Crystal) -> Result<Self> {
        let description = c.description.as_ref().ok_or_else(|| {
            invalid(
                "a hybrid crystal on a k mesh reduced by space-group symmetry needs the \
                 symmetry operations of the snapshot's setup block",
            )
        })?;
        let operations = description.operations.as_ref().ok_or_else(|| {
            invalid("the setup block lists no symmetry operations for the reduced mesh")
        })?;
        // The stars include the time-reversed image of every operation when nonmagnetic
        // (`anti_unitary = None`). A magnetic spin-orbit crystal needs the explicit per-operation
        // unitary/anti-unitary flags from its magnetic little group.
        let anti_unitary = if c.spinor && !c.time_reversal {
            let Some(flags) = operations.anti_unitary.as_ref() else {
                return Err(crate::Error::Unsupported(
                    "the stars of a reduced k mesh include the time-reversed image of every operation, \
                     and a magnetic spin-orbit crystal is not time-reversal symmetric (its states at \
                     -k are not the partners of those at k): use symmetry=False"
                        .into(),
                ));
            };
            if flags.len() != operations.rotations.len() {
                return Err(invalid(
                    "the setup block's magnetic anti_unitary flags must match its operations",
                ));
            }
            Some(flags.as_slice())
        } else {
            None
        };
        if operations.rotations.len() != c.permutations.len() {
            return Err(invalid(
                "the setup block's operations are not the snapshot's node permutations",
            ));
        }
        if c.grid.origin.iter().any(|x| x.abs() > 1e-12) {
            return Err(invalid(
                "the rotation of states needs the grid to start at the cell origin",
            ));
        }
        let shape = c.grid.index.shape();
        let lattice =
            c.grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
        let inverse = lattice
            .try_inverse()
            .ok_or_else(|| invalid("singular lattice"))?;
        let max_l = description
            .projectors
            .iter()
            .map(|p| p.1)
            .max()
            .unwrap_or(0);
        let centers: Vec<Vector3<f64>> = description.centers.iter().map(|&x| x.into()).collect();
        let (mut cartesian, mut translations, mut shifts, mut tables, mut spin) =
            (vec![], vec![], vec![], vec![], vec![]);
        for (s, w) in operations.rotations.iter().enumerate() {
            let w = Matrix3::from_fn(|i, j| f64::from(w[3 * i + j]));
            let r = lattice * w * inverse;
            let t = lattice * Vector3::from(operations.translations[s]);
            check_node_permutation(c, s, &w, operations.translations[s])?;
            let map = &operations.atom_maps[s];
            if map.len() != centers.len() {
                return Err(invalid("an atom map must name every atom"));
            }
            let mut moved = Vec::with_capacity(centers.len());
            for (a, &b) in map.iter().enumerate() {
                let miss = r * centers[a] + t
                    - centers
                        .get(b)
                        .copied()
                        .ok_or_else(|| invalid("an atom map names an atom the crystal lacks"))?;
                // Exactly a lattice vector: the integers `n` of `miss = L n`.
                let n = (inverse * miss).map(f64::round);
                if (miss - lattice * n).norm() > 1e-6 {
                    return Err(invalid(
                        "an operation does not send an atom onto its image in the atom map",
                    ));
                }
                moved.push(lattice * n);
            }
            tables.push(rotation_tables(&r, max_l, 1e-6)?);
            spin.push(su2(&r));
            cartesian.push(r);
            translations.push(t);
            shifts.push(moved);
        }
        let mut reduction = Self {
            stars: Vec::new(),
            cartesian,
            translations,
            atom_maps: operations.atom_maps.clone(),
            shifts,
            tables,
            spin,
            projectors: description.projectors.clone(),
        };
        reduction.stars = reduction.build_stars(c, &lattice, anti_unitary)?;
        Ok(reduction)
    }

    fn build_stars(
        &self,
        c: &Crystal,
        lattice: &Matrix3<f64>,
        anti_unitary: Option<&[bool]>,
    ) -> Result<Vec<Vec<Image>>> {
        let fractional = |k: &Vector3<f64>| lattice.transpose() * k / (2. * std::f64::consts::PI);
        let same = |a: &Vector3<f64>, b: &Vector3<f64>| {
            (0..3).all(|i| {
                let d = a[i] - b[i];
                (d - d.round()).abs() < SAME_POINT
            })
        };
        let mut stars = Vec::with_capacity(c.points.len());
        let mut per_point = Vec::with_capacity(c.points.len());
        for point in &c.points {
            let k = Vector3::from(point.k);
            let mut images: Vec<Image> = Vec::new();
            let mut seen: Vec<Vector3<f64>> = Vec::new();
            for (op, r) in self.cartesian.iter().enumerate() {
                let reversals: &[bool] = match anti_unitary {
                    Some(flags) => {
                        if flags[op] {
                            &[true]
                        } else {
                            &[false]
                        }
                    }
                    None => &[false, true],
                };
                for &reversed in reversals {
                    let moved = r * k * if reversed { -1. } else { 1. };
                    let f = fractional(&moved);
                    if seen.iter().any(|g| same(g, &f)) {
                        continue;
                    }
                    seen.push(f);
                    images.push(Image {
                        op,
                        reversed,
                        k: moved.into(),
                    });
                }
            }
            per_point.push(point.weight / images.len() as f64);
            stars.push(images);
        }
        let first = per_point.first().copied().unwrap_or(0.);
        if per_point
            .iter()
            .any(|w| (w - first).abs() > 1e-10 * first.abs().max(1e-300))
        {
            return Err(crate::Error::Unsupported(
                "a hybrid functional on a reduced k mesh needs every listed point to weigh the \
                 size of its star under the kept operations and time reversal; these weights do \
                 not (the mesh is not that group's reduction)"
                    .into(),
            ));
        }
        Ok(stars)
    }

    /// The weight of one image of listed point `q`.
    pub(super) fn weight(&self, c: &Crystal, q: usize) -> f64 {
        c.points[q].weight / self.stars[q].len() as f64
    }

    /// The states of the image, from the periodic parts `input` (one slice per spin component) of
    /// a state at the listed point with Cartesian wave vector `k`:
    /// `u_{R_s k}[node(n)] = e^{-i k . R_s^-1 t_s} sum_t U_st u_{k,t}[n]` (no `U` for a scalar state),
    /// then, when time reversed, the conjugate (`(up, down) -> (down^*, -up^*)` for a spinor).
    pub(super) fn grid_states(
        &self,
        c: &Crystal,
        k: [f64; 3],
        image: &Image,
        input: &[&[C64]],
    ) -> Result<Vec<Buffer<C64>>> {
        let r = &self.cartesian[image.op];
        let shift = r.transpose() * self.translations[image.op];
        let phase = C64::from_polar(1., -(Vector3::from(k).dot(&shift)));
        let node = &c.permutations[image.op];
        let u = &self.spin[image.op];
        let components = input.len();
        let n = input.first().map_or(0, |x| x.len());
        let mut out = Vec::with_capacity(components);
        for _ in 0..components {
            out.push(c.context.allocate::<C64>(n)?);
        }
        for i in 0..n {
            let at = node[i];
            if components == 2 {
                let v: [C64; 2] = std::array::from_fn(|s| {
                    (u[s][0] * input[0][i] + u[s][1] * input[1][i]) * phase
                });
                let [a, b] = if image.reversed {
                    [v[1].conj(), -v[0].conj()]
                } else {
                    v
                };
                out[0].as_mut_slice()[at] = a;
                out[1].as_mut_slice()[at] = b;
            } else {
                let v = input[0][i] * phase;
                out[0].as_mut_slice()[at] = if image.reversed { v.conj() } else { v };
            }
        }
        Ok(out)
    }

    /// The projections `<p | psi_{R_s k}>` as the linear map `M` of those at `k` (`p x p`).
    pub(super) fn projector_map(&self, c: &Crystal, k: [f64; 3], op: usize) -> Matrix {
        let p = c.p;
        let r = &self.cartesian[op];
        let mut m = Matrix::zeros(p, p);
        let find = |atom: usize, l: usize, m: i32, radial: usize| {
            self.projectors
                .iter()
                .position(|&(a, ll, mm, rr)| a == atom && ll == l && mm == m && rr == radial)
        };
        for (j, &(a, l, m_from, radial)) in self.projectors.iter().enumerate() {
            let b = self.atom_maps[op][a];
            // `R' = R_s^T (lattice vector)`: the phase is `e^{-i k . R'}`.
            let phase = C64::from_polar(
                1.,
                -(Vector3::from(k).dot(&(r.transpose() * self.shifts[op][a]))),
            );
            for m_to in -(l as i32)..=l as i32 {
                let Some(t) = find(b, l, m_to, radial) else {
                    continue;
                };
                // `a'_{b, m_to} += phase conj(T[(m_from + l, m_to + l)]) a_{a, m_from}`
                let table = &self.tables[op][l];
                m[(t, j)] = phase
                    * table[((m_from + l as i32) as usize, (m_to + l as i32) as usize)].conj();
            }
        }
        m
    }

    /// The projections of the image of a state with projections `a` at the listed point `k`
    /// (`components * p`): `M a` per component, mixed by `U` for a spinor, then, when the image is
    /// time reversed, conjugated and reversed through the projectors (the components exchanged
    /// with `(up, down) -> (down^*, -up^*)` for a spinor).
    pub(super) fn project(
        &self,
        c: &Crystal,
        augmentation: &Augmentation,
        k: [f64; 3],
        image: &Image,
        a: &[C64],
    ) -> Result<Vec<C64>> {
        let p = c.p;
        let map = self.projector_map(c, k, image.op);
        let rotated: Vec<Vec<C64>> = a
            .chunks(p)
            .map(|chunk| {
                (&map * Matrix::from_column_slice(p, 1, chunk))
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        let mixed: Vec<Vec<C64>> = if rotated.len() == 2 {
            let u = &self.spin[image.op];
            (0..2)
                .map(|s| {
                    (0..p)
                        .map(|i| u[s][0] * rotated[0][i] + u[s][1] * rotated[1][i])
                        .collect()
                })
                .collect()
        } else {
            rotated
        };
        let mut out = Vec::with_capacity(a.len());
        if !image.reversed {
            mixed.into_iter().for_each(|part| out.extend(part));
            return Ok(out);
        }
        let order: &[(usize, f64)] = if mixed.len() == 2 {
            &[(1, 1.), (0, -1.)]
        } else {
            &[(0, 1.)]
        };
        for &(source, sign) in order {
            let reversed = augmentation.reversed(c, &mixed[source])?;
            out.extend(reversed.into_iter().map(|x| x * sign));
        }
        Ok(out)
    }

    /// The response `gamma'` (`components * p`) to the projections of an image, pulled back to the
    /// projections `a` of the listed point it was made from: `a' = (U x M) a`, `Pi conj(a')` when
    /// time reversed (`dE = 2 Re sum gamma' da'`, so `gamma = (U x M)^T gamma'` and, through the time
    /// reversal, `gamma'_source = sign conj(gamma'')`).
    pub(super) fn pull_back(
        &self,
        c: &Crystal,
        reversal: Option<&[(usize, usize, f64)]>,
        k: [f64; 3],
        image: &Image,
        gamma: &[C64],
    ) -> Result<Vec<C64>> {
        let p = c.p;
        let components = gamma.len() / p;
        let pulled: Vec<Vec<C64>> = if image.reversed {
            let map = reversal.ok_or_else(|| {
                invalid("a time-reversed image needs the time reversal of the projectors")
            })?;
            let order: &[(usize, f64)] = if components == 2 {
                &[(1, 1.), (0, -1.)]
            } else {
                &[(0, 1.)]
            };
            let mut out = vec![vec![C64::default(); p]; components];
            for (i, &(source, sign)) in order.iter().enumerate() {
                for &(target, from, s) in map {
                    out[source][from] += gamma[i * p + target].conj() * (sign * s);
                }
            }
            out
        } else {
            gamma.chunks(p).map(<[C64]>::to_vec).collect()
        };
        let map = self.projector_map(c, k, image.op);
        let by_map: Vec<Vec<C64>> = pulled
            .iter()
            .map(|g| {
                (0..p)
                    .map(|j| (0..p).map(|t| g[t] * map[(t, j)]).sum())
                    .collect()
            })
            .collect();
        if components == 2 {
            let u = &self.spin[image.op];
            Ok((0..2)
                .flat_map(|tau| {
                    let by_map = &by_map;
                    (0..p).map(move |j| u[0][tau] * by_map[0][j] + u[1][tau] * by_map[1][j])
                })
                .collect())
        } else {
            Ok(by_map.into_iter().next().unwrap_or_default())
        }
    }

    /// `P` of a listed point as the image sees it (`p x p`, or the `2p x 2p` spinor with blocks
    /// `(s, t)`): `(U x M) P (U x M)^+`, then `Pi P^* Pi^T` (`U_TR = i sigma_y x Pi` for a spinor)
    /// when time reversed (`Pi` the signed permutation of the time reversal of the projectors).
    pub(super) fn density(
        &self,
        c: &Crystal,
        reversal: Option<&[(usize, usize, f64)]>,
        k: [f64; 3],
        image: &Image,
        projected: &Matrix,
    ) -> Result<Matrix> {
        let p = c.p;
        let map = self.projector_map(c, k, image.op);
        let spinor = projected.nrows() == 2 * p;
        let rotated = if spinor {
            let u = &self.spin[image.op];
            let mut out = Matrix::zeros(2 * p, 2 * p);
            for s in 0..2 {
                for t in 0..2 {
                    let mut block = Matrix::zeros(p, p);
                    for (sp, tp) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let own = projected.view((sp * p, tp * p), (p, p)).into_owned();
                        block += (&map * own * map.adjoint()) * (u[s][sp] * u[t][tp].conj());
                    }
                    out.view_mut((s * p, t * p), (p, p)).copy_from(&block);
                }
            }
            out
        } else {
            &map * projected * map.adjoint()
        };
        if !image.reversed {
            return Ok(rotated);
        }
        let reversal = reversal.ok_or_else(|| {
            invalid("a time-reversed image needs the time reversal of the projectors")
        })?;
        let mut out = Matrix::zeros(rotated.nrows(), rotated.ncols());
        let blocks = if spinor { 2 } else { 1 };
        for s in 0..blocks {
            for t in 0..blocks {
                let sign = if s == t { 1. } else { -1. };
                for &(i, a, si) in reversal {
                    for &(j, b, sj) in reversal {
                        out[(s * p + i, t * p + j)] =
                            rotated[((blocks - 1 - s) * p + a, (blocks - 1 - t) * p + b)].conj()
                                * (sign * si * sj);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Operation `s` (`W`, fractional translation `t`) sends every grid node where the crystal's `s`-th
/// node permutation does (`n' = (W N_i / N_j) n + round(N t)` modulo `N`, the permutation the SCF
/// symmetrizes the density with): the setup block's operations are in the order of the snapshot's
/// permutations, each paired with its own.
fn check_node_permutation(c: &Crystal, s: usize, w: &Matrix3<f64>, t: [f64; 3]) -> Result<()> {
    let shape = c.grid.index.shape().map(|v| v as i64);
    let wrong = |what: &str| {
        invalid(format!(
            "symmetry operation {s} {what}: the operations of the setup block must be in the \
             order of the snapshot's node permutations, each reproducing its own"
        ))
    };
    let mut scale = [[0i64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let x = w[(i, j)] * shape[i] as f64 / shape[j] as f64;
            if (x - x.round()).abs() > 1e-9 {
                return Err(wrong("is not carried by the grid"));
            }
            scale[i][j] = x.round() as i64;
        }
    }
    let shift: [i64; 3] = std::array::from_fn(|i| (t[i] * shape[i] as f64).round() as i64);
    let permutation = &c.permutations[s];
    if permutation.len() as i64 != shape[0] * shape[1] * shape[2] {
        return Err(wrong("has a node permutation of the wrong size"));
    }
    let mut flat = 0;
    for a in 0..shape[0] {
        for b in 0..shape[1] {
            for d in 0..shape[2] {
                let n = [a, b, d];
                let image: [i64; 3] = std::array::from_fn(|i| {
                    ((0..3).map(|j| scale[i][j] * n[j]).sum::<i64>() + shift[i])
                        .rem_euclid(shape[i])
                });
                let want = (image[0] * shape[1] + image[1]) * shape[2] + image[2];
                if permutation[flat] as i64 != want {
                    return Err(wrong("does not reproduce its node permutation"));
                }
                flat += 1;
            }
        }
    }
    Ok(())
}

/// How the images of a reduced mesh were reached, for the tests that must know a gate really uses a
/// feature of the rotation (hidden: a diagnostic, not an API).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageCensus {
    /// Images over every listed point (the identity's included).
    pub images: usize,
    /// Images reached by an operation with a fractional translation.
    pub by_translation: usize,
    /// Of those, the ones whose grid phase `e^{-i k . R^-1 t}` is not 1.
    pub nontrivial_phase: usize,
    /// Images whose atoms land in another cell (`R_s tau_a + t_s - tau_b` not zero for some atom).
    pub with_lattice_shift: usize,
}

impl Reduction {
    pub(super) fn census(&self, c: &Crystal) -> ImageCensus {
        let shape = c.grid.index.shape();
        let lattice =
            c.grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
        let inverse = lattice.try_inverse().unwrap_or_else(Matrix3::identity);
        let mut out = ImageCensus::default();
        for (q, star) in self.stars.iter().enumerate() {
            for image in star {
                out.images += 1;
                let fractional = inverse * self.translations[image.op];
                if fractional.iter().any(|x| (x - x.round()).abs() > 1e-9) {
                    out.by_translation += 1;
                    let shift = self.cartesian[image.op].transpose() * self.translations[image.op];
                    let phase = Vector3::from(c.points[q].k).dot(&shift);
                    let turns = phase / (2. * std::f64::consts::PI);
                    if (turns - turns.round()).abs() > 1e-9 {
                        out.nontrivial_phase += 1;
                    }
                }
                if self.shifts[image.op].iter().any(|v| v.norm() > 1e-9) {
                    out.with_lattice_shift += 1;
                }
            }
        }
        out
    }
}

/// The SU(2) matrix of the rotation of an operation (`super::space_group::double_group::su2`).
fn su2(r: &Matrix3<f64>) -> [[C64; 2]; 2] {
    super::space_group::double_group::su2(r)
}
