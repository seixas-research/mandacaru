//! Real, closed-shell MP2 and frozen virtual natural orbitals (physicists' ERIs).
use super::invalid;
use crate::{product, quantum::C64, Buffer, Context, Result};
use nalgebra::DMatrix;

pub struct Mp2Result {
    pub correlation_energy: f64,
    pub occupied_occupations: Buffer<f64>,
    pub virtual_occupations: Buffer<f64>,
    pub rotation: Buffer<f64>,
    pub occupied_density: Buffer<f64>,
}
fn real(values: &[C64]) -> Result<()> {
    if values
        .iter()
        .any(|c| !c.re.is_finite() || !c.im.is_finite() || c.im.abs() > 1e-8)
    {
        return Err(invalid(
            "MP2 requires finite real integrals (imaginary tolerance 1e-8)",
        ));
    }
    Ok(())
}
/// Ascending eigenpairs from the verified solver: the natural orbitals must diagonalize their
/// density to roundoff, or the active space (and the response of its force) is not the one the
/// occupations describe.
fn eigen(matrix: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> {
    crate::math::linalg::real_eigen((&matrix + matrix.transpose()) * 0.5)
}
pub fn natural_orbitals(
    ctx: &Context,
    m: usize,
    one: &[C64],
    two: &[C64],
    occupied: usize,
) -> Result<Mp2Result> {
    ctx.run(|| {
        if occupied==0 || occupied>=m || one.len()!=product(&[m,m])? || two.len()!=product(&[m,m,m,m])? {
            return Err(invalid("MP2 needs matching spatial integrals and at least one occupied and one virtual orbital"));
        }
        real(one)?; real(two)?;
        let _workspace=ctx.reserve(product(&[m,m,8,16])?)?;
        let idx=|p:usize,q:usize,r:usize,s:usize| ((p*m+q)*m+r)*m+s;
        let fock=DMatrix::from_fn(m,m,|p,q| {
            let entry=|p:usize,q:usize| one[p*m+q].re+(0..occupied).map(|i| 2.*two[idx(p,i,q,i)].re-two[idx(p,i,i,q)].re).sum::<f64>();
            0.5*(entry(p,q)+entry(q,p))
        });
        let mut canonical=DMatrix::<f64>::identity(m,m);
        let mut energies:Vec<_>=(0..m).map(|i| fock[(i,i)]).collect();
        let mut off=0f64;
        for (start,end) in [(0,occupied),(occupied,m)] {
            for p in start..end { for q in start..end { if p!=q { off=off.max(fock[(p,q)].abs()); } } }
        }
        if off>1e-4 {
            for (start,end) in [(0,occupied),(occupied,m)] {
                let size=end-start;
                let (e,mut vectors)=eigen(fock.view((start,start),(size,size)).into_owned())?;
                for j in 0..size {
                    if vectors[(j,j)]<0. { for i in 0..size { vectors[(i,j)] *= -1.; } }
                    energies[start+j]=e[j];
                    for i in 0..size { canonical[(start+i,start+j)]=vectors[(i,j)]; }
                }
            }
        }
        // Four bounded rectangular contractions; never form the rotated M^4 tensor.
        let virtuals=m-occupied;
        let target=[occupied,occupied,virtuals,virtuals];
        let mut shape=[m;4];
        let mut current:Option<Buffer<f64>>=None;
        for axis in 0..4 {
            let old=shape; shape[axis]=target[axis];
            let mut next=ctx.allocate::<f64>(product(&shape)?)?;
            for (flat,out) in next.as_mut_slice().iter_mut().enumerate() {
                let mut rest=flat; let mut indices=[0;4];
                for a in (0..4).rev() { indices[a]=rest%shape[a]; rest/=shape[a]; }
                let column=indices[axis]+if axis<2 {0} else {occupied};
                for source in 0..m {
                    indices[axis]=source;
                    let old_flat=((indices[0]*old[1]+indices[1])*old[2]+indices[2])*old[3]+indices[3];
                    let value=if let Some(ref buffer)=current { buffer.as_slice()[old_flat] } else { two[old_flat].re };
                    *out+=value*canonical[(source,column)];
                }
            }
            current=Some(next);
        }
        let block=current.unwrap();
        from_block_inner(ctx,block.as_slice(),&energies,occupied,&canonical)
    })
}
pub fn from_block(
    ctx: &Context,
    block: &[f64],
    energies: &[f64],
    occupied: usize,
    canonical: &[f64],
) -> Result<Mp2Result> {
    ctx.run(|| {
        let m=energies.len();
        if occupied==0 || occupied>=m || canonical.len()!=product(&[m,m])? || block.len()!=product(&[occupied,occupied,m-occupied,m-occupied])? || block.iter().chain(energies).chain(canonical).any(|v| !v.is_finite()) {
            return Err(invalid("finite MP2 block, energies and square canonical rotation must have consistent dimensions"));
        }
        let _workspace=ctx.reserve(product(&[m,m,8,16])?)?;
        let rotation=DMatrix::from_row_slice(m,m,canonical);
        for i in 0..m { for j in 0..m {
            let dot=(0..m).map(|k| rotation[(k,i)]*rotation[(k,j)]).sum::<f64>();
            if (dot-if i==j {1.} else {0.}).abs()>1e-9 || ((i<occupied)!=(j<occupied) && rotation[(i,j)].abs()>1e-9) {
                return Err(invalid("canonical rotation must be orthogonal and preserve occupied/virtual blocks"));
            }
        } }
        from_block_inner(ctx,block,energies,occupied,&rotation)
    })
}
fn from_block_inner(
    ctx: &Context,
    block: &[f64],
    energies: &[f64],
    o: usize,
    canonical: &DMatrix<f64>,
) -> Result<Mp2Result> {
    let m = energies.len();
    let v = m - o;
    let idx = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
    let mut t = ctx.allocate::<f64>(product(&[o, o, v, v])?)?;
    let mut energy = 0.;
    for i in 0..o {
        for j in 0..o {
            for a in 0..v {
                for b in 0..v {
                    let denominator = energies[i] + energies[j] - energies[o + a] - energies[o + b];
                    if !denominator.is_finite() || denominator.abs() < 1e-6 {
                        return Err(invalid("MP2 denominator is below 1e-6 Ha or overflows; choose a multireference active space"));
                    }
                    t.as_mut_slice()[idx(i, j, a, b)] = block[idx(i, j, a, b)] / denominator;
                }
            }
        }
    }
    let tilde = |i, j, a, b| 2. * t.as_slice()[idx(i, j, a, b)] - t.as_slice()[idx(i, j, b, a)];
    for i in 0..o {
        for j in 0..o {
            for a in 0..v {
                for b in 0..v {
                    energy += tilde(i, j, a, b) * block[idx(i, j, a, b)];
                }
            }
        }
    }
    let mut occupied_density = ctx.allocate::<f64>(product(&[o, o])?)?;
    for i in 0..o {
        for j in 0..o {
            let mut value = 0.;
            for k in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        value -= tilde(i, k, a, b) * t.as_slice()[idx(j, k, a, b)]
                            + tilde(j, k, a, b) * t.as_slice()[idx(i, k, a, b)];
                    }
                }
            }
            occupied_density.as_mut_slice()[i * o + j] = value + if i == j { 2. } else { 0. };
        }
    }
    let mut dvv = DMatrix::<f64>::zeros(v, v);
    for a in 0..v {
        for b in 0..v {
            for i in 0..o {
                for j in 0..o {
                    for c in 0..v {
                        dvv[(a, b)] += tilde(i, j, a, c) * t.as_slice()[idx(i, j, b, c)]
                            + tilde(i, j, b, c) * t.as_slice()[idx(i, j, a, c)];
                    }
                }
            }
        }
    }
    if !energy.is_finite()
        || occupied_density
            .as_slice()
            .iter()
            .chain(dvv.iter())
            .any(|x| !x.is_finite())
    {
        return Err(invalid("MP2 density or correlation energy overflows"));
    }
    let (values, vectors) = eigen(dvv)?;
    let mut virtual_occupations = ctx.allocate::<f64>(v)?;
    let mut occupied_occupations = ctx.allocate::<f64>(o)?;
    let mut rotation = ctx.allocate::<f64>(product(&[m, m])?)?;
    for i in 0..o {
        occupied_occupations.as_mut_slice()[i] = occupied_density.as_slice()[i * o + i];
    }
    for j in 0..v {
        virtual_occupations.as_mut_slice()[j] = values[v - 1 - j];
    }
    for p in 0..m {
        for q in 0..m {
            rotation.as_mut_slice()[p * m + q] = if q < o {
                canonical[(p, q)]
            } else {
                (0..v)
                    .map(|a| canonical[(p, o + a)] * vectors[(a, v - 1 - (q - o))])
                    .sum()
            };
        }
    }
    Ok(Mp2Result {
        correlation_energy: energy,
        occupied_occupations,
        virtual_occupations,
        rotation,
        occupied_density,
    })
}

/// One spin channel of open-shell MP2: its semicanonical orbitals (columns over the incoming
/// model orbitals, occupied then virtual), their energies, and the singles.
struct SpinBlock {
    occupied: DMatrix<f64>,
    virtuals: DMatrix<f64>,
    eps_occupied: Vec<f64>,
    eps_virtual: Vec<f64>,
    f_ov: DMatrix<f64>,
    t1: DMatrix<f64>,
}

/// Open-shell MP2 and the frozen natural orbitals of its spin-summed virtual density.
///
/// The reference is the determinant the open-shell model is built for: one set of spatial
/// orbitals (`model`, the UHF natural orbitals as columns over the orthonormal basis), the
/// lowest `n_alpha` filled with alpha and the lowest `n_beta` with beta electrons. Each spin
/// has its own Fock operator `F^s = h + J[D_a + D_b] - K[D_s]`, block-diagonalized separately in
/// that spin's occupied and virtual spaces (semicanonical orbitals), and `H0` is the sum of
/// those orbital energies: the partition of restricted open-shell MP2 (Knowles, Andrews, Amos,
/// Handy and Pople, Chem. Phys. Lett. 186, 130 (1991); Lauderdale et al., Chem. Phys. Lett. 187,
/// 21 (1991)). The occupied-virtual Fock elements survive in `V`, so the first-order wavefunction
/// has singles `t_i^a = f_ia / (e_i - e_a)` as well as doubles (same-spin antisymmetrized,
/// opposite-spin direct). With `n_alpha == n_beta` on a canonical RHF basis the singles vanish
/// and the result is [`natural_orbitals`]'.
///
/// The second-order density of each spin is rotated back to the model orbitals and the spins
/// summed. Only the true virtuals (empty in both spins, `max(n_alpha, n_beta)` and up) are ranked;
/// the rotation is the identity on every occupied orbital, which keeps the reference determinant.
/// `occupied_occupations` are the reference occupation (2 or 1) plus the density's diagonal,
/// `occupied_density` the occupied block (reference plus second order).
pub fn open_shell_natural_orbitals(
    integrals: &super::MolecularIntegrals,
    model: &DMatrix<C64>,
    n_alpha: usize,
    n_beta: usize,
) -> Result<Mp2Result> {
    let ctx = &integrals.context;
    let m = integrals.m();
    let (high, low) = (n_alpha.max(n_beta), n_alpha.min(n_beta));
    if high == 0 || high >= m || model.shape() != (m, m) {
        return Err(invalid(
            "open-shell MP2 needs an occupied and a virtual orbital in each spin and the model orbitals",
        ));
    }
    if model.iter().any(|v| v.im.abs() > 1e-10) {
        return Err(invalid("open-shell MP2 requires real reference orbitals"));
    }
    ctx.run(|| {
        let _workspace = ctx.reserve(product(&[m, m, 8, 32])?)?;
        let eri = integrals.eri()?;
        let h = DMatrix::from_fn(m, m, |i, j| C64::new(integrals.one.as_slice()[i * m + j], 0.));
        let density = |n: usize| {
            let c = model.columns(0, n);
            c * c.adjoint()
        };
        let (d_a, d_b) = (density(n_alpha), density(n_beta));
        let (coulomb, _) = super::scf::jk(&eri, m, &(&d_a + &d_b));
        let mut blocks = Vec::with_capacity(2);
        for (n, d) in [(n_alpha, &d_a), (n_beta, &d_b)] {
            let (_, exchange) = super::scf::jk(&eri, m, d);
            let fock_ao = &h + &coulomb - exchange;
            let fock = (model.adjoint() * fock_ao * model).map(|v| v.re);
            let fock = (&fock + fock.transpose()) * 0.5;
            let mut rotation = DMatrix::<f64>::zeros(m, m);
            let mut energies = vec![0.; m];
            for (start, size) in [(0, n), (n, m - n)] {
                if size == 0 {
                    continue;
                }
                let (e, vectors) = eigen(fock.view((start, start), (size, size)).into_owned())?;
                for j in 0..size {
                    energies[start + j] = e[j];
                    for i in 0..size {
                        rotation[(start + i, start + j)] = vectors[(i, j)];
                    }
                }
            }
            let occupied = rotation.columns(0, n).into_owned();
            let virtuals = rotation.columns(n, m - n).into_owned();
            let f_ov = occupied.transpose() * &fock * &virtuals;
            let t1 = DMatrix::from_fn(n, m - n, |i, a| f_ov[(i, a)] / gap(energies[i] - energies[n + a]));
            blocks.push(SpinBlock {
                occupied,
                virtuals,
                eps_occupied: energies[..n].to_vec(),
                eps_virtual: energies[n..].to_vec(),
                f_ov,
                t1,
            });
        }
        // <ij|ab> over semicanonical orbitals given as columns over the model orbitals.
        let block = |ci: &DMatrix<f64>, cj: &DMatrix<f64>, ca: &DMatrix<f64>, cb: &DMatrix<f64>| {
            let lift = |c: &DMatrix<f64>| model * c.map(|v| C64::new(v, 0.));
            let values = super::transform::four(ctx, &eri, [&lift(ci), &lift(cj), &lift(ca), &lift(cb)])?;
            real(values.as_slice())?;
            Ok::<Vec<f64>, crate::Error>(values.as_slice().iter().map(|v| v.re).collect())
        };
        let mut correlation = 0.;
        let mut d_vir = Vec::new();
        let mut d_occ = Vec::new();
        let mut same = Vec::new();
        for s in &blocks {
            let (o, v) = (s.eps_occupied.len(), s.eps_virtual.len());
            correlation += s.f_ov.component_mul(&s.t1).sum();
            let g = block(&s.occupied, &s.occupied, &s.virtuals, &s.virtuals)?;
            let at = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
            let mut t = vec![0.; g.len()];
            for i in 0..o {
                for j in 0..o {
                    for a in 0..v {
                        for b in 0..v {
                            let anti = g[at(i, j, a, b)] - g[at(i, j, b, a)];
                            let value = anti
                                / gap(s.eps_occupied[i] + s.eps_occupied[j] - s.eps_virtual[a] - s.eps_virtual[b]);
                            t[at(i, j, a, b)] = value;
                            correlation += 0.25 * anti * value;
                        }
                    }
                }
            }
            same.push(t);
        }
        let (alpha, beta) = (&blocks[0], &blocks[1]);
        let (oa, va, ob, vb) = (
            alpha.eps_occupied.len(),
            alpha.eps_virtual.len(),
            beta.eps_occupied.len(),
            beta.eps_virtual.len(),
        );
        let g_ab = if oa > 0 && ob > 0 {
            block(&alpha.occupied, &beta.occupied, &alpha.virtuals, &beta.virtuals)?
        } else {
            vec![]
        };
        let ab = |i: usize, j: usize, a: usize, b: usize| ((i * ob + j) * va + a) * vb + b;
        let mut t_ab = vec![0.; g_ab.len()];
        for i in 0..oa {
            for j in 0..ob {
                for a in 0..va {
                    for b in 0..vb {
                        let value = g_ab[ab(i, j, a, b)]
                            / gap(alpha.eps_occupied[i] + beta.eps_occupied[j]
                                - alpha.eps_virtual[a]
                                - beta.eps_virtual[b]);
                        t_ab[ab(i, j, a, b)] = value;
                        correlation += g_ab[ab(i, j, a, b)] * value;
                    }
                }
            }
        }
        // The second-order density of each spin in its semicanonical blocks.
        for (spin, s) in blocks.iter().enumerate() {
            let (o, v) = (s.eps_occupied.len(), s.eps_virtual.len());
            let t = &same[spin];
            let at = |i: usize, j: usize, a: usize, b: usize| ((i * o + j) * v + a) * v + b;
            let mut vir = s.t1.transpose() * &s.t1;
            let mut occ = -(&s.t1 * s.t1.transpose());
            for a in 0..v {
                for b in 0..v {
                    let mut sum = 0.;
                    for i in 0..o {
                        for j in 0..o {
                            for c in 0..v {
                                sum += 0.5 * t[at(i, j, a, c)] * t[at(i, j, b, c)];
                            }
                        }
                    }
                    // The opposite-spin pairs, with this spin's virtual free.
                    if spin == 0 {
                        for i in 0..oa {
                            for j in 0..ob {
                                for c in 0..vb {
                                    sum += t_ab[ab(i, j, a, c)] * t_ab[ab(i, j, b, c)];
                                }
                            }
                        }
                    } else {
                        for i in 0..oa {
                            for j in 0..ob {
                                for c in 0..va {
                                    sum += t_ab[ab(i, j, c, a)] * t_ab[ab(i, j, c, b)];
                                }
                            }
                        }
                    }
                    vir[(a, b)] += sum;
                }
            }
            for i in 0..o {
                for j in 0..o {
                    let mut sum = 0.;
                    for k in 0..o {
                        for a in 0..v {
                            for b in 0..v {
                                sum += 0.5 * t[at(i, k, a, b)] * t[at(j, k, a, b)];
                            }
                        }
                    }
                    if spin == 0 {
                        for k in 0..ob {
                            for a in 0..va {
                                for b in 0..vb {
                                    sum += t_ab[ab(i, k, a, b)] * t_ab[ab(j, k, a, b)];
                                }
                            }
                        }
                    } else {
                        for k in 0..oa {
                            for a in 0..va {
                                for b in 0..vb {
                                    sum += t_ab[ab(k, i, a, b)] * t_ab[ab(k, j, a, b)];
                                }
                            }
                        }
                    }
                    occ[(i, j)] -= sum;
                }
            }
            d_vir.push(vir);
            d_occ.push(occ);
        }
        let mut gamma = DMatrix::<f64>::zeros(m, m);
        for (s, (oo, vv)) in blocks.iter().zip(d_occ.iter().zip(&d_vir)) {
            gamma += &s.occupied * oo * s.occupied.transpose() + &s.virtuals * vv * s.virtuals.transpose();
        }
        let gamma = (&gamma + gamma.transpose()) * 0.5;
        if !correlation.is_finite() || gamma.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "open-shell MP2 met a vanishing orbital-energy denominator (a degenerate frontier)",
            ));
        }
        let (values, vectors) = eigen(gamma.view((high, high), (m - high, m - high)).into_owned())?;
        let mut rotation = ctx.allocate::<f64>(product(&[m, m])?)?;
        for p in 0..high {
            rotation.as_mut_slice()[p * m + p] = 1.;
        }
        let v = m - high;
        for i in 0..v {
            for j in 0..v {
                // Descending: the most occupied natural orbital first.
                rotation.as_mut_slice()[(high + i) * m + high + j] = vectors[(i, v - 1 - j)];
            }
        }
        let mut occupied_occupations = ctx.allocate::<f64>(high)?;
        let mut occupied_density = ctx.allocate::<f64>(product(&[high, high])?)?;
        for p in 0..high {
            let reference = if p < low { 2. } else { 1. };
            occupied_occupations.as_mut_slice()[p] = reference + gamma[(p, p)];
            for q in 0..high {
                occupied_density.as_mut_slice()[p * high + q] =
                    gamma[(p, q)] + if p == q { reference } else { 0. };
            }
        }
        let mut virtual_occupations = ctx.allocate::<f64>(v)?;
        for j in 0..v {
            virtual_occupations.as_mut_slice()[j] = values[v - 1 - j];
        }
        Ok(Mp2Result {
            correlation_energy: correlation,
            occupied_occupations,
            virtual_occupations,
            rotation,
            occupied_density,
        })
    })
}

/// An orbital-energy denominator, refused when it vanishes.
fn gap(value: f64) -> f64 {
    if value.abs() < 1e-10 {
        f64::NAN
    } else {
        value
    }
}
