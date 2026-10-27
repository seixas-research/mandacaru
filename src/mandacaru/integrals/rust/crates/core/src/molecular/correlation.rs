//! Real, closed-shell MP2 and frozen virtual natural orbitals (physicists' ERIs).
use super::invalid;
use crate::{product, quantum::C64, Buffer, Context, Result};
use nalgebra::{DMatrix, SymmetricEigen};

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
fn eigen(matrix: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> {
    let e = SymmetricEigen::try_new(matrix, 1e-14, 10000)
        .ok_or_else(|| crate::Error::Kernel("MP2 eigensolver did not converge".into()))?;
    let n = e.eigenvalues.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| e.eigenvalues[i].total_cmp(&e.eigenvalues[j]));
    Ok((
        order.iter().map(|&j| e.eigenvalues[j]).collect(),
        DMatrix::from_fn(n, n, |i, j| e.eigenvectors[(i, order[j])]),
    ))
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
