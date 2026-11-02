//! Building the Rust setup of a frozen case from its manifest and the dataset files: the basis of
//! the sidecar as shells, the raw projectors of every atom, and the block-diagonal `q` and
//! `D_ion` in the raw projector basis.
#![allow(dead_code)]

use mandacaru_core::{
    paw::record::PawDataset,
    periodic::setup::{
        pulay::Matrix,
        shells::{Radial, Shell, Shells},
        spheres::Projector,
    },
    Context,
};
use num_complex::Complex64;
use serde_json::Value;
use std::sync::Arc;

use super::reals;

/// The basis of the sidecar as shells (one per atom and table) and the atom of each function.
pub fn basis(ctx: &Context, setup: &Value) -> (Shells, Vec<usize>) {
    let tables: Vec<(usize, Arc<Radial>)> = setup["basis_radial_tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["l"].as_u64().unwrap() as usize,
                Arc::new(
                    Radial::new(ctx, &reals(&t["r"]["re"]), &reals(&t["values"]["re"]), None)
                        .unwrap(),
                ),
            )
        })
        .collect();
    let mut shells: Vec<Shell> = Vec::new();
    let mut owner = Vec::new();
    for (row, f) in setup["basis_functions"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let (table, m) = (
            f["table"].as_u64().unwrap() as usize,
            f["m"].as_i64().unwrap() as i32,
        );
        let c = reals(&f["center_bohr"]);
        let center = [c[0], c[1], c[2]];
        owner.push(f["atom"].as_u64().unwrap() as usize);
        match shells.last_mut() {
            Some(s) if s.center == center && Arc::ptr_eq(&s.radial, &tables[table].1) => {
                s.ms.push(m);
                s.rows.push(row);
            }
            _ => shells.push(Shell {
                center,
                l: tables[table].0,
                ms: vec![m],
                rows: vec![row],
                radial: tables[table].1.clone(),
            }),
        }
    }
    (Shells::new(shells).unwrap(), owner)
}

/// The raw projectors of one atom, in the order of the Python crystal (`l`, then `m`, then the
/// radial index).
pub fn projector_list(ctx: &Context, ds: &PawDataset, center: [f64; 3]) -> Vec<Projector> {
    let mut out = Vec::new();
    for (&l, channel) in &ds.channels {
        let radials: Vec<_> = channel
            .raw_projectors
            .iter()
            .map(|t| Arc::new(Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).unwrap()))
            .collect();
        for m in -(l as i32)..=l as i32 {
            for (index, radial) in radials.iter().enumerate() {
                out.push(Projector {
                    atom: 0,
                    l: l as usize,
                    m,
                    index,
                    center,
                    r_cut: channel.r_cut.max(ds.r_cut_local),
                    radial: radial.clone(),
                });
            }
        }
    }
    out
}

/// The block-diagonal `p x p` matrix of a dataset block (`coupling`, `overlap_correction`) in the
/// raw projector basis: `B^-1 X B^-T` on every `(atom, l, m)` group.
pub fn assemble(
    projectors: &[Projector],
    datasets: &[PawDataset],
    pick: impl Fn(&mandacaru_core::paw::record::Channel) -> &[f64],
) -> Matrix {
    let p = projectors.len();
    let mut out = Matrix::zeros(p, p);
    let mut start = 0;
    while start < p {
        let head = &projectors[start];
        let group: Vec<usize> = (start..p)
            .take_while(|&i| {
                (projectors[i].atom, projectors[i].l, projectors[i].m)
                    == (head.atom, head.l, head.m)
            })
            .collect();
        let channel = &datasets[head.atom].channels[&(head.l as u32)];
        let n = group.len();
        let x = nalgebra::DMatrix::from_row_slice(n, n, pick(channel));
        let b = nalgebra::DMatrix::from_row_slice(n, n, channel.vanderbilt.data());
        let b_inverse = b
            .try_inverse()
            .expect("the projector transform is invertible");
        let block = &b_inverse * x * b_inverse.transpose();
        for (i, &a) in group.iter().enumerate() {
            for (j, &c) in group.iter().enumerate() {
                out[(a, c)] = Complex64::new(block[(i, j)], 0.);
            }
        }
        start += n;
    }
    out
}
