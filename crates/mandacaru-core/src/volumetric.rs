//! Streaming CUBE/XSF/BXSF writers. Scalar fields are row-major; lengths in Bohr.
use crate::{persistence::atomic_write, product, units::BOHR_TO_ANGSTROM, Context, Error, Result};
use std::{io::Write, path::Path};
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
fn io(e: std::io::Error) -> Error {
    Error::Invalid(e.to_string())
}
pub struct Field<'a> {
    pub shape: [usize; 3],
    pub data: &'a [f64],
    pub origin: [f64; 3],
    pub step: [[f64; 3]; 3],
    pub numbers: &'a [u8],
    pub positions: &'a [[f64; 3]],
    pub charges: &'a [f64],
}
impl Field<'_> {
    fn validate(&self) -> Result<()> {
        if self.shape.contains(&0)
            || product(&self.shape)? != self.data.len()
            || self.numbers.len() != self.positions.len()
            || self.charges.len() != self.numbers.len()
            || self
                .data
                .iter()
                .chain(&self.origin)
                .chain(self.step.iter().flatten())
                .chain(self.positions.iter().flatten())
                .chain(self.charges)
                .any(|x| !x.is_finite())
            || self.numbers.iter().any(|z| *z > 118)
        {
            return Err(invalid(
                "field dimensions, geometry and finite real values must agree",
            ));
        }
        Ok(())
    }
    pub fn write(
        &self,
        ctx: &Context,
        path: &Path,
        format: &str,
        comment: &str,
        cell: Option<[[f64; 3]; 3]>,
        periodic: bool,
    ) -> Result<()> {
        self.validate()?;
        if format != "cube" && format != "xsf" {
            return Err(invalid("scalar format must be cube or xsf"));
        }
        if cell.is_some_and(|v| v.iter().flatten().any(|x| !x.is_finite())) {
            return Err(invalid("cell must be finite"));
        }
        if periodic && format == "xsf" {
            let cell = cell.ok_or_else(|| invalid("a periodic datagrid needs a cell"))?;
            for (a, vector) in cell.iter().enumerate() {
                for (r, &value) in vector.iter().enumerate() {
                    if (value - self.step[r][a] * self.shape[a] as f64).abs()
                        > 1e-8 + 1e-5 * value.abs()
                    {
                        return Err(invalid("periodic grid must span its cell"));
                    }
                }
            }
        }
        let _workspace = ctx.reserve(8192)?;
        atomic_write(path, |out| {
            if format == "cube" {
                self.cube(out, comment)
            } else {
                self.xsf(out, cell, periodic)
            }
        })
    }
    fn cube(&self, out: &mut dyn Write, comment: &str) -> Result<()> {
        let comment = comment.replace('\n', " ").trim().to_owned();
        writeln!(
            out,
            "{}\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z",
            if comment.is_empty() {
                "Mandacaru"
            } else {
                &comment
            }
        )
        .map_err(io)?;
        writeln!(
            out,
            "{:5}{:12.6}{:12.6}{:12.6}",
            self.numbers.len(),
            self.origin[0],
            self.origin[1],
            self.origin[2]
        )
        .map_err(io)?;
        for a in 0..3 {
            writeln!(
                out,
                "{:5}{:12.6}{:12.6}{:12.6}",
                self.shape[a], self.step[0][a], self.step[1][a], self.step[2][a]
            )
            .map_err(io)?;
        }
        for ((&z, &charge), r) in self.numbers.iter().zip(self.charges).zip(self.positions) {
            writeln!(
                out,
                "{z:5}{charge:12.6}{:12.6}{:12.6}{:12.6}",
                r[0], r[1], r[2]
            )
            .map_err(io)?;
        }
        for scan in self.data.chunks(self.shape[2]) {
            for line in scan.chunks(6) {
                for v in line {
                    // Rust omits the '+' and zero-padded exponent used by CUBE.
                    let text = format!("{v:.5E}");
                    let (mantissa, exponent) = text.split_once('E').unwrap();
                    let exponent: i32 = exponent.parse().unwrap();
                    write!(out, "{:>13}", format!("{mantissa}E{exponent:+03}")).map_err(io)?;
                }
                writeln!(out).map_err(io)?;
            }
        }
        Ok(())
    }
    fn xsf(&self, out: &mut dyn Write, cell: Option<[[f64; 3]; 3]>, periodic: bool) -> Result<()> {
        let b = BOHR_TO_ANGSTROM;
        if let Some(cell) = cell {
            writeln!(out, "CRYSTAL\nPRIMVEC").map_err(io)?;
            for r in cell {
                writeln!(out, " {:.14} {:.14} {:.14}", r[0] * b, r[1] * b, r[2] * b).map_err(io)?;
            }
            writeln!(out, "PRIMCOORD\n {} 1", self.numbers.len()).map_err(io)?;
        } else {
            writeln!(out, "ATOMS").map_err(io)?;
        }
        for (&z, r) in self.numbers.iter().zip(self.positions) {
            writeln!(
                out,
                " {z:2} {:.14} {:.14} {:.14}",
                r[0] * b,
                r[1] * b,
                r[2] * b
            )
            .map_err(io)?;
        }
        let shape = self.shape.map(|v| if periodic { v + 1 } else { v });
        writeln!(out,"BEGIN_BLOCK_DATAGRID_3D\n data\n BEGIN_DATAGRID_3Dgrid#1\n  {} {} {}\n  {:.6} {:.6} {:.6}",shape[0],shape[1],shape[2],self.origin[0]*b,self.origin[1]*b,self.origin[2]*b).map_err(io)?;
        for a in 0..3 {
            let span = if periodic {
                cell.unwrap()[a]
            } else {
                std::array::from_fn(|r| self.step[r][a] * (self.shape[a] - 1) as f64)
            };
            writeln!(
                out,
                "  {:.6} {:.6} {:.6}",
                span[0] * b,
                span[1] * b,
                span[2] * b
            )
            .map_err(io)?;
        }
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let v = self.data[((i % self.shape[0]) * self.shape[1] + j % self.shape[1])
                        * self.shape[2]
                        + k % self.shape[2]];
                    write!(out, " {v:.6}").map_err(io)?;
                }
                writeln!(out).map_err(io)?;
            }
        }
        writeln!(out, " END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D").map_err(io)
    }
}
pub fn bxsf(
    ctx: &Context,
    path: &Path,
    shape: [usize; 4],
    data: &[f64],
    cell: [[f64; 3]; 3],
    fermi: f64,
    comment: &str,
) -> Result<()> {
    if shape.contains(&0)
        || product(&shape)? != data.len()
        || data
            .iter()
            .chain(cell.iter().flatten())
            .any(|x| !x.is_finite())
        || !fermi.is_finite()
    {
        return Err(invalid(
            "band grid and reciprocal cell must have consistent finite values",
        ));
    }
    let matrix = nalgebra::Matrix3::from_row_slice(&cell.concat());
    if matrix.determinant().abs() < 1e-12 {
        return Err(invalid("reciprocal cell vectors must be independent"));
    }
    let _workspace = ctx.reserve(8192)?;
    let [bands, nx, ny, nz] = shape;
    let closed = [
        nx.checked_add(1).ok_or(Error::Overflow)?,
        ny.checked_add(1).ok_or(Error::Overflow)?,
        nz.checked_add(1).ok_or(Error::Overflow)?,
    ];
    product(&closed)?;
    atomic_write(path, |out| {
        writeln!(out,"BEGIN_INFO\n  # {}\n  Fermi Energy: {fermi:.8}\nEND_INFO\nBEGIN_BLOCK_BANDGRID_3D\n  band_energies\n  BEGIN_BANDGRID_3D_bands\n  {bands}\n  {} {} {}\n  0.0 0.0 0.0",comment.split_whitespace().collect::<Vec<_>>().join(" "),closed[0],closed[1],closed[2]).map_err(io)?;
        for r in cell {
            writeln!(out, "  {:.10} {:.10} {:.10}", r[0], r[1], r[2]).map_err(io)?;
        }
        for band in 0..bands {
            writeln!(out, "  BAND: {}", band + 1).map_err(io)?;
            let mut column = 0;
            for i in 0..closed[0] {
                for j in 0..closed[1] {
                    for k in 0..closed[2] {
                        let value = data[((band * nx + i % nx) * ny + j % ny) * nz + k % nz];
                        write!(out, " {value:.6}").map_err(io)?;
                        column += 1;
                        if column == 6 {
                            writeln!(out).map_err(io)?;
                            column = 0;
                        }
                    }
                }
            }
            if column != 0 {
                writeln!(out).map_err(io)?;
            }
        }
        writeln!(out, "  END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3D").map_err(io)
    })
}

/// The content of a `.bxsf` band grid: the energies on the periodic mesh (the closing planes of
/// the file dropped), `shape = [bands, n1, n2, n3]`, row-major with `k3` innermost; the
/// reciprocal vectors as rows, as written; the Fermi energy of the header.
pub struct BandGrid {
    pub shape: [usize; 4],
    pub data: Vec<f64>,
    pub cell: [[f64; 3]; 3],
    pub fermi: f64,
}

/// Read a `.bxsf` file written by [`bxsf`] (or any writer of the same single-block layout: the
/// values of a band may wrap at any column).
pub fn read_bxsf(ctx: &Context, path: &Path) -> Result<BandGrid> {
    let name = path.display();
    let size = std::fs::metadata(path).map_err(io)?.len();
    let _text = ctx.reserve(usize::try_from(size).map_err(|_| Error::Overflow)?)?;
    let text = std::fs::read_to_string(path).map_err(io)?;
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let line = |i: usize| -> Result<&str> {
        lines
            .get(i)
            .copied()
            .ok_or_else(|| Error::Invalid(format!("{name}: the band grid ends early")))
    };
    let unreadable = |i: usize| Error::Invalid(format!("{name}: cannot read line {}", i + 1));
    let fermi = lines
        .iter()
        .find(|l| l.to_lowercase().starts_with("fermi energy:"))
        .ok_or_else(|| Error::Invalid(format!("{name}: no 'Fermi Energy:' line in BEGIN_INFO")))?;
    let fermi: f64 = fermi
        .split_once(':')
        .and_then(|(_, value)| value.trim().parse().ok())
        .ok_or_else(|| Error::Invalid(format!("{name}: cannot read the Fermi energy")))?;
    let start = lines
        .iter()
        .position(|l| l.to_uppercase().starts_with("BEGIN_BANDGRID_3D"))
        .ok_or_else(|| Error::Invalid(format!("{name}: no BEGIN_BANDGRID_3D block")))?;
    let bands: usize = line(start + 1)?
        .parse()
        .map_err(|_| unreadable(start + 1))?;
    let closed: Vec<usize> = line(start + 2)?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| unreadable(start + 2))?;
    let closed: [usize; 3] = closed.try_into().map_err(|_| unreadable(start + 2))?;
    if closed.iter().any(|&n| n < 2) {
        return Err(Error::Invalid(format!(
            "{name}: a general band grid has at least two points per direction"
        )));
    }
    let mut cell = [[0.0; 3]; 3];
    for (a, row) in cell.iter_mut().enumerate() {
        let values: Vec<f64> = line(start + 4 + a)?
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| unreadable(start + 4 + a))?;
        *row = values.try_into().map_err(|_| unreadable(start + 4 + a))?;
    }
    let count = product(&closed)?;
    let [nx, ny, nz] = closed.map(|n| n - 1);
    let shape = [bands, nx, ny, nz];
    // The mesh and one band of the file, reserved before either is allocated: a header that
    // names more points than the budget holds is a refusal, not an allocation failure.
    let _data = ctx.reserve(product(&[
        product(&shape)?.checked_add(count).ok_or(Error::Overflow)?,
        8,
    ])?)?;
    let mut data = Vec::with_capacity(product(&shape)?);
    let mut at = start + 7;
    let mut values = Vec::with_capacity(count);
    for band in 0..bands {
        if !line(at)?.to_uppercase().starts_with("BAND:") {
            return Err(Error::Invalid(format!(
                "{name}: expected 'BAND:' at line {}",
                at + 1
            )));
        }
        at += 1;
        values.clear();
        while values.len() < count {
            for word in line(at)?.split_whitespace() {
                values.push(word.parse::<f64>().map_err(|_| unreadable(at))?);
            }
            at += 1;
        }
        if values.len() != count {
            return Err(Error::Invalid(format!(
                "{name}: band {} has {} values for a {}x{}x{} grid",
                band + 1,
                values.len(),
                closed[0],
                closed[1],
                closed[2]
            )));
        }
        for i in 0..nx {
            for j in 0..ny {
                let row = (i * closed[1] + j) * closed[2];
                data.extend_from_slice(&values[row..row + nz]);
            }
        }
    }
    Ok(BandGrid {
        shape,
        data,
        cell,
        fermi,
    })
}
