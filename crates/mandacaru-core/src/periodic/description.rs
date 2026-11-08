//! The description of the PAW setup a snapshot was built from: the `setup` block of the crystal
//! snapshot, parsed and checked once.
//!
//! The derivative engines (forces, stress) rebuild the geometry of the crystal in Rust: the basis
//! functions as shells on radial tables, the projectors from the dataset files, the Gaussian ion
//! width. What the snapshot's other blocks do not say, this block does:
//!
//! ```text
//! "setup": { "version": 1,
//!            "sigma_bohr": 1.27,                              (the Gaussian ion width)
//!            "tables": [ {"l": 0, "r": [...], "values": [...], "confinement": null | 5.1}, ... ],
//!            "functions": {"atom": [...], "m": [...], "table": [...]},   (one entry per basis function)
//!            "projectors": [[atom, l, m, index], ...],          (the order of the p projectors)
//!            "operations": null | {"rotations": [[9 ints]], "translations": [[3]], "atom_maps": [[n]]} }
//! ```
//!
//! The atom centers, the dataset files with their SHA-256 digests and the projector basis are the
//! `sites` block's (read here too, once: the snapshot's source text is not kept); the moment
//! channels are `channel_labels`; the lattice is the grid's. A block that disagrees with the snapshot (function
//! count, atoms, a table index, a projector count, symmetry operations that do not reproduce the
//! snapshot's node permutations) is refused. An old snapshot has no block: its energy is
//! unaffected, and a force or a stress raises naming the missing block.
use super::{invalid, point_group, point_group::PointGroup};
use crate::{product, Context, Error, Reservation, Result};
use nalgebra::Matrix3;
use serde_json::Value;

/// The largest angular momentum of a basis function or a projector: the PAW Gaunt coefficients
/// stop at 8 (`math::angular::gaunt`).
pub const MAX_ORBITAL_L: usize = 8;
/// The largest degree of a moment channel (`l <= 16`, as the snapshot's own labels).
pub const MAX_MOMENT_L: usize = 16;

/// The symmetry operations of a reduced mesh, in the form a `setup` block carries them: integer
/// rotations (row by row), fractional translations and the atom map of each.
#[derive(Clone, Debug, PartialEq)]
pub struct Operations {
    pub rotations: Vec<[i32; 9]>,
    pub translations: Vec<[f64; 3]>,
    pub atom_maps: Vec<Vec<usize>>,
}

impl Operations {
    /// The heap `n` operations on `atoms` atoms take.
    pub fn bytes(n: usize, atoms: usize) -> Result<usize> {
        let each = product(&[atoms, std::mem::size_of::<usize>()])?
            .checked_add(std::mem::size_of::<[i32; 9]>() + std::mem::size_of::<[f64; 3]>()
                         + std::mem::size_of::<Vec<usize>>())
            .ok_or(Error::Overflow)?;
        product(&[n, each])
    }

    fn parse(value: &Value) -> Result<Self> {
        let rows = |key: &str| list(&value[key], "the symmetry operations");
        // Sized exactly, so the description keeps what `Operations::bytes` reserved.
        let n = rows("rotations")?.len();
        let mut out = Operations {
            rotations: Vec::with_capacity(n),
            translations: Vec::with_capacity(n),
            atom_maps: Vec::with_capacity(n),
        };
        for w in rows("rotations")? {
            let w = list(w, "a rotation")?;
            if w.len() != 9 {
                return Err(invalid("a rotation needs 9 numbers"));
            }
            let mut r = [0i32; 9];
            for (slot, x) in r.iter_mut().zip(w) {
                *slot = real(x, "a rotation")?.round() as i32;
            }
            out.rotations.push(r);
        }
        for t in rows("translations")? {
            let t = list(t, "a translation")?;
            if t.len() != 3 {
                return Err(invalid("a translation needs 3 numbers"));
            }
            out.translations.push([real(&t[0], "a translation")?, real(&t[1], "a translation")?,
                                   real(&t[2], "a translation")?]);
        }
        for m in rows("atom_maps")? {
            let entries = list(m, "an atom map")?;
            let mut map = Vec::with_capacity(entries.len());
            for x in entries {
                map.push(index(x, "an atom map entry")?);
            }
            out.atom_maps.push(map);
        }
        Ok(out)
    }

    fn to_json(&self) -> Value {
        serde_json::json!({"rotations": self.rotations, "translations": self.translations,
                           "atom_maps": self.atom_maps})
    }
}

/// One radial table of the basis (several functions of a shell share it).
#[derive(Clone, Debug)]
pub struct Table {
    pub l: usize,
    pub r: Vec<f64>,
    pub values: Vec<f64>,
    /// The radius a confined orbital is cut at; `None`: the end of the table.
    pub confinement: Option<f64>,
}

/// A checked `setup` block.
#[derive(Debug)]
pub struct Description {
    /// The Gaussian ion width of the local-potential split (Bohr).
    pub sigma: f64,
    pub tables: Vec<Table>,
    /// Per basis function: its atom, its `m`, its table.
    pub function_atom: Vec<usize>,
    pub function_m: Vec<i32>,
    pub function_table: Vec<usize>,
    /// Per projector: `(atom, l, m, radial index)`, in the order of the crystal's `p` columns.
    pub projectors: Vec<(usize, usize, i32, usize)>,
    /// The operations a symmetry-reduced mesh was built with, `None` for a full mesh.
    pub point_group: Option<PointGroup>,
    /// The atom centers (Bohr), from the `sites` block.
    pub centers: Vec<[f64; 3]>,
    /// The dataset file of every atom and the SHA-256 it was read with, from the `sites` block.
    pub datasets: Vec<(String, String)>,
    /// The projector basis of the datasets (`"raw"` or `"dual"`).
    pub projector_basis: String,
    /// The moment channels `(atom, L, M)`, in the order of the snapshot's blocks.
    pub channels: Vec<(usize, usize, i32)>,
    /// The operations as written, kept so a snapshot can write the block it was read from;
    /// `None` on a full mesh.
    pub operations: Option<Operations>,
    _memory: Reservation,
}

fn list<'a>(value: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("the setup block needs {what}")))
}

fn real(value: &Value, what: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| invalid(format!("the setup block: {what} must be finite")))
}

/// A list of finite numbers, held at exactly its length.
fn reals(value: &Value, what: &str) -> Result<Vec<f64>> {
    let items = list(value, what)?;
    let mut out = Vec::with_capacity(items.len());
    for x in items {
        out.push(real(x, what)?);
    }
    Ok(out)
}

fn index(value: &Value, what: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid(format!("the setup block: {what} must be an index")))
}

fn integer(value: &Value, what: &str) -> Result<i32> {
    value
        .as_i64()
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| invalid(format!("the setup block: {what} must be an integer")))
}

/// `l <= max` and `|m| <= l`.
fn angular(l: usize, m: i32, max: usize, what: &str) -> Result<()> {
    if l > max || m.unsigned_abs() as usize > l {
        return Err(invalid(format!(
            "the setup block: {what} needs l <= {max} and |m| <= l, it has l = {l}, m = {m}"
        )));
    }
    Ok(())
}

fn sum(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or(Error::Overflow)
}

/// The bytes `Description::parse` will hold, read from the lengths of the arrays the file already
/// contains (no allocation), after the checks that need only those lengths. A count the file
/// declares is bounded by what the arrays hold: functions by the crystal's `m`, projectors by its
/// `p`, operations by the node permutations the snapshot carries, datasets by the centers.
fn declared_bytes(
    root: &Value,
    block: &Value,
    m: usize,
    p: usize,
    permutations: usize,
) -> Result<usize> {
    let sites = &root["sites"];
    let atoms = list(&sites["centers"], "the sites block's atom centers")?.len();
    let datasets = list(&sites["datasets"], "the sites block's dataset files")?;
    if datasets.len() != atoms || atoms == 0 {
        return Err(invalid("the sites block needs one dataset per atom"));
    }
    let mut names = 0usize;
    for d in datasets {
        names = sum(&[
            names,
            d["file"].as_str().map_or(0, str::len),
            d["sha256"].as_str().map_or(0, str::len),
        ])?;
    }
    let basis = sites["projector_basis"]
        .as_str()
        .ok_or_else(|| invalid("the sites block needs the projector basis"))?
        .len();
    let channels = list(&root["channel_labels"], "the moment channel labels")?.len();
    let tables = list(&block["tables"], "the radial tables of the basis")?;
    let mut nodes = 0usize;
    for table in tables {
        let (r, values) = (
            list(&table["r"], "a table radius")?.len(),
            list(&table["values"], "a table")?.len(),
        );
        if r != values || r < 4 {
            return Err(invalid(
                "a radial table needs matching, strictly increasing nodes (at least four)",
            ));
        }
        nodes = sum(&[nodes, r])?;
    }
    let functions = &block["functions"];
    for (what, key) in [
        ("the atom of every basis function", "atom"),
        ("the m of every basis function", "m"),
        ("the table of every basis function", "table"),
    ] {
        let count = list(&functions[key], what)?.len();
        if count != m {
            return Err(invalid(format!(
                "the setup block describes {count} basis functions, the crystal has {m}"
            )));
        }
    }
    let rows = list(&block["projectors"], "the projector list")?.len();
    if rows != p {
        return Err(invalid(format!(
            "the setup block lists {rows} projectors on existing atoms, the crystal has {p}"
        )));
    }
    let group = match &block["operations"] {
        Value::Null => {
            if permutations != 0 {
                return Err(invalid(
                    "the snapshot symmetrizes its density with space-group operations, \
                     the setup block lists none",
                ));
            }
            0
        }
        operations => {
            let n = list(&operations["rotations"], "the symmetry rotations")?.len();
            if n != permutations {
                return Err(invalid(format!(
                    "the setup block lists {n} symmetry operations, the snapshot symmetrizes \
                     with {permutations}"
                )));
            }
            point_group::bytes(n, atoms)?
                .checked_add(Operations::bytes(n, atoms)?)
                .ok_or(Error::Overflow)?
        }
    };
    let (word, pair) = (std::mem::size_of::<usize>(), 2 * std::mem::size_of::<f64>());
    sum(&[
        product(&[nodes, pair])?,
        product(&[tables.len(), std::mem::size_of::<Table>()])?,
        product(&[m, 2 * word + std::mem::size_of::<i32>()])?,
        product(&[p, std::mem::size_of::<(usize, usize, i32, usize)>()])?,
        product(&[
            atoms,
            std::mem::size_of::<[f64; 3]>() + 2 * std::mem::size_of::<String>(),
        ])?,
        names,
        basis,
        product(&[channels, std::mem::size_of::<(usize, usize, i32)>()])?,
        group,
    ])
}

impl Description {
    /// Parses the `setup` block of the snapshot `root` of `m` basis functions and `p` projectors.
    ///
    /// `lattice` is the cell (columns, Bohr), `permutations` the snapshot's grid-node permutations.
    /// Everything the description holds is reserved from `ctx` BEFORE it is allocated, from the
    /// lengths of the arrays the file already carries ([`declared_bytes`]); every angular momentum,
    /// order and atom index is bounded here, before a shell or a shape reads it. Public (and hidden) so
    /// that the memory test can call it on a snapshot it already loaded; a crystal calls it once at load.
    #[doc(hidden)]
    pub fn parse(
        ctx: &Context,
        root: &Value,
        m: usize,
        p: usize,
        lattice: &Matrix3<f64>,
        shape: [usize; 3],
        permutations: &[Vec<usize>],
    ) -> Result<Self> {
        let block = &root["setup"];
        let sites = &root["sites"];
        let memory = ctx.reserve(declared_bytes(root, block, m, p, permutations.len())?)?;
        let center_list = list(&sites["centers"], "the sites block's atom centers")?;
        let mut centers = Vec::with_capacity(center_list.len());
        for c in center_list {
            let c = list(c, "an atom center")?;
            if c.len() != 3 {
                return Err(invalid("an atom center needs three numbers"));
            }
            centers.push([
                real(&c[0], "an atom center")?,
                real(&c[1], "an atom center")?,
                real(&c[2], "an atom center")?,
            ]);
        }
        // A dataset that was not read from a file has no name and no digest: the snapshot loads (its
        // energy never needs them) and the force raises when it has to read the file.
        let mut datasets = Vec::with_capacity(centers.len());
        for d in list(&sites["datasets"], "the sites block's dataset files")? {
            datasets.push((
                d["file"].as_str().unwrap_or("").to_string(),
                d["sha256"].as_str().unwrap_or("").to_string(),
            ));
        }
        let projector_basis = sites["projector_basis"]
            .as_str()
            .ok_or_else(|| invalid("the sites block needs the projector basis"))?
            .to_string();
        let label_list = list(&root["channel_labels"], "the moment channel labels")?;
        let mut channels = Vec::with_capacity(label_list.len());
        for c in label_list {
            let c = list(c, "a channel label")?;
            if c.len() != 3 {
                return Err(invalid("a channel label is [atom, L, M]"));
            }
            let (atom, l, order) = (
                index(&c[0], "a channel's atom")?,
                index(&c[1], "a channel's L")?,
                integer(&c[2], "a channel's M")?,
            );
            angular(l, order, MAX_MOMENT_L, "a moment channel")?;
            if atom >= centers.len() {
                return Err(invalid(
                    "a moment channel names an atom that does not exist",
                ));
            }
            channels.push((atom, l, order));
        }
        if block["version"] != 1 {
            return Err(invalid("the setup block needs version 1"));
        }
        let sigma = block["sigma_bohr"]
            .as_f64()
            .filter(|s| s.is_finite() && *s > 0.)
            .ok_or_else(|| invalid("the setup block needs a positive ion width"))?;
        let table_list = list(&block["tables"], "the radial tables of the basis")?;
        let mut tables = Vec::with_capacity(table_list.len());
        for table in table_list {
            let (r, values) = (
                reals(&table["r"], "a table radius")?,
                reals(&table["values"], "a table")?,
            );
            if r.windows(2).any(|w| w[1] <= w[0]) {
                return Err(invalid(
                    "a radial table needs matching, strictly increasing nodes (at least four)",
                ));
            }
            let l = index(&table["l"], "a table's l")?;
            angular(l, 0, MAX_ORBITAL_L, "a radial table")?;
            let confinement = match &table["confinement"] {
                Value::Null => None,
                x => Some(
                    x.as_f64()
                        .filter(|x| x.is_finite() && *x > 0.)
                        .ok_or_else(|| invalid("a confinement radius must be positive"))?,
                ),
            };
            tables.push(Table {
                l,
                r,
                values,
                confinement,
            });
        }
        let functions = &block["functions"];
        let mut function_atom = Vec::with_capacity(m);
        let mut function_m = Vec::with_capacity(m);
        let mut function_table = Vec::with_capacity(m);
        let atoms = list(&functions["atom"], "the atom of every basis function")?;
        let orders = list(&functions["m"], "the m of every basis function")?;
        let named = list(&functions["table"], "the table of every basis function")?;
        for ((atom, order), table) in atoms.iter().zip(orders).zip(named) {
            let (atom, order, table) = (
                index(atom, "a function's atom")?,
                integer(order, "a function's m")?,
                index(table, "a function's table")?,
            );
            let t = tables
                .get(table)
                .ok_or_else(|| invalid("a basis function names a table the setup block lacks"))?;
            if atom >= centers.len() || order.unsigned_abs() as usize > t.l {
                return Err(invalid(
                    "a basis function names an atom that does not exist or has |m| > l",
                ));
            }
            function_atom.push(atom);
            function_m.push(order);
            function_table.push(table);
        }
        let mut projectors = Vec::with_capacity(p);
        for row in list(&block["projectors"], "the projector list")? {
            let row = list(row, "a projector row")?;
            if row.len() != 4 {
                return Err(invalid("a projector row is [atom, l, m, index]"));
            }
            let q = (
                index(&row[0], "a projector's atom")?,
                index(&row[1], "a projector's l")?,
                integer(&row[2], "a projector's m")?,
                index(&row[3], "a projector's radial index")?,
            );
            angular(q.1, q.2, MAX_ORBITAL_L, "a projector")?;
            if q.0 >= centers.len() || q.3 >= p {
                return Err(invalid(format!(
                    "a projector row names atom {} and radial index {}, outside the {} atoms and \
                     {p} projectors of the crystal",
                    q.0,
                    q.3,
                    centers.len()
                )));
            }
            projectors.push(q);
        }
        let point_group = match &block["operations"] {
            Value::Null => None,
            operations => Some(PointGroup::parse(
                ctx,
                operations,
                lattice,
                &centers,
                shape,
                permutations,
            )?),
        };
        let operations = match &block["operations"] {
            Value::Null => None,
            value => Some(Operations::parse(value)?),
        };
        Ok(Self {
            sigma,
            tables,
            function_atom,
            function_m,
            function_table,
            projectors,
            point_group,
            centers,
            datasets,
            projector_basis,
            channels,
            operations,
            _memory: memory,
        })
    }

    /// The `setup` and `sites` blocks of a snapshot: the exact inverse of [`Description::parse`].
    pub fn blocks(&self) -> (Value, Value) {
        let tables: Vec<Value> = self.tables.iter().map(|t| serde_json::json!({
            "l": t.l, "r": t.r, "values": t.values, "confinement": t.confinement,
        })).collect();
        let setup = serde_json::json!({
            "version": 1,
            "sigma_bohr": self.sigma,
            "tables": tables,
            "functions": {"atom": self.function_atom, "m": self.function_m,
                          "table": self.function_table},
            "projectors": self.projectors.iter()
                .map(|&(atom, l, m, index)| serde_json::json!([atom, l, m, index]))
                .collect::<Vec<_>>(),
            "operations": self.operations.as_ref().map_or(Value::Null, Operations::to_json),
        });
        let sites = serde_json::json!({
            "centers": self.centers,
            "projector_basis": self.projector_basis,
            "datasets": self.datasets.iter()
                .map(|(file, sha256)| serde_json::json!({"file": file, "sha256": sha256}))
                .collect::<Vec<_>>(),
        });
        (setup, sites)
    }

    /// Construct a [`Description`] directly from pre-validated parts and reserve memory.
    pub fn from_parts(
        ctx: &Context,
        sigma: f64,
        tables: Vec<Table>,
        function_atom: Vec<usize>,
        function_m: Vec<i32>,
        function_table: Vec<usize>,
        projectors: Vec<(usize, usize, i32, usize)>,
        point_group: Option<PointGroup>,
        centers: Vec<[f64; 3]>,
        datasets: Vec<(String, String)>,
        projector_basis: String,
        channels: Vec<(usize, usize, i32)>,
        operations: Option<Operations>,
    ) -> Result<Self> {
        let mut bytes = 0;
        for t in &tables {
            bytes += t.r.len() * 8 + t.values.len() * 8;
        }
        bytes += function_atom.len() * 8 + function_m.len() * 4 + function_table.len() * 8;
        bytes += projectors.len() * 32;
        bytes += centers.len() * 24 + datasets.len() * 128 + channels.len() * 24;
        if let Some(o) = &operations {
            bytes += Operations::bytes(o.rotations.len(), centers.len())?;
        }
        let memory = ctx.reserve(bytes)?;
        Ok(Self {
            sigma,
            tables,
            function_atom,
            function_m,
            function_table,
            projectors,
            point_group,
            centers,
            datasets,
            projector_basis,
            channels,
            operations,
            _memory: memory,
        })
    }

    /// Clone this description using `ctx` to reserve memory for the clone.
    pub fn try_clone(&self, ctx: &Context) -> Result<Self> {
        let memory = ctx.reserve(self._memory.bytes())?;
        Ok(Self {
            sigma: self.sigma,
            tables: self.tables.clone(),
            function_atom: self.function_atom.clone(),
            function_m: self.function_m.clone(),
            function_table: self.function_table.clone(),
            projectors: self.projectors.clone(),
            point_group: self.point_group.clone(),
            centers: self.centers.clone(),
            datasets: self.datasets.clone(),
            projector_basis: self.projector_basis.clone(),
            channels: self.channels.clone(),
            operations: self.operations.clone(),
            _memory: memory,
        })
    }
}
