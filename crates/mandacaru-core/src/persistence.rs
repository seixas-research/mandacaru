//! Bounded schema-v1 Hamiltonians, schema-v1/v2 checkpoints and atomic snapshots.
use crate::{
    math::storage::Precision,
    parquet::{self, Codec, Table, WriteOptions},
    product,
    quantum::{
        evolution, solver::by_precision, Amplitude, Basis, CachedProblem, Encoding, Operator,
        SolveOptions, SolveResult, Space, State, C64,
    },
    snapshot_table, Buffer, Context, Error, Reservation, Result,
};
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

pub const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FILE_QUBITS: usize = 50;
pub const HAMILTONIAN: &str = "mandacaru-qubit-hamiltonian";
pub const CHECKPOINT: &str = "mandacaru-wavefunction-checkpoint";
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub fn integer(value: &Value, name: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| invalid(format!("{name} must be a nonnegative integer")))
}
fn number(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| invalid(format!("{name} must be finite")))
}
pub fn pauli(ctx: &Context, n: usize, rows: &Value) -> Result<Operator> {
    let rows = rows
        .as_array()
        .ok_or_else(|| invalid("Pauli terms must be [label, real, imag] triples"))?;
    let _memory = ctx.reserve(product(&[rows.len(), 128])?)?;
    let mut terms = Vec::new();
    for row in rows {
        let row = row
            .as_array()
            .filter(|r| r.len() == 3)
            .ok_or_else(|| invalid("Pauli terms must be [label, real, imag] triples"))?;
        let label = row[0]
            .as_str()
            .ok_or_else(|| invalid("Pauli label must be a string"))?
            .to_owned();
        terms.push((
            label,
            C64::new(
                number(&row[1], "real coefficient")?,
                number(&row[2], "imaginary coefficient")?,
            ),
        ));
    }
    Operator::new(ctx, n, &terms)
}
pub fn pauli_payload(op: &Operator) -> Value {
    let mut terms = op.labels();
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    json!({"num_qubits":op.n,"terms":terms.into_iter().map(|(s,c)| json!([s,c.re,c.im])).collect::<Vec<_>>()})
}
/// Upper bound of the heap bytes a JSON value holds (the `Value` itself is
/// 32 bytes; strings, arrays and map nodes add their contents and a fixed
/// allocator and node overhead).
fn value_bytes(value: &Value) -> usize {
    const NODE: usize = std::mem::size_of::<Value>();
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => NODE,
        Value::String(text) => NODE + text.len() + 16,
        Value::Array(items) => items.iter().fold(NODE + 16, |total, item| {
            total.saturating_add(value_bytes(item))
        }),
        Value::Object(map) => map.iter().fold(NODE + 16, |total, (key, item)| {
            total
                .saturating_add(key.len())
                .saturating_add(64)
                .saturating_add(value_bytes(item))
        }),
    }
}
/// The run a checkpoint was written by: its method and Hartree energy (when
/// recorded), the number of ansatz operators and free parameters it holds, and
/// its status object as JSON text.
#[derive(Debug, Clone, PartialEq)]
pub struct Lineage {
    pub method: Option<String>,
    pub energy: Option<f64>,
    pub operators: usize,
    pub parameters: usize,
    pub status: Option<String>,
}
/// What a Hamiltonian file records about a double-factorized two-body representation:
/// `metadata.two_body = {"representation", "truncation_ev", "error_bound_ev"}`.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoBodyRecord {
    pub representation: String,
    /// The truncation the model was built with, eV.
    pub truncation_ev: f64,
    /// The bound the truncation reached on the change of any energy, eV.
    pub error_bound_ev: f64,
}
pub struct Document {
    pub(crate) payload: Value,
    _memory: Reservation,
}
impl Document {
    pub fn payload(&self) -> &Value {
        &self.payload
    }
    pub fn parse(ctx: &Context, text: &str, kind: &str) -> Result<Self> {
        if text.len() > MAX_JSON_BYTES {
            return Err(invalid("JSON snapshot exceeds the 64 MiB input limit"));
        }
        let memory = ctx.reserve(product(&[
            text.len().checked_add(1).ok_or(Error::Overflow)?,
            32,
        ])?)?;
        Self::parse_inner(ctx, text, kind, memory)
    }
    fn parse_inner(ctx: &Context, text: &str, kind: &str, memory: Reservation) -> Result<Self> {
        let payload: Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))
            .map_err(|e| invalid(format!("invalid JSON snapshot: {e}")))?;
        let doc = Self {
            payload,
            _memory: memory,
        };
        doc.validate(ctx, kind)?;
        Ok(doc)
    }
    /// A document built from an in-memory payload, validated like a parsed
    /// one. The payload's size is charged before it is kept.
    pub fn from_value(ctx: &Context, payload: Value, kind: &str) -> Result<Self> {
        let memory = ctx.reserve(value_bytes(&payload))?;
        let doc = Self {
            payload,
            _memory: memory,
        };
        doc.validate(ctx, kind)?;
        Ok(doc)
    }
    pub fn read(ctx: &Context, path: &Path, kind: &str) -> Result<Self> {
        let file = File::open(path).map_err(|e| invalid(format!("cannot open snapshot: {e}")))?;
        // A leading `PAR1` marks a Parquet file; JSON never starts with it.
        if parquet::is_parquet(path)? {
            drop(file);
            return Self::read_parquet(ctx, path, kind);
        }
        let bytes = usize::try_from(file.metadata().map_err(|e| invalid(e.to_string()))?.len())
            .map_err(|_| Error::Overflow)?;
        if bytes > MAX_JSON_BYTES {
            return Err(invalid("JSON snapshot exceeds the 64 MiB input limit"));
        }
        let memory = ctx.reserve(product(&[bytes + 1, 32])?)?;
        let mut text = String::new();
        text.try_reserve_exact(bytes + 1)
            .map_err(|_| Error::Allocation)?;
        file.take(bytes as u64 + 1)
            .read_to_string(&mut text)
            .map_err(|e| invalid(format!("cannot read snapshot: {e}")))?;
        if text.len() > bytes {
            return Err(invalid("snapshot changed while reading"));
        }
        Self::parse_inner(ctx, &text, kind, memory)
    }
    /// Reads only the header/metadata of a snapshot, returning a validated `Document`
    /// whose payload has an empty `terms` array and `n_terms` set to the term count.
    /// For Parquet files, never reads data pages.
    pub fn read_header(ctx: &Context, path: &Path, kind: &str) -> Result<Self> {
        let file = File::open(path).map_err(|e| invalid(format!("cannot open snapshot: {e}")))?;
        drop(file);
        if parquet::is_parquet(path)? {
            let source = format!("'{}'", path.display());
            let footer = parquet::read_metadata(ctx, path)?;
            let (payload, memory) =
                snapshot_table::header_from_footer(ctx, &footer, kind, &source)?;
            let doc = Self {
                payload,
                _memory: memory,
            };
            doc.validate(ctx, kind)?;
            Ok(doc)
        } else {
            let mut doc = Self::read(ctx, path, kind)?;
            let n_terms = doc.payload["terms"].as_array().map_or(0, Vec::len);
            if let Some(obj) = doc.payload.as_object_mut() {
                obj.insert("n_terms".into(), json!(n_terms));
                obj.insert("terms".into(), json!([]));
            }
            Ok(doc)
        }
    }
    /// Reads a Hamiltonian stored as Parquet. The table is released before the
    /// payload is validated, so the peak is the larger of the two, not their sum.
    fn read_parquet(ctx: &Context, path: &Path, kind: &str) -> Result<Self> {
        let source = format!("'{}'", path.display());
        let table = parquet::read_table(ctx, path)?;
        let (payload, memory) = snapshot_table::payload_from_table(ctx, &table, kind, &source)?;
        drop(table);
        let doc = Self {
            payload,
            _memory: memory,
        };
        doc.validate(ctx, kind)?;
        Ok(doc)
    }
    /// Builds a Hamiltonian document from a Parquet table (`pauli`, `real`,
    /// `imag` columns and the `mandacaru.*` footer entries) and validates it
    /// like any other document. `source` names the table in messages.
    pub fn from_table(ctx: &Context, table: &Table, kind: &str, source: &str) -> Result<Self> {
        let (payload, memory) = snapshot_table::payload_from_table(ctx, table, kind, source)?;
        let doc = Self {
            payload,
            _memory: memory,
        };
        doc.validate(ctx, kind)?;
        Ok(doc)
    }
    /// The Parquet table of a Hamiltonian document, rows sorted by Pauli label.
    pub fn to_table(&self, ctx: &Context) -> Result<Table> {
        snapshot_table::table_from_payload(ctx, &self.payload)
    }
    /// Writes a Hamiltonian document as Parquet, atomically. Nothing is created
    /// when the document cannot be converted.
    pub fn save_parquet(&self, ctx: &Context, path: &Path, codec: Codec) -> Result<()> {
        let table = self.to_table(ctx)?;
        parquet::write_table(
            ctx,
            path,
            &table,
            &WriteOptions {
                codec,
                ..WriteOptions::default()
            },
        )
    }
    fn validate(&self, ctx: &Context, kind: &str) -> Result<()> {
        let p = &self.payload;
        if !p.is_object()
            || p["format"].as_str() != Some(kind)
            || ![HAMILTONIAN, CHECKPOINT].contains(&kind)
        {
            return Err(invalid(format!("expected a {kind} snapshot")));
        }
        let version = integer(&p["version"], "schema version")?;
        if version == 0 || version > if kind == CHECKPOINT { 3 } else { 1 } {
            return Err(invalid("unsupported snapshot schema version"));
        }
        let n = integer(
            &p[if kind == CHECKPOINT {
                "n_qubits"
            } else {
                "num_qubits"
            }],
            "qubit count",
        )?;
        if n > MAX_FILE_QUBITS {
            return Err(invalid("Pauli-string files support at most 50 qubits"));
        }
        let metadata = p.get("metadata").filter(|v| !v.is_null());
        if metadata.is_some_and(|v| !v.is_object())
            || p.get("status")
                .is_some_and(|v| !v.is_null() && !v.is_object())
        {
            return Err(invalid("metadata and status must be JSON objects"));
        }
        let unit = if kind == CHECKPOINT {
            p.get("energy_unit")
        } else {
            metadata.and_then(|v| v.get("energy_unit"))
        };
        if unit.is_some_and(|u| u.as_str() != Some("Ha")) {
            return Err(invalid("snapshot energies must be in Ha"));
        }
        let mapping = p
            .get("mapping")
            .and_then(Value::as_str)
            .unwrap_or("jordan_wigner");
        if !["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"].contains(&mapping) {
            return Err(invalid("unsupported snapshot mapping"));
        }
        let particles = p.get("num_particles").filter(|v| !v.is_null());
        let spatial = p.get("n_spatial_orbitals").filter(|v| !v.is_null());
        if let Some(parts) = particles {
            let parts = parts
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| invalid("num_particles must contain two integers"))?;
            let counts = [
                integer(&parts[0], "alpha count")?,
                integer(&parts[1], "beta count")?,
            ];
            if let Some(spatial) = spatial {
                let m = integer(spatial, "spatial orbital count")?;
                // A Z2-tapered register is the encoding's minus the qubits the taper removed.
                let removed = p["metadata"]["taper"]["anchors"]
                    .as_array()
                    .map_or(0, Vec::len);
                if Encoding::new(mapping, product(&[m, 2])?, counts)?.n_qubits() != n + removed {
                    return Err(invalid(
                        "snapshot qubits disagree with mapping and orbitals",
                    ));
                }
            }
        }
        if kind == HAMILTONIAN {
            pauli(ctx, n, &p["terms"])?;
        } else {
            for q in p["reference_qubits"]
                .as_array()
                .ok_or_else(|| invalid("checkpoint requires reference_qubits"))?
            {
                if integer(q, "reference qubit")? >= n {
                    return Err(invalid("reference qubit outside register"));
                }
            }
            if p.get("preparation")
                .is_some_and(|v| ![Some("product"), Some("sum")].contains(&v.as_str()))
            {
                return Err(invalid("preparation must be product or sum"));
            }
            if p.get("energy").is_some_and(|v| !v.is_null()) {
                number(&p["energy"], "energy")?;
            }
            for op in p["operators"]
                .as_array()
                .ok_or_else(|| invalid("checkpoint requires operators"))?
            {
                number(&op["parameter"], "parameter")?;
                let g = Self::embedded(ctx, &op["generator"], n)?;
                if !g.is_antihermitian() {
                    return Err(invalid("checkpoint generators must be anti-Hermitian"));
                }
            }
            if !p["hamiltonian"].is_null() {
                let h = Self::embedded(ctx, &p["hamiltonian"], n)?;
                if !h.is_hermitian() {
                    return Err(invalid("checkpoint Hamiltonian must be Hermitian"));
                }
            }
        }
        Ok(())
    }
    fn embedded(ctx: &Context, p: &Value, n: usize) -> Result<Operator> {
        if integer(&p["num_qubits"], "operator qubits")? != n {
            return Err(invalid(
                "embedded operator qubit count differs from snapshot",
            ));
        }
        pauli(ctx, n, &p["terms"])
    }
    pub fn text(&self) -> Result<String> {
        serde_json::to_string(&self.payload).map_err(|e| invalid(e.to_string()))
    }
    /// The two-body record of a Hamiltonian file (`None` for a dense one, which writes none);
    /// a malformed record raises.
    pub fn two_body(&self) -> Result<Option<TwoBodyRecord>> {
        let p = &self.payload;
        if p["format"] != HAMILTONIAN {
            return Err(invalid("a two-body record belongs to a qubit-Hamiltonian file"));
        }
        let record = &p["metadata"]["two_body"];
        if record.is_null() {
            return Ok(None);
        }
        let energy = |key: &str| {
            record[key]
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.)
                .ok_or_else(|| invalid(format!("the two-body record needs a nonnegative {key}")))
        };
        let representation = record["representation"]
            .as_str()
            .filter(|r| *r == "double-factorized")
            .ok_or_else(|| invalid("the two-body record names an unknown representation"))?;
        Ok(Some(TwoBodyRecord {
            representation: representation.into(),
            truncation_ev: energy("truncation_ev")?,
            error_bound_ev: energy("error_bound_ev")?,
        }))
    }
    /// What a checkpoint records about the run that wrote it, for a log to say
    /// what a resumed run was restored from. A Hamiltonian snapshot has no
    /// lineage and is refused. Nothing is copied but the status text.
    pub fn lineage(&self) -> Result<Lineage> {
        let p = &self.payload;
        if p["format"] != CHECKPOINT {
            return Err(invalid("lineage requires a checkpoint"));
        }
        let operators = p["operators"]
            .as_array()
            .ok_or_else(|| invalid("checkpoint requires operators"))?
            .len();
        // A fixed Hamiltonian-variational ansatz stores its expanded stages as
        // operators; its free parameters are the logical ones in the metadata.
        let parameters = match p["metadata"]["hva"]["parameters"].as_array() {
            Some(logical) => logical.len(),
            None => operators,
        };
        let status = match p.get("status") {
            Some(value) if !value.is_null() => {
                Some(serde_json::to_string(value).map_err(|e| invalid(e.to_string()))?)
            }
            _ => None,
        };
        Ok(Lineage {
            method: p["method"].as_str().map(str::to_owned),
            energy: p.get("energy").and_then(Value::as_f64),
            operators,
            parameters,
            status,
        })
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write(path, |out| {
            serde_json::to_writer_pretty(&mut *out, &self.payload)
                .map_err(|e| invalid(e.to_string()))?;
            out.write_all(b"\n").map_err(|e| invalid(e.to_string()))
        })
    }
    pub fn problem(
        &self,
        ctx: &Context,
        sector: bool,
        dense: bool,
        singles: bool,
    ) -> Result<CachedProblem> {
        let space = if sector {
            Space::Sector
        } else {
            Space::Register
        };
        self.problem_in(ctx, space, dense, singles)
    }
    /// The quantum problem of this Hamiltonian snapshot in `space` (see [`Space`]).
    pub fn problem_in(
        &self,
        ctx: &Context,
        space: Space,
        dense: bool,
        singles: bool,
    ) -> Result<CachedProblem> {
        let p = &self.payload;
        if p["format"] != HAMILTONIAN {
            return Err(invalid("expected a Hamiltonian snapshot"));
        }
        let parts = p["num_particles"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| invalid("quantum solve requires two particle counts"))?;
        let encoding = Encoding::new(
            p["mapping"].as_str().unwrap_or("jordan_wigner"),
            product(&[integer(&p["n_spatial_orbitals"], "orbital count")?, 2])?,
            [
                integer(&parts[0], "alpha count")?,
                integer(&parts[1], "beta count")?,
            ],
        )?;
        let mut h = pauli(ctx, integer(&p["num_qubits"], "qubits")?, &p["terms"])?;
        if let Some(groups) = p["metadata"]["groups"].as_array() {
            for group in groups {
                h.decomposition.push(Self::embedded(ctx, group, h.n)?);
            }
        }
        CachedProblem::new_in(ctx, encoding, h, space, dense, singles)
    }
    /// The state this checkpoint prepares, stored in `precision` (complex128
    /// or complex64); the generators and parameters stay float64.
    pub fn state(&self, ctx: &Context, precision: Precision) -> Result<State> {
        by_precision!(precision, A => self.state_as::<A>(ctx).map(Amplitude::wrap))
    }
    fn state_as<A: Amplitude>(&self, ctx: &Context) -> Result<Buffer<A>> {
        ctx.run(|| {
            if self.payload["format"] != CHECKPOINT {
                return Err(invalid("state preparation requires a checkpoint"));
            }
            let p = &self.payload;
            let n = integer(&p["n_qubits"], "qubits")?;
            // A wide register (a shot run's) cannot be held as a state vector: say so by name.
            let too_wide = |e: crate::Error| match e {
                crate::Error::Budget { .. } | crate::Error::Overflow => invalid(&format!(
                    "this checkpoint's state has 2^{n} amplitudes, beyond the calculation's memory \
                     budget; a wide (shot-run) checkpoint is prepared on the device: use its circuit"
                )),
                other => other,
            };
            let basis = Basis::register(ctx, n).map_err(too_wide)?;
            let mut initial = ctx.allocate::<A>(basis.len()).map_err(too_wide)?;
            let mut index = 0;
            for q in p["reference_qubits"].as_array().unwrap() {
                index |= 1usize << (n - 1 - integer(q, "reference qubit")?);
            }
            initial.as_mut_slice()[index] = A::narrow(C64::new(1., 0.));
            let operators = p["operators"].as_array().unwrap();
            let _memory = ctx.reserve(product(&[operators.len(), 64])?)?;
            let generators: Vec<_> = operators
                .iter()
                .map(|op| Self::embedded(ctx, &op["generator"], n))
                .collect::<Result<_>>()?;
            let parameters: Vec<_> = operators
                .iter()
                .map(|op| number(&op["parameter"], "parameter"))
                .collect::<Result<_>>()?;
            let ops: Vec<_> = generators.iter().collect();
            evolution::state_from(
                ctx,
                &basis,
                &ops,
                &parameters,
                p["preparation"] != "sum",
                false,
                Some(initial.as_slice()),
            )
        })
    }
    pub fn expectation(&self, ctx: &Context) -> Result<f64> {
        let state = self.state_as::<C64>(ctx)?;
        ctx.run(|| {
            let n = integer(&self.payload["n_qubits"], "qubits")?;
            let basis = Basis::register(ctx, n)?;
            let h = Self::embedded(ctx, &self.payload["hamiltonian"], n)?;
            let action = h.apply(ctx, &basis, state.as_slice())?;
            let value = evolution::inner(state.as_slice(), action.as_slice());
            if !value.re.is_finite() || value.im.abs() > 1e-10 {
                return Err(invalid("nonfinite checkpoint expectation"));
            }
            Ok(value.re)
        })
    }
    /// The checks every resume makes: the register (mapping, width, tapering), the Hamiltonian,
    /// the reference determinant and the particle and orbital counts are the problem's.
    fn check_register(&self, problem: &CachedProblem) -> Result<()> {
        let p = &self.payload;
        if p["format"] != CHECKPOINT
            || p["mapping"] != problem.encoding.mapping
            || integer(&p["n_qubits"], "qubits")? != problem.n_qubits()
            || p["metadata"].get("taper").unwrap_or(&Value::Null) != &taper_record(problem)
        {
            return Err(invalid("checkpoint mapping/register differs from problem"));
        }
        let n = problem.n_qubits();
        let h = Self::embedded(&problem.context, &p["hamiltonian"], n)?;
        if !same_operator(&h, &problem.hamiltonian) {
            return Err(invalid("checkpoint Hamiltonian differs from problem"));
        }
        let mut reference = 0u64;
        for q in p["reference_qubits"].as_array().unwrap() {
            reference |= 1u64 << (n - 1 - integer(q, "reference qubit")?);
        }
        if reference != problem.register_reference() {
            return Err(invalid(
                "checkpoint reference differs from Hartree-Fock reference",
            ));
        }
        let parts = problem.encoding.particles;
        if p["num_particles"] != json!(parts)
            || p["n_spatial_orbitals"] != json!(problem.encoding.modes / 2)
        {
            return Err(invalid(
                "checkpoint particle/orbital counts differ from problem",
            ));
        }
        Ok(())
    }
    /// The stored generators as pool indices, with their parameters.
    fn sequence(&self, problem: &CachedProblem) -> Result<(Vec<usize>, Vec<f64>)> {
        let ctx = &problem.context;
        let n = problem.n_qubits();
        let operators = self.payload["operators"].as_array().unwrap();
        let _memory = ctx.reserve(product(&[operators.len(), 32])?)?;
        let mut indices = Vec::new();
        let mut parameters = Vec::new();
        for op in operators {
            let g = Self::embedded(ctx, &op["generator"], n)?;
            indices.push(
                problem
                    .pool
                    .iter()
                    .position(|p| same_operator(&g, &p.operator))
                    .ok_or_else(|| invalid("checkpoint generator is absent from native pool"))?,
            );
            parameters.push(number(&op["parameter"], "parameter")?);
        }
        Ok((indices, parameters))
    }
    /// `options` for resuming the `count` stored operators: with the curvature estimate the
    /// writer's SLSQP or BFGS growth carried, so the resumed run grows from it as the
    /// uninterrupted one would (the growth continues, see `SolveOptions::continuation`).
    ///
    /// The estimate belongs to the optimizer that kept it (a Hessian for SLSQP, an inverse
    /// Hessian for BFGS). A run with the OTHER one does not use it and says so in the returned
    /// notes, as it does for a file of an older version that recorded none: it starts from the
    /// identity and relaxes the stored sequence. It is not converted: the inverse of one
    /// optimizer's estimate is not what the other would have carried, so a conversion would
    /// promise a continuation it cannot give, and refusing outright would make the file
    /// unusable after a deliberate change of optimizer for what is only a starting guess. A
    /// version-3 file without an entry was written by another optimizer or a fixed ansatz and
    /// starts from the identity silently.
    fn with_stored_hessian(&self, options: &SolveOptions, count: usize) -> Result<(SolveOptions, Vec<String>)> {
        let p = &self.payload;
        let mut options = options.clone();
        let mut notes = Vec::new();
        if options.adapt
            && options.quenching
            && options.optimizer.method.carries_curvature()
            && count > 0
        {
            let method = options.optimizer.method.name();
            match p["status"].get("hessian") {
                None if integer(&p["version"], "schema version")? >= 3 => {}
                None => notes.push(
                    "the checkpoint records no Hessian approximation (it predates schema version 3): \
                     the resumed growth starts from the identity, so it need not reproduce the run \
                     that was never stopped"
                        .to_owned(),
                ),
                // Version-3 entries written before BFGS carried one are SLSQP's.
                Some(entry) if entry["optimizer"].as_str().unwrap_or("SLSQP") != method => notes.push(format!(
                    "the checkpoint's curvature estimate was kept by {}, not {method} (a Hessian and \
                     an inverse Hessian are not interchangeable): the resumed growth starts from \
                     the identity, so it need not reproduce the run that was never stopped",
                    entry["optimizer"].as_str().unwrap_or("SLSQP")
                )),
                Some(entry) => {
                    let values: Vec<f64> = serde_json::from_value(entry["values"].clone())
                        .map_err(|_| invalid("checkpoint Hessian lacks its values"))?;
                    if integer(&entry["dimension"], "Hessian dimension")? != count
                        || values.len() != count * count
                        || values.iter().any(|v| !v.is_finite())
                    {
                        return Err(invalid("checkpoint Hessian differs from its operator count"));
                    }
                    options.hessian = Some(values);
                    options.continuation = true;
                }
            }
        }
        Ok((options, notes))
    }
    /// Continue a subspace calculation from its checkpoint ([`subspace_checkpoint`]): the same
    /// register, method and state weights; `subspace-vqe` restarts its optimization from the
    /// stored parameters, `subspace-adapt-vqe` keeps growing from the stored sequence (continuing from its
    /// stored Hessian, or relaxing it first when the file has none).
    pub fn resume_subspace(
        &self,
        problem: &CachedProblem,
        options: &SolveOptions,
        weights: &[f64],
    ) -> Result<crate::quantum::subspace::SubspaceResult> {
        let p = &self.payload;
        self.check_register(problem)?;
        let method = if options.adapt { "subspace-adapt-vqe" } else { "subspace-vqe" };
        if p["method"] != method {
            return Err(invalid("checkpoint method differs from solver"));
        }
        let stored: Vec<f64> = serde_json::from_value(p["subspace"]["weights"].clone())
            .map_err(|_| invalid("a subspace checkpoint needs its state weights"))?;
        if stored != weights {
            return Err(invalid("checkpoint subspace weights differ from the run's"));
        }
        let (indices, parameters) = self.sequence(problem)?;
        if indices.len() > 1 && (p["preparation"] == "sum") == (options.adapt || options.trotter) {
            return Err(invalid("checkpoint preparation differs from ansatz evolution"));
        }
        if !options.adapt && indices != (0..problem.pool.len()).collect::<Vec<_>>() {
            return Err(invalid(
                "checkpoint generator sequence differs from fixed UCCSD ansatz",
            ));
        }
        let (options, notes) = self.with_stored_hessian(options, indices.len())?;
        let mut result = problem.solve_subspace_sequence(
            &options,
            weights,
            Some(&parameters),
            options.adapt.then_some(indices.as_slice()),
        )?;
        result.notes.extend(notes);
        Ok(result)
    }
    pub fn resume(&self, problem: &CachedProblem, options: &SolveOptions) -> Result<SolveResult> {
        self.resume_observed(problem, options, None)
    }
    pub fn resume_observed(
        &self,
        problem: &CachedProblem,
        options: &SolveOptions,
        observer: Option<&mut crate::quantum::solver::GrowthObserver<'_>>,
    ) -> Result<SolveResult> {
        let p = &self.payload;
        self.check_register(problem)?;
        let n = problem.n_qubits();
        let ctx = &problem.context;
        if p["method"] != if options.adapt { "adapt-vqe" } else { "vqe" } {
            return Err(invalid("checkpoint method differs from solver"));
        }
        let operators = p["operators"].as_array().unwrap();
        if let Some(hva) = &problem.hva {
            let indices: Vec<usize> =
                serde_json::from_value(p["metadata"]["hva"]["indices"].clone())
                    .map_err(|_| invalid("HVA checkpoint lacks its logical group indices"))?;
            let parameters: Vec<f64> =
                serde_json::from_value(p["metadata"]["hva"]["parameters"].clone())
                    .map_err(|_| invalid("HVA checkpoint lacks its logical parameters"))?;
            if indices != (0..problem.pool.len()).collect::<Vec<_>>()
                || p["preparation"] != "product"
                || options.adapt
            {
                return Err(invalid(
                    "HVA checkpoint logical sequence differs from fixed ansatz",
                ));
            }
            let expanded = hva.expand(&problem.pool, &indices, &parameters)?;
            if operators.len() != expanded.operators.len() {
                return Err(invalid("HVA checkpoint evolution differs from the ansatz"));
            }
            for ((entry, operator), &parameter) in operators
                .iter()
                .zip(expanded.operators)
                .zip(&expanded.parameters)
            {
                let actual = Self::embedded(ctx, &entry["generator"], n)?;
                if !same_operator(&actual, operator)
                    || (number(&entry["parameter"], "parameter")? - parameter).abs() > 1e-12
                {
                    return Err(invalid(
                        "HVA checkpoint stages differ from its logical parameters",
                    ));
                }
            }
            return problem.solve(options, Some(&parameters));
        }
        if operators.len() > 1 && (p["preparation"] == "sum") == (options.adapt || options.trotter)
        {
            return Err(invalid(
                "checkpoint preparation differs from ansatz evolution",
            ));
        }
        let (indices, parameters) = self.sequence(problem)?;
        if !options.adapt && indices != (0..problem.pool.len()).collect::<Vec<_>>() {
            return Err(invalid(
                "checkpoint generator sequence differs from fixed UCCSD ansatz",
            ));
        }
        let (options, notes) = self.with_stored_hessian(options, indices.len())?;
        let options = &options;
        let mut result = if let Some(observer) = observer {
            problem.solve_observed(
                options,
                Some(&parameters),
                if options.adapt { Some(&indices) } else { None },
                observer,
            )?
        } else {
            problem.solve_sequence(
                options,
                Some(&parameters),
                if options.adapt { Some(&indices) } else { None },
            )?
        };
        result.notes.extend(notes);
        Ok(result)
    }
}
/// The Z2 tapering a checkpoint was written on (`null` for an untapered register): its anchors,
/// the sector signs and the generators, so a resume on another register is refused.
fn taper_record(problem: &CachedProblem) -> Value {
    problem.taper.as_ref().map_or(
        Value::Null,
        |t| json!({"anchors": t.anchors, "signs": t.signs, "generators": t.labels()}),
    )
}
fn same_operator(a: &Operator, b: &Operator) -> bool {
    a.n == b.n
        && a.terms.len() == b.terms.len()
        && a.terms.iter().zip(&b.terms).all(|(a, b)| {
            a.x == b.x && a.z == b.z && (a.coefficient - b.coefficient).norm() <= 1e-10
        })
}
/// `status` with the curvature estimate of the ansatz parameters (`hessian`, row-major `n x n`)
/// the writer's growth carried at this point: `{"optimizer", "dimension", "values"}`, where
/// `optimizer` names who kept it (`"SLSQP"`: a Hessian approximation; `"BFGS"`: an INVERSE
/// Hessian approximation: the two are not interchangeable). A writer that carried none
/// (another optimizer, a fixed ansatz) adds nothing. Checkpoint schema version 3 introduced
/// the key (a file of an older version that lacks it recorded none because it could not); a
/// version-3 entry without `optimizer` was written when only SLSQP carried one.
pub fn status_with_hessian(mut status: Value, optimizer: &str, hessian: Option<&[f64]>) -> Result<Value> {
    let Some(values) = hessian else {
        return Ok(status);
    };
    let n = (0..=values.len()).find(|n| n * n >= values.len()).unwrap_or(0);
    if n * n != values.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("a checkpoint Hessian must be a finite n x n matrix"));
    }
    status
        .as_object_mut()
        .ok_or_else(|| invalid("a checkpoint status must be a JSON object"))?
        .insert("hessian".into(), json!({"optimizer": optimizer, "dimension": n, "values": values}));
    Ok(status)
}
#[allow(clippy::too_many_arguments)]
pub fn checkpoint(
    ctx: &Context,
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
    product_form: bool,
    method: &str,
    status: Value,
    energy: f64,
) -> Result<Document> {
    if indices.len() != parameters.len()
        || indices.iter().any(|&i| i >= problem.pool.len())
        || parameters.iter().any(|x| !x.is_finite())
        || !energy.is_finite()
        || !["vqe", "adapt-vqe", "vasqa", "valsqa", "subspace-vqe", "subspace-adapt-vqe"]
            .contains(&method)
        || (!["vqe", "subspace-vqe"].contains(&method) && !product_form)
    {
        return Err(invalid("invalid checkpoint ansatz parameters or energy"));
    }
    let n = problem.n_qubits();
    let reference = problem.register_reference();
    let entries = indices
        .iter()
        .try_fold(problem.hamiltonian.terms.len(), |total, &i| {
            total
                .checked_add(problem.pool[i].operator.terms.len())
                .and_then(|t| t.checked_add(1))
                .ok_or(Error::Overflow)
        })?;
    let _build = ctx.reserve(product(&[
        entries.checked_add(32).ok_or(Error::Overflow)?,
        n.checked_add(512).ok_or(Error::Overflow)?,
    ])?)?;
    let (operators, metadata) = if let Some(hva) = &problem.hva {
        let expanded = hva.expand(&problem.pool, indices, parameters)?;
        let operators=expanded.operators.iter().zip(&expanded.parameters).enumerate().map(|(i,(op,t))|json!({"label":format!("HVA-stage({i})"),"kind":"hamiltonian-group","parameter":t,"generator":pauli_payload(op)})).collect::<Vec<_>>();
        (
            operators,
            json!({"engine":"rust","hva":{"indices":indices,"parameters":parameters,"trotter":hva.trotter}}),
        )
    } else {
        (indices.iter().zip(parameters).map(|(&i,&t)|json!({"label":problem.pool[i].label,"kind":"fermionic","parameter":t,"generator":pauli_payload(&problem.pool[i].operator)})).collect::<Vec<_>>(),json!({"engine":"rust"}))
    };
    let mut metadata = metadata;
    if problem.taper.is_some() {
        metadata["taper"] = taper_record(problem);
    }
    let payload = json!({"format":CHECKPOINT,"version":3,"energy_unit":"Ha","n_qubits":n,"mapping":problem.encoding.mapping,"num_particles":problem.encoding.particles,"n_spatial_orbitals":problem.encoding.modes/2,"reference_qubits":(0..n).filter(|q| reference & (1<<(n-1-q))!=0).collect::<Vec<_>>(),"preparation":if problem.hva.is_some() || product_form {"product"} else {"sum"},"method":method,"energy":energy,"operators":operators,"hamiltonian":pauli_payload(&problem.hamiltonian),"status":status,"metadata":metadata});
    Document::parse(ctx, &payload.to_string(), CHECKPOINT)
}
/// A checkpoint of a subspace calculation: the shared ansatz, plus the state weights and the
/// sorted state energies (`subspace`). `energy` is the lowest state's.
#[allow(clippy::too_many_arguments)]
pub fn subspace_checkpoint(
    ctx: &Context,
    problem: &CachedProblem,
    indices: &[usize],
    parameters: &[f64],
    adapt: bool,
    product_form: bool,
    weights: &[f64],
    energies: &[f64],
    status: Value,
) -> Result<Document> {
    if weights.is_empty()
        || weights.len() != energies.len()
        || energies.iter().chain(weights).any(|v| !v.is_finite())
        || problem.hva.is_some()
    {
        return Err(invalid(
            "a subspace checkpoint needs one finite weight and energy per state and no HVA ansatz",
        ));
    }
    let method = if adapt { "subspace-adapt-vqe" } else { "subspace-vqe" };
    let mut doc = checkpoint(ctx, problem, indices, parameters, product_form, method, status, energies[0])?;
    doc.payload["subspace"] = json!({"weights": weights, "energies": energies});
    Ok(doc)
}
/// Same-directory create-new staging, flush/sync before atomic replacement.
pub fn atomic_write(path: &Path, write: impl FnOnce(&mut dyn Write) -> Result<()>) -> Result<()> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|e| invalid(format!("cannot create snapshot directory: {e}")))?;
    let name = path
        .file_name()
        .ok_or_else(|| invalid("snapshot path has no filename"))?;
    let mut stage = None;
    for _ in 0..128 {
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                stage = Some((candidate, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(invalid(format!("cannot stage snapshot: {e}"))),
        }
    }
    let (staging, file) =
        stage.ok_or_else(|| invalid("cannot allocate unique snapshot staging path"))?;
    let result = (|| {
        let mut out = BufWriter::new(file);
        write(&mut out)?;
        out.flush().map_err(|e| invalid(e.to_string()))?;
        out.get_ref()
            .sync_all()
            .map_err(|e| invalid(e.to_string()))?;
        drop(out);
        fs::rename(&staging, path).map_err(|e| invalid(format!("cannot replace snapshot: {e}")))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}
