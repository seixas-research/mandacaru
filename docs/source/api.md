# API reference

`Mandacaru` is an ASE `Calculator` whose numerical implementation runs in Rust.
Attach it with `atoms.calc = Mandacaru(...)`, or call `run` with a standalone
`Geometry`. The package exports seven names: `Mandacaru`, `estimate_qubits`,
`Geometry`, `CalculationResources`, `Viewer3D`, `Camera` and `__version__`. The
result class `CalculationResult` is what `run` returns.

(options-reference)=
## Mandacaru and its options

The class docstring below is the options reference: one entry per constructor
parameter, with its default. The methods the calculator builds, and what each can
do, are in [the methods table](#methods-table); what is not implemented is in
[the not-yet-available table](#not-yet-available).

```{eval-rst}
.. autoclass:: mandacaru.calculation.Mandacaru
   :members:
```

## Qubit counts without a calculation

```{eval-rst}
.. autofunction:: mandacaru.estimate_qubits
```

## Geometry and resources

```{eval-rst}
.. autoclass:: mandacaru.structure.Geometry
   :members:

.. autoclass:: mandacaru.resources.CalculationResources
   :members:
```

## Results

`run` returns a `CalculationResult`; the calculator keeps the last one in
`calc.result`.

```{eval-rst}
.. autoclass:: mandacaru.results.CalculationResult
   :members:
```

The result's `hamiltonian` property explicitly requests dense transformed
integrals. Reading energies, forces or ordinary snapshots leaves that Hamiltonian
unconstructed. Mean-field exchange uses Coulomb Cholesky factors.

For molecular calculations, `natural_orbitals` returns occupations and orbital
coefficients computed in Rust. `volumetric_field` samples the density, the
reference density, the correlation difference density or a natural orbital.
`write_cube` and `Viewer3D` consume these native fields. Fields include frozen
occupied orbitals and exclude deleted virtual orbitals in an active-space run.
Ordinary mean-field fields do not construct four-index integrals. PAW fields
contain smooth valence values; `augmentation_charge` reports the on-site charge
absent from the sampled density. Cube comments identify this separation.

## Visualization

```{eval-rst}
.. autoclass:: mandacaru.Viewer3D
   :members:

.. autoclass:: mandacaru.Camera
   :members:
```

## Native boundary

The compiled extension `mandacaru._native` is the only numerical implementation.
Its signatures are recorded in the stub `src/mandacaru/_native.pyi`, which must
match the bindings in `crates/mandacaru-python`. The rules at the boundary:

- `Context` owns checked allocations and the execution threads. Native calls use
  Hartree and Bohr; the Python layer converts from ASE's angstrom and eV.
- Input arrays are aligned, C-contiguous NumPy arrays of the declared dtype. Rust
  copies and validates them into owned buffers before releasing the interpreter.
  Returned arrays own independent storage; no borrowed or writable Rust view is
  exposed.
- `Context.periodic_problem(metadata_json, arrays)` takes small JSON metadata whose
  strings refer to a dictionary of float64, complex128 and uint64 arrays. Bulk
  numerical values travel through that dictionary. A JSON crystal file is an
  explicit persistence format, read with `Context.load_crystal`.
- `MolecularIntegrals.snapshot(include_hamiltonian=False)` and
  `MeanField.snapshot(include_hamiltonian=False)` export ordinary state without
  building four-index terms; the flag requests them. `MolecularIntegrals.has_eri`
  reports cached Coulomb factors and `MeanField.has_model` a cached transformed
  Hamiltonian.
- Parquet files are read and written by the extension alone, with no Python
  Parquet package: `mandacaru._native.read_parquet`, `read_parquet_metadata` and
  `write_parquet` take a `Context` first, and `Snapshot.save_parquet` writes a
  Hamiltonian ([Parquet files](#native-parquet)).
- An active-space model exposes `orbital_rotation`, `selection_occupations` and
  `mp2_correlation_energy`.

[The architecture page](development/architecture.md) states the dependency rule
between the crates and the boundary rules for contributors.
