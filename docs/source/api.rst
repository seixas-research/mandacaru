API reference
=============

``Mandacaru`` is an ASE Calculator whose numerical implementation runs in Rust.
Attach it directly with ``atoms.calc = Mandacaru(...)``. ``run`` also accepts
standalone ``Geometry`` values. There is no ``engine`` or
``engine_options`` argument.

.. autoclass:: mandacaru.calculation.Mandacaru
   :members:

.. autoclass:: mandacaru.structure.Geometry
   :members:

.. autoclass:: mandacaru.resources.CalculationResources
   :members:

.. autoclass:: mandacaru.results.CalculationResult
   :members:

The result's ``hamiltonian`` property explicitly requests dense transformed
integrals. Reading energies, forces or ordinary snapshots leaves that Hamiltonian
unconstructed. Mean-field exchange uses Coulomb Cholesky factors.

``active_space={"method": "mp2", "orbitals": 6}`` selects closed-shell HF
frozen natural orbitals in Rust. ``threshold`` selects by occupation;
``frozen`` and occupied/virtual counts control the frozen core and register.
Selection constructs only the reduced Hamiltonian. Its native model exposes
``orbital_rotation``, ``selection_occupations`` and ``mp2_correlation_energy``.
Density views retain the selected rotation. Open-shell MP2 selection, DLPNO,
symmetry expansion and correlating-pair constraints remain unavailable.

For molecular calculations, ``natural_orbitals`` returns occupations
and orbital coefficients computed in Rust. ``volumetric_field`` samples the
density, reference density, correlation difference density or a natural orbital.
``write_cube`` and ``Viewer3D`` consume these native fields. Fields include frozen
occupied orbitals and exclude deleted virtual orbitals in an active-space run.
Ordinary mean-field fields do not construct four-index integrals.
PAW fields contain smooth valence values; ``augmentation_charge`` reports the
on-site charge absent from the sampled density. Cube comments identify this
separation, and nuclear element labels remain distinct from valence charges.

Native boundary
---------------

``mandacaru._native.Context`` owns checked allocations and execution threads.
Native calls use Hartree and Bohr. Input arrays are aligned, C-contiguous NumPy
arrays of the declared dtype; Rust copies them into owned buffers before
releasing the interpreter. Returned arrays own independent storage.

``Context.periodic_problem(metadata_json, arrays)`` accepts small JSON metadata
with string references to a dictionary of float64, complex128 and uint64 arrays.
Bulk numerical values travel through that dictionary. JSON crystal files are
an explicit persistence format, loaded with ``Context.load_crystal``.

``MolecularIntegrals.snapshot(include_hamiltonian=False)`` and
``MeanField.snapshot(include_hamiltonian=False)`` export ordinary state without
building four-index terms. The explicit flag requests them.
``MolecularIntegrals.has_eri`` reports cached Coulomb factors;
``MeanField.has_model`` reports a cached transformed Hamiltonian.

Scientific coverage and atomic-unit kernel entry points are described in
:doc:`guide/native_engine`, :doc:`guide/dft` and :doc:`development/architecture`.
The source stub ``src/mandacaru/_native.pyi`` records binding signatures.
