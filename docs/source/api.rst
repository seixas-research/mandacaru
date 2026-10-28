API reference
=============

``Mandacaru`` is the Rust-only calculation interface. ``run`` accepts
``Geometry`` without ASE. The optional ``mandacaru.integrations.ase.ASECalculator``
adapter implements the ASE Calculator protocol. There is no ``engine`` or
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
