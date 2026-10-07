API Reference
=============

This page documents the public modules, classes, and functions of the **Mandacaru** package.


Algorithms & Drivers
--------------------

.. automodule:: mandacaru.algorithms
   :members:
   :undoc-members:
   :show-inheritance:

Convergence Criteria
~~~~~~~~~~~~~~~~~~~~

When an adaptive solver stops growing its ansatz: the ``convergence`` option,
``{"gradient": ..., "energy": ...}``.

.. automodule:: mandacaru.algorithms.convergence
   :members:
   :undoc-members:
   :show-inheritance:

Ansatz Templates
~~~~~~~~~~~~~~~~

The ``ansatz=`` option of ``method="vqe"``: a name or a dictionary choosing a
circuit template, ``"uccsd"`` (the default) or ``"hva"``, and its options.

.. automodule:: mandacaru.algorithms.ansatz_spec
   :members:
   :undoc-members:
   :show-inheritance:

Markov Chain Ansatz Search
~~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture chain behind ``method="mcas-vqe"``: the four moves, their summed
proposal probabilities, the Metropolis-Hastings test and the temperature
schedule.  See :doc:`tutorial/mcas_vqe`.

.. automodule:: mandacaru.algorithms.mcas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.algorithms.orbital_tracking
   :members: OrbitalMatch, OrbitalSnapshot, orbital_overlap,
             match_orbitals, occupation_blocks, transfer_ansatz

Learned Proposals
~~~~~~~~~~~~~~~~~

The operator proposal behind ``method="valqa"`` -- a graph neural network
over the qubit Hamiltonian that scores the pool's operators -- and the edit
store it is trained on.  See :doc:`tutorial/valqa`.

.. automodule:: mandacaru.algorithms.proposal_model
   :members: fit, ProposalModel, HamiltonianGraph, assess_data_volume,
             fit_gnn, GNNModel, gnn_scores, PairwiseRanker,
             fit_pairwise_ranker, fit_gp, compress

.. automodule:: mandacaru.algorithms.proposal_data
   :members: ProblemRecord, EditRecorder, load_edits

Interaction Energies
~~~~~~~~~~~~~~~~~~~~

``E(complex) - sum E(fragments)`` with every energy on one shared grid.  See
:doc:`guide/interaction_energy`.

.. automodule:: mandacaru.algorithms.interaction
   :members:
   :undoc-members:
   :show-inheritance:

Polarizability
~~~~~~~~~~~~~~

A molecule in a uniform electric field (``Mandacaru(electric_field=...)``)
and its static polarizability by finite fields.  See
:doc:`guide/polarizability`.

.. automodule:: mandacaru.algorithms.polarizability
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Berry-Phase Polarization
~~~~~~~~~~~~~~~~~~~~~~~~

The electric polarization of an insulating crystal and its Born effective
charges.  See :doc:`guide/dft`.

.. automodule:: mandacaru.algorithms.berry_phase
   :members:
   :undoc-members:
   :show-inheritance:

Wannier Functions
~~~~~~~~~~~~~~~~~

Maximally localized Wannier functions of a crystal's bands.  See
:doc:`guide/dft`.

.. automodule:: mandacaru.algorithms.wannier
   :members:
   :undoc-members:
   :show-inheritance:

Choosing the Target Space
~~~~~~~~~~~~~~~~~~~~~~~~~

The Bloch states a Wannierization is built from, chosen by their
projectability onto atomic orbitals or by the RPA natural orbitals.  See
:doc:`guide/target_space`.

.. automodule:: mandacaru.algorithms.band_selection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.algorithms.rpa_density
   :members:
   :undoc-members:
   :show-inheritance:

Dry Run
~~~~~~~

The qubit estimate of a calculation, made without integrals, mapping or
circuits.  See :doc:`guide/dry_run`.

.. automodule:: mandacaru.algorithms.dry_run
   :members:
   :undoc-members:
   :show-inheritance:

Active Space
~~~~~~~~~~~~

Which spatial orbitals reach the qubit register, and how the virtuals are
ranked.  See :doc:`guide/active_space`.

.. automodule:: mandacaru.algorithms.active_space
   :members:
   :undoc-members:
   :show-inheritance:

Moller-Plesset Perturbation Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second-order correlation energy and the frozen natural orbitals of its density,
used to select an active space.  See :doc:`guide/active_space`.

.. automodule:: mandacaru.algorithms.mp2
   :members:
   :undoc-members:
   :show-inheritance:

Local Correlation and DLPNO-MP2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Foster-Boys orbitals, projected atomic orbitals and domains, the
integral-direct DLPNO-MP2 behind the ``"dlpno-mp2"`` active-space method, and
the orbital-integral providers it reads.  See :doc:`guide/active_space`.

.. automodule:: mandacaru.algorithms.dlpno_mp2
   :members:

.. automodule:: mandacaru.algorithms.local_correlation
   :members:

.. automodule:: mandacaru.algorithms.orbital_integrals
   :members:

.. automodule:: mandacaru.integrals.direct
   :members:

Orbital Symmetry
~~~~~~~~~~~~~~~~

Molecular point groups and how the molecular orbitals transform under them,
used by the active space's ``symmetry``.  See :doc:`guide/active_space`.

.. automodule:: mandacaru.algorithms.orbital_symmetry
   :members:
   :undoc-members:
   :show-inheritance:

Z2 Symmetry Tapering
~~~~~~~~~~~~~~~~~~~~

One qubit removed per conserved parity of the Hamiltonian, found from its Pauli
terms rather than from a declared point group.  See :doc:`guide/tapering`.

.. automodule:: mandacaru.core.tapering
   :members:
   :undoc-members:
   :show-inheritance:

Pseudopotential Forces
~~~~~~~~~~~~~~~~~~~~~~

Hellmann-Feynman and Pulay forces for the PAW-LCAO family.  See
:doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.algorithms.pseudo_forces
   :members:
   :undoc-members:
   :show-inheritance:

Volumetric Output
~~~~~~~~~~~~~~~~~

One-particle pictures of the many-body state -- electron density, spin density,
correlation density and natural orbitals -- on the calculation's real-space
grid.  See :doc:`guide/visualization`.

.. automodule:: mandacaru.algorithms.volumetric
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.algorithms.charges
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.cube
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.bxsf
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.viewer
   :members:
   :undoc-members:
   :show-inheritance:

Quantum Phase Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

QPE from a checkpointed variational state, with the memory check that guards
its state-vector simulation.  See :doc:`guide/checkpoints_qpe`.

.. automodule:: mandacaru.algorithms.qpe
   :members:
   :undoc-members:
   :show-inheritance:

Quantum Echoes and Time Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weak dipole echoes, nested Pauli OTOCs, and matrix-free Suzuki–Trotter
propagation from prepared states. See :doc:`guide/quantum_echoes`.

.. automodule:: mandacaru.algorithms.quantum_echoes
   :members:
   :no-index:

.. automodule:: mandacaru.algorithms.nested_otoc
   :members:
   :undoc-members:
   :no-index:

.. automodule:: mandacaru.algorithms.time_evolution
   :members: SuzukiTrotter, time_evolve

.. automodule:: mandacaru.core.dipole
   :members:

Wavefunction Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm-agnostic on-disk form of a variational state.  See
:doc:`guide/checkpoints_qpe`.

.. automodule:: mandacaru.core.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

Command Line
~~~~~~~~~~~~

.. automodule:: mandacaru.cli
   :members: main, build_parser, load_geometry, parse_cell, solver_options
   :undoc-members:

----

Basis Sets
----------

.. automodule:: mandacaru.basis
   :members:
   :undoc-members:
   :show-inheritance:

Named Gaussian Families
~~~~~~~~~~~~~~~~~~~~~~~

The Pople, Dunning and Karlsruhe basis sets, generated from the structure
their names encode.  See :doc:`guide/basis_sets`.

.. automodule:: mandacaru.basis.gaussian_families
   :members:
   :undoc-members:
   :show-inheritance:

All-Electron Numerical Atomic Orbitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``NAO-AE`` family -- the LDA atom's own shells under a smooth wall plus
hydrogen-like tiers sized from the atom.  See :doc:`guide/nao_ae`.

.. automodule:: mandacaru.basis.nao_ae
   :members:
   :undoc-members:
   :show-inheritance:

Fourier Filtering
~~~~~~~~~~~~~~~~~

Removing from a radial function the wave-vectors the real-space grid cannot
represent -- the cure for the egg-box, on by default for the PAW-LCAO families.
See :doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.basis.filtering
   :members:
   :undoc-members:
   :show-inheritance:

----

Pseudopotentials
----------------

The valence-only families selected as basis names -- ``"PAW-LCAO"`` (Bloechl) and
``"UPAW-LCAO"`` (Ivanov et al.) -- their
registry, generation, library and the pseudo-atomic orbitals.  See
:doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.pseudopotentials
   :members:
   :undoc-members:
   :show-inheritance:

Families and Registry
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mandacaru.pseudopotentials.families
   :members:
   :undoc-members:
   :show-inheritance:

Radial Partial-Wave Machinery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Numerov reference waves, optimized smooth partial waves and ghost checks
shared by PAW-LCAO and UPAW-LCAO.  See :doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.pseudopotentials.partial_waves
   :members:
   :show-inheritance:

Confined Orbitals
~~~~~~~~~~~~~~~~~

The ``energy_shift`` of a pseudopotential basis (PAW-LCAO, UPAW-LCAO):
the first zeta solved in a smooth
confining potential, its cutoff radius fixed by the eigenvalue shift.  See
:doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.pseudopotentials.confinement
   :members:
   :show-inheritance:

Library Files
~~~~~~~~~~~~~

.. automodule:: mandacaru.pseudopotentials.io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.pseudopotentials.environment
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: LibraryPathError

----

Integral Engine
---------------

.. automodule:: mandacaru.integrals
   :members:
   :undoc-members:
   :show-inheritance:

----

Fermionic Operators & Mappings
------------------------------

.. automodule:: mandacaru.core
   :members:
   :undoc-members:
   :show-inheritance:

Particle-Number Sectors
~~~~~~~~~~~~~~~~~~~~~~~

Operators and states restricted to the determinants with a fixed number of
alpha and beta electrons.

.. automodule:: mandacaru.core.sector
   :members:
   :undoc-members:
   :show-inheritance:

Matrix-Free Operators
~~~~~~~~~~~~~~~~~~~~~

Qubit operators applied to a state vector as Pauli strings, with no matrix
stored.  See :doc:`guide/scalability`.

.. automodule:: mandacaru.core.matrix_free
   :members:
   :undoc-members:
   :show-inheritance:

Pauli Algebra
~~~~~~~~~~~~~

Products, commutators and the fermion-to-qubit map on symplectic bit tables,
at any register width.  See :doc:`guide/scalability`.

.. automodule:: mandacaru.core.pauli_algebra
   :members:
   :undoc-members:
   :show-inheritance:

----

Quantum Circuits & Ansätze
--------------------------

.. automodule:: mandacaru.circuits
   :members:
   :undoc-members:
   :show-inheritance:

----

Classical Optimizers
--------------------

.. automodule:: mandacaru.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

----

Hamiltonian Serialization
-------------------------

The on-disk qubit-Hamiltonian cache (Apache Parquet or JSON) that lets a run skip
the integrals and the fermion-to-qubit mapping entirely.  See
:doc:`guide/hamiltonian_cache`.

.. automodule:: mandacaru.core.serialization
   :members:
   :undoc-members:
   :show-inheritance:

----

Backends: Devices & Circuit Providers
-------------------------------------

.. automodule:: mandacaru.backends

Device Registry
~~~~~~~~~~~~~~~

Which machine a run executes on -- the ideal simulator, or an Amazon Braket
simulator or QPU.  See :doc:`guide/aws_braket`.

.. automodule:: mandacaru.backends.hardware
   :members:
   :undoc-members:
   :show-inheritance:

Circuit Providers
~~~~~~~~~~~~~~~~~

Which SDK builds and executes the ansatz circuits -- Qiskit, Amazon Braket or
Cirq.  See :doc:`guide/backends`.

.. automodule:: mandacaru.backends.providers
   :members:
   :undoc-members:
   :show-inheritance:

Shot-Based Measurement
~~~~~~~~~~~~~~~~~~~~~~

Estimating :math:`\langle H \rangle` from measurement shots via qubit-wise
commuting Pauli groups -- the protocol a real QPU requires.

.. automodule:: mandacaru.backends.measurement
   :members:
   :undoc-members:
   :show-inheritance:

----

Utilities & Profiling
---------------------

.. automodule:: mandacaru.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.dumps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.bibliography
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.citations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mandacaru.utils.shell_config
   :members:
   :undoc-members:
   :show-inheritance:
