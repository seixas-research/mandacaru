API Reference
=============

This page documents the public modules, classes, and functions of the **Mandacaru** package.


Algorithms & Drivers
--------------------

.. automodule:: mandacaru.algorithms
   :members:
   :undoc-members:
   :show-inheritance:

Interaction Energies
~~~~~~~~~~~~~~~~~~~~

``E(complex) - sum E(fragments)`` with every energy on one shared grid.  See
:doc:`guide/interaction_energy`.

.. automodule:: mandacaru.algorithms.interaction
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

Pseudopotential Forces
~~~~~~~~~~~~~~~~~~~~~~

Hellmann-Feynman and Pulay forces for the PAW and ONCVPSP families.  See
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

.. automodule:: mandacaru.utils.cube
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
   :exclude-members: generate_pseudopotential, PseudoPotential, Channel, pseudize_channel, check_channel, report

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
represent -- the cure for the egg-box, on by default for the PAW families.
See :doc:`guide/pseudopotentials`.

.. automodule:: mandacaru.basis.filtering
   :members:
   :undoc-members:
   :show-inheritance:

----

Pseudopotentials
----------------

The valence-only families selected as basis names -- ``"NCPP"``
(Troullier-Martins), ``"ONCVPSP"`` (Hamann) and ``"PAW"`` (Bloechl) -- their
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

Confined Orbitals
~~~~~~~~~~~~~~~~~

The ``energy_shift`` of a PAW basis: the first zeta solved in a smooth
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

.. automodule:: mandacaru.pseudopotentials.link_library
   :members: link_library, status
   :undoc-members:

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
