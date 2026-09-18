API Reference
=============

This page documents the public modules, classes, and functions of the **Carcará** package.


Algorithms & Drivers
--------------------

.. automodule:: carcara.algorithms
   :members:
   :undoc-members:
   :show-inheritance:

Interaction Energies
~~~~~~~~~~~~~~~~~~~~

``E(complex) - sum E(fragments)`` with every energy on one shared grid.  See
:doc:`guide/interaction_energy`.

.. automodule:: carcara.algorithms.interaction
   :members:
   :undoc-members:
   :show-inheritance:

Dry Run
~~~~~~~

The qubit estimate of a calculation, made without integrals, mapping or
circuits.  See :doc:`guide/dry_run`.

.. automodule:: carcara.algorithms.dry_run
   :members:
   :undoc-members:
   :show-inheritance:

Pseudopotential Forces
~~~~~~~~~~~~~~~~~~~~~~

Hellmann-Feynman and Pulay forces for the PAW and ONCVPSP families.  See
:doc:`guide/pseudopotentials`.

.. automodule:: carcara.algorithms.pseudo_forces
   :members:
   :undoc-members:
   :show-inheritance:

Quantum Phase Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

QPE from a checkpointed variational state, with the memory check that guards
its state-vector simulation.  See :doc:`guide/checkpoints_qpe`.

.. automodule:: carcara.algorithms.qpe
   :members:
   :undoc-members:
   :show-inheritance:

Wavefunction Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm-agnostic on-disk form of a variational state.  See
:doc:`guide/checkpoints_qpe`.

.. automodule:: carcara.core.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

Command Line
~~~~~~~~~~~~

.. automodule:: carcara.cli
   :members: main, build_parser, load_geometry, parse_cell, solver_options
   :undoc-members:

----

Basis Sets
----------

.. automodule:: carcara.basis
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: generate_pseudopotential, PseudoPotential, Channel, pseudize_channel, check_channel, report

Named Gaussian Families
~~~~~~~~~~~~~~~~~~~~~~~

The Pople, Dunning and Karlsruhe basis sets, generated from the structure
their names encode.  See :doc:`guide/basis_sets`.

.. automodule:: carcara.basis.gaussian_families
   :members:
   :undoc-members:
   :show-inheritance:

All-Electron Numerical Atomic Orbitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``NAO-AE`` family -- the LDA atom's own shells under a smooth wall plus
hydrogen-like tiers sized from the atom.  See :doc:`guide/nao_ae`.

.. automodule:: carcara.basis.nao_ae
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

.. automodule:: carcara.pseudopotentials
   :members:
   :undoc-members:
   :show-inheritance:

Families and Registry
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: carcara.pseudopotentials.families
   :members:
   :undoc-members:
   :show-inheritance:

Library Files
~~~~~~~~~~~~~

.. automodule:: carcara.pseudopotentials.io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: carcara.pseudopotentials.link_library
   :members: link_library, status
   :undoc-members:

----

Integral Engine
---------------

.. automodule:: carcara.integrals
   :members:
   :undoc-members:
   :show-inheritance:

----

Fermionic Operators & Mappings
------------------------------

.. automodule:: carcara.core
   :members:
   :undoc-members:
   :show-inheritance:

Particle-Number Sectors
~~~~~~~~~~~~~~~~~~~~~~~

Operators and states restricted to the determinants with a fixed number of
alpha and beta electrons.

.. automodule:: carcara.core.sector
   :members:
   :undoc-members:
   :show-inheritance:

----

Quantum Circuits & Ansätze
--------------------------

.. automodule:: carcara.circuits
   :members:
   :undoc-members:
   :show-inheritance:

----

Classical Optimisers
--------------------

.. automodule:: carcara.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

----

Hamiltonian Serialisation
-------------------------

The on-disk qubit-Hamiltonian cache (Apache Parquet or JSON) that lets a run skip
the integrals and the fermion-to-qubit mapping entirely.  See
:doc:`guide/hamiltonian_cache`.

.. automodule:: carcara.core.serialization
   :members:
   :undoc-members:
   :show-inheritance:

----

Backends: Devices & Circuit Providers
-------------------------------------

.. automodule:: carcara.backends

Device Registry
~~~~~~~~~~~~~~~

Which machine a run executes on -- the ideal simulator, or an Amazon Braket
simulator or QPU.  See :doc:`guide/aws_braket`.

.. automodule:: carcara.backends.hardware
   :members:
   :undoc-members:
   :show-inheritance:

Circuit Providers
~~~~~~~~~~~~~~~~~

Which SDK builds and executes the ansatz circuits -- Qiskit, Amazon Braket or
Cirq.  See :doc:`guide/backends`.

.. automodule:: carcara.backends.providers
   :members:
   :undoc-members:
   :show-inheritance:

Shot-Based Measurement
~~~~~~~~~~~~~~~~~~~~~~

Estimating :math:`\langle H \rangle` from measurement shots via qubit-wise
commuting Pauli groups -- the protocol a real QPU requires.

.. automodule:: carcara.backends.measurement
   :members:
   :undoc-members:
   :show-inheritance:

----

Utilities & Profiling
---------------------

.. automodule:: carcara.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: carcara.utils.dumps
   :members:
   :undoc-members:
   :show-inheritance:
