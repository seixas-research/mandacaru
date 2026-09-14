.. raw:: html

   <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">

     <!-- LOGO -->
     <a align="center" href="https://carcara.readthedocs.io/">
       <img src="_static/logo_light.png" width="800px" class="only-light">
       <img src="_static/logo_dark.png" width="800px" class="only-dark">
     </a>

     <!-- BADGES -->
    <p>
       <!-- License -->
       <a href="https://github.com/seixas-research/carcara/blob/main/LICENSE"> 
         <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="license">
       </a>
       <!-- PyPI -->
       <a href="https://pypi.org/project/carcara/">
         <img src="https://img.shields.io/pypi/v/carcara.svg?style=for-the-badge" alt="pypi">
       </a>

      <!-- documentation -->
       <a href="https://carcara.readthedocs.io/en/latest/">
         <img src="https://readthedocs.org/projects/carcara/badge/?version=latest&style=for-the-badge" alt="docs">
       </a>


    </p>

   </div>

**Carcará** is an end-to-end Python framework for fermionic quantum simulations based on variational quantum algorithms (VQAs). The library is engineered to bridge the gap between classical quantum chemistry methods and NISQ-era (Noisy Intermediate-Scale Quantum) physical devices.


.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   GitHub <https://github.com/seixas-research/carcara>
   PyPI <https://pypi.org/project/carcara/>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation
   tutorial/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-To Guides

   guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api


Architecture & Philosophy
=========================

The design of Carcará is built around **loose coupling** and **strict unit boundaries**:

**Data Pipeline:**
  ASE Atoms/XYZ Geometry
    ↳ Grid Configuration & Basis Set Factory
      ↳ Integral Engine (One- & Two-body integrals)
        ↳ Molecular Hamiltonian (Fermionic Operators)
          ↳ Fermion-to-Qubit Mappings (PauliSum qubit Hamiltonian)
            ↳ *(optional)* Parquet/JSON cache — replay later, skipping everything above
              ↳ Carcara (``method=`` selects ADAPT-VQE (default) / VQE / subspace variants; parameterized circuit compiled to a backend & optimized classically)
                ↳ Execution: internal state vector, or Qiskit / Amazon Braket / Cirq circuits (simulator or QPU)

1. Basis-Agnostic Integrals
---------------------------

The real-space integration engine (``carcara.integrals``) has no analytical dependency on the orbitals. It interacts with the basis sets solely through the ``BasisFunction`` interface:

* A basis function must only implement an ``evaluate(x, y, z)`` method returning the orbital's amplitude at any coordinate.
* The engine samples the orbital onto a uniform orthogonal (or non-orthogonal) grid and handles the kinetic and Coulomb integrals numerically.
* This makes it easy to swap Gaussian-type orbitals (GTOs) with Confined Numerical Atomic Orbitals (NAOs) or Wannier functions without changing a single line in the core integral kernels.

2. Standardized Unit Conventions
--------------------------------

To avoid the cognitive load of switching units between physics calculations and chemistry inputs:

* **Internal Core:** The mathematical core—orbital evaluations, grid coordinates, Poisson solvers, and C-backend integrals—operates strictly in **atomic units** (lengths in Bohr, energies in Hartree).
* **User Boundary:** All user-facing APIs (``Grid``, ``FullAtomicOrbital``, ``IntegralEngine``) default to standard chemistry units: **lengths in Ångström** and **energies in electronvolts (eV)**. Conversions happen automatically at the API boundaries (configured via ``units`` / ``energy_units``).
* **Everything you see is eV / Å:** every result a driver returns -- ``optimal_energy``, ``reference_energy``, the energy histories, ``EnergyLevels``, the subspace results, ``InteractionEnergy``, ``ForceResult`` (eV/Å), band structures, hardware measurements (``measured_energy`` / ``measure_energies``) -- every verbose printout and the ``carcara`` command line are in **eV and Ångström**. Result objects record their unit (``energy_unit``) and offer ``in_units("Ha")`` for the atomic-unit view. Hartree and Bohr survive only inside the internal layers (the integral engine, the qubit Hamiltonian's Pauli coefficients and its cache files, the basis and pseudopotential records, the SCF and gradient mathematics). ``atomic_units=True`` on any driver is the single opt-in that switches its outputs to Hartree / Bohr.

3. SDK-Agnostic Circuits
------------------------

The ansatz layer is not tied to one quantum SDK. Every generator is an anti-Hermitian ``PauliSum`` whose terms commute, so :math:`e^{\theta A}` factorizes **exactly** into Pauli rotations. A single provider-independent gate stream (``X``, ``H``, ``S``, ``S^\dagger``, ``CNOT``, ``R_z``) is then translated by :mod:`carcara.backends.providers` into Qiskit, Amazon Braket or Cirq circuits.

* Because the decomposition is exact (no Trotter error), all three SDKs produce the *same* unitary and agree with the internal NumPy state vector to machine precision.
* The same abstraction reaches **real hardware**: with ``shots > 0`` the Braket provider measures :math:`\langle H \rangle` from qubit-wise commuting Pauli groups instead of reading amplitudes — the only protocol a QPU supports.

4. Reusable Hamiltonians
------------------------

Building the qubit Hamiltonian (integrals + mapping) is the most expensive stage of a run and is independent of the algorithm that follows. It can be serialized to **Apache Parquet or JSON** and replayed: ``load_hamiltonian=...`` skips the integral engine and the fermion-to-qubit transformation entirely, and — since the file also records the electron and orbital counts — needs no geometry at all.

----

Current Development State
=========================

Carcará is currently mid-build. The core physical and simulation pipelines are fully implemented and validated. **Execution on real quantum hardware is available** through Amazon Braket (shot-based measurement of :math:`\langle H \rangle`); error mitigation remains a design stub.

.. list-table::
   :header-rows: 1
   :widths: 20 20 45 15

   * - Module
     - Purpose
     - Implemented Capabilities
     - Status
   * - **carcara.basis**
     - Orbital generation
     - FAO (analytic hydrogenic), NAO (numerical confinement, multiple-zeta and polarized), NAO-AE (all-electron numerical orbitals under a smooth wall with hydrogen-like tiers), STO-3G to STO-6G, and the named Gaussian families -- Pople (3-21G ... 6-311+G(2df,2p)), Dunning (cc-pVDZ ... cc-pV5Z, aug-cc-pVDZ, cc-pCVDZ) and Karlsruhe (def2-SV(P) ... def2-QZVPPD) -- generated from the structure their names encode, with the published shell structure and function count. Exponents fitted from scratch via least-squares.
     - **Complete**
   * - **carcara.integrals**
     - Integral evaluation
     - Uniform/anisotropic orthogonal & non-orthogonal grid sampling, FFT Poisson solver, direct real-space sum, OpenMP-parallel C backend with numpy zero-copy ctypes fallback.
     - **Complete**
   * - **carcara.core**
     - Operator algebra & mappings
     - ``Fermion`` creation/annihilation algebra, Jordan-Wigner, Parity (with optional 2-qubit reduction), and Bravyi-Kitaev mappings; Parquet/JSON Hamiltonian serialization with automatic format detection.
     - **Complete**
   * - **carcara.circuits**
     - Parameterized circuits
     - Exact UCC unitary and Trotterized UCCSD circuits, single/double excitation gates, ADAPT operator pools (``fermionic``, ``qubit``, ``qeb``, ``ceo``); ansätze execute on any provider.
     - **Complete**
   * - **carcara.algorithms**
     - Solvers & profiling
     - The unified ``Carcara`` (solver selected with ``method=``, ADAPT-VQE by default) and periodic ``BlochCalculator`` front ends; exact state-vector VQE & ADAPT-VQE solvers, RHF/UHF molecular-orbital (MO) solvers, PQC expressibility trackers (KL-divergence vs. Haar), CNOT/depth compilers, the ``quenching`` parametrization policy, and the ``dry_run`` qubit estimate (also the ``carcara --dry-run`` command line).
     - **Complete**
   * - **carcara.optimizers**
     - Parameter optimization
     - Classical optimizer wrapper (SPSA, COBYLA, Nelder-Mead, SLSQP, Adam, L-BFGS-B) with evaluation counting and history tracking.
     - **Complete**
   * - **carcara.backends.providers**
     - Circuit construction & execution
     - Qiskit, Amazon Braket and Cirq providers sharing one exact Pauli-rotation decomposition; verified to agree to machine precision.
     - **Complete**
   * - **carcara.backends.hardware**
     - Device registry
     - Ideal simulator, IBM Quantum hardware via Qiskit Runtime (``ibm-quantum`` least-busy, any ``ibm_*`` processor, ``fake_*`` local rehearsals), Braket local & managed simulators (SV1/DM1/TN1), and the IonQ / IQM / Rigetti QPUs (or any Braket ARN).
     - **Complete**
   * - **carcara.backends.measurement**
     - Shot-based estimation
     - Qubit-wise commuting Pauli grouping, expectation values from bit-string counts, shot-noise estimates — the protocol a QPU requires.
     - **Complete**
   * - **Hardware execution**
     - Running on a QPU
     - Energy evaluation runs on Amazon Braket devices via the shot path. ADAPT-VQE's *pool-gradient screening* is still classical, so fixed-ansatz ``method="vqe"`` is the fully hardware-native method.
     - **Partial**
   * - **carcara.backends.mitigation**
     - Error mitigation
     - Zero-noise extrapolation (ZNE) and readout mitigation stubs.
     - **Stub (0 LOC)**

----

Detailed Component Walkthrough
==============================

carcara.basis
-------------

Carcará does not ship database tables of basis-set exponents. Instead, it generates all basis sets **from first principles**:

* **All-Electron Numerical Atomic Orbitals (NAO-AE):** every occupied shell of the self-consistent LDA atom (core included) re-solved under a smooth exponential-wall confinement, plus hydrogen-like polarization / diffuse / contracted *tiers* whose effective charges are derived from the atom's own valence radius; each :math:`l` channel is Gram-Schmidt orthonormalized. See :doc:`guide/nao_ae`.
* **Numerical Atomic Orbitals (NAOs):** Confinement is defined by an ``energy_shift`` parameter :math:`\delta E` (default 0.03 eV). The radial Schrödinger equation is solved numerically via finite differences inside a hard-wall sphere of radius :math:`r_c = \pi / \sqrt{2\delta E}` using a screened nuclear potential. A ``size`` argument selects **multiple-zeta and polarized** variants (``SZ``, ``DZ``, ``DZP``, ``TZP``, ``QZP``, ...): extra zetas come from the SIESTA split-valence construction, polarization from an :math:`l+1` shell solved in the same sphere. See :doc:`guide/basis_sets`.
* **Gaussian-Type Orbitals (GTOs):** Exponents and coefficients are computed by a least-squares fit to Slater-Type Orbitals (STOs) with exponents :math:`\zeta` determined by Slater's rules. A reference fit is cached for :math:`\zeta=1` and scaled dynamically by :math:`\zeta^2` for any target atom.
* **Named Gaussian families (Pople, Dunning, Karlsruhe):** a basis-set name such as ``6-311+G(2df,2p)``, ``aug-cc-pVTZ`` or ``def2-TZVP`` is parsed into its *structure* -- core contraction length, valence split, polarization, diffuse and core-correlating functions -- and generated for each atom from the same Slater-orbital fits: contracted and split-valence functions from one fit partitioned tightest-first, polarization exponents scaled from the valence Slater exponent, diffuse and tight functions placed by fixed ratios. The published shell structure and function count are reproduced (``cc-pVTZ`` carbon is ``[4s3p2d1f]``); the exponents are Carcará's own. See :doc:`guide/basis_sets`.

carcara.integrals
-----------------

The integral engine supports three grid geometries to model molecules and crystalline unit cells:

1. **Cubic Grid:** Equal boundaries and spacing :math:`h`. Optimized via a highly efficient 7-point finite difference Laplacian in the C backend.
2. **Orthorhombic Grid:** Different box sizes and spacings per coordinate axis.
3. **Non-Orthogonal Grid:** Full :math:`3 \times 3` voxel steps allowing calculation on skewed lattices. One-body matrices use a generalized finite-difference Laplacian:
   
   .. math::
      \nabla^2 f = \sum_{a,b} g^{ab} \frac{\partial^2 f}{\partial x_a \partial x_b}
   
   employing cross-stencil derivatives implemented in both C and NumPy.

The two-body electron-repulsion integrals are computed via:

* **FFT Method (``method="fft"``):** Evaluates the Poisson equation :math:`\nabla^2 \phi(\mathbf{r}) = -4\pi \rho(\mathbf{r})` on the grid using Fast Fourier Transforms (via ``numpy.fft``) to obtain the potential :math:`\phi`, then contracts it with other orbital densities.
* **Direct Method (``method="direct"``):** Direct real-space double summation over grid voxel coordinates, accelerated with OpenMP in C.

carcara.core
------------

Provides a lightweight second-quantized algebra framework. The ``Fermion`` class handles fermionic addition, multiplication, scaling, and index consolidation. The ``MolecularIntegrals`` class maps the :math:`h_{pq}` core and :math:`g_{pqrs}` ERI matrices to a ``Fermion`` representation.
Three mappings are available to project fermionic operators onto qubit states:

* **Jordan-Wigner:** Maps creation/annihilation operators to localized Pauli strings (:math:`X-iY` padded by :math:`Z` chains).
* **Parity:** Maps occupations to parity states, allowing a 2-qubit reduction by exploiting conservation of total particle number (:math:`Z_0 Z_1 \dots`) and spin parities.
* **Bravyi-Kitaev:** Uses a binary-tree scaling scheme to balance operator locality and weight.

All mappings are generated dynamically through binary transformation matrices.

carcara.circuits
----------------

Prepares parameterized quantum states. The ``UCCSD`` class generates Unitary Coupled-Cluster Singles and Doubles ansätze. By default, it computes the exact matrix exponent of the UCC generators, but ``trotter=True`` compiles a first-order Trotter circuit suitable for execution.
For adaptive VQE algorithms, four operator pools define the candidate generators (:math:`A_i`):

* ``fermionic``: Spin-adapted single and double excitations.
* ``qubit``: Translates each pool operator to individual qubit Pauli strings, maximizing parameter freedom.
* ``qeb`` (Qubit Excitation Basis): Drops Jordan-Wigner :math:`Z`-strings to keep CNOT networks shallow.
* ``ceo`` (Coupled-Exchange Operators): Exploits shared CNOT structures to implement excitations using fewer gates.

carcara.algorithms
------------------

All molecular runs go through the unified ASE calculator ``Carcara``, which selects the variational method with ``method=`` (``"adapt-vqe"`` -- the default -- ``"vqe"``, ``"subspace-vqe"``, ``"subspace-adapt-vqe"``); periodic systems go through ``BlochCalculator``, with the same default. Methods still under development live in ``carcara.experimental`` and are documented separately, outside this manual.

* **VQE:** Computes :math:`\langle \psi(\boldsymbol{\theta})| H |\psi(\boldsymbol{\theta}) \rangle` on exact state vectors, updating parameters until convergence.
* **ADAPT-VQE:** Calculates the commutator gradients :math:`\langle \psi | [H, A_i] | \psi \rangle` for all pool operators, selects the operator with the largest gradient, appends it to the ansatz, and performs a warm-started VQE optimization. Loop terminates when the maximum gradient falls below ``gradient_tolerance``.
* **Hartree-Fock (RHF/UHF):** Solves the self-consistent field equations to yield the molecular-orbital basis. Transforming the Hamiltonian to the MO basis is critical for ADAPT-VQE; it ensures that the starting Hartree-Fock reference is stationary, making single excitation gradients vanish and allowing the algorithm to focus on electron correlations. **Open-shell systems** -- any odd electron count, or a high-spin state requested through the geometry's ``magmoms`` -- are built in the natural-orbital basis of the unrestricted (UHF) solution, one spatial basis shared by both spins, so radicals run through the same solvers, pools and mappings (the reference is then not stationary, and ADAPT-VQE also picks up single excitations).
* **Expressibility:** Computes Kullback-Leibler divergences between random-ansatz state fidelities and the Haar distribution. Because fermionic ansätze conserve symmetries, expressibility is computed against the Haar distribution of the active space dimension:

  .. math::
     d = \binom{M}{n_\alpha}\binom{M}{n_\beta}

  rather than the full :math:`2^N` Hilbert space.
* **Dynamic parametrization (``quenching``):** ``True`` (default) re-optimizes every variational parameter at each iteration — standard ADAPT-VQE. ``False`` freezes the previously optimized angles and varies only the newest one, trading variational freedom for a one-dimensional line search per growth step. See :doc:`guide/quenching`.

carcara.backends
----------------

Two orthogonal registries plus the hardware measurement protocol:

* **Devices** (``backends.hardware``) — *where* a run executes. ``AER_simulator`` is the ideal state-vector default; the Amazon Braket entries cover the local simulator, the managed SV1/DM1/TN1 simulators, and the IonQ / IQM / Rigetti QPUs (any Braket ARN is also accepted verbatim). Naming a QPU without ``shots`` is rejected up front, since hardware cannot return amplitudes.
* **Circuit providers** (``backends.providers``) — *which SDK* builds and runs the circuits: ``"qiskit"``, ``"braket"`` or ``"cirq"``. Each generator's exponential factorizes exactly into commuting Pauli rotations, so one shared gate stream is translated per SDK and all three reproduce the internal state vector to machine precision.
* **Shot-based measurement** (``backends.measurement``) — on real hardware the energy must be *measured*, not read off. The Hamiltonian is partitioned into **qubit-wise commuting** groups (118 Pauli terms → 29 circuits for LiH), one circuit is run per group with the appropriate basis rotation, and :math:`\langle H \rangle` is assembled from the bit-string counts with error falling as :math:`1/\sqrt{\text{shots}}`.

carcara.core.serialization
--------------------------

The qubit Hamiltonian is serializable to **Apache Parquet** (compressed, columnar, queryable from pandas) or **JSON** (plain text, no native dependency), chosen with ``hamiltonian_format``. Loading detects the format automatically — from the extension, else from the file's leading bytes — and skips the integral engine and the fermion-to-qubit mapping entirely. Because the file also records ``num_particles`` and ``n_spatial_orbitals``, a reloaded calculator runs with no geometry at all. See :doc:`guide/hamiltonian_cache`.