
.. image:: _static/logo_light.png
   :alt: Mandacaru
   :class: only-light
   :width: 640
   :align: center

.. image:: _static/logo_dark.png
   :alt: Mandacaru
   :class: only-dark
   :width: 640
   :align: center

**Mandacaru** is a Python framework for simulating fermionic systems with variational
quantum algorithms. Starting from an atomic geometry, it builds a molecular
Hamiltonian, maps it to qubits and optimises a quantum circuit to estimate its
energy. The Atomic Simulation Environment (ASE) provides the geometry and
calculator interface.

Start with lithium hydride (LiH)
--------------------------------

The introductory tutorials follow one molecule from a single energy calculation
to a potential-energy curve:

1. :doc:`Install Mandacaru <installation>` and check your Python environment.
2. :doc:`Run VQE for LiH <tutorial/vqe_lih>` and understand the basis, electron
   count, integration grid and energy units.
3. :doc:`Build an adaptive circuit <tutorial/adapt_vqe_lih>` and compare operator
   pools on the same molecular problem.
4. :doc:`Scan the Li–H distance <tutorial/pes_scan>` and compare basis sets and
   operator pools in a reproducible PNG figure.

How a calculation works
-----------------------

.. raw:: html

   <ol class="workflow" aria-label="Molecular simulation workflow">
     <li><strong>Atomic geometry</strong><span>ASE Atoms: elements and positions in ångströms</span></li>
     <li><strong>Basis and integration grid</strong><span>Spatial orbitals sampled on a real-space grid</span></li>
     <li><strong>Molecular Hamiltonian</strong><span>Integrals, Hartree–Fock orbitals and optional frozen core</span></li>
     <li><strong>Qubit Hamiltonian</strong><span>Jordan–Wigner, parity or Bravyi–Kitaev mapping; optional cache</span></li>
     <li><strong>Variational calculation</strong><span>VQE or ADAPT-VQE; circuit evaluation and classical optimisation</span></li>
     <li><strong>Results</strong><span>Energy, convergence information and circuit parameters</span></li>
   </ol>

A cached Hamiltonian lets you restart at the variational stage when comparing
solvers. Rebuild it whenever the geometry, basis, integration grid or electronic
problem changes. See :doc:`guide/hamiltonian_cache`.

Choose the right controls
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Control
     - What it changes
     - Learn more
   * - Basis set
     - The spatial functions used to represent the electrons
     - :doc:`guide/basis_sets`
   * - Integration grid
     - The numerical accuracy of the molecular integrals
     - :doc:`tutorial/pes_scan`
   * - Frozen core
     - Which occupied orbitals remain fixed during the correlated calculation
     - :doc:`tutorial/vqe_lih`
   * - Operator pool
     - The circuit generators available to ADAPT-VQE
     - :doc:`theory/adapt_vqe`
   * - Backend
     - How circuits are constructed and evaluated
     - :doc:`guide/backends`

Changing a basis changes the approximate Hamiltonian. Changing a pool changes
the variational search within that Hamiltonian. These are separate sources of
error, so the LiH tutorial compares them separately.

Units and interpretation
------------------------

The introductory examples use **ångströms (Å)** for distances and
**electronvolts (eV)** for reported energies. The integral, mapping and
Hamiltonian-cache layers use atomic units internally. A result's
``in_units("Ha")`` method gives a Hartree view; ``atomic_units=True`` changes
the driver output convention. Keep the default convention when following the
ASE examples.

The molecular total energy includes nuclear repulsion. It is different from a
binding energy, which requires a separately defined fragment reference.
Converging a variational optimiser does not establish convergence with respect
to the basis or integration grid.

Mandacaru generates its Gaussian basis functions internally. Named families
reproduce the intended shell structure, but do not use the published exponent
tables. Consult :doc:`guide/basis_sets` before comparing with literature values.

Explore further
---------------

The :doc:`guide/index` covers open shells, pseudopotentials, Hamiltonian caches
and execution backends. The :doc:`theory/index` explains the equations behind
the tutorials, and the :doc:`api` documents the Python interfaces.

Source code and issue reports are hosted on
`GitHub <https://github.com/seixas-research/mandacaru>`_. Releases are available
from `PyPI <https://pypi.org/project/mandacaru/>`_. Mandacaru is distributed under
the `MIT licence <https://github.com/seixas-research/mandacaru/blob/main/LICENSE>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   installation
   tutorial/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to guides

   guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api
   contributing_docs
   GitHub <https://github.com/seixas-research/mandacaru>
   PyPI <https://pypi.org/project/mandacaru/>
