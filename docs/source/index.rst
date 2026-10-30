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

**Mandacaru** is a framework for fermionic quantum simulation. A Rust scientific
engine owns the numerics: molecular integrals, Hartree-Fock and Kohn-Sham
iterations, crystal calculations, quantum states and variational solvers. A Python
interface supplies the ASE calculator, array conversion, visualization and the
adapters for quantum-computing services. The flagship algorithm is ADAPT-VQE; the
same calculator also runs mean-field, density functional, VQE, subspace and
architecture-search methods.

A first calculation
-------------------

.. code-block:: python

   from ase import Atoms
   from mandacaru import Mandacaru

   atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
   atoms.center()
   atoms.calc = Mandacaru(method="dft", xc="pbe",
                          grid_spacing=0.35, trace=False)
   print(atoms.get_potential_energy())  # eV
   print(atoms.get_forces())            # eV / angstrom

Read :doc:`installation` first, then :doc:`start/first_calculation`. The
:doc:`methods table <reference/methods>` says which method does what, and
:doc:`reference/not_yet_available` lists what the program does not do yet.

The migration of the numerical engine from Python to Rust is still in progress;
the :doc:`status page <development/native_cutover>` tells native calculations from
features that remain to be ported. An unsupported option raises an error: the
program never substitutes another calculation, and there is no execution selector.

Scientific conventions
----------------------

ASE geometries use angstroms, and the calculator returns electronvolts and
electronvolts per angstrom. The Rust core works in Bohr and Hartree; the Python
boundary converts units and copies inputs into native ownership. Returned arrays
have independent storage.

Molecular total energies include nuclear repulsion. A binding energy needs separate
fragment calculations. Convergence of an optimizer or SCF loop does not establish
basis or grid convergence. Generated Gaussian families use Mandacaru's exponent
construction; their names do not promise published exponent tables.

.. toctree::
   :maxdepth: 2
   :caption: Start

   installation
   start/first_calculation

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial/index

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   reference/methods
   reference/cli
   reference/run_log
   reference/not_yet_available

.. toctree::
   :maxdepth: 2
   :caption: Engineering

   development/index
   contributing_docs

Source and releases: `GitHub <https://github.com/seixas-research/mandacaru>`_
and `PyPI <https://pypi.org/project/mandacaru/>`_.
