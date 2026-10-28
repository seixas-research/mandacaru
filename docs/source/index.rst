
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

**Mandacaru** provides a Rust scientific engine with a Python interface for ASE,
interactive calculations and external execution services. The native engine
owns molecular integrals, mean-field and Kohn–Sham iterations, quantum state
operations, selected variational solvers and persistence.

Start with a native calculation
-------------------------------

.. code-block:: python

   from ase import Atoms
   from mandacaru import Mandacaru

   atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
   atoms.center()
   atoms.calc = Mandacaru(method="dft", engine="rust", xc="pbe",
                          grid_spacing=0.35, trace=False)
   print(atoms.get_potential_energy())  # eV
   print(atoms.get_forces())            # eV / angstrom

Read :doc:`installation` for the source build and packaged extension,
:doc:`guide/native_engine` for supported methods and execution controls, and
:doc:`guide/dft` for native exchange and correlation.

The cutover is in progress. The :doc:`coverage audit <development/native_cutover>`
distinguishes native calculations from features whose numerical Python code
still needs to move. Selecting ``engine="rust"`` raises an error for unsupported
work; it does not fall back to a different solver.

Scientific conventions
----------------------

ASE geometries use ångströms and the calculator returns electronvolts and
electronvolts per ångström. The Rust core uses Bohr and Hartree. The Python
boundary converts units and copies inputs into native ownership. Returned
arrays have independent storage.

Molecular total energies include nuclear repulsion. A binding energy requires
separate fragment calculations. Convergence of an optimizer or SCF loop does
not establish basis or grid convergence. Generated Gaussian families use
Mandacaru's exponent construction; their names do not promise published
exponent tables.

.. toctree::
   :maxdepth: 2
   :caption: Calculations

   installation
   guide/native_engine
   guide/dft
   tutorial/index
   guide/index

.. toctree::
   :maxdepth: 2
   :caption: Methods and reference

   theory/index
   api
   development/native_cutover
   development/index
   contributing_docs

Source and releases: `GitHub <https://github.com/seixas-research/mandacaru>`_
and `PyPI <https://pypi.org/project/mandacaru/>`_.
