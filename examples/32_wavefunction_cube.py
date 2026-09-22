# -*- coding: utf-8 -*-
# file: examples/32_wavefunction_cube.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""See the ADAPT-VQE state: densities and natural orbitals as ``.cube`` files.

The converged state of an ADAPT-VQE run is a **many-electron** wavefunction on a
qubit register -- a function of :math:`3N` coordinates, which no volumetric file
can hold.  What *can* be put on a three-dimensional grid, exactly, are its
one-particle reductions: the electron density, the natural orbitals, and the
difference between the correlated density and the Hartree-Fock one.  This
example writes all three for a **stretched** H2 molecule, where correlation is
strong enough to see.

At 1.60 Angstrom the bond is half broken.  A single determinant would put both
electrons in the bonding orbital (occupations 2 and 0); the ADAPT-VQE state
moves a substantial fraction into the antibonding one, and the difference
density shows exactly where those electrons went -- out of the bond, onto the
two atoms.  That picture is the whole point of a correlated calculation, and it
is one call away from a finished run.

Output (all in ``examples/data/``)
----------------------------------
``h2_stretched_density.cube``
    The electron density in e/Bohr^3.
``h2_stretched_difference.cube``
    ``n(ADAPT-VQE) - n(Hartree-Fock)``, the density correlation moved.
``h2_stretched_natural_orbital_1.cube``
    The *antibonding* natural orbital -- the one a determinant leaves empty.
``h2_cube_slice.png``
    A two-panel slice through the bond of the first two.

Open the cube files in VESTA (File > Open) or VMD (``vmd -cube ...``); see
``docs/source/guide/visualization.md`` for isosurface values that work.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.integrals import Grid
from mandacaru.units import BOHR_TO_ANGSTROM

# All generated files (cube files, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

#: A stretched bond: at equilibrium (0.74 A) the natural occupations are
#: (1.97, 0.03) and the difference density is barely visible; here they are
#: close to (1.5, 0.5) and the picture is unmistakable.
BOND_LENGTH = 1.60
#: The real-space box.  The cell *is* the integration box in Mandacaru, so it
#: has to hold the (fairly diffuse) hydrogen 1s tails.
CELL = 7.0
#: Grid spacing (Angstrom) of the calculation -- and of the cube files, which
#: are written on the calculation's own grid.  Coarse on purpose: a cube file
#: is one text number per node, so 0.35 A keeps each file around 120 kB.
SPACING = 0.35
#: The plot resamples the basis functions on a finer grid (never interpolating
#: the data), which is what the ``grid=`` argument is for.
PLOT_SPACING = 0.08
#: Half-width (Angstrom) of the plotted window about the molecule.  The cell is
#: mostly vacuum: showing all of it would leave the interesting part tiny.
PLOT_WINDOW = 2.2


def stretched_h2() -> Atoms:
    """H2 along z, centered in its own cell."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, BOND_LENGTH]],
                  cell=[CELL, CELL, CELL])
    atoms.center()
    return atoms


def plane_slice(field, axis: int = 1, window: float = None):
    """``(horizontal, vertical, values, kept_axes)`` of one plane of nodes.

    The plane chosen is the one whose ``axis`` coordinate is closest to the
    molecule.  The grid is orthorhombic here, so a constant-index plane of
    nodes really is a plane in space; the two surviving Cartesian coordinates
    come back in Angstrom, as meshes matching the data.  ``window`` (Angstrom)
    crops the plane to a box that far from the molecule -- most of the cell is
    vacuum, and plotting it only dilutes the picture.
    """
    grid = field.grid
    coordinates = (grid.X, grid.Y, grid.Z)
    nuclei = np.asarray(field.positions) * BOHR_TO_ANGSTROM

    def take(array):
        return np.moveaxis(array, axis, 0)[node]

    line = np.moveaxis(coordinates[axis], axis, 0)[:, 0, 0] * BOHR_TO_ANGSTROM
    node = int(np.argmin(np.abs(line - nuclei[:, axis].mean())))
    keep = [k for k in range(3) if k != axis]
    horizontal = take(coordinates[keep[0]]) * BOHR_TO_ANGSTROM
    vertical = take(coordinates[keep[1]]) * BOHR_TO_ANGSTROM
    values = take(field.data)
    if window is not None:
        rows = np.abs(horizontal[:, 0] - nuclei[:, keep[0]].mean()) <= window
        columns = np.abs(vertical[0] - nuclei[:, keep[1]].mean()) <= window
        cut = np.ix_(rows, columns)
        horizontal, vertical, values = (horizontal[cut], vertical[cut],
                                        values[cut])
    return horizontal, vertical, values, keep


def plot_slices(density, difference, path: str) -> None:
    """Two panels through the bond: the density, and what correlation moved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizontal, vertical, total, keep = plane_slice(density, window=PLOT_WINDOW)
    _h, _v, delta, _keep = plane_slice(difference, window=PLOT_WINDOW)
    labels = "xyz"
    nuclei = np.asarray(density.positions) * BOHR_TO_ANGSTROM

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    # Sequential job (magnitude, one sign): a single hue, light to dark.
    first = axes[0].contourf(horizontal, vertical, total, levels=24,
                             cmap="Blues")
    figure.colorbar(first, ax=axes[0], label="n  [e/Bohr$^3$]")
    axes[0].set_title("Electron density of the ADAPT-VQE state", fontsize=11)

    # Diverging job (polarity about zero): two hues with a neutral midpoint,
    # and limits symmetric about zero so the midpoint really is zero.  Drawn as
    # a continuous map rather than filled contours: the far field is zero to
    # ten decimal places, and banding there would turn arithmetic noise into a
    # visible pattern.
    span = float(np.abs(delta).max()) or 1.0
    second = axes[1].pcolormesh(horizontal, vertical, delta, cmap="RdBu_r",
                                vmin=-span, vmax=span, shading="gouraud")
    figure.colorbar(second, ax=axes[1],
                    label=r"$n_{\mathrm{ADAPT}} - n_{\mathrm{HF}}$  [e/Bohr$^3$]")
    axes[1].set_title("What correlation moved", fontsize=11)

    for axis in axes:
        axis.plot(nuclei[:, keep[0]], nuclei[:, keep[1]], "o",
                  markersize=7, markerfacecolor="none",
                  markeredgecolor="#333333", markeredgewidth=1.4)
        axis.set_xlabel(f"{labels[keep[0]]}  [$\\AA$]", color="#444444")
        axis.set_ylabel(f"{labels[keep[1]]}  [$\\AA$]", color="#444444")
        axis.set_aspect("equal")
        axis.tick_params(colors="#666666", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#cccccc")

    figure.suptitle(f"H$_2$ at {BOND_LENGTH:.2f} $\\AA$  -  "
                    f"one-particle picture of a many-body state", fontsize=12)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    atoms = stretched_h2()
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis="HAO",
                           h=SPACING,
                           pool="fermionic")
    energy = atoms.get_potential_energy()
    print(f"H2 at {BOND_LENGTH:.2f} A: E = {energy:.6f} eV "
          f"({atoms.calc.n_qubits} qubits)")

    orbitals = atoms.calc.natural_orbitals()
    print("\nNatural occupations (a single determinant would give 2 and 0):")
    for index, occupation in enumerate(orbitals.occupations):
        print(f"  natural orbital {index}: {occupation:9.6f}")
    print(f"  sum = {orbitals.n_electrons:.6f} electrons")

    written = []
    for name, options in (
            ("density", dict(quantity="density")),
            ("difference", dict(quantity="difference_density")),
            ("natural_orbital_1", dict(quantity="natural_orbital", index=1))):
        path = os.path.join(DATA, f"h2_stretched_{name}.cube")
        field = atoms.calc.write_cube(path, **options)
        written.append((path, field))
        print()
        print(field.summary())

    # The pictures: resample the basis on a finer grid so the contours are
    # smooth.  The cube files above stay on the calculation's own grid, where
    # the electron count is exact and the files are small.
    fine = Grid(center=atoms.get_positions().mean(axis=0), box_size=0.0,
                h=PLOT_SPACING, units="angstrom", cell=atoms.get_cell())
    density = atoms.calc.volumetric_field("density", grid=fine)
    difference = atoms.calc.volumetric_field("difference_density", grid=fine)
    png = os.path.join(DATA, "h2_cube_slice.png")
    plot_slices(density, difference, png)

    print("\nWritten:")
    for path, _field in written + [(png, None)]:
        print(f"  {os.path.relpath(path, os.path.dirname(DATA))}  "
              f"({os.path.getsize(path) / 1024:.0f} kB)")
    print("\nOpen a cube file in VESTA (File > Open) or VMD "
          "(vmd -cube <file>); see docs/source/guide/visualization.md.")


if __name__ == "__main__":
    main()
