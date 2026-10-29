"""ASE geometry and unit views for native Wannier calculations."""
from __future__ import annotations

import json
import numpy as np
from ..units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, HARTREE_TO_EV


def _trials(guess, atoms):
    centers = atoms.positions * ANGSTROM_TO_BOHR
    if isinstance(guess, str) and guess in ("bonds", "sp3"):
        from ase.neighborlist import neighbor_list

        first, vectors = neighbor_list("iD", atoms, 6.0)
        key = "neighbors" if guess == "sp3" else "bond_neighbors"
        return [
            dict(center=center.tolist(), label=f"{atom.symbol}{i}",
                 **{key: (vectors[first == i] * ANGSTROM_TO_BOHR).tolist()})
            for i, (atom, center) in enumerate(zip(atoms, centers))
        ]
    if isinstance(guess, dict):
        records = []
        for selector, names in guess.items():
            names = [names] if isinstance(names, str) else list(names)
            sites = (
                [i for i, atom in enumerate(atoms) if atom.symbol == selector]
                if isinstance(selector, str) else [int(selector)]
            )
            if not sites:
                raise ValueError(f"the Wannier guess names no atoms for {selector!r}")
            for site in sites:
                for name in names:
                    records.append(dict(center=centers[site].tolist(), orbital=name,
                                        label=f"{atoms[site].symbol}{site}"))
        return records
    centers = np.asarray(guess, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("a Wannier guess names bonds, sp3, atomic orbitals or centers of shape (n,3)")
    return [dict(center=(center * ANGSTROM_TO_BOHR).tolist(), orbital="s",
                 label=f"center{i}") for i, center in enumerate(centers)]


def calculate(calculation, n_functions, *, guess, windows, bands, kpts,
              max_iterations, tolerance, step):
    from .paw import band_input

    if calculation.result is None or calculation.result.periodic is None:
        raise RuntimeError("run periodic DFT before constructing Wannier functions")
    if not hasattr(calculation, "_crystal_adapter"):
        raise ValueError("Wannier setup needs the crystal geometry and PAW setup")
    atoms = calculation.atoms
    adapter = calculation._crystal_adapter
    mesh = calculation._kpts_spec if kpts is None else kpts
    size = mesh["size"] if isinstance(mesh, dict) else mesh
    spec = dict(size=[int(n) for n in size], bands=bands,
                n_functions=n_functions, trials=_trials(guess, atoms),
                centers=np.asarray(adapter.centers).tolist(),
                channel_labels=[list(channel) for channel in adapter.channels])
    if windows is not None:
        if not isinstance(windows, dict) or "outer" not in windows:
            raise ValueError("Wannier windows specify an outer energy pair and an optional frozen pair in eV")
        spec["windows"] = {
            name: None if pair is None else [float(e) / HARTREE_TO_EV for e in pair]
            for name, pair in windows.items()
        }
    setup = calculation.result.periodic.wannier_setup(json.dumps(spec, allow_nan=False))
    for index in np.ndindex(tuple(spec["size"])):
        fractional = np.asarray(index, dtype=float) / np.asarray(spec["size"])
        metadata, arrays = band_input(adapter, fractional[None, :])
        setup.add_point(metadata, arrays)
        del arrays
    native = setup.localize(max_iterations=max_iterations, tolerance=tolerance, step=step)
    return (WannierResult(native, 0) if native.n_spins == 1 else SpinWannierResult(native))


class WannierResult:
    """Unit-converted view of one native localized spin channel."""

    def __init__(self, native, spin):
        self._native = native
        self._spin = spin
        self._snapshot = native.snapshot(spin=spin)

    @property
    def n_functions(self):
        return len(self._snapshot["centers"])

    @property
    def centers(self):
        return self._snapshot["centers"].copy() * BOHR_TO_ANGSTROM

    @property
    def spreads(self):
        return self._snapshot["spreads"].copy() * BOHR_TO_ANGSTROM**2

    @property
    def total_spread(self):
        return sum(self._snapshot[name] for name in
                   ("omega_invariant", "omega_offdiagonal", "omega_diagonal")) * BOHR_TO_ANGSTROM**2

    @property
    def converged(self):
        return self._snapshot["converged"] and self._snapshot["disentanglement_converged"]

    @property
    def labels(self):
        return tuple(self._snapshot["labels"])

    @property
    def kpoints(self):
        return self._snapshot["kpoints"].copy()

    @property
    def lattice_vectors(self):
        return self._snapshot["lattice_vectors"].copy()

    @property
    def hamiltonian_R(self):
        return self._snapshot["hamiltonian_R"].copy() * HARTREE_TO_EV

    def interpolate(self, kpoints):
        return self._native.interpolate(np.ascontiguousarray(kpoints, dtype=float),
                                        spin=self._spin) * HARTREE_TO_EV


class SpinWannierResult:
    """Two spin channels sharing one native calculation owner."""

    def __init__(self, native):
        self._native = native
        self.up = WannierResult(native, 0)
        self.down = WannierResult(native, 1)

    @property
    def centers(self):
        return np.stack([self.up.centers, self.down.centers])

    @property
    def total_spread(self):
        return np.array([self.up.total_spread, self.down.total_spread])

    @property
    def converged(self):
        return self.up.converged and self.down.converged

    def interpolate(self, kpoints):
        return np.stack([self.up.interpolate(kpoints), self.down.interpolate(kpoints)])
