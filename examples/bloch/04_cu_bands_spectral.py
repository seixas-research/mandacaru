"""Copper FCC crystal: Bloch-ADAPT-VQE band structure and spectral function, saved as PNG.

Calculates the correlated band structure along the high-symmetry path Gamma -> X -> W -> L -> Gamma -> K
and the interacting Lehmann spectral function A(E, k) for metallic FCC Cu using Bloch-ADAPT-VQE
with PAW-LCAO DZP basis, multiple k-points (2x1x1 Monkhorst-Pack mesh = 2 cells), active space of 10 orbitals (20 qubits <= 20 qubits),
CEO operator pool, and matrix-free Cholesky decomposition. Saves the resulting figure as a PNG.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk
import numpy as np
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
try:
    os.makedirs(DATA, exist_ok=True)
except OSError:
    pass

# 1. Copper FCC structure (primitive cell: 1 atom)
# Supercell with 2 k-points (2x1x1 Monkhorst-Pack, Gamma-centered) = 2 atoms.
# PAW-LCAO DZP basis with active space of 10 spatial orbitals = 20 qubits (<= 20 qubits).
cu = bulk("Cu", "fcc", a=3.61)

calc = Mandacaru(
    method="bloch-adapt-vqe",
    basis={"name": "PAW-LCAO", "size": "DZP"},
    two_body="cholesky",
    pool="ceo",
    active_space={"orbitals": {"occupied": 9, "virtual": 1}},
    kpts={"size": (2, 1, 1), "gamma": True},
    max_iterations=5,
)
cu.calc = calc

# 2. Total ground-state energy per cell
energy = cu.get_potential_energy()
print(f"Copper total energy per cell: {energy:.6f} eV")

# 3. Continuous band structure along the appropriate FCC high-symmetry path Gamma -> X -> W -> L -> Gamma -> K
path = "GXWLGK"
bs = calc.band_structure(path=path, npoints=60)
e_fermi = float(bs.reference)
print(f"Band structure along '{path}' generated with {bs.energies.shape[1]} k-points. Fermi level reference: {e_fermi:.3f} eV")

# 4. Interacting Lehmann spectral function A(E, k) along the path
spec = bs.spectral_function
print(f"Spectral function A(E, k) computed: shape {spec.weights.shape}, max weight = {np.max(spec.weights):.4f} eV^-1")

# 5. Plot correlated bands, 2D spectral function map along the path, and Integrated Density of States (DOS)
fig, (ax_bands, ax_spec_map, ax_dos) = plt.subplots(
    1, 3, figsize=(14, 5), sharey=True, gridspec_kw={"width_ratios": [2.2, 2.2, 1.0]}
)

# Panel 1: Quasiparticle bands
bs.subtract_reference().plot(ax=ax_bands, emin=-10.0, emax=6.0)
ax_bands.axhline(0.0, color="k", linestyle="--", alpha=0.6)
ax_bands.set_title("Copper FCC: Correlated Bands")

# Panel 2: Spectral function A(E, k) intensity map along the path
_x, X, labels = bs.path.get_linear_kpoint_axis()
mesh = ax_spec_map.pcolormesh(
    spec.distances,
    spec.energies,
    spec.weights.T,
    cmap="Blues",
    shading="gouraud",
)
ax_spec_map.axhline(0.0, color="k", linestyle="--", alpha=0.6, label=r"$E_F$")
ax_spec_map.set_xticks(X)
ax_spec_map.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in labels])
for d in X:
    ax_spec_map.axvline(d, color="grey", linestyle=":", alpha=0.5)
ax_spec_map.set_xlim(spec.distances[0], spec.distances[-1])
ax_spec_map.set_ylim(-10.0, 6.0)
ax_spec_map.set_title(r"Spectral Function $A(E, k)$")
ax_spec_map.set_xlabel("Wavevector")

# Panel 3: Density of States (DOS) via linear tetrahedron method
if getattr(bs, "tetrahedral_dos", None) is not None:
    e_dos, dos = bs.tetrahedral_dos
else:
    e_dos, dos = calc.dos(kpts={"size": (6, 6, 6), "gamma": True}, integration="tetrahedron")
    e_dos = e_dos - e_fermi

ax_dos.plot(dos, e_dos, color="#1f77b4", lw=1.5, label="Tetrahedral DOS")
ax_dos.axhline(0.0, color="k", linestyle="--", alpha=0.6, label=r"$E_F$")
ax_dos.set_xlabel(r"DOS (states / eV)")
ax_dos.set_title("Density of States")
ax_dos.set_ylim(-10.0, 6.0)
ax_dos.grid(True, linestyle="--", alpha=0.5)
ax_dos.legend(loc="upper right")

plt.tight_layout()
png_path = os.path.join(DATA, "cu_bloch_adapt_vqe_bands.png")
try:
    fig.savefig(png_path, dpi=150)
    print(f"Saved band structure figure to {png_path}")
except OSError:
    tmp_path = os.path.join("/tmp", os.path.basename(png_path))
    fig.savefig(tmp_path, dpi=150)
    print(f"Saved band structure figure to {tmp_path} (workspace write restricted)")
