"""Silicon diamond crystal: Bloch-ADAPT-VQE band structure and spectral function, saved as PNG.

Calculates the correlated band structure along the high-symmetry path L -> Gamma -> X -> W -> K -> Gamma
and the interacting Lehmann spectral function A(E, k) using Bloch-ADAPT-VQE with multiple k-points
(2x1x1 Monkhorst-Pack mesh), PAW-LCAO DZP basis, an active space of 12 orbitals (24 qubits)
selected via MP2 natural orbitals with threshold 0.003, CEO operator pool, and matrix-free
Cholesky decomposition. Saves the resulting figure as a PNG.
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

# 1. Silicon diamond structure (primitive cell: 2 atoms)
# Supercell with 2 k-points (2x1x1 Monkhorst-Pack, Gamma-centered) = 4 atoms.
# PAW-LCAO DZP basis with active space of 12 spatial orbitals = 24 qubits via MP2 (threshold 0.003).
si = bulk("Si", "diamond", a=5.43)

calc = Mandacaru(
    method="bloch-adapt-vqe",
    basis={"name": "PAW-LCAO", "size": "DZP"},
    two_body="cholesky",
    pool="ceo",
    active_space={"orbitals": 12, "method": "mp2", "threshold": 0.003},
    kpts={"size": (2, 1, 1), "gamma": True},
    max_iterations=2,
)
si.calc = calc

# 2. Total ground-state energy per cell
energy = si.get_potential_energy()
print(f"Silicon total energy per cell: {energy:.6f} eV")

# 3. Continuous band structure along the appropriate FCC/diamond high-symmetry path L -> Gamma -> X -> W -> K -> Gamma
path = "LGXWKG"
bs = calc.band_structure(path=path, npoints=60)
print(f"Interpolated band structure generated along path '{path}' with {bs.energies.shape[1]} k-points.")
e_fermi = float(bs.reference)
energies = bs.energies[0] - e_fermi
vbm = np.max(energies[energies <= 0.0])
cbm = np.min(energies[energies > 0.0])
print(f"Fermi level reference: {e_fermi:.3f} eV | VBM: {vbm:.3f} eV | CBM: {cbm:.3f} eV | Fundamental Gap: {cbm - vbm:.3f} eV")

# 4. Interacting Lehmann spectral function A(E, k) along the path
spec = bs.spectral_function
print(f"Spectral function A(E, k) computed: shape {spec.weights.shape}, max weight = {np.max(spec.weights):.4f} eV^-1")

# 5. Plot correlated bands, 2D spectral function map along the path, and Integrated Density of States (DOS)
fig, (ax_bands, ax_spec_map, ax_dos) = plt.subplots(
    1, 3, figsize=(14, 5), sharey=True, gridspec_kw={"width_ratios": [2.2, 2.2, 1.0]}
)

# Panel 1: Quasiparticle bands
bs.subtract_reference().plot(ax=ax_bands, emin=-14.0, emax=8.0)
ax_bands.axhline(0.0, color="k", linestyle="--", alpha=0.6)
ax_bands.set_title("Silicon Diamond: Correlated Bands")

# Panel 2: Spectral function A(E, k) intensity map along the path
_x, X, labels = bs.path.get_linear_kpoint_axis()
mesh = ax_spec_map.pcolormesh(
    spec.distances,
    spec.energies,
    spec.weights.T,
    cmap="Blues",
    shading="gouraud",
)
ax_spec_map.axhline(0.0, color="k", linestyle="--", alpha=0.6, label=r"$E_F$ (VBM)")
ax_spec_map.set_xticks(X)
ax_spec_map.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in labels])
for d in X:
    ax_spec_map.axvline(d, color="grey", linestyle=":", alpha=0.5)
ax_spec_map.set_xlim(spec.distances[0], spec.distances[-1])
ax_spec_map.set_ylim(-14.0, 8.0)
ax_spec_map.set_title(r"Spectral Function $A(E, k)$")
ax_spec_map.set_xlabel("Wavevector")

# Panel 3: Density of States (DOS) via linear tetrahedron method
if getattr(bs, "tetrahedral_dos", None) is not None:
    e_dos, dos = bs.tetrahedral_dos
else:
    e_dos, dos = calc.dos(kpts={"size": (6, 6, 6), "gamma": True}, integration="tetrahedron")
    e_dos = e_dos - e_fermi

ax_dos.plot(dos, e_dos, color="#d62728", lw=1.5, label="Tetrahedral DOS")
ax_dos.axhline(0.0, color="k", linestyle="--", alpha=0.6, label=r"$E_F$ (VBM)")
ax_dos.set_xlabel(r"DOS (states / eV)")
ax_dos.set_title("Density of States")
ax_dos.set_ylim(-14.0, 8.0)
ax_dos.grid(True, linestyle="--", alpha=0.5)
ax_dos.legend(loc="upper right")

plt.tight_layout()
png_name = "si_bloch_adapt_vqe_bands.png"
artifact_dir = os.environ.get("MANDACARU_ARTIFACT_DIR", "/Users/leseixas/.gemini/antigravity-cli/brain/51e637a2-f0e3-4d9f-8c02-2601ea8eae40")
for target_dir in [DATA, artifact_dir, "/tmp"]:
    try:
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, png_name)
        fig.savefig(dest, dpi=150)
        print(f"Saved band structure figure to {dest}")
    except OSError:
        pass
