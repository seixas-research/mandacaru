"""A periodic hydrogen chain: energy per cell and quasiparticle bands, saved as PNG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
try:
    os.makedirs(DATA, exist_ok=True)
except OSError:
    pass

chain = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[1.0, 10.0, 10.0], pbc=[True, False, False])
chain.calc = Mandacaru(method="bloch-adapt-vqe",
                       basis="HAO",
                       grid_spacing=0.25,
                       kpts={"size": (4, 1, 1), "gamma": True})

energy = chain.get_potential_energy()
bands = chain.calc.bands()
k_coords = chain.calc.kpoints[:, 0]
print(f"Energy per cell: {energy:.6f} eV")
for k, band in zip(k_coords, bands[:, 0]):
    print(f"k = {k:+.3f}  E = {band:+.3f} eV")

# Save band dispersion figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k_coords, bands[:, 0], "o-", color="#1f77b4", label="Bloch-ADAPT-VQE quasiparticle band")
ax.set_xlabel(r"$k_x$ (fractional)")
ax.set_ylabel("Energy (eV)")
ax.set_title("Quasiparticle Bands of 1D Periodic Hydrogen Chain")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
png_path = os.path.join(DATA, "hydrogen_chain_bands.png")
try:
    fig.savefig(png_path, dpi=150)
    print(f"Saved band structure figure to {png_path}")
except OSError:
    tmp_path = os.path.join("/tmp", os.path.basename(png_path))
    fig.savefig(tmp_path, dpi=150)
    print(f"Saved band structure figure to {tmp_path} (workspace write restricted)")
