"""Stretched H2: the density, the correlation difference density and a natural orbital as .cube files."""
import os
import matplotlib.pyplot as plt
from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.60]], cell=[7.0, 7.0, 7.0])
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35, pool="fermionic")
atoms.get_potential_energy()
print(f"Natural occupations: {atoms.calc.natural_orbitals().occupations.round(4)}")

for name, options in (("density", {"quantity": "density"}),
                      ("difference", {"quantity": "difference_density"}),
                      ("natural_orbital_1", {"quantity": "natural_orbital", "index": 1})):
    atoms.calc.write_cube(os.path.join(DATA, f"h2_stretched_{name}.cube"), **options)

# The plane through the bond: where correlation moved the electrons
field = atoms.calc.volumetric_field("difference_density")
middle = field.data.shape[0] // 2
span = abs(field.data[middle]).max()
plt.imshow(field.data[middle].T, origin="lower", cmap="RdBu_r", vmin=-span, vmax=span)
plt.colorbar(label="n_ADAPT - n_HF (e/Bohr^3)")
plt.savefig(os.path.join(DATA, "h2_difference_slice.png"), dpi=150)
