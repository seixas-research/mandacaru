# -*- coding: utf-8 -*-
# file: examples/28_LiH_relaxation_PAW.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH geometry relaxation with ADAPT-VQE forces (PAW pseudopotentials, DZP basis).

Starts at 2.0 Angstrom, relaxes with ASE's BFGS and writes the final geometry
to examples/data/relax.xyz.
"""

import os

from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

from carcara import Carcara

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = Atoms("LiH",
              positions=[[0.0, 0.0, 0.0],
                         [0.0, 0.0, 2.0]],
              cell=[10.0, 10.0, 10.0])
atoms.center()

atoms.calc = Carcara(method="adapt-vqe",
                     basis={"name": "PAW", "size": "SZ"},
                     h=0.10,
                     pool="ceo",
                     mapping="jordan_wigner",
                     optimizer="L-BFGS-B",
                     max_iterations=80,
                     gradient_tolerance=1e-5,
                     output='output.txt',
                     verbose=False)

opt = BFGS(atoms, trajectory=os.path.join(DATA, "relax.traj"))
opt.attach(lambda: print(f"    Li-H distance {atoms.get_distance(0, 1):.4f} A"))
opt.run(fmax=0.02)

write(os.path.join(DATA, "relax.xyz"), atoms)
print(f"Final Li-H distance: {atoms.get_distance(0, 1):.4f} A")
print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
