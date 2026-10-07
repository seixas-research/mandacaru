"""Train VALQA's proposal model, then search an ansatz for a molecule it has not seen."""
import os
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms import fit

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STORE = os.path.join(DATA, "offline_store")        # written by 01_mcas_vqe_offline_training.py

model = fit(STORE, fractions=(1.0,), folds=5)
print(f"ready: {model.ready} -- {model.report['reason']}")

atoms = molecule("H2CO")
atoms.center(vacuum=3.0)
for method, options in (("valqa", {"proposal_model": os.path.join(STORE, "proposal_model.npz")}),
                        ("mcas-vqe", {})):
    atoms.calc = Mandacaru(method=method,
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.20,
                           pool="qeb",
                           active_space={"orbitals": {"occupied": 3, "virtual": 3},
                                         "method": "mp2"},
                           max_steps=40,
                           length_penalty=1e-3,
                           seed=1,
                           record=False,
                           **options)
    energy = atoms.get_potential_energy()
    print(f"{method:9s} E = {energy:.6f} eV  "
          f"({atoms.calc.result.num_evaluations} evaluations)")
