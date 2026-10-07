"""A hydrogen chain with the fixed-ansatz and the adaptive Bloch methods."""
from ase import Atoms
from mandacaru import Mandacaru

chain = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[1.0, 10.0, 10.0], pbc=[True, False, False])

for method in ("bloch-vqe", "bloch-adapt-vqe"):
    chain.calc = Mandacaru(method=method,
                           basis="HAO",
                           h=0.25,
                           kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer={"method": "L-BFGS", "maxiter": 2000,
                                      "tol": 1e-12})
    energy = chain.get_potential_energy()
    weights = chain.calc.band_weights()
    print(f"{method:16s} E/cell = {energy:.6f} eV  "
          f"quasiparticle weights {weights[:, 0].round(3)}")
