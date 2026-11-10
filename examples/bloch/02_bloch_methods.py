"""A hydrogen chain with the four variational methods: a periodic structure takes the supercell."""
from ase import Atoms
from mandacaru import Mandacaru

chain = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[1.0, 10.0, 10.0], pbc=[True, False, False])

for method in ("vqe", "adapt-vqe", "vasqa", "valsqa"):
    search = {"seed": 0} if method in ("vasqa", "valsqa") else {}
    chain.calc = Mandacaru(method=method,
                           crystal_orbitals="real-space",
                           basis="HAO",
                           grid_spacing=0.25,
                           kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer={"method": "L-BFGS", "maxiter": 2000,
                                      "tol": 1e-12},
                           **search)
    energy = chain.get_potential_energy()
    weights = chain.calc.band_weights()
    print(f"{method:16s} E/cell = {energy:.6f} eV  "
          f"quasiparticle weights {weights[:, 0].round(3)}")
