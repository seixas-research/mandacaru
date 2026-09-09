# Experimental documentation

Notes for the experimental parts of Carcará — code that is usable but still
under development and not yet fully validated (the `carcara.experimental`
package: VASQE and the pseudopotentials). This folder is **independent of
the Sphinx manual**: nothing here is referenced from `docs/source/`, so it is
never part of the documentation build (`cd docs && make html`) or the hosted
manual. Read the Markdown files directly.

| File | Method |
| :--- | :--- |
| `vasqe.md` | VASQE — the Variational Adaptive Stochastic Quantum Eigensolver (ADAPT-VQE with softmax operator selection and temperature annealing), including its subspace-search and periodic (Bloch) usage. |
| `pseudopotentials.md` | Norm-conserving Troullier–Martins pseudopotentials with Kleinman–Bylander projectors (`pseudopotentials=True`), their generation and the valence-only Hamiltonian. |

When a method graduates to the stable API, its page moves into
`docs/source/tutorial/` and is added to that toctree.
