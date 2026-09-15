# Experimental documentation

Notes for the experimental parts of Carcará — code that is usable but still
under development and not yet fully validated (the `carcara.experimental`
package: VASQE). This folder is **independent of the Sphinx manual**: nothing
here is referenced from `docs/source/`, so it is never part of the
documentation build (`cd docs && make html`) or the hosted manual. Read the
Markdown files directly.

| File | Method |
| :--- | :--- |
| `vasqe.md` | VASQE — the Variational Adaptive Stochastic Quantum Eigensolver (ADAPT-VQE with softmax operator selection and temperature annealing), including its subspace-search and periodic (Bloch) usage. |

When a method graduates to the stable API, its page moves into
`docs/source/` and is added to the relevant toctree — as the pseudopotentials
did on 2026-09-15 (`carcara.pseudopotentials`, guide
`docs/source/guide/pseudopotentials.md`, selected with `basis="NCPP"` /
`"ONCVPSP"` / `"PAW"`).
