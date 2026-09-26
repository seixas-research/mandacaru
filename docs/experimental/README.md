# Experimental documentation

Notes for the experimental parts of Mandacaru — code that is usable but still
under development and not yet fully validated (the `mandacaru.experimental`
package). This folder is **independent of the Sphinx manual**: nothing here is
referenced from `docs/source/`, so it is never part of the documentation build
(`cd docs && make html`) or the hosted manual. Read the Markdown files
directly.

There are currently no experimental methods.

When a method graduates to the stable API, its page moves into
`docs/source/` and is added to the relevant toctree — as the pseudopotentials
did on 2026-09-15 (`mandacaru.pseudopotentials`, guide
`docs/source/guide/pseudopotentials.md`, selected with `basis="NCPP"` /
`"ONCVPSP"` / `"PAW-LCAO"`).
