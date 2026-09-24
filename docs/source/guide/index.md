# How-To Guides

Task-oriented guides for the parts of Mandacaru that sit *around* the solvers:
how the single-particle basis and the external potential are built, how to
fit a large basis on a small qubit register, how to
remove the qubits a Hamiltonian's symmetries make redundant, how to
read what a run prints and cite what it used, how to draw the converged state in real space and split it into
per-atom charges and moments, how to
estimate the qubit budget of a run before
launching it, where the Hamiltonian is stored, which quantum SDK builds and executes the circuits, how to
reach real quantum hardware and what a measurement there costs, how to choose the classical optimizer, and how to control the
optimization loop.

```{toctree}
:maxdepth: 1
run_output
visualization
charges
basis_sets
nao_ae
pseudopotentials
active_space
tapering
open_shell
interaction_energy
dry_run
hamiltonian_cache
backends
aws_braket
measurement_cost
optimizers
quenching
checkpoints_qpe
```
