# Reading a run

A variational run reports itself in three places, each with a different job:
the **standard-output trace** while it runs, the structured **`output.txt`**
log, and two optional **JSON dumps** for the objects that are too large to
print.

## The standard-output trace

`verbose=True` (the default) prints a header, one row per ADAPT iteration, and
a closing summary:

```text
======================================================================
ADAPT-VQE | mapping: jordan_wigner | 6 qubits | device: AER_simulator
pool: qeb (QEBPool) | 8 operators | optimizer: COBYLA | gradient: analytic
k-points: Gamma (1x1x1 Monkhorst-Pack) | spin: False | reference: hartree-fock
provider: qiskit | circuits: False | quenching: True
======================================================================
Qubit Hamiltonian: 118 Pauli terms
Hartree-Fock reference energy = -154.59070457 eV
======================================================================
iter   |grad|      E (eV)       dE   expr  cnot    1q depth   type operator
-----------------------------------------------------------------------------
   1 2.72e-01 -154.730899 -1.4e-01   6.21    48    40    65 double QD(0,3->2,5)
   2 8.31e-02 -154.765242 -3.4e-02   2.91    96    74   129 double QD(1,4->2,5)
   3 5.54e-02 -154.776105 -1.1e-02   2.05   144   108   193 double QD(1,3->2,5)
```

The **pool's type and size** and the Hamiltonian's **term count** appear in the
header, before the iterations. Neither the pool's operators nor the
Hamiltonian's Pauli strings are printed: a realistic pool grows with the fourth
power of the number of orbitals and a realistic Hamiltonian runs to thousands of
terms, so printing either buries the run.

Each iteration is one row, one column per property computed at that step:

| Column | Meaning |
| :--- | :--- |
| `iter` | Growth step (1-based). |
| `\|grad\|` | Largest pool gradient; the operator with this gradient is the one selected. Convergence is when it falls below `gradient_tolerance`. |
| `E (eV)` | Energy after the inner re-optimization (Hartree with `atomic_units=True`). |
| `dE` | Change from the previous step. |
| `expr` | Expressivity of the grown ansatz: KL divergence from the Haar distribution over the number-conserving sector. It falls as the ansatz specializes. Reads `-` where it is not computed: by default (`run(log_expressivity="auto")`) it is computed on the sparse and sector backends at any width and on the dense backend up to 10 qubits, because on a *dense* 12-qubit register the estimate costs 78 s per iteration against 0.07 s sparse. `True` computes it regardless, `False` never. |
| `npar` | Variational parameters in the ansatz. Equal to `iter`, so it is the first column dropped on a narrow terminal. |
| `cnot` | CNOT gates after compiling to the native gate set. |
| `1q` | Single-qubit gates in the same compilation. |
| `depth` | Circuit depth in the same compilation. |
| `type` | Kind of the selected operator (`double`, `single`, `ceo`, ...), with the pool's name stripped — the header already carries it. |
| `operator` | Its label, e.g. `QD(0,3->2,5)`. |

**One iteration is always one line.** The row is sized to the terminal
(`shutil.get_terminal_size()`, falling back to 80 columns when the output is
piped): if the full set will not fit, columns are dropped in the order `npar`,
`dE`, `1q`, `depth`, `expr`, `cnot` — the derivable ones first — rather than
letting rows wrap. `iter`, the gradient, the energy, the operator type and its
label are never dropped. Set the `COLUMNS` environment variable to override the
detected width.

The circuit columns need `profile=True` (the default); with `profile=False`
they read `-`.

## The `output.txt` log

`output=` writes the machine-readable protocol of
{mod}`carcara.utils.logging`: metadata, optimizer setup, one block per
iteration, and a summary. Each iteration block names the **selected operator**
and the **pool's size** -- not the pool's contents, for the same reason the
trace does not. {func}`carcara.utils.logging.parse_output` reads it back.

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               output="output.txt")
```

## `verbose_operators` and `verbose_hamiltonian`

Two options write what the trace leaves out, as JSON, once per run:

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               verbose_operators=True,      # -> pool.json
               verbose_hamiltonian=True)    # -> hamiltonian.json
```

`pool.json` carries the pool's name and size and, for every operator, its
label, kind, support and the Pauli expansion of its anti-Hermitian generator.
`hamiltonian.json` carries the qubit Hamiltonian's Pauli terms with their
complex coefficients **in Hartree** -- the operator's own unit, whatever units
the driver reports energies in.

Either option also takes a path, which is how a scan gives each geometry its
own file:

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               verbose_operators=f"data/pool_{distance:.2f}.json",
               verbose_hamiltonian=f"data/hamiltonian_{distance:.2f}.json")
```

Both are capped at
{data}`carcara.core.serialization.MAX_FILE_QUBITS` qubits: above that the file
is skipped with a `RuntimeWarning` and the run continues.

The same flags exist on the command line:

```bash
carcara LiH --cell 10 --verbose-operators --verbose-hamiltonian
carcara LiH --cell 10 --verbose-operators data/pool.json
```

```{note}
`verbose_hamiltonian` writes a file for *reading*. To write one Carcará can
read *back* -- skipping the integrals and the fermion-to-qubit mapping on the
next run -- use `save_hamiltonian=` and `load_hamiltonian=` instead; see
[the Hamiltonian cache](hamiltonian_cache.md).
```
