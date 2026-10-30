# Command line

The `mandacaru` command runs one native calculation and prints the result as JSON.
It is a thin adapter over the same `Mandacaru` calculator: every flag below sets one
constructor option, and every other option keeps its Python default. For anything
else (a text log, an active space, an optimizer, hardware), write a short script.

```bash
mandacaru --version
mandacaru --help
```

## A first run

The geometry is a JSON file with `numbers`, `positions` and `cell` (angstrom, one
row per atom and one row per cell vector), and optionally `pbc`, `units` and
`magnetic_moments`:

```json
{"numbers": [1, 1],
 "positions": [[3.0, 3.0, 2.63], [3.0, 3.0, 3.37]],
 "cell": [[6, 0, 0], [0, 6, 0], [0, 0, 6]]}
```

```bash
mandacaru h2.json --grid-spacing 0.35
```

```text
{
  "energy_ev": -25.87060596308835,
  "converged": true,
  "method": "dft"
}
```

The report has three keys: `energy_ev`, `converged` (the calculation's success flag)
and `method`. With `--forces` it gains `forces_ev_per_angstrom`, one row per atom.
`--output FILE` writes the same JSON to a file and still prints it.

## Flags

| Flag | Default | Sets |
| --- | --- | --- |
| `geometry` (positional) | none | Geometry JSON, or any file ASE can read (needs the `ase` extra). Optional with `--load-hamiltonian` or `--load-crystal`. |
| `--method` | `dft` | `method`: one of `rhf`, `uhf`, `ghf`, `dft`, `vqe`, `adapt-vqe`, `subspace-vqe`, `subspace-adapt-vqe`, `mcas-vqe`, `valqa`. See [the methods table](#methods-table). |
| `--xc` | `lda` for `dft` | `xc`: `lda`, `pbe`, `r2scan`, `hse06`, `pbe+rvv10` or `r2scan+rvv10`. Only `dft` uses it; with another method the calculator raises `ValueError` ("does not use: xc"). |
| `--basis` | `HAO` | `basis`, as a name only (`HAO`, `PAW-LCAO`, a Gaussian recipe such as `STO-3G`). A basis with options needs the Python interface. |
| `--grid-spacing` | `0.2` | `grid_spacing`, in angstrom. |
| `--threads` | automatic | `resources["threads"]`. |
| `--memory-mb` | `512` | The calculation's memory budget in mebibytes (`resources["memory_limit_bytes"]`). |
| `--forces` | off | Also compute forces (`calculate` with the `forces` property). |
| `--load-hamiltonian FILE` | none | `load_hamiltonian`: solve a cached qubit Hamiltonian; no geometry is needed. |
| `--save-hamiltonian FILE` | none | `save_hamiltonian`: write the Hamiltonian the run builds. |
| `--load-crystal FILE` | none | `load_crystal`: solve a cached crystal; no geometry is needed. |
| `--output FILE` | none | Write the JSON report to `FILE`. |
| `--version`, `-h` | | Print the version, or the usage. |

Two defaults differ from the Python class. The command runs `dft` with `lda`, while
`Mandacaru()` defaults to `adapt-vqe`; a quantum run from the shell therefore needs
`--method`. And the command never passes `txt=`, so it writes no run log
([the log format](#run-log-reference) applies to Python runs).

## More examples

Hartree-Fock energy and forces into a file:

```bash
mandacaru h2.json --method rhf --grid-spacing 0.35 --forces --output rhf.json
```

Cache the qubit Hamiltonian of a mean-field run, then solve it with ADAPT-VQE
without a geometry:

```bash
mandacaru h2.json --method rhf --grid-spacing 0.35 --save-hamiltonian h2_ham.json
mandacaru --method adapt-vqe --load-hamiltonian h2_ham.json
```

The second command reports `-26.018025475613708` eV for this H2 geometry.

Limit threads and memory:

```bash
mandacaru h2.json --method rhf --grid-spacing 0.35 --threads 2 --memory-mb 1024
```

Solve a crystal saved earlier with `save_crystal=` (see
[the DFT guide](../guide/dft.md)):

```bash
mandacaru --load-crystal si.json
```

## What the command refuses

- No geometry and no cached problem: `error: provide geometry or a cached problem`.
- An ASE file with no unit cell, such as a plain `.xyz`, fails with
  `ValueError: molecular integration requires a nondegenerate cell or an explicit grid`.
  Mandacaru always needs the cell: it is the box the grid is cut from. Use the JSON
  form, or an extended-XYZ file whose comment line carries `Lattice="..."`.
- A geometry is read as JSON only when its name ends in `.json`; any other suffix
  goes through ASE.
- A flag outside the table, including a dry-run flag, is an `unrecognized arguments`
  error. To estimate the register of a molecule without running it, call
  `estimate_qubits` ([API reference](../api.md)).

## Other commands

The package installs one more command, `mandacaru-paw`, which records and checks
where the PAW-LCAO datasets live. Give it at least one action; they run in the order
`--set`, `--set-upaw`, `--status`.

| Flag | Effect |
| --- | --- |
| `--set DIR` | Record `DIR`, a checkout of `mandacaru-paw`, as `MANDACARU_PAW_PATH` in `~/.zshrc` or `~/.bashrc`, asking before it replaces a different value. |
| `--set-upaw DIR` | The same for the optional UPAW-LCAO library (`MANDACARU_UPAW_PATH`). |
| `--status` | Draw a table of every dataset set each library variable reaches. The exit status is 1 while a required library is unset or not a folder. |
| `--color {auto,always,never}` | Color the status table (default: only on a terminal with `NO_COLOR` unset). |

A shell that does not read `~/.zshrc` (a script runner, for instance) needs
`MANDACARU_PAW_PATH` on its command line. The datasets themselves are described in
[the installation page](../installation.md) and
[the pseudopotential guide](../guide/pseudopotentials.md).
