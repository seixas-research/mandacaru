# Dry run: how many qubits will this need?

Before submitting a calculation to a quantum device — or to a state-vector
simulator whose memory doubles with every qubit — you want to know the size of
the register it needs. A **dry run** answers that without computing a single
integral, mapping a Hamiltonian, or executing a circuit.

## On the command line

```console
$ carcara water.xyz --frozen-core --dry-run
Dry run -- no integrals computed, no circuits executed.

  method            : adapt-vqe
  basis             : FAO
  mapping           : jordan_wigner
  source            : geometry
  basis functions   : 7  (O:5, H:1, H:1)
  frozen core       : 1 spatial orbital(s)
  active orbitals   : 6 spatial / 12 spin
  active electrons  : 8  (n_alpha, n_beta) = (4, 4)

  QUBITS REQUIRED   : 12
  with parity 2-qubit reduction : 10

  device            : AER_simulator  (ideal state-vector simulator (default))
  state vector      : 2^12 amplitudes = 64 KiB
```

Every solver option is a flag (`--basis`, `--basis-option size=DZP`,
`--charge`, `--magmoms`, `--frozen-core`, `--mapping`,
`--device`, ...), so the estimate is for exactly the run you would launch by
dropping `--dry-run`. The geometry can be a file (`.xyz`, `.cif`, `POSCAR`, ...)
or an ASE `g2` molecule name (`H2O`, `LiH`, `NH3`, ...). With
`--load-hamiltonian` only the cache file's header is read and no geometry is
needed at all.

Naming a real device compares the register with its capacity:

```console
$ carcara H2O --cell 8 --basis NAO --basis-option size=DZP --device braket-ionq-aria --dry-run
  ...
  QUBITS REQUIRED   : 46
  device            : braket-ionq-aria  (IonQ Aria-1 trapped-ion QPU (25 qubits))
  device capacity   : 25 qubits  -> DOES NOT FIT
```

The `ibm-quantum` label (the least-busy IBM processor) does not fix a register
size, so pass `--device-qubits 127` to compare against a specific processor. `--json` prints the estimate as machine-readable JSON.

## From Python

```python
from ase.build import molecule
from carcara.algorithms import Carcara, estimate_qubits

water = molecule("H2O"); water.center(vacuum=3.0)

# One-off, on an existing calculator:
calc = Carcara(frozen_core=True)
estimate = calc.dry_run(water)          # -> QubitEstimate
print(estimate.n_qubits)                # 12

# Or make every evaluation a dry run (energies come back as NaN):
water.calc = Carcara(method="adapt-vqe",
                     frozen_core=True,
                     dry_run=True)
water.get_potential_energy()            # nan
water.calc.dry_run_result.summary()

# Or the bare function, mirroring the driver keywords:
estimate_qubits(water, basis={"name": "NAO-AE", "tier": 1}, device="braket-iqm-garnet")
```

`dry_run=True` is honoured by every driver: the ASE hook stores the
{class}`~carcara.algorithms.dry_run.QubitEstimate` on `dry_run_result` and
reports `NaN`, and a direct-mode `run()` returns the estimate instead of a
result. Nothing expensive is touched — the test suite pins this by making the
Hamiltonian builder, the timings and the circuit providers raise.

## How the count is made

```{math}
N_\text{qubits} = 2\,(M_\text{basis} - M_\text{frozen}),
```

one qubit per active **spin-orbital**: the basis family is instantiated exactly
as a run would (so `size`, polarisation and cutoff options are honoured) and
its functions are *counted* rather than integrated; the frozen core removes
doubly occupied spatial orbitals; the plane-wave family counts plane waves
below the cutoff.
Jordan-Wigner, parity and Bravyi-Kitaev all use this many qubits — the parity
mapping's optional two-qubit symmetry reduction is reported separately.

See {mod}`carcara.algorithms.dry_run` and {mod}`carcara.cli`.
