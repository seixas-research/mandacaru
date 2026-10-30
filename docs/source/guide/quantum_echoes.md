# Quantum echoes and dipole response

Time propagation runs natively, but the echo and response orchestration around
it is listed as open work in the [native coverage](../development/native_cutover.md).
Treat the tools below as research instruments, check them against a small exact
calculation, and read their status there before building on them.

`QuantumEchoes` applies a weak potential pulse between forward and backward time
evolution of a prepared state:

$$
|\psi_{\rm echo}(t)\rangle = U_S(t)^\dagger K_S(\tau_p)U_S(t)|\psi\rangle,
\qquad U_S(t)\simeq e^{-iHt},\qquad K_S(\tau_p)\simeq e^{-i\tau_p V}.
$$

This is the single kicked echo primitive. Its return fidelity is not a
four-operator out-of-time-order correlator (OTOC); `NestedOTOC`, below, is the
OTOC. The simulation is classical and exact up to the Suzuki-Trotter error of
the propagation: it stores $O(2^n)$ amplitudes and makes no claim of a quantum
speedup. The Hamiltonian, the potential and the state must share one orbital
basis, fermion encoding and register. Everything is in Hartree and atomic time
units ($\hbar=1$), whatever the calculator's `atomic_units` says.

## An echo from an ADAPT-VQE state

The prepared state is the calculator's checkpoint, which carries the qubit
Hamiltonian. The potential is any `PauliSum` on the same register; here a
Pauli operator stands in for a dipole, so this is a demonstration, not an
optical spectrum.

```python
import numpy as np
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.core import PauliSum
from mandacaru.workflows.quantum_echoes import QuantumEchoes

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
atoms.get_potential_energy()

prepared = atoms.calc.solver.checkpoint
echoes = QuantumEchoes(prepared.hamiltonian, PauliSum({"ZIII": 1.0}))
for time in (0.0, 1.0, 2.0, 4.0):
    result = echoes.run(prepared, time, tau_p=0.01, steps=40)
    print(f"t = {time:.1f}  fidelity = {result.fidelity:.10f}  "
          f"response = {result.response:+.3e} Ha")

spectrum = echoes.spectrum(prepared, time_step=0.5, num_samples=1024)
print("Peaks (eV):", np.round(spectrum.energies_ev[spectrum.peaks()], 4))
```

`prepared` may also be a checkpoint path or a normalized state vector such as
`prepared.state_vector()`. The script is `examples/quantum_echoes/02_echo_spectrum_h2.py`.

A run returns the final state, the return amplitude and fidelity, the signed
response

$$
\Delta V(t)=\langle\psi_{\rm echo}(t)|V|\psi_{\rm echo}(t)\rangle
 -\langle\psi|V|\psi\rangle,
$$

the pulse bound `perturbation_bound` $=|\tau_p|\sum_P|v_P|$ (a run whose bound
exceeds `max_perturbation`, 0.1 by default, raises `ValueError` and names the
largest `|tau_p|` allowed; pulses are never clipped), and the two-point
correlation

$$
C(t)=\langle\psi|V U_S(t)^\dagger V U_S(t)|\psi\rangle .
$$

For an exact ground state and exact evolution
$C(t)=\sum_n |\langle n|V|0\rangle|^2e^{+i(E_n-E_0)t}$. For an exact eigenstate
the return fidelity alone is time-independent, so excitation analysis uses the
response or the correlation. With an approximate ADAPT state, residual
excited-state components and Trotter error also contribute to the signal.

## The spectrum

`echoes.spectrum` samples $C(t)$ by advancing $U(t)|\psi\rangle$ and
$U(t)V|\psi\rangle$ incrementally, so the work is linear in the number of
samples, with `steps_per_sample` Trotter slices per interval. It subtracts the
elastic part $\langle V\rangle^2$, applies a Hann window (`window="none"`
disables it) and returns an `EchoSpectrum`:

* `energies` and `energies_ev`, signed excitation gaps (not total energies);
  positive ones are excitation gaps of a ground-state input;
* `amplitudes`, `intensities` (their magnitudes, in Hartree) and the raw
  `correlation` with its `times`;
* `resolution` $=2\pi/(N\Delta t)$, before window broadening, and
  `nyquist_energy` $=\pi/\Delta t$, above which gaps alias;
* `peaks()`, the sampled bins that are local maxima; no zero padding or peak
  fitting is used;
* `to_csv(path)`, a plain numeric file with the columns
  `energy_ha,energy_ev,magnitude_ha,fft_real_ha,fft_imag_ha`.

`fourier_spectrum(correlation, time_step, elastic=..., window=...)` in the same
module transforms a correlation you supply, sampled uniformly from $t=0$. The
intensities carry the pulse's scale and are **not** oscillator strengths or an
absorption cross section. Only transitions the orbital basis represents appear,
and only dipole-allowed ones for a dipole potential. Increase the observation
time to separate close lines and reduce the interval to avoid aliasing.

## A dipole potential

In atomic units $\boldsymbol\mu=\boldsymbol\mu_{\rm nuc}-\sum_i\mathbf r_i$ and
$V=-\mathbf E\cdot\boldsymbol\mu$. `electric_dipole_potential` in `mandacaru.core`
builds it from three Hermitian position matrices of shape `(3, m, m)` in Bohr,
in an **orthonormal** basis and the same active space as the Hamiltonian
(transform atomic-orbital integrals with `C.conj().T @ r @ C`). The calculator
does not export position integrals, so the matrices are yours to supply.

```python
import numpy as np
from mandacaru.core import electric_dipole_potential

# Illustrative matrices for the two-orbital H2 register above; a calculation
# takes them from its own integrals, in the orbitals of its Hamiltonian.
r_mo = np.zeros((3, 2, 2))
r_mo[2] = [[-0.7, 0.4], [0.4, 0.7]]
V = electric_dipole_potential(
    r_mo, field_direction=[0.0, 0.0, 1.0], field_strength=0.01,
    mapping=prepared.mapping, num_particles=prepared.num_particles)
echoes = QuantumEchoes(prepared.hamiltonian, V)
print(echoes.run(prepared, 1.0, tau_p=0.1, steps=40).response)
```

`field=` takes a Cartesian vector instead of a direction and a strength; the two
spellings cannot be mixed, and a direction must have unit norm to `1e-8`. Spatial
matrices are duplicated into alpha and beta blocks unless
`spatial_orbitals=False`. `nuclear_dipole` adds a constant, a global phase. The
builder applies to molecules and finite clusters: it does not construct a bulk
polarization or current response from periodic coordinates.

## Nested OTOCs

`NestedOTOC` evaluates the eight-operator correlator of Equation (1) of
[Observation of constructive interference at the edge of quantum ergodicity](https://doi.org/10.1038/s41586-025-09526-6)
(Nature 646, 825-830, 2025) with separate signed Pauli strings $B$ and $M$,
$B^2=M^2=I$:

$$
B(t)=U_S(t)^\dagger B U_S(t),\qquad
U_k=B(t)[M B(t)]^{k-1},\qquad
C^{(2k)}=\langle U_k^\dagger M U_k M\rangle=\langle[B(t)M]^{2k}\rangle .
$$

`otoc_order=1` gives the four-operator OTOC and `otoc_order=2` the eight-operator
one. The insertions are Pauli operations, not weak pulses, so there is no
`tau_p`. The weak-dipole `QuantumEchoes` is a different protocol.

```python
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.core import PauliSum
from mandacaru.workflows.nested_otoc import NestedOTOC

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
atoms.get_potential_energy()

prepared = atoms.calc.solver.checkpoint
otoc = NestedOTOC(prepared.hamiltonian,
                  butterfly=PauliSum({"IIIZ": 1}),
                  measurement=PauliSum({"ZIII": 1}))
for time in (0.5, 1.0, 2.0):
    result = otoc.run(prepared, time=time, otoc_order=2, steps=40)
    print(f"t = {time:.1f}  C = {result.correlator:.6f}")
```

This is `examples/quantum_echoes/01_nested_otoc_h2.py`. The correlator is
dimensionless and may be complex: for a generic state the simulator evaluates
the overlap of the branches $U_k|\psi\rangle$ and $U_kM|\psi\rangle$ rather than
the eigenstate shortcut $C^{(2k)}=m\langle M\rangle_{U_k\psi}$ (valid only when
the initial state is an $M$ eigenstate of eigenvalue $m=\pm1$). The Pauli strings
must preserve the symmetries of the register. The paper's Pauli-averaging
decomposition, error mitigation and Hamiltonian-learning fit are not
implemented. An OTOC Fourier spectrum is not generally an excitation spectrum:
it contains sums of $2k$ energy differences, so interference and selection rules
decide which lines appear.
