# Quantum echoes and dipole response

`QuantumEchoes` applies a weak electric dipole pulse between forward and
backward time evolution:

$$
|\psi_{\rm echo}(t)\rangle = U_S(t)^\dagger K_S(\tau_p)U_S(t)|\psi\rangle,
\qquad U_S(t)\simeq e^{-iHt},\qquad K_S(\tau_p)\simeq e^{-i\tau_p V}.
$$

This is the single kicked echo primitive. The returned overlap and fidelity
are not a four-operator out-of-time-order correlator (OTOC). The
[Quantum Echoes description from Google Research](https://research.google/blog/a-verifiable-quantum-advantage/)
explains the broader use of forward/backward evolution in OTOC experiments.

The weak-dipole `QuantumEchoes` API does **not** reproduce the nested OTOC protocol in
[Observation of constructive interference at the edge of quantum ergodicity](https://doi.org/10.1038/s41586-025-09526-6)
(Nature 646, 825–830, 2025). Equation (1) of that article evaluates
$\langle[B(t)M]^{2k}\rangle$ with Pauli insertions and repeated echoes. Here the
kick is a weak dipole rotation, and spectroscopy uses a two-point potential
correlation. The API's `order=2` specifies the Suzuki–Trotter approximation;
it does not implement the article's second-order OTOC ($k=2$). The separate
`NestedOTOC` API implements that nested observable, as described below.

The implementation is a classical state-vector simulation with Trotter error.
It stores O($2^n$) amplitudes and scratch space, avoiding dense $2^n\times2^n$
Hamiltonians and propagators. It does not claim a quantum speedup.

## Nested OTOCs from the article

`NestedOTOC` implements Equation (1) using separate signed Pauli strings
$B$ and $M$, with $B^2=M^2=I$:

$$
B(t)=U_S(t)^\dagger B U_S(t),\qquad
U_k=B(t)[M B(t)]^{k-1},\qquad
C^{(2k)}=\langle U_k^\dagger M U_k M\rangle
        =\langle[B(t)M]^{2k}\rangle.
$$

`otoc_order=1` gives the four-operator OTOC, and `otoc_order=2` gives the
article's eight-operator OTOC. `trotter_order=2` independently selects Strang
splitting for every free-evolution leg. These insertions are Pauli operations,
not weak dipole pulses, and have no `tau_p` parameter.

```python
from mandacaru.algorithms import NestedOTOC
from mandacaru.core import PauliSum

# After the H2 ADAPT-VQE calculation (four Jordan-Wigner qubits):
prepared = calc.solver.checkpoint
otoc = NestedOTOC(
    prepared.hamiltonian,
    butterfly=PauliSum({"IIIZ": 1}),
    measurement=PauliSum({"ZIII": 1}),
    trotter_order=2,
)
result = otoc.run(prepared, time=1.0, otoc_order=2, steps=40)
print(result.correlator)                 # Dimensionless C^(4), possibly complex
print(result.measurement_expectation)   # Physical <M> after the nested sequence
circuit = otoc.circuit(time=1.0, otoc_order=2, steps=40)
```

Every backward leg is the exact adjoint of its discrete forward approximation,
including for first-order Trotter splitting. Input states may be normalized
vectors, checkpoints, or checkpoint paths. The signed Pauli strings may span
multiple qubits but must share the Hamiltonian's register and mapping. They
must preserve any tapered symmetry sectors needed by the calculation.

For an initial $M$ eigenstate with eigenvalue $m=\pm1$, the measured expectation
gives $C^{(2k)}=m\langle M\rangle_{U_k\psi}$. For generic states, including
generic ADAPT states, this shortcut is invalid: the simulator evaluates the
overlap of the two branches $U_k|\psi\rangle$ and $U_kM|\psi\rangle$, retaining
the complex result. `circuit()` exports only $U_k$, without preparation or
readout. Hardware measurement for a generic state needs an additional
interferometric protocol. Mixed-state values can be obtained by probability-
weighted averages of pure-state correlators; the API does not accept a density
matrix. Averaging a complete orthonormal basis gives the normalized trace.

The complete three-dimensional H2 example reuses the molecular builder and
ADAPT checkpoint from the dipole example:

```console
conda run -n mandacaru python examples/nested_otoc.py
```

It writes the standard `[BASIS]` and ADAPT report, then `[NESTED OTOC]` with
compact $k=1$ and $k=2$ sample rows. The full signed FFT goes to
`otoc_spectrum.csv`. `parse_output("output.txt")["nested_otoc"]` reads the
metadata and sample table. Its occupation-parity insertions probe electronic
correlations; they are not electric dipole operators.

## What an OTOC spectrum reveals about a Hamiltonian

```python
spectrum = otoc.spectrum(
    prepared, time_step=0.2, num_samples=128,
    steps_per_sample=2, otoc_order=2,
)
spectrum.to_csv("otoc_spectrum.csv")
```

The negative-phase FFT reports signed angular frequencies and their energy
equivalents with $\hbar=1$. Its amplitudes have atomic-time units because
$C^{(2k)}$ is dimensionless. A Hann window is the default; `window="none"`
selects the unwindowed transform. Neither the mean nor an assumed elastic line
is subtracted. The CSV labels frequencies and amplitude units explicitly.

An OTOC Fourier spectrum is **not generally an excitation spectrum**. In an
energy basis, $B(t)_{ab}=B_{ab}e^{i(E_a-E_b)t}$. The four-operator OTOC therefore
contains frequencies of the form

$$
\omega=(E_a-E_b)+(E_c-E_d),
$$

weighted by matrix elements and the initial state. Order $k$ contains sums of
$2k$ such differences. Interference, selection rules, and cancellation determine
which lines appear. Higher harmonics can exceed the Hamiltonian's spectral
width, so the sampling interval must resolve these combined frequencies.

For example, $H=X\otimes X/2$ has energies $\pm1/2$ Ha, hence gap 1 Ha. With
$B=I\otimes Z$, $M=Z\otimes I$, and initial $|00\rangle$:
$C^{(2)}(t)=\cos(2t)$ and $C^{(4)}(t)=\cos(4t)$. Their FFT lines are at
$\pm2$ and $\pm4$ Ha-equivalent frequencies, respectively, rather than the
Hamiltonian's 1 Ha gap. The tests check this example explicitly.

Nested OTOCs can constrain an unknown Hamiltonian through their sensitivity to
its couplings: fit predicted correlator traces for several operators, times,
and states to observed traces, then calculate the fitted Hamiltonian's spectrum.
The paper demonstrates this Hamiltonian-learning strategy. This implementation
provides the forward correlator calculation; parameter fitting is not included.
The inverse problem need not have a unique solution, and $H$ and $H+cI$ give
identical OTOCs, so absolute energies need an independent reference.

For ground-state excitation spectroscopy, the existing dipole correlation
$\langle0|V V(t)|0\rangle$ remains more directly interpretable: its lines are
$E_n-E_0$ for states coupled by $V$. It can miss dipole-forbidden transitions.
The RHF HOMO–LUMO orbital gap is a separate mean-field quantity.

Nested sampling recomputes each repeated sequence at $t_j=j\Delta t$, with
`j * steps_per_sample` subdivisions per leg to hold Trotter step size fixed.
Its work grows quadratically with sample count and linearly with OTOC order;
it does not have the incremental two-point algorithm's linear sampling cost.
Only current state branches and scalar samples are retained. This API does not
implement the paper's Pauli-averaging interference decomposition or hardware
error mitigation.

## Passing an ADAPT-VQE state

After running an existing `Mandacaru(method="adapt-vqe", ...)` calculator:

```python
from mandacaru.algorithms import QuantumEchoes, time_evolve
from mandacaru.core import electric_dipole_potential

ground_result = calc.run()
prepared = calc.solver.checkpoint
H = prepared.hamiltonian                 # PauliSum in Hartree

# Pure time propagation of the prepared approximate ground state.
psi_t = time_evolve(prepared, H, time=1.0, steps=20, order=2)

# r_mo has shape (3, n_spatial, n_spatial), in Bohr, in H's orbital basis.
V = electric_dipole_potential(r_mo, field=[0.0, 0.0, 0.01],
                             mapping=prepared.mapping,
                             num_particles=prepared.num_particles)
echoes = QuantumEchoes(H, V, order=2)
result = echoes.run(prepared, time=1.0, tau_p=0.01, steps=20)
print(result.fidelity, result.response, result.correlation)
```

`prepared` can also be a checkpoint filename or a normalized one-dimensional
complex vector, such as `prepared.state_vector()`. That method reconstructs
the full register even when ADAPT optimized within a particle-number sector.
For a full-register ansatz, `calc.solver.ansatz.state(ground_result.optimal_parameters)`
is another source of amplitudes. `ADAPTVQEResult` itself contains optimization
results, not state amplitudes. Input arrays are never modified or silently
renormalized.

The complete runnable example is `examples/quantum_echoes.py`, using an H₂
molecule with a 0.74 Å bond oriented along $(1,1,1)$. It evaluates the
interacting molecular Hamiltonian and all three position matrices on a
nonperiodic three-dimensional Cartesian grid. Both operators are transformed
to the same molecular-orbital basis, including the Löwdin orthogonalization
of the original atomic orbitals. The default electric field points along the
bond; its direction and strength can be set independently.
One hydrogen 1s orbital per atom gives four qubits; this minimal basis is a
demonstration of longitudinal response, not a converged optical spectrum.
The example follows the same ASE calculator workflow as `02_ADAPTVQE_LiH.py`:
`atoms.calc = Mandacaru(basis={"name": "HAO"}, ...)` followed by
`atoms.get_total_energy()`. The standard ADAPT report goes to `output.txt`
in the working directory, including `[BASIS]`, integration timings, and circuit
metrics. A `[QUANTUM ECHOES]` block follows in the same report, recording the
propagation settings, field, units, pulse bounds, and a `samples` table of
echo results. It also contains the Fourier sampling settings, resolution,
Nyquist energy, and a `peaks` table in Hartree and eV. Both tables have padded
columns and a header delimiter. The complete signed spectrum is exported to
`spectrum.csv` beside `output.txt`, rather than printed in the main report.
The terminal shows a short summary, the RHF HOMO–LUMO gap, and excitation peaks.
`parse_output("output.txt")["quantum_echoes"]` reads the metadata and tables;
its `spectrum_file` field points to the CSV file.
The dipole calculation reuses the integral engine and orbital rotation
retained by that calculator. The standard builder also supplies the same
basis defaults and nuclear-cusp regularization as the other ASE examples.

```console
conda run -n mandacaru python examples/quantum_echoes.py
```

For an x-polarized field of magnitude 0.01 atomic units:

```console
conda run -n mandacaru python examples/quantum_echoes.py --field-direction 1 0 0 --field-strength 0.01
```

`--output /path/to/output.txt` also places `spectrum.csv` in that directory.

## Dipole coupling and pulse size

In atomic units, $\boldsymbol\mu=\boldsymbol\mu_{\rm nuc}-\sum_i\mathbf r_i$,
so the electric potential is

$$
V=-\mathbf E\cdot\boldsymbol\mu
 =\sum_{pq}\mathbf E\cdot\langle p|\mathbf r|q\rangle a_p^\dagger a_q
 -\mathbf E\cdot\boldsymbol\mu_{\rm nuc}\,I.
$$

`electric_dipole_potential` takes three Hermitian position matrices in an
**orthonormal** basis. Transform atomic-orbital integrals using
`C.conj().T @ r_ao[axis] @ C`, then select the same active space used for `H`.
The default duplicates spatial matrices into alpha and beta spin blocks.
Set `spatial_orbitals=False` for matrices already expressed in spin orbitals.
The field magnitude is retained. The optional `nuclear_dipole` contributes a
global phase; frozen-core dipoles may be included in that constant as well.

Alternatively, supply a real Cartesian **unit vector** and a nonnegative
field strength separately:

```python
V = electric_dipole_potential(
    r_mo, field_direction=[1.0, 0.0, 0.0], field_strength=0.01,
    mapping=prepared.mapping, num_particles=prepared.num_particles,
)
```

The field is `field_strength * field_direction` and couples as
$V=-E_0\hat{\mathbf e}\cdot\boldsymbol\mu$. Directions must have unit norm
within 1e-8; the zero vector, non-unit vectors, and non-finite inputs are
rejected. Supply both directional parameters together, without `field=`.
The existing Cartesian `field=` input remains available. To reverse a field,
reverse its direction; zero strength is allowed.

The Hamiltonian, potential, and state must use the same orbital basis,
fermion encoding, and symmetry reduction. Width checks cannot identify two
different bases with the same dimension. For additional symmetry tapering,
transform the dipole consistently. If the dipole connects sectors removed
by tapering, use a larger register that retains those transitions.

The evolution API always uses **Hartree** and **atomic time units**
($\hbar=1$), independently of the eV output setting of the ADAPT calculator.
One atomic time unit is approximately 0.02418884 fs. `tau_p` is the signed
pulse duration; its effect is the dimensionless product $\tau_p V$.

To enforce a small perturbation, each run requires

$$
|\tau_p|\sum_P|v_P|\leq\epsilon,\qquad
\|\tau_p V\|\leq |\tau_p|\sum_P|v_P|,
\qquad 0<\epsilon\leq0.1.
$$

`epsilon` is `max_perturbation` (default 0.1). The default `tau_p=1e-3`
is still validated against the supplied operator. Oversized pulses raise an
error giving the permitted `abs(tau_p)`; they are never clipped. Tighten
`max_perturbation` and compare responses at `tau_p` and `tau_p/2` to assess
linear response. Negative pulses and `tau_p=0` are supported. Increasing
`kick_steps` improves the pulse's Trotter accuracy but cannot evade the
smallness constraint. The Pauli bound includes identity terms and can be
conservative.

This length-gauge builder applies to molecules and finite solid clusters.
It does not implement periodic bulk polarization: a position operator built
from wrapped cell coordinates is not an appropriate bulk coupling. A
consistent projected dipole operator can be supplied directly as a
`PauliSum`; a bulk gauge-field/current response requires a separate coupling
implementation. See the
[Octopus solid optical-response tutorial](https://www.octopus-code.org/documentation/15/tutorial/periodic_systems/optical_spectra_of_solids/)
for the distinction between finite-system dipoles and bulk current response.

## Propagation order and reversibility

`SuzukiTrotter` uses Strang splitting by default:

$$
S_2(\Delta t)=\left(\prod_{j=1}^m e^{-ih_jP_j\Delta t/2}\right)
             \left(\prod_{j=m}^1 e^{-ih_jP_j\Delta t/2}\right).
$$

Higher even orders use the symmetric recursion

$$
S_{2k}(\Delta t)=S_{2k-2}(p_k\Delta t)^2
 S_{2k-2}((1-4p_k)\Delta t)S_{2k-2}(p_k\Delta t)^2,
\qquad p_k=(4-4^{1/(2k-1)})^{-1}.
$$

`order=1` also permits Lie–Trotter propagation. Higher odd orders are rejected.
The `steps` parameter divides the full interval into equal slices; higher
orders increase the number of rotations rapidly. Check convergence with
increasing `steps` and/or `order`. The
[Qiskit Suzuki–Trotter documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.synthesis.SuzukiTrotter)
describes these product formulas and their references.

The reverse leg reverses the rotation sequence and negates its angles,
giving the exact adjoint of the *approximation*, even for first order.
Consequently the zero-pulse echo returns the input to numerical precision.
This reversibility check alone does not establish accuracy against exact
Hamiltonian evolution; the tests check both properties independently.

For repeated propagation, reuse `SuzukiTrotter(H, order=4).evolve(...)`.
Its `circuit(time, steps=...)` method exports the same evolution as Qiskit
Pauli rotations for composition with state preparation and measurement.
State-vector ordering matches Mandacaru's `PauliSum.to_matrix()`.

## Excited-state information

Results include the final state, complex return amplitude, fidelity, pulse
bound, signed response

$$
\Delta V(t)=\langle\psi_{\rm echo}(t)|V|\psi_{\rm echo}(t)\rangle
 -\langle\psi|V|\psi\rangle,
$$

and the two-point correlation

$$
C(t)=\langle\psi|V U_S(t)^\dagger V U_S(t)|\psi\rangle.
$$

For an exact ground state and exact time evolution,

$$
C(t)=\sum_n |\langle n|V|0\rangle|^2e^{+i(E_n-E_0)t}.
$$

Subtract $\langle V\rangle^2$ to remove the elastic contribution. Fourier
analysis using $e^{-i\omega t}$ then locates dipole-allowed excitation gaps;
finite sampling time limits frequency resolution. Correlation is independent
of pulse strength and is evaluated directly by the state-vector simulator.
`response/tau_p` approaches $2\operatorname{Im}C(t)$ for small pulses. Since
`V` already contains the applied field, these are potential response and
correlation in Hartree and Hartree squared, respectively, not an automatically
normalized absorption cross section.

For an exact eigenstate the return fidelity alone is time-independent.
Excitation analysis must therefore use a response or correlation. With an
approximate ADAPT state, residual excited-state components and Trotter error
also contribute to the signal. The conventional weak-kick spectroscopy
approach is discussed in the
[Octopus time-propagation tutorial](https://www.octopus-code.org/main/tutorials/2-optical-response/1-optical_spectra_from_time_propagation.html).

## Fourier energy spectrum

The spectrum is computed by `QuantumEchoes.spectrum`, rather than inferred
from the time-independent return fidelity:

```python
spectrum = echoes.spectrum(
    prepared, time_step=0.5, num_samples=1024,
    steps_per_sample=10, window="hann",
)
for i in spectrum.peaks():
    print(spectrum.energies_ev[i], spectrum.intensities[i])
```

The method samples $C(t)$ by advancing $U(t)|\psi\rangle$ and
$U(t)V|\psi\rangle$ incrementally. Their contraction
$\langle U(t)V\psi|V U(t)\psi\rangle$ equals the echo correlation and costs
only two evolving state vectors, with work linear in the number of samples.
Every sampling interval uses `steps_per_sample` Trotter slices. This keeps
the time step fixed throughout the trace; a fixed total number of slices for
progressively longer independent trajectories would degrade accuracy.

By default the method removes $\langle V\rangle^2$ and applies a Hann window,
then computes

```{math}
\widetilde C(E_k)=\Delta t\sum_{j=0}^{N-1}w_j
  [C(j\Delta t)-\langle V\rangle^2]e^{-2\pi i jk/N},
\qquad E_k=\frac{2\pi k}{N\Delta t}.
```

The returned energy grid is sorted and includes negative energies. Positive
energies correspond to excitation gaps for a ground-state input. Negative
components may occur with an approximate/nonstationary state. The result
retains complex Fourier amplitudes, their magnitudes (`intensities`), the
raw correlations, and their sampling times. `fourier_spectrum(correlation,
time_step, elastic=..., window=...)` also transforms a supplied uniformly
sampled correlation vector starting at time zero.

The default example samples 1024 times at intervals of 0.5 atomic time units.
Its energy-bin spacing is $2\pi/(N\Delta t)$, approximately 0.334 eV; the Hann
window broadens peaks further. The Nyquist energy is $\pi/\Delta t$.
Increase the total observation time to separate close transitions and reduce
the sampling interval to avoid aliasing high-energy transitions. Also check
convergence in `steps_per_sample` and propagation order. No zero padding or
peak fitting is used; reported peak energies are sampled Fourier bins.

The reported spectrum is the magnitude of a potential-correlation Fourier
transform, in Hartree. It contains the chosen field's scaling and is **not**
a normalized oscillator-strength or absorption spectrum. Direct correlation
sampling is the linear-response calculation and is independent of `tau_p`;
the finite-pulse echo table separately records the validated pulse duration.
Only dipole-allowed transitions represented by the orbital basis appear.

## Spectrum export and orbital gap

The reusable export method writes a plain numeric CSV, with one header and
no comment lines:

```python
spectrum.to_csv("spectrum.csv")
```

Its columns are `energy_ha,energy_ev,magnitude_ha,fft_real_ha,fft_imag_ha`.
Signed energies and both Fourier components are retained. The report writer
`append_quantum_echoes(..., spectrum=spectrum)` exports this file automatically
and keeps only the peak table in the main log. `spectrum_path=` can select a
different destination. Without it, the CSV is next to the main report, or in
the working directory for a report routed only to standard output. Existing
CSV files are replaced on export. Older reports with inline spectrum tables
remain readable by `parse_output`.

For the closed-shell H₂ example, the HOMO–LUMO gap is calculated from a
converged RHF reference using the same cached one- and two-electron integrals:

```python
rhf = integrals.hartree_fock(n_electrons=2)
homo = rhf.mo_energies[rhf.n_occupied - 1]
lumo = rhf.mo_energies[rhf.n_occupied]
gap_ha = lumo - homo
# Validated properties provide the same quantities:
assert gap_ha == rhf.homo_lumo_gap
```

`RHFResult.homo_energy`, `lumo_energy`, and `homo_lumo_gap` use Hartree.
They require a converged reference with both occupied and virtual orbitals.
The reported RHF gap is a mean-field orbital-energy difference; correlated
neutral excitation energies come from the separate Fourier analysis.

An abbreviated report preview for `--field-direction 1 0 0` is shown below
(selected columns and rounded values; the generated tables retain 12-digit
scientific notation and the real/imaginary amplitude columns):

```text
[QUANTUM ECHOES]
    field_direction: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    field_strength_au: 1.000000000000e-02
    orbital_reference: RHF canonical orbitals
    homo_energy_ha: -6.164138436743e-01
    lumo_energy_ha: 3.999367660856e-01
    homo_lumo_gap_ha: 1.016350609760e+00
    homo_lumo_gap_ev: 2.765630900352e+01
    spectrum_file: spectrum.csv
    samples:
        time    tau_p   fidelity      response
        ------------------------------------------
        0.000   0.010   0.9999999941  0.000000e+00
        0.500   0.010   0.9999999941  4.334868e-07
    peaks:
        energy_ha   energy_ev    magnitude
        -------------------------------------
        0.76085447  20.70390488  1.320543e-02
```

The example's numbers depend on its minimal orbital basis and integration
grid; they are not a converged prediction of the molecular spectrum.
