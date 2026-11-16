(real-time)=
# Real-time TD-DFT of molecules

After a converged `method="dft"` run on a molecule, `calc.get_real_time(...)` kicks the
electrons with a short pulse and follows the Kohn-Sham orbitals in time. The same call on a
`method="rhf"` or `"uhf"` run propagates the Hartree-Fock orbitals (time-dependent
Hartree-Fock). The induced dipole as a function of time is the signal, and its Fourier
transform is the absorption spectrum. The numbers are computed in Rust
(`molecular::realtime`, `math::propagator` and `models::time_spectrum`); the returned
`mandacaru.analysis.real_time.RealTime` view converts units and holds no arithmetic of its
own. The references that run are `method="dft"` with `xc="lda"`, `"pbe"` or `"hse06"`, with
or without `hubbard_u`, and `method="rhf"` or `"uhf"`, each closed shell or open shell.
Only molecules are supported: [What raises](#real-time-refusals) lists every refused
request, and [the not-yet-available table](#not-yet-available) what is planned (crystals and
`r2scan` among it).

This is the time-domain companion of [linear-response TD-DFT](excited_states.md). It is not
the route to the excited states of a quantum run: a calculator with a quantum method raises
`ValueError`.

## What is computed

The occupied orbitals of the converged reference, `C`, are propagated in the orthonormal
(Loewdin) frame of the basis by

$$
i\,\frac{dC}{dt} = F[D(t)]\,C, \qquad D = C C^\dagger ,
$$

where `F` is the reference's own Fock matrix, built by the same code as the SCF and the
linear response, now at a complex density matrix. The nuclei are clamped and the basis is
fixed, so the overlap is constant and no basis-derivative term exists. On a PAW-LCAO basis
the one-center terms are linearized at the reference atom, as in the ground state, and live
in a constant matrix: nothing in the spheres is recomputed per step. For `hse06` the
screened exchange is built once and applied to the whole complex density every Fock build;
with `hubbard_u` the +U potential is formed from the complex shell occupations, so the
current through a shell is part of it.

**The kick.** With `kick="density"` (the default) every orbital is multiplied at `t = 0+` by
`exp(-i kappa e.r)`, `kappa` being `strength` (1/Angstrom) and `e` the unit vector along
`direction`. This is the length gauge. The operator `e.r` is the one that gives
`get_excitations` its transition dipoles, including the PAW one-center part. For `t > 0`
there is no external term: the dynamics is the free evolution of the kicked state.

- `kick="spin"` multiplies the up orbitals by `exp(-i kappa e.r)` and the down orbitals by
  `exp(+i kappa e.r)`, and the signal is the spin dipole. A closed shell is expanded to two
  equal channels internally; the kick then excites the triplets with `M_s = 0`.
- `gauge="velocity"` leaves the orbitals alone and switches on a constant vector potential
  of strength `strength` along `e` at `t = 0+`. It has the same poles as the length gauge. It
  is a diagnostic: the `momentum` trace of the view is its velocity-form observable, and
  the spectra methods still transform the dipole trace. A spin kick is a length-gauge kick,
  so `kick="spin"` with `gauge="velocity"` raises `ValueError`.

**The signal.** `view.dipole` is the induced electronic dipole, `-tr[D r]` minus its value
before the kick (the nuclear part is constant), zero at the kick; `view.along` is its
component along the kick direction. For a spin kick the traces are those of the spin dipole.

**The spectrum.** With a damping window `w(t) = exp(-gamma t)` the view forms

$$
\alpha(\omega) = \frac{1}{\kappa}\int_0^T \delta\mu(t)\,e^{i\omega t}\,w(t)\,dt,
\qquad
S(\omega) = \frac{2\omega}{\pi}\,\mathrm{Im}\,\alpha(\omega),
$$

by the trapezoid rule, and `window="gaussian"` gives Gaussian lines of standard deviation
`damping` instead. With the exponential window `polarizability` is `alpha(omega + i gamma)`
exactly: Lorentzian lines of half width `damping`. A discrete spectrum gives
`S = sum_n f_n delta(omega - omega_n)` with `f_n = 2 omega_n |<0|e.r|n>|^2`: the lines of
`strength_function` have the areas of the oscillator strengths **along the kick direction**,
where `Excitations.oscillator_strengths` is the average over three directions. The integral
of `S` is the oscillator-strength sum of the finite basis, not the electron count
([the sum rule](#real-time-linear-response)).

Frequencies and `damping` are in eV (Hartree with `atomic_units=True`), `strength_function`
is in 1/eV and `polarizability` in Angstrom^3. The trace is finite, so a line cannot be
narrower than about `2 pi / T` for a run of duration `T`, and `damping` should be large
enough that `exp(-gamma T)` (with `gamma` converted to an inverse time) is small: the view does not report the window's weight on the
last sample, and a window that is not small leaves truncation ripples on the spectrum.
Check a spectrum by doubling `duration`.

## A first calculation

```python
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
atoms.center()
atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.3)

rt = atoms.calc.get_real_time(duration=6.0, time_step=0.01, direction="z")
print(rt.step_width, rt.step_width_bound)     # dt * W, and the largest value allowed
print(rt.energy_drift, rt.worst_norm_error)   # conservation checks of the run

energies = np.linspace(5.0, 40.0, 3501)       # eV
spectrum = rt.strength_function(energies, damping=0.5)   # 1/eV
print(energies[int(np.argmax(spectrum))])     # the strongest line, eV

casida = atoms.calc.get_excitations(9, approximation="casida")
bright = int(np.argmax(casida.energies * casida.transition_dipoles[:, 2] ** 2))
print(casida.energies[bright])                # the same line from linear response
atoms.calc.close()
```

If no calculation has run yet, `get_real_time` runs it first; the PAW-LCAO basis needs the
dataset library ([PAW-LCAO bases](pseudopotentials.md)). The kick along the bond excites the
`sigma_u` roots only, and the strongest line of the spectrum sits at the first bright Casida
root. The test suite checks this for this molecule, with a tolerance inside the damping. The
signature is

```python
calc.get_real_time(duration=40.0, time_step=0.005, *, direction="x", strength=0.002,
                   kick="density", gauge=None, propagator="etrs",
                   corrector_tolerance=1e-6, max_energy_drift=0.05)
```

`duration` and `time_step` are in femtoseconds (atomic time units with
`atomic_units=True`); the number of steps is their ratio, rounded, and a duration that rounds to
zero steps raises `ValueError`. `direction` is `"x"`, `"y"`, `"z"` or a three-vector,
which is normalized before use (the run log records it as given). `strength` is in
1/Angstrom (1/Bohr with `atomic_units=True`). `gauge=None` is the length gauge. A tensor
needs one call per axis: `direction="all"` is not offered.

## The view

| Attribute | Content |
| --- | --- |
| `times` | `(nt,)` sample times from the kick, fs |
| `dipole`, `along` | `(nt, 3)` induced dipole and its component along the kick, e Angstrom |
| `momentum` | `(nt, 3)` the electrons' kinetic momentum `sum tr[D (p + A)]`, 1/Angstrom (`hbar = 1`); the velocity-form observable |
| `energy`, `norm_error` | `(nt,)` total electronic energy of the propagated state, eV, and `max abs(C^dagger C - 1)` |
| `ground_energy`, `kick_energy`, `energy_drift` | the energy of the unkicked reference from the same builder, the energy the kick put in, and the largest deviation of the energy from its value after the kick in units of the kick energy |
| `width`, `step_width`, `step_width_bound` | the basis width `W` of the reference Fock matrix (eV), `dt W`, and the propagator's bound on it |
| `aliased_weight` | the bare Kohn-Sham oscillator weight above the Nyquist frequency, as a fraction of the total |
| `builds`, `corrector_iterations` | Fock builds of the run, and the largest corrector iteration count any step used |
| `propagator`, `kick`, `gauge`, `direction`, `strength`, `time_step`, `steps`, `functional` | the request; `functional` is `None` for Hartree-Fock |
| `corrector_tolerance`, `effective_corrector_tolerance`, `max_energy_drift` | the tolerance requested, the one applied ([below](#real-time-propagator)) and the drift limit |
| `strength_function(energies, damping=0.1, window="exponential")` | `S(E)` along the kick, 1/eV |
| `polarizability(energies, damping=0.1, window="exponential")` | complex `alpha(E)`, Angstrom^3 |

Units follow the calculator: atomic units (Hartree, Bohr, atomic time) with
`atomic_units=True`.

(real-time-propagator)=
## The propagator and its limits

`propagator` is `"etrs"` (the default), `"aetrs"` or `"midpoint"`. Every half-step is the
exact exponential of a Hermitian matrix, so the orbitals stay orthonormal to round-off at any
step and a constant `F` would be propagated exactly. That is not unconditional stability:
`F` depends on the propagated state, and the self-consistent map can amplify the error.

- **ETRS**, enforced time-reversal symmetry (`C_{n+1} = exp(-i dt F_{n+1}/2) exp(-i dt F_n/2) C_n`).
  `F_{n+1}` is built at the predicted state and rebuilt until the largest change of any
  density element falls below the corrector tolerance, in at most eight iterations. Converged,
  the step is time-reversal symmetric. It spends two or more Fock builds per step.
- **AETRS**, the same with `F_{n+1}` extrapolated from `2 F_n - F_{n-1}`: one Fock build per
  step, and the time-reversal defect is of third order in `dt` per step.
- **`"midpoint"`**, the exponential midpoint with `F_{n+1/2}` extrapolated as
  `(3 F_n - F_{n-1}) / 2`: one Fock build per step, not time-reversal symmetric.

The first step after the kick is an ETRS step with one corrector iteration for every scheme.
All three are second order in `time_step`: halve it to see the error of a run fall by about
four, as long as it stays above the error of the kick itself (second order in `strength`).

**The `dt W` limit.** `W = e_max - e_min` is the width of the reference Fock matrix in the
orthonormal frame (the largest over the spin channels), reported as `width` and logged with
`dt W` as `step_width`. A run **raises `ValueError`**, naming `time_step`, when `dt W`
exceeds the bound of its propagator: **4 for ETRS, 1 for AETRS and `"midpoint"`**. There is
no override. The bounds were measured on an interacting many-level model and on water:
beyond them the extrapolated schemes lose energy conservation and diverge, while ETRS stays
bounded to its limit. At the default step the production molecules of the
[frozen references](#real-time-frozen) are well inside the ETRS bound; at a step several
times larger they would not be, and the run says so before it starts.

**The corrector tolerance scales with the kick.** `corrector_tolerance` bounds the change of
a density element, an absolute number, while the density the kick induces scales with
`strength`. The run therefore applies

```text
effective_corrector_tolerance = corrector_tolerance * |strength| / default_strength
```

with `default_strength` the default kick, 0.002 per Angstrom. The default kick applies
`corrector_tolerance` exactly, and a weaker kick applies a proportionally tighter tolerance
and costs more corrector iterations. Only ETRS has a corrector; the applied value is
reported for every propagator (`effective_corrector_tolerance` on the view, and the same
key in the run log) but means nothing for `"aetrs"` and `"midpoint"`. An ETRS corrector that
does not reach it in eight iterations raises `RuntimeError` naming `corrector_tolerance`
and the step; the cap is not an option of `get_real_time`.

**Checks that raise.** After a kick the field is constant (none, or a constant vector
potential), so the energy is conserved. The run raises `RuntimeError` when the energy
deviates from its value after the kick by more than `max_energy_drift` kick energies
(`None` checks nothing), and when `max abs(C^dagger C - 1)` exceeds `1e-8` in any step;
orthonormality is never restored by hand. It raises `ValueError` before starting when the
reference is not a fixed point of its own Fock matrix: the occupied orbitals must be the
lowest eigenvectors of it to `1e-6`.

**Aliasing.** A transition above the Nyquist frequency `pi / dt` folds back into the window.
`aliased_weight` is the fraction of the bare Kohn-Sham oscillator weight along the kick above
it. It is a proxy, not a bound (an interacting line can lie above the bare gaps), and it is
reported, not checked.

**Cost.** The cost is the Fock builds: the Poisson solver, the grid scratch and, for
`hse06`, the screening are built once, but the grid and exchange work is redone at every
build. A `hse06` run costs much more per build than `lda` or `pbe`, because the
complex exchange is applied to the screened factors every time.

(real-time-frozen)=
## Frozen reference traces

`test/migration/reference/real_time.json` freezes five short runs, one fs in steps of
0.005 fs (200 steps), ETRS with `corrector_tolerance=1e-6`, PAW-LCAO DZP and
`grid_spacing=0.25` on two threads: the induced dipole along x after a kick of 0.002 per
Angstrom (for `hse06`, 1e-4 per Bohr, the kick of its weak-kick test), sampled every
twenty steps. The replay holds each trace to 1e-6 of its largest value; the `hse06` replay
is in the slow tier. The summaries as frozen (the wall times are those of the freezing
machine, a macOS arm64 one):

| Case | Kick energy (eV) | Energy drift (kick energies) | Fock builds | `W` (eV) | Wall time (s) |
| --- | --- | --- | --- | --- | --- |
| water, `lda` | 1.314e-4 | 5.51e-3 | 551 | 100.4 | 36 |
| water, `pbe` | 1.314e-4 | 5.44e-3 | 550 | 100.7 | 33 |
| water, `hse06` | 1.176e-6 | 5.77e-3 | 572 | 105.5 | 342 |
| NH2 (unrestricted), `lda` | 1.116e-4 | 3.32e-3 | 454 | 83.6 | 22 |
| CuH, `lda`, `hubbard_u` Cu d U = 4 eV | 7.00e-5 | 1.70e-3 | 403 | 162.3 | 17 |

The `hse06` case was frozen again on 2026-11-12, when the corrector tolerance began to scale
with the kick: the weak kick it uses had carried the error of an absolute tolerance. Its
Fock builds went from 402 to 572 and its energy drift from 1.00e-2 to 5.77e-3 kick energies,
and the old and new summaries are kept in the file under `refrozen`. The other four did not
move.

(real-time-linear-response)=
## Relation to linear response

For a weak kick the response is the linear-response one, and in the finite basis the
identity is exact: the propagated Fock matrix and the response kernel of
[`get_excitations`](excited_states.md) are the first and second derivatives of the same
energy. With `kappa` the kick,

$$
\frac{\delta\mu(t)}{\kappa} \to \sum_n 2\,|\langle 0|e\cdot r|n\rangle|^2 \sin(\omega_n t)
$$

over **all** the roots of the full equations, so the peaks of `strength_function` sit at the
roots of `approximation="casida"` (not the Tamm-Dancoff roots, which differ slightly), with
areas equal to the strengths along the kick. The Rust tests check this against the dense
response function of the Casida matrices, with the error falling as `dt^2`, for `lda`, `pbe`,
Hartree-Fock, an unrestricted reference (`lda` and `uhf`), `hse06` and `hubbard_u` (the
Dudarev and Liechtenstein forms); the spin kick of a closed shell is checked against the
triplet matrices (`lda`, Hartree-Fock). The response is linear in the kick up to a
quadratic remainder, which the tests verify too, so keep `strength` small; the `hse06` test
uses 1e-4 per Bohr, weaker than the default, because the model of its semilocal exchange hole
has a kink that a larger displacement reaches. The initial slope of the dipole is the
finite-basis sum `sum_n f_n` over every root: it is the sum rule of this basis, not the
electron count `N` (on a PAW-LCAO basis it can exceed `N` even in a large basis, because the
frozen core is not part of the sum).

The two routes differ in what they resolve. A real-time spectrum has the resolution
`2 pi / T` and the line shape of the window, and merges lines closer than that, where
`get_excitations` returns each root. The velocity-gauge run has the poles of the
length-gauge one; the two gauges agree in weights only as the basis is completed, as for
the oscillator strengths of `get_excitations` ([the gauge limit](#excited-states-limits)).

## Limits of the method

- **Adiabatic kernels.** The kernel has no memory, so the limits of adiabatic linear
  response carry over unchanged: charge-transfer and Rydberg lines are placed poorly, and
  double excitations are absent ([the limits](#excited-states-limits)).
- **A kick, not a pulse.** The only perturbation is the delta kick. Ions stay clamped: no
  ionic motion, no field of finite duration, no restart file. A run that fails late is
  lost.
- **A finite basis and box.** The spectrum is the one of this basis, grid and box; read it as
  in [the excited-states guide](excited_states.md), and converge the basis (diffuse shells)
  before reading Rydberg or near-threshold lines.
- **Open shells.** An open-shell molecule (or `method="uhf"`, `spin=True` or
  `unrestricted=True`) propagates both spin channels; a closed shell propagates one, or two
  equal ones for a spin kick.

(real-time-refusals)=
## What raises

`get_real_time` refuses what it cannot compute and never substitutes another calculation.
Each row was read from the code, not requested on this build. An unknown option value
raises `ValueError` and names the accepted ones.

| Request | What happens |
| --- | --- |
| Any quantum method ([the methods table](#methods-table)) | `ValueError`: the propagation needs a mean-field state |
| `method="ghf"` | `ValueError` |
| A spin-orbit (fully relativistic) dataset, whose Kohn-Sham state is a spinor | `NotImplementedError`; a scalar-relativistic dataset runs |
| `xc="r2scan"` | `NotImplementedError` naming it: its Fock matrix of a kicked state needs the current-density correction of the kinetic energy density, which is not implemented |
| `xc="pbe+rvv10"` | `NotImplementedError`: the nonlocal correlation is not implemented in real time |
| `xc="r2scan+rvv10"` | `NotImplementedError` for the kinetic-density reason of `r2scan`, which is checked first |
| A crystal (`pbc`, or `load_crystal=`) | `NotImplementedError` |
| `precision="float32"` | `ValueError`: the propagation runs in `float64` only |
| An unconverged SCF | `ValueError` |
| Fractional occupations ([Occupations](#dft-degenerate-shell)), or a state that is not aufbau | `NotImplementedError`; give the spin state explicitly instead |
| `rhf` or `uhf` of a molecule whose PAW frozen core has no stored core states or a partially filled subshell | `ValueError`, naming the element: the Hartree-Fock exchange of that core is not available (Kohn-Sham runs on every element) |
| `hubbard_u=` where the ground state refuses it (a screened hybrid, a spin-orbit dataset, `float32`; [DFT+U on molecules](#dft-u-molecules)) | The ground state's own error |
| A reference that is not a fixed point of its Fock matrix | `ValueError` |
| `kick="spin"` with `gauge="velocity"` | `ValueError` |
| An unknown `kick`, `gauge`, `propagator` or `direction`; a zero `direction` or `strength`; a `duration` that rounds to zero steps | `ValueError` |
| `dt W` above the bound of the propagator | `ValueError` naming `time_step` |
| An ETRS corrector that misses its tolerance, a norm error above `1e-8`, an energy drift above `max_energy_drift` | `RuntimeError` naming the option and the step |
| `window` other than `"exponential"` or `"gaussian"` in a spectrum method | `ValueError` |

## The run log

Each call appends one `[REAL TIME]` block to the log (`txt=` or `trace=True`): the request,
the propagator's record and the checks. It does not repeat the ground-state energy or the
orbital energies, which `[SCF SUMMARY]` and `[SCF SETUP]` own, and the damping and window of
a spectrum belong to the view and are not logged ([the run log](#run-log-real-time)).

## Scientific references

The bibliography registers these records (`mandacaru.utils.bibliography`); each is marked
unverified there, so check one against the journal before relying on it in a manuscript.

- Yabana and Bertsch, *Physical Review B* **54**, 4484 (1996),
  [doi:10.1103/PhysRevB.54.4484](https://doi.org/10.1103/PhysRevB.54.4484)
  (`YabanaBertsch1996`): real-time TD-DFT after a delta kick and the dipole strength
  function.
- Castro, Marques and Rubio, *Journal of Chemical Physics* **121**, 3425 (2004),
  [doi:10.1063/1.1774980](https://doi.org/10.1063/1.1774980)
  (`CastroMarquesRubio2004`): the enforced time-reversal symmetry propagator and its
  approximation.
