(excited-states)=
# Excited states of molecules: linear-response TD-DFT

After a converged `method="dft"` run on a molecule, `calc.get_excitations(...)`
returns its lowest electronic excitations from adiabatic linear-response
time-dependent density functional theory (TD-DFT). The same call on a
`method="rhf"` run gives CIS and time-dependent Hartree-Fock. The numbers are
computed in Rust (`molecular::excitations`); the returned
`mandacaru.analysis.excitations.Excitations` view converts units and holds no
arithmetic of its own. Closed-shell and open-shell (spin-unrestricted) molecules are
supported, with the `lda`, `pbe`, `r2scan` and `hse06` functionals and Hartree-Fock,
and with the on-site DFT+U correction (`hubbard_u=`, Dudarev and Liechtenstein forms);
[What raises](#excited-states-refusals) lists every request that is refused, and
[the not-yet-available table](#not-yet-available) what is planned.

The time-domain route, a delta kick followed by real-time propagation of the orbitals, is
[its own guide](real_time.md); its weak-kick spectrum peaks at the roots of
`approximation="casida"` below.

This is not the route to the excited states of a quantum run. `energy_levels()` and
the `subspace-*` methods give the many-body levels of a qubit Hamiltonian
([the methods table](#methods-table)); the levels here are
single-excitation responses of a mean-field state, with different fields
(oscillator strengths, amplitudes, transition dipoles). The two views never convert
into each other, and `get_excitations` on a calculator with a quantum method raises
`ValueError`.

## What is computed

The ground state is the converged Kohn-Sham (or Hartree-Fock) determinant, restricted
for a closed shell and spin-unrestricted for an open one, with `o` occupied and `v`
empty orbitals. Each excitation is a linear combination of
occupied-to-empty transitions `i -> a`. With the adiabatic kernel (frequency
independent, evaluated at the ground-state density) the response equations for a
closed shell, separately for singlets and triplets, are

$$
\begin{pmatrix} A & B \\ B & A \end{pmatrix}
\begin{pmatrix} X \\ Y \end{pmatrix}
= \omega \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\begin{pmatrix} X \\ Y \end{pmatrix},
\qquad
A_{ia,jb} = (\epsilon_a - \epsilon_i)\,\delta_{ij}\delta_{ab} + K_{ia,jb},
$$

where `K` is the response kernel: the Hartree kernel and the second derivative of the
exchange-correlation energy for Kohn-Sham, Coulomb and exact exchange for Hartree-Fock.
For `hse06` the kernel also carries the short-range exact exchange, in both `A` and `B`,
and with `hubbard_u` it carries the +U kernel ([r2SCAN and DFT+U](#excited-states-meta-u)).
Beyond the orbital-energy differences, the exchange part and the +U kernel are the
pieces that act in `A - B`.

- `approximation="tda"` (the default) is the Tamm-Dancoff approximation, `B = 0`: the
  Hermitian problem `A X = omega X`. It is robust, and on an `rhf` state it is CIS. If a
  root of `A` is not positive, the reference is unstable in that symmetry: the call raises
  `ValueError` naming the spin and advising a spin-unrestricted reference (a restricted
  state) or another converged state (an unrestricted one), and never
  returns a negative or zero excitation energy.
- `approximation="casida"` solves the full equations, `(A - B)(A + B)(X + Y) =
  omega^2 (X + Y)` (Casida's equations; on an `rhf` or `uhf` state, TDHF). If a root has
  `omega^2 < 0` the reference is unstable in that symmetry: the call raises
  `ValueError` naming the spin and the advice above, and never
  returns an imaginary root or falls back to the other approximation. For a restricted
  state, the advice has a route: run the molecule unrestricted ([open shells](#excited-states-open-shell)).
- `spin=None` (the default) is a singlet on a closed shell. `spin="singlet"` or
  `"triplet"` selects one of the two spin-adapted solves of a closed shell; an open
  shell has no spin adaptation and refuses either name
  ([open shells](#excited-states-open-shell)). A triplet has no transition dipole
  without spin-orbit coupling, so its oscillator strengths are exactly zero.
- `kernel="adiabatic"` (the default) is the reference's whole kernel. `"hartree"`
  keeps the Coulomb kernel only (the random-phase approximation); `"independent"` has
  no kernel, so the roots are the bare orbital-energy differences. The last two are
  reference limits, not alternative physics.

The kernel is the exact second derivative of the energy that the SCF minimizes. The
test suite checks it against finite differences of the Fock matrix, in the
symmetric and the exchange channel, for every piece (Poisson Hartree, PAW compensation
pairs, the exchange-correlation kernel of `lda`, `pbe` and `hse06`, Hartree-Fock exchange, the
short-range exchange of `hse06` and, for its triplets, the spin polarization of the PAW
spheres' short-range semilocal exchange, which a singlet does not see; the unrestricted
kernel is checked the same way, both spin blocks, on NH2 and on a polarized PAW reference).
The `r2scan` kernel is checked against differences of the Fock matrix too (singlets and
triplets of a closed shell, and the unrestricted kernel on NH2), and the +U kernel in the
symmetric channel the same way, for both forms, restricted and unrestricted. The +U
`A - B` channel is checked against the closed form of the Dudarev correction and, for the
Liechtenstein form, against the second difference of an independently built real-harmonic
energy; singlet and triplet `A - B` agree exactly, as they must, because an imaginary change
of the occupations carries no density. The lowest roots
come from a matrix-free block Davidson solver: the Hermitian one for TDA, the paired
(product-form) one for Casida. No dense matrix is built in production. Every
reduction runs in a fixed order, so energies and strengths are bit-identical at
every thread count.

## A first calculation

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=4.0)
atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25, parallelization={"threads": 2})

singlets = atoms.calc.get_excitations(5)                       # TDA, singlets
triplets = atoms.calc.get_excitations(5, spin="triplet")
casida = atoms.calc.get_excitations(5, approximation="casida")

print(singlets.energies)               # eV, ascending
print(singlets.oscillator_strengths)   # dimensionless, length gauge
print(singlets.velocity_oscillator_strengths)   # dimensionless, velocity gauge
atoms.calc.close()
```

If no calculation has run yet, `get_excitations` runs it first. The PAW-LCAO basis needs
the dataset library ([PAW-LCAO bases](pseudopotentials.md)), and a calculation reads
the set of its functional: `pbe-sr` for `xc="pbe"`, `xc="hse06"` and `xc="r2scan"`, `lda-sr` for
`xc="lda"`. Each call solves one spin symmetry (closed shell) and one approximation; ask
again for the other. The
solves reuse the converged state, so a semilocal request costs a fraction of the SCF: the
eight `lda` and `pbe` water requests below took 1.2 to 2.2 s each on two threads, SCF
included. An `hse06` request is not cheap: each one builds the screened exchange once (and
shares it between the Fock matrix and the kernel), and a water request takes about 8 s
after an SCF of about 8.6 s, on two threads.

The lowest roots of this water molecule (PAW-LCAO DZP, grid spacing 0.25 Angstrom, 4
Angstrom of vacuum, residual tolerance `1e-8`) are frozen in
`test/migration/reference/excitations.json`. Energies in eV; the length-gauge oscillator
strength of the first singlet root is in parentheses:

| | singlet, TDA | singlet, Casida | triplet, TDA | triplet, Casida |
| --- | --- | --- | --- | --- |
| `lda` | 7.894 (0.0324) | 7.872 (0.0319) | 7.345 | 7.335 |
| `pbe` | 7.876 (0.0330) | 7.857 (0.0326) | 7.221 | 7.199 |
| `hse06` | 8.469 (0.0343) | 8.449 (0.0343) | 7.767 | 7.734 |

To six decimals the `hse06` energies are 8.469147 (singlet, TDA), 8.448506 (singlet,
Casida), 7.767175 (triplet, TDA) and 7.733697 eV (triplet, Casida). The five lowest `lda` TDA
singlets are 7.894, 9.878, 10.016, 12.145 and 14.149 eV, with
oscillator strengths 0.0324, 0 (a dark root, to 1e-24), 0.0836, 0.0462 and 0.2751. Read these numbers as
what this basis, grid and box give, not as benchmarks: see [the limits](#excited-states-limits).

## The `Excitations` view

`get_excitations(num_states=5, *, approximation="tda", spin=None,
kernel="adiabatic", tolerance=1e-5, max_iterations=200)` returns the view. `tolerance`
bounds the residual norm of each root, in atomic units; a solve that does not reach it
in `max_iterations` raises `RuntimeError` instead of returning unconverged roots, so the view has no
`converged` flag. The fields:

| Field | Meaning |
| --- | --- |
| `energies` | Excitation energies, ascending, in eV (Ha with `atomic_units=True`) |
| `wavelengths` | The vacuum wavelengths of the same photons, nm |
| `oscillator_strengths` | Dimensionless, `2/3 omega abs(mu)^2` (length gauge); zero for triplets |
| `velocity_oscillator_strengths` | Dimensionless, `2/3 abs(p)^2 / omega` with `p = <0 abs(grad) abs n>` (velocity gauge, from `X - Y`, with the PAW one-center part of the gradient); zero for triplets. Not the same number as the length form in a finite basis: see [the limits](#excited-states-limits) |
| `transition_dipoles` | `(n, 3)` transition moments `<0 abs(r) abs n>` in e Angstrom (e Bohr with `atomic_units=True`); the sign of each root is a convention |
| `one_center_hartree` | PAW-LCAO Kohn-Sham only, else `None`: the first-order shift of each root from the one-center Hartree kernel, in the energy unit (see below) |
| `spin_squared`, `reference_spin_squared` | Open shell only, else `None`: `<S^2>` of each root and of the reference determinant (see [open shells](#excited-states-open-shell)) |
| `approximation`, `spin`, `kernel`, `functional` | What was solved; `spin` reads `"singlet"` or `"triplet"` on a closed shell and `"unrestricted"` on an open one; `functional` is `None` on a Hartree-Fock state |
| `orbital_energies`, `occupied`, `virtual` | Tuples with one entry per spin channel (one for a closed shell; alpha, then beta, for an open one): the reference's orbital energies (ascending) and the numbers of occupied and empty orbitals of that channel |
| `requested`, `added`, `degenerate_tolerance` | The roots asked for, how many the multiplet rule added, and the tolerance (1e-4 Ha) below which roots count as one multiplet |
| `residuals`, `iterations`, `applications` | The solver's final residual per root, its iterations and its operator applications |

`len(view)` and `view.num_states` are the number of roots returned. A request that
would cut a degenerate multiplet (roots closer than `degenerate_tolerance`) is
extended to the whole multiplet, so `len(view)` can exceed `num_states`; `added` says by
how much. Within a degenerate multiplet only gauge-invariant quantities are meaningful:
the energies and the strengths summed over the multiplet, not the strength of one member.

The methods take a **0-based** root index:

- `transitions(root)` lists the dominant orbital transitions as `Transition(spin,
  occupied, empty, weight)` records. `spin` is `None` on a closed shell and `"alpha"` or
  `"beta"` on an open one; the orbital indices are the reference's in that spin (0-based,
  ascending energy, so the highest occupied orbital of channel `c` is
  `view.occupied[c] - 1`); the weights are `X^2` (TDA) or
  `X^2 - Y^2` (Casida). Weights below 0.01 are left out, at most ten are listed, and the
  largest is always kept.
- `amplitudes(root, channel=0)` returns `(X + Y, X - Y)` of that spin channel (0 on a
  closed shell; 0 alpha and 1 beta on an open one), each `(occupied, virtual)` of the
  channel. Summed over the channels, `(X + Y) . (X - Y) = 1`. In TDA both are `X`.
- `natural_transition_orbitals(root, channel=0)` returns `(weights, occupied, empty)`: the
  weights of the natural transition orbital pairs of that channel in descending order
  (they sum to 1 within the channel), and the pairs as columns over the atomic basis.
  A weight near 1 for the first pair means the excitation is one orbital to one orbital.
  `nto_weights(root)` gives the weights of the whole root in descending order: the same
  numbers on a closed shell, and on an open shell those of both spin blocks together,
  summing to 1 over the two.
- `transition_density_matrix(root, channel=0)` is the symmetrized transition density
  matrix in the atomic basis, `C_o (X + Y) C_v^T`, of that channel: on an open shell, the
  spatial factor of that spin's first-order density change.
- `spectrum(energies, width=0.1, shape="gaussian")` broadens the roots into
  `sum_n f_n g(E - E_n)`, with the length-gauge `oscillator_strengths` as `f_n`, on the
  `energies` you give (in the view's energy unit), with
  unit-area `"gaussian"` (standard deviation `width`) or `"lorentzian"` (half width
  `width`) lines, computed in Rust. Any other `shape` raises `ValueError`.

The natural transition orbitals and the transition density matrix are one-particle
quantities of the response, not a wavefunction. They are returned as arrays over the
atomic basis; no `write_cube` or `Viewer3D` route takes them yet
([not yet available](#not-yet-available)).

(excited-states-open-shell)=
## Open shells: unrestricted references

A radical or any state with unequal alpha and beta populations is solved on its
spin-unrestricted reference: `method="uhf"` (UHF, CIS or TDHF) or a `method="dft"` state
with `xc="lda"`, `"pbe"`, `"r2scan"` or `"hse06"` and an open shell (unequal populations, or
`spin=True` or `unrestricted=True`). `get_excitations` follows the reference it finds, not
a keyword.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("NH2")                      # a doublet
atoms.center(vacuum=4.0)
atoms.calc = Mandacaru(method="dft", xc="pbe", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25, parallelization={"threads": 2})

view = atoms.calc.get_excitations(5)         # spin is left unset
print(view.energies)                         # eV
print(view.spin_squared)                     # <S^2> of each root
print(view.reference_spin_squared)           # <S^2> of the reference determinant
for t in view.transitions(0):                # Transition(spin, occupied, empty, weight)
    print(t.spin, t.occupied, t.empty, t.weight)
atoms.calc.close()
```

What is solved:

- **Spin-conserving excitations of both spins.** The alpha and the beta transitions sit
  in one vector (the alpha block, then the beta block), coupled through the Hartree
  kernel of the total transition density, the spin-resolved exchange-correlation kernel
  at the polarized ground-state density (and, for `hse06`, each spin's short-range exact
  exchange and the kernel of the spheres' spin polarization at the polarized
  reference), or, for Hartree-Fock, each spin's exchange. With `r2scan` the spin-resolved
  meta-GGA kernel acts the same way, and with `hubbard_u` each spin channel carries its
  +U kernel ([r2SCAN and DFT+U](#excited-states-meta-u)). Both the Tamm-Dancoff and the
  Casida forms exist. Spin-flip excitations are not computed.
- **`spin=None` is the default.** A closed shell takes it as a singlet; an open shell has
  no spin adaptation, and `spin="singlet"` or `"triplet"` raises `ValueError` naming that.
  The transition moments are summed over both spins in spin orbitals, so the strengths
  carry no spin-adaptation factor.
- **A closed shell run unrestricted** (`unrestricted=True`) returns the union of its
  singlets and its triplets: the roots match the spin-adapted solves, with `<S^2>` equal to
  0 and 2 exactly. The test suite uses this identity as a gate for LDA, PBE and
  Hartree-Fock and, on an all-electron basis, HSE06.
- **Each spin channel needs a gap** between its highest occupied and lowest empty orbital,
  and an unrestricted reference that is not the lowest of its symmetry raises
  `ValueError` (the message suggests converging another state, for example from other
  initial magnetic moments).

`<S^2>` of each root, `view.spin_squared`, is evaluated from the singles combination of the
reference's determinants with the root's `X` amplitudes. It is an approximation of the
interacting state's value, as the reference determinant is, and it is a **contamination
diagnostic**, not a quantum number. A clean doublet has 0.75. The first roots of NH2
below sit at 0.752 to 0.754, but states built from excitations into singly occupied
orbitals can be strongly contaminated: root 3 of NH2 (counting from 1) has `<S^2>` of 2.59
to 2.70 across the functionals, a state with a large quartet component, which is the known
weakness of unrestricted adiabatic TD-DFT. `view.reference_spin_squared` is the same
number for the reference (0.752 to 0.754 for these runs). Both are `None` on a closed
shell, whose roots are pure singlets or triplets.

The channel-resolved views take `channel=0` (alpha) or `channel=1` (beta): `amplitudes(root,
channel)`, `natural_transition_orbitals(root, channel)` and `transition_density_matrix(root,
channel)`; `transitions(root)` carries the spin in each record. The first root of NH2
(PAW-LCAO DZP, grid spacing 0.25 Angstrom, 4 Angstrom of vacuum, residual tolerance `1e-8`,
frozen in `test/migration/reference/excitations.json`) is, in eV, with its length-gauge
oscillator strength and `<S^2>` in parentheses:

| | TDA | Casida |
| --- | --- | --- |
| `lda` | 1.952860 (0.00224, 0.7521) | 1.885818 (0.00151, 0.7521) |
| `pbe` | 2.281644 (0.00255, 0.7535) | 2.194107 (0.00209, 0.7535) |
| `hse06` | 2.271213 (0.00266, 0.7543) | 2.186831 (0.00226, 0.7543) |

A request takes 2.3 to 3.6 s for `lda` and `pbe` and about 25 s for `hse06`, the SCF
included, on two threads. As for water, read the numbers as what this basis, grid and box
give, not as benchmarks.

**An HSE06 caveat for finite-difference checks.** The exchange hole model of `hse06` has a
kink in its dependence on the reduced density gradient (a slope discontinuity at a reduced
gradient of 1, of about 5e-4), so a finite-difference check of an `hse06` response quantity
(the kernel against a difference of Fock matrices, or a response against a finite-field
energy) needs steps of at most 1e-4: a larger step can cross the kink at some grid points
and give erratic differences (the unrestricted `hse06` check in the test suite uses steps of
1e-5 and 2e-5), and such a check keeps a tolerance of about 1e-4, where the `lda`, `pbe` and
Hartree-Fock checks hold to about 1e-6. The kink is kept as the functional is implemented, so
no `hse06` number depends on it. The analytic response is
the exact derivative of the implemented functional, evaluated one-sidedly there, and
this does not change it; `pbe` shows nothing like it.

(excited-states-meta-u)=
## r2SCAN and DFT+U

**`xc="r2scan"`.** The meta-GGA kernel is the second derivative of the r2SCAN energy the
SCF minimizes, with respect to the density matrix. A trial transition density changes three
grid fields: the density, its gradient (the product-rule gradient of the orbitals, as the
ground state builds it) and the kinetic energy density, which the response takes from the
orbitals' gradients exactly as the ground state does. On a PAW-LCAO basis the smooth core
density and its kinetic density are included, as in the ground state. The kernel acts in
`A` and `B` of a closed shell (singlets and triplets) and, per spin, of an unrestricted
reference. `kernel="hartree"` and `kernel="independent"` drop it, like every
exchange-correlation term. The kernel is tested on molecules without nodes of the
magnetization; r2SCAN is sharp near such nodes, and a response there is not validated.

**`hubbard_u=`.** The +U energy (Dudarev or Liechtenstein form) is quadratic in the shell
occupations `n = Q^T D Q`, so its kernel is exact and constant: a linear map from the change
of the occupations to the change of the +U potential, carried to the occupied-empty blocks
by the projections of the canonical orbitals on the shell. Its structure is
exchange-type, not a function of the density of each transition: between the transitions
`i -> a` and `j -> b` of one spin, its `A` element is a product of the shell overlaps of the
occupied pair `(i, j)` and of the empty pair `(a, b)`, and its `B` element one of `(i, b)`
and `(a, j)` (Dudarev form, up to the factor `U_eff`). So it enters `A - B` as well as
`A + B`, and a state with `hubbard_u` has no density-only kernel. The request follows the limits of the ground state's `hubbard_u`: `method="dft"` on a
PAW-LCAO basis, no screened hybrid, no spin-orbit dataset, no `float32`
([DFT+U on molecules](#dft-u-molecules)). The test suite exercises the response with
`xc="lda"` on copper hydride and its cation.

The CuH roots frozen in `test/migration/reference/excitations.json` come from `xc="lda"`
with the Dudarev correction on the copper d shell, U = 4 eV, a 1.46 Angstrom bond in a 9
Angstrom cell, PAW-LCAO DZP and grid spacing 0.25 Angstrom:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[9.0] * 3)
atoms.center()
atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25, hubbard_u={"Cu": {"l": "d", "U": 4.0}},
                       parallelization={"threads": 2})
view = atoms.calc.get_excitations(5, tolerance=1e-8)
atoms.calc.close()
```

The first root, in eV, with the length-gauge oscillator strength of the singlets in
parentheses (the lowest five roots are completed to whole multiplets; a request takes 3.0 to
5.5 s, the SCF included, on two threads):

| | singlet | triplet |
| --- | --- | --- |
| TDA | 3.728 (0.0851) | 2.636 |
| Casida | 3.610 (0.0699) | 2.583 |

As for water, read these as what this basis, grid and box give, not as benchmarks. No
calculation without +U is frozen for this molecule, so the table says nothing about how
large the +U contribution is.

## PAW-LCAO: the consistent kernel and what it omits

On a PAW-LCAO basis the kernel is the exact second derivative of the energy the
calculation implements, and that energy is "frozen": the one-center Hartree and
exchange-correlation terms of each atom are linearized at the reference atom. A
linear functional of the density matrix has no second derivative, so the kernel has no
one-center term. The ground state omits the second-order one-center term too (the
audit that measured it is in the maintainers' plan files), and the response omits the same physics.
The effect is largest for excitations localized on one atom, such as lone-pair and
semicore transitions. (For `hse06` the exact-exchange part of the kernel does carry the PAW
compensation and one-center pieces of the screened factors, and a triplet's kernel carries
the linearized spin polarization of the spheres' short-range semilocal exchange; what is
omitted is the one-center Hartree and exchange-correlation response described here.)

The view reports the Hartree part of what is missing as `one_center_hartree`: the
first-order shift each root would receive from the one-center Hartree kernel. It is
not added to `energies`; adding it would make the response differ from the second
derivative of the ground-state energy. For singlets it is a small shift whose sign is
not fixed, because the one-center factors are signed: for the five lowest `lda` TDA
singlets of the water above it is positive, 0.04 to 0.10 eV, while some of the frozen
CuH roots ([r2SCAN and DFT+U](#excited-states-meta-u)) carry a negative one. For
triplets it is exactly zero, because the Coulomb-type kernel cancels between the spins. **The
one-center exchange-correlation part is not computed yet**, so the diagnostic is a
partial estimate: the run log and the view docstring say so. On an all-electron
basis (`HAO`) and on a Hartree-Fock state, the kernel has no PAW pieces and
`one_center_hartree` is `None`.

A semilocal response works for every element the ground state supports. Hartree-Fock
(CIS, TDHF) needs the factored Coulomb integrals, and so it has the same limit as the
Hartree-Fock ground state on a PAW-LCAO basis: a core beyond one closed 1s shell is
refused ([not yet available](#not-yet-available)).

(excited-states-limits)=
## Limits of the method and of the basis

- **Adiabatic semilocal kernels** (`lda`, `pbe`, `r2scan`) place charge-transfer excitations too
  low, and place Rydberg states poorly. The screened hybrid `hse06` does not repair this
  either: its exact exchange is short-range. Double
  excitations are absent from any adiabatic linear response.
- **Spin contamination on open shells.** An unrestricted reference is not a spin
  eigenstate, and the roots built on it inherit that: check `spin_squared` against 0.75
  (a doublet) before reading a root, and distrust the ones far above it
  ([open shells](#excited-states-open-shell)). No adiabatic response here repairs it.
- **A confined basis without diffuse functions** pushes diffuse states up. In plain DZP the
  water LUMO is unbound (+1.03 eV) and the first singlet above (7.87 eV, f = 0.032) lies above
  the ionization threshold (minus the highest occupied orbital energy, 6.57 eV): a basis
  artifact in the Kohn-Sham continuum, whose closeness to the experimental first band near
  7.4 eV is partly an accident. With `basis={"name": "PAW-LCAO", "size": "DZP", "diffuse": True}`
  ([diffuse functions](#paw-lcao-diffuse)) and 8 Å of vacuum, the LUMO is bound (-0.91 eV) and
  the first singlet drops to 6.60 eV with LDA (6.44 eV with PBE), f = 0.047, still the HOMO to
  LUMO transition; a larger box (11 Å) and a finer grid (0.20 Å) move it by under 10 meV. That
  is about 0.8 eV below experiment, the known underbinding of Rydberg-like states by adiabatic
  LDA and PBE. Converge the basis (diffuse shells) and the box before reading Rydberg or
  near-threshold states.
- **The one-center response is omitted** (above); only its Hartree part is estimated.
- **Gauge.** The view carries two oscillator strengths: `oscillator_strengths` in the
  length gauge and `velocity_oscillator_strengths` in the velocity gauge. They agree
  only in a complete basis (and, for the velocity form, only for a local Hamiltonian:
  with exact exchange the nonlocal potential makes them differ too). In a finite basis
  neither is gauge-invariant, and the length form satisfies the sum rule only
  approximately. In plain DZP the two differ a lot for the first water root: the
  `lda` Casida singlet has 0.0319 in the length gauge and 0.1505 in the velocity gauge.
  That root lies in the Kohn-Sham continuum, an artifact of the basis (see above). With
  diffuse shells (8 Å vacuum) the Casida velocity strengths come within 10 to 13 % in
  amplitude, about 25 % in f, of the length ones for the first water roots, with LDA and
  HSE06 alike; a finer grid, more vacuum or a TZP basis do not close the rest, which is
  not yet attributed (the PAW one-center scheme or the angular completeness of the LCAO
  basis). Read the velocity form as good to about 25 % in f here, and judge gauge
  agreement on Casida roots: TDA velocity strengths are further off. With a density-only
  kernel the full Casida length-gauge strengths sum exactly to the bare Kohn-Sham ones
  within the basis, and the TDA strengths do not. A +U kernel is not density-only, so that
  identity does not hold with `hubbard_u`.
- **Crystals** are not supported. A semilocal adiabatic kernel gives no bound excitons
  in an insulator anyway: the planned route is the Bethe-Salpeter equation on the
  screened interaction the constrained-RPA code already builds.

(excited-states-refusals)=
## What raises

`get_excitations` refuses what it cannot compute and never substitutes another
calculation. Unknown option values raise `ValueError` and name the accepted ones.

| Request | What happens |
| --- | --- |
| Any quantum method ([the methods table](#methods-table)) | `ValueError`: the response needs a mean-field state |
| `method="ghf"` | `ValueError`: a two-component state is not a restricted or unrestricted reference |
| A spin-orbit (fully relativistic) dataset, whose Kohn-Sham state is a spinor | `NotImplementedError` naming spinor Kohn-Sham; a scalar-relativistic dataset runs collinear Kohn-Sham, restricted or unrestricted |
| `spin="singlet"` or `"triplet"` on an unrestricted reference (an open shell, `method="uhf"`, `spin=True` or `unrestricted=True`) | `ValueError`: an unrestricted reference has spin-conserving excitations of both spins, and the spin adaptation is that of a closed shell; leave `spin` unset |
| An unrestricted reference with no gap between the highest occupied and lowest empty orbital of one spin | `ValueError` naming the spin channel |
| `xc="pbe+rvv10"` or `"r2scan+rvv10"` | `NotImplementedError`: the nonlocal (rVV10) kernel is not implemented, and neglecting it would run a different calculation |
| `hubbard_u=` where the ground state refuses it (a screened hybrid, a spin-orbit dataset, `precision="float32"`; [DFT+U on molecules](#dft-u-molecules)) | The ground state's own error, raised when the calculation starts: no response is attempted |
| A fractional-occupation state (a degenerate or nearly degenerate frontier shell, [Occupations](#dft-degenerate-shell)), or a state that is not aufbau | `NotImplementedError` naming the fractional occupations; give the spin state explicitly instead |
| `precision="float32"` | `ValueError`: the response runs in `float64` only |
| A crystal (`pbc`, or `load_crystal=`) | `NotImplementedError` |
| An unconverged SCF, or no gap between the highest occupied and lowest empty orbital | `ValueError` |
| `num_states` below 1 or above the number of occupied-to-empty transitions | `ValueError` |
| An unstable restricted reference: a Tamm-Dancoff root that is not positive | `ValueError` naming the spin and a spin-unrestricted reference (a stretched H2 triplet is an example) |
| An unstable restricted reference: a Casida root with `omega^2 < 0` | `ValueError` naming the spin and a spin-unrestricted reference; the call never returns an imaginary root or switches to `approximation="tda"` |
| An unrestricted reference that is not the lowest of its symmetry (a non-positive Tamm-Dancoff root or an imaginary Casida root) | `ValueError` advising to converge another state, for example from other initial magnetic moments |
| A solve that does not reach `tolerance` in `max_iterations` | `RuntimeError` |

## The run log

Each call appends one `[EXCITED STATES]` block to the log (`txt=` or `trace=True`): the
request, the solver's record and one row per root; on an open shell it adds the
reference's `<S^2>` and a `spin_squared` column, and each transition carries its spin,
as in `3a->5a` (alpha) or `2b->4b` (beta). It does not repeat the ground-state
energy or the orbital energies, which `[SCF SUMMARY]` and `[SCF SETUP]` own
([the run log](#run-log-excited-states)).

## Scientific references

The bibliography registers these records (`mandacaru.utils.bibliography`); each is marked
unverified there, so check one against the journal before relying on it in a manuscript.

- Runge and Gross, *Physical Review Letters* **52**, 997 (1984),
  [doi:10.1103/PhysRevLett.52.997](https://doi.org/10.1103/PhysRevLett.52.997)
  (`RungeGross1984`): the foundation of time-dependent DFT.
- Casida, in *Recent Advances in Density Functional Methods, Part I*, World Scientific
  (1995), pp. 155-192 (`Casida1995`): the response equations of `approximation="casida"`.
- Hirata and Head-Gordon, *Chemical Physics Letters* **314**, 291 (1999),
  [doi:10.1016/S0009-2614(99)01149-5](https://doi.org/10.1016/S0009-2614(99)01149-5)
  (`HirataHeadGordon1999`): the Tamm-Dancoff approximation.
- Stratmann, Scuseria and Frisch, *Journal of Chemical Physics* **109**, 8218 (1998),
  [doi:10.1063/1.477483](https://doi.org/10.1063/1.477483)
  (`StratmannScuseriaFrisch1998`): the paired Davidson solver of the Casida equations.
- Martin, *Journal of Chemical Physics* **118**, 4775 (2003),
  [doi:10.1063/1.1558471](https://doi.org/10.1063/1.1558471) (`Martin2003`): natural
  transition orbitals.
