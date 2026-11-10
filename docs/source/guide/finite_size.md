(finite-size)=
# Finite-size corrections for periodic supercells

In periodic electronic-structure calculations, a finite simulation cell (the supercell)
is periodically replicated throughout space. While this eliminates artificial surface effects,
it introduces spurious finite-size errors arising from the periodic boundary conditions.
Mandacaru provides three complementary, physics-based finite-size correction schemes
implemented entirely in its native Rust core (`mandacaru-core`):

1. **Model Periodic Coulomb (MPC)**: A real-space and reciprocal-space modification of
   the electron–electron Coulomb interaction replacing the periodic Ewald kernel with the
   truncated Wigner–Seitz kernel for exchange and correlation.
2. **Kwee–Zhang–Krakauer (KZK)**: A size-dependent local density approximation (LDA) functional
   parameterized from quantum Monte Carlo simulations of the finite uniform electron gas.
3. **Chiesa–Ceperley–Martin–Holzmann (CCMH)**: An analytic reciprocal-space extrapolation of the
   static structure factor $S(\mathbf{k})$ that accounts for long-wavelength plasmon excitations
   missing at $\mathbf{k} = \mathbf{0}$.

---

## Physical origins of finite-size errors

Finite-size errors in periodic supercells belong to two distinct physical classes:

### One-body (kinetic) errors
One-body finite-size errors arise from the discrete sampling of the Brillouin zone.
For a supercell of linear dimension $L$, momentum space is sampled on a grid with spacing
$\Delta k \sim 2\pi/L$. In metallic systems with a sharp Fermi surface, this leads to an
oscillatory error decaying as $\mathcal{O}(L^{-1})$ or $\mathcal{O}(L^{-2})$. In insulating
systems with a band gap, the density matrix decays exponentially, leading to an exponential
convergence with supercell size.

### Two-body (Coulomb and correlation) errors
Two-body finite-size errors stem from the periodic Ewald summation used to evaluate Coulomb
interactions:

$$
v_E(\mathbf{r}) = \sum_{\mathbf{R}} \frac{1}{|\mathbf{r} - \mathbf{R}|}
$$

Because of periodic boundary conditions:
- **Exchange–correlation hole self-interaction**: The localized exchange–correlation hole
  $\rho_{\text{xc}}(\mathbf{r}_1, \mathbf{r}_2)$ surrounding an electron interacts not only with
  that electron but also with periodic image holes in neighboring replica cells. This creates an
  artificial attractive self-interaction that decays slowly as $\mathcal{O}(L^{-1})$.
- **Missing long-wavelength excitations**: In a finite supercell, collective charge fluctuations
  (plasmons) with wavelengths $\lambda > L$ (i.e., $|\mathbf{k}| < 2\pi/L$) cannot fit inside the box.
  The long-range $1/k^2$ Coulomb singularity at $\mathbf{k} = \mathbf{0}$ is conventionally dropped,
  omitting the zero-point energy of long-wavelength plasmons.

---

## Model Periodic Coulomb (MPC)

The Model Periodic Coulomb method replaces the periodic Ewald interaction $v_E(\mathbf{r})$
with the truncated Wigner–Seitz potential $v_{\text{WS}}(\mathbf{r})$ when evaluating the
electron–electron exchange and correlation energy.

Inside the Wigner–Seitz primitive cell $\Omega_{\text{WS}}$, $v_{\text{WS}}(\mathbf{r}) = 1/|\mathbf{r}|$,
and outside it conforms to the minimum-image convention:

$$
v_{\text{WS}}(\mathbf{r}) = \frac{1}{|\mathbf{r} - \mathbf{R}_{\text{min}}(\mathbf{r})|}
$$

### The exchange–correlation hole correction
The hole energy difference between the periodic Ewald kernel and the Wigner–Seitz kernel is:

$$
\Delta E_{\text{hole}} = \frac{1}{2} \int_\Omega d\mathbf{r}_1 \int_\Omega d\mathbf{r}_2\,
\left[ v_E(\mathbf{r}_1 - \mathbf{r}_2) - v_{\text{WS}}(\mathbf{r}_1 - \mathbf{r}_2) \right]
\rho_{\text{xc}}(\mathbf{r}_1, \mathbf{r}_2)
$$

In a correlated wavefunction or variational state, the exchange–correlation hole is computed
from the one-body reduced density matrix (1-RDM) $D_{pr}$ and two-body reduced density matrix
(2-RDM) $\Gamma_{pqrs}$:

$$
\rho_{\text{xc}}(\mathbf{r}_1, \mathbf{r}_2) = \rho_2(\mathbf{r}_1, \mathbf{r}_2) - \rho(\mathbf{r}_1)\rho(\mathbf{r}_2)
$$

A fundamental exact trace identity guarantees electron conservation in the hole:

$$
\sum_q \left( \Gamma_{pqrq} - D_{pr} D_{qq} \right) = -D_{pr}
$$

Mandacaru verifies this trace identity to machine precision before contracting tensors.

### Madelung double-counting resolution
A standard periodic Hamiltonian already includes single-particle Madelung self-interaction
corrections $v_M/2$ per electron in the total energy. Evaluating $\Delta E_{\text{hole}}$ without
accounting for the single-particle self-energy double-counts the Madelung contribution.
Mandacaru applies the exact corrected relation:

$$
\Delta E_{\text{MPC}} = \Delta E_{\text{hole}} - \frac{1}{2} N_e v_M
$$

where $N_e$ is the number of electrons in the supercell and $v_M$ is the lattice Madelung constant:

$$
v_M = \lim_{\mathbf{r} \to \mathbf{0}} \left[ v_E(\mathbf{r}) - \frac{1}{|\mathbf{r}|} \right]
$$

### Numerical implementation in Rust
- **Voxel self-potential**: The self-potential of an orthogonal grid voxel is integrated analytically
  using the divergence theorem over the six polyhedral faces (12 planar triangles):
  $$
  \Phi_{\text{voxel}} = \frac{1}{V_{\text{box}}} \int_{\text{box}} \frac{d\mathbf{r}}{|\mathbf{r}|}
  $$
- **FFT circular convolution**: The truncated Wigner–Seitz convolution is evaluated in $\mathcal{O}(N \log N)$
  time using 3D fast Fourier transforms.
- **Richardson extrapolation**: On discretized grids, the contact potential exhibits an $\mathcal{O}(h^2)$
  quadrature discretization error. Mandacaru employs Richardson extrapolation across grid densities
  to satisfy analytic integration gates within 1%.

---

## Kwee–Zhang–Krakauer (KZK) functional

The KZK method replaces the conventional uniform electron gas (LDA) exchange and correlation
functional with a size-dependent functional $e_{\text{KZK}}(r_s, L)$ parameterized from
quantum Monte Carlo calculations in finite periodic cubic boxes:

$$
\Delta E_{\text{KZK}} = \int_\Omega \rho(\mathbf{r}) \left[ e_{\text{KZK}}(\rho(\mathbf{r}), L) - e_{\text{LDA}}(\rho(\mathbf{r})) \right] d\mathbf{r}
$$

where $r_s(\mathbf{r}) = \left(\frac{3}{4\pi \rho(\mathbf{r})}\right)^{1/3}$ is the local Seitz radius
and $L = \Omega^{1/3}$ is the effective cubic box length.

### Functional formulation
KZK partitions the Seitz radius into three distinct regimes:
1. **Low $r_s$ (high density)**:
   $$
   e_x(r_s, L) = -\frac{A_0}{r_s} + \frac{A_1}{L} + \frac{A_2 r_s}{L^2} + \frac{A_3 r_s^2}{L^3}
   $$
   where $A_0 = \frac{3}{4}\left(\frac{3}{2\pi}\right)^{2/3} \approx 0.45816529$ is the Slater exchange constant.
2. **High $r_s$ (low density)**:
   $$
   e_{\text{KZK}}(r_s, L) = e_{\text{LDA}}(r_s) + \frac{G_1}{L} + \frac{G_2}{L^2 r_s^{1/2}} + \frac{G_3}{L^3 r_s} + \frac{G_4}{L^4 r_s^{3/2}}
   $$
3. **Intermediate $r_s$**:
   A continuous $C^1$ cubic polynomial matches values and first derivatives continuously across the
   boundaries $\gamma_l$, $\gamma$, and $\gamma_h$.

Mandacaru uses the exact derivative from the Perdew–Zunger correlation relation:

$$
v_c(r_s) = e_c(r_s) - \frac{r_s}{3} \frac{de_c(r_s)}{dr_s} \implies \frac{de_c}{dr_s} = \frac{3}{r_s} \left( e_c - v_c \right)
$$

Compensated summation guarantees that in the thermodynamic limit $L \to \infty$, the difference
$e_{\text{KZK}} - e_{\text{LDA}}$ cancels identically to machine precision ($0.0$).

---

## Chiesa–Ceperley–Martin–Holzmann (CCMH)

The CCMH scheme corrects the Coulomb potential energy by extrapolating the static structure factor
$S(\mathbf{k})$ to the long-wavelength limit $\mathbf{k} \to \mathbf{0}$.

The structure factor measures electron–electron pair correlations in reciprocal space:

$$
S(\mathbf{k}) = \frac{1}{N_e} \langle \rho_{\mathbf{k}} \rho_{-\mathbf{k}} \rangle
$$

### Asymptotic behavior and plasmon model
In the uniform electron gas and isotropic solids, long-range Coulomb interactions cause $S(k)$
to vanish quadratically as $k \to 0$:

$$
\lim_{k \to 0} \frac{S(k)}{k^2} = \frac{\hbar \omega_p}{8\pi \rho_0}
$$

where $\omega_p = \sqrt{4\pi \rho_0 / m}$ is the plasma frequency and $\rho_0 = N_e / \Omega$ is the
average electron density.

The missing potential energy per electron due to the omission of the $\mathbf{k} = \mathbf{0}$
component in the periodic supercell is:

$$
\Delta V = \frac{2\pi}{\Omega} \lim_{k \to 0} \frac{S(k)}{k^2}
$$

For the entire supercell containing $N_e$ electrons, the total supercell correction is:

$$
\Delta E_{\text{CCMH}} = N_e \Delta V
$$

### Structure factor from density matrices
Mandacaru computes the connected structure factor $S_c(\mathbf{k})$ directly from the native 1-RDM and 2-RDM:

$$
S_c(\mathbf{k}) = \frac{1}{N_e} \sum_{pqrs} M_{pr}(\mathbf{k}) M_{qs}(-\mathbf{k}) \left( \Gamma_{prqs} - D_{pr} D_{qs} \right)
$$

where $M_{pr}(\mathbf{k}) = \langle \phi_p | e^{i\mathbf{k}\cdot\mathbf{r}} | \phi_r \rangle$ are the plane-wave
matrix elements. The vectors $\mathbf{k}$ are grouped into symmetry shells $|\mathbf{k}|$, and a polynomial fit
$S_c(k) = a k^2 + b k^4$ reliably extracts the curvature $\lim_{k \to 0} S_c(k)/k^2 = a$.

---

## Usage in Mandacaru

### Python interface

Finite-size corrections are accessible via the `finite_size_correction` method on the calculator:

```python
from ase import Atoms
from mandacaru import Mandacaru

# The primitive cell of a hydrogen chain; the mesh is solved as its 2-cell supercell
# (see the crystal guide, bloch.md)
atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2.0, 10.0, 10.0], pbc=[True, False, False])

calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                 grid_spacing=0.35,
                 kpts={"size": (2, 1, 1), "gamma": True})
atoms.calc = calc
energy = atoms.get_potential_energy()

# Evaluate Model Periodic Coulomb correction
mpc = calc.finite_size_correction(scheme="mpc")
print(f"Supercell MPC correction: {mpc.correction:.6f} Ha")
print(f"Per-cell MPC correction:   {mpc.correction_per_cell:.6f} Ha")
print(mpc.summary())

# Convert to electronvolts
mpc_ev = mpc.in_units("eV")
print(f"Corrected energy: {mpc_ev.mpc_energy_per_cell:.4f} eV per cell")
```

Low-level functions are also available in `mandacaru.core`:

```python
from mandacaru.core.kzk import kzk_correction
from mandacaru.core.mpc import wigner_seitz_kernel, voxel_self_potential
from mandacaru.core.ccmh import ccmh_extrapolate

# Direct KZK evaluation from a real-space density grid
delta_e = kzk_correction(density_array, cell_matrix)

# Compute Wigner-Seitz kernel via Rust
kernel = wigner_seitz_kernel(grid_shape=(32, 32, 32), step=step_matrix)
```

### Native Rust API

In native Rust code, finite-size routines reside under `mandacaru_core::periodic::finite_size`:

```rust
use mandacaru_core::periodic::finite_size::{
    kzk::{kzk, kzk_energy_correction},
    mpc::{Mpc, mpc_from_hole_energies},
    ccmh::{Ccmh, ccmh_extrapolate},
};
use mandacaru_core::math::wigner_seitz::RealKernel;

// Evaluate KZK exchange-correlation difference
let delta_kzk = kzk(rs, box_length);

// Compute MPC from exchange-correlation hole energies
let mpc_result = mpc_from_hole_energies(
    e_periodic_hole,
    e_truncated_hole,
    n_electrons,
    madelung_constant,
);

// CCMH quadratic extrapolation from reciprocal shells
let ccmh_result = ccmh_extrapolate(&k_magnitudes, &s_values, cell_volume, n_electrons)?;
```

---

## Conventions and units

- **Internal units**: All native Rust routines operate in atomic units (distances in Bohr, energies in Hartree).
- **Python reporting**: The Python interface reports energies in Hartree by default and converts to electronvolts
  via `correction.in_units("eV")`.
- **Per-cell vs. per-supercell**: A supercell with $N_{\text{cells}}$ primitive units has:
  $$
  E_{\text{per-cell}} = \frac{E_{\text{supercell}}}{N_{\text{cells}}}
  $$
  `FiniteSizeCorrection` explicitly tracks `n_cells` to prevent mixing per-cell and per-supercell quantities.
