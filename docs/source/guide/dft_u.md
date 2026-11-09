(dft-u)=
# DFT+U: the on-site Hubbard correction

Standard semilocal exchange–correlation functionals (LDA, PBE) spread the electrons of
partly filled, localized $d$ and $f$ shells over too many orbitals due to spurious
self-interaction. In transition-metal oxides, rare-earth compounds, and magnetic
insulators, this over-delocalization severely underestimates band gaps (for instance,
yielding only 0.6 eV in Rocksalt NiO compared to the experimental 4.3 eV), places localized
multiplet levels at unphysical energies, and underestimates local magnetic moments.

DFT+U remedies this by adding an on-site Coulomb penalty that discourages fractional
occupations within the correlated shell. In Mandacaru, DFT+U is implemented directly on the
native Rust engine (`mandacaru-core`), where projector projections, occupation matrices,
energy corrections, and effective Hamiltonian potential operators are evaluated
self-consistently during the SCF cycle. Python provides the user API, ASE integration, and
snapshot presentation.

The option is `hubbard_u` in `Mandacaru(method="dft", ...)`; an example is in
`examples/dft/08_dft_nio_afm_hubbard_u.py`.

## The `hubbard_u` option

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms.calc = Mandacaru(
    method="dft",
    xc="lda",
    basis={"name": "PAW-LCAO", "size": "SZ"},
    kpts={"size": (2, 2, 2), "gamma": True},
    spin=True,
    hubbard_u={"Ni": {"l": "d", "U": 6.0, "J": 0.0, "form": "dudarev-fll"}},
)
```

`hubbard_u` maps an element symbol to a dictionary configuring its correlated shell:
* `"l"`: the angular momentum shell, `"d"` ($l = 2$) or `"f"` ($l = 3$).
* `"U"`: on-site Coulomb interaction parameter in eV (finite, $\ge 0$).
* `"J"`: on-site Hund's exchange coupling parameter in eV (finite, $\ge 0$; defaults to 0.0).
* `"form"`: functional formulation:
  * `"dudarev-fll"`: simplified rotationally invariant Dudarev functional with fully localized limit (FLL) double counting (default).
  * `"liechtenstein-fll"`: rotationally invariant Slater integral formulation with separate $U$ and $J$.
* Optional Slater ratio parameters (Liechtenstein form): `"f4_f2"` ($F^4/F^2$) and `"f6_f2"` ($F^6/F^2$).

Each atom of a listed element receives the correction on that shell; an element may carry
one shell. Specifying no `hubbard_u` runs plain DFT, and passing $U = 0$ reproduces plain DFT
to the precision of the SCF (`result.hubbard_energy` is then 0.0 eV).

`hubbard_u` requires `method="dft"`: any other method raises `ValueError`
([the methods reference](#methods-table)).

## Subspace projection: what `n` is

All DFT+U variants map the Kohn–Sham single-particle density onto an on-site shell
occupation matrix $n^\sigma$. In Mandacaru, this subspace projection uses the atom's PAW
augmentation partial waves:

$$
n^\sigma_{mm'} = \sum_{ij} W_i R^\sigma_{(m i), (m' j)} W_j = \left( W^T R^\sigma W \right)_{mm'},
$$

where $R^\sigma$ is the density matrix projected onto the projectors of the shell summed over
the Brillouin zone:

$$
R^\sigma_{(m i), (m' j)} = \sum_{\mathbf{k}} w_{\mathbf{k}} \langle \tilde{p}_{mi} | \psi_{\mathbf{k}} \rangle \langle \psi_{\mathbf{k}} | \tilde{p}_{m'j} \rangle = \sum_{\mathbf{k}} w_{\mathbf{k}} \left( C_{\mathbf{k}}^\dagger D^\sigma_{\mathbf{k}} C_{\mathbf{k}} \right)_{(mi), (m'j)}.
$$

Here, $m, m' \in [-l, l]$ are orbital indices, $i, j$ index the radial projector channels of
the shell, $w_{\mathbf{k}}$ are $k$-point integration weights, and $W_i = \langle \chi | \phi_i \rangle$
are the radial overlap weights between the first bound all-electron partial wave $\chi$ and
the all-electron basis waves $\phi_i$, evaluated inside the augmentation sphere radius $r_c$.
The weights $W$ are computed once per element and projector basis in Rust (`paw/shell.rs`).

For a restricted calculation (`spin=False`, `unrestricted=False`), the density matrix $D$
represents the spin-summed density, so each spin channel is assigned half:
$n^\sigma = \frac{1}{2} W^T R W$, and $E_U$ counts both spin channels.

### Truncated partial-wave normalization: a filled shell has $n < 1$

The projector basis is non-orthonormal and strictly sphere-truncated inside $r_c$. Because a
bound all-electron wave has a small tail extending outside $r_c$, the projected occupation of
a completely filled shell evaluates to less than 1 inside the sphere:
* Ni 3d: $\approx 0.951$ (scalar-relativistic `lda-sr`, $r_c = 2.2$ Bohr)
* Fe 3d: $\approx 0.925$
* Cu 3d: $\approx 0.960$
* Mn 3d: $\approx 0.905$
* Gd 4f: $\approx 0.458$ ($r_c = 0.744$ Bohr)
* O 2p: $\approx 0.680$
* H 1s: $\approx 0.281$

In fully relativistic datasets (`lda-fr`, `pbe-fr`), the sphere cutoffs differ from the
scalar-relativistic sets (for instance, Ni 3d $r_c = 1.060$ Bohr, yielding a filled-shell
fraction of $\approx 0.638$).

Because $n < 1$ for a filled shell, $E_U$ is slightly non-zero even for an isolated atom with
a closed shell. The $U$ value that reproduces an experimental gap is calibrated to this
specific projection and is not directly transferable to codes that orthonormalize projector
spheres.

## Theoretical formulations

Mandacaru implements multiple DFT+U formalisms on the native engine:

### 1. Dudarev simplified rotationally invariant functional (FLL)

The Dudarev formulation (`form="dudarev-fll"`) is parameterized by a single effective
interaction:

$$
U_\mathrm{eff} = U - J.
$$

The Hubbard energy and effective potential operator are:

$$
E_U = \frac{U_\mathrm{eff}}{2} \sum_\sigma \mathrm{Tr}\left[ n^\sigma \left( \mathbf{I} - n^\sigma \right) \right] = \frac{U_\mathrm{eff}}{2} \sum_\sigma \sum_{m} \left( n^\sigma_{mm} - \sum_{m'} n^\sigma_{mm'} n^\sigma_{m'm} \right),
$$

$$
V^\sigma_{mm'} = \frac{\partial E_U}{\partial n^\sigma_{m'm}} = U_\mathrm{eff} \left( \frac{1}{2} \delta_{mm'} - n^\sigma_{mm'} \right).
$$

$E_U$ splits into an on-site Coulomb term and the fully localized limit (FLL) double-counting
subtraction:

$$
E_\mathrm{dc} = \frac{U}{2} N(N - 1) - \frac{J}{2} \sum_\sigma N_\sigma(N_\sigma - 1).
$$

The operator $V^\sigma$ acts on the Kohn–Sham Hamiltonian through the shell projectors:

$$
H^\sigma \leftarrow H^\sigma + C \left( W V^\sigma W^T \right) C^\dagger.
$$

Nearly filled orbitals ($n \approx 1$) shift downward in energy by approximately $-U_\mathrm{eff}/2$,
while nearly empty orbitals ($n \approx 0$) shift upward by $+U_\mathrm{eff}/2$, opening the
band gap. The Hellmann–Feynman derivative with respect to $U_\mathrm{eff}$ is:

$$
\frac{dE_U}{dU_\mathrm{eff}} = \frac{1}{2} \sum_\sigma \mathrm{Tr}\left[ n^\sigma \left( \mathbf{I} - n^\sigma \right) \right].
$$

### 2. Liechtenstein form: rotationally invariant with Slater integrals

The Liechtenstein formulation (`form="liechtenstein-fll"`) accounts for the full angular
structure of the screened Coulomb interaction via Slater integrals $F^k$, distinguishing the
direct Coulomb repulsion $U = F^0$ from the exchange parameter $J$:

$$
v_{abcd} = \left\langle ab \left| \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \right| cd \right\rangle = \sum_{k=0,2,\dots,2l} F^k a_k(a,b,c,d),
$$

$$
a_k(a,b,c,d) = \frac{4\pi}{2k+1} \sum_{q=-k}^k G(k, q; l, m_a; l, m_c) \, G(k, q; l, m_d; l, m_b),
$$

where $G$ are real/complex Gaunt coefficients computed by `math::angular::gaunt`.

The on-site interaction energy and effective potential are:

$$
E_\mathrm{Hub} = \frac{1}{2} \sum_{abcd} v_{abcd} \left( n_{ca} n_{db} - \sum_\sigma n_{da}^\sigma n_{cb}^\sigma \right), \qquad E_U = E_\mathrm{Hub} - E_\mathrm{dc},
$$

$$
V^\sigma_{ab} = \sum_{cd} \left( v_{acbd} n_{dc} - v_{acdb} n^\sigma_{dc} \right) - \delta_{ab} \left[ U\left(N - \tfrac12\right) - J\left(N_\sigma - \tfrac12\right) \right],
$$

with $N = \sum_\sigma N_\sigma$ and $N_\sigma = \mathrm{Tr}[n^\sigma]$.

The exchange coupling $J$ is consistent with the FLL double counting:

$$
J = \frac{1}{p(p - 1)} \sum_{k \ge 2} F^k S_k, \qquad S_k = \sum_{ab} a_k(a,b,b,a).
$$

For $d$ shells ($p = 5$), $J = \frac{F^2 + F^4}{14}$. The default Slater integral ratio is
$F^4/F^2 = 5/8 \approx 0.625$. For $f$ shells ($p = 7$), $J = \frac{2}{45}F^2 + \frac{1}{33}F^4 + \frac{50}{1287}F^6$,
with hydrogenic defaults $F^4/F^2 = 451/675$ and $F^6/F^2 = 1001/2025$. These ratios can be
overridden via the optional keys `"f4_f2"` and `"f6_f2"`.

### 3. Noncollinear spinor Dudarev functional

For two-component spinor calculations with spin-orbit coupling, the occupation matrix is a
full $2p \times 2p$ complex Hermitian matrix over spin and orbital indices:

$$
n_{(a s), (b s')} = \sum_{\mathbf{k}} f_{\mathbf{k}} \, c_{a s}^{\mathbf{k}} \left(c_{b s'}^{\mathbf{k}}\right)^*, \qquad s, s' \in \{\uparrow, \downarrow\}, \quad a, b \in [-l, l].
$$

The rotationally invariant functional evaluates the trace over the combined spin-orbital space:

$$
E_U = \frac{U_\mathrm{eff}}{2} \mathrm{Tr}_{2p}\left[ n - n^2 \right], \qquad V = U_\mathrm{eff} \left( \frac{1}{2}\mathbf{I}_{2p} - n \right).
$$

On time-reversal-invariant meshes, the time-reversal completion:

$$
\Theta |m \uparrow\rangle = (-1)^m |-m \downarrow\rangle, \qquad \Theta |m \downarrow\rangle = -(-1)^m |-m \uparrow\rangle,
$$

enforces Kramers pairing and preserves exact orbital ordering.

### 4. First-principles linear response: Cococcioni–de Gironcoli

Rather than empirical tuning, the Hubbard $U$ parameter can be determined from first principles
via constrained density-functional perturbation theory. A localized potential shift $\alpha_J$
perturbs shell $J$:

$$
H \leftarrow H + \alpha_J P_J, \qquad P_J = \sum_{m} |p_{Jm}\rangle \langle p_{Jm}|.
$$

The non-interacting (bare) susceptibility $\chi^0$ and interacting susceptibility $\chi$ measure
the response of the shell occupations $N_I = \mathrm{Tr}[n_I]$:

$$
\chi^0_{IJ} = \left. \frac{dN_I}{d\alpha_J} \right|_\mathrm{bare}, \qquad \chi_{IJ} = \left. \frac{dN_I}{d\alpha_J} \right|_\mathrm{scf}.
$$

The bare susceptibility $\chi^0$ is computed directly as a sum over ground-state Kohn–Sham
eigenpairs with a chemical potential shift term for metals:

$$
\chi^0_{IJ} = \sum_{\mathbf{k}} w_{\mathbf{k}} g \sum_{nm} \frac{f_n - f_m}{\varepsilon_n - \varepsilon_m} M_I(n, m) M_J(m, n) - \frac{\left(\sum f' M_I\right)\left(\sum f' M_J\right)}{\sum f'}.
$$

Inverting the full multi-site susceptibility matrices yields the screened on-site interaction:

$$
\mathbf{U} = \left(\boldsymbol{\chi}^0\right)^{-1} - \boldsymbol{\chi}^{-1}.
$$

In Mandacaru, this response can be invoked on a converged calculator via:

```python
resp = calc.hubbard_response(alphas=(0.05, 0.1))
hubbard_u_dict = resp.hubbard_u(form="dudarev-fll")
```

The resulting parameters can then be supplied directly into subsequent SCF runs.

## Exposed properties

On the calculator result (`calc.result`):

| Property | Type | Description |
| --- | --- | --- |
| `hubbard_energy` | `float` or `None` | Total $E_U$ in eV (already included inside `get_potential_energy()`); `None` without `hubbard_u` |
| `hubbard` | `list[dict]` or `None` | List of per-shell occupation dictionaries, or `None` |

Each per-shell dictionary in `result.hubbard` contains:
* `"atom"`: integer atom index in the geometry.
* `"symbol"`: element chemical symbol.
* `"l"`: shell angular momentum (`"d"` or `"f"`).
* `"U"`: applied on-site $U$ in eV.
* `"J"`: applied on-site $J$ in eV.
* `"energy"`: shell contribution to $E_U$ in eV.
* `"trace"`: occupation count $\mathrm{Tr}[n^\sigma]$ for each spin channel.
* `"n"`: occupation matrix array of shape `(spins, 2l+1, 2l+1)`, complex Hermitian, ordered $m = -l, \dots, +l$.

```python
result = atoms.calc.result
print(f"E_U = {result.hubbard_energy:.6f} eV")

for shell in result.hubbard:
    atom_idx = shell["atom"]
    sym = shell["symbol"]
    trace = shell["trace"]
    print(f"Atom {atom_idx} ({sym}): Tr(n_up) = {trace[0]:.4f}, Tr(n_dn) = {trace[1]:.4f}")
    
    # Extract eigenvalues of the occupation matrix:
    import numpy as np
    eigenvalues_up = np.linalg.eigvalsh(shell["n"][0])
    print(f"  Majority d eigenvalues: {np.round(eigenvalues_up, 4)}")
```

The local magnetic moment of the correlated shell follows directly from the occupation trace:
$m_d = \mathrm{Tr}[n^\uparrow] - \mathrm{Tr}[n^\downarrow]$.

## A saved crystal owns its correction

`save_crystal="path.json"` saves the complete DFT+U specification (form, shells, projector indices,
weights, and seed moments) inside the snapshot. When restarting with `load_crystal="path.json"`:
* Omit `hubbard_u`: the file's stored correction applies automatically.
* If passing `hubbard_u`: it must agree exactly with the file's settings; any disagreement
  raises `ValueError("disagrees with the crystal file")`.

## The run log

Calculations run with `txt=` record DFT+U details in the `[SCF SUMMARY]` block:

```text
    hubbard_energy_eV: 4.2560147802
    hubbard_Ni_d: U 6.000000 eV, J 0.000000 eV, U_eff 6.000000 eV
```

When linear response is run, response matrices and condition diagnostics are reported in
`[HUBBARD RESPONSE]`.

## Status of forces and stresses

Plain-DFT crystal forces and stresses for LDA, PBE, and HSE06 are analytic in native Rust
([Forces](forces_relaxation.md)).

For a DFT+U crystal calculation (`hubbard_u`), requesting crystal forces or stress currently
raises:

```text
NotImplementedError: crystal forces and stress are not available for a DFT+U crystal (hubbard_u):
the +U terms of the analytic force and stress are not implemented; the DFT+U energy is available (with symmetry=False)
```

This refusal is checked before any SCF cycle begins. Mandacaru deliberately refuses numerical
finite-difference fallbacks through ASE, ensuring that all forces and stresses remain exact,
analytic, and computationally bounded.

The analytic force contribution from DFT+U requires differentiating the basis-projector
overlap matrices $C(X) = \langle \chi_\mu | p_p \rangle$:

$$
F_X^{(U)} = -\sum_{s, \mathbf{k}} w_{\mathbf{k}} 2 \mathrm{Re}\mathrm{Tr}\left[ D_{s\mathbf{k}} \, dC^X \, O_s \, C_{\mathbf{k}}^\dagger \right],
$$

where $O_s = W V_s W^T$. The implementation of these analytic $+U$ gradient terms is scheduled
in the development plan (`dev_plan/DFTU_PLAN.md`).

## Supported features and refusal matrix

Every unsupported combination raises an explicit error; none silently runs a different calculation.

| Configuration / Request | Status / Behavior |
| --- | --- |
| Periodic crystal, collinear spin, semilocal functional (`lda`, `pbe`), float64 | **Supported** (native Rust SCF, energies, band structures, DOS/PDOS, Wannier) |
| Periodic crystal with `symmetry=True` on a reduced $k$-mesh | Raises `ValueError` ("DFT+U needs the shell occupations symmetrized with the density... use symmetry=False"). Nonzero magnetic moments preserve the full mesh automatically. |
| Periodic crystal with `xc="hse06"` (screened hybrid) | Raises `NotImplementedError` ("DFT+U is implemented for semilocal functionals only"). |
| Single precision (`precision="float32"`) | Raises `ValueError` ("precision='float64'"). |
| Periodic crystal on a fully relativistic / spin-orbit dataset (`lda-fr`, `pbe-fr`) | Raises `NotImplementedError` ("scalar-relativistic datasets only"). A molecule on such a dataset runs spinor Kohn–Sham without DFT+U; spinor molecular DFT+U is not implemented. |
| Isolated molecule (`pbc=False`) | Raises `ValueError` ("needs a periodic cell: DFT+U is implemented for crystals only"). |
| Basis without PAW datasets (`HAO`, Gaussian STO-3G) | Raises `ValueError` ("hubbard_u needs a PAW-LCAO or UPAW-LCAO basis"). |
| Element not in system, or channel absent from dataset | Raises `ValueError` naming element and shell. |
| Shell with no occupied reference wave (e.g. Ba 5d, Au 5f in `lda-sr`) | Raises `ValueError` naming the element and channel. |
| Crystal forces or stress with `hubbard_u` | Raises `NotImplementedError` ("crystal forces and stress are not available for a DFT+U crystal"). |

## Example: NiO type-II antiferromagnet

`examples/dft/08_dft_nio_afm_hubbard_u.py` calculates rocksalt NiO (4-atom AFM-II unit cell,
$a = 4.17$ Å, LDA, PAW-LCAO SZ basis, $2 \times 2 \times 2$ $k$-mesh, 0.05 eV Fermi–Dirac smearing,
initial magnetic moments $+2.0$ and $-2.0$ on the Ni atoms):

```python
from ase.build import bulk
from mandacaru import Mandacaru

# Construct rocksalt NiO AFM-II cell
# (2 Ni atoms with opposite magnetic moments, 2 O atoms)
...
atoms.calc = Mandacaru(
    method="dft",
    xc="lda",
    basis={"name": "PAW-LCAO", "size": "SZ"},
    kpts={"size": (2, 2, 2), "gamma": True},
    symmetry=False,
    spin=True,
    hubbard_u={"Ni": {"l": "d", "U": 6.0, "J": 0.0}},
)
energy = atoms.get_potential_energy()
```

Comparison between plain DFT ($U = 0$) and DFT+U ($U = 6$ eV):

| Observable | Plain LDA ($U = 0$) | LDA+U ($U = 6$ eV) |
| --- | --- | --- |
| Total energy | -10112.3922 eV | -10107.4683 eV |
| Kohn–Sham band gap | 0.597 eV | 2.229 eV |
| Hubbard energy ($E_U$) | 0.0000 eV | 4.2560 eV (2.1280 eV per Ni) |
| Ni 3d occupation trace ($\uparrow / \downarrow$) | 4.35 / 3.82 | 4.789 / 3.382 |
| Ni local $d$ magnetic moment | $0.53 \, \mu_B$ | $1.41 \, \mu_B$ |
| $n$ eigenvalues | 0.412 to 0.910 | 0.256 to 0.959 |

The on-site potential penalty pushes the occupied majority $e_g$ states down and empty minority
states up, opening the band gap by 1.63 eV and strongly localizing the magnetic moment on the
nickel sites.

## Scientific references

- Dudarev, S. L., Botton, G. A., Savrasov, S. Y., Humphreys, C. J., and Sutton, A. P.,
  *Electron-energy-loss spectra and the structural stability of nickel oxide: an LSDA+U study*,
  *Phys. Rev. B* **57**, 1505–1509 (1998),
  [doi:10.1103/PhysRevB.57.1505](https://doi.org/10.1103/PhysRevB.57.1505).
- Liechtenstein, A. I., Anisimov, V. I., and Zaanen, J.,
  *Density-functional theory and strong interactions: Orbital ordering in Mott-Hubbard insulators*,
  *Phys. Rev. B* **52**, R5467–R5470 (1995),
  [doi:10.1103/PhysRevB.52.R5467](https://doi.org/10.1103/PhysRevB.52.R5467).
- Cococcioni, M. and de Gironcoli, S.,
  *Linear response approach to the calculation of the effective interaction parameters in the LDA+U method*,
  *Phys. Rev. B* **71**, 035105 (2005),
  [doi:10.1103/PhysRevB.71.035105](https://doi.org/10.1103/PhysRevB.71.035105).
- Anisimov, V. I., Zaanen, J., and Andersen, O. K.,
  *Band theory and Mott insulators: Hubbard U instead of Stoner I*,
  *Phys. Rev. B* **44**, 943–954 (1991),
  [doi:10.1103/PhysRevB.44.943](https://doi.org/10.1103/PhysRevB.44.943).
- Timrov, I., Marzari, N., and Cococcioni, M.,
  *Hubbard parameters from density-functional perturbation theory*,
  *Phys. Rev. B* **98**, 085127 (2018),
  [doi:10.1103/PhysRevB.98.085127](https://doi.org/10.1103/PhysRevB.98.085127).
