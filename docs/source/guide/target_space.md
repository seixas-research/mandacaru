# Choosing the target space

Wannier functions and the downfolded problems built from them
(see the [DFT guide](dft.md), "Wannier functions") need a target
space: the Bloch states the functions are made of. Energy windows say
*where* those states lie, and you set them by hand. This page covers two
ways to choose them automatically: by **what the states are** (their
atomic character) and by **how correlated they are** (RPA natural orbitals).

## By character: projectability

Name the target atomic orbitals as for `guess` and ask for their
projectability, each Bloch state's weight in the span of those orbitals:

```python
p = si.calc.projectabilities("sp3")            # or {"Cu": "d"}, [("Cu", ["s", "p", "d"])]
print(p.summary())                             # per band: energy range, p min / mean / max
p.values                                       # (k-points, bands), between 0 and 1
w = si.calc.wannier(guess="sp3", windows="auto")
```

$$p_{n\mathbf k} = \sum_j |\langle\tilde g_{j\mathbf k}|\psi_{n\mathbf k}\rangle|^2,$$

where the $\tilde g_j$ are the target orbitals, Löwdin-orthogonalized among
themselves.

- **The targets.** They are the atoms' own first-zeta basis orbitals,
  whatever radial functions the trial orbitals of the starting gauge use.
  The overlaps are therefore exact, PAW augmentation included. Summed over
  every band of the basis, $p$ equals the number of targets at each
  k-point (to 1e-14 for silicon's eight sp³ hybrids). Only the span of the
  targets matters: four sp³ hybrids on an atom give the same $p$ as its s
  and p orbitals. Targets must sit on atoms; `"bonds"` and plain centers
  are refused.
- **`windows="auto"`.** The number of functions is the number of targets.
  The frozen window is the widest energy window in which every state has
  $p \ge 0.95$. States with $p < 0.02$ are left out, and the subspace of
  the rest is disentangled as for energy windows.
  `windows={"projectability": (outer, frozen)}` sets the two thresholds.
- **Why a window.** The frozen states form one energy window, not every
  state above the threshold. Freezing every state with $p \ge 0.9$ also
  froze silicon's antibonding states up to 10 eV above the gap, and the
  s–d states of copper up to 8 eV above the Fermi level. Each case raised
  $\Omega_I$ and spoiled the interpolation of the bands near the Fermi
  level, by up to several eV for copper.

Silicon sp³ (LDA, PAW-LCAO SZP), with the hand-set windows of the DFT
guide ($E_F + 1$ eV frozen) as the reference. The interpolation error is
the largest and root-mean-square distance (meV) from each true band below
$E_F + 1$ eV to the nearest interpolated one, at 60 random k-points:

| mesh | frozen window top | $\Omega_I$ auto / hand (Å²) | error auto | error hand |
|---|---|---|---|---|
| 3³ | $E_F$ + 1.98 eV | 7.67 / 7.73 | 462 / 114 | 332 / 98 |
| 4³ | $E_F$ + 1.43 eV | 8.21 / 8.66 | 129 / 31 | 128 / 36 |
| 6³ | $E_F$ + 1.11 eV | 8.85 / 9.35 | 15.6 / 3.9 | 14.6 / 3.4 |
| 8³ | $E_F$ + 1.43 eV | 9.48 / 9.68 | 6.0 / 1.3 | 2.9 / 0.8 |

On 6³, freezing from $p \ge 0.9$ ends the window at $E_F$ + 3.3 eV and
doubles the error (30 meV). A lower threshold of 0.01 or 0.05 instead of
0.02 gives $\Omega_I$ 8.41 or 10.51 Å² and errors of 25 or 13 meV.

Copper (LDA, PAW-LCAO DZP, 6³ SCF), targets s, p and d (nine functions).
The nine lowest bands all have $p \ge 0.97$, and band 9 starts 20 eV
above the Fermi level. The frozen window spans them, from 9.9 eV below
$E_F$ to 11.7 eV above it, so it holds the d bands (4.6 eV below to
1.2 eV above $E_F$). Inside it, the d-like states are interpolated within
36 / 3.9 meV on 8³ and 9.5 / 1.7 meV on 10³; the states below $E_F$
within 20 / 3.4 meV on 10³. The d shell alone, or s and d, is not a closed
target space in copper: the lowest d band carries up to 40% p
character, so no energy window around the d bands has every state above
0.95.

### Without windows: selected columns of the density matrix

`windows="scdm"` uses the erfc-weighted projections of the
selected-columns scheme (Damle, Lin and Ying). Each state's projection onto
the targets is weighted by $f(\varepsilon) = \tfrac12
\operatorname{erfc}((\varepsilon - \mu)/\sigma)$ and Löwdin-orthonormalized,
which gives the subspace and the starting gauge in one step, with no
iteration. By default $\mu$ and $\sigma$ come from an erfc fit of the
projectability against energy, $\mu = \mu_\text{fit} - 3\sigma_\text{fit}$
(Vitale et al.); `windows={"scdm": (mu, sigma)}` (eV) sets them. This
suits a target manifold that starts at the bottom of the spectrum. On
silicon sp³ it gives a larger $\Omega_I$ than the disentanglement (9.07
against 7.73 Å² on 3³, 11.38 against 8.66 Å² on 4³), and an error below
$E_F + 1$ eV of 384 / 129 meV on 3³ and 236 / 71 meV on 4³. It is a start
that needs no windows, not a replacement for them.

## By correlation: RPA natural orbitals

`natural_orbitals(method="rpa")` builds the crystal's one-particle density
matrix in the direct random-phase approximation and diagonalizes it at
each k-point. It is the crystal analogue of `active_space={"method":
"mp2"}`, and it stays finite in a metal, where MP2 diverges:

```python
nos = si.calc.natural_orbitals(method="rpa")    # on the SCF mesh
print(nos.summary())                            # occupations, ranked
nos.occupations, nos.vectors                    # per k-point, in the band basis
w = si.calc.wannier(8, guess="sp3", windows=nos)  # or windows="rpa"
```

- **The amplitudes.** Direct RPA is ring coupled-cluster doubles with the
  Coulomb rings only (Scuseria, Henderson and Sorensen). Its amplitudes
  solve a Riccati equation, here iterated with DIIS through the
  $\mathbf q + \mathbf G$ vectors (the eigenproblem of the RPA matrix is
  the fallback). The pairs are those of the Born-von Karman supercell of
  the mesh. Momentum conservation splits them into one problem per
  $\mathbf q$, coupled to $-\mathbf q$: this is the Γ-point supercell
  problem, block-diagonalized. The Coulomb integrals come from the
  transition densities on $|\mathbf q + \mathbf G|$ below 3 Bohr⁻¹,
  compensation charges included, the same densities the cRPA screening
  uses. A metal's smeared occupations enter as the pair weights
  $F_n - F_m$.
- **The density.** The amplitudes are contracted like the unrelaxed MP2
  density, with direct terms only. The density stays block-diagonal in k,
  its trace is the electron count exactly, and its eigenvectors are Bloch
  natural orbitals.
- **The selection.** The orbitals are ranked by how far their occupation
  lies from 2 or 0. `threshold` (default 0.02) sets the deviation an orbital
  must reach; `nos.n_functions` is the mean count per k-point. Wannierizing
  with `windows=nos` takes each band's projectability onto the selected
  natural orbitals at its k-point and uses the same windows as `"auto"`.
  `n_functions` overrides the count.

The checks:

- **Molecules.** For H₂O and N₂ in cc-pVDZ (dRPA on PBE, density-fitted
  integrals from PySCF), the correlation energy from the amplitudes equals
  PySCF's imaginary-frequency dRPA to 5e-12 Ha. The density matrix equals
  an independent spin-orbital ring-CCD density to 3e-15. The traces are 10
  and 14, and the occupations lie between 2e-4 and 1.9999.
- **Crystals.** On silicon, three formulas give the same correlation
  energy to 2e-11 eV: the amplitudes, the excitation energies and the
  polarizability on the imaginary axis. At zero frequency, the blocks'
  polarizability equals that of the cRPA screening to 7e-16.
- **The k-mesh.** A hydrogen chain on a (3, 1, 1) mesh gives the
  correlation energy of its three-cell supercell at Γ to 1e-7 eV, and
  every occupation to 4e-9.

Silicon (LDA, SZP, 3³; 18 bands): $E_c = -6.70$ eV per cell. The occupied
natural orbitals lie between 1.88 and 1.96. The empty ones, ranked at each
k-point, are 0.073–0.087, 0.042–0.080, 0.040–0.080 and 0.034–0.068, then
0.024–0.033 and 0.022–0.032, then below 0.013. The eight most correlated
orbitals, bonding and antibonding, have sp³ projectability 0.89–1.00. The
next two are d-like, and the rest have at most 0.12 sp³ character. The
default threshold keeps those ten at every k-point.

Aluminum (SZP, 4³, Fermi-Dirac 0.05 eV) is a metal, and the occupations
stay between 9e-4 and 1.96. At each k-point there is one strongly occupied
orbital, and the most occupied empty ones reach 0.14 and 0.11.

**Cost.** A block holds $N_k n_o n_v$ pairs: 1512 for silicon on 3³. The
eigenproblem costs $O(N_k^4(n_on_v)^3)$ and took 31 minutes for silicon on
3³. The DIIS iteration converges in 11–15 steps of $O(N_k^4(n_on_v)^2
n_G)$: 7 minutes for silicon on 3³ and 5 minutes for aluminum on 4³. All
times are on a heavily loaded machine.

## What remains open

- The RPA density is unrelaxed and spin-restricted. Its cost grows as
  $N_k^4 (n_o n_v)^2 n_G$ per DIIS sweep (the eigenproblem as $N_k^4(n_o
  n_v)^3$), which keeps it to coarse meshes.
- The $\mathbf q \to 0$ head of the Coulomb interaction is left out. It is
  a finite-size error that falls with the mesh.
- SCDM is offered with the targets as its columns. The QR-pivoted grid
  columns of the original scheme, which need no guess at all, are not
  implemented.
