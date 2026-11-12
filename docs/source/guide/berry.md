(berry-phase)=
# Berry phase, curvature, polarization and Hall conductivity

A crystal solved with `method="dft"` has Bloch states at every k-point of the Brillouin
zone, not only on the mesh of its self-consistent run. The calculator uses them to report
the geometry of the occupied bands: the Berry phase of a closed string of k-points, the
Wannier charge centers and Berry connection along it, the Berry curvature and Chern number of a
k plane, the electric polarization of an insulator, and the intrinsic anomalous Hall
conductivity of an insulator or a metal.
Everything is computed in Rust (`periodic::berry`); Python validates the request and
converts units. These are DFT-only: a quantum method on a crystal solves a
Born-von Karman supercell with no Bloch bands and raises `NotImplementedError`
([crystals with the quantum methods](bloch.md)).

| Method | Returns |
| --- | --- |
| `calc.get_berry_phase(direction, kpts=None, bands=None)` | `BerryPhase`: the Berry phase of every string along $\mathbf G_{\text{direction}}$, the Wannier charge centers and the Berry connection |
| `calc.get_berry_curvature(plane, kpts=None, offset=0.0, bands=None)` | `BerryCurvature`: the plaquette flux and curvature of a k plane and its Chern number, both oriented along $\mathbf G_a\times\mathbf G_b$ |
| `calc.get_chern_number(plane, kpts=None, offset=0.0, bands=None)` | the Chern number of that plane, an `int` summed over spin channels, oriented along $\mathbf G_a\times\mathbf G_b$ |
| `calc.get_polarization(kpts=None)` | `Polarization`: the Berry-phase polarization in C/m², its quanta and its raw coordinates |
| `calc.get_anomalous_hall(kpts=None)` | `AnomalousHall`: the Hall conductivity of an insulator or a metal with the plane numbers behind it, the per-layer values and, for a metal, its convergence estimate |
| `calc.get_anomalous_hall_conductivity(kpts=None)` | the `sigma` of that view, a `(3, 3)` array in S/cm |

They are methods returning views, not ASE properties: each takes a mesh, a plane or a band
group, and a polarization means nothing without its quantum. `get_property("polarization")`
keeps raising. The views are the classes of `mandacaru.analysis.berry`; they are not exported
by the package.

## An example

Polar LiH in a skew cell: no inversion, so a nonzero polarization, and time-reversal
symmetric, so no Hall response. The cell is small and the run takes about a minute.

```python
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru

bohr = 0.529177210903
cell = np.array([[7.0, 0.0, 0.0], [0.8, 6.5, 0.0], [0.4, 0.6, 7.5]]) * bohr
lih = Atoms("LiH", scaled_positions=[[0.1, 0.2, 0.15], [0.1 + 3.0 / 7.0, 0.2, 0.15]],
            cell=cell, pbc=True)
lih.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                     grid_spacing=0.4 * bohr, kpts={"size": (2, 1, 1), "gamma": True},
                     smearing={"method": "fermi-dirac", "width": 0.01})

mesh = (4, 2, 2)                                  # strings of 4 points, 2 x 2 of them
p = lih.calc.get_polarization(kpts=mesh)          # runs the SCF first
print(p.vector)                                   # C/m^2, branch nearest zero
print(p.fractional, p.raw)                        # in quanta, folded and unfolded
print(p.quanta[0])                                # the quantum along a_1, C/m^2

phase = lih.calc.get_berry_phase(0, kpts=mesh)    # strings along G_1
print(phase.phase, phase.phase_spread)            # rad

plane = lih.calc.get_berry_curvature(2, kpts=(4, 4, 1))   # the plane normal to G_3
print(plane.chern_number, plane.max_flux)

print(lih.calc.get_anomalous_hall_conductivity(kpts=(3, 3, 3)))   # S/cm
```

`smearing` is narrow on purpose: the polarization, the phases and the curvature refuse a
fractionally occupied state (see [what raises](#berry-refusals)), and the default 0.1 eV of the
crystal SCF leaves a cell with a small gap above the occupation tolerance. The Hall
conductivity would not refuse it but take the [metal's weighted field](#berry-metals), whose
value is close to the quantized one without equaling it. The mesh `(2, 1, 1)` is the SCF's; the
Berry mesh is separate and is passed with `kpts=`.

On this cell the polarization along $\mathbf a_1$ comes out at $-0.50$ C/m² (the other two
components are below $10^{-3}$ C/m²), which is $-0.43$ quanta; its unfolded coordinate is
$1.57$, two quanta higher, and the quantum along $\mathbf a_1$ is $1.17$ C/m². Along a string
the unfolded coordinate converges as $1/N^2$: at $N = 2, 4, 6, 8$ points it is $1.572256$,
$1.569524$, $1.569185$, $1.569096$ (the transverse mesh stays $2\times 2$). The string
phase spread is $0.03$ rad, far below the warning threshold. The Chern number of the plane
is 0, its largest plaquette flux $0.02$ rad, and the Hall conductivity is zero. The grid is
coarse and the basis single-zeta: the numbers show the shape of the output, not a converged
LiH polarization.

## The mesh

`kpts` is the Berry mesh `(n1, n2, n3)`, or `{"size": (n1, n2, n3)}` (only `size` is read; the
mesh is always Gamma-anchored, $\mathbf k = \sum_j (i_j/n_j)\,\mathbf G_j$). It
defaults to the size of the calculation's own `kpts`, which for a small SCF mesh is too
coarse (a string of one point raises), so pass it explicitly. It is independent of the SCF
mesh: the states at every Berry k-point are diagonalized again with the converged
potential, so no k-point has to be on the SCF mesh.

- `get_berry_phase(direction, ...)`: `direction` is the index 0, 1 or 2 of a reciprocal
  lattice vector. The entry of `kpts` along it is the string length (at least 2); the other two
  are the transverse mesh, one string per transverse point.
- `get_berry_curvature(plane, ...)` and `get_chern_number`: `plane` is the index $c$ of the
  reciprocal vector $\mathbf G_c$ normal to the plane, spanned by the other two in cyclic
  order ($c = 2$: $\mathbf G_1, \mathbf G_2$; $c = 0$: $\mathbf G_2, \mathbf G_3$). The two in-plane entries
  of `kpts` (each at least 2) are the plane's mesh; the entry along the normal is ignored.
  The curvature and the Chern number are oriented along $\mathbf G_a\times\mathbf G_b$, which
  points against $\mathbf a_c$ in a left-handed cell: relabeling the cell into the other
  handedness (swapping $\mathbf a_1$ and $\mathbf a_2$, say) flips their sign. The Hall
  conductivity does not flip (see below).
  `offset` is the plane's fractional coordinate along $\mathbf G_c$, in [0, 1).
- `get_polarization`, `get_anomalous_hall` and `get_anomalous_hall_conductivity`: at least 2
  points along every direction. The polarization cuts its strings from the mesh in all three
  directions; the Hall conductivity takes the plane number of every plane of the mesh in all
  three families (the Chern number for an insulator). A metal's Hall conductivity forms its
  convergence estimate only when every entry is even and at least 8 (see [metals](#berry-metals)).
- `bands=None` is the occupied manifold of an insulator, taken per spin channel. A band
  index or a half-open `(lo, hi)` takes that group: with one band, the band-resolved curvature.
  The group must be separated from the bands below and above it by more than $10^{-6}$ Ha at
  every k-point of the mesh, otherwise `ValueError` names the k-point. An isolated group is
  also accepted for a metal.

A bad direction, plane, mesh, offset or band group raises `ValueError`; booleans are
not indices.

## What is computed

The links between neighboring k-points are the overlaps of the periodic parts of the
Bloch states, $M_{mn}(\mathbf k, \mathbf b) = \langle u_{m\mathbf k}|u_{n,\mathbf k+\mathbf b}\rangle$, with
the PAW one-center terms (the same overlaps the Wannier localization uses). A link that
crosses the zone boundary reuses the state of the wrapped point with the true $\mathbf b$, so
every string closes. Each link is made unitary (the polar factor of its singular value
decomposition) before it is used. The one-center term is first order in $\mathbf b$; the
dropped second-order term is Hermitian and enters $\arg\det$ only at third order in
$\mathbf b$, so the Berry phase of a string converges as $1/N^2$ in its length $N$, not
exactly at any mesh.

**Berry phase.** For the bands $B$ along a closed string of $N$ points,

$$
\gamma = -\sum_{s=1}^{N} \arg\det M_B(\mathbf k_s, \mathbf k_{s+1})
 = \oint \mathbf A\cdot d\mathbf k, \qquad \mathbf A = i\langle u|\nabla_{\mathbf k} u\rangle,
$$

reported in $(-\pi, \pi]$ and gauge invariant modulo $2\pi$.

**Wilson loop and Wannier charge centers.** The product $W$ of the unitary links around a
string has unit-modulus eigenvalues $e^{i\theta_n}$. The hybrid Wannier centers are
$x_n = -\theta_n/2\pi$, fractional along $\mathbf a_j$ in [0, 1) (`wannier_charge_centers`);
their sum is $\gamma/2\pi$ modulo 1.

**Berry connection.** The connection is gauge dependent. It is reported only in the
parallel-transport gauge of each string, where it is constant along the string and diagonal:
$A_n = 2\pi x_n/|\mathbf G_j|$ with $x_n$ on $(-1/2, 1/2]$, a length in Å (`connection`): the
projection of the hybrid Wannier center on $\mathbf G_j/|\mathbf G_j|$.

**Berry curvature and Chern number** (Fukui, Hatsugai and Suzuki). On a plane with in-plane
vectors $\mathbf G_a, \mathbf G_b$ (cyclic) each plaquette carries the flux

$$
\phi = -\arg\left[\det U_a(\mathbf k)\,\det U_b(\mathbf k+\mathbf b_a)\,
 \det U_a(\mathbf k+\mathbf b_b)^*\,\det U_b(\mathbf k)^*\right]\in(-\pi,\pi],
$$

with $U$ the unitary links of the band group (non-Abelian through the determinant). The
curvature $\Omega_c = \phi/(|\mathbf G_a\times\mathbf G_b|/n_a n_b)$ is the component along the
plane normal $\mathbf G_a\times\mathbf G_b$ at the plaquette center, and the Chern number is
$C_c = (1/2\pi)\sum\phi$, an integer by construction. Both change sign when the cell is
relabeled into the other handedness, since $\mathbf G_a\times\mathbf G_b$ then points
against $\mathbf a_c$; the Hall conductivity below does not. A plaquette flux above
$3\pi/4$ is refused: near $\pi$ the lattice field aliases and the integer would be silently
wrong. The sign is tied to the phase's: the flux of the strip between two strings along $a$
is the difference of their Berry phases modulo $2\pi$.

**Polarization** (King-Smith and Vanderbilt; Resta). Strings along each $\mathbf G_j$ are cut
over the transverse mesh, their phases are unwrapped onto the branch of their circular mean
and averaged. The electronic fractional coordinate is $p^{\text{el}}_j = -f\gamma_j/2\pi$, with $f$
the electrons per band (below), and the ionic one $p^{\text{ion}}_j = \sum_a Z_a s_{a,j}$, with $Z_a$ the
valence charge of the PAW dataset and $s_a$ the fractional position. The polarization is

$$
\mathbf P = \frac{e}{\Omega}\sum_j p_j\,\mathbf a_j,
$$

and is defined only modulo the quantum $g\,e\,\mathbf a_j/\Omega$ with
$g = \gcd(f, Z_1, Z_2, \dots)$ (just $f$ when a valence charge is not an integer).

**Anomalous Hall conductivity** of an insulator. With the Chern number $C_c$ of the planes
normal to every $\mathbf G_c$ (the same on every layer of the mesh, or `ValueError`), summed
over spin channels and weighted by the electrons per band $f$ of a channel ($\nu_c = f\,C_c$;
an unpolarized crystal is time-reversal symmetric and has $C_c = 0$), the Chern vector is
$\mathbf K = \operatorname{sign}(\det \mathbf G)\sum_c \nu_c\mathbf G_c$, where
$\det\mathbf G$ has the handedness of the cell, and

$$
\sigma_{ab} = \frac{e^2}{h}\,\frac{1}{2\pi}\,\epsilon_{abd}K_d ,
$$

the quantized conductivity $\sigma_{xy} = C\,e^2/h$ per plane (Thouless and colleagues; the
sign is that of Xiao, Chang and Niu). The factor $\operatorname{sign}(\det\mathbf G)$ cancels
the handedness carried by each $C_c$, so the conductivity does not depend on how the cell is
labeled, right- or left-handed. A time-reversal-symmetric insulator gives zero.

(berry-metals)=
## The anomalous Hall conductivity of a metal

A metal has no occupied manifold, so there is no Chern number to count, and its polarization
and Berry phase are undefined. Its intrinsic anomalous Hall conductivity is not: it is the
Fermi-sea term, the Berry curvature of every band weighted by its occupation. The calculator
replaces the integer $f\,C_c$ of an insulator by the filling-weighted plane number

$$
\nu_c = \frac{1}{2\pi}\Big\langle \sum_n \int f_n\,\Omega_n \Big\rangle ,
$$

the integral over a plane normal to $\mathbf G_c$, averaged over the layers of the mesh along
$\mathbf G_c$, with $f_n$ the occupation of band $n$ in electrons: the electrons per band
times the smearing function of the converged calculation (its `smearing` and width) at a Fermi
level solved again on the Berry mesh (see "The Fermi level" below). $\mathbf K$ and $\sigma_{ab}$ follow from $\nu_c$ exactly as above. Only the
intrinsic term is computed: no extrinsic contribution, no orbital magnetization and no
Fermi-surface form of the conductivity is provided.

**The weighted field.** The Berry curvature of a single band is not defined where bands
touch, so the lattice field is built from groups of bands. On every plaquette and in every
spin channel the bands up to a cap are cut into groups of contiguous bands. Each group
contributes its non-Abelian plaquette flux (the determinants of the unitarized sub-block of
its links, as in the Chern number above), multiplied by the mean occupation of its members over
the four corners; $\nu_c$ is $1/2\pi$ times the sum over the plaquettes. The result is gauge
invariant, and in the continuum limit the sum tends to $\int f_n\,\Omega_n$ for every isolated
band. The rules that make the groups:

- A cut between two neighboring bands is allowed only where they are more than $10^{-6}$ Ha
  apart at all four corners: degenerate and touching bands always share a group.
- Consecutive groups that are full at every corner (an occupation of at least $1 - 10^{-6}$ of a
  full band) merge into one sea. An insulator is one sea, so it gives exactly the integers of
  the insulator section: the insulator limit is $\sigma_{xy} = C\,e^2/h$, to the last digit.
- A group that is empty at every corner (at most $10^{-6}$ of a full band) contributes nothing
  and is skipped.
- Two bands can swap order inside a plaquette without touching at a corner. That shows as a
  link of the group whose smallest singular value falls below 0.5, and the group merges with its
  neighbor (the one above, or below at the cap) until every group is clean. A merged group has
  one weight, so on such a plaquette the result is off by $(f_n - f_m)\,\phi$ when the two
  bands are at different fillings. The error lives only on the plaquettes that hold a crossing
  and vanishes as the mesh refines; `merged` counts those plaquettes (and `plaquettes` the
  total).
- The cap, the number of bands linked at every point (`cap`), is the number of SCF levels with
  a non-negligible occupation (above $10^{-6}$ of a full band) plus two. A Berry k-point that
  needs more bands rebuilds the mesh with a larger cap. An occupied group that reaches the
  largest band the crystal has, with no gap above it, raises `ValueError`.
- A group whose flux nears $\pi$ (above $3\pi/4$), where the lattice field aliases, merges with
  its neighbor as a crossing does and is counted in `merged`. A point node near the center of a
  plaquette keeps every link large and carries $\pi$ at any mesh, so a refusal there would never
  clear; the merge gives the group's flux a shared weight instead. Only a group that already spans
  every linked band cannot merge further, and that raises `ValueError`.

**The Fermi level.** The occupations use a Fermi level solved again on the Berry mesh, with the
converged smearing and width, so that the mesh holds the electrons of the cell. The level of the
SCF run does not carry over to another mesh: it holds the electrons on the SCF mesh, and with a
narrow smearing its bisection can stop anywhere on the flat plateau of an insulator's electron
count. On the $2^3$ Berry mesh of a LiH cell with a narrow smearing, the SCF level misplaced 0.44
electrons (a figure from running the weighted formula on that insulator, which a public request
would send to the Chern path). The view reports the level used (`fermi`, in eV), `scf_electron_error` (the electrons
the SCF level would put in the cell on this mesh, minus the cell's electrons: what re-solving
corrected) and `plaquettes` (the plaquettes of every layer of every plane family and channel, of which
`merged` merged band groups). For an insulator `fermi` is the SCF level and `scf_electron_error`
is 0.

Which path a request takes follows from the SCF occupations. A crystal whose occupations are all
empty or full within $10^{-4}$ and whose occupied manifold is separated from the empty bands
is an insulator and gives its Chern numbers. Any other crystal takes the weighted field,
including a gapped crystal whose smearing is wider than its gap: its answer is then close to
the quantized value but not equal to it, and narrowing the smearing recovers the integer.

```python
hall = calc.get_anomalous_hall(kpts=(8, 8, 8))     # runs the SCF first; a metal or an insulator
print(hall.sigma)                                  # (3, 3), S/cm
print(hall.sigma_coarse)                           # the same on the every-other-point sub-mesh
print(hall.numbers, hall.merged, hall.plaquettes)  # nu_c, merged plaquettes, of how many
print(hall.fermi, hall.scf_electron_error, hall.cap)   # eV on the Berry mesh, electrons, bands linked

sigma = calc.get_anomalous_hall_conductivity(kpts=(8, 8, 8))   # the same sigma, S/cm
```

`get_anomalous_hall_conductivity` still returns the `(3, 3)` array in S/cm, antisymmetric,
with the factor $\operatorname{sign}(\det\mathbf G)$, for an insulator and now for a metal;
`get_anomalous_hall` returns the same array in `sigma` together with what is behind it (the
fields are in the table below). Both run the SCF first if the calculator has no result.

**Convergence.** The Hall conductivity of a metal converges slowly with the k mesh: the Fermi
surface cuts the mesh, and the curvature spikes where bands come close. The result therefore
carries an estimate of its own error. `sigma_coarse` is the same formula on the sub-mesh of every
other point of the same states (links over two steps, so it costs no new diagonalization). It
exists when every entry of `kpts` is even and at least 8, and is `None` otherwise: a 4-point
direction leaves a 2-point sub-mesh, whose plane number is exactly zero, so below 8 no estimate is
formed and the warning says so, naming 8. A `UserWarning` is issued, and the value still returned,
when

- `sigma_coarse` cannot be formed (an odd entry, or an entry below 8),
- it differs from `sigma` by more than 10 % of the largest component of `sigma` (differences
  below $5\times10^{-5}$ S/cm count as zero), or
- more than 25 % of the plaquettes merged band groups: the mesh does not resolve the bands near
  the Fermi level, and the value is mostly the shared weight of those groups.

None is a certificate. The sub-mesh agreeing with the mesh does not prove convergence, and a
single mesh never does: compute a metal on two meshes (for example $8^3$ and $16^3$) and compare
before using a value. On the doped lattice model used to test the formula the error fell about
fourfold per halving of the mesh step, through merged crossings too; a real Fermi surface can
converge more slowly.

For a magnet whose symmetry fixes the shape of the tensor, the symmetry itself is the most
direct convergence check. A cubic ferromagnet magnetized along $z$ must give $\sigma_{xy}$ alone,
so $\sigma_{yz}/\sigma_{xy}$ (zero when converged) measures how far the mesh is from
converged: bcc Fe on a $5^3$ mesh gives $\sigma_{yz}/\sigma_{xy} = 0.38$, and the mesh breaks
the cubic symmetry visibly. A small ratio is necessary, not sufficient: fcc Ni on $4^3$ keeps
the symmetry to $10^{-3}$ with most of its plaquettes merged.

**Spin.**

- An unpolarized metal gives zero: time reversal makes each channel's Hamiltonian satisfy
  $H(-\mathbf k) = H(\mathbf k)^*$, so its curvature is odd in $\mathbf k$.
- A collinear spin-polarized metal without spin-orbit coupling also gives zero: each channel's
  Hamiltonian is real, so each channel's curvature is odd and integrates to zero. Both channels
  are computed and summed anyway; a nonzero value there would be a defect.
- A nonzero intrinsic conductivity needs a spinor calculation with spin-orbit coupling (a fully
  relativistic dataset such as `lda-fr`) in a magnetic state, which breaks time reversal.
- Each channel's plane number is weighted by its electrons per band $f$ (2 unpolarized, 1 for a
  spin-polarized channel and for the spinor channel), for an insulator and a metal alike.
- Every functional, `vdw="ts"` and `hubbard_u` enter through the converged states, as in the rest
  of this guide.

**What is tested, and what is not.** Validated at the lattice-model level and by symmetry
relations; a quantitative value for a metal is not yet available. On a doped
Qi-Wu-Zhang model the plane number agrees with an independent continuum integral of the
occupation-weighted curvature: quantized in the gap, continuous inside a band, and converging
with the mesh. Two decoupled copies of the model whose bands cross exercise the merge and agree
with the sum of their continua. Random gauges change no flux, a real model gives zero at every
filling, and the insulator limit equals the Chern-number field plaquette by plaquette. On DFT
crystals the tests check relations the answer must satisfy. An insulator through the weighted
formula reproduces its Chern layer numbers. The Hall conductivity is odd in the magnetization
(fcc Ni and bcc Fe), unchanged by relabeling the cell (swapping $\mathbf a_1$ and $\mathbf a_2$ of
Ni), and Fe has $\nu_1 = \nu_2$. In an
unpolarized noncentrosymmetric metal (BeLi) and a collinear spin-polarized one without
spin-orbit coupling (a displaced zincblende BeB), the layers of either sign, individually
nonzero, cancel to $10^{-10}$ in every family and channel: a nontrivial cancellation, not a
vanishing field.

**A metal's value is not quantitative on the meshes that are affordable.** On fcc Ni with a
$4^3$ mesh, 180 of 192 plaquettes merged band groups (the warning above fires), so the value is
mostly the shared weight of merged groups; bcc Fe on $5^3$ breaks the cubic symmetry
($\sigma_{yz}/\sigma_{xy} = 0.38$). Neither value is comparable with a published conductivity.
A link costs about a second on Ni, so a mesh of 12 points per direction takes about an hour.
A quantitative value waits on denser meshes or a Wannier-interpolation route, which does
not exist yet. Treat a metal's value as unvalidated until it is the same on two meshes, obeys
the symmetry of the material and matches a reference you trust.

## Units and the returned views

Angles are in radians, k-points in fractional coordinates of the $\mathbf G_j$. Lengths are in Å,
areas in Å², the polarization in C/m² and the conductivity in S/cm; a calculator built with
`atomic_units=True` reports atomic units instead.

| View | Fields |
| --- | --- |
| `BerryPhase` | `phases` `(spins, n1, n2)` of the strings over the transverse mesh (`transverse_axes`, from the fractional origins `starts`); `phase` and `phase_spread`: the string average per channel and the largest distance of a string from it; `wannier_charge_centers` and `connection`: one `(n1, n2, n_bands)` array per channel (fractional, and Å); `bands`; `minimum_singular_value` |
| `BerryCurvature` | `curvature` (Å²) and `flux` (rad), `(spins, n_a, n_b)`, at the fractional plaquette `centers`; `orientation` (the unit vector of $\mathbf G_a\times\mathbf G_b$); `chern_numbers` per channel and `chern_number`; `axes`, `size`, `offset`, `bands`; `max_flux` |
| `AnomalousHall` | `sigma` `(3, 3)` (S/cm) and `sigma_coarse` (the every-other-point sub-mesh value of a metal, or `None`); `metallic`; `numbers`: the three plane numbers $\nu_c$, electrons per band included; `layers`: the plane number of every layer, indexed `[c][channel][layer]`; `chern_numbers`: the integers per plane family and channel (empty for a metal); `strings`: the mesh; `max_flux` (rad); `cap`, `merged` and `plaquettes` (bands linked per point, the plaquettes where a crossing or a node merged two groups, and all plaquettes of the mesh; 0 for an insulator); `fermi` (eV, the level re-solved on the Berry mesh for a metal, the SCF level for an insulator) and `scf_electron_error` (the electrons the SCF level would misplace on this mesh; 0 for an insulator) |
| `Polarization` | `vector` (C/m², branch nearest zero); `fractional` (in quanta, each in $(-1/2, 1/2]$); `raw`; `quanta` (rows $g\,e\,\mathbf a_j/\Omega$, C/m²); `electronic` and `ionic` (Cartesian, unreduced); `phases` `(spins, 3)`; `phase_spread`; `strings`; `quantum` ($g$); `degeneracy` ($f$) |

**The branch.** A polarization is defined modulo its quanta, so only a change along a path is
physical. `vector` and `fractional` are folded to the branch nearest zero; `raw` is the
ionic plus electronic coordinate, in the same units of the quanta, before the fold. To follow
the polarization along a path (a displacement of the atoms, a strain), work with `raw`, not
with the folded `vector`s, which jump by a quantum wherever the fold changes. `raw` is itself
continuous only modulo $f/g$ quanta (a jump of the electronic branch) and $Z_a/g$ quanta (an
ion crossing the cell boundary), so fold each difference to the nearest lattice point:
$\Delta = d - \operatorname{round}(d)$ for a raw difference $d$ in quanta, with points along the
path close enough that the true change stays well below half a quantum.

**The spread warning.** If the string phases along some $\mathbf G_j$ spread by more than
$\pi/2$ about their average, the polarization is still returned, with a `UserWarning` that
names the direction and the spread. $\pi/2$ is the edge of branch uniqueness: beyond it the
pairwise differences of the string phases may exceed $\pi$ and the branch of their average is
no longer unique. The warning is not a bound on the transverse sampling error, and its absence
is not a certificate (a single transverse point never warns). To measure that error, compare
the polarization on two transverse meshes, one denser along the directions transverse to
$\mathbf G_j$.

## Spin and spin-orbit coupling

- **Unpolarized.** One channel, $f = 2$ electrons per band; the views carry a leading
  `spins` axis of length 1.
- **Spin-polarized** (`spin=True`). Two channels, each with $f = 1$. Phases, centers, curvature
  and Chern numbers are per channel; the polarization and `get_chern_number` sum them. With
  zero moments the result equals the unpolarized one channel by channel. The Hall conductivity
  weights each channel's plane number by its $f$ (a metal's is
  [zero without spin-orbit coupling](#berry-metals)).
- **Spin-orbit coupling** (a crystal on a fully relativistic dataset such as `lda-fr`, a
  two-component spinor calculation). One channel of spinor bands, $f = 1$, the number of occupied
  bands equal to the number of electrons. The overlaps sum both spinor components. For light
  elements the fully relativistic (`lda-fr`) and scalar-relativistic (`lda-sr`) datasets give the
  same result: the same energy and polarization, with every spinor string phase and plaquette
  flux twice the collinear one (the spinor channel has $f = 1$ and the collinear one $f = 2$
  electrons per band). With real spin-orbit coupling, on a time-reversal-invariant string the
  Wilson centers come in Kramers pairs.

## Functionals

The Berry quantities read the converged Hamiltonian of the crystal: the potential, the
DFT+U operator, the exact exchange of a hybrid and the meta-GGA kinetic potential are all
inside the states that are diagonalized again at each k-point. No functional needs its
own Berry code. `lda`, `pbe`, `r2scan`, `pbe+rvv10` and `hse06` run, with `vdw="ts"`
(a dispersion energy that does not change the states, so the Berry quantities are the
same) and with `hubbard_u=`. What the crystal SCF itself refuses stays refused: a
hybrid or a spin-orbit run needs the whole k mesh, and DFT+U with a hybrid or on a spinor
crystal does not run ([DFT](dft.md), [DFT+U](dft_u.md)).

(berry-refusals)=
## What raises

| Request | Error |
| --- | --- |
| a molecule (no periodic axis) | `ValueError`: a molecule has no Brillouin zone |
| a quantum method on a crystal | `NotImplementedError`: the supercell state has no Bloch bands |
| a crystal loaded from a snapshot (`load_crystal=`) | `NotImplementedError`: states off the SCF mesh need the native setup, which a snapshot does not have |
| a metal, a semimetal or a fractionally occupied state: polarization, `bands=None` phase and curvature | `ValueError`: undefined; narrow the smearing for an insulator |
| a metal: `get_anomalous_hall` and `get_anomalous_hall_conductivity` | no error: the intrinsic Fermi-sea term, with a `UserWarning` when its mesh estimate is missing (an entry of `kpts` that is odd or below 8), differs by more than 10 %, or more than 25 % of the plaquettes merged band groups ([metals](#berry-metals)) |
| a metal: an occupied band group that reaches the last band the crystal has, a singular group link, or a group spanning every linked band whose flux is above $3\pi/4$ | `ValueError`: more bands are needed, or refine the mesh (a group whose flux nears $\pi$ but has a neighbor merges with it instead) |
| a charged cell: polarization | `ValueError` |
| a mesh below the minimum, or a bad direction, plane, offset or band group | `ValueError` |
| a link overlap with a singular value below $10^{-6}$, a plaquette flux above $3\pi/4$, or a band group that is not isolated | `ValueError`: refine the mesh |
| Chern numbers that differ between the layers of a plane family | `ValueError`: the gap closes or the mesh is too coarse |

An occupied manifold, which the insulator paths need, is accepted when every SCF occupation is empty or full within $10^{-4}$,
there is a direct gap above $10^{-6}$ Ha at every Berry k-point, and the highest occupied
level lies below the lowest empty one over the whole mesh and every channel. The Fermi level
plays no part. The Hall conductivity does not need it: a crystal that fails these checks
because of its occupations is a metal for that request, and takes the weighted field.

## What is native and what is not

The six methods above are native and tested on lattice models with known answers
(a Su-Schrieffer-Heeger chain, a Rice-Mele cycle, a Qi-Wu-Zhang model, and for a metal's Hall
conductivity a doped Qi-Wu-Zhang model against its continuum integral), on gauge and
rigid-translation invariance of the crystal's links, on time-reversal and inversion
symmetry, and on each functional and spin case above. A metal's Hall conductivity is also
checked on DFT crystals by symmetry relations (odd in the magnetization, invariant under
relabeling the cell, nontrivial cancellations in BeLi and a displaced BeB). None of the six is
yet compared with a published value of the polarization or the Hall conductivity of a real
material, and a metal's Hall conductivity is not quantitative on the meshes that are
affordable: a quantitative value waits on denser meshes or a Wannier-interpolation route
([what is tested](#berry-metals)).

The Berry phase, the polarization and the occupied-manifold curvature of a metal are undefined
and raise `ValueError`, as above. The orbital magnetization and the Fermi-surface form of the
Hall conductivity are not provided. The Born effective charges and the piezoelectric tensor
(B3) and the $\mathbb Z_2$ invariant of a time-reversal-symmetric insulator (B4) are planned and
not available: see [not yet available](#not-yet-available).

## Scientific references

The bibliography registers these under the analyses `"berry-phase"`, `"berry-curvature"`,
`"anomalous-hall"` and `"polarization"`.

- Berry, *Proceedings of the Royal Society A* **392**, 45 (1984) (`Berry1984`),
  [doi:10.1098/rspa.1984.0023](https://doi.org/10.1098/rspa.1984.0023).
- Thouless, Kohmoto, Nightingale and den Nijs, *Physical Review Letters* **49**, 405 (1982)
  (`Thouless1982`), [doi:10.1103/PhysRevLett.49.405](https://doi.org/10.1103/PhysRevLett.49.405).
- King-Smith and Vanderbilt, *Physical Review B* **47**, 1651 (1993) (`KingSmith1993`),
  [doi:10.1103/PhysRevB.47.1651](https://doi.org/10.1103/PhysRevB.47.1651).
- Resta, *Reviews of Modern Physics* **66**, 899 (1994) (`Resta1994`),
  [doi:10.1103/RevModPhys.66.899](https://doi.org/10.1103/RevModPhys.66.899).
- Fukui, Hatsugai and Suzuki, *Journal of the Physical Society of Japan* **74**, 1674 (2005)
  (`Fukui2005`), [doi:10.1143/JPSJ.74.1674](https://doi.org/10.1143/JPSJ.74.1674).
- Xiao, Chang and Niu, *Reviews of Modern Physics* **82**, 1959 (2010) (`Xiao2010`),
  [doi:10.1103/RevModPhys.82.1959](https://doi.org/10.1103/RevModPhys.82.1959).
- Yu, Qi, Bernevig, Fang and Dai, *Physical Review B* **84**, 075119 (2011) (`Yu2011`),
  [doi:10.1103/PhysRevB.84.075119](https://doi.org/10.1103/PhysRevB.84.075119).
