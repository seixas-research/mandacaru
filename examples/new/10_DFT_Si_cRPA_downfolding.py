# -*- coding: utf-8 -*-
# file: examples/new/10_DFT_Si_cRPA_downfolding.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Downfolding a crystal to a small many-body model with the constrained random
# phase approximation (cRPA), then solving the model with ADAPT-VQE.
#
# Diamond silicon, Kohn-Sham DFT (LDA, PAW-LCAO DZP).  The pipeline:
#
#   1. a self-consistent crystal run;
#   2. maximally localized Wannier functions of the sp3 manifold: eight
#      hybrids, two atoms x four, disentangled from the valence and the lowest
#      conduction bands (everything up to 1 eV above the Fermi level frozen,
#      so those bands are reproduced exactly);
#   3. the fragment: the two hybrids that face each other across one Si-Si
#      bond, a two-site, two-electron model
#
#          H = sum_pq,s t_pq c+_ps c_qs
#              + 1/2 sum_pqrs,ss' W_pqrs c+_ps c+_qs' c_ss' c_rs ;
#
#   4. the one-body part t = h_KS - V_dc: the Kohn-Sham Hamiltonian between
#      the two hybrids minus the double counting V_dc, the Hartree and exchange
#      potential of the fragment's own electrons at the Kohn-Sham density
#      (so that the model's Fock matrix at that density is h_KS again --
#      checked below);
#   5. the two-body part W, bare and screened.  The static cRPA screens the
#      bare Coulomb interaction with every transition except those inside the
#      fragment's own Bloch subspace -- the model treats those explicitly --
#
#          W = [1 - v chi_r]^-1 v ,   chi_r = chi_KS - chi_target ,
#
#      at omega = 0, then projected on the hybrids: on-site U, inter-site U'
#      (= V) and exchange J.  Full RPA (nothing excluded) is shown for
#      comparison; it screens a little more than cRPA;
#   6. ADAPT-VQE on the screened model through Mandacaru, against the exact
#      diagonalization of the same model.
#
# The table gives the Kohn-Sham hopping t_KS (the same for every interaction)
# and the model's hopping t = t_KS - V_dc, which moves with W because the
# double counting does.  The band check compares the Wannier-interpolated
# bands with the Kohn-Sham ones along a path: exact on the mesh, and between
# mesh points an error that falls with the mesh (Si: ~100 meV at 4x4x4, a few
# meV at 8x8x8).
#
# What to expect: the bare U is ~12 eV; screening brings it to ~4 eV, while J
# is screened least.  These numbers are NOT converged on this mesh: the
# screened U still falls ~0.1 eV per step of the k-mesh at 6x6x6, and with
# the number of empty bands (HISTORY.md, Phase J).  Converge both before
# quoting a value.  The model is static (no frequency dependence) and not
# self-consistent.
#
# Slow: one crystal run, a disentanglement and three downfolds; expect
# several minutes.

import os

import matplotlib

matplotlib.use("Agg")                                     # write the file, no window
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.core.mapping import Fermion
from mandacaru.units import HARTREE_TO_EV

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

LATTICE = 5.43                                            # experimental a (Angstrom)
BASIS = {"name": "PAW-LCAO", "size": "DZP"}
KPTS = {"size": (4, 4, 4), "gamma": True}                 # SCF and Wannier mesh
FROZEN_ABOVE_FERMI = 1.0                                  # frozen window top (eV)
PATH, NPOINTS = "LGXWKG", 120                             # band path for the check

# 1. The crystal
atoms = bulk("Si", "diamond", a=LATTICE)                  # two atoms per cell
atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis=BASIS,
                       h=0.25,                            # Grid spacing (Angstrom)
                       kpts=KPTS,
                       txt=os.path.join(OUT, "output_10.txt"),
                       references=os.path.join(OUT, "references_10.bib"))
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()

# 2. Wannier functions of the sp3 manifold (energy windows in eV, on the
#    scale of the Fermi level)
windows = {"outer": (fermi - 20.0, fermi + 20.0),
           "frozen": (fermi - 20.0, fermi + FROZEN_ABOVE_FERMI)}
wannier = atoms.calc.wannier(8, guess="sp3", windows=windows)

# 3. The fragment: the hybrid on each atom closest to the bond between them
start, end = atoms.positions[0], atoms.positions[1]
bond = [int(np.argmin(np.linalg.norm(wannier.centers - (a + 0.2 * (b - a)),
                                     axis=1)))
        for a, b in ((start, end), (end, start))]


# 4-5. The model with a bare, a fully screened and a cRPA-screened interaction
def couplings(problem):
    """The Kohn-Sham hopping, the model's hopping t (after the double
    counting, so it moves with the interaction), on-site U, inter-site U'
    (here "V") and exchange J of the two-site model, in eV (the arrays are
    in Hartree)."""
    h, g = problem.one_body, problem.two_body
    return {"t_KS": problem.kohn_sham[0, 1].real * HARTREE_TO_EV,
            "t": h[0, 1].real * HARTREE_TO_EV,
            "U": g[0, 0, 0, 0].real * HARTREE_TO_EV,
            "V": g[0, 1, 0, 1].real * HARTREE_TO_EV,
            "J": g[0, 1, 1, 0].real * HARTREE_TO_EV}


models = {screening: wannier.downfold(bond, screening=screening)
          for screening in (None, "rpa", "crpa")}

# The double counting restores the Kohn-Sham Fock matrix at the KS density
crpa = models["crpa"]
gamma, g = crpa.density_matrix, crpa.two_body
fock = (crpa.one_body + np.einsum("rs,prqs->pq", gamma, g)
        - 0.5 * np.einsum("rs,prsq->pq", gamma, g))
fock_error = float(np.abs(fock - crpa.kohn_sham).max()) * HARTREE_TO_EV


# 6. ADAPT-VQE on the cRPA model, against exact diagonalization
def exact_ground_state(problem):
    """Lowest eigenvalue (eV) of the model in its (up, down) sector."""
    n = 2 * problem.n_spatial_orbitals
    H = problem.fermion_hamiltonian().to_matrix(n)
    number = [np.real(np.diag(Fermion({((i, True), (i, False)): 1.0},
                                      n_modes=n).to_matrix(n)))
              for i in range(n)]
    up = sum(number[:n // 2])
    down = sum(number[n // 2:])
    n_up, n_down = problem.num_particles
    sector = np.flatnonzero((np.abs(up - n_up) < 1e-9)
                            & (np.abs(down - n_down) < 1e-9))
    return np.linalg.eigvalsh(H[np.ix_(sector, sector)])[0] * HARTREE_TO_EV


exact = exact_ground_state(crpa)
adapt = Mandacaru(method="adapt-vqe", **crpa.as_quantum_problem()).run()

# Band check: Wannier-interpolated bands against the non-self-consistent
# Kohn-Sham bands along a path
bands = atoms.calc.band_structure(path=PATH, npoints=NPOINTS)
direct = bands.energies[0] - fermi                        # (nk, nb), eV
interpolated = wannier.interpolate(bands.path.kpts) - fermi
frozen = direct <= FROZEN_ABOVE_FERMI
worst = 0.0
for k in range(len(direct)):
    inside = direct[k][frozen[k]]
    worst = max(worst, float(np.abs(interpolated[k][:inside.size]
                                    - inside).max()))

x, ticks, labels = bands.path.get_linear_kpoint_axis()
fig, ax = plt.subplots(figsize=(6.0, 4.5))
ax.plot(x, direct, color="lightgray", lw=2.5)
ax.plot(x, interpolated, color="C3", lw=1.0, ls="--")
ax.axhline(FROZEN_ABOVE_FERMI, color="C0", ls=":", lw=0.8)
ax.axhline(0.0, color="gray", ls=":", lw=0.8)
ax.set_xticks(ticks, [label.replace("G", r"$\Gamma$") for label in labels])
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-14.0, 8.0)
ax.set_ylabel("E - E_F (eV)")
ax.set_title("Si: Kohn-Sham bands (gray) and the sp3 Wannier interpolation")
fig.savefig(os.path.join(OUT, "10_si_crpa_bands.png"), dpi=200,
            bbox_inches="tight")

# Report
lines = [f"Diamond Si, LDA, {BASIS['name']} {BASIS['size']}, "
         f"{'x'.join(map(str, KPTS['size']))} Gamma-centered mesh",
         f"sp3 Wannier functions: Omega_I = {wannier.omega_invariant:.4f} A^2, "
         f"spreads {np.round(wannier.spreads, 3).tolist()} A^2",
         f"interpolated bands inside the frozen window: max error "
         f"{1e3 * worst:.3f} meV",
         f"fragment: hybrids {bond} (one Si-Si bond), "
         f"{crpa.n_electrons} electrons",
         "",
         "interaction  t_KS (eV)   t (eV)   U (eV)  U' (eV)   J (eV)"]
for screening, problem in models.items():
    c = couplings(problem)
    lines.append(f"{screening or 'bare':<12} {c['t_KS']:9.3f} {c['t']:8.3f} "
                 f"{c['U']:8.3f} "
                 f"{c['V']:8.3f} {c['J']:8.3f}")
lines += ["",
          f"Fock matrix at the KS density - h_KS (cRPA model): "
          f"{1e3 * fock_error:.2e} meV",
          f"ground state of the cRPA model: exact {exact:.6f} eV, "
          f"ADAPT-VQE {adapt.optimal_energy:.6f} eV "
          f"(difference {abs(adapt.optimal_energy - exact):.1e} eV)"]
report = "\n".join(lines)
with open(os.path.join(OUT, "10_si_crpa.dat"), "w") as table:
    table.write(report + "\n")
print(report)
