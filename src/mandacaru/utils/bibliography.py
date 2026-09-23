# -*- coding: utf-8 -*-
# file: src/mandacaru/utils/bibliography.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The reference database a run's ``references.bib`` is assembled from.

One :class:`Reference` per method, basis family, mapping, pool, optimizer and
code Mandacaru can use.  :func:`references_for` turns a run's configuration
into the subset that actually applies, and :func:`bibtex` renders it.  Nothing
here inspects a calculation — the mapping from a run to a set of keys lives in
:mod:`mandacaru.utils.citations`, so this file stays a data table that can be
checked against the literature line by line.

**Journal names are abbreviated** (ISO-4 style, as the journals themselves
print them: ``Nat. Commun.``, ``Phys. Rev. B``, ``J. Chem. Phys.``), which is
what most quantum-chemistry and physics styles expect.

**Provenance.** Every entry whose PDF sits in the owner's library was taken
from that file's own metadata and title page, not from memory: the ADAPT-VQE
family (Grimsley 2019, Tang 2021, Yordanov 2021, Ramoa 2025, Anastasiou 2024,
Vaquero-Sabater 2025, Van Dyke 2024, Shkolnikov 2021, Seki 2020, Singh 2025,
Li 2025) came from ``~/Dropbox/2.Library/.../ADAPT-VQE/``.  The rest are
standard citations; **check one before relying on it in a manuscript** — a
`` % unverified`` comment marks those that were not read from a local source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reference:
    """One bibliography entry.

    ``key`` is the BibTeX citation key and the identifier the rest of the code
    uses.  ``entry`` is the rendered BibTeX record, already abbreviated and
    indented.  ``note`` says what in a run pulls the entry in, and is written
    into the file as a comment so a reader can tell why a paper is there.
    ``verified`` is ``False`` when the record was not read from a local PDF.
    """

    key: str
    entry: str
    note: str
    verified: bool = True
    topics: tuple = field(default_factory=tuple)


def _ref(key, entry, note, verified=True, topics=()):
    return Reference(key=key, entry=entry.strip("\n"), note=note,
                     verified=verified, topics=tuple(topics))


# --------------------------------------------------------------------------- #
# Algorithms.
# --------------------------------------------------------------------------- #

_ALGORITHMS = [
    _ref("Grimsley2019", """
@article{Grimsley2019,
  author  = {Grimsley, Harper R. and Economou, Sophia E. and Barnes, Edwin and
             Mayhall, Nicholas J.},
  title   = {An adaptive variational algorithm for exact molecular simulations
             on a quantum computer},
  journal = {Nat. Commun.},
  volume  = {10},
  pages   = {3007},
  year    = {2019},
  doi     = {10.1038/s41467-019-10988-2}
}""", "ADAPT-VQE; also the fermionic operator pool", topics=("adapt", "pool")),
    _ref("Peruzzo2014", """
@article{Peruzzo2014,
  author  = {Peruzzo, Alberto and McClean, Jarrod and Shadbolt, Peter and
             Yung, Man-Hong and Zhou, Xiao-Qi and Love, Peter J. and
             Aspuru-Guzik, Al{\\'a}n and O'Brien, Jeremy L.},
  title   = {A variational eigenvalue solver on a photonic quantum processor},
  journal = {Nat. Commun.},
  volume  = {5},
  pages   = {4213},
  year    = {2014},
  doi     = {10.1038/ncomms5213}
}""", "the variational quantum eigensolver", verified=False, topics=("vqe",)),
    _ref("Romero2019", """
@article{Romero2019,
  author  = {Romero, Jonathan and Babbush, Ryan and McClean, Jarrod R. and
             Hempel, Cornelius and Love, Peter J. and Aspuru-Guzik, Al{\\'a}n},
  title   = {Strategies for quantum computing molecular energies using the
             unitary coupled cluster ansatz},
  journal = {Quantum Sci. Technol.},
  volume  = {4},
  pages   = {014008},
  year    = {2019},
  doi     = {10.1088/2058-9565/aad3e4}
}""", "the UCCSD ansatz", verified=False, topics=("uccsd",)),
    _ref("Higgott2019", """
@article{Higgott2019,
  author  = {Higgott, Oscar and Wang, Daochen and Brierley, Stephen},
  title   = {Variational quantum computation of excited states},
  journal = {Quantum},
  volume  = {3},
  pages   = {156},
  year    = {2019},
  doi     = {10.22331/q-2019-07-01-156}
}""", "excited states by variational quantum deflation (energy_levels)",
         verified=False, topics=("deflation",)),
    _ref("Nakanishi2019", """
@article{Nakanishi2019,
  author  = {Nakanishi, Ken M. and Mitarai, Kosuke and Fujii, Keisuke},
  title   = {Subspace-search variational quantum eigensolver for excited states},
  journal = {Phys. Rev. Res.},
  volume  = {1},
  pages   = {033062},
  year    = {2019},
  doi     = {10.1103/PhysRevResearch.1.033062}
}""", "the subspace-search solvers", verified=False, topics=("subspace",)),
    _ref("Kitaev1995", """
@misc{Kitaev1995,
  author        = {Kitaev, A. Yu.},
  title         = {Quantum measurements and the {A}belian stabilizer problem},
  year          = {1995},
  eprint        = {quant-ph/9511026},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.quant-ph/9511026}
}""", "quantum phase estimation", verified=False, topics=("qpe",)),
    _ref("AspuruGuzik2005", """
@article{AspuruGuzik2005,
  author  = {Aspuru-Guzik, Al{\\'a}n and Dutoi, Anthony D. and Love, Peter J.
             and Head-Gordon, Martin},
  title   = {Simulated quantum computation of molecular energies},
  journal = {Science},
  volume  = {309},
  pages   = {1704--1707},
  year    = {2005},
  doi     = {10.1126/science.1113479}
}""", "phase estimation applied to molecular energies", verified=False,
         topics=("qpe",)),
    _ref("Sim2019", """
@article{Sim2019,
  author  = {Sim, Sukin and Johnson, Peter D. and Aspuru-Guzik, Al{\\'a}n},
  title   = {Expressibility and entangling capability of parameterized quantum
             circuits for hybrid quantum-classical algorithms},
  journal = {Adv. Quantum Technol.},
  volume  = {2},
  pages   = {1900070},
  year    = {2019},
  doi     = {10.1002/qute.201900070}
}""", "the expressibility diagnostic", verified=False,
         topics=("expressivity",)),
]

# --------------------------------------------------------------------------- #
# Operator pools and growth strategies.
# --------------------------------------------------------------------------- #

_POOLS = [
    _ref("Tang2021", """
@article{Tang2021,
  author  = {Tang, Ho Lun and Shkolnikov, V. O. and Barron, George S. and
             Grimsley, Harper R. and Mayhall, Nicholas J. and Barnes, Edwin
             and Economou, Sophia E.},
  title   = {Qubit-{ADAPT}-{VQE}: An adaptive algorithm for constructing
             hardware-efficient ans{\\"a}tze on a quantum processor},
  journal = {PRX Quantum},
  volume  = {2},
  pages   = {020310},
  year    = {2021},
  doi     = {10.1103/PRXQuantum.2.020310}
}""", "the qubit operator pool", topics=("pool", "qubit")),
    _ref("Yordanov2021", """
@article{Yordanov2021,
  author  = {Yordanov, Yordan S. and Armaos, V. and Barnes, Crispin H. W. and
             Arvidsson-Shukur, David R. M.},
  title   = {Qubit-excitation-based adaptive variational quantum eigensolver},
  journal = {Commun. Phys.},
  volume  = {4},
  pages   = {228},
  year    = {2021},
  doi     = {10.1038/s42005-021-00730-0}
}""", "the qubit-excitation (QEB) pool", topics=("pool", "qeb")),
    _ref("Ramoa2025", """
@article{Ramoa2025,
  author  = {Ram{\\^o}a, Mafalda and Anastasiou, Panagiotis G. and
             Santos, Luis Paulo and Mayhall, Nicholas J. and Barnes, Edwin and
             Economou, Sophia E.},
  title   = {Reducing the resources required by {ADAPT}-{VQE} using coupled
             exchange operators and improved subroutines},
  journal = {npj Quantum Inf.},
  volume  = {11},
  pages   = {86},
  year    = {2025},
  doi     = {10.1038/s41534-025-01039-4}
}""", "the coupled-exchange-operator (CEO) pool", topics=("pool", "ceo")),
    _ref("Anastasiou2024", """
@article{Anastasiou2024,
  author  = {Anastasiou, Panagiotis G. and Chen, Yanzhu and
             Mayhall, Nicholas J. and Barnes, Edwin and Economou, Sophia E.},
  title   = {{TETRIS}-{ADAPT}-{VQE}: An adaptive algorithm that yields
             shallower, denser circuit ans{\\"a}tze},
  journal = {Phys. Rev. Res.},
  volume  = {6},
  pages   = {013254},
  year    = {2024},
  doi     = {10.1103/PhysRevResearch.6.013254}
}""", "the TETRIS growth strategy (tetris=True)", topics=("tetris",)),
    _ref("VaqueroSabater2025", """
@article{VaqueroSabater2025,
  author  = {Vaquero-Sabater, Nonia and Carreras, Abel and Casanova, David},
  title   = {Pruned-{ADAPT}-{VQE}: Compacting molecular ans{\\"a}tze by
             removing irrelevant operators},
  journal = {J. Chem. Theory Comput.},
  volume  = {21},
  pages   = {8720--8728},
  year    = {2025},
  doi     = {10.1021/acs.jctc.5c00535}
}""", "ansatz pruning (prune=True)", topics=("prune",)),
]

# --------------------------------------------------------------------------- #
# Fermion-to-qubit mappings.
# --------------------------------------------------------------------------- #

_MAPPINGS = [
    _ref("Jordan1928", """
@article{Jordan1928,
  author  = {Jordan, P. and Wigner, E.},
  title   = {{\\"U}ber das {P}aulische {\\"A}quivalenzverbot},
  journal = {Z. Phys.},
  volume  = {47},
  pages   = {631--651},
  year    = {1928},
  doi     = {10.1007/BF01331938}
}""", "the Jordan-Wigner transformation", verified=False,
         topics=("mapping", "jordan_wigner")),
    _ref("BravyiKitaev2002", """
@article{BravyiKitaev2002,
  author  = {Bravyi, Sergey B. and Kitaev, Alexei Yu.},
  title   = {Fermionic quantum computation},
  journal = {Ann. Phys.},
  volume  = {298},
  pages   = {210--226},
  year    = {2002},
  doi     = {10.1006/aphy.2002.6254}
}""", "the Bravyi-Kitaev transformation", verified=False,
         topics=("mapping", "bravyi_kitaev")),
    _ref("Seeley2012", """
@article{Seeley2012,
  author  = {Seeley, Jacob T. and Richard, Martin J. and Love, Peter J.},
  title   = {The {B}ravyi-{K}itaev transformation for quantum computation of
             electronic structure},
  journal = {J. Chem. Phys.},
  volume  = {137},
  pages   = {224109},
  year    = {2012},
  doi     = {10.1063/1.4768229}
}""", "the parity and Bravyi-Kitaev encodings of electronic structure",
         verified=False, topics=("mapping", "parity", "bravyi_kitaev")),
    _ref("Bravyi2017", """
@misc{Bravyi2017,
  author        = {Bravyi, Sergey and Gambetta, Jay M. and
                   Mezzacapo, Antonio and Temme, Kristan},
  title         = {Tapering off qubits to simulate fermionic {H}amiltonians},
  year          = {2017},
  eprint        = {1701.08213},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.1701.08213}
}""", "the two-qubit reduction of the parity mapping (parity_reduced)",
         verified=False, topics=("mapping", "parity_reduced")),
]

# --------------------------------------------------------------------------- #
# Bases, pseudopotentials and the real-space machinery.
# --------------------------------------------------------------------------- #

_BASES = [
    _ref("Bloechl1994", """
@article{Bloechl1994,
  author  = {Bl{\\"o}chl, P. E.},
  title   = {Projector augmented-wave method},
  journal = {Phys. Rev. B},
  volume  = {50},
  pages   = {17953--17979},
  year    = {1994},
  doi     = {10.1103/PhysRevB.50.17953}
}""", "the PAW / UPAW basis", verified=False, topics=("paw", "upaw")),
    _ref("Ivanov2024", """
@misc{Ivanov2024,
  author        = {Ivanov, Aleksei V. and S{\"u}nderhauf, Christoph and
                   Holzmann, Nicole and Ellaby, Tom and Kerber, Rachel N. and
                   Jones, Glenn and Camps, Joan},
  title         = {Quantum computation for periodic solids in second
                   quantization},
  year          = {2024},
  eprint        = {2408.03159},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2408.03159}
}""", "unitary PAW (basis='UPAW')", verified=False, topics=("upaw",)),
    _ref("Troullier1991", """
@article{Troullier1991,
  author  = {Troullier, N. and Martins, Jos{\\'e} Lu{\\'i}s},
  title   = {Efficient pseudopotentials for plane-wave calculations},
  journal = {Phys. Rev. B},
  volume  = {43},
  pages   = {1993--2006},
  year    = {1991},
  doi     = {10.1103/PhysRevB.43.1993}
}""", "the NCPP (Troullier-Martins) pseudopotentials", verified=False,
         topics=("ncpp",)),
    _ref("Kleinman1982", """
@article{Kleinman1982,
  author  = {Kleinman, Leonard and Bylander, D. M.},
  title   = {Efficacious form for model pseudopotentials},
  journal = {Phys. Rev. Lett.},
  volume  = {48},
  pages   = {1425--1428},
  year    = {1982},
  doi     = {10.1103/PhysRevLett.48.1425}
}""", "the Kleinman-Bylander separable nonlocal form", verified=False,
         topics=("ncpp", "oncvpsp")),
    _ref("Hamann2013", """
@article{Hamann2013,
  author  = {Hamann, D. R.},
  title   = {Optimized norm-conserving {V}anderbilt pseudopotentials},
  journal = {Phys. Rev. B},
  volume  = {88},
  pages   = {085117},
  year    = {2013},
  doi     = {10.1103/PhysRevB.88.085117}
}""", "the ONCVPSP pseudopotentials", verified=False, topics=("oncvpsp",)),
    _ref("Sankey1989", """
@article{Sankey1989,
  author  = {Sankey, Otto F. and Niklewski, David J.},
  title   = {Ab initio multicenter tight-binding model for molecular-dynamics
             simulations and other applications in covalent systems},
  journal = {Phys. Rev. B},
  volume  = {40},
  pages   = {3979--3995},
  year    = {1989},
  doi     = {10.1103/PhysRevB.40.3979}
}""", "confined numerical atomic orbitals", verified=False,
         topics=("nao", "energy_shift")),
    _ref("Junquera2001", """
@article{Junquera2001,
  author  = {Junquera, Javier and Paz, {\\'O}scar and S{\\'a}nchez-Portal,
             Daniel and Artacho, Emilio},
  title   = {Numerical atomic orbitals for linear-scaling calculations},
  journal = {Phys. Rev. B},
  volume  = {64},
  pages   = {235111},
  year    = {2001},
  doi     = {10.1103/PhysRevB.64.235111}
}""", "the smooth confining potential behind energy_shift", verified=False,
         topics=("nao", "energy_shift")),
    _ref("Artacho1999", """
@article{Artacho1999,
  author  = {Artacho, Emilio and S{\\'a}nchez-Portal, Daniel and Ordej{\\'o}n,
             Pablo and Garc{\\'i}a, Alberto and Soler, Jos{\\'e} M.},
  title   = {Linear-scaling ab-initio calculations for large and complex
             systems},
  journal = {Phys. Status Solidi B},
  volume  = {215},
  pages   = {809--817},
  year    = {1999},
  doi     = {10.1002/(SICI)1521-3951(199909)215:1<809::AID-PSSB809>3.0.CO;2-0}
}""", "the split-valence multiple-zeta construction", verified=False,
         topics=("multizeta", "split_norm")),
    _ref("Anglada2006", """
@article{Anglada2006,
  author  = {Anglada, Emilio and Soler, Jos{\\'e} M.},
  title   = {Filtering a distribution simultaneously in real and {F}ourier
             space},
  journal = {Phys. Rev. B},
  volume  = {73},
  pages   = {115122},
  year    = {2006},
  doi     = {10.1103/PhysRevB.73.115122}
}""", "Fourier filtering of the radial basis (filter=)", verified=False,
         topics=("filter",)),
    _ref("Slater1930", """
@article{Slater1930,
  author  = {Slater, J. C.},
  title   = {Atomic shielding constants},
  journal = {Phys. Rev.},
  volume  = {36},
  pages   = {57--64},
  year    = {1930},
  doi     = {10.1103/PhysRev.36.57}
}""", "Slater-type orbitals and the shielding rules behind their exponents",
         verified=False, topics=("slater",)),
    _ref("Hehre1969", """
@article{Hehre1969,
  author  = {Hehre, W. J. and Stewart, R. F. and Pople, J. A.},
  title   = {Self-consistent molecular-orbital methods. {I}. {U}se of {G}aussian
             expansions of {S}later-type atomic orbitals},
  journal = {J. Chem. Phys.},
  volume  = {51},
  pages   = {2657--2664},
  year    = {1969},
  doi     = {10.1063/1.1672392}
}""", "the STO-nG contraction scheme", verified=False, topics=("gto",)),
    _ref("Hehre1972", """
@article{Hehre1972,
  author  = {Hehre, W. J. and Ditchfield, R. and Pople, J. A.},
  title   = {Self-consistent molecular orbital methods. {XII}. {F}urther
             extensions of {G}aussian-type basis sets},
  journal = {J. Chem. Phys.},
  volume  = {56},
  pages   = {2257--2261},
  year    = {1972},
  doi     = {10.1063/1.1677527}
}""", "the Pople split-valence scheme (6-31G)", verified=False,
         topics=("pople",)),
    _ref("Dunning1989", """
@article{Dunning1989,
  author  = {Dunning, Thom H.},
  title   = {Gaussian basis sets for use in correlated molecular calculations.
             {I}. {T}he atoms boron through neon and hydrogen},
  journal = {J. Chem. Phys.},
  volume  = {90},
  pages   = {1007--1023},
  year    = {1989},
  doi     = {10.1063/1.456153}
}""", "the correlation-consistent (cc-pVXZ) shell structure", verified=False,
         topics=("dunning",)),
    _ref("Weigend2005", """
@article{Weigend2005,
  author  = {Weigend, Florian and Ahlrichs, Reinhart},
  title   = {Balanced basis sets of split valence, triple zeta valence and
             quadruple zeta valence quality for {H} to {R}n},
  journal = {Phys. Chem. Chem. Phys.},
  volume  = {7},
  pages   = {3297--3305},
  year    = {2005},
  doi     = {10.1039/B508541A}
}""", "the def2 shell structure", verified=False, topics=("def2",)),
    _ref("Kwee2008", """
@article{Kwee2008,
  author  = {Kwee, Hendra and Zhang, Shiwei and Krakauer, Henry},
  title   = {Finite-Size Correction in Many-Body Electronic Structure
             Calculations},
  journal = {Phys. Rev. Lett.},
  volume  = {100},
  pages   = {126404},
  year    = {2008},
  doi     = {10.1103/PhysRevLett.100.126404}
}""", "the size-dependent LDA of the KZK finite-size correction",
         topics=("periodic", "finite-size")),
    _ref("Fraser1996", """
@article{Fraser1996,
  author  = {Fraser, Louisa M. and Foulkes, W. M. C. and Rajagopal, G. and
             Needs, R. J. and Kenny, S. D. and Williamson, A. J.},
  title   = {Finite-size effects and {C}oulomb interactions in quantum {M}onte
             {C}arlo calculations for homogeneous systems with periodic
             boundary conditions},
  journal = {Phys. Rev. B},
  volume  = {53},
  pages   = {1814--1832},
  year    = {1996},
  doi     = {10.1103/PhysRevB.53.1814}
}""", "the model periodic Coulomb interaction", verified=False,
         topics=("periodic", "finite-size")),
    _ref("Williamson1997", """
@article{Williamson1997,
  author  = {Williamson, A. J. and Rajagopal, G. and Needs, R. J. and
             Fraser, L. M. and Foulkes, W. M. C. and Wang, Y. and
             Chou, M.-Y.},
  title   = {Elimination of {C}oulomb finite-size effects in quantum many-body
             simulations},
  journal = {Phys. Rev. B},
  volume  = {55},
  pages   = {R4851--R4854},
  year    = {1997},
  doi     = {10.1103/PhysRevB.55.R4851}
}""", "the MPC separation of Hartree and exchange-correlation kernels",
         verified=False, topics=("periodic", "finite-size")),
    _ref("Chiesa2006", """
@article{Chiesa2006,
  author  = {Chiesa, Simone and Ceperley, David M. and Martin, Richard M. and
             Holzmann, Markus},
  title   = {Finite-Size Error in Many-Body Simulations with Long-Range
             Interactions},
  journal = {Phys. Rev. Lett.},
  volume  = {97},
  pages   = {076404},
  year    = {2006},
  doi     = {10.1103/PhysRevLett.97.076404}
}""", "the structure-factor finite-size correction", verified=False,
         topics=("periodic", "finite-size")),
    _ref("BornKarman1912", """
@article{BornKarman1912,
  author  = {Born, Max and von K{\\'a}rm{\\'a}n, Theodore},
  title   = {{\\"U}ber {S}chwingungen in {R}aumgittern},
  journal = {Phys. Z.},
  volume  = {13},
  pages   = {297--309},
  year    = {1912}
}""", "the periodic boundary conditions a k-point mesh realizes as a supercell",
         verified=False, topics=("periodic",)),
    _ref("MonkhorstPack1976", """
@article{MonkhorstPack1976,
  author  = {Monkhorst, Hendrik J. and Pack, James D.},
  title   = {Special points for {B}rillouin-zone integrations},
  journal = {Phys. Rev. B},
  volume  = {13},
  pages   = {5188--5192},
  year    = {1976},
  doi     = {10.1103/PhysRevB.13.5188}
}""", "the Brillouin-zone sampling `kpts` resolves", verified=False,
         topics=("periodic",)),
    _ref("Lehmann1954", """
@article{Lehmann1954,
  author  = {Lehmann, Harry},
  title   = {{\\"U}ber {E}igenschaften von {A}usbreitungsfunktionen und
             {R}enormierungskonstanten quantisierter {F}elder},
  journal = {Nuovo Cimento},
  volume  = {11},
  pages   = {342--357},
  year    = {1954},
  doi     = {10.1007/BF02783624}
}""", "the spectral representation `get_spectral_function` evaluates",
         verified=False, topics=("periodic", "spectral")),
    _ref("Togo2018", """
@article{Togo2018,
  author  = {Togo, Atsushi and Tanaka, Isao},
  title   = {Spglib: a software library for crystal symmetry search},
  journal = {arXiv:1808.01590},
  year    = {2018},
  doi     = {10.48550/arXiv.1808.01590}
}""", "the space-group search behind the irreducible Brillouin zone",
         verified=False, topics=("periodic", "symmetry")),
    _ref("Ewald1921", """
@article{Ewald1921,
  author  = {Ewald, P. P.},
  title   = {Die {B}erechnung optischer und elektrostatischer {G}itterpotentiale},
  journal = {Ann. Phys.},
  volume  = {369},
  pages   = {253--287},
  year    = {1921},
  doi     = {10.1002/andp.19213690304}
}""", "the Ewald sum for periodic ion-ion energies", verified=False,
         topics=("periodic",)),
    _ref("Loewdin1950", """
@article{Loewdin1950,
  author  = {L{\\"o}wdin, Per-Olov},
  title   = {On the non-orthogonality problem connected with the use of atomic
             wave functions in the theory of molecules and crystals},
  journal = {J. Chem. Phys.},
  volume  = {18},
  pages   = {365--375},
  year    = {1950},
  doi     = {10.1063/1.1747632}
}""", "symmetric orthonormalization of the basis", verified=False,
         topics=("lowdin",)),
    _ref("Pulay1980", """
@article{Pulay1980,
  author  = {Pulay, P{\\'e}ter},
  title   = {Convergence acceleration of iterative sequences. {T}he case of
             {SCF} iteration},
  journal = {Chem. Phys. Lett.},
  volume  = {73},
  pages   = {393--398},
  year    = {1980},
  doi     = {10.1016/0009-2614(80)80396-4}
}""", "DIIS convergence acceleration in the SCF", verified=False,
         topics=("scf",)),
]

# --------------------------------------------------------------------------- #
# Classical optimizers.
# --------------------------------------------------------------------------- #

_OPTIMIZERS = [
    _ref("Spall1992", """
@article{Spall1992,
  author  = {Spall, James C.},
  title   = {Multivariate stochastic approximation using a simultaneous
             perturbation gradient approximation},
  journal = {IEEE Trans. Autom. Control},
  volume  = {37},
  pages   = {332--341},
  year    = {1992},
  doi     = {10.1109/9.119632}
}""", "the SPSA optimizer", verified=False, topics=("optimizer", "spsa")),
    _ref("Powell1994", """
@incollection{Powell1994,
  author    = {Powell, M. J. D.},
  title     = {A direct search optimization method that models the objective
               and constraint functions by linear interpolation},
  booktitle = {Advances in Optimization and Numerical Analysis},
  pages     = {51--67},
  publisher = {Springer},
  year      = {1994},
  doi       = {10.1007/978-94-015-8330-5_4}
}""", "the COBYLA optimizer", verified=False,
         topics=("optimizer", "cobyla")),
    _ref("Nelder1965", """
@article{Nelder1965,
  author  = {Nelder, J. A. and Mead, R.},
  title   = {A simplex method for function minimization},
  journal = {Comput. J.},
  volume  = {7},
  pages   = {308--313},
  year    = {1965},
  doi     = {10.1093/comjnl/7.4.308}
}""", "the Nelder-Mead optimizer", verified=False,
         topics=("optimizer", "nelder-mead")),
    _ref("Byrd1995", """
@article{Byrd1995,
  author  = {Byrd, Richard H. and Lu, Peihuang and Nocedal, Jorge and
             Zhu, Ciyou},
  title   = {A limited memory algorithm for bound constrained optimization},
  journal = {SIAM J. Sci. Comput.},
  volume  = {16},
  pages   = {1190--1208},
  year    = {1995},
  doi     = {10.1137/0916069}
}""", "the L-BFGS-B optimizer", verified=False,
         topics=("optimizer", "l-bfgs-b")),
    _ref("Kraft1988", """
@techreport{Kraft1988,
  author      = {Kraft, Dieter},
  title       = {A software package for sequential quadratic programming},
  institution = {DFVLR, Oberpfaffenhofen},
  number      = {DFVLR-FB 88-28},
  year        = {1988}
}""", "the SLSQP optimizer", verified=False, topics=("optimizer", "slsqp")),
    # BFGS was published independently in 1970 by all four of its initials;
    # Fletcher's is the one usually cited for the update formula itself.
    _ref("Fletcher1970", """
@article{Fletcher1970,
  author  = {Fletcher, R.},
  title   = {A new approach to variable metric algorithms},
  journal = {Comput. J.},
  volume  = {13},
  pages   = {317--322},
  year    = {1970},
  doi     = {10.1093/comjnl/13.3.317}
}""", "the BFGS optimizer", verified=False, topics=("optimizer", "bfgs")),
    _ref("Liu1989", """
@article{Liu1989,
  author  = {Liu, Dong C. and Nocedal, Jorge},
  title   = {On the limited memory {BFGS} method for large scale optimization},
  journal = {Math. Program.},
  volume  = {45},
  pages   = {503--528},
  year    = {1989},
  doi     = {10.1007/BF01589116}
}""", "the L-BFGS optimizer", verified=False,
         topics=("optimizer", "l-bfgs")),
    _ref("Polak1969", """
@article{Polak1969,
  author  = {Polak, E. and Ribi{\\`e}re, G.},
  title   = {Note sur la convergence de m{\\'e}thodes de directions
             conjugu{\\'e}es},
  journal = {Rev. Fr. Inform. Rech. Oper.},
  volume  = {3},
  pages   = {35--43},
  year    = {1969},
  doi     = {10.1051/m2an/196903R100351}
}""", "the Polak-Ribiere nonlinear conjugate gradient", verified=False,
         topics=("optimizer", "nlcg-pr")),
]

# --------------------------------------------------------------------------- #
# Codes.
# --------------------------------------------------------------------------- #

_CODES = [
    _ref("Mandacaru", """
@software{Mandacaru,
  author  = {Seixas, Leandro},
  title   = {Mandacaru: fermionic quantum simulation with variational quantum algorithms},
  url     = {https://github.com/seixas-research/mandacaru},
  year    = {2026}
}""", "this code", topics=("code",)),
    _ref("Larsen2017", """
@article{Larsen2017,
  author  = {Larsen, Ask Hjorth and Mortensen, Jens J{\\o}rgen and Blomqvist,
             Jakob and others},
  title   = {The atomic simulation environment---a {P}ython library for working
             with atoms},
  journal = {J. Phys.: Condens. Matter},
  volume  = {29},
  pages   = {273002},
  year    = {2017},
  doi     = {10.1088/1361-648X/aa680e}
}""", "the ASE calculator interface", verified=False, topics=("code",)),
    _ref("Harris2020", """
@article{Harris2020,
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt,
             St{\\'e}fan J. and others},
  title   = {Array programming with {N}um{P}y},
  journal = {Nature},
  volume  = {585},
  pages   = {357--362},
  year    = {2020},
  doi     = {10.1038/s41586-020-2649-2}
}""", "NumPy", verified=False, topics=("code",)),
    _ref("Virtanen2020", """
@article{Virtanen2020,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
             others},
  title   = {{S}ci{P}y 1.0: fundamental algorithms for scientific computing in
             {P}ython},
  journal = {Nat. Methods},
  volume  = {17},
  pages   = {261--272},
  year    = {2020},
  doi     = {10.1038/s41592-019-0686-2}
}""", "SciPy", verified=False, topics=("code",)),
    _ref("Qiskit2024", """
@misc{Qiskit2024,
  author        = {Javadi-Abhari, Ali and Treinish, Matthew and
                   Krsulich, Kevin and others},
  title         = {Quantum computing with {Q}iskit},
  year          = {2024},
  eprint        = {2405.08810},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2405.08810}
}""", "circuit construction, transpilation and the Estimator primitive",
         verified=False, topics=("code", "qiskit")),
]

#: Every reference, keyed by its BibTeX key.
REFERENCES = {r.key: r for r in
              (_ALGORITHMS + _POOLS + _MAPPINGS + _BASES + _OPTIMIZERS
               + _CODES)}

#: Entries whose record was not read from a local source; a manuscript should
#: check these against the journal.
UNVERIFIED = tuple(k for k, r in REFERENCES.items() if not r.verified)


def resolve(keys) -> list:
    """The :class:`Reference` objects for ``keys``, in database order.

    Order is the database's, not the caller's, so the same run always produces
    the same file whatever order the keys were collected in.  An unknown key
    raises rather than being dropped -- a silently missing citation is worse
    than a loud one.
    """
    wanted = set(keys)
    unknown = sorted(wanted - set(REFERENCES))
    if unknown:
        raise KeyError(f"no bibliography entry for {unknown}; known keys are "
                       f"{sorted(REFERENCES)}")
    return [r for key, r in REFERENCES.items() if key in wanted]


def bibtex(keys, header: str | None = None) -> str:
    """Render ``keys`` as a BibTeX file.

    Each entry is preceded by a comment saying what in the run pulled it in,
    and by `` % unverified`` when the record was not read from a local source.
    """
    lines = []
    if header:
        lines += [f"% {line}" if line else "%" for line in header.splitlines()]
        lines.append("")
    for reference in resolve(keys):
        lines.append(f"% {reference.note}"
                     + ("" if reference.verified else "  [unverified record]"))
        lines.append(reference.entry)
        lines.append("")
    return "\n".join(lines)
