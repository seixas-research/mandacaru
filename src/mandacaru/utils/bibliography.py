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
    _ref("Roothaan1951", """
@article{Roothaan1951,
  author  = {Roothaan, C. C. J.},
  title   = {New developments in molecular orbital theory},
  journal = {Rev. Mod. Phys.},
  volume  = {23},
  pages   = {69--89},
  year    = {1951},
  doi     = {10.1103/RevModPhys.23.69}
}""", "restricted Hartree-Fock molecular orbitals", verified=False,
         topics=("rhf", "scf")),
    _ref("PopleNesbet1954", """
@article{PopleNesbet1954,
  author  = {Pople, J. A. and Nesbet, R. K.},
  title   = {Self-consistent orbitals for radicals},
  journal = {J. Chem. Phys.},
  volume  = {22},
  pages   = {571--572},
  year    = {1954},
  doi     = {10.1063/1.1740120}
}""", "unrestricted Hartree-Fock for open shells", verified=False,
         topics=("uhf", "scf")),
    _ref("JimenezHoyos2011", """
@article{JimenezHoyos2011,
  author  = {Jim{\\'e}nez-Hoyos, Carlos A. and Henderson, Thomas M. and
             Scuseria, Gustavo E.},
  title   = {Generalized {Hartree-Fock} description of molecular dissociation},
  journal = {J. Chem. Theory Comput.},
  volume  = {7},
  pages   = {2667--2674},
  year    = {2011},
  doi     = {10.1021/ct200345a}
}""", "generalized (spinor) Hartree-Fock", verified=True,
         topics=("ghf", "scf")),
    _ref("HohenbergKohn1964", """
@article{HohenbergKohn1964,
  author  = {Hohenberg, P. and Kohn, W.},
  title   = {Inhomogeneous Electron Gas},
  journal = {Phys. Rev.},
  volume  = {136},
  pages   = {B864--B871},
  year    = {1964},
  doi     = {10.1103/PhysRev.136.B864}
}""", "density functional theory", verified=False,
         topics=("dft", "scf")),
    _ref("KohnSham1965", """
@article{KohnSham1965,
  author  = {Kohn, W. and Sham, L. J.},
  title   = {Self-Consistent Equations Including Exchange and Correlation
             Effects},
  journal = {Phys. Rev.},
  volume  = {140},
  pages   = {A1133--A1138},
  year    = {1965},
  doi     = {10.1103/PhysRev.140.A1133}
}""", "the Kohn-Sham self-consistent equations", verified=False,
         topics=("dft", "scf")),
    _ref("vonBarthHedin1972", """
@article{vonBarthHedin1972,
  author  = {von Barth, U. and Hedin, L.},
  title   = {A local exchange-correlation potential for the spin polarized
             case: {I}},
  journal = {J. Phys. C: Solid State Phys.},
  volume  = {5},
  pages   = {1629--1642},
  year    = {1972},
  doi     = {10.1088/0022-3719/5/13/012}
}""", "the spin interpolation of the local spin-density correlation",
         verified=False, topics=("xc", "spin")),
    _ref("PerdewZunger1981", """
@article{PerdewZunger1981,
  author  = {Perdew, J. P. and Zunger, Alex},
  title   = {Self-interaction correction to density-functional approximations
             for many-electron systems},
  journal = {Phys. Rev. B},
  volume  = {23},
  pages   = {5048--5079},
  year    = {1981},
  doi     = {10.1103/PhysRevB.23.5048}
}""", "the LDA correlation (xc='lda')", verified=False,
         topics=("xc", "lda")),
    _ref("Furness2020", """
@article{Furness2020,
  author  = {Furness, James W. and Kaplan, Aaron D. and Ning, Jinliang and
             Perdew, John P. and Sun, Jianwei},
  title   = {Accurate and Numerically Efficient {r$^2$SCAN} Meta-Generalized
             Gradient Approximation},
  journal = {J. Phys. Chem. Lett.},
  volume  = {11},
  pages   = {8208--8215},
  year    = {2020},
  doi     = {10.1021/acs.jpclett.0c02405}
}""", "the r2SCAN meta-GGA (xc='r2scan')", verified=False,
         topics=("xc", "meta-gga")),
    _ref("Vydrov2010", """
@article{Vydrov2010,
  author  = {Vydrov, Oleg A. and Van Voorhis, Troy},
  title   = {Nonlocal van der {W}aals density functional: The simpler the
             better},
  journal = {J. Chem. Phys.},
  volume  = {133},
  pages   = {244103},
  year    = {2010},
  doi     = {10.1063/1.3521275}
}""", "VV10 nonlocal correlation (xc='r2scan-rvv10')", verified=False,
         topics=("xc", "dispersion")),
    _ref("Sabatini2013", """
@article{Sabatini2013,
  author  = {Sabatini, Riccardo and Gorni, Tommaso and de Gironcoli,
             Stefano},
  title   = {Nonlocal van der {W}aals density functional made simple and
             efficient},
  journal = {Phys. Rev. B},
  volume  = {87},
  pages   = {041108},
  year    = {2013},
  doi     = {10.1103/PhysRevB.87.041108}
}""", "the rVV10 kernel (xc='r2scan-rvv10')", verified=False,
         topics=("xc", "dispersion")),
    _ref("RomanPerez2009", """
@article{RomanPerez2009,
  author  = {Rom{\'a}n-P{\'e}rez, Guillermo and Soler, Jos{\'e} M.},
  title   = {Efficient Implementation of a van der {W}aals Density Functional:
             Application to Double-Wall Carbon Nanotubes},
  journal = {Phys. Rev. Lett.},
  volume  = {103},
  pages   = {096102},
  year    = {2009},
  doi     = {10.1103/PhysRevLett.103.096102}
}""", "the q interpolation of the nonlocal kernel (xc='r2scan-rvv10')",
         verified=False, topics=("xc", "dispersion")),
    _ref("KingSmith1993", """
@article{KingSmith1993,
  author  = {King-Smith, R. D. and Vanderbilt, David},
  title   = {Theory of polarization of crystalline solids},
  journal = {Phys. Rev. B},
  volume  = {47},
  pages   = {1651--1654},
  year    = {1993},
  doi     = {10.1103/PhysRevB.47.1651}
}""", "the Berry-phase polarization (get_polarization)",
         verified=False, topics=("polarization", "berry phase")),
    _ref("Resta1994", """
@article{Resta1994,
  author  = {Resta, Raffaele},
  title   = {Macroscopic polarization in crystalline dielectrics: the
             geometric phase approach},
  journal = {Rev. Mod. Phys.},
  volume  = {66},
  pages   = {899--915},
  year    = {1994},
  doi     = {10.1103/RevModPhys.66.899}
}""", "the polarization as a geometric phase (get_polarization)",
         verified=False, topics=("polarization", "berry phase")),
    _ref("Vanderbilt2000", """
@article{Vanderbilt2000,
  author  = {Vanderbilt, David},
  title   = {Berry-phase theory of proper piezoelectric response},
  journal = {J. Phys. Chem. Solids},
  volume  = {61},
  pages   = {147--151},
  year    = {2000},
  doi     = {10.1016/S0022-3697(99)00273-5}
}""", "the proper piezoelectric tensor (piezoelectric_tensor)",
         verified=False, topics=("polarization", "piezoelectric")),
    _ref("Souza2002", """
@article{Souza2002,
  author  = {Souza, Ivo and {\\'I}{\\~n}iguez, Jorge and Vanderbilt, David},
  title   = {First-Principles Approach to Insulators in Finite Electric
             Fields},
  journal = {Phys. Rev. Lett.},
  volume  = {89},
  pages   = {117602},
  year    = {2002},
  doi     = {10.1103/PhysRevLett.89.117602}
}""", "the Berry-phase finite electric field (electric_field on a crystal)",
         verified=False, topics=("polarization", "electric field")),
    _ref("Umari2002", """
@article{Umari2002,
  author  = {Umari, Paolo and Pasquarello, Alfredo},
  title   = {Ab initio Molecular Dynamics in a Finite Homogeneous Electric
             Field},
  journal = {Phys. Rev. Lett.},
  volume  = {89},
  pages   = {157602},
  year    = {2002},
  doi     = {10.1103/PhysRevLett.89.157602}
}""", "the Berry-phase finite electric field (electric_field on a crystal)",
         verified=False, topics=("polarization", "electric field")),
    _ref("Marzari1997", """
@article{Marzari1997,
  author  = {Marzari, Nicola and Vanderbilt, David},
  title   = {Maximally localized generalized {W}annier functions for composite
             energy bands},
  journal = {Phys. Rev. B},
  volume  = {56},
  pages   = {12847--12865},
  year    = {1997},
  doi     = {10.1103/PhysRevB.56.12847}
}""", "maximally localized Wannier functions (wannier)",
         verified=False, topics=("wannier",)),
    _ref("Marzari2012", """
@article{Marzari2012,
  author  = {Marzari, Nicola and Mostofi, Arash A. and Yates, Jonathan R. and
             Souza, Ivo and Vanderbilt, David},
  title   = {Maximally localized {W}annier functions: {T}heory and
             applications},
  journal = {Rev. Mod. Phys.},
  volume  = {84},
  pages   = {1419--1475},
  year    = {2012},
  doi     = {10.1103/RevModPhys.84.1419}
}""", "the review of maximally localized Wannier functions (wannier)",
         verified=False, topics=("wannier",)),
    _ref("Souza2001", """
@article{Souza2001,
  author  = {Souza, Ivo and Marzari, Nicola and Vanderbilt, David},
  title   = {Maximally localized {W}annier functions for entangled energy
             bands},
  journal = {Phys. Rev. B},
  volume  = {65},
  pages   = {035109},
  year    = {2001},
  doi     = {10.1103/PhysRevB.65.035109}
}""", "the disentanglement of entangled bands (wannier with windows)",
         verified=False, topics=("wannier",)),
    _ref("Aryasetiawan2004", """
@article{Aryasetiawan2004,
  author  = {Aryasetiawan, F. and Imada, M. and Georges, A. and Kotliar, G.
             and Biermann, S. and Lichtenstein, A. I.},
  title   = {Frequency-dependent local interactions and low-energy effective
             models from electronic structure calculations},
  journal = {Phys. Rev. B},
  volume  = {70},
  pages   = {195104},
  year    = {2004},
  doi     = {10.1103/PhysRevB.70.195104}
}""", "the constrained random-phase approximation (downfold screening)",
         verified=False, topics=("wannier", "screening")),
    _ref("Sasioglu2011", """
@article{Sasioglu2011,
  author  = {{\\c{S}}a{\\c{s}}{\\i}o{\\u{g}}lu, E. and Friedrich, C. and
             Bl{\\"u}gel, S.},
  title   = {Effective {C}oulomb interaction in transition metals from
             constrained random-phase approximation},
  journal = {Phys. Rev. B},
  volume  = {83},
  pages   = {121101},
  year    = {2011},
  doi     = {10.1103/PhysRevB.83.121101}
}""", "the projector-weighted constrained RPA for entangled bands "
         "(downfold screening)",
         verified=False, topics=("wannier", "screening")),
    _ref("Spencer2008", """
@article{Spencer2008,
  author  = {Spencer, James and Alavi, Ali},
  title   = {Efficient calculation of the exact exchange energy in periodic
             systems using a truncated {C}oulomb potential},
  journal = {Phys. Rev. B},
  volume  = {77},
  pages   = {193110},
  year    = {2008},
  doi     = {10.1103/PhysRevB.77.193110}
}""", "the truncated Coulomb interaction of the downfolded fragment",
         verified=False, topics=("wannier",)),
    _ref("Sayfutyarova2017", """
@article{Sayfutyarova2017,
  author  = {Sayfutyarova, Elvira R. and Sun, Qiming and Chan, Garnet
             Kin-Lic and Knizia, Gerald},
  title   = {Automated Construction of Molecular Active Spaces from Atomic
             Valence Orbitals},
  journal = {J. Chem. Theory Comput.},
  volume  = {13},
  pages   = {4063--4078},
  year    = {2017},
  doi     = {10.1021/acs.jctc.7b00128}
}""", "the projection onto target atomic orbitals (projectability "
         "windows, wannier windows='auto')",
         verified=False, topics=("wannier", "selection")),
    _ref("Qiao2023", """
@article{Qiao2023,
  author  = {Qiao, Junfeng and Pizzi, Giovanni and Marzari, Nicola},
  title   = {Projectability disentanglement for accurate and automated
             electronic-structure {H}amiltonians},
  journal = {npj Comput. Mater.},
  volume  = {9},
  pages   = {208},
  year    = {2023},
  doi     = {10.1038/s41524-023-01146-w}
}""", "states chosen by projectability for the disentanglement "
         "(wannier windows='auto' or 'rpa')",
         verified=False, topics=("wannier", "selection")),
    _ref("Damle2017", """
@article{Damle2017,
  author  = {Damle, Anil and Lin, Lin and Ying, Lexing},
  title   = {{SCDM-k}: Localized orbitals for solids via selected columns of
             the density matrix},
  journal = {J. Comput. Phys.},
  volume  = {334},
  pages   = {1--15},
  year    = {2017},
  doi     = {10.1016/j.jcp.2016.12.053}
}""", "selected columns of the density matrix for crystals "
         "(wannier windows='scdm')",
         verified=False, topics=("wannier", "selection")),
    _ref("Damle2018", """
@article{Damle2018,
  author  = {Damle, Anil and Lin, Lin},
  title   = {Disentanglement via Entanglement: A Unified Method for {W}annier
             Localization},
  journal = {Multiscale Model. Simul.},
  volume  = {16},
  pages   = {1392--1410},
  year    = {2018},
  doi     = {10.1137/18M1167164}
}""", "the erfc-weighted quasi-density matrix for entangled bands "
         "(wannier windows='scdm')",
         verified=False, topics=("wannier", "selection")),
    _ref("Vitale2020", """
@article{Vitale2020,
  author  = {Vitale, Valerio and Pizzi, Giovanni and Marrazzo, Antimo and
             Yates, Jonathan R. and Marzari, Nicola and Mostofi, Arash A.},
  title   = {Automated high-throughput {W}annierisation},
  journal = {npj Comput. Mater.},
  volume  = {6},
  pages   = {66},
  year    = {2020},
  doi     = {10.1038/s41524-020-0312-y}
}""", "the SCDM erfc parameters fitted to the projectability "
         "(wannier windows='scdm' without mu and sigma)",
         verified=False, topics=("wannier", "selection")),
    _ref("Scuseria2008", """
@article{Scuseria2008,
  author  = {Scuseria, Gustavo E. and Henderson, Thomas M. and Sorensen,
             Danny C.},
  title   = {The ground state correlation energy of the random phase
             approximation from a ring coupled cluster doubles approach},
  journal = {J. Chem. Phys.},
  volume  = {129},
  pages   = {231101},
  year    = {2008},
  doi     = {10.1063/1.3043729}
}""", "direct RPA as ring coupled-cluster doubles (the amplitudes of "
         "natural_orbitals(method='rpa'))",
         verified=False, topics=("rpa", "selection")),
    _ref("Furche2008", """
@article{Furche2008,
  author  = {Furche, Filipp},
  title   = {Developing the random phase approximation into a practical
             post-{K}ohn-{S}ham correlation model},
  journal = {J. Chem. Phys.},
  volume  = {129},
  pages   = {114105},
  year    = {2008},
  doi     = {10.1063/1.2977789}
}""", "the RPA correlation energy from the excitation energies and the "
         "imaginary-frequency polarizability (natural_orbitals checks)",
         verified=False, topics=("rpa",)),
    _ref("Ning2022", """
@article{Ning2022,
  author  = {Ning, Jinliang and Kothakonda, Manish and Furness, James W. and
             Kaplan, Aaron D. and Ehlert, Sebastian and Brandenburg, Jan Gerit
             and Perdew, John P. and Sun, Jianwei},
  title   = {Workhorse minimally empirical dispersion-corrected density
             functional with tests for weakly bound systems: {r$^2$SCAN+rVV10}},
  journal = {Phys. Rev. B},
  volume  = {106},
  pages   = {075422},
  year    = {2022},
  doi     = {10.1103/PhysRevB.106.075422}
}""", "r2SCAN+rVV10, b = 11.95 (xc='r2scan-rvv10')", verified=False,
         topics=("xc", "dispersion")),
    _ref("Heyd2003", """
@article{Heyd2003,
  author  = {Heyd, Jochen and Scuseria, Gustavo E. and Ernzerhof, Matthias},
  title   = {Hybrid functionals based on a screened {Coulomb} potential},
  journal = {J. Chem. Phys.},
  volume  = {118},
  pages   = {8207--8215},
  year    = {2003},
  note    = {Erratum: J. Chem. Phys. 124, 219906 (2006)},
  doi     = {10.1063/1.1564060}
}""", "the screened hybrid and its short-range exchange hole (xc='hse06')",
         verified=False, topics=("xc", "hybrid")),
    _ref("Krukau2006", """
@article{Krukau2006,
  author  = {Krukau, Aliaksandr V. and Vydrov, Oleg A. and Izmaylov, Artur F.
             and Scuseria, Gustavo E.},
  title   = {Influence of the exchange screening parameter on the performance
             of screened hybrid functionals},
  journal = {J. Chem. Phys.},
  volume  = {125},
  pages   = {224106},
  year    = {2006},
  doi     = {10.1063/1.2404663}
}""", "the HSE06 screening parameter omega = 0.11 (xc='hse06')",
         verified=False, topics=("xc", "hybrid")),
    _ref("Paier2005", """
@article{Paier2005,
  author  = {Paier, Joachim and Hirschl, Robin and Marsman, Martijn
             and Kresse, Georg},
  title   = {The {Perdew-Burke-Ernzerhof} exchange-correlation functional
             applied to the {G2-1} test set using a plane-wave basis set},
  journal = {J. Chem. Phys.},
  volume  = {122},
  pages   = {234102},
  year    = {2005},
  doi     = {10.1063/1.1926272}
}""", "exact exchange in the projector augmented-wave method: the "
         "one-center correction of a PAW-LCAO hybrid",
         verified=False, topics=("xc", "hybrid", "paw")),
    _ref("Caldeweyher2019", """
@article{Caldeweyher2019,
  author  = {Caldeweyher, Eike and Ehlert, Sebastian and Hansen, Andreas and
             Neugebauer, Hagen and Spicher, Sebastian and Bannwarth, Christoph
             and Grimme, Stefan},
  title   = {A generally applicable atomic-charge dependent London dispersion
             correction},
  journal = {J. Chem. Phys.},
  volume  = {150},
  pages   = {154122},
  year    = {2019},
  doi     = {10.1063/1.5090222}
}""", "the D4 dispersion correction (dispersion='d4')", verified=False,
         topics=("dispersion",)),
    _ref("Mermin1965", """
@article{Mermin1965,
  author  = {Mermin, N. David},
  title   = {Thermal Properties of the Inhomogeneous Electron Gas},
  journal = {Phys. Rev.},
  volume  = {137},
  pages   = {A1441--A1443},
  year    = {1965},
  doi     = {10.1103/PhysRev.137.A1441}
}""", "finite-temperature (smeared) Kohn-Sham occupations", verified=False,
         topics=("dft", "smearing")),
    _ref("MethfesselPaxton1989", """
@article{MethfesselPaxton1989,
  author  = {Methfessel, M. and Paxton, A. T.},
  title   = {High-precision sampling for {Brillouin}-zone integration in
             metals},
  journal = {Phys. Rev. B},
  volume  = {40},
  pages   = {3616--3621},
  year    = {1989},
  doi     = {10.1103/PhysRevB.40.3616}
}""", "Methfessel-Paxton smearing (smearing='methfessel-paxton')",
         verified=False, topics=("dft", "smearing")),
    _ref("Kerker1981", """
@article{Kerker1981,
  author  = {Kerker, G. P.},
  title   = {Efficient iteration scheme for self-consistent pseudopotential
             calculations},
  journal = {Phys. Rev. B},
  volume  = {23},
  pages   = {3082--3084},
  year    = {1981},
  doi     = {10.1103/PhysRevB.23.3082}
}""", "the Kerker preconditioner of the periodic density mixing",
         verified=False, topics=("dft", "mixing")),
    _ref("Wecker2015", """
@article{Wecker2015,
  author  = {Wecker, Dave and Hastings, Matthew B. and Troyer, Matthias},
  title   = {Progress towards practical quantum variational algorithms},
  journal = {Phys. Rev. A},
  volume  = {92},
  pages   = {042303},
  year    = {2015},
  doi     = {10.1103/PhysRevA.92.042303}
}""", "Hamiltonian variational ansatz", verified=False,
         topics=("hva", "vqe")),
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
    _ref("Metropolis1953", """
@article{Metropolis1953,
  author  = {Metropolis, Nicholas and Rosenbluth, Arianna W. and
             Rosenbluth, Marshall N. and Teller, Augusta H. and Teller, Edward},
  title   = {Equation of state calculations by fast computing machines},
  journal = {J. Chem. Phys.},
  volume  = {21},
  pages   = {1087},
  year    = {1953},
  doi     = {10.1063/1.1699114}
}""", "the Metropolis acceptance of the MCAS-VQE architecture chain",
         verified=False, topics=("mcas",)),
    _ref("Hastings1970", """
@article{Hastings1970,
  author  = {Hastings, W. K.},
  title   = {Monte {C}arlo sampling methods using {M}arkov chains and their
             applications},
  journal = {Biometrika},
  volume  = {57},
  pages   = {97},
  year    = {1970},
  doi     = {10.1093/biomet/57.1.97}
}""", "the proposal-ratio correction of the MCAS-VQE architecture chain",
         verified=False, topics=("mcas",)),
    _ref("Kirkpatrick1983", """
@article{Kirkpatrick1983,
  author  = {Kirkpatrick, S. and Gelatt, C. D. and Vecchi, M. P.},
  title   = {Optimization by simulated annealing},
  journal = {Science},
  volume  = {220},
  pages   = {671},
  year    = {1983},
  doi     = {10.1126/science.220.4598.671}
}""", "the annealed architecture temperature of MCAS-VQE", verified=False,
         topics=("mcas",)),
    _ref("Gilmer2017", """
@inproceedings{Gilmer2017,
  author    = {Gilmer, Justin and Schoenholz, Samuel S. and Riley, Patrick F.
               and Vinyals, Oriol and Dahl, George E.},
  title     = {Neural message passing for quantum chemistry},
  booktitle = {Proceedings of the 34th International Conference on Machine
               Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {1263--1272},
  year      = {2017}
}""", "the Hamiltonian message-passing network of VALQA's proposal",
         verified=False, topics=("mcas",)),
    _ref("Rasmussen2006", """
@book{Rasmussen2006,
  author    = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  title     = {Gaussian Processes for Machine Learning},
  publisher = {MIT Press},
  year      = {2006}
}""", "the Gaussian-process head of VALQA's proposal", verified=False,
         topics=("mcas",)),
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
}""", "the PAW-LCAO / UPAW-LCAO basis", verified=False, topics=("paw-lcao", "upaw-lcao")),
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
}""", "unitary PAW-LCAO (basis='UPAW-LCAO')", verified=False, topics=("upaw-lcao",)),
    _ref("Hamann2013", """
@article{Hamann2013,
  author  = {Hamann, D. R.},
  title   = {Optimized norm-conserving {V}anderbilt pseudopotentials},
  journal = {Phys. Rev. B},
  volume  = {88},
  pages   = {085117},
  year    = {2013},
  doi     = {10.1103/PhysRevB.88.085117}
}""", "the optimized smooth partial waves of PAW-LCAO and UPAW-LCAO",
         verified=False, topics=("paw-lcao", "upaw-lcao")),
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
    _ref("PBE1996", """
@article{PBE1996,
  author  = {Perdew, John P. and Burke, Kieron and Ernzerhof, Matthias},
  title   = {Generalized Gradient Approximation Made Simple},
  journal = {Phys. Rev. Lett.},
  volume  = {77},
  pages   = {3865--3868},
  year    = {1996},
  doi     = {10.1103/PhysRevLett.77.3865}
}""", "the GGA reference atom (xc='pbe')", verified=False,
         topics=("xc", "gga", "pseudopotential")),
    _ref("PerdewWang1992", """
@article{PerdewWang1992,
  author  = {Perdew, John P. and Wang, Yue},
  title   = {Accurate and simple analytic representation of the
             electron-gas correlation energy},
  journal = {Phys. Rev. B},
  volume  = {45},
  pages   = {13244--13249},
  year    = {1992},
  doi     = {10.1103/PhysRevB.45.13244}
}""", "the uniform-gas correlation PBE is built on", verified=False,
         topics=("xc", "gga")),
    _ref("KoellingHarmon1977", """
@article{KoellingHarmon1977,
  author  = {Koelling, D. D. and Harmon, B. N.},
  title   = {A technique for relativistic spin-polarised calculations},
  journal = {J. Phys. C},
  volume  = {10},
  pages   = {3107--3114},
  year    = {1977},
  doi     = {10.1088/0022-3719/10/16/019}
}""", "the scalar-relativistic radial equation (relativity='scalar')",
         verified=False, topics=("relativity", "pseudopotential")),
    _ref("MacDonaldVosko1979", """
@article{MacDonaldVosko1979,
  author  = {MacDonald, A. H. and Vosko, S. H.},
  title   = {A relativistic density functional formalism},
  journal = {J. Phys. C},
  volume  = {12},
  pages   = {2977--2990},
  year    = {1979},
  doi     = {10.1088/0022-3719/12/15/007}
}""", "the relativistic correction to LDA exchange of a relativistic "
         "reference atom", verified=True,
         topics=("relativity", "xc", "pseudopotential")),
    _ref("Louie1982", """
@article{Louie1982,
  author  = {Louie, Steven G. and Froyen, Sverre and Cohen, Marvin L.},
  title   = {Nonlinear ionic pseudopotentials in spin-density-functional
             calculations},
  journal = {Phys. Rev. B},
  volume  = {26},
  pages   = {1738--1742},
  year    = {1982},
  doi     = {10.1103/PhysRevB.26.1738}
}""", "the nonlinear core correction and its partial core density",
         verified=False, topics=("pseudopotential", "nlcc")),
    _ref("Kleinman1980", """
@article{Kleinman1980,
  author  = {Kleinman, Leonard},
  title   = {Relativistic norm-conserving pseudopotential},
  journal = {Phys. Rev. B},
  volume  = {21},
  pages   = {2630--2631},
  year    = {1980},
  doi     = {10.1103/PhysRevB.21.2630}
}""", "splitting a j-resolved pseudopotential into an average and an L.S term",
         verified=False, topics=("relativity", "spin-orbit")),
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
    _ref("SetyawanCurtarolo2010", """
@article{SetyawanCurtarolo2010,
  author  = {Setyawan, Wahyu and Curtarolo, Stefano},
  title   = {High-throughput electronic band structure calculations:
             {C}hallenges and tools},
  journal = {Comput. Mater. Sci.},
  volume  = {49},
  pages   = {299--312},
  year    = {2010},
  doi     = {10.1016/j.commatsci.2010.05.010}
}""", "the default high-symmetry band path of each Bravais lattice",
         verified=False, topics=("periodic",)),
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
    _ref("Moller1934", """
@article{Moller1934,
  author  = {M{\\o}ller, Chr. and Plesset, Milton S.},
  title   = {Note on an Approximation Treatment for Many-Electron Systems},
  journal = {Phys. Rev.},
  volume  = {46},
  pages   = {618--622},
  year    = {1934},
  doi     = {10.1103/PhysRev.46.618}
}""", "second-order perturbation theory (the MP2 active-space selector)",
         verified=False, topics=("mp2", "active-space")),
    _ref("Loewdin1955", """
@article{Loewdin1955,
  author  = {L{\\"o}wdin, Per-Olov},
  title   = {Quantum Theory of Many-Particle Systems. I. Physical
             Interpretations by Means of Density Matrices, Natural Spin-Orbitals,
             and Convergence Problems in the Method of Configurational
             Interaction},
  journal = {Phys. Rev.},
  volume  = {97},
  pages   = {1474--1489},
  year    = {1955},
  doi     = {10.1103/PhysRev.97.1474}
}""", "natural orbitals and their occupation numbers", verified=False,
         topics=("natural-orbitals", "active-space")),
    _ref("Sosa1989", """
@article{Sosa1989,
  author  = {Sosa, Carlos and Geertsen, Jan and Trucks, Gary W. and
             Bartlett, Rodney J. and Franz, Judy A.},
  title   = {Selection of the reduced virtual space for correlated
             calculations. An application to the energy and dipole moment of
             H2O},
  journal = {Chem. Phys. Lett.},
  volume  = {159},
  pages   = {148--154},
  year    = {1989},
  doi     = {10.1016/0009-2614(89)87399-3}
}""", "the frozen natural orbital truncation of the virtual space",
         verified=False, topics=("fno", "active-space", "mp2")),
    _ref("TaubeBartlett2005", """
@article{TaubeBartlett2005,
  author  = {Taube, Andrew G. and Bartlett, Rodney J.},
  title   = {Frozen natural orbitals: Systematic basis set truncation for
             coupled-cluster theory},
  journal = {Collect. Czech. Chem. Commun.},
  volume  = {70},
  pages   = {837--850},
  year    = {2005},
  doi     = {10.1135/cccc20050837}
}""", "frozen natural orbitals as a systematic basis truncation",
         verified=False, topics=("fno", "active-space")),
    _ref("Pinski2015", """
@article{Pinski2015,
  author  = {Pinski, Peter and Riplinger, Christoph and Valeev, Edward F. and
             Neese, Frank},
  title   = {Sparse maps---{A} systematic infrastructure for reduced-scaling
             electronic structure methods. {I}. {A}n efficient and simple
             linear scaling local {MP2} method that uses an intermediate basis
             of pair natural orbitals},
  journal = {J. Chem. Phys.},
  volume  = {143},
  pages   = {034108},
  year    = {2015},
  doi     = {10.1063/1.4926879}
}""", "DLPNO-MP2, the local pair natural orbital MP2 active-space selector",
         verified=False, topics=("mp2", "active-space", "local-correlation")),
    _ref("FosterBoys1960", """
@article{FosterBoys1960,
  author  = {Foster, J. M. and Boys, S. F.},
  title   = {Canonical Configurational Interaction Procedure},
  journal = {Rev. Mod. Phys.},
  volume  = {32},
  pages   = {300--302},
  year    = {1960},
  doi     = {10.1103/RevModPhys.32.300}
}""", "Foster-Boys localized orbitals (DLPNO-MP2's occupied space)",
         verified=False, topics=("localization", "local-correlation")),
    _ref("Pulay1983", """
@article{Pulay1983,
  author  = {Pulay, Peter},
  title   = {Localizability of dynamic electron correlation},
  journal = {Chem. Phys. Lett.},
  volume  = {100},
  pages   = {151--154},
  year    = {1983},
  doi     = {10.1016/0009-2614(83)80703-9}
}""", "projected atomic orbitals and local correlation domains",
         verified=False, topics=("local-correlation",)),
    _ref("Rozzi2006", """
@article{Rozzi2006,
  author  = {Rozzi, Carlo A. and Varsano, Daniele and Marini, Andrea and
             Gross, Eberhard K. U. and Rubio, Angel},
  title   = {Exact {C}oulomb cutoff technique for supercell calculations},
  journal = {Phys. Rev. B},
  volume  = {73},
  pages   = {205119},
  year    = {2006},
  doi     = {10.1103/PhysRevB.73.205119}
}""", "the two-dimensionally truncated Coulomb kernel for slabs",
         verified=False, topics=("slab", "periodic", "coulomb-cutoff")),
    _ref("Bravyi2017", """
@misc{Bravyi2017,
  author        = {Bravyi, Sergey and Gambetta, Jay M. and Mezzacapo, Antonio
                   and Temme, Kristan},
  title         = {Tapering off qubits to simulate fermionic {H}amiltonians},
  year          = {2017},
  eprint        = {1701.08213},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.1701.08213}
}""", "Z2 symmetry tapering (mandacaru.core.tapering)", verified=False,
         topics=("tapering", "mapping", "symmetry")),
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
}""", "the limited-memory BFGS implementation behind L-BFGS",
         verified=False, topics=("optimizer", "l-bfgs")),
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
         topics=("optimizer", "cg")),
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
