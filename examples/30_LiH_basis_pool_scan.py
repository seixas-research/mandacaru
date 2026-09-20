"""Reproduce the documentation's LiH basis and operator-pool comparison.

Run from the repository root after installing Mandacaru::

    python examples/30_LiH_basis_pool_scan.py
    python examples/30_LiH_basis_pool_scan.py --plot-only

The CSV contains calculated total energies, including nuclear and frozen-core
contributions. The PNG is a numerical demonstration, not a converged prediction
of LiH spectroscopy. No hardware jobs are submitted. Each geometry and basis
gets a new Hamiltonian; all pools reuse that identical Hamiltonian.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import tempfile

import numpy as np
from mandacaru.optimizers import Optimizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/source/_static/lih"
BASES = ("STO-3G", "3-21G")
POOLS = ("fermionic", "qubit", "qeb", "ceo")
FIELDS = (
    "basis", "pool", "distance_A", "energy_eV", "reference_eV",
    "num_operators", "converged", "inner_failures", "n_qubits", "particle_sector_weight",
)


def calculate(output: Path, spacing: float, half_width: float,
              points: int, max_iterations: int) -> None:
    """Calculate every point and retain its convergence information."""
    import ase
    import mandacaru
    import scipy
    from ase import Atoms
    from mandacaru.algorithms import Mandacaru
    from mandacaru.core.sector import ParticleSector
    from mandacaru.integrals import Grid
    from mandacaru.units import BOHR_TO_ANGSTROM

    grid = Grid(center=[0.0, 0.0, 0.0], box_size=half_width, h=spacing)
    actual_spacing = float(grid.dz * BOHR_TO_ANGSTROM)
    distances = 1.12 + 2 * actual_spacing * np.arange(points)
    if distances[-1] / 2 >= half_width:
        raise ValueError("The scan extends outside the integration box.")

    metadata = {
        "description": "Calculated LiH total energies; numerical teaching example",
        "mandacaru_version": mandacaru.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "ase_version": ase.__version__,
        "basis_sets": list(BASES), "pools": list(POOLS),
        "mapping": "jordan_wigner", "frozen_core": True,
        "charge": 0, "active_electrons": 2,
        "box_half_width_A": half_width,
        "requested_spacing_A": spacing, "actual_spacing_A": actual_spacing,
        "distances_A": distances.tolist(),
        "gradient_tolerance_Ha": 1e-5, "max_iterations": max_iterations,
        "optimizer": "L-BFGS-B", "optimizer_maxiter": 1000,
        "execute_circuits": False, "profile": False,
        "sparse": True, "sector": "fixed particle numbers except for the qubit pool",
        "energy_unit": "eV",
        "limitations": [
            "Native generated basis functions, not published exponent tables",
            "Finite grid and box; compact 3-21G functions trigger resolution warnings",
            "A finite growth budget may leave some pool searches unconverged",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with (output / "energies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        with tempfile.TemporaryDirectory(prefix="mandacaru-lih-") as directory:
            for basis in BASES:
                for index, distance in enumerate(distances):
                    atoms = Atoms(
                        "LiH",
                        positions=[[0, 0, -distance / 2], [0, 0, distance / 2]],
                        pbc=False,
                    )
                    cache = str(Path(directory) / f"{basis}-{index}.json")
                    for pool in POOLS:
                        settings = dict(
                            method="adapt-vqe", pool=pool,
                            optimizer=Optimizer(method="L-BFGS-B",
                                                maxiter=1000,
                                                tol=1e-12),
                            gradient="analytic", gradient_tolerance=1e-5,
                            max_iterations=max_iterations,
                            sparse=True, sector=(pool != "qubit"),
                            execute_circuits=False, profile=False,
                        )
                        if pool == POOLS[0]:
                            calc = Mandacaru(
                                **settings, basis=basis, grid=grid, h=spacing,
                                frozen_core=True, mapping="jordan_wigner",
                                save_hamiltonian=cache, hamiltonian_format="json",
                            )
                            atoms.calc = calc
                            atoms.get_potential_energy()
                            result = calc.result
                        else:
                            calc = Mandacaru(**settings, load_hamiltonian=cache)
                            result = calc.run()
                        if not np.isfinite(result.optimal_energy):
                            raise RuntimeError(f"Non-finite energy: {basis}, {pool}, {distance}")
                        state = calc.solver.ansatz.state(result.optimal_parameters)
                        if calc.solver.ansatz.sector is None:
                            sector = ParticleSector(calc.solver.n_qubits, (1, 1))
                            sector_weight = float(np.sum(np.abs(state[sector.indices]) ** 2))
                        else:
                            sector_weight = float(np.vdot(state, state).real)
                        writer.writerow(dict(
                            basis=basis, pool=pool, distance_A=float(distance),
                            energy_eV=result.optimal_energy,
                            reference_eV=result.reference_energy,
                            num_operators=result.num_operators,
                            converged=result.converged,
                            inner_failures=len(result.optimizer_failures),
                            n_qubits=calc.solver.n_qubits,
                            particle_sector_weight=sector_weight,
                        ))
                        handle.flush()
                        print(f"{basis:8s} {pool:10s} R={distance:.3f} Å "
                              f"E={result.optimal_energy:.6f} eV "
                              f"converged={result.converged}", flush=True)


def plot(output: Path) -> None:
    """Plot the saved data without rerunning the quantum calculations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "energies.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads((output / "metadata.json").read_text())
    expected = len(BASES) * len(POOLS) * len(metadata["distances_A"])
    if len(rows) != expected:
        raise ValueError(f"Incomplete scan: expected {expected} rows, found {len(rows)}.")

    colors = ("#0072B2", "#D55E00", "#009E73", "#882255")
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(2, len(BASES), figsize=(10.5, 7.2), sharex=True,
                             layout="constrained")
    for column, basis in enumerate(BASES):
        reference_rows = sorted(
            (r for r in rows if r["basis"] == basis and r["pool"] == "fermionic"),
            key=lambda r: float(r["distance_A"]),
        )
        reference = np.array([float(r["energy_eV"]) for r in reference_rows])
        for pool, color, marker in zip(POOLS, colors, markers):
            series = sorted(
                (r for r in rows if r["basis"] == basis and r["pool"] == pool),
                key=lambda r: float(r["distance_A"]),
            )
            distances = np.array([float(r["distance_A"]) for r in series])
            energies = np.array([float(r["energy_eV"]) for r in series])
            good = np.array([r["converged"] == "True" and int(r["inner_failures"]) == 0
                             for r in series])
            for ax, values in ((axes[0, column], energies),
                               (axes[1, column], 1000 * (energies - reference))):
                ax.plot(distances, values, color=color, marker=marker,
                        markersize=6, markerfacecolor="none", linewidth=1.2,
                        label=pool)
                if not good.all():
                    ax.scatter(distances[~good], values[~good], color="black",
                               marker="x", s=65, zorder=10)
        axes[0, column].set_title(f"Mandacaru {basis} · {reference_rows[0]['n_qubits']} qubits")
        axes[0, column].set_ylabel("Total energy (eV)")
        axes[0, column].ticklabel_format(useOffset=False, axis="y")
        axes[0, column].legend(frameon=False, fontsize=9)
        axes[1, column].axhline(0, color="0.6", linewidth=0.8)
        axes[1, column].set_ylabel("Difference from fermionic pool (meV)")
        axes[1, column].set_xlabel("Li–H distance (Å)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LiH: basis sets and ADAPT-VQE operator pools\n"
                 "Frozen core · Jordan–Wigner · numerical teaching example", fontsize=13)
    fig.supxlabel("Lines connect calculated points; black crosses mark incomplete convergence.",
                  fontsize=9)
    fig.savefig(output / "basis_pool_scan.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {output / 'basis_pool_scan.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing", type=float, default=0.12)
    parser.add_argument("--half-width", type=float, default=4.8)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=40)
    args = parser.parse_args()
    if args.spacing <= 0 or args.half_width <= 0 or args.points < 2 or args.max_iterations < 1:
        parser.error("Use positive lengths and iterations, and at least two points.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        calculate(args.output_dir, args.spacing, args.half_width,
                  args.points, args.max_iterations)
    plot(args.output_dir)


if __name__ == "__main__":
    main()
