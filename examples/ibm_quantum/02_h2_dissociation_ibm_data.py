"""Plot the H2 dissociation curves measured on IBM processors against the local curve.

The CSV files in data/ are paid hardware results: this script only reads them.
"""
import csv
import os
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def read(name):
    with open(os.path.join(DATA, name), newline="") as handle:
        return list(csv.DictReader(handle))


local = read("h2_dissociation_ibm_fez.csv")
plt.plot([float(r["distance_A"]) for r in local],
         [float(r["energy_local_eV"]) for r in local], "o-", label="local state vector")
for name in ("h2_dissociation_ibm_fez.csv", "h2_dissociation_ibm_level2.csv"):
    rows = [r for r in read(name) if r["energy_measured_eV"]]
    plt.errorbar([float(r["distance_A"]) for r in rows],
                 [float(r["energy_measured_eV"]) for r in rows],
                 yerr=[float(r["std_measured_eV"]) for r in rows], fmt="s--", capsize=3,
                 label=f"{rows[0]['backend']} ({rows[0]['shots']} shots)")
    print(f"{name}: {len(rows)} points measured on {rows[0]['backend']}")
plt.xlabel("H-H distance (Angstrom)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.savefig(os.path.join(DATA, "replot_h2_dissociation.png"), dpi=150)
