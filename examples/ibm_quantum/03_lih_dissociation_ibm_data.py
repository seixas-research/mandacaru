"""Plot the LiH dissociation curve measured on an IBM processor against the local curve.

The CSV file in data/ is a paid hardware result: this script only reads it.
"""
import csv
import os
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(DATA, "lih_dissociation_ibm.csv"), newline="") as handle:
    rows = list(csv.DictReader(handle))
measured = [r for r in rows if r["energy_measured_eV"]]

plt.plot([float(r["distance_A"]) for r in rows],
         [float(r["energy_local_eV"]) for r in rows], "o-", label="local state vector")
plt.errorbar([float(r["distance_A"]) for r in measured],
             [float(r["energy_measured_eV"]) for r in measured],
             yerr=[float(r["std_measured_eV"]) for r in measured], fmt="s--", capsize=3,
             label=f"{measured[0]['backend']} (job {measured[0]['job_id']})")
print(f"{len(measured)} points measured on {measured[0]['backend']}")
plt.xlabel("Li-H distance (Angstrom)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.savefig(os.path.join(DATA, "replot_lih_dissociation.png"), dpi=150)
