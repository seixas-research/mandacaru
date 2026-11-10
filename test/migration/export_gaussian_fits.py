"""Freeze scale-invariant unit Slater fits as native engine data, never at runtime.

These are Mandacaru's generated fits, not published basis-set tables. The Rust
engine applies Slater scaling, recipe construction and normalization itself.

A fit is a non-convex maximization of the overlap with the Slater function, and
the native optimizer can land in a local optimum: on 2026-11-09 it gave 2s with
two Gaussians an overlap squared of 0.99514 against the shipped 0.99733 (and was
worse in 20 of 80 entries, better in 17 by at most 3e-7). So an entry already in
the output is replaced only by a strictly better fit; a regeneration never
worsens a shipped contraction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy
import scipy
from mandacaru._native import fit_sto_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "crates/mandacaru-python/src/orbitals.rs"
    report = {"format": "mandacaru-unit-slater-fits", "version": 1,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "numpy": numpy.__version__, "scipy": scipy.__version__,
              "fits": {}, "unavailable": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shipped = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        shipped = previous.get("fits", {})
        if all(previous.get(k) == report[k] for k in ("source_sha256", "numpy", "scipy")):
            report["fits"] = previous["fits"]
            report["unavailable"] = previous["unavailable"]
    # All occupied s/p/d shells through krypton; long core contractions need
    # only the first two shells for these elements. HAO has no fit table.
    states = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)]
    for n, l in states:
        counts = range(1, 11)
        for count in counts:
            key = f"{n},{l},{count}"
            if key in report["fits"] or key in report["unavailable"]:
                continue
            try:
                alpha, coeff, overlap = fit_sto_reference(n, l, count)
                fit = {"exponents": alpha, "coefficients": coeff, "overlap_squared": overlap}
                old = shipped.get(key)
                report["fits"][key] = (fit if old is None or overlap > old["overlap_squared"] + 1e-12
                                       else old)
            except RuntimeError as exc:
                report["unavailable"][key] = str(exc)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.output)
            print(key, "ok" if key in report["fits"] else "unavailable", flush=True)


if __name__ == "__main__":
    main()
