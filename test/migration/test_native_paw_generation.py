# -*- coding: utf-8 -*-
# file: test/migration/test_native_paw_generation.py

# This code is part of Mandacaru.
# MIT License

"""The native PAW-LCAO / UPAW-LCAO generator through the Python API against the frozen output of the
Python generator it replaced (the A0 freeze, ``reference/paw_generate``).

What is gated, and why the gates differ.  Production generation uses the variational radial
exchange-correlation scheme (the scheme of the molecular PAW path); the freeze was made with the
atom-solver scheme (``PAW_MIGRATION_LOG.md``, "G0 / XC-SW").  For LDA the two agree to round-off
wherever the density exceeds the exchange floor and differ by a uniform-gas step of 1.9e-4 Ha
beyond about 14 Bohr, which reaches the dataset through the box-normalized *scattering*
references and the screened local potential only.  So, per case:

- the selection is exact: the index and label of the accepted construction, the number of
  constructions built and the text of every attempt that was not accepted;
- the quantities of the sphere (matrices, projectors, bound partial waves, local potential,
  densities, energies, the spin-orbit chain of a Dirac dataset) are gated at 1e-6 of their array
  maximum (measured 1.8e-8 at worst over the cases below);
- the tail-sensitive ones (``v_local_screened`` and the partial waves of the references that hold
  no electron) at 3e-3 of their maximum (measured 1.3e-3);
- the diagnosis of the generated dataset equals the checks the freeze recorded for the attempt it
  selected, to 1e-6 rad;
- for PBE the move of the scheme is itself gated: the one-center energy of oxygen moves by
  4.15e-5 Ha (the G0 measurement), so a production atom that fell back to the old scheme would
  fail the lower bound, and nothing else is compared at a tight gate.

The arrays of the freeze (``gen_<set>_<El>_data.npz``, git-ignored) live in
``reference/paw_generate_archive``, copied there while the Python generator still existed; a
missing input FAILS with the restore command, it never skips.  The scripts that wrote the
freeze (`freeze_paw_generation.py`, `freeze_paw_recipe.py`) were deleted with the Python generator
they drove (A7): the freeze is not regenerable, and the archive (75 files, byte-identical to
`reference/paw_generate`) is its only home.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mandacaru.pseudopotentials.diagnostics import diagnose
from mandacaru.pseudopotentials.generation import generate_paw, generate_upaw
from mandacaru.pseudopotentials.record import record_of

REFERENCE = Path(__file__).parent / "reference"
RESTORE = ("restore the archive of the A0 freeze: copy the git-ignored files "
           "`gen_*_data.npz` of a checkout that has them into "
           "test/migration/reference/paw_generate_archive/")

#: ``set -> (functional, treatment)``
SETS = {"lda-sr": ("lda", "scalar"), "pbe-sr": ("pbe", "scalar"), "lda-fr": ("lda", "dirac")}
TIGHT = 1e-6
TAIL = 3e-3
#: Arrays smaller than this (matrices of an empty channel are round-off) are compared absolutely.
NOISE_FLOOR = 1e-6

LIGHT = ["lda-sr_H", "lda-sr_O", "lda-sr_Na", "lda-sr_Si", "lda-fr_Na", "upaw_H", "upaw_O"]


def frozen(case: str):
    stem = f"gen_{case}"
    meta = REFERENCE / "paw_generate" / f"{stem}.json"
    if not meta.exists():
        pytest.fail(f"the frozen case {meta} is missing: it is tracked, `git checkout` it")
    for folder in ("paw_generate_archive", "paw_generate"):
        npz = REFERENCE / folder / f"{stem}_data.npz"
        if npz.exists():
            return json.loads(meta.read_text()), np.load(npz)
    pytest.fail(f"the frozen arrays of {case} are missing; {RESTORE}")


def generated(case: str):
    state, element = case.rsplit("_", 1)
    if state == "upaw":
        return generate_upaw(element)
    functional, relativity = SETS[state]
    return generate_paw(element, xc=functional, relativity=relativity)


def deviations(case: str, pp):
    """``{item: (deviation, gate)}`` of the generated dataset against the freeze."""
    meta, data = frozen(case)
    payload, tables = record_of(pp)
    record = json.loads(payload)
    out = {}
    for name, info in meta["arrays"].items():
        key = "d_" + name
        if key not in data.files or name.startswith("so_"):
            continue
        ours = tables[name]
        ours = ours[math.ceil(ours.size / 2000) - 1::math.ceil(ours.size / 2000)]
        scale = max(info.get("abs_max", abs(info["max"])), 1e-300)
        tail = name == "v_local_screened"
        wave = name.startswith(("ae_wave_l", "pseudo_wave_l"))
        if wave:
            l, i = name.rsplit("_l", 1)[1].split("_")
            tail = record["channels"][l]["occupations"][int(i)] == 0.0
        out["table." + name] = (float(np.max(np.abs(ours - data[key])) / scale),
                                TAIL if tail else TIGHT)
    for l, channel in record["channels"].items():
        for name in ("coupling", "coupling_screened", "overlap_correction", "norm_correction",
                     "kinetic_difference", "potential_difference", "vanderbilt"):
            info = meta["arrays"][f"{name}_l{l}"]
            want = np.array(info["values"]).reshape(info["shape"])
            scale = max(np.max(np.abs(want)), NOISE_FLOOR)
            out[f"matrix.{name}_l{l}"] = (float(np.max(np.abs(np.array(channel[name]) - want)) / scale), TIGHT)
        for name in ("reference_energies", "occupations", "residual_kinetic"):
            want = np.array(meta["arrays"][f"{name}_l{l}"]["values"])
            out[f"vector.{name}_l{l}"] = (
                float(np.max(np.abs(np.array(channel[name]) - want)) / max(1.0, np.max(np.abs(want)))), TIGHT)
        out[f"channel.r_cut_l{l}"] = (abs(channel["r_cut"] - meta["scalars"]["channels"][l]["r_cut"]), 1e-12)
    for l, entry in record["spin_orbit"].items():
        for name, value in entry.items():
            if name == "projectors":
                for i, p in enumerate(value):
                    info = meta["arrays"][f"so_projector_l{l}_{i}"]
                    ours = np.asarray(p)
                    stride = math.ceil(ours.size / 2000)
                    ours = ours[stride - 1::stride]
                    out[f"spin_orbit.projector_l{l}_{i}"] = (
                        float(np.max(np.abs(ours - data[f"d_so_projector_l{l}_{i}"])) / info.get("abs_max", abs(info["max"]))), 1e-7)
            else:
                info = meta["arrays"][f"so_{name}_l{l}"]
                want = np.array(info["values"]).reshape(info["shape"])
                out[f"spin_orbit.{name}_l{l}"] = (
                    float(np.max(np.abs(np.array(value) - want)) / max(np.max(np.abs(want)), NOISE_FLOOR)), 1e-7)
    scalars = meta["scalars"]
    for name in ("valence_charge", "r_cut_local", "local_shift", "compensation_radius", "q_cut",
                 "energy_offset", "norm_deficit"):
        out["scalar." + name] = (abs(record[name] - scalars[name]), 1e-12)
    for name in ("compensation_charge", "hartree_screening", "one_center_energy"):
        out["scalar." + name] = (abs(record[name] - scalars[name]) / max(1.0, abs(scalars[name])), 1e-9)
    for name, value in scalars["energies"].items():
        out["energy." + name] = (abs(record["energies"][name] - value) / max(1.0, abs(value)), 1e-9)
    for name in ("symbol", "atomic_number", "family", "xc", "relativity", "extra_l",
                 "relativistic_exchange"):
        assert record[name] == scalars[name], name
    assert [list(s) for s in record["frozen_subshells"]] == scalars["frozen_subshells"]
    assert [list(s) for s in record["semicore_subshells"]] == scalars["semicore_subshells"]
    return out


def check_selection(case: str, pp):
    meta, _ = frozen(case)
    chosen = meta["ghost_free"]
    assert pp.search.index == chosen["selected_index"]
    assert pp.search.label == chosen["selected_label"]
    assert list(pp.search.tried) == chosen["tried"]
    assert pp.search.flagged is False and pp.defects == {}


def check_diagnosis(case: str, pp):
    meta, _ = frozen(case)
    checks = meta["ghost_free"]["attempts"][meta["ghost_free"]["selected_index"]]["checks"]
    found = diagnose(pp)
    assert found["ghosts"] == {} and checks["ghost_errors"] == {}
    for l, (near, far) in checks["scattering_errors"].items():
        assert found["phases"][int(l)] == pytest.approx((near, far), abs=1e-6)
    for l, value in checks["unprojected_scattering_errors"].items():
        assert found["unprojected"][int(l)] == pytest.approx(value, abs=1e-6)
    assert found["s_miss"] == pytest.approx(meta["intruding_s_miss"], abs=1e-6)


@pytest.mark.parametrize("case", LIGHT)
def test_the_generated_dataset_is_the_frozen_one(case):
    pp = generated(case)
    check_selection(case, pp)
    check_diagnosis(case, pp)
    items = deviations(case, pp)
    failures = [f"{name}: {dev:.3e} > {gate:.0e}" for name, (dev, gate) in items.items()
                if not dev <= gate]
    assert not failures, "\n".join(failures)


def test_a_wrong_dataset_is_beyond_the_gates():
    """The negative control: the same comparison fails on a dataset that is not the frozen one
    (hydrogen against the frozen oxygen's arrays has other shapes, so use a perturbed copy)."""
    import dataclasses

    pp = generated("lda-sr_O")
    channels = dict(pp.channels)
    channels[0] = dataclasses.replace(
        pp.channels[0], coupling_screened=pp.channels[0].coupling_screened * (1 + 1e-4))
    items = deviations("lda-sr_O", dataclasses.replace(pp, channels=channels))
    dev, gate = items["matrix.coupling_screened_l0"]
    assert dev > 20 * gate


def test_pbe_moves_with_the_variational_scheme_and_keeps_the_selection():
    case = "pbe-sr_O"
    pp = generated(case)
    check_selection(case, pp)
    meta, _ = frozen(case)
    frozen_energy = meta["scalars"]["one_center_energy"]
    move = pp.one_center_energy - frozen_energy
    # G0 measured +4.148e-5 Ha for this dataset; the atom-solver scheme would give ~1e-12.
    assert 3e-5 < move < 5e-5
    # What does not depend on the scheme stays exactly where the freeze put it.
    items = deviations(case, pp)
    for name in ("scalar.r_cut_local", "scalar.compensation_radius", "scalar.norm_deficit",
                 "channel.r_cut_l0", "channel.r_cut_l1", "scalar.valence_charge"):
        dev, gate = items[name]
        assert dev <= gate, name
    # And the move is the documented size: the sphere's matrices follow the potential by a
    # few percent of their maximum, the freeze's own gates (1e-6) are far beyond.
    assert 1e-3 < items["matrix.coupling_screened_l0"][0] < 6e-2
    check = diagnose(pp)
    assert check["ghosts"] == {} and check["wrong_phases"] == {}
