"""The Pople and def2 shell rules (``basis::ladders::pople``, ``karlsruhe``) against the frozen
Python parser.

``reference/gaussian_names.json`` holds what ``models/basis.py::parse_basis_name`` returned for
every named Pople and def2 basis and a set of edge spellings (case, spaces, empty groups, the
3-21G* row limit) while the rules were Python (2026-11-14). Three spellings the Python read by
skipping characters or a group now raise; they are tested below."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from mandacaru.models.basis import parse_basis_name

FROZEN = json.loads(
    (Path(__file__).resolve().parent / "reference" / "gaussian_names.json").read_text())["recipes"]


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_the_recipe_is_the_frozen_one(name):
    frozen = FROZEN[name]
    if "error" in frozen:
        with pytest.raises(ValueError):
            parse_basis_name(name)
        return
    recipe = json.loads(json.dumps(asdict(parse_basis_name(name))))
    assert recipe == frozen


@pytest.mark.parametrize("name", ["6-31G(1x d)", "6-31G(dq)", "6-31G(d,p,f)", "6-31G*(d)",
                                  "6-31G(0d)"])
def test_a_spelling_the_rule_cannot_read_whole_raises(name):
    with pytest.raises(ValueError):
        parse_basis_name(name)
