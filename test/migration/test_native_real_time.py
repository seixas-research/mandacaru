"""The real-time traces of the native propagator against their frozen values
(``reference/real_time.json``, written by ``freeze_real_time.py``). Needs ``MANDACARU_PAW_PATH``."""

import json
from pathlib import Path

import pytest

from .freeze_real_time import STRIDE, run

PATH = Path(__file__).parent / "reference" / "real_time.json"
REFERENCE = json.loads(PATH.read_text()) if PATH.exists() else {"cases": {}}
# The SCF replays to about 1e-9 eV, and the corrector stops at 1e-6 in the density, so a replay
# can take a different iteration count; the trace is held to 1e-6 of its largest value.
RELATIVE = 1e-6


# A case whose propagation took over two minutes when frozen (water HSE06, 203 s: the complex
# exchange of every build) runs only with the slow tier.
CASES = [pytest.param(key, marks=pytest.mark.slow) if record["seconds"] > 120 else key
         for key, record in sorted(REFERENCE["cases"].items())]


@pytest.mark.parametrize("key", CASES)
def test_real_time_traces_match_the_frozen_values(key):
    frozen = REFERENCE["cases"][key]
    rt, _, calc = run(key)
    found = rt.along[::STRIDE]
    assert len(found) == len(frozen["along_eA"])
    scale = max(abs(v) for v in frozen["along_eA"])
    for value, reference in zip(found, frozen["along_eA"]):
        assert abs(value - reference) < RELATIVE * scale, key
    assert abs(rt.kick_energy - frozen["kick_energy_eV"]) < 1e-6 * frozen["kick_energy_eV"]
    calc.close()
