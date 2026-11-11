"""Freeze the seeded VASQA chains before `driver.rs` changes (VASQA plan, step 0).

Run from the repository root with the extension of the build to pin:
``python test/migration/freeze_search_chain.py [case ...]``. Every case runs at one and at eight
threads and the script stops unless the two summaries are identical, then writes
``test/migration/reference/search_chain.json`` (naming cases merges them into the existing file),
which ``test_native_search_chain.py`` replays.
"""

import json
import sys
from pathlib import Path

from search_chain_reference import CASES, run_in_temporary_store

OUT = Path(__file__).parent / "reference" / "search_chain.json"


def main(names):
    reference = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names or CASES:
        serial = run_in_temporary_store(name, 1)
        parallel = run_in_temporary_store(name, 8)
        if serial != parallel:
            raise SystemExit(f"{name}: the chain differs between 1 and 8 threads")
        reference[name] = serial
        print(f"{name}: {len(serial['steps'])} steps, identical at 1 and 8 threads", flush=True)
    OUT.write_text(json.dumps(reference, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
