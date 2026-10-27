"""Explicitly capture a portable Si PAW setup; runtime never regenerates it."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('MANDACARU_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from ase.build import bulk
    from mandacaru import Mandacaru
    from mandacaru.integrals import ensure_backend
    from inventory import source_manifest, digest_manifest
    ensure_backend(build=False)
    here = Path(__file__).parent
    case = next(c for c in json.loads((here/'cases.json').read_text())['cases'] if c['id']=='si_periodic_lda')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atoms = bulk(case['element'], case['structure'], a=case['a'])
    atoms.calc = Mandacaru(**case['options'], engine='rust', engine_options={'threads': 1},
                           trace=False, save_crystal=str(args.output))
    energy = atoms.get_potential_energy()
    payload = json.loads(args.output.read_text())
    payload['provenance'].update(reference_case=case['id'], source_sha256=digest_manifest(source_manifest()),
                                 generator='test/migration/export_crystal.py', energy_ev=energy)
    args.output.write_text(json.dumps(payload, separators=(',', ':'), sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                         energy_ev=energy, dataset_sources=payload['provenance']['dataset_sources']), indent=2))


if __name__=='__main__':
    main()
