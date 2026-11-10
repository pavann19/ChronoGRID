"""Create once or verify a separate review sidecar; never rewrites model metadata."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronogrid.operations.feature_contracts import MANIFEST, SOURCE_PATHS, feature_dependencies, load_binding, sha

EXPERIMENTS = ('issuance_v2', 'local_calendar_v3')


def create():
    policies = {}
    for experiment in EXPERIMENTS:
        directory = ROOT / 'artifacts/experiments' / experiment
        metadata = json.loads((directory / 'fallback_metadata.json').read_bytes())
        policies[metadata['feature_policy']] = dict(
            experiment=experiment,
            calendar_timezone='Europe/Berlin' if experiment == 'local_calendar_v3' else 'UTC',
            artifacts={name: sha((directory / name).read_bytes()) for name in
                       ('fallback.joblib', 'fallback_calibration.json', 'fallback_metadata.json')},
            sources={name: sha((ROOT / name).read_bytes()) for name in SOURCE_PATHS},
            dependencies=feature_dependencies(),
        )
    manifest = dict(schema_version='1.0',
                    scope='Retrospective binding review; not original training provenance or independent attestation',
                    policies=policies)
    with (ROOT / MANIFEST).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')


def verify(wheel=None):
    for experiment in EXPERIMENTS:
        directory = ROOT / 'artifacts/experiments' / experiment
        metadata = json.loads((directory / 'fallback_metadata.json').read_bytes())
        load_binding(directory, metadata)
    if wheel:
        with zipfile.ZipFile(wheel) as archive:
            for name in (*SOURCE_PATHS, MANIFEST):
                if archive.read(name) != (ROOT / name).read_bytes():
                    raise ValueError(f'wheel binding/source byte mismatch: {name}')
        print('PASS: wheel contains byte-identical binding manifest and all eight source files')
    print('PASS: two frozen load-only adapters bound to metadata, artifacts, sources and 31 feature rules')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='exclusive create; refuses to overwrite')
    parser.add_argument('--wheel', type=Path, help='also verify packaged binding/source bytes')
    args = parser.parse_args()
    if args.create:
        create()
    verify(args.wheel)
