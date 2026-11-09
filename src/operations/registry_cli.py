"""Explicit local diagnostic registry administration; never production promotion."""
import argparse
import json
from pathlib import Path
import sys
import sqlite3

from src.operations.artifact_registry import ArtifactRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('verify')
    legacy = sub.add_parser('import-legacy')
    legacy.add_argument('source', type=Path)
    register = sub.add_parser('register')
    register.add_argument('name')
    register.add_argument('model', type=Path)
    register.add_argument('metadata', type=Path)
    register.add_argument('--model-sha256', required=True)
    register.add_argument('--metadata-sha256', required=True)
    activate = sub.add_parser('activate')
    activate.add_argument('name')
    activate.add_argument('bundle')
    activate.add_argument('--expected-current', required=True, help='bundle ID, or none for first activation')
    rollback = sub.add_parser('rollback')
    rollback.add_argument('name')
    rollback.add_argument('--expected-current', required=True)
    backup = sub.add_parser('backup')
    backup.add_argument('target', type=Path)
    restore = sub.add_parser('restore')
    restore.add_argument('source', type=Path)
    restore.add_argument('--expected-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            result = ArtifactRegistry.create(args.db).verify()
        elif args.command == 'restore':
            _, result = ArtifactRegistry.restore(args.source, args.db, expected_sha256=args.expected_sha256)
        else:
            registry = ArtifactRegistry(args.db)
            if args.command == 'verify':
                result = registry.verify()
            elif args.command == 'import-legacy':
                result = registry.import_legacy(args.source)
            elif args.command == 'register':
                result = dict(bundle_id=registry.register(args.name, args.model.read_bytes(), args.metadata.read_bytes(),
                              expected_model_sha256=args.model_sha256, expected_metadata_sha256=args.metadata_sha256))
            elif args.command == 'activate':
                result = dict(bundle_id=registry.activate(args.name, args.bundle,
                              expected_current=None if args.expected_current == 'none' else args.expected_current))
            elif args.command == 'rollback':
                result = dict(bundle_id=registry.rollback(args.name, expected_current=args.expected_current))
            else:
                result = registry.backup(args.target)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(json.dumps(dict(error=str(exc), scope='local diagnostic registry operation rejected')), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
