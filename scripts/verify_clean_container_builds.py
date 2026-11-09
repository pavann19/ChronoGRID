"""Build a committed Git archive without local wheels, then probe isolated containers.

Evidence distinguishes clean-context buildability from bit-for-bit image reproducibility,
security approval, signed provenance and production readiness.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {'operations': ('deployment/Dockerfile.operations', 8765),
            'service': ('deployment/Dockerfile', 8000)}
INVENTORY = """
import importlib.metadata as m, json, platform, subprocess
print(json.dumps(dict(python=platform.python_version(),
    packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in m.distributions()],key=lambda x:x['name'].lower()),
    os_packages=subprocess.check_output(['dpkg-query','-W','-f=${Package}\\t${Version}\\n'],text=True).splitlines())))
"""


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def logged(command, path, *, timeout=1200):
    with path.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {path.name}')


def build_profile(profile, context, output, commit):
    dockerfile, port = PROFILES[profile]
    tag = f'chronogrid-clean-{profile}:{commit[:12]}'
    logged(['docker', 'build', '--platform', 'linux/amd64', '--no-cache', '--progress=plain',
            '-f', str(context / dockerfile), '-t', tag, str(context)], output / f'{profile}-build.log')
    inspection = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag], text=True))[0]
    (output / f'{profile}-image.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
    if inspection['Config']['User'] != '10001:10001':
        raise ValueError('image must run as the dedicated non-root user')
    logged(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python', tag,
            '-c', INVENTORY], output / f'{profile}-runtime-inventory.json', timeout=60)
    if profile == 'operations':
        logged(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python', tag,
                '-c', "import json; from pathlib import Path; from src.operations.feature_contracts import load_binding; "
                "[(load_binding(p,json.loads((p/'fallback_metadata.json').read_bytes()))) for p in "
                "[Path('/app/artifacts/experiments/issuance_v2'),Path('/app/artifacts/experiments/local_calendar_v3')]]; print('PASS')"],
               output / 'operations-binding.log', timeout=60)
    container = subprocess.check_output([
        'docker', 'run', '-d', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges:true', '--tmpfs', '/state:uid=10001,gid=10001',
        '--tmpfs', '/tmp:uid=10001,gid=10001', tag], text=True).strip()
    try:
        probe = f"import urllib.request; [print(urllib.request.urlopen('http://127.0.0.1:{port}'+p,timeout=3).read().decode()) for p in ['/health','/ready']]"
        for _ in range(45):
            result = subprocess.run(['docker', 'exec', container, 'python', '-c', probe],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                (output / f'{profile}-probes.log').write_text(result.stdout, encoding='utf-8')
                break
            time.sleep(1)
        else:
            raise RuntimeError(f'{profile} probes did not become ready: {result.stderr}')
    finally:
        with (output / f'{profile}-runtime.log').open('w', encoding='utf-8') as stream:
            subprocess.run(['docker', 'logs', container], stdout=stream, stderr=subprocess.STDOUT, check=False)
        subprocess.run(['docker', 'rm', '-f', container], check=True, capture_output=True)
    return dict(status='PASS', image_id=inspection['Id'], user=inspection['Config']['User'],
                platform='linux/amd64', isolated_health_and_ready='PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD', help='committed revision; uncommitted files are excluded')
    parser.add_argument('--profile', choices=PROFILES, action='append')
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory; never overwritten')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = git('rev-parse', args.ref + '^{commit}')
    report = dict(commit=commit, tree=git('rev-parse', commit + '^{tree}'),
                  created_at=datetime.now(timezone.utc).isoformat(), status='FAIL', profiles={},
                  limitations=['Not bit-identical image reproducibility', 'No fresh vulnerability scan',
                               'Inventory is not a signed SBOM', 'No production readiness or hosted CI claim'])
    try:
        with tempfile.TemporaryDirectory(prefix='chronogrid-clean-') as temporary:
            temp = Path(temporary)
            archive = temp / 'context.tar'
            subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive), commit], cwd=ROOT, check=True)
            report['context_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            context = temp / 'context'
            context.mkdir()
            with tarfile.open(archive) as bundle:
                bundle.extractall(context, filter='data')
            if (context / 'deployment/wheelhouse').exists():
                raise ValueError('clean build context unexpectedly contains a wheelhouse')
            for profile in args.profile or PROFILES:
                print(f'Building and probing {profile} from {commit[:12]}', flush=True)
                report['profiles'][profile] = build_profile(profile, context, output, commit)
            report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        report['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(output.iterdir()) if p.is_file()}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: committed contexts built and isolated probes passed; see report.json', flush=True)


if __name__ == '__main__':
    main()
