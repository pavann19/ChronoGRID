"""Verify clean-context custody and failure evidence without depending on Docker."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import verify_clean_container_builds as verifier


@pytest.mark.parametrize('failure', [False, True])
def test_build_uses_committed_bytes_and_preserves_failure_report(tmp_path, monkeypatch, failure):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
    git('init', '-q')
    (repo / 'tracked.txt').write_text('committed content')
    git('add', 'tracked.txt')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture')
    commit = git('rev-parse', 'HEAD')
    (repo / 'tracked.txt').write_text('dirty content')
    wheels = repo / 'deployment/wheelhouse'
    wheels.mkdir(parents=True)
    (wheels / 'local.whl').write_text('must not enter context')
    output = tmp_path / 'evidence'
    seen = []
    def build(profile, context, evidence, actual_commit):
        assert (context / 'tracked.txt').read_text() == 'committed content'
        assert not (context / 'deployment/wheelhouse').exists()
        assert actual_commit == commit
        seen.append(profile)
        (evidence / 'build.log').write_text('retained output')
        if failure:
            raise RuntimeError('synthetic build failure')
        return dict(status='PASS')
    monkeypatch.setattr(verifier, 'ROOT', repo)
    monkeypatch.setattr(verifier, 'build_profile', build)
    monkeypatch.setattr(sys, 'argv', ['verify', '--profile', 'operations', '--output', str(output)])
    if failure:
        with pytest.raises(RuntimeError, match='synthetic build failure'):
            verifier.main()
    else:
        verifier.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['commit'] == commit and seen == ['operations']
    assert report['status'] == ('FAIL' if failure else 'PASS')
    assert 'build.log' in report['files'] and len(report['context_archive_sha256']) == 64
    with pytest.raises(FileExistsError):
        verifier.main()
