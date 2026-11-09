import json

import pytest

from scripts.collect_local_release_evidence import counts, verify


def test_vulnerability_rows_not_confused_with_distinct_advisories():
    report = dict(Results=[dict(Class='os-pkgs', Vulnerabilities=[
        dict(VulnerabilityID='CVE-example', Severity='HIGH', PkgName='lib-one'),
        dict(VulnerabilityID='CVE-example', Severity='HIGH', PkgName='lib-two'),
    ])])
    result = counts(report)
    assert result['package_vulnerability_rows'] == 2
    assert result['distinct_ids'] == 1 and result['severities']['HIGH'] == 2


def test_modified_manifest_rejected_before_signature_verification(tmp_path):
    (tmp_path / 'evidence-manifest.json').write_text(json.dumps(dict(files={})))
    (tmp_path / 'evidence-manifest.sha256').write_text('0' * 64)
    with pytest.raises(ValueError, match='checksum'):
        verify(tmp_path)
