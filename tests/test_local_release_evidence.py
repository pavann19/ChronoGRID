import hashlib
import json
import shutil
import subprocess

import pytest

from scripts.collect_local_release_evidence import counts, require_security_policy, verify


def test_vulnerability_rows_not_confused_with_distinct_advisories():
    report = dict(
        Results=[
            dict(
                Class="os-pkgs",
                Vulnerabilities=[
                    dict(VulnerabilityID="CVE-example", Severity="HIGH", PkgName="lib-one"),
                    dict(VulnerabilityID="CVE-example", Severity="HIGH", PkgName="lib-two"),
                ],
            )
        ]
    )
    result = counts(report)
    assert result["package_vulnerability_rows"] == 2
    assert result["distinct_ids"] == 1 and result["severities"]["HIGH"] == 2
    with pytest.raises(ValueError, match="HIGH/CRITICAL"):
        require_security_policy(dict(profiles={"operations": result, "service": result}))


def test_modified_manifest_rejected_before_signature_verification(tmp_path):
    (tmp_path / "evidence-manifest.json").write_text(json.dumps(dict(files={})))
    (tmp_path / "evidence-manifest.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


def test_signed_evidence_mutation_cannot_be_hidden_by_rehashing_unsigned_manifest(tmp_path):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL required for the signature integration check")

    def run(*args):
        subprocess.run(["openssl", *map(str, args)], check=True, capture_output=True)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    key = tmp_path / "private.pem"
    pub = tmp_path / "local-review.pub.pem"
    payload = tmp_path / "evidence.txt"
    attestation = tmp_path / "attestation.json"
    payload.write_text("original evidence")
    run("genpkey", "-algorithm", "ED25519", "-out", key)
    run("pkey", "-in", key, "-pubout", "-out", pub)
    attestation.write_text(
        json.dumps(
            dict(
                predicate=dict(
                    evidence={"evidence.txt": digest(payload)}, local_public_key_sha256=digest(pub)
                )
            )
        )
    )
    run(
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        key,
        "-in",
        attestation,
        "-out",
        tmp_path / "attestation.sig",
    )

    def inventory():
        path = tmp_path / "evidence-manifest.json"
        path.write_text(
            json.dumps(
                dict(
                    files={
                        p.name: digest(p)
                        for p in [pub, payload, attestation, tmp_path / "attestation.sig"]
                    }
                )
            )
        )
        (tmp_path / "evidence-manifest.sha256").write_text(digest(path))

    inventory()
    verify(tmp_path, digest(pub))
    with pytest.raises(ValueError, match="fingerprint"):
        verify(tmp_path, "0" * 64)
    payload.write_text("tampered evidence")
    inventory()
    with pytest.raises(ValueError, match="signed evidence"):
        verify(tmp_path, digest(pub))
