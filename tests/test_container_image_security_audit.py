from scripts.build_container_image_security_audit import vulnerability_rows


def test_vulnerability_rows_reconcile_package_level_findings():
    image = {
        "Results": [
            {
                "Target": "image",
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "PkgName": "liba",
                        "InstalledVersion": "1.0",
                        "Severity": "HIGH",
                        "FixedVersion": None,
                    },
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "PkgName": "libb",
                        "InstalledVersion": "2.0",
                        "Severity": "HIGH",
                        "FixedVersion": "2.1",
                    },
                ],
            }
        ]
    }
    rootfs = {
        "Results": [
            {
                "Target": "rootfs",
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "PkgName": "libb",
                        "InstalledVersion": "2.0",
                        "Severity": "HIGH",
                        "FixedVersion": "2.1",
                    },
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "PkgName": "liba",
                        "InstalledVersion": "1.0",
                        "Severity": "HIGH",
                        "FixedVersion": None,
                    },
                ],
            }
        ]
    }

    assert vulnerability_rows(image) == vulnerability_rows(rootfs)
