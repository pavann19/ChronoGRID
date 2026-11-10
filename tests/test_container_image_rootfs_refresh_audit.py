from scripts.build_container_image_rootfs_refresh_audit import row_difference, vulnerability_rows


def test_rootfs_reconciliation_ignores_target_labels_but_preserves_package_rows():
    image = {
        "Results": [
            {
                "Target": "chronogrid:v4 (debian 13.7)",
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-109518",
                        "PkgName": "apt",
                        "InstalledVersion": "3.0.3",
                        "Severity": "UNKNOWN",
                        "FixedVersion": "3.0.4",
                    }
                ],
            }
        ]
    }
    rootfs = {
        "Results": [
            {
                "Target": "rootfs-123 (debian 13.7)",
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-109518",
                        "PkgName": "apt",
                        "InstalledVersion": "3.0.3",
                        "Severity": "UNKNOWN",
                        "FixedVersion": "3.0.4",
                    }
                ],
            }
        ]
    }

    image_rows = vulnerability_rows(image)
    rootfs_rows = vulnerability_rows(rootfs)

    assert image_rows == rootfs_rows
    assert row_difference(image_rows, rootfs_rows) == {"image_only": [], "rootfs_only": []}


def test_rootfs_reconciliation_surfaces_a_missing_package_row():
    image = {
        "Results": [
            {
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-109518",
                        "PkgName": "apt",
                        "InstalledVersion": "3.0.3",
                        "Severity": "UNKNOWN",
                        "FixedVersion": "3.0.4",
                    }
                ],
            }
        ]
    }
    rootfs = {"Results": [{"Type": "debian", "Vulnerabilities": []}]}

    delta = row_difference(vulnerability_rows(image), vulnerability_rows(rootfs))

    assert len(delta["image_only"]) == 1
    assert delta["rootfs_only"] == []
