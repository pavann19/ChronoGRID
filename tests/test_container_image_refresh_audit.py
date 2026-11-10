from scripts.build_container_image_refresh_audit import package_rows, row_delta


def test_package_row_reconciliation_preserves_duplicate_ids_across_packages():
    old = {
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
                    },
                    {
                        "VulnerabilityID": "CVE-2026-109518",
                        "PkgName": "libapt-pkg7.0",
                        "InstalledVersion": "3.0.3",
                        "Severity": "UNKNOWN",
                        "FixedVersion": "3.0.4",
                    },
                ],
            }
        ]
    }
    candidate = {"Results": [{"Type": "debian", "Vulnerabilities": []}]}

    before_rows = package_rows(old)
    after_rows = package_rows(candidate)
    delta = row_delta(before_rows, after_rows)

    assert len(delta["removed"]) == 2
    assert {row["PkgName"] for row in delta["removed"]} == {"apt", "libapt-pkg7.0"}
    assert delta["added"] == []


def test_package_row_reconciliation_surfaces_new_rows_instead_of_hiding_them():
    previous = {"Results": [{"Type": "debian", "Vulnerabilities": []}]}
    candidate = {
        "Results": [
            {
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-9999",
                        "PkgName": "new-package",
                        "InstalledVersion": "1.0",
                        "Severity": "HIGH",
                        "FixedVersion": "1.1",
                    }
                ],
            }
        ]
    }

    delta = row_delta(package_rows(previous), package_rows(candidate))

    assert len(delta["added"]) == 1
    assert delta["added"][0]["VulnerabilityID"] == "CVE-2026-9999"


def test_package_version_transition_keeps_the_same_advisory_visible():
    previous = {
        "Results": [
            {
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2011-3374",
                        "PkgName": "apt",
                        "InstalledVersion": "3.0.3",
                        "Severity": "LOW",
                        "FixedVersion": None,
                    }
                ],
            }
        ]
    }
    candidate = {
        "Results": [
            {
                "Type": "debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2011-3374",
                        "PkgName": "apt",
                        "InstalledVersion": "3.0.4",
                        "Severity": "LOW",
                        "FixedVersion": None,
                    }
                ],
            }
        ]
    }

    delta = row_delta(package_rows(previous), package_rows(candidate))

    assert delta["removed"][0]["InstalledVersion"] == "3.0.3"
    assert delta["added"][0]["InstalledVersion"] == "3.0.4"
    assert delta["added"][0]["VulnerabilityID"] == "CVE-2011-3374"
