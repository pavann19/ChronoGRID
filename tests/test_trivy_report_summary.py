from scripts.summarize_trivy_report import summarize


def test_summary_deduplicates_cves_and_keeps_highest_severity():
    report = {
        "Metadata": {"OS": {"Family": "debian", "Name": "13.7"}},
        "Results": [
            {
                "Target": "image (debian 13.7)",
                "Type": "debian",
                "Packages": [{"Name": "alpha"}, {"Name": "beta"}],
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "Severity": "MEDIUM",
                        "PkgName": "alpha",
                        "InstalledVersion": "1.0",
                        "FixedVersion": "1.1",
                        "Title": "shared issue",
                    },
                    {
                        "VulnerabilityID": "CVE-2026-0001",
                        "Severity": "HIGH",
                        "PkgName": "beta",
                        "InstalledVersion": "2.0",
                        "Title": "shared issue",
                    },
                ],
            },
            {"Target": "Python", "Type": "python-pkg", "Packages": [], "Vulnerabilities": []},
        ],
    }

    summary = summarize(report, "a" * 64)

    assert summary["totals"] == {
        "vulnerability_entries": 2,
        "unique_vulnerability_ids": 1,
        "unique_severity_counts": {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 0,
            "LOW": 0,
            "UNKNOWN": 0,
        },
    }
    assert summary["vulnerabilities"][0]["severity"] == "HIGH"
    assert summary["vulnerabilities"][0]["packages"] == ["alpha@1.0", "beta@2.0"]
    assert summary["vulnerabilities"][0]["fixed_versions"] == ["1.1"]
