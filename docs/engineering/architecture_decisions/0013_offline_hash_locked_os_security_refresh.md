# ADR 0013: Offline hash-locked OS package security refresh

Date: 2026-11-04
Status: Accepted for the local operations research image

## Context

The digest-pinned Python 3.13 slim image contained Debian Trixie APT 3.0.3. A newer Trivy database identified `CVE-2026-109518` on `apt` and `libapt-pkg7.0` 3.0.3 and listed 3.0.4 as fixed. The official Debian `trixie-security` index exposed the two 3.0.4 packages, but resolving packages over the network during every image build would make the build depend on mutable repository state.

## Decision

Retain the digest-pinned base and frontend. Store the exact APT 3.0.4 `.deb` files in a versioned evidence directory, record their repository locations, sizes, and SHA-256 hashes, and install them through a read-only BuildKit bind mount. Verify the package hashes before installation and verify the installed versions after `dpkg -i`. Do not run `apt-get update` or network package resolution in the image build.

The two package payloads were selected from the HTTPS Debian security repository after the disposable base container refreshed and verified the configured Debian archive keyring. Their checksums match the package metadata from that signed repository snapshot. See `artifacts/engineering/container_base_candidate_20261104_v2/security_package_lock.json` and `debian_security_packages/SHA256SUMS`.

## Alternatives considered

- Refresh the upstream base digest: the official tag resolved to the same older base snapshot during this audit, so it did not include the available APT security update.
- Run `apt-get update` and `apt-get upgrade` in the Dockerfile: this makes the image depend on the repository contents at build time and can silently broaden the package changes.
- Leave APT 3.0.3 unchanged: rejected because the refreshed vulnerability database identified a fixed package release available from Debian security.

## Consequences and limits

The image build is deterministic with respect to the two added package payloads and remains network-independent after the base image is present. Updating them requires a new evidence directory, a new checksum lock, an image rebuild, runtime checks, and a fresh scan against a recorded vulnerability database. The local candidate removes the two `CVE-2026-109518` package rows and adds no unique advisory ID, but nine HIGH Debian IDs remain. Trivy continues to show LOW `CVE-2011-3374` matches on the updated APT package versions; this decision does not classify those rows as false positives or fixed. The image remains conditional for local research use.
