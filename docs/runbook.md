# Local runbook

I install the package with `make install`, then run `make lint`, `make test` and `make audit`. I run `make benchmark` only when I intend to regenerate the retrospective reports. I compare the saved fold boundaries and dataset identity before interpreting a changed result.

For diagnostics I configure `CHRONOGRID_DIAGNOSTICS_TOKEN` and use `make serve`. I check `/health`, then call `/model/info` with `Authorization: Bearer <token>`. An unconfigured token returns service unavailable; a missing or incorrect bearer value returns unauthorized. Replay requires `CHRONOGRID_LEGACY_REPLAY_TOKEN` and `CHRONOGRID_ENABLE_LEGACY_REPLAY=1`; its outputs remain explicitly historical or synthetic.

If startup reports a checksum mismatch, I stop and inspect the registry entry, model bytes and metadata. I do not replace the expected checksum merely to silence the check. A new model needs its own reviewed identity and compatible features/calibration. The repository's historical ONNX models are not the corrected benchmark winner.

For Docker I use `make docker`, then the loopback-only container command in the README. Neither Dockerfile reads a local wheelhouse. Dependencies are hash-locked; OS package refreshes use the build-time repository state, so I do not claim bit-for-bit image reproducibility.

Run `python scripts/preserve_research_evidence.py --verify` only for optional restored closure evidence. Without the private frozen files it reports a skip. Public weather replay tests validate numerical reconstruction without authenticating unavailable historical source bytes.

I keep local runtime databases and logs outside Git tracking. I back up a journal before experimenting with it and use isolated fixture paths for tests. A successful local check does not imply that hosted CI, a published image or a production deployment has passed.
