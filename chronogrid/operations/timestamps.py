"""Offline RFC3161 request/receipt verification. Never contacts a TSA automatically."""

import base64
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chronogrid.evaluation.prospective import utc
from chronogrid.operations.journal import digest


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode:
        raise ValueError("OpenSSL timestamp verification failed: " + result.stderr[-500:])
    return result.stdout


def make_request(sha256: str, target: Path, openssl="openssl"):
    if not re.fullmatch("[0-9a-f]{64}", sha256):
        raise ValueError("invalid SHA-256")
    if Path(target).exists():
        raise FileExistsError(target)
    # OpenSSL adds a nonce by default; retain exact request for verification.
    raw = subprocess.run(
        [openssl, "ts", "-query", "-digest", sha256, "-sha256", "-cert"],
        capture_output=True,
        timeout=15,
        check=True,
    ).stdout
    with Path(target).open("xb") as f:
        f.write(raw)
    return digest(raw)


def verify_receipt(
    store,
    run_id,
    day,
    arm,
    query: Path,
    response: Path,
    ca: Path,
    expected_ca_sha256: str,
    authority: str,
    trust_scope: str,
    openssl="openssl",
):
    if trust_scope not in ("external_configured", "local_test"):
        raise ValueError("explicit receipt trust scope required")
    if digest(Path(ca).read_bytes()) != expected_ca_sha256:
        raise ValueError("timestamp trust anchor mismatch")
    p = store.get(f"forecast:{run_id}:{day}:{arm}")
    from chronogrid.operations.journal import encode

    h = digest(encode(p))
    # Validate both retained query (nonce+imprint) and the actual forecast imprint.
    command(
        [
            openssl,
            "ts",
            "-verify",
            "-in",
            str(response),
            "-queryfile",
            str(query),
            "-CAfile",
            str(ca),
        ]
    )
    command([openssl, "ts", "-verify", "-in", str(response), "-digest", h, "-CAfile", str(ca)])
    text = command([openssl, "ts", "-reply", "-in", str(response), "-text"])
    match = re.search(r"Time stamp:\s*(.+)", text)
    if not match:
        raise ValueError("timestamp response missing generation time")
    stamp = match.group(1).strip()
    fmt = "%b %d %H:%M:%S.%f %Y GMT" if "." in stamp else "%b %d %H:%M:%S %Y GMT"
    generated = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
    from zoneinfo import ZoneInfo

    from chronogrid.operations.contracts import Forecast

    forecast = Forecast.model_validate(p)
    gate = datetime.combine(
        forecast.delivery_date - timedelta(days=1), datetime.min.time(), ZoneInfo("Europe/Berlin")
    ).replace(hour=12)
    if not utc(forecast.issued_at) <= generated <= utc(gate) or generated > utc(store.clock()):
        raise ValueError("independent commitment outside issuance/gate timeline")
    payload = dict(
        prediction_sha256=h,
        authority=authority,
        trust_scope=trust_scope,
        ca_sha256=expected_ca_sha256,
        generated_at=generated.isoformat(),
        response_base64=base64.b64encode(Path(response).read_bytes()).decode(),
        query_base64=base64.b64encode(Path(query).read_bytes()).decode(),
        verification="openssl_signature_chain_nonce_imprint_time",
    )
    with store.transaction() as db:
        return store._add(
            db,
            f"receipt:{run_id}:{day}:{arm}:{digest(Path(response).read_bytes())}",
            "receipt",
            payload,
        )
