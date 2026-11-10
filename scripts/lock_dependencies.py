"""Lock one installed runtime profile against its declared constraints and PyPI hashes."""

import argparse
import json
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import requests
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def requirements(path, profile="service"):
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    declarations = list(project["dependencies"])
    if profile == "service":
        declarations.extend(project["optional-dependencies"]["diagnostics"])
    return [Requirement(value) for value in declarations]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deployment/requirements-service.lock"
    )
    parser.add_argument(
        "--versions",
        type=Path,
        default=ROOT / "output/service_runtime_versions.json",
    )
    parser.add_argument("--profile", choices=["service"], default="service")
    args = parser.parse_args()
    if args.output.exists() or args.versions.exists():
        raise FileExistsError(
            "lock/version output exists; use a new version instead of overwriting evidence"
        )
    pending = requirements(args.input, args.profile)
    selected = {}
    while pending:
        req = pending.pop()
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(req.name)
        package = distribution(name)
        if package.version not in req.specifier:
            raise ValueError(f"Installed {name}=={package.version} violates {req.specifier}")
        if name in selected:
            continue
        selected[name] = package.version
        pending.extend(Requirement(raw) for raw in package.requires or [])
    lines = [
        "# Exact runtime versions and PyPI distribution hashes; selected profile: "
        + args.input.name
    ]
    for name, version in sorted(selected.items()):
        response = requests.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=20)
        response.raise_for_status()
        hashes = sorted({x["digests"]["sha256"] for x in response.json()["urls"]})
        if not hashes:
            raise ValueError("No distribution hashes: " + name)
        lines.append(
            name
            + "=="
            + version
            + " \\\n    "
            + " \\\n    ".join("--hash=sha256:" + h for h in hashes)
        )
    with args.output.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with args.versions.open("x", encoding="utf-8") as f:
        f.write(json.dumps(selected, indent=2) + "\n")
    print(
        f"Locked {len(selected)} runtime distributions with verified constraints and distribution hashes."
    )


if __name__ == "__main__":
    main()
