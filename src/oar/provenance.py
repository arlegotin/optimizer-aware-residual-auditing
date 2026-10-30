"""Content identities and conservative asset notice inspection; no bulk downloads."""

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse


def hash_file(path: Path) -> str:
    """SHA-256 over bytes with bounded memory, including wheel and notice files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_asset(uri: str, revision: str) -> dict[str, object]:
    """Inspect local bytes plus <asset>.notice.json evidence; remote assets stay blocked.

    Sidecars carry source, revision, sha256, notices [{path, license, sha256}], restrictions and an
    explicit review_status. Approval records a supplied review, never a loader result
    or an automatic legal inference. Conflicting or missing notices fail closed.
    """
    report = {"source": uri, "revision": revision, "sha256": None, "size_bytes": None,
              "notice_evidence": [], "local_status": "missing", "restrictions": [],
              "approval_status": "blocked", "reason": "missing local bytes or notice evidence"}
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision is required")
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        report.update(local_status="not_downloaded", reason="remote metadata and notices require review; no download attempted")
        return report
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        raise ValueError("file URI must be local")
    path = Path(unquote(parsed.path) if parsed.scheme else uri)
    if not path.is_file():
        return report
    report.update(sha256=hash_file(path), size_bytes=path.stat().st_size, local_status="present")
    try:
        sidecar = json.loads(path.with_name(path.name + ".notice.json").read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            raise ValueError("invalid sidecar")
        restrictions = sidecar.get("restrictions")
        if not isinstance(restrictions, list) or not all(isinstance(item, str) for item in restrictions):
            raise ValueError("missing restrictions")
        report["restrictions"] = restrictions
        if (sidecar.get("revision") != revision or not sidecar.get("source")
                or sidecar.get("sha256") != report["sha256"]):
            raise ValueError("revision/source mismatch")
        report["source"] = sidecar["source"]
        notices = sidecar.get("notices")
        if not isinstance(notices, list) or not notices:
            raise ValueError("missing notices")
        for item in notices:
            notice_path = (path.parent / item["path"]).resolve()
            license_id = item["license"]
            if (not notice_path.is_relative_to(path.parent.resolve())
                    or not isinstance(license_id, str)
                    or license_id.strip().lower() in ("", "unknown", "unresolved", "noassertion")):
                raise ValueError("invalid notice")
            digest = hash_file(notice_path)
            if item.get("sha256") != digest:
                raise ValueError("notice review content mismatch")
            report["notice_evidence"].append({"path": item["path"], "license": item["license"],
                                               "sha256": digest})
        if len({item["license"] for item in report["notice_evidence"]}) != 1:
            raise ValueError("conflicting notices require reconciliation")
        if sidecar.get("review_status") != "approved":
            raise ValueError("notice review unresolved")
        report.update(approval_status="approved", reason="explicit review with consistent archived notices")
    except (OSError, ValueError, TypeError, KeyError):
        report["reason"] = "missing, conflicting, mismatched or unreviewed notice evidence"
    return report
