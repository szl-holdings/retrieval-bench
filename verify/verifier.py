"""Receipt verifier for the SZL bench planes.

Schema (v1):
  plane        str   one of "engine" | "retrieval" | "quant" | "calibration"
  status       str   one of MEASURED | BLOCKED | INVALID | FAILED | PROMOTED
  machine      obj   {cpu: str, ram_gb: number, gpu: str}
  measured_at  str   ISO-8601 date or datetime (UTC)
  method       str   free text naming harness + version
  metrics      obj   plane-specific measured values (empty for non-MEASURED)
  prev_hash    str   sha256 hex of previous receipt, or 64 zeros for genesis
  hash         str   sha256 of canonical JSON of this receipt without "hash"

Rules: fail closed. Any malformed receipt, bad hash, or broken chain exits 1.
Only MEASURED receipts are publishable; other states are reported, never published.
Integrity only (UNSIGNED-honest): proves chain continuity, not signer identity.
"""
import hashlib
import json
import sys

STATUSES = {"MEASURED", "BLOCKED", "INVALID", "FAILED", "PROMOTED"}
PLANES = {"engine", "retrieval", "quant", "calibration"}
GENESIS = "0" * 64
REQUIRED = ("plane", "status", "machine", "measured_at", "method", "metrics", "prev_hash", "hash")


def digest(receipt):
    body = {k: v for k, v in receipt.items() if k != "hash"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def unique_object(pairs):
    """Reject ambiguous JSON before a duplicate key can replace evidence."""
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def check_receipt(r, i):
    if not isinstance(r, dict):
        return [f"[{i}] receipt must be an object"]
    errs = [f"[{i}] missing field: {f}" for f in REQUIRED if f not in r]
    if errs:
        return errs
    if not isinstance(r["plane"], str) or r["plane"] not in PLANES:
        errs.append(f"[{i}] unknown plane: {r['plane']}")
    if not isinstance(r["status"], str) or r["status"] not in STATUSES:
        errs.append(f"[{i}] unknown status: {r['status']}")
    m = r["machine"]
    if not isinstance(m, dict) or not all(k in m for k in ("cpu", "ram_gb", "gpu")):
        errs.append(f"[{i}] machine must name cpu, ram_gb, gpu")
    if not isinstance(r["metrics"], dict):
        errs.append(f"[{i}] metrics must be an object")
    elif r["status"] == "MEASURED" and not r["metrics"]:
        errs.append(f"[{i}] MEASURED with empty metrics")
    try:
        if digest(r) != r["hash"]:
            errs.append(f"[{i}] hash mismatch (recomputed != declared)")
    except (TypeError, ValueError, OverflowError) as exc:
        errs.append(f"[{i}] receipt is not finite JSON: {exc}")
    return errs


def verify(paths):
    try:
        paths = list(paths)
    except TypeError:
        return ["receipt paths must be an iterable collection"], []
    if not paths:
        return ["empty receipt chain"], []
    all_errs, measured, prev = [], [], GENESIS
    for i, p in enumerate(paths):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f, object_pairs_hook=unique_object)
        except Exception as e:
            all_errs.append(f"[{i}] {p}: unreadable JSON: {e}")
            continue
        errors = check_receipt(r, i)
        if errors:
            all_errs.extend(errors)
            continue
        if r.get("prev_hash", prev) != prev:
            all_errs.append(f"[{i}] {p}: chain break (prev_hash mismatch)")
        prev = r.get("hash", prev)
        if r.get("status") == "MEASURED":
            measured.append(p)
    return all_errs, [] if all_errs else measured


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: verifier.py <receipt.json> [...]")
        sys.exit(2)
    errors, ok = verify(args)
    for e in errors:
        print("FAIL", e)
    print(f"checked={len(args)} measured={len(ok)} errors={len(errors)}")
    sys.exit(1 if errors else 0)
