# VENDORED COPY. Source: pod/background/src/podbg/provenance.py at background commit f1e93f5 (file last changed in e85dd0f);
# sha256 of the source file: 4033a5c0fcc8b8a7fc2de4258a284870374301cab82747557991108f6e174c08. Copied 2026-11-05 so that bgsub never imports podbg at run time.
# Edits relative to the source: `import podbg` -> `import bgsub` (pod_version = bgsub.__version__); repo root parents[2] -> parents[3].
# KEEP IN STEP: if the source changes in background, re-copy it here (or record why not) and update this header.
"""Every produced table gets a JSON sidecar: what made it, from what, when."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path


def git_commit(repo: Path | None = None) -> str | None:
    repo = repo or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() != ""
        return out + ("-dirty" if dirty else "")
    except Exception:
        return None


def file_stamp(p) -> dict:
    p = Path(p)
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}


def write_sidecar(target: Path, *, inputs: list, params: dict, note: str = "", extra: dict | None = None) -> Path:
    import bgsub
    side = Path(str(target) + ".provenance.json")
    doc = {
        "target": str(target),
        "produced": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pod_version": bgsub.__version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "argv": sys.argv,
        "params": params,
        "inputs": [file_stamp(i) if isinstance(i, (str, Path)) and Path(i).exists() else {"ref": str(i)} for i in inputs],
        "note": note,
    }
    if extra:
        doc.update(extra)
    side.write_text(json.dumps(doc, indent=2))
    return side
