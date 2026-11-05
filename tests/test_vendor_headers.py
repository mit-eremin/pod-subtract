"""Every vendored file names its source, the background commit and the source sha256, and says KEEP IN STEP.
When ../background is present, sources that changed since the copy are reported (a warning, not a failure: the
decision to re-copy is a session's, and it goes in LOG.md)."""
import hashlib
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT.parent / "background"
VENDORED = sorted(list((ROOT / "src/bgsub/vendor").glob("[!_]*.py")) +
                  [p for p in (ROOT / "src/bgsub/validation").glob("*.py") if p.name in ("peaks.py", "sim_geometry.py", "sim_detectors.py", "harness.py")])
HDR = re.compile(r"# VENDORED COPY\. Source: pod/background/(\S+) at background commit (\w+).*?\n# sha256 of the source file: ([0-9a-f]{64})", re.S)


def test_headers_present():
    assert len(VENDORED) == 9
    for p in VENDORED:
        head = p.read_text()[:1200]
        m = HDR.search(head)
        assert m, p
        assert "KEEP IN STEP" in head, p


def test_sources_in_step():
    if not BACKGROUND.exists():
        return
    drift = []
    for p in VENDORED:
        src, _, sha = HDR.search(p.read_text()[:1200]).groups()
        f = BACKGROUND / src
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() != sha:
            drift.append(src)
    if drift:
        warnings.warn(f"background sources changed since vendoring (re-copy or record why not): {drift}")
