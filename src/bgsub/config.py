"""Data locations. bgsub writes ONLY under DATA_ROOT; the background caches are read-only inputs, referenced by path.

Override with environment variables: BGSUB_DATA (own data root), POD_DATA (parent of both), BGSUB_BACKGROUND_DATA.
Nothing here is copied: frame references, never copies (umbrella rule).
"""
from __future__ import annotations

import os
from pathlib import Path

POD_DATA = Path(os.environ.get("POD_DATA", Path.home() / "pod_data"))
DATA_ROOT = Path(os.environ.get("BGSUB_DATA", POD_DATA / "bgsub"))                 # own outputs (versioned subfolders)
BACKGROUND = Path(os.environ.get("BGSUB_BACKGROUND_DATA", POD_DATA / "background"))  # READ-ONLY
CACHE = POD_DATA / "cache"                                                          # shared small-file mirror (READ-ONLY)

# Simulator-v1 bank (frozen radial tables; see bgsub.validation.simbank and scripts/make_simbank_v1.py)
SIMBANK_V1 = DATA_ROOT / "simbank_v1"

# READ-ONLY inputs used by the v0-v2.1 validations (all produced by pod/background; do not write here)
BG_INPUTS = {
    "spot_shape": BACKGROUND / "peak_recovery_v0" / "spot_shape_v0.csv",          # bin-8 spot sigmas (injection)
    "flyscan_hosts": BACKGROUND / "peak_recovery_v0" / "hosts",                     # cached fly-scan host frames (bin 8)
    "classic_frames": BACKGROUND / "s_level_bin4_v0" / "frames",                    # cached classic raw-count bin-4 No frames
    "classic_profiles": BACKGROUND / "s_level_bin4_v0" / "classic_profiles_v0.pkl",
    "features": BACKGROUND / "features",                                            # <sid>_bin8_features_v1.parquet, _shells_v1.npz
    "atlas_v1b": BACKGROUND / "atlas" / "v1b",                                       # <sid>_build_v1b.npz (gain_flag)
    "bgsub_v0": BACKGROUND / "bgsub_v0",                                            # published result tables, per version
    "bgsub_v1": BACKGROUND / "bgsub_v1",
    "bgsub_v2": BACKGROUND / "bgsub_v2",
    "bgsub_v2_1": BACKGROUND / "bgsub_v2_1",
}


def out_dir(name: str) -> Path:
    """A versioned output folder under DATA_ROOT (created)."""
    p = DATA_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p
