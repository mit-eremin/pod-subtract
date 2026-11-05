"""End to end: one frame of the published bgsub v2.1 synthetic test set, re-made from the simulator bank and re-fitted
by this package, equals the background checkpoint row for row. Skipped when ~/pod_data is not present."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from bgsub import config

ROOT = Path(__file__).resolve().parents[1]
KEY = "formvar_carbon_tedpella_b8_d1.0_plain_r0"
REF = config.BG_INPUTS["bgsub_v2_1"] / "checkpoints" / "synth_test" / f"{KEY}.pkl"
ENTRY = config.SIMBANK_V1 / "v1synth_test" / f"{KEY}.npz"


@pytest.mark.skipif(not (REF.exists() and ENTRY.exists() and config.BG_INPUTS["spot_shape"].exists()), reason="pod_data caches absent")
def test_v21_synthetic_frame_reproduces_published_checkpoint(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import synth_v2_1 as SV
    import check_rerun_vs_background as CHK
    from bgsub.validation import synth as S1
    job = [c for c in S1.cells("test") if S1.cell_key(*c[1:6]) == KEY][0] + ("test",)
    SV.run_one(job, ck_dir=tmp_path)
    ok, msg = CHK.compare(pickle.load(open(tmp_path / f"{KEY}.pkl", "rb")), pickle.load(open(REF, "rb")))
    assert ok, msg
