"""bgsub.validation.simbank reproduces podbg's simulator-v1 expected frames bit for bit (sha256 recorded by
scripts/make_simbank_v1.py in pod/background's venv), without importing podbg."""
import sys
from pathlib import Path

import numpy as np

from bgsub.validation import simbank as SB

DATA = Path(__file__).resolve().parent / "data" / "simbank_v1_sample.npz"


def test_sample_entries_render_bit_for_bit():
    ents = SB.load_bundle(DATA)
    assert set(ents) == {"lacey_b16", "lacey_b8_struct"}
    for e in ents.values():
        assert SB.verify(e), e.key
    assert ents["lacey_b8_struct"].ellipticity == 1.008


def test_no_podbg_import():
    import bgsub  # noqa: F401
    import bgsub.validation.harness, bgsub.validation.synth  # noqa: F401,E401
    assert not [m for m in sys.modules if m.split(".")[0] == "podbg"]


def test_stored_frame_mean_follows_count_law():
    import bgsub as BG
    e = SB.load_bundle(DATA)["lacey_b16"]
    mu = SB.expected(e)
    law = BG.CountLaw.apollo(16, SB.DT.FILM_P, dead_time_tau=SB.DT.apollo(16).tau_binned)
    rng = np.random.default_rng(0)
    K = np.mean([SB.stored(mu, 16, rng) for _ in range(8)], 0)
    sel = ~SB.stop_mask(e) & (mu > 0.5)
    mu_c = mu[sel] * np.exp(-mu[sel] * SB.DT.apollo(16).tau_binned)
    assert abs(K[sel].mean() / law.E(mu_c).mean() - 1) < 0.01
