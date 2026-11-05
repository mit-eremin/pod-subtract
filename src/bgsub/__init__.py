# MOVED from pod/background/src/podbg/bgsub/__init__.py at background commit f1e93f5 (file last changed in 98139b6; sha256 7aa213bce9d45b18c22b545cabf25af2e41563d7968485008a99ca4bc0de8e19).
# bgsub owns this code from 2026-11-05; the background copy is its historical origin, not a source to keep in step.
"""bgsub: fast, unbiased, universal background ESTIMATION AND SUBTRACTION for single diffraction frames.

Extracted from pod/background (podbg.bgsub) on 2026-11-05. The product is the background estimate and its per-pixel
variance; the integrators (G-ML, LC, intervals) are kept but are not product criteria (owner scope note 2026-11-05).
Design, derivations and validation: notes/reports/2026-11-04_bgsub_algorithm_v0.md (derivations), _v1, 2026-11-05 _v2,
_v2_1. Versions: version="v1" (default), "v2", "v2.1" (opt-in), "v0" (reproducible).

    from bgsub import CountLaw, Geometry, estimate_background, make_plan, disc_mask
    law = CountLaw.apollo(binning=8, tie_up=0.35)          # or CountLaw.poisson(gain); .with_offset(1) for MRC -> TIFF + 1
    res = estimate_background(frame, mask, Geometry((cy, cx)), law, exclude=spot_discs, version="v2.1")
    bg, var = res                                           # stored counts; variance of the estimate
    subtracted = frame - bg
"""
__version__ = "0.1.0"

from .law import CountLaw
from .core import Geometry, Plan, make_plan, estimate_background, BackgroundResult
from .integrate import integrate_spots, disc_mask
from . import reference
from . import integrate, zone, spotlaw

__all__ = ["CountLaw", "Geometry", "Plan", "make_plan", "estimate_background", "BackgroundResult", "integrate_spots",
           "disc_mask", "reference", "integrate", "zone", "spotlaw"]
