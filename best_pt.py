"""The fitted stress-gate value, in one place.

``pT`` (``psigma`` in the code) is the stress at which the Hill gate that makes
differentiation stress-dependent is half open. It is fitted by the psigma sweep
(``full_model.find_psigma``), and the manuscript figures all compare the fitted
value against the pT = 0 baseline.

WHY THIS MODULE EXISTS. The previous value, 0.162, was written as a literal in
six different scripts -- panel lists, hatch selectors, axis labels, two table
builders. When the edge-tension fix forced a re-fit, every one of them had to be
found and changed by hand, and a missed one does not fail: it quietly plots or
tabulates a pT that no longer exists, producing an empty panel or a silently
dropped series. One constant, imported.

THE CURRENT VALUE is 0.08556, which is NOT the nominal minimum of the
objective. That minimum is 0.08667 (18.05 against 20.82), and the difference is
not meaningful:

  * The gap is 2.8. The sweep's noise floor is far larger -- the near duplicate
    pair 0.08333/0.08334, parameter values differing by 1e-5, scored 52.97 and
    83.82. A 2.8 gap is ~9% of that, so the two points are statistically
    indistinguishable on the objective.
  * The whole of 0.08667's advantage is P0's score 2, which there is computed
    from 13 differentiation events in 3 arrays. That term is degenerate: across
    the sweep it improves monotonically as the events vanish, reaching 0.000 at
    pT = 0.12 where no run produces a single event. Every pT with >= 66 events
    scores 63-86% of events in the 0-HC-neighbour bucket against an
    experimental 24%.
  * On every quantity that is NOT that term, 0.08556 is better: E17.5 score 1
    (2.55 vs 2.68) and score 2 (5.01 vs 5.73), P0 score 1 (3.11 vs 8.65), the
    experimental differentiation rate (0.42% against 0.57% observed, where
    0.08667 gives 0.08%), and the single-HC ablation distance (ratio z^2 11.0
    vs 55.2).

So the fit localises pT to about 0.085-0.087, and 0.08556 is the value in that
range consistent with the observables that were never fitted on.

The old 0.162 was fitted against the pre-correction tension, whose face-level
median ratio to the corrected one was ~1.95 -- which is why the re-fit landed
near 0.162/1.95.

RESOLUTION. Given the noise floor above, do not quote this to more precision
than it is written here, and do not read the third decimal as meaningful.
"""
import os

from run_model import _psigma_tag

#: The fitted value. Override with the TISSUE_BEST_PT environment variable to
#: redraw any of the figures at a RIVAL pT without editing ten call sites --
#: e.g. 0.08556, the point whose differentiation fraction matches the
#: experiment's 0.57% for P0. Give those runs their own ``--out-dir``: the
#: figure stems do not carry the pT, so they would otherwise overwrite the
#: fitted set.
BEST_PT = float(os.environ.get("TISSUE_BEST_PT", "0.08556"))

# What to print in a figure label or an axis tick. Trailing zeros stripped, so
# 0.08667 reads as itself rather than as 0.086670.
BEST_PT_LABEL = ("%g" % BEST_PT)


def pt_tag(pt=BEST_PT):
    """The folder/file tag for a pT, matching how run folders are named.

    ``_psigma_tag`` writes 3 decimals when that is exact and 5 otherwise. A bare
    ``%.3f`` here would write 0.08667 as ``0.087``, which is both wrong and a
    collision: every pT in 0.0865-0.0874 would share the name.
    """
    return _psigma_tag(float(pt))
