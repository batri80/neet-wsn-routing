# Experiment A — Conclusion (Corrected)

## HND (primary lifetime metric): advantage is real but NOT universal
EqOpt's HND advantage holds cleanly and consistently only against `random`
(403 vs 219 at N=30, widening in relative terms down to 74 vs 40 at N=500).

Against every other baseline the picture is mixed and *degrades with scale*:
- vs `greedy`: EqOpt wins at N=30-200, but LOSES at N=500 (74.0 vs 79.5).
- vs `mincost`: EqOpt LOSES at every single N tested.
- vs `static`: EqOpt wins narrowly at N=30-60, then LOSES from N=100 onward
  (209.9 vs 216.8 at N=100, widening to 74.0 vs 80.8 at N=500).

## Matched-window Psi_max (equilibrium-quality metric): same pattern
- N=30: EqOpt wins outright.
- N>=60: EqOpt underperforms static/greedy/mincost, confirmed statistically
  significant at N=100 (Mann-Whitney p<0.001, Cohen's d>1.4 vs all three).

## Root cause (see model reference doc, Section 14.6)
One-step greedy Psi-minimization selects marginal-improvement candidates
whose structural cost outweighs their tiny Psi benefit, accumulating over
many decisions per episode. This was diagnosed via three falsified
alternative hypotheses (Lambda-neglect, distance-blindness, single bad
decision) and confirmed by a rigorous negative result on a minimum-
improvement-margin mitigation (converges toward the no-action baseline,
never exceeds it -- see Section 14.6 table).

Confirmed by the full Experiment A sweep: this doesn't just cost equilibrium
quality, it costs SURVIVAL TIME ITSELF at scale -- the cumulative structural
cost compounds with N (more triggers, more decisions per episode), eventually
outweighing both static's do-nothing baseline and mincost's equilibrium-
blind-but-structurally-conservative behavior.

## Status: CONCLUDED for one-step EqOpt.
One-step EqOpt's advantage is real and reportable, but scale-limited: reliable
only against unmanaged/naive baselines (random), eroding against structurally
conservative ones (static, mincost, and greedy at large N) as N grows. This
directly motivates EqOpt-K (bounded receding-horizon controller, Section 14.6)
as necessary future work -- not merely beneficial -- since it is required to
restore the model's primary lifetime-advantage claim at realistic network
scales, not just to improve a secondary equilibrium metric.

Follow-up required: re-run this experiment once EqOpt-K is designed and
implemented, to confirm the K-step extension restores a consistent HND/Psi_max
advantage across all baselines and all tested N.
