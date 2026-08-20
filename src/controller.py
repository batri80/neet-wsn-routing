"""
controller.py -- EqOpt v2 and baseline reconfiguration strategies.

Implements Algorithm 1 (event-triggered topology selection) from the model
reference doc, plus four comparison strategies operating on the SAME
candidate-tree mechanism (only the selection rule differs), analogous to
the original Table I role.

Strategies:
    static   -- never reconfigures (uncontrolled baseline)
    random   -- picks a uniformly random candidate (incl. null) on trigger
    greedy   -- picks the candidate maximizing minimum post-txn residual
                energy among affected relays (energy-greedy heuristic)
    mincost  -- picks the candidate minimizing total analytical epoch cost
                (distance-aware, ignores equilibrium)
    eqopt    -- NEET-optimal: minimizes Psi(E+(T))  [Algorithm 1]

Candidate generation:
    local trigger  (a specific node died): Delta(t) = orphan roots of the
        dead node; full/near-full enumeration via cartesian product
        (capped) since this set is small.
    global trigger (Psi/R_h threshold crossed, no specific death): Delta(t)
        = all alive nodes is intractable to enumerate; implemented as a
        greedy coordinate-descent over the top-K highest-load nodes, each
        considering in-range reattachment (this is the practical
        realization of "explore candidates from a large Delta(t) without
        full combinatorial search," explicitly noted in the model doc).
"""
import numpy as np
from energy import epoch_cost
import metrics as M


MAX_LOCAL_PARENTS = 6      # cap on alternate parents offered per orphan
GLOBAL_TOP_K_FRAC = 0.05   # fraction of alive nodes considered per global trigger
GLOBAL_TOP_K_MIN = 5       # floor, so small N still gets a reasonable candidate pool
GLOBAL_MAX_PARENTS_PER_NODE = 4  # cap on alternate parents offered per node

def _global_top_k(n_connected):
    return max(GLOBAL_TOP_K_MIN, int(GLOBAL_TOP_K_FRAC * n_connected))


def _null_diff():
    return {}


def _apply_diff(net, diff):
    parent2 = net.parent.copy()
    for node, newp in diff.items():
        parent2[node] = newp
    return parent2


def evaluate_diff(net, diff, beta=0.5):
    """Analytically evaluate a candidate parent-diff: returns dict of
    resulting Psi, E_plus, cost, L, parent array -- no simulation."""
    parent2 = _apply_diff(net, diff)
    cost, L2 = epoch_cost(net, parent_override=parent2)
    E_plus = np.clip(net.E - cost, 0.0, None)
    Psi_val, Ebar = M.psi(E_plus)
    return dict(Psi=Psi_val, Ebar=Ebar, E_plus=E_plus, cost=cost, L=L2, parent=parent2)


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------
def generate_local_candidates(net, orphan):
    """
    Candidate reattachment options for a SINGLE orphan (one node whose
    parent just died).

    IMPORTANT: connectivity restoration is treated as a precondition, not
    a Psi-tradeoff. If a valid in-range reattachment exists, the null
    (stay-disconnected) candidate is EXCLUDED from the set entirely --
    otherwise a one-step-greedy strategy (EqOpt) will systematically
    prefer leaving a reconnectable node disconnected whenever reattaching
    it would concentrate its subtree's relay burden onto one new parent,
    since that looks like a smaller ONE-STEP Psi increase than it actually
    is. Empirically this caused EqOpt to leave orphans disconnected 82-85%
    of the time an alternative existed, driving up long-run Psi via
    frozen-energy partition pockets -- the same mechanism documented for
    the 'static' baseline, just emerging locally. Null is only offered
    when NO feasible reattachment exists (genuine unavoidable partition).
    """
    subtree = net.subtree_of(orphan) | {orphan}
    parents = net.candidate_parents_for(orphan, exclude_subtree=subtree)
    if len(parents) == 0:
        return [_null_diff()]  # genuinely unavoidable -- no candidate can reconnect it
    candidates = []
    for p in parents[:MAX_LOCAL_PARENTS]:
        candidates.append({int(orphan): int(p)})
    return candidates


def generate_global_candidates(net, beta=0.5):
    """
    Structural, strategy-agnostic candidate generation for global triggers:
    for each of the top-K highest-load nodes, offer reattachment to each
    in-range alternate parent (subject to the depth-change axiom), as a
    single pass -- NOT a Psi-driven iterative search. Selection among
    these candidates is left entirely to the calling strategy (eqopt,
    random, greedy, mincost), so all four pay the same generation cost
    and only differ in which candidate they pick.
    """
    L, _ = net.loads()
    connected = net.alive_connected()
    if len(connected) == 0:
        return [_null_diff()]

    k = _global_top_k(len(connected))
    order = connected[np.argsort(-L[connected])][:k]
    candidates = [_null_diff()]
    for node in order:
        subtree = net.subtree_of(node) | {node}
        parents = net.candidate_parents_for(node, exclude_subtree=subtree)
        for p in parents[:GLOBAL_MAX_PARENTS_PER_NODE]:
            candidates.append({int(node): int(p)})
    return candidates


# ----------------------------------------------------------------------
# Strategy selection rules (all operate over the same candidate list)
# ----------------------------------------------------------------------
def select_eqopt(net, candidates, beta, rng):
    evals = [evaluate_diff(net, c, beta) for c in candidates]
    best = min(evals, key=lambda e: e['Psi'])
    return best


def select_random(net, candidates, beta, rng):
    c = candidates[rng.integers(len(candidates))]
    return evaluate_diff(net, c, beta)


def select_greedy(net, candidates, beta, rng):
    """Maximize the minimum post-transmission residual energy among
    nodes touched by the candidate (energy-greedy heuristic)."""
    best_ev, best_score = None, -np.inf
    for c in candidates:
        ev = evaluate_diff(net, c, beta)
        touched = list(c.keys())
        if touched:
            score = ev['E_plus'][touched].min()
        else:
            score = ev['E_plus'][net.alive_connected()].min() if len(net.alive_connected()) else 0.0
        if score > best_score:
            best_score, best_ev = score, ev
    return best_ev


def select_mincost(net, candidates, beta, rng):
    """Minimize total analytical epoch cost (distance-aware, equilibrium-blind)."""
    best_ev, best_cost = None, np.inf
    for c in candidates:
        ev = evaluate_diff(net, c, beta)
        total_cost = ev['cost'].sum()
        if total_cost < best_cost:
            best_cost, best_ev = total_cost, ev
    return best_ev


STRATEGY_FUNCS = {
    'eqopt': select_eqopt,
    'random': select_random,
    'greedy': select_greedy,
    'mincost': select_mincost,
}


def reconfigure(net, strategy, trigger, beta, rng, orphan=None, K=None, alpha=0.6, cost_tolerance=0.15):
    """
    trigger: 'local' (requires orphan=node id) or 'global'.
    strategy: one of STRATEGY_FUNCS keys.
    K: horizon length for 'eqopt_k'; ignored otherwise.
    alpha: hybrid weight for 'eqopt_h'; ignored otherwise.
    Returns (chosen evaluation dict, number of candidates considered).
    """
    if trigger == 'local':
        candidates = generate_local_candidates(net, orphan)
    else:
        candidates = generate_global_candidates(net, beta)
    if strategy == 'eqopt_k':
        return STRATEGY_FUNCS[strategy](net, candidates, beta, rng, K=K), len(candidates)
    if strategy == 'eqopt_h':
        return STRATEGY_FUNCS[strategy](net, candidates, beta, rng, cost_tolerance=cost_tolerance), len(candidates)
    return STRATEGY_FUNCS[strategy](net, candidates, beta, rng), len(candidates)


def _light_clone(net):
    """Cheap clone for K-step projection: shares immutable arrays
    (pos, sink, w) by reference, copies only the mutable state
    (E, alive, disconnected, parent, depth). Avoids full deepcopy
    overhead since this runs once per candidate per trigger event."""
    shadow = object.__new__(_Network)
    shadow.N = net.N
    shadow.L = net.L
    shadow.R_c = net.R_c
    shadow.rng = net.rng
    shadow.sink = net.sink
    shadow.pos = net.pos
    shadow.w = net.w
    shadow.E0 = net.E0
    shadow.E = net.E.copy()
    shadow.alive = net.alive.copy()
    shadow.disconnected = net.disconnected.copy()
    shadow.parent = net.parent.copy()
    shadow.depth = net.depth.copy()
    return shadow


def project_k_rounds(net, parent_after_diff, K, beta, max_cascade_reattachments=8):
    """
    Simulate K rounds of ordinary depletion under a fixed candidate
    topology (parent_after_diff), starting from net's current state.
    No DELIBERATE (Psi-driven, global) reconfiguration occurs within the
    horizon -- that is what this projection evaluates. Mandatory
    connectivity restoration (Section 14.2) still applies within the
    horizon, since it is a precondition, not a strategic choice.

    CASCADE CAP (Section 14.9 of the model reference doc): a death
    partway through the projection can orphan a subtree, whose repair
    can itself precipitate further deaths later in the same projection,
    compounding. Left unbounded, this was found to occasionally blow up
    projection cost by 18x or more on specific seeds (K=50 vs K=40,
    same seed), and at full episode length (~3000 rounds, vs the 362
    used for calibration) was observed to cause multi-hour single-
    episode runtimes. max_cascade_reattachments caps the number of
    local reattachments performed WITHIN a single projection call; once
    the cap is hit, any further orphans in that projection are simply
    left disconnected for the remainder of the horizon (matching the
    'no feasible reattachment' fallback already defined in Section 14.2
    -- this is a legitimate degraded-but-bounded outcome for that
    projection, not a new rule). This bounds worst-case projection cost
    to O(max_cascade_reattachments) local repairs regardless of how
    many deaths actually occur within the horizon.

    Returns the final Psi at round K (or at natural termination if the
    network empties before K rounds elapse).
    """
    shadow = _light_clone(net)
    shadow.parent = parent_after_diff.copy()
    _mark_disc_fn(shadow)

    cascade_count = 0
    for _ in range(K):
        connected = shadow.alive_connected()
        if len(connected) == 0:
            break
        cost, L = epoch_cost(shadow)
        shadow.E = np.clip(shadow.E - cost, 0.0, None)
        newly_dead = [i for i in connected if shadow.E[i] <= 0.0]
        for i in newly_dead:
            shadow.kill(i)
        if newly_dead:
            _mark_disc_fn(shadow)
            for dn in newly_dead:
                for orph in shadow.orphans_of(dn):
                    if not shadow.alive[orph]:
                        continue
                    if cascade_count >= max_cascade_reattachments:
                        continue  # cap hit -- leave remaining orphans disconnected this projection
                    local_cands = generate_local_candidates(shadow, orph)
                    local_evals = [evaluate_diff(shadow, c, beta) for c in local_cands]
                    local_best = local_evals[int(np.argmin([e["Psi"] for e in local_evals]))]
                    shadow.parent = local_best["parent"]
                    _mark_disc_fn(shadow)
                    cascade_count += 1

    Psi_final, _ = M.psi(shadow.E)
    return Psi_final


def _mark_disc_fn(net):
    """Local import wrapper to avoid a circular import at module load
    time (simulate.py imports controller.py)."""
    from simulate import _mark_unreachable_as_disconnected
    _mark_unreachable_as_disconnected(net)


def select_eqopt_k(net, candidates, beta, rng, K=None):
    """K-step analogue of select_eqopt: choose the candidate whose
    K-round-projected Psi, MINUS the null candidate's K-round-projected
    Psi (same horizon, same starting state), is lowest -- drift relative
    to null, extending Theorem 1's comparison structure to K steps,
    rather than comparing raw absolute K-round Psi. Absolute comparison
    was found empirically to underperform even one-step EqOpt: over a
    K-round horizon every candidate's Psi rises from shared background
    depletion common to all candidates, which swamps the much smaller
    candidate-specific structural signal the projection is meant to
    isolate. Subtracting the null baseline cancels that common trend.
    Falls back to DEFAULT_K if K not supplied."""
    k = K if K is not None else DEFAULT_K
    null_ev = evaluate_diff(net, {}, beta)
    null_k_psi = project_k_rounds(net, null_ev['parent'], k, beta)

    best_ev, best_k_drift = None, np.inf
    for c in candidates:
        ev = evaluate_diff(net, c, beta)
        k_psi = project_k_rounds(net, ev['parent'], k, beta)
        k_drift = k_psi - null_k_psi
        if k_drift < best_k_drift:
            best_k_drift = k_drift
            best_ev = dict(ev)
            best_ev['k_psi'] = k_psi
            best_ev['k_drift'] = k_drift
    return best_ev


STRATEGY_FUNCS['eqopt_k'] = select_eqopt_k


# ----------------------------------------------------------------------
# EqOpt-H: Hybrid Cost-Equilibrium controller.
#
# Motivated by a direct empirical finding, not a guess: in Experiment A,
# 'mincost' -- a strategy fully blind to equilibrium, minimizing only raw
# transmission cost -- matched or beat both one-step EqOpt and EqOpt-K on
# HND at every tested N. This suggests pure Psi-greedy optimization
# (regardless of lookahead depth) systematically undervalues raw energy
# conservation. EqOpt-H blends both signals directly, using only the
# existing one-step evaluate_diff() output -- no K-round projection, so
# it is as cheap as one-step EqOpt.
# ----------------------------------------------------------------------

def select_eqopt_h(net, candidates, beta, rng, cost_tolerance=0.15):
    """
    Lexicographic hybrid (replaces an earlier linear-blend design that
    collapsed to mincost-like behavior across alpha in [0.3,0.7] --
    traced to min-max normalization treating a noise-level absolute
    Psi spread (~1e-6) as equally decisive as a substantively real
    cost spread once both are rescaled to [0,1]).

    Among candidates whose cost is within cost_tolerance (relative) of
    the cheapest candidate, select the one minimizing projected Psi.
    cost_tolerance=0 reduces to pure mincost; large tolerance reduces
    to pure one-step EqOpt.
    """
    evals = [evaluate_diff(net, c, beta) for c in candidates]
    total_costs = np.array([e["cost"].sum() for e in evals])

    min_cost = total_costs.min()
    within_budget = total_costs <= min_cost * (1 + cost_tolerance)
    eligible_idx = np.where(within_budget)[0]

    psi_vals = np.array([evals[i]["Psi"] for i in eligible_idx])
    best_local_idx = eligible_idx[int(np.argmin(psi_vals))]

    best_ev = dict(evals[best_local_idx])
    best_ev["n_eligible"] = len(eligible_idx)
    return best_ev


STRATEGY_FUNCS['eqopt_h'] = select_eqopt_h
