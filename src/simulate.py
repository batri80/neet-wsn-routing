"""
simulate.py -- runs one NEET episode: normal traffic-driven depletion every
round, event-triggered reconfiguration (Algorithm 1) on Psi/R_h threshold
crossing or node death, until LND (last node dead) or a round cap.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import metrics as M
import controller as C


def run_episode(N=100, E0=0.5, R_c=30.0, sink_pos=None, strategy='eqopt',
                 beta=0.5, Psi_c=None, Rc_thresh=None, max_rounds=4000,
                 seed=0, w=None, hole_delta_h=0.3, record_series=True,
                 min_improvement_delta=0.0, K=None, cooldown_rounds=None, alpha=0.6, cost_tolerance=0.15):
    """
    Returns a dict of episode results: FND, HND, LND, AUC(Psi), AUC(Psi_N),
    Psi_max, R_h_mean, hole_events, partition events, reconfig log, and
    (optionally) full Psi(t)/R_h(t) time series.
    """
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, sink_pos=sink_pos, E0=E0, rng=rng, w=w)

    # Empirical Psi_c and R_c, calibrated separately per controller
    # (Section 14.6/14.9 of the model reference doc -- one-step EqOpt and
    # EqOpt-K operate in genuinely different Psi/R_h regimes, so a single
    # shared default would silently misconfigure whichever strategy
    # didn't match it -- this is the same class of silent-mismatch risk
    # documented for R_c/comm-range in Section 14.9).
    if Psi_c is None:
        Psi_c = 0.045 if strategy == 'eqopt_k' else 0.065
    if Rc_thresh is None:
        Rc_thresh = 0.400 if strategy == 'eqopt_k' else 0.664
    if cooldown_rounds is None:
        # Debounce global re-triggering (Section 14.9/14.10 of the model
        # reference doc): without this, a controller whose Psi_c is
        # tuned aggressively (as eqopt_k's is, 0.045 vs one-step's
        # 0.065) can re-trigger on nearly every round over a long
        # episode, since the topology has no time to settle before
        # being re-evaluated -- observed directly causing 2967
        # reconfigurations in a single 3000-round episode. Tied to K for
        # eqopt_k (half the projection horizon -- give the topology at
        # least that long to show its effect before re-checking);
        # disabled (0) for all other strategies, unchanged behavior.
        cooldown_rounds = (K // 2) if (strategy == 'eqopt_k' and K) else 0
    cooldown_remaining = 0

    series_Psi, series_PsiN, series_Rh, series_alive = [], [], [], []
    fnd = hnd = lnd = None
    hole_events = 0
    partition_events = 0
    reconfig_log = []  # (round, trigger_type, |T(t)|, dPsi)
    null_with_alt_local = 0
    total_with_alt_local = 0
    null_with_alt_global = 0
    total_with_alt_global = 0
    n0 = N

    # Running summary stats, tracked incrementally so they are correct
    # even when record_series=False (needed for large sweeps).
    sum_Psi = sum_PsiN = sum_Rh = 0.0
    max_Psi = 0.0
    n_rounds_counted = 0

    prev_alive_count = N
    for t in range(max_rounds):
        connected = net.alive_connected()
        if len(connected) == 0:
            lnd = t
            break

        # --- normal traffic-driven depletion ---
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        dead_this_round = False
        for i in newly_dead:
            net.kill(i)
            dead_this_round = True

        # Disconnection bookkeeping must happen for EVERY strategy
        # (including static) -- a dead relay breaks its descendants'
        # path to the sink regardless of whether the controller attempts
        # repair. Previously this only ran inside _apply_evaluation(),
        # which static never calls, so orphaned nodes under static kept
        # computing transmission cost toward a dead parent forever and
        # were never excluded from alive_connected() -- silently
        # inflating static's HND and distorting its Psi_max.
        if dead_this_round:
            _mark_unreachable_as_disconnected(net)

        m = M.compute_all(net, beta=beta)
        Psi_t, PsiN_t, Rh_t = m['Psi'], m['Psi_N'], m['R_h']

        sum_Psi += Psi_t; sum_PsiN += PsiN_t; sum_Rh += Rh_t
        max_Psi = max(max_Psi, Psi_t)
        n_rounds_counted += 1

        # Hole detection: risk crosses empirical threshold
        if Rh_t >= Rc_thresh:
            hole_events += 1

        alive_count = int((net.alive & ~net.disconnected).sum())
        if fnd is None and alive_count <= n0 - 1 and prev_alive_count > alive_count:
            fnd = t
        if hnd is None and alive_count <= n0 / 2:
            hnd = t
        prev_alive_count = alive_count

        if record_series:
            series_Psi.append(Psi_t); series_PsiN.append(PsiN_t)
            series_Rh.append(Rh_t); series_alive.append(alive_count)

        # --- local trigger: handle each newly dead node's orphans, one
        # orphan at a time (avoids the combinatorial blowup of jointly
        # reassigning multiple orphans at once) ---
        if strategy != 'static' and dead_this_round:
            for dn in newly_dead:
                orphans = net.orphans_of(dn)
                for orph in orphans:
                    if not net.alive[orph]:
                        continue
                    pre_partition = net.disconnected.sum()
                    prev_parent_orph = net.parent[orph]
                    ev, ncand = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph, K=K, alpha=alpha, cost_tolerance=cost_tolerance)
                    null_ev = C.evaluate_diff(net, {}, beta)
                    dPsi = ev['Psi'] - null_ev['Psi']
                    if ncand > 1:
                        total_with_alt_local += 1
                        if ev['parent'][orph] == prev_parent_orph:
                            null_with_alt_local += 1
                    _apply_evaluation(net, ev)
                    reconfig_log.append((t, 'local', ncand, dPsi))
                    if net.disconnected.sum() > pre_partition:
                        partition_events += 1

        # --- global trigger (subject to cooldown, see initialization) ---
        elif (strategy != 'static' and cooldown_remaining == 0
              and (Psi_t > Psi_c or Rh_t > Rc_thresh)):
            ev, ncand = C.reconfigure(net, strategy, 'global', beta, rng, K=K, alpha=alpha, cost_tolerance=cost_tolerance)
            null_ev = C.evaluate_diff(net, {}, beta)
            dPsi = ev['Psi'] - null_ev['Psi']
            if ncand > 1:
                total_with_alt_global += 1
                if ev['parent'] is net.parent or (ev['parent'] == net.parent).all():
                    null_with_alt_global += 1
            if -dPsi > min_improvement_delta:   # improvement clears the margin -> act
                _apply_evaluation(net, ev)
            else:                                 # not worth disturbing the tree
                _apply_evaluation(net, null_ev)
            reconfig_log.append((t, 'global', ncand, dPsi))
            cooldown_remaining = cooldown_rounds

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if alive_count == 0:
            lnd = t
            break

    if lnd is None:
        lnd = max_rounds
    if fnd is None:
        fnd = lnd
    if hnd is None:
        hnd = lnd

    series_Psi = np.array(series_Psi); series_PsiN = np.array(series_PsiN)
    series_Rh = np.array(series_Rh)

    result = dict(
        FND=fnd, HND=hnd, LND=lnd,
        AUC_Psi=float(sum_Psi),      # trapezoid ~ sum for unit round-spacing
        AUC_PsiN=float(sum_PsiN),
        Psi_max=float(max_Psi),
        Rh_mean=float(sum_Rh / n_rounds_counted) if n_rounds_counted else 0.0,
        hole_events=hole_events,
        partition_events=partition_events,
        n_reconfigs=len(reconfig_log),
        n_local=sum(1 for r in reconfig_log if r[1] == 'local'),
        n_global=sum(1 for r in reconfig_log if r[1] == 'global'),
        mean_dPsi_per_reconfig=float(np.mean([r[3] for r in reconfig_log])) if reconfig_log else 0.0,
        hole_rate=float(hole_events / n_rounds_counted) if n_rounds_counted else 0.0,
        AUC_Psi_per_round=float(sum_Psi / n_rounds_counted) if n_rounds_counted else 0.0,
        AUC_PsiN_per_round=float(sum_PsiN / n_rounds_counted) if n_rounds_counted else 0.0,
        null_with_alt_local=null_with_alt_local,
        total_with_alt_local=total_with_alt_local,
        null_with_alt_global=null_with_alt_global,
        total_with_alt_global=total_with_alt_global,
    )
    if record_series:
        result['series'] = dict(Psi=series_Psi, Psi_N=series_PsiN, Rh=series_Rh,
                                 alive=np.array(series_alive))
    return result


def _apply_evaluation(net, ev):
    # IMPORTANT: only adopt the candidate's TOPOLOGY, not its projected
    # energy. ev['E_plus'] is a hypothetical one-round-ahead projection
    # used purely to compare candidates (Theorem 1's drift comparison);
    # writing it back into net.E would double-charge energy this round
    # (once for the real depletion already applied above, again here).
    # The real depletion under the new tree happens naturally next round.
    net.parent = ev['parent']
    _mark_unreachable_as_disconnected(net)


def _mark_unreachable_as_disconnected(net):
    """After a parent-array edit, mark any alive node whose parent chain
    does not terminate at the sink (-1) within N steps as disconnected."""
    alive_idx = np.where(net.alive)[0]
    for i in alive_idx:
        cur = i
        steps = 0
        reached = False
        while steps <= net.N:
            p = net.parent[cur]
            if p == -1:
                reached = True
                break
            if p == -2 or p not in set(alive_idx.tolist()):
                break
            cur = p
            steps += 1
        net.disconnected[i] = not reached
