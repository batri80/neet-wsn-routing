"""
protocols.py -- LEACH, PEGASIS, HEED baselines for Experiment F.

Unlike v1 (single-source, one-shot comparison undermined by Remark 4),
every node here generates w_i traffic every round -- the same multi-source
assumption EqOpt uses -- so cross-protocol comparison of the NEET
observables (Psi, Psi_N, R_h) is now meaningful. Relay load L_i is
generalized per-protocol: cluster-head load = weighted member traffic for
LEACH/HEED; chain-forwarding load for PEGASIS.

These are standard-textbook simplified implementations (2-tier member/CH/
sink for LEACH & HEED; single rotating-leader chain for PEGASIS), not
full protocol-stack faithful reproductions -- adequate for the purpose
stated in the model doc: demonstrating Psi/R_h provide a common analytical
language across architecturally distinct strategies, not benchmarking
routing-protocol engineering quality.
"""
import numpy as np
from energy import etx, erx
import metrics as M

E_DA = 5e-9  # J/bit/signal, data-aggregation cost at cluster heads


def _init_state(N, L, sink, E0, rng, w):
    pos = rng.uniform(0, L, size=(N, 2))
    E = np.full(N, E0, dtype=float)
    alive = np.ones(N, dtype=bool)
    return pos, E, alive


def _psi_series_step(E, L_load, alive_mask, beta=0.5):
    Psi, Ebar = M.psi(E)
    Psin = M.psi_n(Psi, Ebar)
    Ch, Lambda = M.structural_covariance(E, L_load, alive_mask)
    Rh = M.risk(Psin, Lambda, beta)
    return Psi, Psin, Rh


def _summ(fnd, hnd, lnd, sum_psi, sum_psin, sum_rh, max_psi, n_rounds, holes, n0):
    return dict(FND=fnd if fnd is not None else lnd, HND=hnd if hnd is not None else lnd,
                LND=lnd, AUC_Psi=float(sum_psi), AUC_PsiN=float(sum_psin),
                Psi_max=float(max_psi), Rh_mean=float(sum_rh / n_rounds) if n_rounds else 0.0,
                hole_events=holes)


# ----------------------------------------------------------------------
def run_leach_episode(N=100, E0=1.0, L=100.0, sink_pos=None, p=0.05,
                       max_rounds=2000, seed=0, w=None, beta=0.5,
                       Rc_thresh=0.82):
    rng = np.random.default_rng(seed)
    sink = np.array(sink_pos if sink_pos is not None else [L / 2, L / 2])
    w = np.ones(N) if w is None else np.asarray(w, float)
    pos, E, alive = _init_state(N, L, sink, E0, rng, w)

    fnd = hnd = lnd = None
    sum_psi = sum_psin = sum_rh = max_psi = 0.0
    n_rounds = 0; holes = 0
    n0 = N
    round_in_epoch = 0
    prev_alive = N
    already_ch_this_epoch = np.zeros(N, dtype=bool)

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            lnd = t; break

        # CH election (standard LEACH threshold). BUG FIX: the
        # threshold formula correctly escalates toward 1.0 near the
        # end of each epoch (intended LEACH behavior), but without
        # excluding nodes that already served as CH earlier in the
        # SAME epoch, this caused every node to become a CH
        # simultaneously in late-epoch rounds -- an unintended,
        # cost-skipping degenerate state that inflated LEACH's
        # measured performance. Fixed by tracking and excluding
        # already-served nodes within each epoch, resetting at the
        # start of every new epoch.
        if round_in_epoch == 0:
            already_ch_this_epoch[:] = False
        r = round_in_epoch % int(round(1.0 / p))
        thresh = p / (1 - p * r) if (1 - p * r) > 0 else 1.0
        is_ch = np.zeros(N, dtype=bool)
        for i in alive_idx:
            if already_ch_this_epoch[i]:
                continue
            if rng.random() < thresh:
                is_ch[i] = True
        if not is_ch[alive_idx].any():
            eligible = [i for i in alive_idx if not already_ch_this_epoch[i]]
            if eligible:
                is_ch[rng.choice(eligible)] = True
            else:
                is_ch[rng.choice(alive_idx)] = True
        already_ch_this_epoch[is_ch] = True
        round_in_epoch = (round_in_epoch + 1) % int(round(1.0 / p))

        ch_idx = alive_idx[is_ch[alive_idx]]
        member_idx = alive_idx[~is_ch[alive_idx]]

        # assign members to nearest CH
        assign = {}
        for m_ in member_idx:
            d = np.linalg.norm(pos[ch_idx] - pos[m_], axis=1)
            assign[m_] = ch_idx[np.argmin(d)]

        cost = np.zeros(N)
        L_load = np.zeros(N)
        for m_, ch in assign.items():
            d = np.linalg.norm(pos[m_] - pos[ch])
            cost[m_] += w[m_] * etx(d)
            cost[ch] += w[m_] * erx()
            L_load[ch] += w[m_]
        for ch in ch_idx:
            d_sink = np.linalg.norm(pos[ch] - sink)
            # Fair (iso-morphic) cost: charge for own traffic PLUS full
            # relayed member volume, no aggregation discount.
            cost[ch] += (w[ch] + L_load[ch]) * etx(d_sink)

        E = np.clip(E - cost, 0.0, None)
        newly_dead = alive_idx[E[alive_idx] <= 0.0]
        alive[newly_dead] = False

        Psi_t, Psin_t, Rh_t = _psi_series_step(E, L_load, alive, beta)
        sum_psi += Psi_t; sum_psin += Psin_t; sum_rh += Rh_t
        max_psi = max(max_psi, Psi_t); n_rounds += 1
        if Rh_t >= Rc_thresh:
            holes += 1

        na = int(alive.sum())
        if fnd is None and na < prev_alive:
            fnd = t
        if hnd is None and na <= n0 / 2:
            hnd = t
        prev_alive = na
        if na == 0:
            lnd = t; break

    if lnd is None:
        lnd = max_rounds
    return _summ(fnd, hnd, lnd, sum_psi, sum_psin, sum_rh, max_psi, n_rounds, holes, n0)


# ----------------------------------------------------------------------
def run_heed_episode(N=100, E0=1.0, L=100.0, sink_pos=None, c_prob=0.05,
                      max_rounds=2000, seed=0, w=None, beta=0.5,
                      Rc_thresh=0.82):
    """HEED: CH election probability proportional to residual energy."""
    rng = np.random.default_rng(seed)
    sink = np.array(sink_pos if sink_pos is not None else [L / 2, L / 2])
    w = np.ones(N) if w is None else np.asarray(w, float)
    pos, E, alive = _init_state(N, L, sink, E0, rng, w)

    fnd = hnd = lnd = None
    sum_psi = sum_psin = sum_rh = max_psi = 0.0
    n_rounds = 0; holes = 0
    n0 = N
    prev_alive = N

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            lnd = t; break

        ch_prob_i = c_prob * (E[alive_idx] / (E0 + 1e-12))
        draws = rng.random(len(alive_idx))
        is_ch_mask = draws < np.clip(ch_prob_i, 0, 1)
        ch_idx = alive_idx[is_ch_mask]
        if len(ch_idx) == 0:
            ch_idx = np.array([alive_idx[np.argmax(E[alive_idx])]])
        member_idx = np.array([i for i in alive_idx if i not in set(ch_idx.tolist())])

        assign = {}
        for m_ in member_idx:
            d = np.linalg.norm(pos[ch_idx] - pos[m_], axis=1)
            assign[m_] = ch_idx[np.argmin(d)]

        cost = np.zeros(N)
        L_load = np.zeros(N)
        for m_, ch in assign.items():
            d = np.linalg.norm(pos[m_] - pos[ch])
            cost[m_] += w[m_] * etx(d)
            cost[ch] += w[m_] * erx()
            L_load[ch] += w[m_]
        for ch in ch_idx:
            d_sink = np.linalg.norm(pos[ch] - sink)
            cost[ch] += (w[ch] + L_load[ch]) * etx(d_sink)

        E = np.clip(E - cost, 0.0, None)
        newly_dead = alive_idx[E[alive_idx] <= 0.0]
        alive[newly_dead] = False

        Psi_t, Psin_t, Rh_t = _psi_series_step(E, L_load, alive, beta)
        sum_psi += Psi_t; sum_psin += Psin_t; sum_rh += Rh_t
        max_psi = max(max_psi, Psi_t); n_rounds += 1
        if Rh_t >= Rc_thresh:
            holes += 1

        na = int(alive.sum())
        if fnd is None and na < prev_alive:
            fnd = t
        if hnd is None and na <= n0 / 2:
            hnd = t
        prev_alive = na
        if na == 0:
            lnd = t; break

    if lnd is None:
        lnd = max_rounds
    return _summ(fnd, hnd, lnd, sum_psi, sum_psin, sum_rh, max_psi, n_rounds, holes, n0)


# ----------------------------------------------------------------------
def _build_chain(alive_idx, pos, sink):
    """Greedy nearest-neighbor chain starting from the node farthest from sink."""
    if len(alive_idx) == 0:
        return []
    d_sink = np.linalg.norm(pos[alive_idx] - sink, axis=1)
    start = alive_idx[np.argmax(d_sink)]
    remaining = set(alive_idx.tolist()) - {start}
    chain = [start]
    cur = start
    while remaining:
        rem_arr = np.array(list(remaining))
        d = np.linalg.norm(pos[rem_arr] - pos[cur], axis=1)
        nxt = rem_arr[np.argmin(d)]
        chain.append(int(nxt))
        remaining.remove(int(nxt))
        cur = nxt
    return chain


def run_pegasis_episode(N=100, E0=1.0, L=100.0, sink_pos=None,
                         max_rounds=2000, seed=0, w=None, beta=0.5,
                         Rc_thresh=0.82, rebuild_frac=0.2):
    rng = np.random.default_rng(seed)
    sink = np.array(sink_pos if sink_pos is not None else [L / 2, L / 2])
    w = np.ones(N) if w is None else np.asarray(w, float)
    pos, E, alive = _init_state(N, L, sink, E0, rng, w)

    fnd = hnd = lnd = None
    sum_psi = sum_psin = sum_rh = max_psi = 0.0
    n_rounds = 0; holes = 0
    n0 = N
    prev_alive = N
    chain = _build_chain(np.where(alive)[0], pos, sink)
    n_at_last_build = len(chain)
    leader_ptr = 0

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            lnd = t; break

        chain = [c for c in chain if alive[c]]
        if n_at_last_build == 0 or len(chain) < max(1, int((1 - rebuild_frac) * n_at_last_build)):
            chain = _build_chain(alive_idx, pos, sink)
            n_at_last_build = len(chain)
            leader_ptr = 0
        if len(chain) == 0:
            lnd = t; break

        leader = chain[leader_ptr % len(chain)]
        leader_ptr += 1

        # Fair (iso-morphic) cost: full accumulated relay volume, no
        # aggregation discount -- each node forwards its own traffic
        # plus everything accumulated from further down its side of
        # the chain, at full volume.
        cost = np.zeros(N)
        L_load = np.zeros(N)
        li = chain.index(leader)
        accumulated = 0.0
        for idx in range(0, li):
            src, dst = chain[idx], chain[idx + 1]
            total_out = w[src] + accumulated
            d = np.linalg.norm(pos[src] - pos[dst])
            cost[src] += total_out * etx(d)
            cost[dst] += total_out * erx()
            L_load[dst] += total_out
            accumulated = total_out
        left_total = accumulated
        accumulated = 0.0
        for idx in range(len(chain) - 1, li, -1):
            src, dst = chain[idx], chain[idx - 1]
            total_out = w[src] + accumulated
            d = np.linalg.norm(pos[src] - pos[dst])
            cost[src] += total_out * etx(d)
            cost[dst] += total_out * erx()
            L_load[dst] += total_out
            accumulated = total_out
        right_total = accumulated
        d_sink = np.linalg.norm(pos[leader] - sink)
        leader_total_out = w[leader] + left_total + right_total
        cost[leader] += leader_total_out * etx(d_sink)

        E = np.clip(E - cost, 0.0, None)
        newly_dead = alive_idx[E[alive_idx] <= 0.0]
        alive[newly_dead] = False

        Psi_t, Psin_t, Rh_t = _psi_series_step(E, L_load, alive, beta)
        sum_psi += Psi_t; sum_psin += Psin_t; sum_rh += Rh_t
        max_psi = max(max_psi, Psi_t); n_rounds += 1
        if Rh_t >= Rc_thresh:
            holes += 1

        na = int(alive.sum())
        if fnd is None and na < prev_alive:
            fnd = t
        if hnd is None and na <= n0 / 2:
            hnd = t
        prev_alive = na
        if na == 0:
            lnd = t; break

    if lnd is None:
        lnd = max_rounds
    return _summ(fnd, hnd, lnd, sum_psi, sum_psin, sum_rh, max_psi, n_rounds, holes, n0)


PROTOCOL_FUNCS = {
    'LEACH': run_leach_episode,
    'HEED': run_heed_episode,
    'PEGASIS': run_pegasis_episode,
}
