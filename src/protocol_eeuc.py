"""
protocol_eeuc.py -- EEUC (Energy-Efficient Unequal Clustering), the
modern-baseline comparison. Chosen over metaheuristic-based recent
protocols because EEUC has a single, closed-form, unambiguous
specification, and remains the active reference baseline that papers
published as recently as 2024-2026 compare themselves against.

Core mechanism: competition radius shrinks as a cluster-head
candidate's distance to the sink decreases, so cluster heads near the
sink form smaller clusters and retain spare relay capacity.

Contains TWO bug fixes found during testing:
  1. Epoch tracking (same class as the LEACH bug): candidates must
     exclude nodes already served as CH earlier in the same epoch.
  2. Relay selection: prefer the relay making the most progress
     toward the sink (smallest resulting distance), not the spatially
     nearest neighbor -- the latter creates long chains of small hops
     and inflates cost via repeated fixed per-transmission overhead.
"""
import numpy as np
from energy import etx, erx
import metrics as M


def _init_state(N, L, rng, E0):
    pos = rng.uniform(0, L, size=(N, 2))
    E = np.full(N, E0, dtype=float)
    alive = np.ones(N, dtype=bool)
    return pos, E, alive


def _psi_series_step(E, L_load, alive_mask, beta=1.0):
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


E_DA = 5e-9  # unused here, kept for interface consistency with protocols.py


def run_eeuc_episode(N=100, E0=1.0, L=100.0, sink_pos=None, R_c=35.0,
                      p_candidate=0.1, c_param=0.5,
                      max_rounds=2000, seed=0, w=None, beta=1.0,
                      Rc_thresh=0.82):
    """
    EEUC episode loop, matching run_leach_episode/run_heed_episode's
    interface and return-dict shape exactly.
    """
    rng = np.random.default_rng(seed)
    sink = np.array(sink_pos if sink_pos is not None else [L / 2, L / 2])
    w = np.ones(N) if w is None else np.asarray(w, float)
    pos, E, alive = _init_state(N, L, rng, E0)

    d_to_sink_all = np.linalg.norm(pos - sink, axis=1)
    d_max = float(d_to_sink_all.max())
    d_min = float(d_to_sink_all.min())
    d_range = max(d_max - d_min, 1e-9)

    fnd = hnd = lnd = None
    sum_psi = sum_psin = sum_rh = max_psi = 0.0
    n_rounds = 0; holes = 0
    n0 = N
    prev_alive = N
    round_in_epoch = 0
    already_ch_this_epoch = np.zeros(N, dtype=bool)

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            lnd = t; break

        # --- Step 1: candidate election (LEACH-style epoch threshold,
        # BUG FIX: exclude nodes already served as CH this epoch) ---
        if round_in_epoch == 0:
            already_ch_this_epoch[:] = False
        r = round_in_epoch % int(round(1.0 / p_candidate))
        thresh = p_candidate / (1 - p_candidate * r) if (1 - p_candidate * r) > 0 else 1.0
        is_candidate = np.zeros(N, dtype=bool)
        for i in alive_idx:
            if already_ch_this_epoch[i]:
                continue
            if rng.random() < thresh:
                is_candidate[i] = True
        if not is_candidate[alive_idx].any():
            eligible = [i for i in alive_idx if not already_ch_this_epoch[i]]
            if eligible:
                is_candidate[rng.choice(eligible)] = True
            else:
                is_candidate[rng.choice(alive_idx)] = True
        round_in_epoch = (round_in_epoch + 1) % int(round(1.0 / p_candidate))
        cand_idx = alive_idx[is_candidate[alive_idx]]

        # --- Step 2: unequal competition radius per candidate ---
        d_cand_sink = np.linalg.norm(pos[cand_idx] - sink, axis=1)
        R_comp = (1.0 - c_param * (d_max - d_cand_sink) / d_range) * R_c
        R_comp = np.clip(R_comp, 0.1 * R_c, R_c)

        # --- Step 3: competition-radius-based suppression ---
        surviving = np.ones(len(cand_idx), dtype=bool)
        for a in range(len(cand_idx)):
            if not surviving[a]:
                continue
            i = cand_idx[a]
            for b in range(len(cand_idx)):
                if a == b or not surviving[b]:
                    continue
                j = cand_idx[b]
                d_ij = np.linalg.norm(pos[i] - pos[j])
                if d_ij <= min(R_comp[a], R_comp[b]):
                    if E[j] > E[i]:
                        surviving[a] = False
                        break
                    elif E[j] == E[i] and j < i:
                        surviving[a] = False
                        break
        ch_idx = cand_idx[surviving]
        if len(ch_idx) == 0:
            ch_idx = np.array([alive_idx[np.argmax(E[alive_idx])]])

        # --- Step 4: cluster formation -- members join nearest CH ---
        ch_set = set(ch_idx.tolist())
        member_idx = np.array([i for i in alive_idx if i not in ch_set])
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

        # --- Step 5: inter-cluster routing ---
        ch_to_sink_dist = {ch: np.linalg.norm(pos[ch] - sink) for ch in ch_idx}
        ch_total_out = {ch: w[ch] + L_load[ch] for ch in ch_idx}

        order = sorted(ch_idx.tolist(), key=lambda c: -ch_to_sink_dist[c])
        relay_accum = {ch: 0.0 for ch in ch_idx}

        for ch in order:
            total_out = ch_total_out[ch] + relay_accum[ch]
            own_dist = ch_to_sink_dist[ch]
            # Relay selection FIX: prefer relay making the most
            # progress toward the sink, not the spatially nearest.
            best_relay, best_relay_dist = None, None
            for other in ch_idx:
                if other == ch:
                    continue
                if ch_to_sink_dist[other] >= own_dist:
                    continue
                d = np.linalg.norm(pos[ch] - pos[other])
                if d <= R_c:
                    if best_relay_dist is None or ch_to_sink_dist[other] < best_relay_dist:
                        best_relay_dist = ch_to_sink_dist[other]; best_relay = other
            best_d = np.linalg.norm(pos[ch] - pos[best_relay]) if best_relay is not None else None
            if best_relay is not None:
                cost[ch] += total_out * etx(best_d)
                cost[best_relay] += total_out * erx()
                relay_accum[best_relay] += total_out
            else:
                cost[ch] += total_out * etx(own_dist)

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
