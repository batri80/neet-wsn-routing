import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
from simulate import _mark_unreachable_as_disconnected

def track(strategy, N=100, E0=1.0, R_c=35.0, max_rounds=400, seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    n_orphan_events = 0
    n_had_candidates = 0
    n_actually_reassigned = 0
    e_percentiles, l_percentiles = [], []

    for t in range(max_rounds):
        connected = net.alive_connected()
        if len(connected) == 0:
            break
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        for i in newly_dead:
            net.kill(i)
        if newly_dead:
            _mark_unreachable_as_disconnected(net)

        if strategy != 'static' and newly_dead:
            for dn in newly_dead:
                for orph in net.orphans_of(dn):
                    if not net.alive[orph]:
                        continue
                    n_orphan_events += 1
                    candidates = C.generate_local_candidates(net, orph)
                    if len(candidates) > 0 and any(len(c) > 0 for c in candidates):
                        n_had_candidates += 1
                    L_before, _ = net.loads()
                    E_before = net.E.copy()
                    old_parent = net.parent[orph]
                    ev, ncand = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph)
                    new_parent = ev['parent'][orph]
                    if new_parent != old_parent and new_parent >= 0:
                        n_actually_reassigned += 1
                        alive_idx = net.alive_connected()
                        if len(alive_idx) > 1 and new_parent in alive_idx:
                            e_percentiles.append((E_before[alive_idx] < E_before[new_parent]).mean())
                            l_percentiles.append((L_before[alive_idx] < L_before[new_parent]).mean())
                    net.parent = ev['parent']
                    _mark_unreachable_as_disconnected(net)
        if net.alive.sum() == 0:
            break
    return n_orphan_events, n_had_candidates, n_actually_reassigned, e_percentiles, l_percentiles

for strat in ['greedy', 'eqopt']:
    tot_orph = tot_cand = tot_reassign = 0
    all_e, all_l = [], []
    for seed in range(8):
        oe, hc, ar, e, l = track(strat, seed=seed)
        tot_orph += oe; tot_cand += hc; tot_reassign += ar
        all_e += e; all_l += l
    print(f"{strat:8s}: orphan_events={tot_orph}  had_feasible_candidates={tot_cand}  actually_reassigned={tot_reassign}")
    if all_e:
        print(f"{'':8s}  chosen-parent energy percentile={np.mean(all_e):.3f}  load percentile={np.mean(all_l):.3f}")
    else:
        print(f"{'':8s}  (no reassignments captured -- see counts above)")
