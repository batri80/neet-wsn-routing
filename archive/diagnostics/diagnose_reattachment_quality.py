"""
For local-trigger reattachment decisions specifically, compare WHICH
parent eqopt vs greedy chooses, and whether eqopt's choice concentrates
load onto nodes that are already low-energy or high-load -- checking for
a second, more subtle one-step myopia in reattachment TARGET selection
(as opposed to the reconnect/don't-reconnect myopia already fixed).
"""
import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M

def track_reattachment_quality(strategy, N=100, E0=1.0, R_c=35.0, max_rounds=400, seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    chosen_parent_energy_percentile = []
    chosen_parent_load_percentile = []

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
            from simulate import _mark_unreachable_as_disconnected
            _mark_unreachable_as_disconnected(net)

        if strategy != 'static' and newly_dead:
            for dn in newly_dead:
                for orph in net.orphans_of(dn):
                    if not net.alive[orph]:
                        continue
                    L_before, _ = net.loads()
                    E_before = net.E.copy()
                    ev, ncand = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph)
                    new_parent = ev['parent'][orph]
                    if new_parent >= 0 and new_parent != net.parent[orph]:
                        alive_idx = net.alive_connected()
                        if len(alive_idx) > 1:
                            e_rank = (E_before[alive_idx] < E_before[new_parent]).mean()
                            l_rank = (L_before[alive_idx] < L_before[new_parent]).mean()
                            chosen_parent_energy_percentile.append(e_rank)
                            chosen_parent_load_percentile.append(l_rank)
                    net.parent = ev['parent']
                    from simulate import _mark_unreachable_as_disconnected
                    _mark_unreachable_as_disconnected(net)
        if net.alive.sum() == 0:
            break

    return chosen_parent_energy_percentile, chosen_parent_load_percentile

for strat in ['greedy', 'eqopt']:
    all_e, all_l = [], []
    for seed in range(8):
        e, l = track_reattachment_quality(strat, seed=seed)
        all_e += e; all_l += l
    print(f"{strat:8s}: chosen-parent energy percentile (higher=picks stronger nodes) = {np.mean(all_e):.3f}")
    print(f"{'':8s}  chosen-parent load percentile (higher=picks already-busier nodes, BAD) = {np.mean(all_l):.3f}")
