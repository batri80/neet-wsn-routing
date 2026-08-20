"""
Does EqOpt's candidate selection systematically pick FARTHER parents than
mincost/static's baseline, accumulating extra real transmission cost
over many reconfigurations?
"""
from simulate import run_episode
import numpy as np

for strat in ['static', 'mincost', 'eqopt']:
    total_costs = []
    for seed in range(10):
        res = run_episode(N=100, E0=1.0, strategy=strat, max_rounds=362, seed=seed, record_series=False)
        total_costs.append(res['AUC_Psi'])  # proxy: cumulative dispersion, but let's get raw energy spent too
    print(f"{strat:8s}: mean AUC_Psi over window = {np.mean(total_costs):.4f}")

print()
print("Direct distance check:")
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected

for strat in ['mincost', 'eqopt']:
    rng = np.random.default_rng(2)
    net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
    dist_deltas = []
    for t in range(100):
        connected = net.alive_connected()
        if len(connected) == 0: break
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        for i in newly_dead: net.kill(i)
        if newly_dead: _mark_unreachable_as_disconnected(net)
        m = M.compute_all(net, beta=0.5)
        if m['Psi'] > 999 or m['R_h'] > 0.664:
            old_parent = net.parent.copy()
            ev, ncand = C.reconfigure(net, strat, 'global', 0.5, rng)
            for i in range(net.N):
                if net.alive[i] and ev['parent'][i] != old_parent[i] and ev['parent'][i] >= 0:
                    old_d = net.dist_to_sink(i) if old_parent[i] == -1 else (net.dist(i, old_parent[i]) if old_parent[i] >= 0 else 0)
                    new_d = net.dist_to_sink(i) if ev['parent'][i] == -1 else net.dist(i, ev['parent'][i])
                    dist_deltas.append(new_d - old_d)
            net.parent = ev['parent']
            _mark_unreachable_as_disconnected(net)
        if net.alive.sum() == 0: break
    print(f"{strat:8s}: mean distance change per reassignment = {np.mean(dist_deltas) if dist_deltas else float('nan'):.2f} m  (n={len(dist_deltas)})")
