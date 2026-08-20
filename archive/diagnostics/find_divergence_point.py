"""
Round-by-round comparison of true_noop (hand-rolled) vs the real
run_episode(strategy='static') -- find the FIRST round they disagree,
rather than only comparing final Psi_max. Given tonight's repeated
pattern of hand-rolled scripts drifting from the real implementation,
treat run_episode as ground truth throughout.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import metrics as M
from simulate import _mark_unreachable_as_disconnected, run_episode

def true_noop(N=100, E0=1.0, R_c=35.0, max_rounds=362, seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    psi_series = []
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
        m = M.compute_all(net, beta=beta)
        psi_series.append(m['Psi'])
        if net.alive.sum() == 0:
            break
    return np.array(psi_series)

noop_psi = true_noop(seed=0)
real_res = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=0, record_series=True)
real_psi = real_res['series']['Psi']

print(f"lengths: true_noop={len(noop_psi)}  real_static={len(real_psi)}")
n = min(len(noop_psi), len(real_psi))
for r in range(n):
    if not np.isclose(noop_psi[r], real_psi[r], atol=1e-9):
        print(f"FIRST DIVERGENCE at round {r}: true_noop={noop_psi[r]:.8f}  real={real_psi[r]:.8f}")
        break
else:
    print("no divergence found in overlapping range")
