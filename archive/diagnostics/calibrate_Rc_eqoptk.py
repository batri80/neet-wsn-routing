"""
Non-circular R_c calibration for eqopt_k at K=40, using the same
structural-hole definition as the original calibration (node death
carrying above-average relay load), independent of any R_h threshold.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import metrics as M
import controller as C
from simulate import _mark_unreachable_as_disconnected

def scan_structural_holes(N=60, E0=1.0, R_c=35.0, strategy='eqopt_k', K=40,
                           max_rounds=400, seed=0, Psi_c=0.045, beta=0.5):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    rh_at_holes = []

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
        Rh_t = m['R_h']

        if newly_dead:
            mean_load = L[connected].mean() if len(connected) else 0.0
            for nd in newly_dead:
                if L[nd] > mean_load:
                    rh_at_holes.append(Rh_t)
                for orph in net.orphans_of(nd):
                    if net.alive[orph]:
                        ev, _ = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph, K=K)
                        net.parent = ev['parent']
                        _mark_unreachable_as_disconnected(net)

        if strategy != 'static' and (m['Psi'] > Psi_c or Rh_t > 0.99):
            ev, _ = C.reconfigure(net, strategy, 'global', beta, rng, K=K)
            net.parent = ev['parent']
            _mark_unreachable_as_disconnected(net)

        if net.alive.sum() == 0:
            break
    return rh_at_holes

all_rh = []
for seed in range(10):
    all_rh += scan_structural_holes(seed=seed)

print(f"structural hole events captured: {len(all_rh)}")
if all_rh:
    print("10th percentile R_h at real holes (candidate R_c):", np.percentile(all_rh, 10))
    print("median R_h at real holes:", np.percentile(all_rh, 50))
else:
    print("No structural holes captured -- may need lower Psi_c above, or more seeds.")
