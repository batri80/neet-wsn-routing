"""
Non-circular R_c calibration: 'hole' = a node dying while carrying
above-average relay load (a real structural failure), independent of
any R_h threshold. R_h is then measured AT that moment, across many
episodes, and the 10th percentile becomes the calibrated R_c.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import metrics as M
import controller as C

def scan_structural_holes(N=60, E0=1.0, R_c=35.0, strategy='random',
                           max_rounds=800, seed=0, Psi_c=0.065):
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
        m = M.compute_all(net, beta=0.5)
        Rh_t = m['R_h']

        if newly_dead:
            mean_load = L[connected].mean() if len(connected) else 0.0
            for nd in newly_dead:
                if L[nd] > mean_load:          # real structural failure
                    rh_at_holes.append(Rh_t)
                net.kill(nd)

        if strategy != 'static' and (m['Psi'] > Psi_c or Rh_t > 0.99):
            ev, _ = C.reconfigure(net, strategy, 'global', 1.0, rng)
            net.parent = ev['parent']; net.E = ev['E_plus']

        if net.alive.sum() == 0:
            break
    return rh_at_holes

all_rh = []
for strat in ['random', 'mincost']:
    for seed in range(8):
        all_rh += scan_structural_holes(strategy=strat, seed=seed)

print(f"structural hole events captured: {len(all_rh)}")
if all_rh:
    print("10th percentile R_h at real holes (candidate R_c):", np.percentile(all_rh, 10))
    print("median R_h at real holes:", np.percentile(all_rh, 50))
else:
    print("No structural holes captured — widen seed/strategy range or lower Psi_c above.")
