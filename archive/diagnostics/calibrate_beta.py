"""
Sweep beta and check which value makes R_h best separate real structural
hole events from ordinary rounds, using the same non-circular hole
definition as calibrate_Rc.py.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import metrics as M
import controller as C

def scan(beta, N=60, E0=1.0, R_c=35.0, strategy='random',
         max_rounds=800, seed=0, Psi_c=0.065):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    rh_holes, rh_normal = [], []

    for t in range(max_rounds):
        connected = net.alive_connected()
        if len(connected) == 0:
            break
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        m = M.compute_all(net, beta=beta)
        Rh_t = m['R_h']

        is_hole_round = False
        if newly_dead:
            mean_load = L[connected].mean() if len(connected) else 0.0
            for nd in newly_dead:
                if L[nd] > mean_load:
                    is_hole_round = True
                net.kill(nd)
        (rh_holes if is_hole_round else rh_normal).append(Rh_t)

        if strategy != 'static' and (m['Psi'] > Psi_c or Rh_t > 0.99):
            ev, _ = C.reconfigure(net, strategy, 'global', beta, rng)
            net.parent = ev['parent']; net.E = ev['E_plus']
        if net.alive.sum() == 0:
            break
    return rh_holes, rh_normal

for beta in [0.5, 1.0, 2.0]:
    all_holes, all_normal = [], []
    for strat in ['random', 'mincost']:
        for seed in range(6):
            h, n = scan(beta, strategy=strat, seed=seed)
            all_holes += h; all_normal += n
    if all_holes and all_normal:
        sep = np.mean(all_holes) - np.mean(all_normal)
        print(f"beta={beta}: mean R_h at holes={np.mean(all_holes):.3f}, "
              f"mean R_h normal={np.mean(all_normal):.3f}, separation={sep:.3f}")
