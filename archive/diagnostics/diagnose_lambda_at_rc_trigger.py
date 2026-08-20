"""
For R_h-triggered global reconfigurations specifically, check whether
EqOpt's chosen candidate improves Psi (which it's selected for) while
leaving Lambda^2 unchanged or worse (which it's blind to).
"""
from simulate import run_episode
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected
import numpy as np

def track(N=100, E0=1.0, R_c=35.0, max_rounds=362, seed=0, beta=0.5, Psi_c=999, Rc_thresh=0.664):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    lambda_deltas, psi_deltas = [], []

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

        m_before = M.compute_all(net, beta=beta)
        if m_before['Psi'] > Psi_c or m_before['R_h'] > Rc_thresh:
            ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
            net.parent = ev['parent']
            _mark_unreachable_as_disconnected(net)
            m_after = M.compute_all(net, beta=beta)
            lambda_deltas.append(m_after['Lambda']**2 - m_before['Lambda']**2)
            psi_deltas.append(m_after['Psi'] - m_before['Psi'])
        if net.alive.sum() == 0:
            break
    return lambda_deltas, psi_deltas

all_ld, all_pd = [], []
for seed in range(10):
    ld, pd_ = track(seed=seed)
    all_ld += ld; all_pd += pd_
print(f"R_h-triggered reconfigs captured: {len(all_ld)}")
print(f"mean d(Psi) from chosen candidate  = {np.mean(all_pd):.6f}  (should be <=0, it's what EqOpt optimizes)")
print(f"mean d(Lambda^2) from chosen candidate = {np.mean(all_ld):.6f}  (EqOpt is blind to this)")
print(f"fraction where Lambda^2 got WORSE = {np.mean([x>0 for x in all_ld]):.1%}")
