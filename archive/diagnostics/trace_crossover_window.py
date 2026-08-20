"""
Zoom into rounds 100-140 specifically -- the crossover window identified
by the extended trace -- and log every reconfiguration event in detail,
the same trustworthy way as before.
"""
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected
import numpy as np

rng = np.random.default_rng(2)
net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
Psi_c, Rc_thresh, beta = 0.065, 0.664, 0.5

for t in range(140):
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

    if 95 <= t <= 140 and (newly_dead or m_before['Psi'] > Psi_c or m_before['R_h'] > Rc_thresh):
        trigger = 'local' if newly_dead else 'global'
        print(f"round {t}: trigger={trigger}  Psi_before={m_before['Psi']:.6f}  "
              f"alive_connected={len(connected)}  n_dead_this_round={len(newly_dead)}")

    if newly_dead:
        for dn in newly_dead:
            for orph in net.orphans_of(dn):
                if net.alive[orph]:
                    ev, ncand = C.reconfigure(net, 'eqopt', 'local', beta, rng, orphan=orph)
                    net.parent = ev['parent']
                    _mark_unreachable_as_disconnected(net)
    elif m_before['Psi'] > Psi_c or m_before['R_h'] > Rc_thresh:
        ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
        net.parent = ev['parent']
        _mark_unreachable_as_disconnected(net)

    if net.alive.sum() == 0:
        break
