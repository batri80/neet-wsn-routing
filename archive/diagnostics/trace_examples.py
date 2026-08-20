"""
Trace 5 individual R_h-triggered reconfiguration events in full detail --
what candidates existed, what EqOpt actually picked, and what happened
to Psi and Lambda^2 as a direct consequence. No aggregation, no
percentiles -- just print the raw numbers for each event so we can see
mechanically what's happening.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected

rng = np.random.default_rng(2)
net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
events_shown = 0
Psi_c, Rc_thresh, beta = 999, 0.664, 0.5

for t in range(362):
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
        candidates = C.generate_global_candidates(net, beta)
        evals = [C.evaluate_diff(net, c, beta) for c in candidates]
        psis = [e['Psi'] for e in evals]
        chosen_idx = int(np.argmin(psis))
        is_null_chosen = (chosen_idx == 0)  # candidates[0] is always the null diff

        ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
        net.parent = ev['parent']
        _mark_unreachable_as_disconnected(net)
        m_after = M.compute_all(net, beta=beta)

        print(f"--- round {t}, event {events_shown+1} ---")
        print(f"  R_h_before={m_before['R_h']:.4f}  Lambda_before={m_before['Lambda']:.4f}  Psi_before={m_before['Psi']:.6f}")
        print(f"  num candidates={len(candidates)}  candidate Psi values={[round(p,6) for p in psis]}")
        print(f"  chosen candidate index={chosen_idx}  (null? {is_null_chosen})")
        print(f"  R_h_after={m_after['R_h']:.4f}  Lambda_after={m_after['Lambda']:.4f}  Psi_after={m_after['Psi']:.6f}")
        events_shown += 1
        if events_shown >= 5:
            break
    if net.alive.sum() == 0:
        break
