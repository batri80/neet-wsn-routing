"""
Trace actual K-step candidate scores at one real decision point, same
method that worked for diagnosing the one-step issue. Compares the
SPREAD of one-step vs K-step candidate scores to test whether K-step
is actually washing out the signal, as hypothesized.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected

rng = np.random.default_rng(2)
net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
Psi_c, Rc_thresh, beta, K = 0.065, 0.664, 0.5, 10

events_shown = 0
for t in range(200):
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
    if m['Psi'] > Psi_c or m['R_h'] > Rc_thresh:
        candidates = C.generate_global_candidates(net, beta)
        one_step_scores = [C.evaluate_diff(net, c, beta)['Psi'] for c in candidates]
        k_step_scores = [C.project_k_rounds(net, C.evaluate_diff(net, c, beta)['parent'], K, beta)
                          for c in candidates]

        os_arr, ks_arr = np.array(one_step_scores), np.array(k_step_scores)
        print(f"--- round {t}, {len(candidates)} candidates ---")
        print(f"  one-step: min={os_arr.min():.6f} max={os_arr.max():.6f} "
              f"spread={os_arr.max()-os_arr.min():.6f} relative_spread={100*(os_arr.max()-os_arr.min())/os_arr.min():.2f}%")
        print(f"  K-step  : min={ks_arr.min():.6f} max={ks_arr.max():.6f} "
              f"spread={ks_arr.max()-ks_arr.min():.6f} relative_spread={100*(ks_arr.max()-ks_arr.min())/ks_arr.min():.2f}%")
        print(f"  one-step argmin == K-step argmin? {np.argmin(os_arr) == np.argmin(ks_arr)}")

        ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
        net.parent = ev['parent']
        _mark_unreachable_as_disconnected(net)
        events_shown += 1
        if events_shown >= 5:
            break
    if net.alive.sum() == 0:
        break
