"""
At every decision where one-step and K-step disagree, check which
candidate ACTUALLY performs better empirically (ground truth: real
20-round unfolding under each disputed candidate, not another
projection -- i.e., what really happens if we commit to each).
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

disagreements_checked = 0
k_step_was_right = 0

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
        os_pick = int(np.argmin(one_step_scores))
        ks_pick = int(np.argmin(k_step_scores))

        if os_pick != ks_pick:
            disagreements_checked += 1
            # Ground truth: unfold 40 REAL rounds under each disputed candidate
            os_parent = C.evaluate_diff(net, candidates[os_pick], beta)['parent']
            ks_parent = C.evaluate_diff(net, candidates[ks_pick], beta)['parent']
            true_os = C.project_k_rounds(net, os_parent, 40, beta)
            true_ks = C.project_k_rounds(net, ks_parent, 40, beta)
            winner = 'K-step pick' if true_ks < true_os else 'one-step pick'
            if true_ks < true_os:
                k_step_was_right += 1
            print(f"round {t}: one-step_Psi_at_40={true_os:.6f}  K-step_Psi_at_40={true_ks:.6f}  "
                  f"TRUE WINNER: {winner}")

        ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
        net.parent = ev['parent']
        _mark_unreachable_as_disconnected(net)
    if net.alive.sum() == 0:
        break
    if disagreements_checked >= 10:
        break

print(f"\nK-step was actually right in {k_step_was_right}/{disagreements_checked} disagreements")
