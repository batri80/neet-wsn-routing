"""
Corrected version: minimum-improvement margin applies ONLY to global
triggers. Local (connectivity-restoration) reattachment remains
unconditional, per the Section 14.2 amendment -- gating it would risk
silently reintroducing the disconnection bug we already fixed.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected

def run_with_delta(delta, N=100, E0=1.0, R_c=35.0, max_rounds=362, seed=0,
                    beta=0.5, Psi_c=0.065, Rc_thresh=0.664):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    psi_max = 0.0
    n_global_acted, n_global_skipped, n_local_acted = 0, 0, 0

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
        psi_max = max(psi_max, m['Psi'])

        if newly_dead:
            # LOCAL: unconditional, no margin -- matches Section 14.2 amendment
            for dn in newly_dead:
                for orph in net.orphans_of(dn):
                    if net.alive[orph]:
                        cands = C.generate_local_candidates(net, orph)
                        evals = [C.evaluate_diff(net, c, beta) for c in cands]
                        best = evals[int(np.argmin([e['Psi'] for e in evals]))]
                        net.parent = best['parent']
                        n_local_acted += 1
                        _mark_unreachable_as_disconnected(net)
        elif m['Psi'] > Psi_c or m['R_h'] > Rc_thresh:
            # GLOBAL: margin-gated
            cands = C.generate_global_candidates(net, beta)
            null_ev = C.evaluate_diff(net, {}, beta)
            evals = [C.evaluate_diff(net, c, beta) for c in cands]
            best = evals[int(np.argmin([e['Psi'] for e in evals]))]
            if null_ev['Psi'] - best['Psi'] > delta:
                net.parent = best['parent']
                n_global_acted += 1
            else:
                net.parent = null_ev['parent']
                n_global_skipped += 1
            _mark_unreachable_as_disconnected(net)

        if net.alive.sum() == 0:
            break
    return psi_max, n_global_acted, n_global_skipped, n_local_acted

print("=== Margin applied ONLY to global triggers; local stays unconditional ===\n")
for delta in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
    vals, ga, gs, la = [], [], [], []
    for s in range(10):
        pm, a, sk, l = run_with_delta(delta, seed=s)
        vals.append(pm); ga.append(a); gs.append(sk); la.append(l)
    print(f"delta={delta:8.5f}: mean Psi_max={np.mean(vals):.5f}  std={np.std(vals):.5f}  "
          f"global_acted={np.mean(ga):.1f}  global_skipped={np.mean(gs):.1f}  local_acted={np.mean(la):.1f}")

print()
from simulate import run_episode
static_vals = []
for s in range(10):
    res = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=s, record_series=True)
    psi = res['series']['Psi']
    if len(psi) >= 362:
        static_vals.append(psi[:362].max())
print(f"static baseline    : mean Psi_max={np.mean(static_vals):.5f}")
