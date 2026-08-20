"""
Sanity check: if a strategy takes literally zero reconfiguration actions,
its trajectory MUST be bit-identical to static on the same seed (same
physics, same rng draws for deployment). If it isn't, something in the
test harness has a side effect we haven't found yet.
"""
import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected
from simulate import run_episode

def run_true_noop(N=100, E0=1.0, R_c=35.0, max_rounds=362, seed=0, beta=0.5):
    """Identical setup to test_hysteresis_v2 but WITHOUT ever calling
    C.generate_global_candidates / evaluate_diff at all -- true no-op,
    to isolate whether just CALLING those functions (even if discarded)
    has a side effect."""
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
        # NOTHING ELSE HAPPENS -- true no-op, no candidate generation at all
        if net.alive.sum() == 0:
            break
    return np.array(psi_series)

for seed in [0, 1, 2]:
    true_noop = run_true_noop(seed=seed)
    res_static = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=seed, record_series=True)
    static_psi = res_static['series']['Psi']
    n = min(len(true_noop), len(static_psi))
    match = np.allclose(true_noop[:n], static_psi[:n])
    print(f"seed={seed}: true_noop vs static IDENTICAL? {match}  "
          f"(true_noop Psi_max={true_noop[:n].max():.6f}, static Psi_max={static_psi[:n].max():.6f})")
