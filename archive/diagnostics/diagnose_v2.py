from simulate import run_episode
import numpy as np

print("=== Concern 1: is n_local=0 real, or a counting bug? ===")
for strat in ['mincost', 'eqopt']:
    total_deaths_with_children = 0
    total_deaths = 0
    for seed in range(5):
        res = run_episode(N=100, E0=1.0, strategy=strat, max_rounds=1500, seed=seed, record_series=False)
        # n_local counts RECONFIG events, not deaths -- check raw death count separately
    print(f"{strat}: (see next block for direct orphan check)")

from network import Network
from energy import epoch_cost
rng = __import__('numpy').random.default_rng(0)
net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
orphan_counts = []
for t in range(1500):
    connected = net.alive_connected()
    if len(connected) == 0:
        break
    cost, L = epoch_cost(net)
    net.E = np.clip(net.E - cost, 0.0, None)
    for i in connected:
        if net.E[i] <= 0.0:
            n_orphans = len(net.orphans_of(i))
            orphan_counts.append(n_orphans)
            net.kill(i)
print(f"deaths observed: {len(orphan_counts)}  deaths WITH orphans: {sum(1 for o in orphan_counts if o>0)}  "
      f"mean orphans-per-death: {np.mean(orphan_counts) if orphan_counts else 0:.3f}")

print("\n=== Concern 2: is LND pinned at cap for eqopt at N=500 due to leftover frozen nodes? ===")
for strat in ['random', 'eqopt', 'mincost']:
    rows = [run_episode(N=500, E0=1.0, strategy=strat, max_rounds=3000, seed=s, record_series=False)
            for s in range(5)]
    lnds = [r['LND'] for r in rows]
    hnds = [r['HND'] for r in rows]
    print(f"{strat:8s}: LNDs={lnds}  HNDs={hnds}")

print("\n=== proper matched-window Psi_max at N=500 (window = min observed LND across strategies) ===")
# first find the real minimum survival time
min_lnd = 99999
for strat in ['random', 'eqopt']:
    rows = [run_episode(N=500, E0=1.0, strategy=strat, max_rounds=3000, seed=s, record_series=False)
            for s in range(5)]
    min_lnd = min(min_lnd, min(r['LND'] for r in rows))
window = int(min_lnd * 0.9)  # a bit under the shortest actual survival, so all episodes reach it
print(f"using matched window = {window} rounds (90% of shortest observed LND={min_lnd})")
for strat in ['random', 'eqopt']:
    vals = []
    for seed in range(8):
        res = run_episode(N=500, E0=1.0, strategy=strat, max_rounds=window, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= window:
            vals.append(psi[:window].max())
    print(f"{strat:8s}: mean Psi_max over 0-{window} = {np.mean(vals) if vals else float('nan'):.5f}  (n={len(vals)}/8)")
