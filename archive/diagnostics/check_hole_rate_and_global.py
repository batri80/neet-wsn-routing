from simulate import run_episode
import numpy as np

print("=== hole rate (holes per surviving round), properly normalized ===")
for strat in ['random', 'eqopt']:
    rates, holes_list, lnds = [], [], []
    for seed in range(15):
        res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=3000,
                           seed=seed, record_series=False)
        rates.append(res['hole_events'] / max(res['LND'], 1))
        holes_list.append(res['hole_events'])
        lnds.append(res['LND'])
    print(f"{strat:8s}: mean hole_events={np.mean(holes_list):.1f}  "
          f"mean LND={np.mean(lnds):.1f}  mean hole_RATE={np.mean(rates):.4f}")

print("\n=== global-trigger null-selection rate, post local-fix ===")
for strat in ['random', 'eqopt']:
    tg, ng = [], []
    for seed in range(8):
        res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                           seed=seed, record_series=False)
        tg.append(res['total_with_alt_global'])
        ng.append(res['null_with_alt_global'])
    t, n = sum(tg), sum(ng)
    print(f"{strat:8s}: global null-rate={100*n/max(t,1):.1f}% ({n}/{t})")
