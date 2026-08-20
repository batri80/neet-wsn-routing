from simulate import run_episode
import numpy as np

for strat in ['eqopt']:
    res = run_episode(N=500, E0=1.0, strategy=strat, max_rounds=3000, seed=0, record_series=True)
    alive_series = res['series']['alive']
    print(f"{strat}: alive-node count at rounds 40,60,100,500,1500,2999:")
    for r in [40, 60, 100, 500, 1500, 2999]:
        if r < len(alive_series):
            print(f"  round {r}: alive_connected={alive_series[r]}")
    print(f"  HND={res['HND']}  LND={res['LND']}  final alive_connected={alive_series[-1] if len(alive_series) else 'N/A'}")
