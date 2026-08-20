from simulate import run_episode
import numpy as np

for strat in ['random', 'eqopt']:
    lnds = []
    for seed in range(10):
        res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                           seed=seed, record_series=True)
        lnds.append(res['LND'])
    lnds = np.array(lnds)
    print(f"{strat:8s}: LNDs = {sorted(lnds.tolist())}")
    print(f"          episodes surviving past round 600: {(lnds > 600).sum()}/10")
    print(f"          episodes surviving past round 1000: {(lnds > 1000).sum()}/10")
