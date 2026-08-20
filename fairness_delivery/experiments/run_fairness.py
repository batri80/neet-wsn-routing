from fairness_sweep_lib import run_episode_with_fairness
import numpy as np
import json
import os
import time
import sys

# Writes to results/fairness_results/, matching every other experiment
# script's convention in this repository (results/ as a top-level sibling
# of experiments/), not a subfolder of experiments/ itself.
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fairness_results')
os.makedirs(OUTDIR, exist_ok=True)

obj = sys.argv[1]
seed_start = int(sys.argv[2])
seed_end = int(sys.argv[3])

for seed in range(seed_start, seed_end):
    fname = os.path.join(OUTDIR, f'{obj}_seed{seed}.json')
    if os.path.exists(fname):
        print(f'{obj} seed={seed} [cached]')
        continue
    t0 = time.time()
    res = run_episode_with_fairness(N=100, E0=1.0, R_c=35.0, beta=1.0, max_rounds=3000, seed=seed, objective=obj)
    with open(fname, 'w') as f:
        json.dump({'HND': res['HND'], 'jains_fairness': res['jains_fairness']}, f)
    print(f'{obj} seed={seed}: HND={res["HND"]} Jain={res["jains_fairness"]:.4f} time={time.time()-t0:.1f}s')

