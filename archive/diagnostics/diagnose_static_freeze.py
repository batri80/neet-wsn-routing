from network import Network
from energy import epoch_cost
import metrics as M
import numpy as np

for N in [60, 500]:
    frozen_fracs, psi_maxes = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        net = Network(N=N, R_c=35.0, E0=1.0, rng=rng)
        psi_max = 0.0
        for t in range(1500):
            connected = net.alive_connected()
            if len(connected) == 0:
                break
            cost, L = epoch_cost(net)
            net.E = np.clip(net.E - cost, 0.0, None)
            for i in connected:
                if net.E[i] <= 0.0:
                    net.kill(i)
            m = M.compute_all(net, beta=0.5)
            psi_max = max(psi_max, m['Psi'])
        disc_frac = net.disconnected.sum() / N
        frozen_fracs.append(disc_frac)
        psi_maxes.append(psi_max)
    print(f"N={N:4d} (static): mean disconnected fraction at end={np.mean(frozen_fracs):.2%}  "
          f"mean Psi_max={np.mean(psi_maxes):.4f}")
