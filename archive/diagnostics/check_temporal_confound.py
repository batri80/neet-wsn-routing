from network import Network
from energy import epoch_cost
import metrics as M
import controller as C
import numpy as np

def run_and_track(strategy, N=60, E0=1.0, R_c=35.0, max_rounds=1500, seed=0,
                   beta=0.5, Psi_c=0.065, Rc_thresh=0.664):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    psin_series, lambda_sq_series, ebar_series = [], [], []

    for t in range(max_rounds):
        connected = net.alive_connected()
        if len(connected) == 0:
            break
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        for i in newly_dead:
            net.kill(i)

        m = M.compute_all(net, beta=beta)
        psin_series.append(m['Psi_N'])
        lambda_sq_series.append(m['Lambda'] ** 2)
        ebar_series.append(m['Ebar'])

        if strategy != 'static' and newly_dead:
            for dn in newly_dead:
                for orph in net.orphans_of(dn):
                    if net.alive[orph]:
                        ev, _ = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph)
                        net.parent = ev['parent']
        elif strategy != 'static' and (m['Psi'] > Psi_c or m['R_h'] > Rc_thresh):
            ev, _ = C.reconfigure(net, strategy, 'global', beta, rng)
            net.parent = ev['parent']
        if net.alive.sum() == 0:
            break
    return np.array(psin_series), np.array(lambda_sq_series), np.array(ebar_series)

WINDOW = 300  # only compare the first 300 rounds -- comparable network age
for strat in ['random', 'eqopt']:
    psins_early, lam2s_early, ebars_end = [], [], []
    for seed in range(8):
        psin, lam2, ebar = run_and_track(strat, seed=seed)
        n = min(WINDOW, len(psin))
        psins_early.append(psin[:n].mean())
        lam2s_early.append(lam2[:n].mean())
        ebars_end.append(ebar[-1] if len(ebar) else 0.0)
    print(f"{strat:8s}: EARLY(rounds 0-{WINDOW}) mean Psi_N={np.mean(psins_early):.4f}  "
          f"mean Lambda^2={np.mean(lam2s_early):.4f}   |  final Ebar={np.mean(ebars_end):.4f}")
