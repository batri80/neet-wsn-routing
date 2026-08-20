"""
flow_simulate_static_ct.py -- episode loop for the STATIC Chang &
Tassiulas maximum-lifetime-routing baseline: solve the min-max
relative-depletion LP ONCE at t=0 (identical formulation to v3's
original fairness objective), then apply that FIXED flow allocation
for the entire episode with NO re-optimization.

Past the first death, mandatory-only local reconnection is applied
(same semantics as v2's 'static' baseline: patch around dead nodes
for basic connectivity, never re-optimize the global routing).
"""
import numpy as np
from network import Network
from flow_controller import solve_flow_lp
import metrics as M


def run_episode_static_ct(N=100, E0=0.5, R_c=30.0, sink_pos=None, beta=1.0,
                           max_rounds=4000, seed=0, w=None, max_neighbors=15,
                           record_series=True):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, sink_pos=sink_pos, E0=E0, rng=rng, w=w)

    series_Psi, series_PsiN, series_Rh, series_alive = [], [], [], []
    fnd = hnd = lnd = None
    n0 = N
    sum_Psi = sum_PsiN = sum_Rh = 0.0
    max_Psi = 0.0
    n_rounds_counted = 0
    prev_alive_count = N

    sol = solve_flow_lp(net, beta=beta, max_neighbors=max_neighbors)
    net.disconnected[:] = False
    for i in sol['disconnected']:
        net.disconnected[i] = True
    predicted_T = 1.0 / sol['objective'] if sol['objective'] and sol['objective'] > 0 else float('inf')
    fixed_cost = sol['cost'].copy()
    fixed_load = sol['load'].copy()
    n_resolves_local_only = 0

    for t in range(max_rounds):
        alive_idx = np.where(net.alive)[0]
        if len(alive_idx) == 0:
            lnd = t
            break

        cost = np.where(net.alive & ~net.disconnected, fixed_cost, 0.0)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in alive_idx if net.E[i] <= 0.0]
        for i in newly_dead:
            net.kill(i)

        if newly_dead:
            n_resolves_local_only += 1
            resolve = solve_flow_lp(net, beta=beta, max_neighbors=max_neighbors)
            net.disconnected[:] = False
            for i in resolve['disconnected']:
                net.disconnected[i] = True
            fixed_cost = resolve['cost'].copy()
            fixed_load = resolve['load'].copy()

        Psi_t, Ebar_t = M.psi(net.E)
        PsiN_t = M.psi_n(Psi_t, Ebar_t)
        alive_mask = net.alive & ~net.disconnected
        _, Lambda_t = M.structural_covariance(net.E, fixed_load, alive_mask)
        Rh_t = M.risk(PsiN_t, Lambda_t, beta)

        sum_Psi += Psi_t; sum_PsiN += PsiN_t; sum_Rh += Rh_t
        max_Psi = max(max_Psi, Psi_t)
        n_rounds_counted += 1

        alive_count = int(alive_mask.sum())
        if fnd is None and alive_count <= n0 - 1 and prev_alive_count > alive_count:
            fnd = t
        if hnd is None and alive_count <= n0 / 2:
            hnd = t
        prev_alive_count = alive_count

        if record_series:
            series_Psi.append(Psi_t); series_PsiN.append(PsiN_t)
            series_Rh.append(Rh_t); series_alive.append(alive_count)

        if alive_count == 0:
            lnd = t
            break

    if lnd is None: lnd = max_rounds
    if fnd is None: fnd = lnd
    if hnd is None: hnd = lnd

    result = dict(FND=fnd, HND=hnd, LND=lnd,
                  predicted_T=predicted_T, n_resolves_local_only=n_resolves_local_only,
                  AUC_Psi=float(sum_Psi), AUC_PsiN=float(sum_PsiN),
                  Psi_max=float(max_Psi),
                  Rh_mean=float(sum_Rh / n_rounds_counted) if n_rounds_counted else 0.0)
    if record_series:
        result['series'] = dict(Psi=np.array(series_Psi), Psi_N=np.array(series_PsiN),
                                 Rh=np.array(series_Rh), alive=np.array(series_alive))
    return result
