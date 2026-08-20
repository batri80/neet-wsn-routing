"""
flow_simulate_agg.py -- episode loop for Direction C (in-network
aggregation, discounted reception cost matching LEACH's E_DA
mechanism, applied fairly to v3's own multi-path flow model).
"""
import numpy as np
from network import Network
from flow_controller_agg import solve_flow_lp_agg
import metrics as M


def run_episode_v3_agg(N=100, E0=0.5, R_c=30.0, sink_pos=None, beta=1.0,
                        Psi_c=None, Rc_thresh=None, max_rounds=4000, seed=0,
                        w=None, max_neighbors=15, agg_savings=0.5,
                        record_series=True):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, sink_pos=sink_pos, E0=E0, rng=rng, w=w)

    if Psi_c is None:
        Psi_c = 0.05579
    if Rc_thresh is None:
        Rc_thresh = 1.02488

    series_Psi, series_PsiN, series_Rh, series_alive = [], [], [], []
    fnd = hnd = lnd = None
    n0 = N
    n_solves = 0
    sum_Psi = sum_PsiN = sum_Rh = 0.0
    max_Psi = 0.0
    n_rounds_counted = 0
    prev_alive_count = N
    current_solution = None

    for t in range(max_rounds):
        alive_idx = np.where(net.alive)[0]
        if len(alive_idx) == 0:
            lnd = t
            break

        if current_solution is None:
            current_solution = solve_flow_lp_agg(
                net, beta=beta, max_neighbors=max_neighbors, agg_savings=agg_savings)
            n_solves += 1
            net.disconnected[:] = False
            for i in current_solution['disconnected']:
                net.disconnected[i] = True

        cost = current_solution['cost']
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in alive_idx if net.E[i] <= 0.0]
        dead_this_round = False
        for i in newly_dead:
            net.kill(i)
            dead_this_round = True

        Psi_t, Ebar_t = M.psi(net.E)
        PsiN_t = M.psi_n(Psi_t, Ebar_t)
        alive_mask = net.alive & ~net.disconnected
        load_arr = current_solution.get('load', np.zeros(net.N))
        _, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
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

        if dead_this_round or Psi_t > Psi_c or Rh_t > Rc_thresh:
            current_solution = solve_flow_lp_agg(
                net, beta=beta, max_neighbors=max_neighbors, agg_savings=agg_savings)
            n_solves += 1
            net.disconnected[:] = False
            for i in current_solution['disconnected']:
                net.disconnected[i] = True

        if alive_count == 0:
            lnd = t
            break

    if lnd is None: lnd = max_rounds
    if fnd is None: fnd = lnd
    if hnd is None: hnd = lnd

    result = dict(FND=fnd, HND=hnd, LND=lnd, n_solves=n_solves,
                  AUC_Psi=float(sum_Psi), AUC_PsiN=float(sum_PsiN),
                  Psi_max=float(max_Psi),
                  Rh_mean=float(sum_Rh / n_rounds_counted) if n_rounds_counted else 0.0)
    if record_series:
        result['series'] = dict(Psi=np.array(series_Psi), Psi_N=np.array(series_PsiN),
                                 Rh=np.array(series_Rh), alive=np.array(series_alive))
    return result
