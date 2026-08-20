"""
flow_simulate.py -- NEET v3 episode loop. Replaces v2's tree-based
reconfiguration loop with periodic re-solves of the flow LP
(flow_controller.solve_flow_lp), triggered on node death (mandatory)
or Psi/R_h threshold crossing (preventive) -- see Section 6 of the v3
model specification. v2's Psi_c/R_c defaults are used as an initial,
explicitly-unvalidated approximation (Open Item 2 of the spec).
"""
import numpy as np
from network import Network
from flow_controller import solve_flow_lp
import metrics as M


def run_episode_v3(N=100, E0=0.5, R_c=30.0, sink_pos=None, beta=0.5,
                    Psi_c=None, Rc_thresh=None, max_rounds=4000, seed=0,
                    w=None, max_neighbors=8, record_series=True):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, sink_pos=sink_pos, E0=E0, rng=rng, w=w)

    if Psi_c is None:
        Psi_c = 0.065
    if Rc_thresh is None:
        Rc_thresh = 0.664

    series_Psi, series_PsiN, series_Rh, series_alive = [], [], [], []
    fnd = hnd = lnd = None
    hole_events = 0
    n0 = N
    n_lp_solves = 0

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
            current_solution = solve_flow_lp(net, beta=beta, max_neighbors=max_neighbors)
            n_lp_solves += 1
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
        Ch_t, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
        Rh_t = M.risk(PsiN_t, Lambda_t, beta)

        sum_Psi += Psi_t; sum_PsiN += PsiN_t; sum_Rh += Rh_t
        max_Psi = max(max_Psi, Psi_t)
        n_rounds_counted += 1
        if Rh_t >= Rc_thresh:
            hole_events += 1

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
            current_solution = solve_flow_lp(net, beta=beta, max_neighbors=max_neighbors)
            n_lp_solves += 1
            net.disconnected[:] = False
            for i in current_solution['disconnected']:
                net.disconnected[i] = True

        if alive_count == 0:
            lnd = t
            break

    if lnd is None:
        lnd = max_rounds
    if fnd is None:
        fnd = lnd
    if hnd is None:
        hnd = lnd

    series_Psi = np.array(series_Psi); series_PsiN = np.array(series_PsiN)
    series_Rh = np.array(series_Rh)

    result = dict(
        FND=fnd, HND=hnd, LND=lnd,
        AUC_Psi=float(sum_Psi), AUC_PsiN=float(sum_PsiN),
        Psi_max=float(max_Psi),
        Rh_mean=float(sum_Rh / n_rounds_counted) if n_rounds_counted else 0.0,
        hole_events=hole_events, n_lp_solves=n_lp_solves,
    )
    if record_series:
        result['series'] = dict(Psi=series_Psi, Psi_N=series_PsiN, Rh=series_Rh,
                                 alive=np.array(series_alive))
    return result
