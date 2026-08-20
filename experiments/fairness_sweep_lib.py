"""
fairness_sweep.py -- extends the existing, verified NEET/NEET-Cost episode
loops (flow_simulate.py / flow_simulate_mincost.py) to additionally track
per-node cumulative carried traffic (own w_i=1 plus relayed load[i], summed
over every round the node is alive), then computes Jain's fairness index
over that cumulative array at episode end.

This is NOT a new simulator: it reuses solve_flow_lp / solve_flow_lp_mincost
(the same verified LP solvers used throughout the paper) and Network exactly
as-is. The only addition is a per-node accumulator array and a fairness
index computed from it -- addressing the "sampling fairness" evaluation-gap
limitation (Discussion, Section 9.2) for the relay-burden interpretation
that actually applies to NEET's uniform-traffic-generation model (w_i=1 for
every alive node, confirmed in network.py -- there is no transmission-
opportunity inequality to measure, unlike the source-selection EqOpt model
this paper's predecessor used).

Jain's fairness index: J(x) = (sum(x))^2 / (N * sum(x^2)), x_i = node i's
total cumulative carried traffic over its alive lifetime. J=1 is perfectly
fair (every node carries equal total burden); J=1/N is maximally unfair
(one node carries everything).

Assumes this file lives in experiments/ alongside the repo's src/ directory
(the standard neet-wsn-routing layout); adds src/ to the import path
automatically so network.py, flow_controller.py, flow_controller_mincost.py,
and metrics.py are found without manual PYTHONPATH configuration.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from network import Network
from flow_controller import solve_flow_lp
from flow_controller_mincost import solve_flow_lp_mincost


def jains_index(x):
    x = np.asarray(x, dtype=float)
    if x.sum() <= 0:
        return float('nan')
    return float((x.sum() ** 2) / (len(x) * np.sum(x ** 2)))


def run_episode_with_fairness(N=100, E0=0.5, R_c=30.0, sink_pos=None,
                               beta=0.5, Psi_c=None, Rc_thresh=None,
                               max_rounds=4000, seed=0, w=None,
                               max_neighbors=8, objective='neet'):
    """
    objective: 'neet' (min-max, solve_flow_lp) or 'neet_cost'
               (total-cost-min, solve_flow_lp_mincost).
    Returns dict with HND (for a sanity cross-check against existing
    results) plus cumulative_burden (per-node array) and jains_fairness.
    """
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, sink_pos=sink_pos, E0=E0, rng=rng, w=w)
    solver = solve_flow_lp if objective == 'neet' else solve_flow_lp_mincost

    if Psi_c is None:
        Psi_c = 0.065
    if Rc_thresh is None:
        Rc_thresh = 0.664

    cumulative_burden = np.zeros(N)  # own (w_i) + relayed (load_i), summed over alive rounds
    fnd = hnd = lnd = None
    n0 = N
    prev_alive_count = N
    current_solution = None

    for t in range(max_rounds):
        alive_idx = np.where(net.alive)[0]
        if len(alive_idx) == 0:
            lnd = t
            break

        if current_solution is None:
            current_solution = solver(net, beta=beta, max_neighbors=max_neighbors)
            net.disconnected[:] = False
            for i in current_solution['disconnected']:
                net.disconnected[i] = True

        cost = current_solution['cost']
        load_arr = current_solution.get('load', np.zeros(net.N))
        # own traffic (w_i, =1 for alive nodes in this paper's setup) + relayed load
        for i in alive_idx:
            if i not in current_solution['disconnected']:
                cumulative_burden[i] += net.w[i] + load_arr[i]

        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in alive_idx if net.E[i] <= 0.0]
        dead_this_round = False
        for i in newly_dead:
            net.kill(i)
            dead_this_round = True

        import metrics as M
        Psi_t, Ebar_t = M.psi(net.E)
        alive_mask = net.alive & ~net.disconnected
        Ch_t, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
        PsiN_t = M.psi_n(Psi_t, Ebar_t)
        Rh_t = M.risk(PsiN_t, Lambda_t, beta)

        alive_count = int(alive_mask.sum())
        if fnd is None and alive_count <= n0 - 1 and prev_alive_count > alive_count:
            fnd = t
        if hnd is None and alive_count <= n0 / 2:
            hnd = t
        prev_alive_count = alive_count

        if dead_this_round or Psi_t > Psi_c or Rh_t > Rc_thresh:
            current_solution = solver(net, beta=beta, max_neighbors=max_neighbors)
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

    return dict(HND=hnd, FND=fnd, LND=lnd,
                cumulative_burden=cumulative_burden.tolist(),
                jains_fairness=jains_index(cumulative_burden))
