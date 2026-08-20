"""
flow_controller_blend.py -- J_alpha(F) = alpha*J_fair + (1-alpha)*J_cost,
properly normalized. alpha=1 recovers NEET-v3 (min-max fairness)
exactly; alpha=0 recovers adaptive mincost-v3 exactly.

Normalization: each network state's fairness-optimal value (t_min) and
cost-optimal value (C_min) are computed first (two extra LP solves),
then used as fixed scaling constants in the blended objective:

  minimize (alpha/t_min)*t + ((1-alpha)/C_min)*sum_i c_i(F)
  subject to c_i(F)/E_i <= t for all i (epigraph, as in the base model)
             conservation constraints
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from energy import etx, erx
from flow_controller import build_eligible_edges, compute_reachability


def solve_flow_lp_blend(net, alpha=0.5, beta=0.5, max_neighbors=8):
    edges, direct_to_sink, dts, order = build_eligible_edges(net, max_neighbors=max_neighbors)
    reachable = compute_reachability(edges, direct_to_sink, order)
    alive_idx = np.where(net.alive)[0]
    disconnected = set(int(i) for i in alive_idx if int(i) not in reachable)

    if len(reachable) == 0:
        return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                    load=np.zeros(net.N), reachable=reachable,
                    disconnected=disconnected, objective=0.0, feasible=True,
                    t_min=0.0, C_min=0.0)

    edges_r = [(i, j) for (i, j) in edges if i in reachable and j in reachable]
    d2s_r = {i for i in reachable if i in direct_to_sink}
    reach_list = sorted(reachable)
    row_of = {node: r for r, node in enumerate(reach_list)}
    n_rows = len(reach_list)

    var_index = {}
    idx = 0
    for (i, j) in edges_r:
        var_index[('edge', i, j)] = idx; idx += 1
    for i in d2s_r:
        var_index[('sink', i)] = idx; idx += 1
    t_idx = idx; idx += 1
    n_vars = idx

    eq_rows, eq_cols, eq_data = [], [], []
    for (i, j) in edges_r:
        col = var_index[('edge', i, j)]
        eq_rows.append(row_of[i]); eq_cols.append(col); eq_data.append(1.0)
        eq_rows.append(row_of[j]); eq_cols.append(col); eq_data.append(-1.0)
    for i in d2s_r:
        col = var_index[('sink', i)]
        eq_rows.append(row_of[i]); eq_cols.append(col); eq_data.append(1.0)
    A_eq = sp.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_rows, n_vars))
    b_eq = np.array([net.w[i] for i in reach_list])
    bounds = [(0, None)] * n_vars

    ub_rows, ub_cols, ub_data = [], [], []
    for r in range(n_rows):
        ub_rows.append(r); ub_cols.append(t_idx); ub_data.append(-1.0)
    for (i, j) in edges_r:
        d = net.dist(i, j)
        Ei = max(net.E[i], 1e-12); Ej = max(net.E[j], 1e-12)
        col = var_index[('edge', i, j)]
        ub_rows.append(row_of[i]); ub_cols.append(col); ub_data.append(etx(d) / Ei)
        ub_rows.append(row_of[j]); ub_cols.append(col); ub_data.append(erx() / Ej)
    for i in d2s_r:
        d = net.dist_to_sink(i)
        Ei = max(net.E[i], 1e-12)
        col = var_index[('sink', i)]
        ub_rows.append(row_of[i]); ub_cols.append(col); ub_data.append(etx(d) / Ei)
    A_ub = sp.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_rows, n_vars))
    b_ub = np.zeros(n_rows)

    cost_coef = np.zeros(n_vars)
    for (i, j) in edges_r:
        d = net.dist(i, j)
        cost_coef[var_index[('edge', i, j)]] = etx(d) + erx()
    for i in d2s_r:
        d = net.dist_to_sink(i)
        cost_coef[var_index[('sink', i)]] = etx(d)

    c_fair = np.zeros(n_vars); c_fair[t_idx] = 1.0
    res_fair = linprog(c_fair, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
    if not res_fair.success:
        return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                    load=np.zeros(net.N), reachable=reachable,
                    disconnected=set(int(i) for i in alive_idx),
                    objective=None, feasible=False, t_min=None, C_min=None)
    t_min = max(res_fair.x[t_idx], 1e-12)

    res_cost = linprog(cost_coef, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res_cost.success:
        return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                    load=np.zeros(net.N), reachable=reachable,
                    disconnected=set(int(i) for i in alive_idx),
                    objective=None, feasible=False, t_min=t_min, C_min=None)
    C_min = max(res_cost.fun, 1e-12)

    if alpha >= 1.0 - 1e-12:
        x = res_fair.x
    elif alpha <= 1e-12:
        c_blend = cost_coef.copy()
        res_blend = linprog(c_blend, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
        x = res_blend.x if res_blend.success else res_cost.x
    else:
        c_blend = (alpha / t_min) * c_fair + ((1 - alpha) / C_min) * cost_coef
        res_blend = linprog(c_blend, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
        if not res_blend.success:
            return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                        load=np.zeros(net.N), reachable=reachable,
                        disconnected=set(int(i) for i in alive_idx),
                        objective=None, feasible=False, t_min=t_min, C_min=C_min)
        x = res_blend.x

    flows, sink_flows = {}, {}
    for (i, j) in edges_r:
        val = x[var_index[('edge', i, j)]]
        if val > 1e-9:
            flows[(i, j)] = val
    for i in d2s_r:
        val = x[var_index[('sink', i)]]
        if val > 1e-9:
            sink_flows[i] = val

    cost = np.zeros(net.N)
    load = np.zeros(net.N)
    for (i, j), f in flows.items():
        d = net.dist(i, j)
        cost[i] += f * etx(d)
        cost[j] += f * erx()
        load[j] += f
    for i, f in sink_flows.items():
        d = net.dist_to_sink(i)
        cost[i] += f * etx(d)

    return dict(flows=flows, sink_flows=sink_flows, cost=cost, load=load,
                reachable=reachable, disconnected=disconnected,
                objective=float(x[t_idx]), feasible=True,
                t_min=float(t_min), C_min=float(C_min))
