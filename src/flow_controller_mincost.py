"""
flow_controller_mincost.py -- pure total-cost-minimizing objective,
replacing v3's min-max fairness objective entirely. Direct test of the
root-cause finding: v3 pays ~14% more total network energy than LEACH
for the same traffic delivery task, attributed to the min-max fairness
objective's deliberate sacrifice of total efficiency to protect the
worst-off node. This removes that sacrifice completely: minimize
sum_i c_i(F) instead of minimize max_i rho_i(F).

Simpler LP than the base model: no epigraph variable t, no per-node
depletion-rate constraint -- just a direct linear cost sum, subject to
the same conservation constraints.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from energy import etx, erx
from flow_controller import build_eligible_edges, compute_reachability


def solve_flow_lp_mincost(net, beta=0.5, max_neighbors=8):
    """
    Same eligible-edge construction and conservation constraints as
    the base model (flow_controller.py) -- only the objective changes,
    from min-max relative depletion to minimize total network cost.
    """
    edges, direct_to_sink, dts, order = build_eligible_edges(net, max_neighbors=max_neighbors)
    reachable = compute_reachability(edges, direct_to_sink, order)
    alive_idx = np.where(net.alive)[0]
    disconnected = set(int(i) for i in alive_idx if int(i) not in reachable)

    if len(reachable) == 0:
        return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                    load=np.zeros(net.N), reachable=reachable,
                    disconnected=disconnected, objective=0.0, feasible=True)

    edges_r = [(i, j) for (i, j) in edges if i in reachable and j in reachable]
    d2s_r = {i for i in reachable if i in direct_to_sink}

    var_index = {}
    idx = 0
    for (i, j) in edges_r:
        var_index[('edge', i, j)] = idx; idx += 1
    for i in d2s_r:
        var_index[('sink', i)] = idx; idx += 1
    n_vars = idx

    c_obj = np.zeros(n_vars)
    for (i, j) in edges_r:
        d = net.dist(i, j)
        c_obj[var_index[('edge', i, j)]] = etx(d) + erx()
    for i in d2s_r:
        d = net.dist_to_sink(i)
        c_obj[var_index[('sink', i)]] = etx(d)

    reach_list = sorted(reachable)
    row_of = {node: r for r, node in enumerate(reach_list)}
    n_rows = len(reach_list)

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

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return dict(flows={}, sink_flows={}, cost=np.zeros(net.N),
                    load=np.zeros(net.N), reachable=reachable,
                    disconnected=set(int(i) for i in alive_idx),
                    objective=None, feasible=False)

    x = res.x
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
                objective=float(res.fun), feasible=True)
