"""
flow_controller_agg.py -- Direction C: in-network data aggregation
added to the validated v3 flow LP. Modeled as a discount on RECEPTION
cost (not transmission), matching LEACH's E_DA which is explicitly a
processing cost paid at the aggregation point. This avoids an edge-
ambiguity problem a transmission-side discount would create when a
node splits flow across multiple distances -- Erx is already
node-level (not edge-specific) and already linear in the flow
variables, so this is a one-constant change to the validated base LP
(flow_controller.py), no new variables, no MILP risk.

Effective reception cost per unit: erx()*(1-agg_savings) + E_DA*K_BITS_AGG*agg_savings,
replacing the flat erx() used in the base (no-aggregation) model.
agg_savings=0 recovers the original model's objective EXACTLY (verified
to machine precision) -- confirms the change introduces no side effects
when aggregation is disabled.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from energy import etx, erx

E_DA = 5e-9  # J/bit, aggregation processing cost (matches protocols.py LEACH/HEED value)
K_BITS_AGG = 4000  # bits/packet, matches energy.py K_BITS -- E_DA is per-bit, needs this scaling


def build_eligible_edges(net, max_neighbors=8):
    """
    Eligible node-to-node edges (i,j): j strictly closer to sink, in
    range, restricted to each node's `max_neighbors` NEAREST such
    options. Also returns the set of nodes directly in range of the
    sink itself, and each node's distance-to-sink.
    """
    alive_idx = np.where(net.alive)[0]
    dts = {int(i): net.dist_to_sink(i) for i in alive_idx}
    order = sorted(alive_idx.tolist(), key=lambda i: dts[i])  # nearest-to-sink first

    edges = []
    direct_to_sink = set()
    for i in alive_idx:
        i = int(i)
        if dts[i] <= net.R_c:
            direct_to_sink.add(i)
        candidates = []
        for j in alive_idx:
            j = int(j)
            if i == j:
                continue
            if dts[j] < dts[i] and net.dist(i, j) <= net.R_c:
                candidates.append((net.dist(i, j), j))
        candidates.sort(key=lambda x: x[0])
        for _, j in candidates[:max_neighbors]:
            edges.append((i, j))
    return edges, direct_to_sink, dts, order


def compute_reachability(edges, direct_to_sink, order):
    """A node is reachable if it can reach the sink directly, or has an
    eligible edge to an already-reachable (nearer) node."""
    reachable = set()
    adj = {}
    for (i, j) in edges:
        adj.setdefault(i, []).append(j)
    for i in order:
        if i in direct_to_sink:
            reachable.add(i)
        else:
            for j in adj.get(i, []):
                if j in reachable:
                    reachable.add(i)
                    break
    return reachable


def solve_flow_lp_agg(net, beta=0.5, max_neighbors=8, agg_savings=0.5):
    """
    Direction C flow LP with aggregation-discounted reception cost.

    Returns a dict:
      flows: {(i,j): f_ij} node-to-node flow values
      sink_flows: {i: f_i_sink} direct-to-sink flow values
      cost: per-node energy cost array (length N)
      load: per-node incoming-flow array (length N)
      reachable, disconnected: node-index sets
      objective: the solved min-max relative depletion rate (or None if infeasible)
      feasible: bool
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
    t_idx = idx; idx += 1
    n_vars = idx

    c_obj = np.zeros(n_vars)
    c_obj[t_idx] = 1.0

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

    ub_rows, ub_cols, ub_data = [], [], []
    for r in range(n_rows):
        ub_rows.append(r); ub_cols.append(t_idx); ub_data.append(-1.0)
    for (i, j) in edges_r:
        d = net.dist(i, j)
        Ei = max(net.E[i], 1e-12)
        Ej = max(net.E[j], 1e-12)
        col = var_index[('edge', i, j)]
        ub_rows.append(row_of[i]); ub_cols.append(col); ub_data.append(etx(d) / Ei)
        ub_rows.append(row_of[j]); ub_cols.append(col)
        ub_data.append((erx() * (1 - agg_savings) + E_DA * K_BITS_AGG * agg_savings) / Ej)
    for i in d2s_r:
        d = net.dist_to_sink(i)
        Ei = max(net.E[i], 1e-12)
        col = var_index[('sink', i)]
        ub_rows.append(row_of[i]); ub_cols.append(col); ub_data.append(etx(d) / Ei)
    A_ub = sp.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_rows, n_vars))
    b_ub = np.zeros(n_rows)

    bounds = [(0, None)] * n_vars

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method='highs')

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
        cost[j] += f * (erx() * (1 - agg_savings) + E_DA * K_BITS_AGG * agg_savings)
        load[j] += f
    for i, f in sink_flows.items():
        d = net.dist_to_sink(i)
        cost[i] += f * etx(d)

    return dict(flows=flows, sink_flows=sink_flows, cost=cost, load=load,
                reachable=reachable, disconnected=disconnected,
                objective=float(x[t_idx]), feasible=True)
