"""
flow_controller.py -- NEET v3 multi-path flow routing controller.

Replaces v2's discrete candidate generation + selection with a single
linear program solved per trigger: minimize the maximum relative
depletion rate (cost/remaining-energy) across all nodes, subject to
flow conservation over a DAG oriented toward the sink (edges only run
from farther to nearer, guaranteeing acyclicity by construction).

Uses sparse constraint matrices (scipy.sparse) -- an initial dense-
matrix implementation was profiled and found to scale far worse than
linear (N=500: 7.1s/solve, vs N=30: 0.005s/solve), inconsistent with
the constraint system's actual sparsity (each row only touches the
handful of edges incident to that node, not all N variables). This is
the first thing checked before any experiment, per the model
specification's explicit "profile before committing" requirement.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from energy import etx, erx


def build_eligible_edges(net, max_neighbors=8):
    """
    Eligible node-to-node edges (i,j): j strictly closer to sink, in
    range, restricted to each node's `max_neighbors` NEAREST such
    options. Also returns the set of nodes directly in range of the
    sink itself, and each node's distance-to-sink.

    The unbounded version (every in-range, closer node, no cap) was
    profiled at N=500 and found to generate 35,838 candidate edges,
    of which only 899 were actually used in the LP solution --
    handing the solver a hugely oversized problem for no benefit.
    Capping to nearest neighbors mirrors v2's GLOBAL_TOP_K fix
    (Section 14.6 of the v2 addendum) and is expected to preserve
    solution quality while dramatically shrinking solve time, since a
    node's 30th-nearest option is essentially never going to be part
    of an optimal min-max allocation over its 5 nearest.
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
    eligible edge to an already-reachable (nearer) node. Processing in
    increasing distance-to-sink order makes this a single pass."""
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


def solve_flow_lp(net, beta=0.5, max_neighbors=8):
    """
    Solve the NEET-v3 flow LP. Returns a dict:
      flows: {(i,j): f_ij} node-to-node flow values
      sink_flows: {i: f_i_sink} direct-to-sink flow values
      cost: per-node energy cost array (length N)
      load: per-node incoming-flow array (length N) -- the v3 analogue of L_i
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
        ub_rows.append(row_of[j]); ub_cols.append(col); ub_data.append(erx() / Ej)
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
        cost[j] += f * erx()
        load[j] += f
    for i, f in sink_flows.items():
        d = net.dist_to_sink(i)
        cost[i] += f * etx(d)

    return dict(flows=flows, sink_flows=sink_flows, cost=cost, load=load,
                reachable=reachable, disconnected=disconnected,
                objective=float(x[t_idx]), feasible=True)
