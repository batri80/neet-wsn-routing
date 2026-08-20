"""
network.py -- NEET v2 core network model.

Implements the frozen model: N nodes, communication graph G(t), BFS routing
tree T(t) rooted at the sink, weighted relay load L_i(t), and the
reconnection/partition rule (Section 3 of the model reference doc).
"""
import numpy as np


class Network:
    def __init__(self, N, L=100.0, R_c=30.0, sink_pos=None, E0=0.5,
                 rng=None, w=None, max_deploy_retries=20):
        self.N = N
        self.L = L
        self.R_c = R_c
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sink = np.array(sink_pos if sink_pos is not None else [L / 2, L / 2], dtype=float)
        self.E0 = E0
        self.w = np.ones(N) if w is None else np.asarray(w, dtype=float)

        # Deploy nodes, retrying if the initial graph is badly disconnected
        # from the sink (keeps experiments from silently starting dead).
        for attempt in range(max_deploy_retries):
            self.pos = self.rng.uniform(0, L, size=(N, 2))
            self.E = np.full(N, E0, dtype=float)
            self.alive = np.ones(N, dtype=bool)
            self.disconnected = np.zeros(N, dtype=bool)
            self.parent = np.full(N, -2, dtype=int)
            self.depth = np.full(N, -1, dtype=int)
            self._rebuild_tree()
            frac_disc = self.disconnected.sum() / N
            if frac_disc < 0.15:
                break

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def dist_to_sink(self, i):
        return np.linalg.norm(self.pos[i] - self.sink)

    def dist(self, i, j):
        return np.linalg.norm(self.pos[i] - self.pos[j])

    def _in_range(self, i, candidate_idx):
        """Subset of candidate_idx within R_c of node i (excluding i itself)."""
        candidate_idx = candidate_idx[candidate_idx != i]
        if len(candidate_idx) == 0:
            return candidate_idx
        d = np.linalg.norm(self.pos[candidate_idx] - self.pos[i], axis=1)
        return candidate_idx[d <= self.R_c]

    # ------------------------------------------------------------------
    # Routing tree construction (full BFS -- used at init and after
    # multi-node die-offs where local repair is not well-defined)
    # ------------------------------------------------------------------
    def _rebuild_tree(self):
        alive_idx = np.where(self.alive)[0]
        self.parent[:] = -2
        self.depth[:] = -1
        self.disconnected[:] = False

        visited = set()
        queue = []
        for i in alive_idx:
            if self.dist_to_sink(i) <= self.R_c:
                self.parent[i] = -1  # direct to sink
                self.depth[i] = 1
                visited.add(i)
                queue.append(i)

        head = 0
        while head < len(queue):
            cur = queue[head]; head += 1
            neighbors = self._in_range(cur, alive_idx)
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    self.parent[nb] = cur
                    self.depth[nb] = self.depth[cur] + 1
                    queue.append(nb)

        for i in alive_idx:
            if i not in visited:
                self.disconnected[i] = True
                self.parent[i] = -2
                self.depth[i] = -1

    # ------------------------------------------------------------------
    # Local repair (Section 6: local trigger -- node death)
    # ------------------------------------------------------------------
    def orphans_of(self, dead_node):
        """Direct children of dead_node in the current tree (orphan roots)."""
        connected = np.where(self.alive & ~self.disconnected)[0]
        return connected[self.parent[connected] == dead_node]

    def candidate_parents_for(self, orphan, exclude_subtree, depth_slack=1,
                               parent_override=None, depth_override=None):
        """
        In-range alive-and-connected nodes eligible to become orphan's new
        parent, excluding orphan's own subtree (no cycles) and respecting
        the hop-depth-change <= 1 axiom. Accepts an optional working parent
        array / depth map so callers (e.g. coordinate-descent) can query
        against a partially-modified hypothetical tree without mutating
        network state or reintroducing cycles from stale subtree info.
        """
        connected = np.where(self.alive & ~self.disconnected)[0]
        pool = np.array([n for n in connected if n not in exclude_subtree and n != orphan])
        if len(pool) == 0:
            return np.array([], dtype=int)
        in_range = self._in_range(orphan, pool)
        if len(in_range) == 0:
            return in_range
        old_depth = depth_override.get(orphan, self.depth[orphan]) if depth_override else self.depth[orphan]
        if depth_override:
            new_depths = np.array([depth_override.get(n, self.depth[n]) for n in in_range]) + 1
        else:
            new_depths = self.depth[in_range] + 1
        keep = np.abs(new_depths - old_depth) <= depth_slack
        return in_range[keep]

    def subtree_of(self, root, parent_override=None):
        """All descendants of root (root excluded), via given (or current) parent array."""
        parent = self.parent if parent_override is None else parent_override
        connected = np.where(self.alive & ~self.disconnected)[0]
        children = {i: [] for i in connected}
        for i in connected:
            p = parent[i]
            if p >= 0 and p in children:
                children[p].append(i)
        out = set()
        stack = list(children.get(root, []))
        while stack:
            n = stack.pop()
            if n not in out:
                out.add(n)
                stack.extend(children.get(n, []))
        return out

    # ------------------------------------------------------------------
    # Weighted relay load L_i(t) = sum_{j in D(i,t)} w_j
    # ------------------------------------------------------------------
    def loads(self, parent_override=None):
        """
        Compute weighted relay load for every node given either the live
        tree or a hypothetical parent array (parent_override), used by the
        controller to analytically evaluate candidate trees without
        mutating network state.
        """
        parent = self.parent if parent_override is None else parent_override
        connected = np.where(self.alive & ~self.disconnected)[0]
        children = {i: [] for i in connected}
        for i in connected:
            p = parent[i]
            if p >= 0 and p in children:
                children[p].append(i)

        depth = self._depths_from_parent(parent, connected)
        order = sorted(connected, key=lambda i: -depth.get(i, 0))
        L = np.zeros(self.N)
        for i in order:
            load = 0.0
            for c in children.get(i, []):
                load += self.w[c] + L[c]
            L[i] = load
        return L, depth

    def _depths_from_parent(self, parent, connected_idx):
        """Iterative (cycle-safe) depth computation via parent-chain climb."""
        connected_set = set(connected_idx.tolist())
        depth = {}
        for i in connected_idx:
            if i in depth:
                continue
            chain = []
            cur = i
            visited_this_chain = set()
            while True:
                if cur in depth:
                    base = depth[cur]
                    break
                if cur in visited_this_chain:
                    # cycle detected -- break it by treating cur as a root
                    base = 0
                    break
                visited_this_chain.add(cur)
                chain.append(cur)
                p = parent[cur]
                if p == -1:
                    base = 0
                    chain_root_is_sink = True
                    break
                if p not in connected_set:
                    base = 0
                    break
                cur = p
            # unwind: assign depths from base outward
            for k, node in enumerate(reversed(chain)):
                base += 1
                depth[node] = base
        return depth

    # ------------------------------------------------------------------
    # Node death / reconnection rule (Section 3)
    # ------------------------------------------------------------------
    def kill(self, i):
        self.alive[i] = False
        self.E[i] = 0.0

    def alive_connected(self):
        return np.where(self.alive & ~self.disconnected)[0]
