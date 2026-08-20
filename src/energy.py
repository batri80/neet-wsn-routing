"""
energy.py -- First-order radio model + analytical multi-hop epoch cost.

Derivation (see model reference Sec. 4 / algorithm doc Sec. 9):
  Node i's total per-epoch cost, given weighted relay load L_i(t) and
  parent distance d(i, pa(i)):

      c_i = (w_i + L_i) * Etx(k, d(i,pa(i)))  +  L_i * Erx(k)

  i.e. i must transmit its own w_i packets plus every relayed packet
  (w_i + L_i total departures), and must receive each relayed packet once
  (L_i arrivals). This is deterministic given (W, T) -- no simulation of
  individual packets required, which is what makes candidate-tree
  evaluation (Delta(T)) cheap enough for the controller to use directly.
"""
import numpy as np

EELEC = 50e-9      # J/bit
EAMP = 100e-12     # J/bit/m^2
K_BITS = 4000      # bits per packet


def etx(d, k=K_BITS):
    return k * EELEC + k * EAMP * d ** 2


def erx(k=K_BITS):
    return k * EELEC


def epoch_cost(net, parent_override=None, L_override=None):
    """
    Vectorized per-node energy cost for one epoch under the network's
    current tree, or a hypothetical parent array (for candidate evaluation).
    Returns (cost_array[N], L_array[N]).
    """
    parent = net.parent if parent_override is None else parent_override
    if L_override is None:
        L, _ = net.loads(parent_override=parent_override)
    else:
        L = L_override

    connected = net.alive_connected() if parent_override is None else \
        np.where(net.alive & ~net.disconnected)[0]

    cost = np.zeros(net.N)
    for i in connected:
        p = parent[i]
        if p == -1:
            d = net.dist_to_sink(i)
        elif p >= 0:
            d = net.dist(i, p)
        else:
            continue
        cost[i] = (net.w[i] + L[i]) * etx(d) + L[i] * erx()
    return cost, L
