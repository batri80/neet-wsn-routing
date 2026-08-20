"""
metrics.py -- NEET v2 equilibrium observables (model reference Sec. 5-6).

Psi:     global-mean functional over ALL N nodes (dead contribute 0),
         fixes the v1 mixed-population bug.
Psi_N:   scale-invariant normalized form.
Lambda:  Pearson correlation between relay load and residual energy
         (Structural Energy Covariance, normalized) -- the spatial term
         missing from v1.
R_h:     sqrt(Psi_N + beta * Lambda^2) -- strictly positive, no sign
         cancellation (v1's (1+Lambda) form could hit exactly zero).
"""
import numpy as np


def psi(E):
    """E: length-N array, dead nodes = 0. Global mean/variance over all N."""
    Ebar = E.mean()
    Psi = np.mean((E - Ebar) ** 2)
    return Psi, Ebar


def psi_n(Psi, Ebar, eps=1e-12):
    return Psi / (Ebar ** 2 + eps)


def structural_covariance(E, L, alive_mask):
    """C_h and its normalized form Lambda, computed over the alive set."""
    Ea, La = E[alive_mask], L[alive_mask]
    if len(Ea) < 2:
        return 0.0, 0.0
    Ebar, Lbar = Ea.mean(), La.mean()
    Ch = -np.mean((La - Lbar) * (Ea - Ebar))
    sigE, sigL = Ea.std(), La.std()
    denom = sigE * sigL
    Lambda = Ch / denom if denom > 1e-9 else 0.0
    return Ch, float(np.clip(Lambda, -1.0, 1.0))


def risk(psin, Lambda, beta=0.5):
    val = psin + beta * Lambda ** 2
    return float(np.sqrt(max(val, 0.0)))


def compute_all(net, beta=0.5):
    """Convenience: compute (Psi, Psi_N, Lambda, R_h) from current net state."""
    Psi, Ebar = psi(net.E)
    Psin = psi_n(Psi, Ebar)
    alive_mask = net.alive & ~net.disconnected
    L, _ = net.loads()
    Ch, Lambda = structural_covariance(net.E, L, alive_mask)
    Rh = risk(Psin, Lambda, beta)
    return dict(Psi=Psi, Ebar=Ebar, Psi_N=Psin, Lambda=Lambda, R_h=Rh, L=L)
