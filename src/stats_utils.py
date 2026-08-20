"""stats_utils.py -- Mann-Whitney U, Cohen's d, Spearman partial correlation."""
import numpy as np
from scipy import stats


def mann_whitney(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return stat, p


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-12:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def spearman_partial(x, y, z):
    """Partial Spearman correlation of x,y controlling for z, via rank +
    linear-residual method (standard approach for partial rank correlation)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    rx_resid = resid(rx, rz)
    ry_resid = resid(ry, rz)
    r, p = stats.pearsonr(rx_resid, ry_resid)
    return r, p


def spearman_simple(x, y):
    r, p = stats.spearmanr(x, y)
    return r, p
