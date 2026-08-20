"""
Overnight pipeline to close every open calibration item for EqOpt-K
(Section 14.10 of the model reference doc): Psi_c and R_c at FULL
episode length (not the 400-round window used originally), the
cooldown period (currently an unvalidated K/2 guess), and a final
properly-powered significance test using all newly-calibrated values.

SAFETY: every full-length eqopt_k episode is wrapped in a per-episode
timeout. Every stage's results are saved to disk immediately on
completion and skipped on rerun if already present -- safe to
interrupt and resume at any point, same pattern as k_sweep_resumable.py.
"""
import numpy as np
import json
import time
import signal
import os
from scipy import stats
from simulate import run_episode
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected

OUTDIR = 'freeze_results'
os.makedirs(OUTDIR, exist_ok=True)

N, E0, R_c_comm = 100, 1.0, 35.0
K = 40
EPISODE_TIMEOUT = 240  # seconds, per full-length episode


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(**kwargs):
    """run_episode wrapped with a hard timeout. Returns None on timeout."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EPISODE_TIMEOUT)
    try:
        res = run_episode(**kwargs)
        signal.alarm(0)
        return res
    except TimeoutException:
        return None


def save_json(name, obj):
    with open(f'{OUTDIR}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    path = f'{OUTDIR}/{name}.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ======================================================================
# STAGE 1: Psi_c knee-finding at FULL episode length (3000 rounds).
# Global trigger suppressed (Psi_c=999, Rc_thresh=999) -- only mandatory
# local reconnection fires, so this is cooldown-independent and cheap.
# ======================================================================
def stage1_psi_c():
    cached = load_json('stage1_psi_c')
    if cached:
        log(f"STAGE 1 [cached]: Psi_c knee = {cached['psi_c']:.5f}")
        return cached['psi_c']

    log("STAGE 1: finding Psi_c knee at full episode length (3000 rounds)")
    checkpoints = [100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
    results = {c: [] for c in checkpoints}

    for seed in range(10):
        res = run_with_timeout(N=N, E0=E0, R_c=R_c_comm, strategy='eqopt_k', K=K,
                                Psi_c=999, Rc_thresh=999, max_rounds=3000,
                                seed=seed, record_series=True)
        if res is None:
            log(f"  seed {seed}: TIMED OUT (unexpected for suppressed-global stage)")
            continue
        psi = res['series']['Psi']
        for c in checkpoints:
            if c < len(psi):
                results[c].append(psi[c])

    means = {}
    log("  round   mean_Psi   n")
    for c in checkpoints:
        vals = results[c]
        if vals:
            means[c] = float(np.mean(vals))
            log(f"  {c:5d}   {np.mean(vals):.5f}   {len(vals)}")

    # Find the knee: first checkpoint where the round-over-round delta
    # drops below 25% of the previous delta (deceleration plateau).
    ckpts = sorted(means.keys())
    deltas = [means[ckpts[i+1]] - means[ckpts[i]] for i in range(len(ckpts)-1)]
    knee_idx = 0
    for i in range(1, len(deltas)):
        if deltas[i] < 0.5 * deltas[i-1]:
            knee_idx = i
        else:
            break
    knee_round = ckpts[knee_idx + 1]
    psi_c = means[knee_round]

    log(f"STAGE 1 result: knee at round {knee_round}, Psi_c = {psi_c:.5f}")
    save_json('stage1_psi_c', {'psi_c': psi_c, 'knee_round': knee_round,
                                 'checkpoints': means})
    return psi_c


# ======================================================================
# STAGE 2: Cooldown sweep. Uses Stage 1's Psi_c, current R_c=0.400
# placeholder (refined in Stage 3). Finds the cooldown value that
# maximizes mean HND without pathological runtime.
# ======================================================================
def stage2_cooldown(psi_c):
    cached = load_json('stage2_cooldown')
    if cached:
        log(f"STAGE 2 [cached]: best cooldown = {cached['best_cooldown']}")
        return cached['best_cooldown']

    log("STAGE 2: sweeping cooldown period")
    candidates = [5, 10, 15, 20, 30, 40, 60]
    N_SEEDS = 20
    R_c_thresh_placeholder = 0.400

    results = {}
    for cd in candidates:
        hnds, times, reconfigs, timeouts = [], [], [], 0
        t0 = time.time()
        for seed in range(N_SEEDS):
            res = run_with_timeout(N=N, E0=E0, R_c=R_c_comm, strategy='eqopt_k', K=K,
                                    Psi_c=psi_c, Rc_thresh=R_c_thresh_placeholder,
                                    cooldown_rounds=cd, max_rounds=3000,
                                    seed=seed, record_series=False)
            if res is None:
                timeouts += 1
                continue
            hnds.append(res['HND'])
            reconfigs.append(res['n_reconfigs'])
        elapsed = time.time() - t0
        results[cd] = dict(mean_hnd=float(np.mean(hnds)) if hnds else None,
                            std_hnd=float(np.std(hnds)) if hnds else None,
                            mean_reconfigs=float(np.mean(reconfigs)) if reconfigs else None,
                            n_completed=len(hnds), n_timeouts=timeouts,
                            elapsed=elapsed)
        log(f"  cooldown={cd:3d}: mean_HND={results[cd]['mean_hnd']}  "
            f"n_completed={len(hnds)}/{N_SEEDS}  timeouts={timeouts}  "
            f"mean_reconfigs={results[cd]['mean_reconfigs']}  time={elapsed:.0f}s")

    # Best cooldown: highest mean HND among those with zero timeouts
    # (a cooldown that can't reliably complete episodes is disqualified
    # regardless of its HND looking good on the seeds that did finish).
    viable = {cd: r for cd, r in results.items() if r['n_timeouts'] == 0 and r['mean_hnd'] is not None}
    if not viable:
        log("  WARNING: no cooldown value completed all seeds without timeout; "
            "picking best among partial completions")
        viable = {cd: r for cd, r in results.items() if r['mean_hnd'] is not None}
    best_cooldown = max(viable, key=lambda cd: viable[cd]['mean_hnd'])

    log(f"STAGE 2 result: best cooldown = {best_cooldown}")
    save_json('stage2_cooldown', {'best_cooldown': best_cooldown, 'all_results': results})
    return best_cooldown


# ======================================================================
# STAGE 3: R_c recalibration at full episode length, using Stage 2's
# cooldown. Same non-circular structural-hole methodology as the
# original calibration (Section 14.5): hole = death carrying
# above-average relay load, independent of any R_h threshold.
# ======================================================================
def stage3_r_c(psi_c, cooldown):
    cached = load_json('stage3_r_c')
    if cached:
        log(f"STAGE 3 [cached]: R_c = {cached['r_c']:.5f}")
        return cached['r_c']

    log(f"STAGE 3: recalibrating R_c at full length (cooldown={cooldown})")

    def scan_structural_holes(seed, beta=0.5, max_rounds=3000):
        rng = np.random.default_rng(seed)
        net = Network(N=N, R_c=R_c_comm, E0=E0, rng=rng)
        rh_at_holes = []
        cooldown_remaining = 0

        for t in range(max_rounds):
            connected = net.alive_connected()
            if len(connected) == 0:
                break
            cost, L = epoch_cost(net)
            net.E = np.clip(net.E - cost, 0.0, None)
            newly_dead = [i for i in connected if net.E[i] <= 0.0]
            for i in newly_dead:
                net.kill(i)
            if newly_dead:
                _mark_unreachable_as_disconnected(net)

            m = M.compute_all(net, beta=beta)
            Rh_t = m['R_h']

            if newly_dead:
                mean_load = L[connected].mean() if len(connected) else 0.0
                for nd in newly_dead:
                    if L[nd] > mean_load:
                        rh_at_holes.append(Rh_t)
                    for orph in net.orphans_of(nd):
                        if net.alive[orph]:
                            ev, _ = C.reconfigure(net, 'eqopt_k', 'local', beta, rng,
                                                   orphan=orph, K=K)
                            net.parent = ev['parent']
                            _mark_unreachable_as_disconnected(net)

            if cooldown_remaining == 0 and (m['Psi'] > psi_c or Rh_t > 0.99):
                ev, _ = C.reconfigure(net, 'eqopt_k', 'global', beta, rng, K=K)
                net.parent = ev['parent']
                _mark_unreachable_as_disconnected(net)
                cooldown_remaining = cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            if net.alive.sum() == 0:
                break
        return rh_at_holes

    all_rh = []
    N_SEEDS = 25
    t0 = time.time()
    for seed in range(N_SEEDS):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EPISODE_TIMEOUT)
        try:
            all_rh += scan_structural_holes(seed)
            signal.alarm(0)
        except TimeoutException:
            log(f"  seed {seed}: TIMED OUT, skipping")
    elapsed = time.time() - t0

    log(f"  structural hole events captured: {len(all_rh)}  (time={elapsed:.0f}s)")
    if not all_rh:
        log("  WARNING: no hole events captured, falling back to prior R_c=0.400")
        r_c = 0.400
    else:
        r_c = float(np.percentile(all_rh, 10))
        log(f"  10th percentile R_h at holes: {r_c:.5f}  "
            f"(median: {np.percentile(all_rh, 50):.5f})")

    save_json('stage3_r_c', {'r_c': r_c, 'n_holes': len(all_rh)})
    return r_c


# ======================================================================
# STAGE 4: beta re-verification at full length, using Stage 2/3's
# cooldown and R_c. Quick check, not a full sweep -- beta governs the
# R_h formula itself, not controller behavior, so it's expected (but
# not assumed) to be stable.
# ======================================================================
def stage4_beta(psi_c, cooldown, r_c):
    cached = load_json('stage4_beta')
    if cached:
        log(f"STAGE 4 [cached]: beta = {cached['beta']}")
        return cached['beta']

    log("STAGE 4: re-verifying beta at full length")

    def scan(beta, seed, max_rounds=3000):
        rng = np.random.default_rng(seed)
        net = Network(N=N, R_c=R_c_comm, E0=E0, rng=rng)
        rh_holes, rh_normal = [], []
        cooldown_remaining = 0
        for t in range(max_rounds):
            connected = net.alive_connected()
            if len(connected) == 0:
                break
            cost, L = epoch_cost(net)
            net.E = np.clip(net.E - cost, 0.0, None)
            newly_dead = [i for i in connected if net.E[i] <= 0.0]
            for i in newly_dead:
                net.kill(i)
            if newly_dead:
                _mark_unreachable_as_disconnected(net)
            m = M.compute_all(net, beta=beta)
            Rh_t = m['R_h']
            is_hole = False
            if newly_dead:
                mean_load = L[connected].mean() if len(connected) else 0.0
                for nd in newly_dead:
                    if L[nd] > mean_load:
                        is_hole = True
                    for orph in net.orphans_of(nd):
                        if net.alive[orph]:
                            ev, _ = C.reconfigure(net, 'eqopt_k', 'local', beta, rng,
                                                   orphan=orph, K=K)
                            net.parent = ev['parent']
                            _mark_unreachable_as_disconnected(net)
            (rh_holes if is_hole else rh_normal).append(Rh_t)
            if cooldown_remaining == 0 and (m['Psi'] > psi_c or Rh_t > 0.99):
                ev, _ = C.reconfigure(net, 'eqopt_k', 'global', beta, rng, K=K)
                net.parent = ev['parent']
                _mark_unreachable_as_disconnected(net)
                cooldown_remaining = cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            if net.alive.sum() == 0:
                break
        return rh_holes, rh_normal

    best_beta, best_sep = 0.5, -np.inf
    results = {}
    for beta in [0.5, 1.0, 2.0]:
        all_h, all_n = [], []
        for seed in range(12):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(EPISODE_TIMEOUT)
            try:
                h, n = scan(beta, seed)
                signal.alarm(0)
                all_h += h; all_n += n
            except TimeoutException:
                pass
        if all_h and all_n:
            sep = np.mean(all_h) - np.mean(all_n)
            results[beta] = sep
            log(f"  beta={beta}: separation={sep:.4f}")
            if sep > best_sep:
                best_sep = sep
                best_beta = beta

    log(f"STAGE 4 result: beta = {best_beta}")
    save_json('stage4_beta', {'beta': best_beta, 'all_separations': results})
    return best_beta


# ======================================================================
# STAGE 5: Final significance validation, n=50, using ALL newly
# calibrated constants. This is the result that determines whether
# EqOpt-K's HND claim is real at N=100, full episode length.
# ======================================================================
def stage5_final_validation(psi_c, cooldown, r_c, beta):
    cached = load_json('stage5_final')
    if cached:
        log("STAGE 5 [cached] -- see freeze_results/stage5_final.json")
        return cached

    log(f"STAGE 5: final n=50 validation (Psi_c={psi_c:.4f}, R_c={r_c:.4f}, "
        f"cooldown={cooldown}, beta={beta})")
    N_SEEDS = 50
    results = {}

    for strat, kwargs in [('static', {}), ('mincost', {}), ('eqopt', {}),
                           ('eqopt_k', dict(K=K, Psi_c=psi_c, Rc_thresh=r_c,
                                             cooldown_rounds=cooldown, beta=beta))]:
        t0 = time.time()
        hnds, timeouts = [], 0
        for seed in range(N_SEEDS):
            res = run_with_timeout(N=N, E0=E0, R_c=R_c_comm, strategy=strat,
                                    max_rounds=3000, seed=seed, record_series=False,
                                    **kwargs)
            if res is None:
                timeouts += 1
                continue
            hnds.append(res['HND'])
        elapsed = time.time() - t0
        results[strat] = hnds
        log(f"  {strat:10s}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
            f"n={len(hnds)}/{N_SEEDS}  timeouts={timeouts}  time={elapsed:.0f}s")

    log("\n  === eqopt_k vs each baseline ===")
    ek = np.array(results['eqopt_k'])
    summary = {}
    for strat in ['static', 'mincost', 'eqopt']:
        other = np.array(results[strat])
        u, p = stats.mannwhitneyu(ek, other, alternative='two-sided')
        pooled_std = np.sqrt((ek.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (ek.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
        result = 'eqopt_k WINS' if ek.mean() > other.mean() else 'eqopt_k loses'
        summary[strat] = dict(p=p, d=d, significant=p < 0.05)
        log(f"  eqopt_k vs {strat:10s}: {result}  p={p:.4f}  d={d:.3f}  {sig}")

    final = dict(psi_c=psi_c, r_c=r_c, cooldown=cooldown, beta=beta,
                 raw_hnds=results, significance=summary)
    save_json('stage5_final', final)
    return final


# ======================================================================
def main():
    t_start = time.time()
    log("=== EqOpt-K full-episode calibration freeze: starting ===")

    psi_c = stage1_psi_c()
    cooldown = stage2_cooldown(psi_c)
    r_c = stage3_r_c(psi_c, cooldown)
    beta = stage4_beta(psi_c, cooldown, r_c)
    final = stage5_final_validation(psi_c, cooldown, r_c, beta)

    total_elapsed = time.time() - t_start
    log(f"\n=== ALL STAGES COMPLETE in {total_elapsed/3600:.2f} hours ===")
    log(f"Final calibrated constants: Psi_c={psi_c:.5f}  R_c={r_c:.5f}  "
        f"cooldown={cooldown}  beta={beta}")
    log("Significance results:")
    for strat, s in final['significance'].items():
        log(f"  eqopt_k vs {strat}: p={s['p']:.4f}  d={s['d']:.3f}  "
            f"{'SIGNIFICANT' if s['significant'] else 'not significant'}")


if __name__ == '__main__':
    main()
