"""
calibrate_v3.py -- v3-specific calibration, closing Open Item 2 of the
v3 model specification. v2's borrowed constants (Psi_c=0.065,
Rc_thresh=0.664, beta=0.5) were used for the N-sweep validation
(Section 8 of the spec); this derives v3's own values using the same
non-circular, staged methodology used throughout the v2 investigation
(model reference doc Sections 14.4-14.5, 14.10).

Resumable: each stage's result is saved to disk immediately and
skipped on rerun if already present.
"""
import numpy as np
import json
import time
import os
from network import Network
from flow_controller import solve_flow_lp
from flow_simulate import run_episode_v3
import metrics as M

OUTDIR = 'v3_calibration_results'
os.makedirs(OUTDIR, exist_ok=True)

N, E0, R_c_comm = 100, 1.0, 35.0
MAX_ROUNDS = 3000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def save_json(name, obj):
    with open(f'{OUTDIR}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    path = f'{OUTDIR}/{name}.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ======================================================================
# STAGE 1: Psi_c knee-finding at full episode length.
# ======================================================================
def stage1_psi_c():
    cached = load_json('stage1_psi_c')
    if cached:
        log(f"STAGE 1 [cached]: Psi_c knee = {cached['psi_c']:.5f}")
        return cached['psi_c']

    log("STAGE 1: finding Psi_c knee at full episode length (3000 rounds)")
    checkpoints = [50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000]
    N_SEEDS_STAGE1 = 10
    results = {c: [] for c in checkpoints}

    for seed in range(N_SEEDS_STAGE1):
        res = run_episode_v3(N=N, E0=E0, R_c=R_c_comm, max_rounds=MAX_ROUNDS,
                              Psi_c=999, Rc_thresh=999, seed=seed, record_series=True)
        psi = res['series']['Psi']
        for c in checkpoints:
            if c < len(psi):
                results[c].append(psi[c])

    means, ns = {}, {}
    log("  round   mean_Psi   n")
    for c in checkpoints:
        vals = results[c]
        if vals:
            means[c] = float(np.mean(vals))
            ns[c] = len(vals)
            log(f"  {c:5d}   {np.mean(vals):.5f}   {len(vals)}")

    ckpts_full = sorted([c for c in means if ns[c] == N_SEEDS_STAGE1])
    if len(ckpts_full) < 3:
        log("  WARNING: matched window too short, using all available checkpoints"
            " (results may include survivorship bias)")
        ckpts_full = sorted(means.keys())
    log(f"  matched window (full n={N_SEEDS_STAGE1}): rounds {ckpts_full}")

    deltas_per_round = []
    for i in range(len(ckpts_full) - 1):
        span = ckpts_full[i+1] - ckpts_full[i]
        d = (means[ckpts_full[i+1]] - means[ckpts_full[i]]) / span
        deltas_per_round.append(d)

    if len(deltas_per_round) < 2:
        knee_round = ckpts_full[-1]
        psi_c = means[knee_round]
        log(f"  (too few matched-window points for peak-detection; using last matched point)")
    else:
        peak_idx = int(np.argmax(deltas_per_round))
        peak_rate = deltas_per_round[peak_idx]
        knee_idx = peak_idx
        found_deceleration = False
        for i in range(peak_idx + 1, len(deltas_per_round)):
            if deltas_per_round[i] < 0.5 * peak_rate:
                knee_idx = i
                found_deceleration = True
                break
        if not found_deceleration:
            log("  NOTE: no deceleration below 50% of peak rate found within the "
                "matched window -- growth was still accelerating or steady "
                "throughout; using the last matched-window point as a "
                "conservative (safe, not artificially low) Psi_c.")
            knee_idx = len(deltas_per_round) - 1
        knee_round = ckpts_full[knee_idx + 1]
        psi_c = means[knee_round]
        log(f"  peak growth rate at round {ckpts_full[peak_idx+1]} "
            f"({peak_rate:.6f}/round); knee at round {knee_round}")

    log(f"STAGE 1 result: Psi_c = {psi_c:.5f} (at round {knee_round})")
    save_json('stage1_psi_c', {'psi_c': psi_c, 'knee_round': knee_round,
                                 'checkpoints': means, 'matched_window': ckpts_full})
    return psi_c


# ======================================================================
# STAGE 2: R_c recalibration, non-circular structural-hole methodology.
# ======================================================================
def stage2_r_c(psi_c):
    cached = load_json('stage2_r_c')
    if cached:
        log(f"STAGE 2 [cached]: R_c = {cached['r_c']:.5f}")
        return cached['r_c']

    log("STAGE 2: recalibrating R_c (non-circular structural-hole method)")

    def scan_structural_holes(seed, beta=0.5, max_rounds=MAX_ROUNDS):
        rng = np.random.default_rng(seed)
        net = Network(N=N, R_c=R_c_comm, E0=E0, rng=rng)
        rh_at_holes = []
        current_solution = None

        for t in range(max_rounds):
            alive_idx = np.where(net.alive)[0]
            if len(alive_idx) == 0:
                break
            if current_solution is None:
                current_solution = solve_flow_lp(net, beta=beta)
                net.disconnected[:] = False
                for i in current_solution['disconnected']:
                    net.disconnected[i] = True

            cost = current_solution['cost']
            net.E = np.clip(net.E - cost, 0.0, None)
            newly_dead = [i for i in alive_idx if net.E[i] <= 0.0]

            if newly_dead:
                load_arr = current_solution.get('load', np.zeros(net.N))
                mean_load = load_arr[alive_idx].mean() if len(alive_idx) else 0.0
                Psi_t, Ebar_t = M.psi(net.E)
                PsiN_t = M.psi_n(Psi_t, Ebar_t)
                alive_mask = net.alive & ~net.disconnected
                _, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
                Rh_t = M.risk(PsiN_t, Lambda_t, beta)
                for nd in newly_dead:
                    if load_arr[nd] > mean_load:
                        rh_at_holes.append(Rh_t)
                    net.kill(nd)
                current_solution = solve_flow_lp(net, beta=beta)
                net.disconnected[:] = False
                for i in current_solution['disconnected']:
                    net.disconnected[i] = True
            else:
                Psi_t, Ebar_t = M.psi(net.E)
                PsiN_t = M.psi_n(Psi_t, Ebar_t)
                alive_mask = net.alive & ~net.disconnected
                load_arr = current_solution.get('load', np.zeros(net.N))
                _, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
                Rh_t = M.risk(PsiN_t, Lambda_t, beta)
                if Psi_t > psi_c or Rh_t > 0.99:
                    current_solution = solve_flow_lp(net, beta=beta)
                    net.disconnected[:] = False
                    for i in current_solution['disconnected']:
                        net.disconnected[i] = True

            if net.alive.sum() == 0:
                break
        return rh_at_holes

    all_rh = []
    t0 = time.time()
    for seed in range(15):
        all_rh += scan_structural_holes(seed)
    elapsed = time.time() - t0

    log(f"  structural hole events captured: {len(all_rh)}  (time={elapsed:.0f}s)")
    if not all_rh:
        log("  WARNING: no hole events captured, falling back to v2's R_c=0.664")
        r_c = 0.664
    else:
        r_c = float(np.percentile(all_rh, 10))
        log(f"  10th percentile R_h at holes: {r_c:.5f}  (median: {np.percentile(all_rh, 50):.5f})")

    save_json('stage2_r_c', {'r_c': r_c, 'n_holes': len(all_rh)})
    return r_c


# ======================================================================
# STAGE 3: beta re-verification, bounded range.
# ======================================================================
def stage3_beta(psi_c, r_c):
    cached = load_json('stage3_beta')
    if cached:
        log(f"STAGE 3 [cached]: beta = {cached['beta']}")
        return cached['beta']

    log("STAGE 3: re-verifying beta (bounded range, avoiding the v2 unbounded-sweep pitfall)")

    def scan(beta, seed, max_rounds=MAX_ROUNDS):
        rng = np.random.default_rng(seed)
        net = Network(N=N, R_c=R_c_comm, E0=E0, rng=rng)
        rh_holes, rh_normal = [], []
        current_solution = None
        for t in range(max_rounds):
            alive_idx = np.where(net.alive)[0]
            if len(alive_idx) == 0:
                break
            if current_solution is None:
                current_solution = solve_flow_lp(net, beta=beta)
                net.disconnected[:] = False
                for i in current_solution['disconnected']:
                    net.disconnected[i] = True

            cost = current_solution['cost']
            net.E = np.clip(net.E - cost, 0.0, None)
            newly_dead = [i for i in alive_idx if net.E[i] <= 0.0]
            load_arr = current_solution.get('load', np.zeros(net.N))
            Psi_t, Ebar_t = M.psi(net.E)
            PsiN_t = M.psi_n(Psi_t, Ebar_t)
            alive_mask = net.alive & ~net.disconnected
            _, Lambda_t = M.structural_covariance(net.E, load_arr, alive_mask)
            Rh_t = M.risk(PsiN_t, Lambda_t, beta)

            is_hole = False
            if newly_dead:
                mean_load = load_arr[alive_idx].mean() if len(alive_idx) else 0.0
                for nd in newly_dead:
                    if load_arr[nd] > mean_load:
                        is_hole = True
                    net.kill(nd)
            (rh_holes if is_hole else rh_normal).append(Rh_t)

            if newly_dead or Psi_t > psi_c or Rh_t > r_c:
                current_solution = solve_flow_lp(net, beta=beta)
                net.disconnected[:] = False
                for i in current_solution['disconnected']:
                    net.disconnected[i] = True

            if net.alive.sum() == 0:
                break
        return rh_holes, rh_normal

    best_beta, best_sep = 0.5, -np.inf
    results = {}
    for beta in [0.5, 1.0, 1.5]:
        all_h, all_n = [], []
        for seed in range(10):
            h, n = scan(beta, seed)
            all_h += h; all_n += n
        if all_h and all_n:
            sep = np.mean(all_h) - np.mean(all_n)
            results[beta] = sep
            log(f"  beta={beta}: separation={sep:.4f}")
            if sep > best_sep:
                best_sep = sep
                best_beta = beta

    log(f"STAGE 3 result: beta = {best_beta}")
    save_json('stage3_beta', {'beta': best_beta, 'all_separations': results})
    return best_beta


# ======================================================================
# STAGE 4: Final validation.
# ======================================================================
def stage4_validation(psi_c, r_c, beta):
    cached = load_json('stage4_validation')
    if cached:
        log("STAGE 4 [cached] -- see v3_calibration_results/stage4_validation.json")
        return cached

    log(f"STAGE 4: final validation with Psi_c={psi_c:.4f}, R_c={r_c:.4f}, beta={beta}")
    N_SEEDS = 15
    results = {}
    for test_N in [100, 500]:
        t0 = time.time()
        hnds = []
        for seed in range(N_SEEDS):
            res = run_episode_v3(N=test_N, E0=E0, R_c=R_c_comm, Psi_c=psi_c,
                                  Rc_thresh=r_c, beta=beta, max_rounds=MAX_ROUNDS,
                                  seed=seed, record_series=False)
            hnds.append(res['HND'])
        elapsed = time.time() - t0
        results[test_N] = hnds
        log(f"  N={test_N}: mean HND={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
            f"n={N_SEEDS}  time={elapsed:.0f}s")

    log("\n  Reference (borrowed v2 constants, from Section 8 of the v3 spec):")
    log("  N=100: mean=629.4 (n=20)   N=500: mean=434.7 (n=12)")

    save_json('stage4_validation', results)
    return results


# ======================================================================
def main():
    t_start = time.time()
    log("=== v3 calibration pipeline: starting ===")

    psi_c = stage1_psi_c()
    r_c = stage2_r_c(psi_c)
    beta = stage3_beta(psi_c, r_c)
    stage4_validation(psi_c, r_c, beta)

    total_elapsed = time.time() - t_start
    log(f"\n=== ALL STAGES COMPLETE in {total_elapsed/3600:.2f} hours ===")
    log(f"Final calibrated constants: Psi_c={psi_c:.5f}  R_c={r_c:.5f}  beta={beta}")


if __name__ == '__main__':
    main()
