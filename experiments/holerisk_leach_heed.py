"""
holerisk_leach_heed.py -- fair, non-circular hole-risk measurement for
LEACH and HEED, analogous to the hole-risk verification done for
adaptive mincost-v3.

Reusing R_h's hardcoded default threshold (0.82) for LEACH/HEED would
repeat the exact circularity mistake this project learned to avoid --
that value was never derived from either protocol's own dynamics.
This script instead:

  STAGE 1 & 2: non-circularly calibrates R_c specifically for LEACH's
               and HEED's own dynamics independently (a "hole" = a
               node death carrying above-average relay load AT THAT
               MOMENT, independent of any R_h threshold; R_c is set to
               the 10th percentile of R_h observed at those genuine
               hole events).
  STAGE 3 & 4: measures hole-risk-rounds fraction for each protocol
               using its own fairly-calibrated threshold, at N=100 and
               N=500 (10 seeds, 6000-round episodes, matching the
               mincost-v3 hole-risk verification methodology exactly).

Resumable: every stage's result cached to disk immediately. Safe to
interrupt and rerun -- completed stages are skipped.

Run in parallel with any other background job:
  caffeinate -i nohup python3 holerisk_leach_heed.py > holerisk_lh_log.txt 2>&1 &
"""
import numpy as np
import json
import time
import os
from energy import etx, erx
import metrics as M

OUTDIR = 'holerisk_leach_heed_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, L = 1.0, 100.0
BETA = 1.0
LONG_ROUNDS = 6000


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def save_json(name, obj):
    with open(OUTDIR + '/' + name + '.json', 'w') as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    path = OUTDIR + '/' + name + '.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _init_state(N, L, rng):
    pos = rng.uniform(0, L, size=(N, 2))
    E = np.full(N, E0, dtype=float)
    alive = np.ones(N, dtype=bool)
    return pos, E, alive


def leach_round(pos, E, alive, sink, p, round_in_epoch, rng, N, already_ch_this_epoch):
    alive_idx = np.where(alive)[0]
    if round_in_epoch == 0:
        already_ch_this_epoch[:] = False
    r = round_in_epoch % int(round(1.0 / p))
    thresh = p / (1 - p * r) if (1 - p * r) > 0 else 1.0
    is_ch = np.zeros(N, dtype=bool)
    for i in alive_idx:
        if already_ch_this_epoch[i]:
            continue
        if rng.random() < thresh:
            is_ch[i] = True
    if not is_ch[alive_idx].any():
        eligible = [i for i in alive_idx if not already_ch_this_epoch[i]]
        if eligible:
            is_ch[rng.choice(eligible)] = True
        else:
            is_ch[rng.choice(alive_idx)] = True
    already_ch_this_epoch[is_ch] = True
    ch_idx = alive_idx[is_ch[alive_idx]]
    member_idx = alive_idx[~is_ch[alive_idx]]
    assign = {}
    for m_ in member_idx:
        d = np.linalg.norm(pos[ch_idx] - pos[m_], axis=1)
        assign[m_] = ch_idx[np.argmin(d)]
    cost = np.zeros(N)
    L_load = np.zeros(N)
    for m_, ch in assign.items():
        d = np.linalg.norm(pos[m_] - pos[ch])
        cost[m_] += etx(d)
        cost[ch] += erx()
        L_load[ch] += 1.0
    for ch in ch_idx:
        d_sink = np.linalg.norm(pos[ch] - sink)
        cost[ch] += (1.0 + L_load[ch]) * etx(d_sink)
    return cost, L_load


def heed_round(pos, E, alive, sink, c_prob, rng, N):
    alive_idx = np.where(alive)[0]
    ch_prob_i = c_prob * (E[alive_idx] / (E0 + 1e-12))
    draws = rng.random(len(alive_idx))
    is_ch_mask = draws < np.clip(ch_prob_i, 0, 1)
    ch_idx = alive_idx[is_ch_mask]
    if len(ch_idx) == 0:
        ch_idx = np.array([alive_idx[np.argmax(E[alive_idx])]])
    ch_set = set(ch_idx.tolist())
    member_idx = np.array([i for i in alive_idx if i not in ch_set])
    assign = {}
    for m_ in member_idx:
        d = np.linalg.norm(pos[ch_idx] - pos[m_], axis=1)
        assign[m_] = ch_idx[np.argmin(d)]
    cost = np.zeros(N)
    L_load = np.zeros(N)
    for m_, ch in assign.items():
        d = np.linalg.norm(pos[m_] - pos[ch])
        cost[m_] += etx(d)
        cost[ch] += erx()
        L_load[ch] += 1.0
    for ch in ch_idx:
        d_sink = np.linalg.norm(pos[ch] - sink)
        cost[ch] += (1.0 + L_load[ch]) * etx(d_sink)
    return cost, L_load


def scan_structural_holes(protocol, N, seed, max_rounds=3000):
    rng = np.random.default_rng(seed)
    sink = np.array([L / 2, L / 2])
    pos, E, alive = _init_state(N, L, rng)
    rh_at_holes = []
    round_in_epoch = 0
    already_ch_this_epoch = np.zeros(N, dtype=bool)

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break
        if protocol == 'LEACH':
            cost, L_load = leach_round(pos, E, alive, sink, 0.05, round_in_epoch, rng, N, already_ch_this_epoch)
            round_in_epoch = (round_in_epoch + 1) % int(round(1.0 / 0.05))
        else:
            cost, L_load = heed_round(pos, E, alive, sink, 0.05, rng, N)

        E_new = np.clip(E - cost, 0.0, None)
        newly_dead = alive_idx[E_new[alive_idx] <= 0.0]

        if len(newly_dead) > 0:
            mean_load = L_load[alive_idx].mean() if len(alive_idx) else 0.0
            Psi_t, Ebar_t = M.psi(E_new)
            PsiN_t = M.psi_n(Psi_t, Ebar_t)
            alive_mask_now = alive.copy()
            _, Lambda_t = M.structural_covariance(E_new, L_load, alive_mask_now)
            Rh_t = M.risk(PsiN_t, Lambda_t, BETA)
            for nd in newly_dead:
                if L_load[nd] > mean_load:
                    rh_at_holes.append(float(Rh_t))

        E = E_new
        alive[newly_dead] = False
        if alive.sum() == 0:
            break
    return rh_at_holes


def measure_hole_risk(protocol, N, seed, threshold, max_rounds=LONG_ROUNDS):
    rng = np.random.default_rng(seed)
    sink = np.array([L / 2, L / 2])
    pos, E, alive = _init_state(N, L, rng)
    round_in_epoch = 0
    n_hole_rounds = 0
    n_rounds = 0
    already_ch_this_epoch = np.zeros(N, dtype=bool)

    for t in range(max_rounds):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break
        if protocol == 'LEACH':
            cost, L_load = leach_round(pos, E, alive, sink, 0.05, round_in_epoch, rng, N, already_ch_this_epoch)
            round_in_epoch = (round_in_epoch + 1) % int(round(1.0 / 0.05))
        else:
            cost, L_load = heed_round(pos, E, alive, sink, 0.05, rng, N)

        E = np.clip(E - cost, 0.0, None)
        newly_dead = alive_idx[E[alive_idx] <= 0.0]
        alive[newly_dead] = False

        Psi_t, Ebar_t = M.psi(E)
        PsiN_t = M.psi_n(Psi_t, Ebar_t)
        _, Lambda_t = M.structural_covariance(E, L_load, alive)
        Rh_t = M.risk(PsiN_t, Lambda_t, BETA)
        n_rounds += 1
        if Rh_t > threshold:
            n_hole_rounds += 1

        if alive.sum() == 0:
            break
    return n_hole_rounds, n_rounds


if __name__ == '__main__':
    for protocol in ['LEACH', 'HEED']:
        cache_name = 'calibration_' + protocol
        cached = load_json(cache_name)
        if cached:
            log('STAGE [' + protocol + ' calibration] [cached]: R_c = ' + str(round(cached['r_c'], 5)))
            continue

        log('STAGE [' + protocol + ' calibration]: non-circular structural-hole scan')
        t0 = time.time()
        all_rh = []
        for seed in range(15):
            all_rh += scan_structural_holes(protocol, N=100, seed=seed, max_rounds=3000)
        elapsed = time.time() - t0

        if not all_rh:
            log('  WARNING: no hole events captured for ' + protocol + ', falling back to Rc=0.82 default')
            r_c = 0.82
        else:
            r_c = float(np.percentile(all_rh, 10))
            log('  ' + protocol + ': ' + str(len(all_rh)) + ' hole events, 10th pct R_h = ' +
                str(round(r_c, 5)) + '  (median: ' + str(round(np.percentile(all_rh, 50), 5)) + ')  time=' +
                str(round(elapsed, 0)) + 's')

        save_json(cache_name, dict(r_c=r_c, n_holes=len(all_rh)))

    log('')

    for protocol in ['LEACH', 'HEED']:
        calib = load_json('calibration_' + protocol)
        threshold = calib['r_c']

        for N in [100, 500]:
            cache_name = 'holerisk_' + protocol + '_N' + str(N)
            cached = load_json(cache_name)
            if cached:
                log('STAGE [' + protocol + ' N=' + str(N) + '] [cached]: mean_hole_frac=' +
                    str(round(cached['mean_hole_frac'], 3)))
                continue

            log('STAGE [' + protocol + ' N=' + str(N) + ']: hole-risk measurement (threshold=' +
                str(round(threshold, 5)) + ')')
            t0 = time.time()
            hole_fracs = []
            for seed in range(10):
                n_hole, n_total = measure_hole_risk(protocol, N, seed, threshold)
                hole_fracs.append(n_hole / n_total if n_total > 0 else 0.0)
            elapsed = time.time() - t0

            result = dict(hole_fracs=hole_fracs, mean_hole_frac=float(np.mean(hole_fracs)),
                          threshold=threshold)
            save_json(cache_name, result)
            log('  ' + protocol + ' N=' + str(N) + ': mean_hole_fraction=' +
                str(round(result['mean_hole_frac'], 3)) + '  time=' + str(round(elapsed, 0)) + 's')

    log('')
    log('=== SUMMARY ===')
    for protocol in ['LEACH', 'HEED']:
        calib = load_json('calibration_' + protocol)
        log(protocol + ' calibrated R_c = ' + str(round(calib['r_c'], 5)))
        for N in [100, 500]:
            r = load_json('holerisk_' + protocol + '_N' + str(N))
            log('  N=' + str(N) + ': mean_hole_fraction=' + str(round(r['mean_hole_frac'], 3)))

    log('')
    log('reference (adaptive mincost-v3, own calibrated R_c=1.02488):')
    log('  N=100: hole_fraction=0.177')
    log('  N=500: hole_fraction=0.710')
    log('')
    log('=== DONE ===')
