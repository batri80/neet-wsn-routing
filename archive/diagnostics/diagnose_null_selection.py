import numpy as np
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M

def count_null_choices(strategy, N=60, E0=1.0, R_c=35.0, max_rounds=1500, seed=0, beta=0.5):
    rng = np.random.default_rng(seed)
    net = Network(N=N, R_c=R_c, E0=E0, rng=rng)
    null_chosen_with_alternatives = 0
    total_local_triggers_with_alternatives = 0

    for t in range(max_rounds):
        connected = net.alive_connected()
        if len(connected) == 0:
            break
        cost, L = epoch_cost(net)
        net.E = np.clip(net.E - cost, 0.0, None)
        newly_dead = [i for i in connected if net.E[i] <= 0.0]
        for i in newly_dead:
            net.kill(i)

        if strategy != 'static' and newly_dead:
            for dn in newly_dead:
                for orph in net.orphans_of(dn):
                    if not net.alive[orph]:
                        continue
                    candidates = C.generate_local_candidates(net, orph)
                    if len(candidates) > 1:  # a real alternative to null existed
                        total_local_triggers_with_alternatives += 1
                        chosen_ev, _ = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph)
                        if chosen_ev['parent'][orph] == net.parent[orph]:  # stayed as before = null was picked
                            null_chosen_with_alternatives += 1
                    ev, _ = C.reconfigure(net, strategy, 'local', beta, rng, orphan=orph)
                    net.parent = ev['parent']

        m = M.compute_all(net, beta=beta)
        if strategy != 'static' and (m['Psi'] > 0.3 or m['R_h'] > 999):  # global trigger suppressed
            pass
        if net.alive.sum() == 0:
            break
    return null_chosen_with_alternatives, total_local_triggers_with_alternatives

for strat in ['random', 'eqopt']:
    nulls, totals = [], []
    for seed in range(5):
        n, tot = count_null_choices(strat, seed=seed)
        nulls.append(n); totals.append(tot)
    print(f"{strat:8s}: null chosen {sum(nulls)}/{sum(totals)} times an alternative existed "
          f"({100*sum(nulls)/max(sum(totals),1):.1f}%)")
