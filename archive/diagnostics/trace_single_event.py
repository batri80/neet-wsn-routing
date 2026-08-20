"""
Reproduce the exact state up to round 101 and inspect the single local
reattachment decision in full detail -- which orphan, which candidate
parents were available, what each would cost, and which was chosen.
"""
from network import Network
from energy import epoch_cost
import controller as C
import metrics as M
from simulate import _mark_unreachable_as_disconnected
import numpy as np

rng = np.random.default_rng(2)
net = Network(N=100, R_c=35.0, E0=1.0, rng=rng)
Psi_c, Rc_thresh, beta = 0.065, 0.664, 0.5

for t in range(102):
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

    if t == 101 and newly_dead:
        for dn in newly_dead:
            print(f"Dead node: {dn}  position={net.pos[dn]}  was at depth={net.depth[dn]}")
            orphans = net.orphans_of(dn)
            print(f"Orphans: {orphans}")
            for orph in orphans:
                if not net.alive[orph]:
                    continue
                subtree = net.subtree_of(orph)
                print(f"\n  Orphan {orph}: position={net.pos[orph]}  subtree_size={len(subtree)}  own_load_contribution(w)={net.w[orph]}")
                L_now, _ = net.loads()
                print(f"  Orphan's pre-death relay load L={L_now[orph]:.2f}")

                candidates = C.generate_local_candidates(net, orph)
                print(f"  {len(candidates)} candidates available:")
                for c in candidates:
                    if not c:
                        continue
                    new_parent = c[orph]
                    d = net.dist(orph, new_parent) if new_parent >= 0 else net.dist_to_sink(orph)
                    ev_c = C.evaluate_diff(net, c, beta)
                    L_parent_before = L_now[new_parent] if new_parent < len(L_now) else 0
                    print(f"    -> parent {new_parent}: dist={d:.1f}m  "
                          f"parent's_existing_load={L_parent_before:.2f}  "
                          f"parent's_energy={net.E[new_parent]:.4f}  "
                          f"resulting_Psi={ev_c['Psi']:.6f}")

                ev, ncand = C.reconfigure(net, 'eqopt', 'local', beta, rng, orphan=orph)
                chosen = ev['parent'][orph]
                print(f"  EqOpt CHOSE parent: {chosen}")
                net.parent = ev['parent']
                _mark_unreachable_as_disconnected(net)
    elif newly_dead:
        for dn in newly_dead:
            for orph in net.orphans_of(dn):
                if net.alive[orph]:
                    ev, ncand = C.reconfigure(net, 'eqopt', 'local', beta, rng, orphan=orph)
                    net.parent = ev['parent']
                    _mark_unreachable_as_disconnected(net)
    else:
        m_before = M.compute_all(net, beta=beta)
        if m_before['Psi'] > Psi_c or m_before['R_h'] > Rc_thresh:
            ev, ncand = C.reconfigure(net, 'eqopt', 'global', beta, rng)
            net.parent = ev['parent']
            _mark_unreachable_as_disconnected(net)

    if net.alive.sum() == 0:
        break
