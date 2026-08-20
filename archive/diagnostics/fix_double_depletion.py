import re

with open('simulate.py') as f:
    content = f.read()

old = """def _apply_evaluation(net, ev):
    net.parent = ev['parent']
    net.E = ev['E_plus']
    _mark_unreachable_as_disconnected(net)"""

new = """def _apply_evaluation(net, ev):
    # IMPORTANT: only adopt the candidate's TOPOLOGY, not its projected
    # energy. ev['E_plus'] is a hypothetical one-round-ahead projection
    # used purely to compare candidates (Theorem 1's drift comparison);
    # writing it back into net.E would double-charge energy this round
    # (once for the real depletion already applied above, again here).
    # The real depletion under the new tree happens naturally next round.
    net.parent = ev['parent']
    _mark_unreachable_as_disconnected(net)"""

assert old in content, "pattern not found -- check simulate.py manually"
content = content.replace(old, new)
with open('simulate.py', 'w') as f:
    f.write(content)
print("patched _apply_evaluation")
