"""
Round-0 divergence -- likely a network CONSTRUCTION difference (node
positions), not a dynamics difference. Compare initial node positions
directly between the two construction paths.
"""
import numpy as np
from network import Network

rng1 = np.random.default_rng(0)
net1 = Network(N=100, R_c=35.0, E0=1.0, rng=rng1)

rng2 = np.random.default_rng(0)
net2 = Network(N=100, R_c=35.0, E0=1.0, rng=rng2)

print(f"positions identical? {np.array_equal(net1.pos, net2.pos)}")
print(f"initial E identical? {np.array_equal(net1.E, net2.E)}")
print(f"sink identical? {np.array_equal(net1.sink, net2.sink)}")
print(f"disconnected count: net1={net1.disconnected.sum()}  net2={net2.disconnected.sum()}")

# now check whether run_episode's internal Network() call differs in any way
import inspect
from simulate import run_episode
src = inspect.getsource(run_episode)
# print just the Network(...) construction line
for line in src.split('\n'):
    if 'Network(' in line:
        print(f"\nrun_episode's Network() call: {line.strip()}")
