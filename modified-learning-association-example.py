import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput



num_items = 6

num_periods = 2
d_key = 2
d_value = 3

rng = np.random.RandomState(seed=7)
keysA = nengo.dists.UniformHypersphere(surface=True).sample(2, d_key, rng=rng)
keys = nengo.dists.UniformHypersphere(surface=True).sample(num_items, d_key, rng=rng)
values = nengo.dists.UniformHypersphere(surface=False).sample(
    num_items, d_value, rng=rng
)

keys[0][0] = keysA[0][0]
keys[0][1] = keysA[0][1]
keys[1][0] = keysA[0][0]
keys[1][1] = keysA[0][1]
keys[2][0] = keysA[0][0]
keys[2][1] = keysA[0][1]
keys[3][0] = keysA[1][0]
keys[3][1] = keysA[1][1]
keys[4][0] = keysA[1][0]
keys[4][1] = keysA[1][1]
keys[5][0] = keysA[1][0]
keys[5][1] = keysA[1][1]
print(keys)

print(np.shape(values))

values[0][0] = 1.0
values[0][1] = 0.0
values[0][2] = 0.0

#print(values)

values[1][0] = 0.0
values[1][1] = 1.0
values[1][2] = 0.0

#print(values)

values[2][0] = 0.0
values[2][1] = 0.0
values[2][2] = 1.0

#print(values)


values[3][0] = 0.0
values[3][1] = 1.0
values[3][2] = 0.0

#print(values)

values[4][0] = 1.0
values[4][1] = 0.0
values[4][2] = 0.0

values[5][0] = 0.0
values[5][1] = 0.0
values[5][2] = 1.0

#print(values)


intercept = (np.dot(keysA, keysA.T) - np.eye(num_periods)).flatten().max()
print(f"Intercept: {intercept}")
intercept = 0.0

def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError(f"dt ({dt}) does not divide period ({period})")

    def f(t):
        i = int(round((t - dt) / dt))  # t starts at dt
        return x[int(i / i_every) % len(x)]

    return f

# Model constants
n_neurons = 200
dt = 0.001
period = 0.3
T = period * num_items * 2

# Model network
model = nengo.Network()
with model:

    # Create the inputs/outputs
    stim_keys = nengo.Node(output=cycle_array(keys, period, dt))
    stim_values = nengo.Node(output=cycle_array(values, period, dt))
    learning = nengo.Node(output=lambda t: -int(t >= T / 2))
    recall = nengo.Node(size_in=d_value)

    # Create the memory
    memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons)

    # Learn the encoders/keys
    voja = nengo.Voja(learning_rate=5e-2, post_synapse=None)
    conn_in = nengo.Connection(stim_keys, memory, synapse=None, learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(
        memory,
        recall,
        learning_rule_type=nengo.PES(1e-3),
        function=lambda x: np.zeros(d_value),
    )

    # Create the error population
    error = nengo.Ensemble(n_neurons, d_value)
    nengo.Connection(
        learning, error.neurons, transform=[[10.0]] * n_neurons, synapse=None
    )

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(stim_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(stim_keys, synapse=None)
    p_values = nengo.Probe(stim_values, synapse=None)
    p_learning = nengo.Probe(learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_recall = nengo.Probe(recall, synapse=None)
    p_encoders = nengo.Probe(conn_in.learning_rule, "scaled_encoders")


with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)
t = sim.trange()

plt.figure()
plt.title("Keys")
plt.plot(t, sim.data[p_keys])
plt.ylim(-1, 1)
plt.show()
plt.figure()
plt.title("Values")
plt.plot(t, sim.data[p_values])
plt.ylim(-1, 1)
plt.show()
plt.figure()
plt.title("Learning")
plt.plot(t, sim.data[p_learning])
plt.ylim(-1.2, 0.2)
plt.show()

train = t <= T / 2
test = ~train

plt.figure()
plt.title("Value Error During Training")
plt.plot(t[train], sim.data[p_error][train])
plt.show()

plt.figure()
plt.title("Value Error During Recall")
plt.plot(t[test], sim.data[p_recall][test])# - sim.data[p_values][test])
plt.show()
scale = (sim.data[memory].gain / memory.radius)[:, np.newaxis]


def plot_2d(text, xy):
    plt.figure()
    plt.title(text)
    plt.scatter(xy[:, 0], xy[:, 1], label="Encoders")
    plt.scatter(keys[:, 0], keys[:, 1], c="red", s=150, alpha=0.6, label="Keys")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 2)
    plt.legend()
    plt.gca().set_aspect("equal")


plot_2d("Before", sim.data[p_encoders][0].copy() / scale)
plt.show()
plot_2d("After", sim.data[p_encoders][-1].copy() / scale)
plt.show()