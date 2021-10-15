import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput
from nengo.utils.ensemble import tuning_curves


def phase_automata(driving_symbol='0', number_of_symbols=3, id_of_starting_symbol=0, timesteps=9,
                   probability_of_transition=False):
    code = np.zeros((number_of_symbols, timesteps), dtype=float)
    code = code - 1
    state = id_of_starting_symbol
    i = 0
    while i < timesteps:
        u = True
        j = 0
        while j < number_of_symbols:
            if state == j and u:
                mu, sigma = 1, 0.5  # mean and standard deviation
                if probability_of_transition:
                    s = np.random.normal(mu, sigma)
                else:
                    s = 1
                if s >= 0.8:
                    if driving_symbol == '0':
                        state = (j + 1) % number_of_symbols
                    elif driving_symbol == '1':
                        state = ((j - 1) % number_of_symbols)
                    else:
                        state = id_of_starting_symbol
                        print("ILLEGAL DRIVING SYMBOL")
                    # print('passing to state ', state, 'driving symbol ', driving_symbol)
                    code[j][i] = 1
                    u = False
                else:
                    state = j
                    # print('staying in state', state)
            j += 1
        i += 1
    ending_state = state
    return code, ending_state


reseed = 91274 # 91264 # 91254  # 91273
good = False
number_of_samples = 6
ts = 45  # number of timesteps to hold a driving symbol constant for.

threeChannelsOF1, end_channel = phase_automata(driving_symbol="1", probability_of_transition=True, timesteps=ts)
padded_zeros = np.zeros((3, ts * 2), dtype=float)
threeChannels1 = np.concatenate((threeChannelsOF1, padded_zeros), axis=1)
threeChannelsOF0, end_channel0 = phase_automata(driving_symbol="0", probability_of_transition=True, timesteps=ts)
threeChannels0 = np.concatenate((threeChannelsOF0, padded_zeros), axis=1)
labels0 = np.zeros((ts * 3, 1), dtype=float)
labels1 = np.ones((ts * 3, 1), dtype=float)
bothLabels = np.concatenate((labels0, labels1), axis=0)
bothPatterns = np.concatenate((threeChannels0, threeChannels1), axis=1)

plt.figure()
plt.title("Input Pattern and Label Example ProbTran:True PaddedZeros:True")
t = np.arange(ts*3*2)
plt.xlabel('ts (Timestep)' )
plt.ylabel('Value')
plt.plot(t,bothPatterns[0],color="blue",label="pattern C1")
plt.plot(t,bothPatterns[1],color="green",label="pattern C2")
plt.plot(t,bothPatterns[2],color="orange",label="pattern C3")
plt.plot(t,bothLabels+2.0,color="black",label="label")
plt.legend()
plt.savefig("fig/input_pattern_example_probTran_True_padded_zeros_true.png")
i = 1
while i < number_of_samples:
    threeChannelsOF1, end_channel = phase_automata(driving_symbol="1", probability_of_transition=True, timesteps=ts)
    padded_zeros = np.zeros((3, ts * 2), dtype=float)
    threeChannels1 = np.concatenate((threeChannelsOF1, padded_zeros), axis=1)
    threeChannelsOF0, end_channel0 = phase_automata(driving_symbol="0", probability_of_transition=True, timesteps=ts)
    threeChannels0 = np.concatenate((threeChannelsOF0, padded_zeros), axis=1)
    labels0 = np.zeros((ts * 3, 1), dtype=float)
    labels1 = np.ones((ts * 3, 1), dtype=float)
    bothLabelsB = np.concatenate((labels0, labels1), axis=0)
    bothPatternsB = np.concatenate((threeChannels0, threeChannels1), axis=1)
    bothLabelsA = np.copy(bothLabels)
    bothPatternsA = np.copy(bothPatterns)
    bothLabels = np.concatenate((bothLabelsA, bothLabelsB), axis=0)
    bothPatterns = np.concatenate((bothPatternsA, bothPatternsB), axis=1)
    i += 1
    
tC = bothPatterns.transpose((1, 0))
labels = bothLabels


while not good:
    model = nengo.Network(label='Three Layers with feedback', seed=reseed)
    num_neurons_l1 = 4
    num_neurons_l2 = 6
    num_neurons_l3 = 2
    with model:
        with model:
            layer1 = nengo.Ensemble(
                num_neurons_l1,  # Number of neurons
                dimensions=3,  # each neuron is connected to all (3) input channels.
                # Set intercept to 0.5
                neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                max_rates=Uniform(1 / 6e-2, 1 / 6e-2),  # Set the maximum firing rate of the neuron 500Mhz
            )

            layer2 = nengo.Ensemble(
                num_neurons_l2,  # Number of neurons
                dimensions=3,
                # Set intercept to 0.5
                neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                max_rates=Uniform(1 / 6e-2, 1 / 6e-2),  # Set the maximum firing rate of the neuron 500Mhz
            )

            layer3 = nengo.Ensemble(
                num_neurons_l3,  # Number of neurons
                dimensions=3,
                # Set intercept to 0.5
                neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                max_rates=Uniform(1 / 6e-2, 1 / 6e-2),  # Set the maximum firing rate of the neuron 500Mhz
            )

    with model:
        input_signal = nengo.Node(PresentInput(tC, presentation_time=3e-2))
        input_keys = nengo.Node(PresentInput(labels, presentation_time=3e-2))

    with model:
        nengo.Connection(input_signal, layer1, synapse=None)
        nengo.Connection(layer1, layer2, synapse=1e-1)
        nengo.Connection(layer2, layer1, synapse=ts*1e-2)
        nengo.Connection(layer2, layer3, synapse=1e-2)
        nengo.Connection(layer3, layer2, synapse=ts*1e-1)

    simT = 180
    with model:
        input_probe = nengo.Probe(input_signal)  # The original input
        spikes = nengo.Probe(layer1.neurons)  # Raw spikes from each neuron
        # Subthreshold soma voltages of the neurons
        
        #voltage = nengo.Probe(layer1.neurons, 'voltage')
        # Spikes filtered by a 10ms post-synaptic filter
        filteredl1 = nengo.Probe(layer1.neurons, 'voltage', synapse=2e-2)
        filteredl2 = nengo.Probe(layer2.neurons, 'voltage', synapse=2e-2)
        filteredl3 = nengo.Probe(layer3.neurons, 'voltage', synapse=2e-2)
        
        # Setup probes
        p_keys = nengo.Probe(input_keys, synapse=None, label="p_keys")
        p_values = nengo.Probe(input_signal, synapse=None, label="p_values")
        
    with nengo.Simulator(model) as sim:  # Create a simulator
        sim.run(simT)  # Run 

    t = sim.trange()

    plot_range = -1  # index

    train = t <= simT / 2
    test = ~train

    # plt.figure()
    # plt.title("Value Error During Training")
    # plt.plot(t[train], sim.data[p_error][train])
    # plt.show()

    plt.figure()
    plt.title("Filtered L1 output")
    # print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
    # plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
    plt.plot(t, sim.data[p_keys] + 0, color="g")
    plt.plot(t, sim.data[input_probe] + 2)
    print(np.shape(sim.data[filteredl1]))
    plt.plot(t, sim.data[filteredl1][0:, 0:1] + 3, color="b")
    plt.plot(t, sim.data[filteredl1][0:, 1:2] + 4, color="y")
    plt.plot(t, sim.data[filteredl1][0:, 2:3] + 5, color="k")
    plt.plot(t, sim.data[filteredl1][0:, 3:4] + 6, color="m")
    plt.savefig('fig/3layersfeedbck_neuronsl1_4.png')
#    plt.plot(t, sim.data[filtered][0:, 4:5] + 7, color="r")
#    plt.plot(t, sim.data[filtered][0:, 5:6] + 8, color="g")
#    plt.plot(t, sim.data[filtered][0:, 6:7] + 9, color="#aabbcc")

    plt.figure()
    plt.title("Filtered L2 output")
    # print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
    # plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
    plt.plot(t, sim.data[p_keys] + 0, color="g")
    plt.plot(t, sim.data[input_probe] + 2)
    print(np.shape(sim.data[filteredl2]))
    plt.plot(t, sim.data[filteredl2][0:, 0:1] + 3, color="b")
    plt.plot(t, sim.data[filteredl2][0:, 1:2] + 4, color="y")
    plt.plot(t, sim.data[filteredl2][0:, 2:3] + 5, color="k")
    plt.plot(t, sim.data[filteredl2][0:, 3:4] + 6, color="m")
    plt.plot(t, sim.data[filteredl2][0:, 4:5] + 7, color="r")
    plt.plot(t, sim.data[filteredl2][0:, 5:6] + 8, color="g")
    plt.savefig('fig/3layersfeedbck_neuronsl2_6.png')

    plt.figure()
    plt.title("Filtered L3 output")

    plt.plot(t[test], sim.data[p_keys][test] + 0, color="g")
    plt.plot(t[test], sim.data[input_probe][test] + 2)
    print(np.shape(sim.data[filteredl2]))
    plt.plot(t[test], sim.data[filteredl2][test][0:, 0:1] + 3, color="b")
    plt.plot(t[test], sim.data[filteredl2][test][0:, 1:2] + 4, color="y")
    plt.savefig('fig/3layersfeedbck_neuronsl3_2.png')

    plt.clf()
    
    i = 0
    
    best_neuron_value = np.abs(np.sum(sim.data[filteredl3][0:, 0:0 + 1] - sim.data[p_keys]))
    print(best_neuron_value)
    best_neuron_index = 0
    while i < num_neurons_l3:
        sum = np.abs(np.sum(sim.data[filteredl3][0:, i:i + 1] - sim.data[p_keys]))
        # print(i, sum)

        if (sum < best_neuron_value):
            best_neuron_index = i
            best_neuron_value = sum
        print(best_neuron_value, best_neuron_index, reseed)
        i += 1
    if (best_neuron_value < 4000):
        good = True
    else:
        reseed += 1
    good = True

plt.figure()
plt.title("Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[p_keys] + 0, color="g")
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[filteredl3][0:, best_neuron_index:best_neuron_index + 1] + 3, color="black")
plt.savefig('fig/3layersfeedbck_best_neuron_index_l3.png')

