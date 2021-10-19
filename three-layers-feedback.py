import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput
from nengo.utils.ensemble import tuning_curves

def preset_timing_plot():
    print("sim.dt",sim.dt)
    print("Symbol presentation time",pt)
    print("Label length", label_length)
    print("layer 1 Tau RC",layer1.neuron_type.tau_rc)
    print("layer1 refractory period",layer1.neuron_type.tau_ref)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    people = ('Sim dt', 'Pulse Time', 'Pulse Spacing', 'Rotation Time', 'Label Time', 'L1 Tau RC', 'L1 Tau Ref', 
    'L2 Tau RC', 
              "L2 Tau Ref", 
              "L3 Tau RC", "L3 Tau REf")
    y_pos = np.arange(len(people))
    print (y_pos)
    performance = np.zeros(len(people),dtype=float)
    lbs = [1,2,3,4,5,6,7,8,9,10]
    #performance = 3 + 10 * np.random.rand(len(people))
    performance[0] = sim.dt
    performance[1] = pt
    performance[2] = 0.0
    performance[3] = ts*pt
    performance[4] = label_length*pt
    performance[5] = layer1.neuron_type.tau_rc
    performance[6] = layer1.neuron_type.tau_ref
    performance[7] = layer2.neuron_type.tau_rc
    performance[8] = layer2.neuron_type.tau_ref
    performance[9] = layer3.neuron_type.tau_rc
    performance[10] = layer3.neuron_type.tau_ref
    ax.barh(y_pos, performance)
    ax.set_yticks(y_pos)
    #plt.xscale("log")
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time (s)')
    ax.set_title('Preset Timing Parameters')

    plt.show()

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


reseed = 91332 #91323 #91280 #91274 # 91264 # 91254  # 91273
good = False
number_of_samples = 64
ts = 9  # number of timesteps to hold a driving symbol constant for.
pb = False
pt = 3e-2 # seconds to present each step of input
label_length = ts * 3 # In Timesteps, multiply by dt to get actual length of time
threeChannelsOF1, end_channel = phase_automata(driving_symbol="1", probability_of_transition=pb, timesteps=ts)
padded_zeros = np.zeros((3, ts * 2), dtype=float)
padded_zeros = padded_zeros - 1.0
threeChannels1 = np.concatenate((threeChannelsOF1, padded_zeros), axis=1)
threeChannelsOF0, end_channel0 = phase_automata(driving_symbol="0", probability_of_transition=pb, timesteps=ts)
threeChannels0 = np.concatenate((threeChannelsOF0, padded_zeros), axis=1)
labels0 = np.zeros((label_length, 1), dtype=float)
labels1 = np.ones((label_length, 1), dtype=float)
bothLabels = np.concatenate((labels0, labels1), axis=0)
bothPatterns = np.concatenate((threeChannels0, threeChannels1), axis=1)

plt.figure()
plt.title("Input Pattern and Label Example ProbTran:"+str(pb)+" PaddedZeros:True")
t = np.arange(ts*3*2)
plt.xlabel('ts (Timestep)' )
plt.ylabel('Value')
plt.plot(t,bothPatterns[0],color="blue",label="pattern C1")
plt.plot(t,bothPatterns[1],color="green",label="pattern C2")
plt.plot(t,bothPatterns[2],color="orange",label="pattern C3")
plt.plot(t,bothLabels+2.0,color="black",label="label")
plt.legend()
plt.savefig("fig/input_pattern_example_probTran_"+str(pb)+"_padded_zeros_true.png")
i = 1
bitstring = "0101100001010000000111111101010101010011001100011100101010000111100000000000000000000000000000000000000000000000000000000000000000"
w = 0
while i < number_of_samples:
    threeChannelsOF1, end_channel = phase_automata(driving_symbol=bitstring[w], probability_of_transition=pb, 
                                                                  timesteps=ts)
    threeChannels1 = np.concatenate((threeChannelsOF1, padded_zeros), axis=1)
    if (bitstring[w] == "0"):
        labels0 = np.zeros((ts * 3, 1), dtype=float)
    else:
        labels0 = np.ones((ts * 3, 1), dtype=float)
    w += 1
    w  = w % len(bitstring)
    threeChannelsOF0, end_channel0 = phase_automata(driving_symbol=bitstring[w], probability_of_transition=pb,
                                                    timesteps=ts)
    if (bitstring[w] == "0"):
        labels1 = np.zeros((ts * 3, 1), dtype=float)
    else:
        labels1 = np.ones((ts * 3, 1), dtype=float)

    w += 1
    w = w % len(bitstring)

    threeChannels0 = np.concatenate((threeChannelsOF0, padded_zeros), axis=1)
    #labels0 = np.zeros((ts * 3, 1), dtype=float)
    #labels1 = np.ones((ts * 3, 1), dtype=float)
    bothLabelsB = np.concatenate((labels0, labels1), axis=0)
    bothPatternsB = np.concatenate((threeChannels0, threeChannels1), axis=1)
    bothLabelsA = np.copy(bothLabels)
    bothPatternsA = np.copy(bothPatterns)
    bothLabels = np.concatenate((bothLabelsA, bothLabelsB), axis=0)
    bothPatterns = np.concatenate((bothPatternsA, bothPatternsB), axis=1)
    i += 1
    
tC = bothPatterns.transpose((1, 0))
labels = bothLabels


#while not good:
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
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-1, tau_rc=1e-1),  # Specify type of neuron
            max_rates=Uniform(1 / 6e-1, 1 / 6e-1),  # Set the maximum firing rate of the neuron 500Mhz
        )


with model:
    input_signal = nengo.Node(PresentInput(tC, presentation_time=pt))
    input_keys = nengo.Node(PresentInput(labels, presentation_time=pt))

with model:
    nengo.Connection(input_signal, layer1, synapse=None,label="input signals to layer 1")
    nengo.Connection(layer1, layer2, synapse=1e-1, label="layer 1 to layer2")
    nengo.Connection(layer2, layer1, synapse=ts*1e-1, label="layer 2 to layer 1")
    conn = nengo.Connection(layer2, layer3, synapse=1e-2, label="layer 2 to layer 3")
    nengo.Connection(layer3, layer2, synapse=ts*1e-1, label="layer3 to layer2")

simT = 10
with model:
    input_probe = nengo.Probe(input_signal)  # The original input
    spikes = nengo.Probe(layer1.neurons)  # Raw spikes from each neuron
    # Subthreshold soma voltages of the neurons
    
    #voltage = nengo.Probe(layer1.neurons, 'voltage')
    # Spikes filtered by a 10ms post-synaptic filter
    filteredl1 = nengo.Probe(layer1.neurons, 'voltage', synapse=3e-2)
    filteredl2 = nengo.Probe(layer2.neurons, 'voltage', synapse=3e-2)
    filteredl3 = nengo.Probe(layer3.neurons, 'voltage', synapse=3e-2)
    
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

plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[p_keys] + 0, color="black")
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
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[p_keys] + 0, color="black")
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

plt.plot(t[test], sim.data[p_keys][test] + 0, color="black")
plt.plot(t[test], sim.data[input_probe][test] + 2)
print(np.shape(sim.data[filteredl2]))
plt.plot(t[test], sim.data[filteredl2][test][0:, 0:1] + 3, color="b")
plt.plot(t[test], sim.data[filteredl2][test][0:, 1:2] + 4, color="y")
plt.savefig('fig/3layersfeedbck_neuronsl3_2_'+str(reseed)+'.png')

plt.clf()

plt.figure()
time_input_signal = t <= pt*ts
plt.title("single timesteps of input")
plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 0:1], color="blue")
plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 1:2]+2.1, color="green")
plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 2:3] + 4.2 , color="orange")
plt.savefig('fig/inputTiming.png')
i = 0

best_neuron_value = np.sum((sim.data[filteredl3][0:, 0:0 + 1]+1.0) - (sim.data[p_keys]*2.0))
print(best_neuron_value)
best_neuron_index = 0
while i < num_neurons_l3:
    sum = np.sum((sim.data[filteredl3][0:, i:i + 1]+1.0) - (sim.data[p_keys]*2.0))
    # print(i, sum)

    if (sum < best_neuron_value):
        best_neuron_index = i
        best_neuron_value = sum
    print(best_neuron_value, best_neuron_index, reseed)
    i += 1
if (np.abs(best_neuron_value) < 20000):
    good = True
else:
    reseed += 1

#    preset_timing_plot()    
    good = True

plt.figure()
plt.title("Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[p_keys] + 0, color="g")
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[filteredl3][0:, best_neuron_index:best_neuron_index + 1] + 3, color="black")
plt.savefig('fig/3layersfeedbck_best_neuron_index_l3.png')



with model:
    print(model)
    print(model.all_connections)
    #nengo.Connection(input_signal, layer1, synapse=None)
    #conn = nengo.Connection(layer1, layer2, synapse=1e-1)
    #nengo.Connection(layer2, layer1, synapse=ts*1e-1)
    #nengo.Connection(layer2, layer3, synapse=1e-2)
    #nengo.Connection(layer3, layer2, synapse=ts*1e-1)
    error = nengo.Ensemble(40,dimensions=1,label="error network")
    error_values = nengo.Probe(error, synapse=None, label="error_values")
    nengo.Connection(input_keys,error,transform=1,synapse=3e-3, label="input keys to error")
    nengo.Connection(layer3,error,transform=[[-1,-1,-1]], synapse=ts*1e-3, label="layer3 to error network")
    conn.learning_rule_type = nengo.PES(pre_synapse=9e-1)
    nengo.Connection(error,conn.learning_rule,transform=[[1],[1],[1]], label="error network connect to learning rule")
    print(model.all_connections)
with nengo.Simulator(model) as sim:  # Create a simulator
    sim.run(simT)  # Run 

plt.figure()
plt.title("Learned Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[error_values],color="red")
plt.plot(t, sim.data[p_keys] + 0, color="g")
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[filteredl3][0:, best_neuron_index:best_neuron_index + 1] + 3, color="black")

plt.savefig('fig/3layersfeedbck_best_neuron_index_l3_learned.png')