import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput
from nengo.utils.ensemble import tuning_curves


def phase_automata(driving_symbol='0',number_of_symbols=3,id_of_starting_symbol=0,timesteps=9,
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
                        state = (j+1) % number_of_symbols
                    elif driving_symbol == '1':
                        state = ((j-1) % number_of_symbols)
                    else:
                        state = id_of_starting_symbol
                        print ("ILLEGAL DRIVING SYMBOL")
                    #print('passing to state ', state, 'driving symbol ', driving_symbol)
                    code[j][i] = 1
                    u = False
                else:
                    state = j
                    #print('staying in state', state)
            j += 1
        i += 1
    ending_state = state
    return code, ending_state

reseed = 91253 # 91273
good = False
while not good:
    model = nengo.Network(label='Two Layers', seed=reseed)
    #sim = nengo.Simulator(model,dt=1e-8)
    ts = 45
    threeChannels1, end_channel = phase_automata(driving_symbol="1", probability_of_transition=False,timesteps=ts)
    threeChannels0, end_channel0 = phase_automata(driving_symbol="0", probability_of_transition=False,timesteps=ts)
    labels0 = np.zeros((ts,1),dtype=float)
    labels1 = np.ones((ts,1),dtype=float)
    bothLabels = np.concatenate((labels0,labels1),axis=0)
    bothPatterns = np.concatenate((threeChannels0,threeChannels1),axis=1)
    
    tC = bothPatterns.transpose((1, 0))
    labels = bothLabels
    num_neurons = 1000
    with model:
        with model:
            neurons = nengo.Ensemble(
                num_neurons,  # Number of neurons
                dimensions=3,  # each neuron is connected to all (3) input channels.
                # Set intercept to 0.5
                neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                max_rates=Uniform(1/6e-2,1/6e-2),             # Set the maximum firing rate of the neuron 500Mhz
            )
    
    
    with model:
        input_signal = nengo.Node(PresentInput(tC, presentation_time=3e-2))
        input_keys = nengo.Node(PresentInput(labels,presentation_time=3e-2))
    
    with model:
        nengo.Connection(input_signal, neurons, synapse=None)
       
    simT = 18
    with model:
        input_probe = nengo.Probe(input_signal)  # The original input
        spikes = nengo.Probe(neurons.neurons)  # Raw spikes from each neuron
        # Subthreshold soma voltages of the neurons
        voltage = nengo.Probe(neurons.neurons, 'voltage')
        # Spikes filtered by a 10ms post-synaptic filter
        filtered = nengo.Probe(neurons.neurons,'voltage', synapse=2e-2)
        learning = nengo.Node(output=lambda t: -int(t >= simT / 2))# nengo.Node(output=lambda t:-1)
        recall = nengo.Node(size_in=1)
        #guess = nengo.Probe(best_guess,synapse=1e-8)
    
        # Learn the encoders/keys
       
        # Setup probes
        p_keys = nengo.Probe(input_keys, synapse=None, label="p_keys")
        p_values = nengo.Probe(input_signal, synapse=None, label="p_values")
        p_learning = nengo.Probe(learning, synapse=None, label="p_learning")
    
    with nengo.Simulator(model) as sim:  # Create a simulator
       sim.run(simT)  # Run 
    
    t = sim.trange()
    
    plot_range = -1 # index
    
    
    # train = t <= simT / 2
    # test = ~train
    
    # plt.figure()
    # plt.title("Value Error During Training")
    # plt.plot(t[train], sim.data[p_error][train])
    # plt.show()
    
    plt.figure()
    plt.title("Filtered output")
    # print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
    # plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
    plt.plot(t, sim.data[p_keys]+0,color="g")
    plt.plot(t,sim.data[input_probe]+2)
    print(np.shape(sim.data[filtered]))
    plt.plot(t, sim.data[filtered][0:,0:1]+3,color="b")
    plt.plot(t, sim.data[filtered][0:,1:2]+4,color="y")
    plt.plot(t, sim.data[filtered][0:,2:3]+5,color="k")
    plt.plot(t, sim.data[filtered][0:,3:4]+6,color="m")
    plt.plot(t, sim.data[filtered][0:,4:5]+7,color="r")
    plt.plot(t, sim.data[filtered][0:,5:6]+8,color="g")
    plt.plot(t, sim.data[filtered][0:,6:7]+9,color="#aabbcc")
    
    i = 0
    best_neuron_value = np.abs(np.sum(sim.data[filtered][0:,0:0+1]-sim.data[p_keys]))
    print(best_neuron_value)
    best_neuron_index = 0
    while i <num_neurons:
        sum = np.abs(np.sum(sim.data[filtered][0:,i:i+1]-sim.data[p_keys]))
        #print(i, sum)
        
        if (sum < best_neuron_value):
            best_neuron_index = i
            best_neuron_value = sum
        print(best_neuron_value,best_neuron_index, reseed)
        i += 1
    if (best_neuron_value < 20):
        good = True
    else:
        reseed += 1
    

plt.figure()
plt.title("Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[p_keys]+0,color="g")
plt.plot(t,sim.data[input_probe]+2)
print(np.shape(sim.data[filtered]))
plt.plot(t, sim.data[filtered][0:,best_neuron_index:best_neuron_index+1]+3,color="b")
print(neurons.encoders)

# #plt.plot(t, sim.data[p_recall][0:,2:3] - sim.data[p_values][0:,2:3]+4,color="orange")
# #plt.plot(t[test],sim.data[input_probe][test])
plt.show()
exit()
plt.savefig("best_neuron_fit.png")

# #scale = (sim.data[neuronsL2].gain / neuronsL2.radius)[:, np.newaxis]


# # def plot_2d(text, xy):
# #     plt.figure()
# #     plt.title(text)
# #     plt.scatter(xy[:, 0], xy[:, 1], label="Encoders")
# #     plt.scatter(labels[:, 0], labels[:, 1], c="red", s=150, alpha=0.6, label="Keys")
# #     plt.xlim(-1.5, 1.5)
# #     plt.ylim(-1.5, 2)
# #     plt.legend()
# #     plt.gca().set_aspect("equal")
# # 
# # 
# # plot_2d("Before", sim.data[p_encoders][0].copy() / scale)
# # plt.show()
# # plot_2d("After", sim.data[p_encoders][-1].copy() / scale)
# # plt.show()


# # Plot the decoded output of the ensemble
# plt.figure()
# plt.plot(t, sim.data[input_probe])
# plt.xlim(0, t[plot_range])
# plt.xlabel("Time (s)")
# plt.title("Input probe for " + str(plot_range) + " timesteps")
# plt.savefig("fig/two_layers_input_probe.png")
# plt.clf()
# plt.figure()
# plt.title("Neurons filtered probe for " + str(plot_range) + " timesteps")
# plt.plot(t, sim.data[filteredL2])
# plt.xlabel("Time (s)")
# plt.xlim(0, t[plot_range])
# plt.savefig("fig/two_layers_filtered.png")

# plt.clf()
# plt.figure()
# plt.title("NeuronsL2 Best guess filtered probe for " + str(plot_range) + " timesteps")
# #plt.plot(t, sim.data[guess])
# plt.xlabel("Time (s)")
# plt.xlim(0, t[plot_range])
# plt.savefig("fig/two_layers_best_guess.png")

# # Plot the spiking output of the ensemble
# plt.figure(figsize=(8, 8))


# plt.subplot(2, 2, 1)
# plt.title("Neuron Spikes")
# plt.xlabel("Time (s)")
# rasterplot(t[0:plot_range], sim.data[spikes][0:plot_range], colors=['y', 'm', 'k', 'r'])
# plt.yticks((1,2,3,4), ("0" ,"1" ,"2","3"))
# plt.ylim(4.5, 0.5)

# # Plot the soma voltages of the neurons
# plt.subplot(2, 2, 2)
# plt.title("Neuron Soma Voltage")

# plt.plot(t, sim.data[voltage][:, 3] + 0, 'r')
# plt.plot(t, sim.data[voltage][:, 2]+1, 'k')
# plt.plot(t, sim.data[voltage][:, 1]+2, 'm')
# plt.plot(t, sim.data[voltage][:, 0]+3, 'y')
# plt.xlabel("Time (s)")
# plt.yticks(())
# plt.subplots_adjust(wspace=0.1)


# plt.subplots_adjust(hspace=0.4)
# plt.subplot(2, 2, 3)
# plt.title("Neuron Spikes L2")
# plt.xlabel("Time (s)")
# rasterplot(t[0:plot_range], sim.data[spikesL2][0:plot_range], colors=['y', 'm', 'k', 'r','b','g'])
# plt.yticks((1,2,3,4,5,6), ("0" ,"1" ,"2","3", "4","5"))
# plt.ylim(6.5, 0.5)

# # Plot the soma voltages of the neurons
# plt.subplot(2, 2, 4)
# plt.title("Neuron Soma Voltage L2")
# plt.plot(t, sim.data[voltageL2][:, 5] + 0, 'g')
# plt.plot(t, sim.data[voltageL2][:, 4] + 1, 'b')
# plt.plot(t, sim.data[voltageL2][:, 3] + 2, 'r')
# plt.plot(t, sim.data[voltageL2][:, 2]+3, 'k')
# plt.plot(t, sim.data[voltageL2][:, 1]+4, 'm')
# plt.plot(t, sim.data[voltageL2][:, 0]+5, 'y')
# plt.xlabel("Time (s)")
# plt.yticks(())
# plt.subplots_adjust(wspace=0.1)
# plt.savefig("fig/two_layers.png")