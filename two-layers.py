import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput

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


model = nengo.Network(label='Two Layers', seed=91195)

with model:
    with model:
        neurons = nengo.Ensemble(
            4,  # Number of neurons
            dimensions=3,  # each neuron is connected to all (3) input channels.
            # Set intercept to 0.5
            intercepts=Uniform(-0.00001, 0.00001),  # Set the intercepts at 0.00001 (threshold for Soma voltage)
            neuron_type=nengo.LIF(min_voltage=-1, tau_ref=5e-11, tau_rc=0.00000001),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(500e+6, 500e+6),             # Set the maximum firing rate of the neuron 500Mhz
            # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
            #encoders=[[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]],
        )

        neuronsL2 = nengo.Ensemble(
            6,  # Number of neurons
            dimensions=3,
            # Set intercept to 0.5
            intercepts=Uniform(-0.1, 0.1),  # Set the threshold for Soma voltage
            neuron_type=nengo.LIF(min_voltage=-1, tau_ref=5e-11, tau_rc=5e-6),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(5e+8, 5e+8),  # Set the maximum firing rate of the neuron 500Mhz
            # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
            #encoders=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [ -1, -1, -1], [1, 1, 1], [1, 1, -1]],
        )

threeChannels1, end_channel = phase_automata(driving_symbol="1", probability_of_transition=False,timesteps=9)
threeChannels0, end_channel0 = phase_automata(driving_symbol="0", probability_of_transition=False,timesteps=9)
labels0 = np.zeros((9,2),dtype=float)
labels1 = np.ones((9,2),dtype=float)
bothLabels = np.concatenate((labels0,labels1),axis=0)
bothPatterns = np.concatenate((threeChannels0,threeChannels1),axis=1)

tC = bothPatterns.transpose((1, 0))
labels = bothLabels
with model:
    input_signal = nengo.Node(PresentInput(tC, presentation_time=1e-7))
    input_keys = nengo.Node(PresentInput(labels,presentation_time=1e-7))

with model:
    nengo.Connection(input_signal, neurons, synapse=None)
    nengo.Connection(neurons, neuronsL2, synapse=1e-7)

simT = 1e-4
with model:
    input_probe = nengo.Probe(input_signal)  # The original input
    spikes = nengo.Probe(neurons.neurons)  # Raw spikes from each neuron
    spikesL2 = nengo.Probe(neuronsL2.neurons)  # Raw spikes from each neuron
    # Subthreshold soma voltages of the neurons
    voltage = nengo.Probe(neurons.neurons, 'voltage')
    voltageL2 = nengo.Probe(neuronsL2.neurons, 'voltage')
    # Spikes filtered by a 10ms post-synaptic filter
    filteredL2 = nengo.Probe(neuronsL2, synapse=1e-8)
    learning = nengo.Node(output=lambda t: -int(t >= simT / 2))
    recall = nengo.Node(size_in=3)
    #guess = nengo.Probe(best_guess,synapse=1e-8)

    # Learn the encoders/keys
    voja = nengo.Voja(learning_rate=5e-5, post_synapse=None)
    conn_in = nengo.Connection(input_signal, neuronsL2, synapse=None, learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)
    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(
        neuronsL2,
        recall,
        learning_rule_type=nengo.PES(1e-2),
        function=lambda x: np.zeros(3),
    )

    # Create the error population
    error = nengo.Ensemble(6, 3,intercepts=Uniform(-0.1, 0.1),  # Set the threshold for Soma voltage
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-12, tau_rc=5e-7),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(5e+9, 5e+9),  # Set the maximum firing rate of the neuron 500Mhz
     )
    nengo.Connection(
        learning, error.neurons,transform=[[10.0]] * 6, synapse=None
    )

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(input_signal, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(input_keys, synapse=None)
    p_values = nengo.Probe(input_signal, synapse=None)
    p_learning = nengo.Probe(learning, synapse=None)
    p_error = nengo.Probe(error, synapse=1e-8)
    p_recall = nengo.Probe(recall, synapse=None)
    p_encoders = nengo.Probe(conn_in.learning_rule, "scaled_encoders")
    
    
with nengo.Simulator(model, dt=1e-8) as sim:  # Create a simulator
    sim.run(simT)  # Run 
    
t = sim.trange()


plt.figure()
plt.title("Keys")
plt.plot(t, sim.data[p_keys])
#plt.ylim(-1, 1)
plt.show()

plt.figure()
plt.title("Values")
plt.plot(t, sim.data[p_values])
#plt.ylim(-1, 1)
plt.show()

plt.figure()
plt.title("Learning")
plt.plot(t, sim.data[p_learning])
#plt.ylim(-1.2, 0.2)
plt.show()


train = t <= simT / 2
test = ~train

plt.figure()
plt.title("Value Error During Training")
plt.plot(t[train], sim.data[p_error][train])
plt.show()

plt.figure()
plt.title("Value Error During Recall")
plt.plot(t[test], sim.data[p_recall][test] - sim.data[p_values][test])
plt.show()

plot_range = 100 # index



# Plot the decoded output of the ensemble
plt.figure()
plt.plot(t, sim.data[input_probe])
plt.xlim(0, t[plot_range])
plt.xlabel("Time (s)")
plt.title("Input probe for " + str(plot_range) + " timesteps")
plt.savefig("fig/two_layers_input_probe.png")
plt.clf()
plt.figure()
plt.title("Neurons filtered probe for " + str(plot_range) + " timesteps")
plt.plot(t, sim.data[filteredL2])
plt.xlabel("Time (s)")
plt.xlim(0, t[plot_range])
plt.savefig("fig/two_layers_filtered.png")

plt.clf()
plt.figure()
plt.title("NeuronsL2 Best guess filtered probe for " + str(plot_range) + " timesteps")
#plt.plot(t, sim.data[guess])
plt.xlabel("Time (s)")
plt.xlim(0, t[plot_range])
plt.savefig("fig/two_layers_best_guess.png")

# Plot the spiking output of the ensemble
plt.figure(figsize=(8, 8))


plt.subplot(2, 2, 1)
plt.title("Neuron Spikes")
plt.xlabel("Time (s)")
rasterplot(t[0:plot_range], sim.data[spikes][0:plot_range], colors=['y', 'm', 'k', 'r'])
plt.yticks((1,2,3,4), ("0" ,"1" ,"2","3"))
plt.ylim(4.5, 0.5)

# Plot the soma voltages of the neurons
plt.subplot(2, 2, 2)
plt.title("Neuron Soma Voltage")

plt.plot(t, sim.data[voltage][:, 3] + 0, 'r')
plt.plot(t, sim.data[voltage][:, 2]+1, 'k')
plt.plot(t, sim.data[voltage][:, 1]+2, 'm')
plt.plot(t, sim.data[voltage][:, 0]+3, 'y')
plt.xlabel("Time (s)")
plt.yticks(())
plt.subplots_adjust(wspace=0.1)


plt.subplots_adjust(hspace=0.4)
plt.subplot(2, 2, 3)
plt.title("Neuron Spikes L2")
plt.xlabel("Time (s)")
rasterplot(t[0:plot_range], sim.data[spikesL2][0:plot_range], colors=['y', 'm', 'k', 'r','b','g'])
plt.yticks((1,2,3,4,5,6), ("0" ,"1" ,"2","3", "4","5"))
plt.ylim(6.5, 0.5)

# Plot the soma voltages of the neurons
plt.subplot(2, 2, 4)
plt.title("Neuron Soma Voltage L2")
plt.plot(t, sim.data[voltageL2][:, 5] + 0, 'g')
plt.plot(t, sim.data[voltageL2][:, 4] + 1, 'b')
plt.plot(t, sim.data[voltageL2][:, 3] + 2, 'r')
plt.plot(t, sim.data[voltageL2][:, 2]+3, 'k')
plt.plot(t, sim.data[voltageL2][:, 1]+4, 'm')
plt.plot(t, sim.data[voltageL2][:, 0]+5, 'y')
plt.xlabel("Time (s)")
plt.yticks(())
plt.subplots_adjust(wspace=0.1)
plt.savefig("fig/two_layers.png")