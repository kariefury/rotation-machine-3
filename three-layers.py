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


model = nengo.Network(label='Three Layers', seed=91195)

with model:
    with model:
        neurons = nengo.Ensemble(
            4,  # Number of neurons
            dimensions=3,  # each neuron is connected to all (3) input channels.
            # Set intercept to 0.5
            intercepts=Uniform(-0.00001, 0.00001),  # Set the intercepts at 0.00001 (threshold for Soma voltage)
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=0.0000000005, tau_rc=0.00000001),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(500e+6, 500e+6),             # Set the maximum firing rate of the neuron 500Mhz
            # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
            encoders=[[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]],
        )

        neuronsL2 = nengo.Ensemble(
            6,  # Number of neurons
            dimensions=3,
            # Set intercept to 0.5
            intercepts=Uniform(-0.00001, 0.00001),  # Set the intercepts at 0.00001 (threshold for Soma voltage)
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=0.0000000005, tau_rc=0.00000001),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(500e+6, 500e+6),  # Set the maximum firing rate of the neuron 500Mhz
            # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
            encoders=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [ -1, -1, -1], [1, 1, 1], [1, 1, -1]],
        )
        
        neuronsL3 = nengo.Ensemble(
            2,  # Number of neurons
            dimensions=3,
            # Set intercept to 0.5
            intercepts=Uniform(-0.00001, 0.00001),  # Set the intercepts at 0.00001 (threshold for Soma voltage)
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=0.0000000005, tau_rc=0.00000001),  # Specify type of neuron
            # Set tau_ref= or tau_rc = here to
            # change those
            # parms
            # for the
            # neurons.
            max_rates=Uniform(500e+6, 500e+6),  # Set the maximum firing rate of the neuron 500Mhz
            # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
            encoders=[[1, -1, -1], [-1, -1, 1]],
        )

driving_symbol = "0"
noise = True
threeChannels, end_channel = phase_automata(driving_symbol=driving_symbol, probability_of_transition=noise,timesteps=90)
print(threeChannels)
tC = threeChannels.transpose((1, 0))
print(tC)
with model:
    input_signal = nengo.Node(PresentInput(tC, presentation_time=1e-7))

with model:
    nengo.Connection(input_signal, neurons, synapse=None)
    nengo.Connection(neurons, neuronsL2, synapse=None)
    nengo.Connection(neuronsL2,neuronsL3,synapse=None)


with model:
    input_probe = nengo.Probe(input_signal)  # The original input
    spikesL3 = nengo.Probe(neuronsL3.neurons)  # Raw spikes from each neuron
    # Subthreshold soma voltages of the neurons
    #voltage = nengo.Probe(neurons.neurons, 'voltage')
    voltageL3 = nengo.Probe(neuronsL3.neurons, 'voltage')
    # Spikes filtered by a 10ms post-synaptic filter
    filteredL3 = nengo.Probe(neuronsL3, synapse=1e-8)
    
with nengo.Simulator(model, dt=1e-8) as sim:  # Create a simulator
    sim.run(1000000e-9)  # Run it for 10k nanosecond
    
t = sim.trange()

plot_range = 100  # index
# Plot the decoded output of the ensemble
plt.figure()
plt.plot(t, sim.data[input_probe])
plt.xlim(0, t[plot_range])
plt.xlabel("Time (s)")
plt.title("Input probe for " + str(plot_range) + " timesteps " +"Driving Symbol:"+driving_symbol+" Noise "
                                                                                                        ""+str(noise))
plt.savefig("fig/three_layers_input_probe"+driving_symbol+".png")
plt.clf()
plt.figure()
plt.title("Neurons filtered probe for " + str(plot_range) + " timesteps Driving Symbol:"+driving_symbol+" Noise "
                                                                                                        ""+str(noise))
plt.plot(t, sim.data[filteredL3])
plt.xlabel("Time (s)")
plt.xlim(0, t[plot_range])
plt.savefig("fig/three_layers_filtered"+driving_symbol+".png")

# Plot the spiking output of the ensemble
plt.figure(figsize=(10, 4))
plt.title("Neuron Spikes Driving Symbol "+driving_symbol+" Noise "+str(noise))
plt.subplot(1, 2, 1)
plt.xlabel("Time (s)")
rasterplot(t[0:plot_range], sim.data[spikesL3][0:plot_range], colors=['y', 'm'])
plt.yticks((1,2), ("0" ,"1" ))
plt.ylim(2.5, 0.5)

# Plot the soma voltages of the neurons
plt.subplot(1, 2, 2)
plt.title("Neuron Soma Voltage Driving Symbol "+driving_symbol+" Noise "+str(noise))
plt.plot(t, sim.data[voltageL3][:, 1]+4, 'm')
plt.plot(t, sim.data[voltageL3][:, 0]+5, 'y')
plt.xlabel("Time (s)")
plt.yticks(())
plt.subplots_adjust(wspace=0.05)
plt.savefig("fig/three_layers"+driving_symbol+".png")