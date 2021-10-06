import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import nengo
from nengo.dists import Uniform
from nengo.processes import WhiteNoise
from nengo.processes import WhiteSignal
from nengo.processes import PresentInput
from nengo.utils.matplotlib import rasterplot

#phase automata 
# Creates a matrix of channels and events. For 3 channels (0,1,2) and timestamps 0-9:
#[[1. 0. 0. 1. 0. 0. 1. 0. 0.] Channel 0
# [0. 1. 0. 0. 1. 0. 0. 1. 0.] Channel 1
# [0. 0. 1. 0. 0. 1. 0. 0. 1.]] Channel 2
def phase_automata(driving_symbol='0',number_of_symbols=3,id_of_starting_symbol=0,timesteps=9,
                 probability_of_transition=False):
    code = np.zeros((number_of_symbols, timesteps), dtype=float)
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
    #print(code, ending_state)
    return code, ending_state


def usePhaseMachine():
    threeChannels, end_channel = phase_automata(driving_symbol="1",probability_of_transition=True)
    print(threeChannels)
    tC = threeChannels.transpose((1,0))
    simt = 0.000001#0.000000008

    model = nengo.Network(label="A Single Neuron", seed=91195)
    with model:
        neuron = nengo.Ensemble(
            1,
            dimensions=3,  # Represent a scalar
            # Set intercept to 0.5
            intercepts=Uniform(-0.00001, 0.00001),
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=0.0000000005, tau_rc=0.00000001),
            # Set tau_ref= or tau_rc = here to 
            # change those 
            # parms 
            # for the 
            # neurons.
            # Set the maximum firing rate of the neuron to 100hz
            #max_rates=Uniform(20000000, 20000000),
            max_rates=Uniform(500000000,500000000),
            # Set the neuron's firing rate to increase for positive input
            encoders=[[1,1,1]],
        )

    with model:
        #input_signal = nengo.Node(PresentInput(tC,presentation_time=0.0000000001))
        #input_signal = nengo.Node(WhiteNoise(dist=nengo.dists.Gaussian(0.0, 0.1), seed=1,default_size_out=3))
        #input_signal = nengo.Node(WhiteSignal(1, high=10,rms=0.2,y0=0.1,seed=5), size_out=3)
        input_signal = nengo.Node(PresentInput(tC, presentation_time=0.0000001))
    with model:
        # Connect the input signal to the neuron
        nengo.Connection(input_signal, neuron)

    syn = 0.001

    with model:
        # The original input
        input_signal_probe = nengo.Probe(input_signal)
        # The raw spikes from the neuron
        spikes = nengo.Probe(neuron.neurons)
        # Subthreshold soma voltage of the neuron
        voltage = nengo.Probe(neuron.neurons, "voltage")
        # Spikes filtered by a 10ms post-synaptic filter
        filtered = nengo.Probe(neuron, synapse=syn)

    with nengo.Simulator(model, dt=0.00000001) as sim:  # Create the simulator
    #with nengo.Simulator(model, dt=0.0000000001) as sim:  # Create the simulator
        sim.run(simt)  # Run it for simt seconds
        # print(neuron.neurons.ensemble.neuron_type.state)
        #print('\n'.join(f"* {op}" for op in sim.step_order)) # For debug

    with model:
        print(neuron.neurons.ensemble.neuron_type.state)
    # Plot the decoded output of the ensemble
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("Input Signals")
    plt.ylabel("signal (Input 3 phase signal)")
    plt.xlabel("Time (s)")

    plt.plot(sim.trange(), sim.data[input_signal_probe])
    plt.xlim(0, simt)

    plt.subplot(122)
    plt.title("Spikes")
    plt.ylabel("signal (soma spike)")
    plt.xlabel("Time (s)")
    plt.plot(sim.trange(), sim.data[spikes])
    #plt.plot(sim.trange(), (sim.data[filtered]))
    #plt.plot(sim.trange(), (sim.data[filtered])*10000)
    plt.xlim(0, simt)

    plt.savefig("fig/input_signals.png")
    plt.clf()
    # Plot the spiking output of the ensemble
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    rasterplot(sim.trange(), sim.data[spikes])
    plt.ylabel("Neuron")
    plt.xlabel("Time (s)")
    plt.xlim(0, simt)

    # Plot the soma voltages of the neurons
    plt.subplot(122)

    plt.plot(sim.trange(), sim.data[voltage][:, 0], "r")
    plt.ylabel("Soma Voltage")
    plt.xlabel("Time (s)")
    plt.xlim(0, simt)
    plt.savefig("fig/short_time_neuron.png")
    # plt.show()
    return 0



# phase_automata()

# str_of_events = timeseries_to_events(info="0", starting_symbol=0, timesteps=9, number_of_symbols=3,
#                                      probability_of_transition=False,save_path_and_name="",time_to_save_at = 1)
# 
# print(str_of_events)

usePhaseMachine()