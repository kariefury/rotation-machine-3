import matplotlib as mpl
print(mpl.get_backend())

import matplotlib.pyplot as plt
import numpy as np
mpl.use('agg')
import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.processes import PresentInput
from nengo.utils.ensemble import tuning_curves
from numpy.random import RandomState


def preset_timing_plot():
    print("sim.dt",sim.dt)
    print("Symbol presentation time",pt)
    print("Label length", label_length)
    print("layer 1 Tau RC",layer1.neuron_type.tau_rc)
    print("layer1 refractory period",layer1.neuron_type.tau_ref)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    people = ('Sim dt', 'Pulse Length', 'Pulse Gap', 'Rotation Time', 'Label Time', 'L1 Tau RC', 'L1 Tau Ref', 
    'L2 Tau RC', 
              "L2 Tau Ref", 
              "Feed Forward", "Feed Back")
    y_pos = np.arange(len(people))
    print (y_pos)
    performance = np.zeros(len(people),dtype=float)
    lbs = [1,2,3,4,5,6,7,8,9,10]
    #performance = 3 + 10 * np.random.rand(len(people))
    performance[0] = sim.dt
    performance[1] = (pt*pulse_length)
    performance[2] = (pt*pulse_gap)
    performance[3] = ts*(pt*(pulse_gap+pulse_length))
    performance[4] = label_length*pt
    performance[5] = layer1.neuron_type.tau_rc
    performance[6] = layer1.neuron_type.tau_ref
    performance[7] = layer2.neuron_type.tau_rc
    performance[8] = layer2.neuron_type.tau_ref
    performance[9] = ff
    performance[10] = fb
    ax.barh(y_pos, performance)
    ax.set_yticks(y_pos)
    #plt.xscale("log")
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time (s)')
    ax.set_title('Preset Timing Parameters')
    plt.tight_layout() 
    plt.savefig('fig/preset_timing_parameters/preset_timing_parmeters.png')

def plot_data(q,neuronid,b):
    print("plotting")
    plt.figure()
    plt.title("Filtered L1 output")

    plt.plot(t, sim.data[input_probe] + 2.1)
    plt.plot(t, sim.data[p_keys] + 0, color="black")
    print(np.shape(sim.data[filteredl1]))
    plt.plot(t, sim.data[filteredl1][0:, neuronid:neuronid+1] + 3, color="b")
    plt.plot(t, sim.data[filteredl1][0:, b:b+1] + 4, color="y")
    plt.plot(t, sim.data[filteredl1][0:, 2:3] + 5, color="k")
    plt.plot(t, sim.data[filteredl1][0:, 3:4] + 6, color="m")
    plt.savefig('fig/two-layers-feedback/layer1-4-'+str(q)+'.png')
    #    plt.plot(t, sim.data[filtered][0:, 4:5] + 7, color="r")
    #    plt.plot(t, sim.data[filtered][0:, 5:6] + 8, color="g")
    #    plt.plot(t, sim.data[filtered][0:, 6:7] + 9, color="#aabbcc")
    plt.close()
    plt.figure()
    plt.title("Filtered L2 output")
    # print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
    # plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
    plt.plot(t, sim.data[input_probe] + 2.1)
    plt.plot(t, sim.data[p_keys] + 0, color="black")
    print(np.shape(sim.data[filteredl2]))
    plt.plot(t, sim.data[filteredl2][0:, 0:1] + 3, color="b")
    plt.plot(t, sim.data[filteredl2][0:, 1:2] + 4, color="y")
    plt.plot(t, sim.data[filteredl2][0:, 2:3] + 5, color="k")
    plt.plot(t, sim.data[filteredl2][0:, 3:4] + 6, color="m")
    plt.plot(t, sim.data[filteredl2][0:, 4:5] + 7, color="r")
    plt.plot(t, sim.data[filteredl2][0:, 5:6] + 8, color="g")
    plt.savefig('fig/two-layers-feedback/layer2-6-'+str(q)+'.png')
    plt.close("all")
    

def phase_automata_fractional_pulse(driving_symbol='0', number_of_symbols=3, id_of_starting_symbol=0, timesteps=9,
                   probability_of_transition=False,pulse_length=100,pulse_gap=100):
    code = np.zeros((number_of_symbols, timesteps*(pulse_length+pulse_gap)), dtype=float)
    code = code - 1
    state = id_of_starting_symbol
    i = 0
    while i < (timesteps*(pulse_length+pulse_gap)):
        u = True
        j = 0
#        print("i",i)
        while j < number_of_symbols:
 #           print("j",j)
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
                    k = 0
                    while k < pulse_length:
                        #print("k",k)
                        code[j][i+k] = 1
                        k += 1

                    while k < (pulse_length+pulse_gap):
                        #print("k",k)
                        code[j][i+k] = -1.0
                        k += 1
                    u = False
                else:
                    state = j
                    # print('staying in state', state)
            j += 1
        i += (pulse_length+pulse_gap)
    ending_state = state
    return code, ending_state


reseed = 13544 #13537# 11524#9166#8 #91521 #91427 #91377 #91362 #91332 #91323 #91280 #91274 # 91264 # 91254  # 91273
good = False
number_of_samples = 2
ts = 6  # number of possible transitions to hold a driving symbol constant for.
pb = False
pt = 1e-3 # seconds to present each step of input
pulse_length = 10
pulse_gap = 5
label_length = ts * 3 * (pulse_length+pulse_gap)# In Timesteps, multiply by dt to get actual length of time
padded_zeros = np.zeros((3, ts * 2 * (pulse_length+pulse_gap)), dtype=float)
padded_zeros = padded_zeros - 1.0


i = 1

threeChannelsOF1, end_channel = phase_automata_fractional_pulse(driving_symbol="1", probability_of_transition=pb, timesteps=ts,pulse_length=pulse_length,pulse_gap=pulse_gap)
threeChannels1 = np.concatenate((threeChannelsOF1, padded_zeros), axis=1)
threeChannelsOF0, end_channel0 = phase_automata_fractional_pulse(driving_symbol="0", probability_of_transition=pb, timesteps=ts,pulse_length=pulse_length,pulse_gap=pulse_gap)
threeChannels0 = np.concatenate((threeChannelsOF0, padded_zeros), axis=1)
labels0 = np.zeros((label_length, 1), dtype=float)
labels1 = np.ones((label_length, 1), dtype=float)

bothLabels = np.concatenate((labels0, labels1), axis=0)
bothPatterns = np.concatenate((threeChannels0, threeChannels1), axis=1)

bitstring = "10110011100001010000000111111101010101010011001100011100101010000111100000000000000000000000000000000000000000000000000000000000000000"
plt.figure()
plt.title("Input Pattern and Label Example ProbTran:"+str(pb)+" PaddedZeros:True")
t = np.arange(ts*(pulse_gap+pulse_length)*3*2)
plt.xlabel('ts (Timestep)' )
plt.ylabel('Value')
plt.plot(t,bothPatterns[0],color="blue",label="pattern C1")
plt.plot(t,bothPatterns[1],color="green",label="pattern C2")
plt.plot(t,bothPatterns[2],color="orange",label="pattern C3")
plt.plot(t,bothLabels+2.0,color="black",label="label")
plt.legend()
plt.savefig("fig/input_pattern_example_probTran_"+str(pb)+"_padded_zeros_true.png")

plot_now = True
w = 0
while i < number_of_samples:
    threeChannelsOF, end_channel = phase_automata_fractional_pulse(driving_symbol=bitstring[w], probability_of_transition=pb, 
                                                                  timesteps=ts,id_of_starting_symbol=np.random.randint(0, 3),pulse_length=pulse_length,pulse_gap=pulse_gap)
    threeChannels = np.concatenate((threeChannelsOF, padded_zeros), axis=1)
    if (bitstring[w] == "0"):
        labels0or1 = np.zeros((label_length, 1), dtype=float)
    else:
        labels0or1 = np.ones((label_length, 1), dtype=float)
    w += 1
    w  = w % len(bitstring)

    bothLabelsA = np.copy(bothLabels)
    bothPatternsA = np.copy(bothPatterns)
    bothLabels = np.concatenate((bothLabelsA, labels0or1), axis=0)
    bothPatterns = np.concatenate((bothPatternsA, threeChannels), axis=1)
    i += 1
    
tC = bothPatterns.transpose((1, 0))
labels = bothLabels

sweep = 0
ntr_sweep = np.arange(pt, 9e-2, pt,dtype=float)
ntr = ntr_sweep[10]#Uniform(9e-2,1e-3).sample(n=100,d=None,rng=RandomState(reseed))[0]
ntrc_sweep = np.arange(pt, 9e-2, pt,dtype=float)
ntrc = ntrc_sweep[10]
l2_ntr_sweep = np.arange(pt, 9e-2, pt,dtype=float)
l2_ntr = l2_ntr_sweep[40]#Uniform(5e-2,1e-3).sample(n=1,d=None,rng=RandomState(reseed))[0]
l2_ntrc_sweep = np.arange(pt, 9e-2, pt,dtype=float)
l2_ntrc = l2_ntrc_sweep[10]#Uniform(4e-2,1e-3).sample(n=1,d=None,rng=RandomState(reseed))[0]
ff_sweep = np.arange(pt, 9e-2, pt,dtype=float)
ff = ff_sweep[60]#Uniform(1e-2,1e-3).sample(n=1,d=None,rng=RandomState(reseed))[0]
fb_sweep = np.arange(pt, 9e-2, pt,dtype=float)
fb = fb_sweep[0]#Uniform(2e-2,1e-3).sample(n=1,d=None,rng=RandomState(reseed))[0]
while not good:
    model = nengo.Network(label='Two Layers with feedback', seed=reseed)
    sim = nengo.Simulator(model)
    pt = sim.dt
    num_neurons_l1 = 4
    num_neurons_l2 = 6
    with model:
        print("l2_ntr",l2_ntr)
        fb = fb_sweep[sweep]
        layer1 = nengo.Ensemble(
            num_neurons_l1,  # Number of neurons
            dimensions=3,  # each neuron is connected to all (3) input channels.
            # Set intercept to 0.5
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=ntr, tau_rc=ntrc),  # Specify type of neuron
            max_rates=Uniform(1 / (ntr+sim.dt), 1 / (ntr+sim.dt)),  # Set the maximum firing rate of the neuron 500Mhz
        )
        
        layer2 = nengo.Ensemble(
            num_neurons_l2,  # Number of neurons
            dimensions=3,
            # Set intercept to 0.5
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=l2_ntr, tau_rc=l2_ntrc),  # Specify type of neuron
            max_rates=Uniform(1 / (l2_ntr+sim.dt), 1 / (l2_ntr+sim.dt)),  # Set the maximum firing rate of the neuron 500Mhz
        )

    with model:
        input_signal = nengo.Node(PresentInput(tC, presentation_time=pt))
        input_keys = nengo.Node(PresentInput(labels, presentation_time=pt))

    with model:
        nengo.Connection(input_signal, layer1, synapse=None,label="input signals to layer 1")
        nengo.Connection(layer1, layer2, synapse=ff, label="layer 1 to layer2")
        nengo.Connection(layer2, layer1, synapse=fb, label="layer 2 to layer 1")

    simT = label_length*sim.dt*2
    with model:
        input_probe = nengo.Probe(input_signal)  # The original input
        spikes = nengo.Probe(layer1.neurons)  # Raw spikes from each neuron
        # Subthreshold soma voltages of the neurons
        
        #voltage = nengo.Probe(layer1.neurons, 'voltage')
        # Spikes filtered by a 10ms post-synaptic filter
        filteredl1 = nengo.Probe(layer1.neurons, 'voltage', synapse=1e-3)
        filteredl2 = nengo.Probe(layer2.neurons, 'voltage', synapse=1e-3)
        
        # Setup probes
        p_keys = nengo.Probe(input_keys, synapse=None, label="p_keys")
        p_values = nengo.Probe(input_signal, synapse=None, label="p_values")
        
    with nengo.Simulator(model) as sim:  # Create a simulator
        sim.run(simT)  # Run 

    t = sim.trange()

    plot_range = -1  # index

    train = t <= simT / 2
    test = ~train
    #plot_data()

    i = 0
    best_neuron_valueA = np.abs(np.sum(sim.data[filteredl1][train][0:, 0:0 + 1] - (sim.data[p_keys][train])))
    best_neuron_valueB = np.abs(np.sum(sim.data[filteredl1][test][0:, 0:0 + 1] - (sim.data[p_keys][test])))
    best_neuron_indexA = 0
    best_neuron_indexB = 0
    while i < num_neurons_l1:
        sumA = np.abs(np.sum(sim.data[p_keys][train]) - np.sum(sim.data[filteredl1][train][0:, i:i + 1] ))
        sumB = np.abs(np.sum(sim.data[p_keys][test]) - np.sum(sim.data[filteredl1][test][0:, i:i + 1] ))

        if sumA < best_neuron_valueA:
            best_neuron_indexA = i
            best_neuron_valueA = sumA
            #best_neuron_indexB = i
            #best_neuron_valueB = sumB

        if sumB < best_neuron_valueB: 
            #best_neuron_indexA = i
            #best_neuron_valueA = sumA
            best_neuron_indexB = i
            best_neuron_valueB = sumB

        print(best_neuron_valueA, best_neuron_indexA, best_neuron_valueB, best_neuron_indexB, reseed)
        i += 1
        target = np.sum(sim.data[p_keys][test])
        print("Target", target)
    if (best_neuron_valueA < target):
        #print("here")
        if (best_neuron_valueB < target):
            print("now")
             
          #      print("here")
                #good = True
            xls = str(l2_ntrc)
            if len(xls) > 8:
                xls = xls[0:8]
            restart_conditions = "ntr="+str(ntr)+"\nntrc="+str(ntrc)+"\nl2_ntr="+str(l2_ntr)+"\nl2_ntrc="+str(l2_ntrc)+"\nff="+str(ff)+"\nfb="+str(fb)+"\nreseed="+str(reseed)
            print(best_neuron_valueA, best_neuron_indexA, best_neuron_valueB, best_neuron_indexB, reseed)
            plot_data(str(reseed)+"_fb_"+str(sweep),best_neuron_indexA,best_neuron_indexB)
            with open("fig/two-layers-feedback/"+str(reseed)+"_fb_"+str(sweep)+".txt","w") as f:
                f.write(restart_conditions)
                f.close()
    #reseed += 1
    #preset_timing_plot()
    sweep += 1
    if sweep == 99:
        good = True