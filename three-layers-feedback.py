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
    
    people = ('Sim dt', 'Pulse Length', 'Pulse Gap', 'Rotation Time', 'Label Time', 'L1 Tau RC', 'L1 Tau Ref', 
    'L2 Tau RC', 
              "L2 Tau Ref", 
              "L3 Tau RC", "L3 Tau REf")
    y_pos = np.arange(len(people))
    print (y_pos)
    performance = np.zeros(len(people),dtype=float)
    lbs = [1,2,3,4,5,6,7,8,9,10]
    #performance = 3 + 10 * np.random.rand(len(people))
    performance[0] = sim.dt
    performance[1] = (pt*pulse_length)
    performance[2] = (pt*pulse_gap)
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
    plt.tight_layout() 
    plt.savefig('fig/preset_timing_parameters/preset_timing_parmeters.png')

def plot_data(q):
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
    plt.close()
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
    plt.close()
    plt.figure()
    plt.title("Filtered L3 output")

    plt.plot(t, sim.data[p_keys] + 0, color="black")
    plt.plot(t, sim.data[input_probe] + 2)
    print(np.shape(sim.data[filteredl2]))
    plt.plot(t, sim.data[filteredl2][0:, 0:1] + 3, color="b")
    plt.plot(t, sim.data[filteredl2][0:, 1:2] + 4, color="y")
    plt.savefig('fig/3layersfeedbck_neuronsl3_2_search'+q+'.png')
    #plt.show()

    plt.clf()
    plt.close()
    plt.figure()
    time_input_signal = t <= (pt*ts*(pulse_length+pulse_gap))
    plt.title("single timesteps of input")
    plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 0:1], color="blue")
    plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 1:2]+2.1, color="green")
    plt.plot(t[time_input_signal],sim.data[input_probe][time_input_signal][0: , 2:3] + 4.2 , color="orange")
    plt.savefig('fig/inputTiming.png')

    plt.close()

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


reseed = 11524#9166#8 #91521 #91427 #91377 #91362 #91332 #91323 #91280 #91274 # 91264 # 91254  # 91273
good = False
number_of_samples = 2
ts = 6  # number of possible transitions to hold a driving symbol constant for.
pb = False
pt = 3e-3 # seconds to present each step of input
pulse_length = 2
pulse_gap = 0
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

bitstring = "0111100001010000000111111101010101010011001100011100101010000111100000000000000000000000000000000000000000000000000000000000000000"
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


while not good:
    model = nengo.Network(label='Three Layers with feedback', seed=reseed)
    sim = nengo.Simulator(model)
    pt = sim.dt
    num_neurons_l1 = 4
    num_neurons_l2 = 6
    num_neurons_l3 = 2
    with model:
        ntr = Uniform(7e-2,4e-3).sample(1)[0]
        ntrc = Uniform(1e-2,4e-3).sample(1)[0]
        l2_ntr = Uniform(1e-2,4e-3).sample(1)[0]
        l2_ntrc = Uniform(1e-1,4e-3).sample(1)[0]
        l3_ntr = Uniform(1e-2,4e-3).sample(1)[0]
        l3_ntrc = Uniform(1e-2,1e-3).sample(1)[0]
        # ntr= 0.04857797742868027
        # ntrc= 0.005298126253483592
        # l2_ntr= 0.07872599281604377
        # l2_ntrc= 0.004172924890959869
        # l3_ntr= 0.008004251267465179
        # l3_ntrc= 0.0015320812730265365
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

        layer3 = nengo.Ensemble(
            num_neurons_l3,  # Number of neurons
            dimensions=3,
            # Set intercept to 0.5
            neuron_type=nengo.LIF(min_voltage=0, tau_ref=l3_ntr, tau_rc=l3_ntrc),  # Specify type of neuron
            max_rates=Uniform(1 / (l3_ntr+sim.dt), 1 / (l3_ntr+sim.dt)),  # Set the maximum firing rate of the neuron 500Mhz
        )
        


    with model:
        input_signal = nengo.Node(PresentInput(tC, presentation_time=pt))
        input_keys = nengo.Node(PresentInput(labels, presentation_time=pt))

    with model:
        nengo.Connection(input_signal, layer1, synapse=None,label="input signals to layer 1")
        ff = Uniform(2e-1,2e-2).sample(1)[0]
        fb = Uniform(2e-1,2e-2).sample(1)[0]
        nengo.Connection(layer1, layer2, synapse=ff, label="layer 1 to layer2")
        nengo.Connection(layer2, layer1, synapse=fb, label="layer 2 to layer 1")
        conn = nengo.Connection(layer2, layer3, synapse=ff, label="layer 2 to layer 3")
        nengo.Connection(layer3, layer2, synapse=fb, label="layer3 to layer2")

    simT = label_length*sim.dt*2
    with model:
        input_probe = nengo.Probe(input_signal)  # The original input
        spikes = nengo.Probe(layer1.neurons)  # Raw spikes from each neuron
        # Subthreshold soma voltages of the neurons
        
        #voltage = nengo.Probe(layer1.neurons, 'voltage')
        # Spikes filtered by a 10ms post-synaptic filter
        filteredl1 = nengo.Probe(layer1.neurons, 'voltage', synapse=1e-3)
        filteredl2 = nengo.Probe(layer2.neurons, 'voltage', synapse=1e-3)
        filteredl3 = nengo.Probe(layer3.neurons, 'voltage', synapse=1e-3)
        
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
    #plot_data()

    i = 0
    best_neuron_valueA = np.abs(np.sum(sim.data[filteredl3][train][0:, 0:0 + 1] - (sim.data[p_keys][train])))
    #print(best_neuron_valueA)
    best_neuron_valueB = np.abs(np.sum(sim.data[filteredl3][test][0:, 0:0 + 1] - (sim.data[p_keys][test])))
    #print(best_neuron_valueB)
    best_neuron_indexA = 0
    best_neuron_indexB = 0
    while i < num_neurons_l3:
        #print(i, "sim.data[filteredl3]", np.sum(sim.data[filteredl3][0:, i:i + 1]) , "sim.data[p_keys] ", np.sum(sim.data[p_keys]) )
        sumA = np.abs(np.sum(sim.data[p_keys][train]) - np.sum(sim.data[filteredl3][train][0:, i:i + 1] ))
        #print("train",i, sumA)
        sumB = np.abs(np.sum(sim.data[p_keys][test]) - np.sum(sim.data[filteredl3][test][0:, i:i + 1] ))
        #print("test",i, sumB)

        if sumA < 20:
            best_neuron_indexA = i
            best_neuron_valueA = sumA
            #best_neuron_indexB = i
            #best_neuron_valueB = sumB

        if sumB < 20:
            #best_neuron_indexA = i
            #best_neuron_valueA = sumA
            best_neuron_indexB = i
            best_neuron_valueB = sumB

        print(best_neuron_valueA, best_neuron_indexA, best_neuron_valueB, best_neuron_indexB, reseed)
        i += 1

    if (best_neuron_valueA < 20):
        #print("here")
        if (best_neuron_valueB < 20):
         #   print("now")
            if(best_neuron_indexA == best_neuron_indexB):
          #      print("here")
                #good = True
                print("ntr=",ntr,
                "\nntrc=",ntrc,
                "\nl2_ntr=",l2_ntr,
                "\nl2_ntrc=",l2_ntrc,
                "\nl3_ntr=",l3_ntr,
                "\nl3_ntrc=",l3_ntrc,
                "\nff=",ff,
                "\nfb=",fb)
                print(best_neuron_valueA, best_neuron_indexA, best_neuron_valueB, best_neuron_indexB, reseed)
                plot_data(str(reseed))
            else:
                reseed += 1
        else:
            reseed += 1
    else:
        reseed += 1

    #preset_timing_plot()    
    #good = True

plt.figure()
plt.title("Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[p_keys] + 0, color="g")
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[filteredl3][0:, best_neuron_indexA:best_neuron_indexA + 1] + 3, color="black")
plt.plot(t, sim.data[filteredl3][0:, best_neuron_indexB:best_neuron_indexB + 1] + 5, color="gray")
#plt.savefig('fig/3layersfeedbck_best_neuron_index_l3.png')



with model:
    #print(model)
    #print(model.all_connections)
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
    #print(model.all_connections)

with nengo.Simulator(model) as sim:  # Create a simulator
    sim.run(simT)  # Run 

plt.figure()
plt.title("Learned Filtered output")
# print(np.shape(sim.data[p_recall][test][0:,0:1]), np.shape(sim.data[p_values][test][0:,0:1]))
# plt.plot(t, sim.data[p_recall][0:,0:1])# - sim.data[p_values][0:,0:1])
plt.plot(t, sim.data[error_values],color="red")
plt.plot(t, sim.data[p_keys] + 0, color="g")
plt.plot(t, sim.data[input_probe] + 2)
plt.plot(t, sim.data[filteredl3][0:, best_neuron_indexA:best_neuron_indexA + 1] + 3, color="black")
plt.plot(t, sim.data[filteredl3][0:, best_neuron_indexB:best_neuron_indexB + 1] + 5, color="gray")

#plt.savefig('fig/3layersfeedbck_best_neuron_index_l3_learned.png')