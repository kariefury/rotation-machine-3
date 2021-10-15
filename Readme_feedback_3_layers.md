### Learning with a 3 Layer, 12 LIF Neuron Network using 3 feedback connections.

Starting off with training data generated from the 3 phase rotation machine, a set of training data is created. Two examples of timeseries from the data is shown in the following figures. The label is the black line and when it switching from low to high, the direction of the pattern reverses.

![input_pattern_example_probTran_False_padded_zeros_true](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_pattern_example_probTran_False_padded_zeros_true.png)

The second example shows how the pattern looks when there is a non-zero chance of failing to transition. **p = 0.2**

![input_pattern_example_probTran_True_padded_zeros_true](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_pattern_example_probTran_True_padded_zeros_true.png)

The network has 3 layers (ensembles). An ensemble is fully connected within it's layer.
```
  model = nengo.Network(label='Three Layers with feedback', seed=reseed)
      num_neurons_l1 = 4
      num_neurons_l2 = 6
      num_neurons_l3 = 2
      with model:
          with model:
              layer1 = nengo.Ensemble(num_neurons_l1,  dimensions=3, 
                  neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                  max_rates=Uniform(1 / 6e-2, 1 / 6e-2))  # Set the maximum firing rate of the ensemble
              
              layer2 = nengo.Ensemble(num_neurons_l2,dimensions=3,
                  neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                  max_rates=Uniform(1 / 6e-2, 1 / 6e-2))  # Set the maximum firing rate of the ensemble
              
              layer3 = nengo.Ensemble(num_neurons_l3,dimensions=3,
                neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-2, tau_rc=1e-2),  # Specify type of neuron
                max_rates=Uniform(1 / 6e-2, 1 / 6e-2))  # Set the maximum firing rate of the ensemble
```

The connections between the input and layer 1 are setup. Then the connections both feed-forward and feedback between layers 1, 2 and 3.

```
# ts = 45 in this example. ts is the number of time steps to devote to generating a input signal corresponding to the driving symbol.
with model:
    nengo.Connection(input_signal, layer1, synapse=None)
    nengo.Connection(layer1, layer2, synapse=1e-1)
    nengo.Connection(layer2, layer1, synapse=ts*1e-2)
    nengo.Connection(layer2, layer3, synapse=1e-2)
    nengo.Connection(layer3, layer2, synapse=ts*1e-1)
```

![three-layers-feedback](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three-layers-feedback.png)

To train the model, a series of training data and labels is created and then the model is ran though the dataset.
While it is running, the output on the layer 3 neurons is probed.
The difference between each L3 neuron probe and the labels is measured and if the difference is small eneough it is a good model, capable of reproducing the driving symbol from data stored only in the rotation direction of the finite state machine.

![3layersfeedbck_neuronsl3_2time180s](https://github.com/kariefury/rotation-machine-3/blob/main/fig/3layersfeedbck_neuronsl3_2time180s.png)

The figure above shows the output of the 2 neurons from Layer 3.

When the experiment is running the spike pattern changes dependent on the direction of the ```input_signal```.
![Input Signal](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three-layers-feedback-plots.png)
