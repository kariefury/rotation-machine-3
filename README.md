# rotation-machine-3
When a driving symbol that is 1 binary input (x) into a 3 state (A,B,C) Markov Chain with a probability transition p = 100%, to the next state is simulated, with a driving symbol 0 and 3 pulse outputs (A,B,C), it randomly walks C,B,A if x = 0.

![Image of 3 Channels, driving symbol 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_signals_driving_symbol0.png)

When x = 1 it randomly walks A,B,C. 
![Image of 3 Channels, driving symbol 1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_signals_driving_symbol1.png)

When the three pulse channels are used as inputs into a LIF neuron, the neuron produces events.
![Image of 1 neuron](https://github.com/kariefury/rotation-machine-3/blob/main/fig/short_time_neuron.png)

The spike events are shown in blue on the left and the subthreshold neuron voltage on the right in red.

## Examples of Layers and Networks
1. [3 Layer network](https://github.com/kariefury/rotation-machine-3/blob/main/Readme_three_layers.md)


