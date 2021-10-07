Using the driven state machine to produce 3 channels of events, now, classify the 3 channels back into the 2 original driving symbols '0' or '1'.

## Layer 1

Start with 4 neurons, each receiving 3 channels of input and outputting 3 channels of output. 

![Image of 4 neurons](https://github.com/kariefury/rotation-machine-3/blob/main/fig/four_neurons.png)

## Layer 2
Now connect the outputs of the 4 neurons to Layer 2 consisting of 6 neurons, each receiving 3 channels of input and outputting 3 channels of output each.

![Image of 6 neurons](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_layers.png)

## Layer 3
Finally, connect the outputs of the 6 neurons to Layer 3 consisting of 2 neurons, each receiving 3 channels of input and outputting 3 channels each.

![Image of 2 neurons, driving symbol 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three_layers0.png)

![Image of 2 neurons, driving symbol 1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three_layers1.png)

Probing the neurons with a synapse filter results in:
![Image of filter](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three_layers_filtered0.png)
![Image of filter](https://github.com/kariefury/rotation-machine-3/blob/main/fig/three_layers_filtered1.png)

