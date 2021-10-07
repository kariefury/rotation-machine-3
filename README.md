# rotation-machine-3
The following experiment describes a spiking neural network for studying the output of a driven state machine, represented in the picture below with three states: A,B,C.

![Image of 3 states and their connections](https://github.com/kariefury/rotation-machine-3/blob/main/fig/driven_state_machine.png)

The state machine can be represented as a 3 x 3 matrix of transition probabilities, where the probability of transition **p** represents how unlikely it is for the machine to remain in the same state for two measurement periods. **p** is a positive number between 0 - 1.

With a driving symbol of 1 binary input (x) into the 3 state (A,B,C) when x = 0, p = 0 it walks A,C,B. If **p** is non-zero and positive it will randomly walk A,C, C, C, B, B, A, A, C ... where the number of times it occurs in the same state represented by no spikes on the 3 output channels (represented as the edges where x = 0 or 1).

The function that implements the driven state machine is called `phase_automata()` and its source code can be found [here](https://github.com/kariefury/rotation-machine-3/blob/main/three-phase-rotation-machine.py).
```
phase_automata(driving_symbol='0', # Corresponds to x = 0, x = 1
                number_of_symbols=3, # All of these experiments use 3 states for A,B,C. The function is flexible and allows for different numbers.
                id_of_starting_symbol=0, # All of these experiments start from state A, that means x = 0 produces A,C,B,A while x = 1 produces A,B,C,A. The starting symbol is not changed with the driving symbol.
                timesteps=9, # The resulting matrix will contain the pattern for 9 timesteps. This is only of consequence when probability of transition = True 
                probability_of_transition=False #Causes p to be non-zero and positive, since it uses a random number, causes non-deterministic behaviror for simulation.)
```
                
The output of `phase_automata()` is a matrix number_of_symbols x timesteps, where 1 represents a spike and a non-spike is represented by a 0. It can also be rewritten for 1 to represent a spike and a -1 for non-spike.

Then it is fed into the Nengo neuron simulation using the Nengo function `PresentInput()`.
When x = 0 it randomly walks A,C,B.

![Image of 3 Channels, driving symbol 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_signals_driving_symbol0.png)

When x = 1 it randomly walks A,B,C. 
![Image of 3 Channels, driving symbol 1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/input_signals_driving_symbol1.png)

When the three pulse channels are used as inputs into a LIF neuron, the neuron produces events.
![Image of 1 neuron](https://github.com/kariefury/rotation-machine-3/blob/main/fig/short_time_neuron.png)

The spike events are shown in blue on the left and the subthreshold neuron voltage on the right in red.

## Examples of Layers and Networks
1. **2 Neurons** 2 Neurons in an ensemble, used to produce plots of 2 neurons
2. **2 Layer network** 2 Layers connected feedforward
3. **[3 Layer network]**(https://github.com/kariefury/rotation-machine-3/blob/main/Readme_three_layers.md)


