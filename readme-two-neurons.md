This study looks at an ensemble of two neurons. The variable behavior of the neurons can be controlled by changing parameters. 

This readme looks at the effects of changing the synapse connection for the inputs, and also at the effects 

The inputs are provided by the driven state machine, and they connect to the network through a synapse or through a synapse set to None
```
nengo.Connection(input_signal, neurons, synapse=None)
```
![Image of two neurons with synapse none](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_none.png)

```
nengo.Connection(input_signal, neurons, synapse=1)
```
![Image of two neurons with synapse 1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_1.png)


```
nengo.Connection(input_signal, neurons, synapse=0)
```
![Image of two neurons with synapse 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_0.png)

```
nengo.Connection(input_signal, neurons, synapse=1e-2)
```
![Image of two neurons with synapse 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_1e-2.png)

```
nengo.Connection(input_signal, neurons, synapse=1e-5)
```
![Image of two neurons with synapse 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_1e-5.png)

The synapse range from None, 0 to 1 shows how the number of resulting spikes increases with the stronger synapse. The effects are controlled by the neuron configuration.
```
neurons = nengo.Ensemble(
    2,  # Number of neurons
    dimensions=3,  # each neuron is connected to all (3) input channels.
    intercepts=Uniform(-1e-5, 1e-5),  # Set the intercepts at 0.00001 (threshold for Soma voltage)
    neuron_type=nengo.LIF(min_voltage=0, tau_ref=5e-11, tau_rc=2e-8),  # Specify type of neuron
    # Set tau_ref= or tau_rc = here to
    # change those
    # parms
    # for the
    # neurons.
    max_rates=Uniform(2e+9, 2e+9),             # Set the maximum firing rate of the neuron 500Mhz
    # Set the neuron's firing rate to increase for 2 combinations of 3 channel input.
    encoders=[[-1, -1, 1], [1, -1, -1]],
)
 ```
 
 The dimensions corresponds to the size of the input vectors and is constant. Changing the intercepts does not have much effect.
 
```
intercepts=Uniform(-1e-1, 1e-1),
``` 
 ![Two Neurons with intercepts 1e-1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01.png)
 
 Reducing the Max Rate cause the number of spikes to decrease.
 ```
   max_rates=Uniform(1e+7, 1e+7),
  ```
  ![Two Neurons with intercepts 1e-1 Max Rate 1e+7](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate1e+7.png)

 ```
   max_rates=Uniform(1e+8, 1e+8),
  ```
  
![Two Neurons with intercepts 1e-1 Max rate 1e+8](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate1e+8.png)

 ```
 max_rates=Uniform(1e+9, 1e+9),
 ```
 ![Two Neurons with intercepts 1e-1 Max Rate 1e+9](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate1e+9.png)
 
 
Now, looking at some specific parameters of the [nengo LIF](https://www.nengo.ai/nengo/frontend-api.html#nengo.LIF) model, 

> ***tau_rc*** (float)
> Membrane RC time constant, in seconds. Affects how quickly the membrane voltage decays to zero in the absence of input (larger = slower decay).
> 
> ***tau_ref*** (float)
> Absolute refractory period, in seconds. This is how long the membrane voltage is held at zero after a spike.


> neuron_type=nengo.LIF(min_voltage=0, ***tau_ref=5e-11***, tau_rc=2e-8), 

 ![Tau refractory](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate2e+9_tau_ref5e-11.png)
 
 > neuron_type=nengo.LIF(min_voltage=0, ***tau_ref=2e-10***, tau_rc=2e-8), 
 ![Tau Refractory period lower](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate2e+9_tau_ref2e-10_tau_rc=2e-8.png)
 
  > neuron_type=nengo.LIF(min_voltage=0, ***tau_ref=2e-9***, tau_rc=2e-8), 
 ![Tau Refractory period lower](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate2e+9_tau_ref2e-10_tau_rc=2e-8.png)
 
 Tau_ref is related to the maximum rate, for instance, trying to make Tau_ref = 2e-8 results in the error message:

>LIF.max_rates: Max rates must be below the inverse refractory period (500000000.000) 

That is because the max rate I am using is 2Ghz, which is not possible with a refractory period so long.

Now the min voltage setting.
 ![Min voltage -1](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate2e+9_tau_ref2e-11_tau_rc=2e-8_min_voltage_-1.png)

 ![Min Voltage 0](https://github.com/kariefury/rotation-machine-3/blob/main/fig/two_neuronsinput_signal_synapse_None_intercepts_-1e-1-1e-01_maxrate2e+9_tau_ref2e-11_tau_rc=2e-8_min_voltage_0.png)
