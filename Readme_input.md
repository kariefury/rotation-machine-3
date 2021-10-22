The global timing of any simulation of a 3 layer feedback network consists of several parameters shown below: (all in seconds)
![Preset timing parameters](https://github.com/kariefury/rotation-machine-3/blob/main/fig/preset_timing_parameters/preset_timing_parmeters.png)

Sim dt is the simulation timestep (default is 0.001s). Pulse Length is the length of time a single pulse is high. Pulse Gap is the length of time a single pulse is low.

Rotation time is the seconds the machine is emitting pulses from the driving symbol. Label length is the seconds the label signal is held constant against the rotation machine.
The layer neuron parameters for refractory time **tau_ref** and RC time **tau_RC** are shown for each of the 3 layers.

The rotating phase machine can operate for varying numbers of timesteps. The image below shows the output when it is set to rotate for 6 timesteps.

![Input timing with 6 timesteps](https://raw.githubusercontent.com/kariefury/rotation-machine-3/main/fig/inputTiming.png)
