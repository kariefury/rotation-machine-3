The global timing of any simulation of a 3 layer feedback network consists of several parameters shown below: (all in seconds)
![Preset timing parameters](https://github.com/kariefury/rotation-machine-3/blob/main/fig/preset_timing_parameters.png)
Sim dt is the simulation timestep. Pulse time is the length of time a single pulse is logical 1. Pulse Spacing is the time between rotating pulses that is 0 for all channels. It is not a functional part of the program through because increasing that to a non-zero value would change the graph of the Rotation Machine generating the 3 channel symbols.

The rotating phase machine can operate for varying numbers of timesteps. The image below shows the output when it is set to rotate for 9 timesteps.

![Input timing with 9 timesteps](https://raw.githubusercontent.com/kariefury/rotation-machine-3/main/fig/inputTiming.png)
