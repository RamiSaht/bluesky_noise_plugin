# bluesky_noise_plugin
This is a plugin for the Bluesky air traffic simulator [link](https://github.com/TUDelft-CNS-ATM/bluesky)
The plugin simulates the noise impact of an aircraft based on the flight path. Current parameters are set to a B737. The plugin relies on the ANOPP model to simulate the noise impact of an aircraft.
You can change the model by replacing or modifying the "ANOPP_OASPL" function.

## Installation
1. Copy the 'NOISECONTOUR.py' file to the plugins directory.
2. Add "NOISECONTOUR" to "enabled_plugins" in the "settings.cfg" file.
3. (Optional) Add the Noise folder to the scenario folder of Bluesky.
4. Start the simulator and load the scenario using: IC Noise/mini_demp
5. When the simulation is finished, the noise contour will be saved and shown.