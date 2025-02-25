'''
Plugin to create a noise heat map
The user initializes the plugin
For now:
The user has to define which aircrafts are to be considered by their aircraft id
When an aircraft is added to the "to be considered" list, for now we only print the aircraft id to the console
'''

# import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# from random import randint
import numpy as np
import pandas as pd
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools


def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    example = NoiseHeatMap()

    # Configuration parameters
    config = {
        'plugin_name':     'NoiseMap',
        'plugin_type':     'sim',
        }

    return config


class NoiseHeatMap(core.Entity):
    '''Implement a noise heat map plugin for BlueSky'''
    def __init__(self):
        super().__init__()
        self.ac_to_be_considered = []
        self.consider_all_ac = False
        self.noise_area_lats = []
        self.noise_area_lons = []
        self.mesh_size = -1
        self.debugging = False
        self.rng = np.random.default_rng(12345)
        self.mesh = None

    def create(self, n=1):
        super().create(n)
        return
    
    # echo "to be considered aircrafts" to the console
    @core.timed_function(name='update', dt=5)
    def update(self):
        
        # update the list of aircrafts to be considered
        if self.consider_all_ac:
            if not self.consider_all_ac == traf.id:
                self.ac_to_be_considered = traf.id.copy()
        else:
            for ac in self.ac_to_be_considered:
                if ac not in traf.id:
                    self.ac_to_be_considered.remove(ac)
        stack.stack(f"ECHO Aircrafts to be considered for the noise heat map: {[ac for ac in self.ac_to_be_considered]} \n Updating noise heat map...")
        
        
        # update the mesh
        
        return
        
    # command to add an aircraft to the list of aircrafts to be considered
    @stack.command
    def noiseaddac(self, arg):
        if str(arg) == 'all':
            self.consider_all_ac = True
            self.ac_to_be_considered = traf.id.copy()
            stack.stack(f"ECHO All aircrafts will be considered for the noise heat map {[ac for ac in self.ac_to_be_considered]}")
            return
        
        self.ac_to_be_considered.append(arg)
        stack.stack(f"ECHO Aircraft {arg} added to the list of aircrafts to be considered for the noise heat map")
        return
    
    @stack.command
    def noiseac(self):
        stack.stack("ECHO Aircrafts to be considered for the noise heat map:")
        for ac in self.ac_to_be_considered:
            stack.stack(f"ECHO {ac}")
        return
    
    @stack.command
    def noisermvac(self, acid):
        if acid in self.ac_to_be_considered:
            self.ac_to_be_considered.remove(acid)
            stack.stack(f"ECHO Aircraft {acid} removed from the list of aircrafts to be considered for the noise heat map")
        else:
            stack.stack(f"ECHO Aircraft {acid} not in the list of aircrafts to be considered for the noise heat map")
    
    @stack.command
    def noisesetup(self, 
                   lat1:float, lon1:float, 
                   lat2:float, lon2:float, 
                   mesh_x:int=10, mesh_y:int=10,
                   show_grid="false"):
        self.mesh_size = (mesh_y, mesh_x)
        self.noise_area_lats = [lat1, lat2]
        self.noise_area_lons = [lon1, lon2]
        self.mesh = self.rng.integers(0, 100, size=self.mesh_size)
        

        stack.stack(f"ECHO Noise area set up with coordinates: ({self.noise_area_lats[0]}, {self.noise_area_lons[0]}), ({self.noise_area_lats[1]}, {self.noise_area_lons[1]})")
        
        # indicate the area on the radar map example polygon noiseareae,54,3,54,8,50,8,50,3
        stack.stack(f"POLYGON NOISEAREA,{self.noise_area_lats[0]},{self.noise_area_lons[0]},{self.noise_area_lats[1]},{self.noise_area_lons[0]},{self.noise_area_lats[1]},{self.noise_area_lons[1]},{self.noise_area_lats[0]},{self.noise_area_lons[1]}")
        stack.stack(f"COLOUR NOISEAREA,255,0,0")
        
        
        if show_grid=="true":
            for i, lat in enumerate(np.linspace(self.noise_area_lats[0], self.noise_area_lats[1], self.mesh_size[0])[1:-1]):
                stack.stack(f"LINE GRID{i}, {lat}, {self.noise_area_lons[0]}, {lat}, {self.noise_area_lons[1]}")
                stack.stack(f"COLOUR GRID{i}, 0, 0, 255")
            for i, lon in enumerate(np.linspace(self.noise_area_lons[0], self.noise_area_lons[1], self.mesh_size[1])[1:-1]):
                stack.stack(f"LINE GRID{i+self.mesh_size[0]}, {self.noise_area_lats[0]}, {lon}, {self.noise_area_lats[1]}, {lon}")
                stack.stack(f"COLOUR GRID{i+self.mesh_size[1]}, 0, 0, 255")
        elif show_grid=="false":
            pass
        else:
            stack.stack(f"ECHO Invalid input for show_grid={show_grid}. Please enter 'true' or 'false'")
        
        
        return

    @stack.command
    def noisesavempl(self, filename:str):
        if not self.mesh:
            stack.stack("ECHO Noise mesh not set up. Please set up the noise mesh with the 'noisesetup' command")
            return
        num_ticks = 5
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.mesh, cmap='Plasma', interpolation='none', alpha=1, origin='lower')
        ax.set_xticks(np.linspace(0+self.mesh_size[0]/(2*num_ticks), self.mesh_size[0]-self.mesh_size[0]/(2*num_ticks), num_ticks), labels=np.linspace(self.noise_area_lons[0], self.noise_area_lons[1], num_ticks))
        ax.set_yticks(np.linspace(0+self.mesh_size[1]/(2*num_ticks), self.mesh_size[1]-self.mesh_size[1]/(2*num_ticks), num_ticks), labels=np.linspace(self.noise_area_lats[0], self.noise_area_lats[1], num_ticks))
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Noise Heat Map')
        plt.savefig(f'{filename}.png', dpi=300)
        plt.close()
        
        return

    @stack.command
    def noisesavepx(self, filename:str):
        lats_flat = np.zeros(self.mesh_size[0]**2)
        lons_flat = np.zeros(self.mesh_size[1]**2)
        
        lats = np.linspace(self.noise_area_lats[0], 
                           self.noise_area_lats[1], 
                           self.mesh_size[0])
        lons = np.linspace(self.noise_area_lons[0], 
                           self.noise_area_lons[1], 
                           self.mesh_size[1])
        
        lats_flat, lons_flat = np.meshgrid(lats, lons)
        lats_flat = lats_flat.flatten()
        lons_flat = lons_flat.flatten()
        
        mesh_df = pd.DataFrame({'lon': lons_flat, 
                                'lat': lats_flat, 
                                'noise': self.mesh.flatten()})
        
        figpx = px.density_map(mesh_df, 
                       lon='lon', 
                       lat='lat', 
                       z='noise', 
                       title="Noise Heat Map", 
                       opacity=0.25, 
                       zoom=6)
        
        figpx.write_html(f'{filename}.html')

        return
