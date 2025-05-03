"""
Plugin for noise heatmap visualization. Using knowledge of course AE4431-23 Aircraft noise 
Basic usage:
With the simulator cleared and paused, run the following commands:
0.Load using: Plugin load NOISECONTOUR
1. NOISESETUP, lat1, lon1, lat2, lon2, mesh_x, mesh_y, num, plot_type
    This command sets up the noise monitoring area and starts the noise visualization.
    The rectangle defined by opposite corners (lat1, lon1) and (lat2, lon2) is divided into mesh_x x mesh_y grid points.
    Example: Noisesetup 52.1 4.4 52.7 5.2 35 35 1 mpl  -> Would give a 35x35 grid with the area between 52.1, 4.4 and 52.7, 5.2 (Around EHAM)
    num: Number of flyovers to expect.
    plot_type: 'mpl' for matplotlib, 'px' for plotly express.
2. Start the simulation and let the aircraft fly over the monitoring area.
3. The plugin will automatically pause the simulation after the last flyover.
4. A noise heatmap will be generated and saved as a PNG or HTML file in the output folder.

Note: The plugin uses the ANOPP model to calculate noise levels. Currently, the plugin is set up for a Boeing 737 aircraft.
You can edit the model or use a different one by modifying/replacing the ANOPP_OASPL function. The function needs to return a grid of noise levels in dBA each timestep.
"""


import numpy as np
import matplotlib.pyplot as plt
from bluesky import core, stack, traf, sim
from bluesky.tools import geo
import time

### Initialization function of the plugin (neede for all plugins)
def init_plugin():
    ''' Plugin initialisation function. '''
    traf.noise_contour = NoiseContour()

    config = {
        'plugin_name': 'NOISECONTOUR',
        'plugin_type': 'sim',
    }

    return config

### Define NoiseContour Entity
class NoiseContour(core.Entity):
    def __init__(self):
        super().__init__()
        self.active = False
        self.grid_points = None
        self.E_period = None
        self.active_flyovers = {}  # Track active flyovers
        self.E_sum = {}  # Track per-aircraft energy summation
        self.finalization_time = {} # Track when aircraft left monitoring area
        self.L_DEN = None
        

    @stack.command
    def noisesetup(self, 
                   lat1:float, lon1:float, 
                   lat2:float, lon2:float, 
                   mesh_x:int=25, mesh_y:int=25,
                   num: int = 1):
        
        # set up the noise monitoring area
        if lat1 == lat2 or lon1 == lon2:
            return False, "Noise area must have non-zero area."
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        self.mesh_size = (mesh_y, mesh_x)
        self.L_DEN = np.zeros(self.mesh_size)
        self.generate_grid()         # Define grid
        self.E_period = np.zeros(self.grid_points.shape[:2]) # setup energy sum

        
        if num is not None:
            if num <= 0:
                return False, "Number of flyovers must be positive."
            self.num_flyovers = num

        self.active = not self.active
        
        if self.active:
            self.remaining_flyovers = self.num_flyovers

        if self.active and num is None:
            return False, "Noise contour is already active."

        return True, f"Noise visualization {'enabled' if self.active else 'disabled'}, flyovers: {self.num_flyovers}"

    @core.timed_function(name='update_noise', dt=1)
    def update(self):
        ''' Update noise contour based on aircraft position and altitude. '''
        if len(traf.id) == 0:
            return  # No aircraft in simulation
        
        min_lat, max_lat = np.min(self.grid_points[:, :, 0]), np.max(self.grid_points[:, :, 0])
        min_lon, max_lon = np.min(self.grid_points[:, :, 1]), np.max(self.grid_points[:, :, 1])
        
        for i in range(len(traf.id)):  # Loop over all aircraft
            ac_id = traf.id[i]  # Get aircraft ID
            lat, lon, alt, spd, hdg = traf.lat[i], traf.lon[i], traf.alt[i], traf.tas[i], traf.hdg[i]
            
            
            if not ((min_lat <= lat) and (lat <= max_lat) and (min_lon <= lon) and (lon <= max_lon)):
                continue  # Skip if aircraft is outside the grid
            
            if alt > 7000 * 0.3048:  # Ignore aircraft above 7000 ft
                continue  
            
            # Compute noise contribution for this aircraft
            self.calculate_noise_contributions(ac_id, lat, lon, alt, spd, hdg, dt=1)

    @core.timed_function(name='monitor_flyovers', dt=1)  # Run every 0.05s
    def monitor_flyovers(self):
        ''' Monitors and finalizes flyovers in the background. '''
        if self.remaining_flyovers <= 0:  
            return  # Stop when all flyovers are completed

        active_aircraft = set(traf.id)  # Get current aircraft IDs in the simulation
        # Iterate over all tracked aircraft
        for ac_id in list(self.active_flyovers.keys()):
            # if aircraft is no longer active or has left the monitoring area, finalize its flyover
            if ac_id not in active_aircraft or not self.flyover_is_active(ac_id):  
                if ac_id not in self.finalization_time:
                    self.finalization_time[ac_id] = time.time()  # Record exit time

                elif (time.time() - self.finalization_time[ac_id] > 1) and (self.remaining_flyovers > 1): # Wait for 1s before finalizing
                    # If aircraft is no longer active, finalize its flyover
                    end_time = sim.utc
                    self.finalize_flyover(ac_id, end_time)
                    del self.active_flyovers[ac_id] # Remove from active list
                    self.finalization_time.pop(ac_id, None)  # Safe delete. Remove from finalization tracking
                    self.remaining_flyovers -= 1
                else:
                    end_time = sim.utc
                    self.finalize_flyover(ac_id, end_time)
                    del self.active_flyovers[ac_id] # Remove from active list
                    self.finalization_time.pop(ac_id, None)  # Safe delete. Remove from finalization tracking
                    self.remaining_flyovers -= 1
                    self.sim_end_time = sim.simt

            else:
                if ac_id in self.finalization_time:  # Reset finalization timer if aircraft returns
                    del self.finalization_time[ac_id]
                    

        # Check for new aircraft entering the monitored area
        for i in range(len(traf.id)):
            ac_id = traf.id[i]
            if ac_id not in self.active_flyovers and self.flyover_is_active(ac_id):
                self.active_flyovers[ac_id] = True  # Mark new flyover as active
                if self.remaining_flyovers == self.num_flyovers:
                    self.sim_start_time = sim.simt
        
        # Check if all flyovers are completed
        if self.remaining_flyovers == 0:            
            flyover_time = self.sim_end_time - self.sim_start_time
            final_L_DEN = self.finalize_simulation(flyover_time)
            self.L_DEN = final_L_DEN
            sim.hold() # Pause simulation
            self.visualize_noise_mpl(self.L_DEN)

    def generate_grid(self):
        ''' Generate grid points for noise monitoring area. '''
        # Create a 3D array to store lat, lon, and altitude. Use self.grid_points[:,:,0] and self.grid_points[:,:,1] to acces lat and lon respectively.
        
        self.noise_area_lats = np.linspace(self.lat1, self.lat2, num=self.mesh_size[0])
        self.noise_area_lats.sort()
        self.noise_area_lons = np.linspace(self.lon1, self.lon2, num=self.mesh_size[1])
        self.noise_area_lons.sort()

        # Create a meshgrid of latitudes and longitudes
        lat_grid, lon_grid = np.meshgrid(self.noise_area_lats, self.noise_area_lons, indexing='ij')
        self.grid_points = np.stack((lat_grid, lon_grid), axis=-1)
        
        # Plot the noise area on the radar screen
        stack.stack(f"POLYGON NOISEAREA,{self.lat1},{self.lon1},{self.lat1},{self.lon2},{self.lat2},{self.lon2},{self.lat2},{self.lon1}")
        stack.stack(f"COLOUR NOISEAREA,255,0,0")
        
        return
    
    def latlon2xy(self, lat1, lon1, lat2, lon2):
            ''' Convert lat/lon difference to dx and dy in meters. '''
            dx = geo.latlondist(lat1, lon1, lat1, lon2)
            dy = geo.latlondist(lat1, lon1, lat2, lon1)

            dy = np.where(lat2 < lat1, -dy, dy)
            dx = np.where(lon2 < lon1, -dx, dx)
            return dx, dy

    def calculate_noise_contributions(self, ac_id, ac_lat, ac_lon, ac_alt, ac_spd, ac_hdg, dt):
        ''' Compute noise at each grid point based on aircraft position. Add the result to the energy sum. '''
        
        if ac_id not in self.E_sum:
            self.E_sum[ac_id] = np.zeros(self.grid_points.shape[:2])

        # Compute relative position in meters
        dx, dy = self.latlon2xy(ac_lat, ac_lon, self.grid_points[:, :, 0], self.grid_points[:, :, 1]) # Convert lat/lon diff to meters
        dz = ac_alt  # Altitude difference (assuming ground level at 0m) 
        spd = ac_spd 
        ac_hdg_rad = np.radians(ac_hdg)
        # Rotate dx, dy into aircraft body frame (map coordinates: x: east, y: north, z: up) (aircraft-centric coordinates, x: forward (out of nose), y: out of right wing, z: down)
        dx_body = dy * np.cos(ac_hdg_rad) + dx * np.sin(ac_hdg_rad)  # Convert to aircraft body frame
        dy_body = -dy * np.sin(ac_hdg_rad) + dx * np.cos(ac_hdg_rad)  # Convert to aircraft body frame
        
        dist = np.sqrt(dx_body**2 + dy_body**2 + dz**2) # 3D distance to aircraft 

        # Compute polar directivity angle (wrt aircraft nose direction)
        polar = np.arctan2(dz, dx_body) # Angle from horizontal axis

        # Compute azimuth angle (angle from aircraft yaw axis)
        azimuth = np.arctan2(np.abs(dy_body), dz) # Angle from vertical axis

        OASPL = np.zeros(self.grid_points.shape[:2])
        
        for i in range(self.mesh_size[0]):
            for j in range(self.mesh_size[1]):
                OASPL[i,j] = self.ANOPP_OASPL(azimuth[i, j], polar[i, j], dz, dist[i, j], spd)

        # Update energy sum for SEL
        self.E_sum[ac_id] += 10**(OASPL / 10) * dt
    
    def ANOPP_OASPL(self, azimuth, polar, dz, dist, spd):
        
        
        # Constatns and parameters for the airframe are defined in the script below to avoid allocating memory each time the function is called
        
        theta = polar
        phi = azimuth
        

        #Flap control
        V = spd * 1.94384 #speed [kts]
        flaps = 0
        df = 0
        if 191 <= V < 201:
            flaps = 1
            df = np.radians(5)
        elif 175 <= V < 191:
            flaps = 1
            df = np.radians(15)
        elif 161 < V < 175:
            flaps = 1
            df = np.radians(25)
        elif V <= 161:
            flaps = 1
            df = np.radians(30)

        #Landing gear control
        gear = 0
        if dz < 2000 * 0.3048:
            gear = 1


        #Frequencies for 1/3 octave band
        band_numbers = np.arange(1, 44) #band numbers
        f_n = 10**(band_numbers / 10) #centre frequencies
        f_U = 2**(1/6) * f_n #upper bound
        f_L = f_n / (2**(1/6)) #lower bound
        delta_f = f_U - f_L #bandwidth

        #Ambient conditions
        temp = T_0 + l * dz
        c = np.sqrt(gamma * R * temp)
        M = spd / c
        rho = rho_0 * (temp / T_0)**(-g / (R * l) - 1)  

        
        #clean wing calculations
        G_cw = 0.37 * (A_w/b_w**2) * ((rho * M * c * A_w)/(mu * b_w))**(-0.2)
        L_cw = G_cw * b_w
        S_cw = (f_n * L_cw * (1 - M * np.cos(theta))) / (M * c)
        F_cw = 0.613 * (10 * S_cw)**4 * ((10 * S_cw)**1.5 + 0.5)**(-4)
        D_cw = 4 * (np.cos(phi))**2 * (np.cos(theta/2))**2
        P_cw = (K_CW * M**a_CW * G_cw * (rho * c**3 * b_w**2 ))
        p_e_2_cw = (rho * c * P_cw * D_cw * F_cw) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)


        #slats calculations
        G_sl = 0.37 * (A_w/b_w**2) * ((rho * M * c * A_w)/(mu * b_w))**(-0.2)
        L_sl = G_sl * b_w
        S_sl = (f_n * L_sl * (1 - M * np.cos(theta))) / (M * c)
        F_sl = 0.613 * (10 * S_sl)**4 * ((10 * S_sl)**1.5 + 0.5)**(-4) + 0.613 * (2.19 * S_sl)**4 * ((2.19 * S_sl)**1.5 + 0.5)**(-4)
        D_sl = 4 * (np.cos(phi))**2 * (np.cos(theta/2))**2
        P_sl = (K_SL * M**a_SL * G_sl * (rho * c**3 * b_w**2 ))
        p_e_2_sl = (rho * c * P_sl * D_sl * F_sl) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)


        #flaps calculations
        G_fl = A_f/b_w**2 * (np.sin(df))**2
        L_fl = A_f / b_f
        S_fl = (f_n * L_fl * (1 - M * np.cos(theta))) / (M * c)
        F_fl = np.empty(S_fl.shape)
        F_fl[S_fl < 2] = 0.0480 * S_fl[S_fl < 2]
        F_fl[(2 <= S_fl) & (S_fl <= 20)] = 0.1406 * (S_fl[(2 <= S_fl) & (S_fl <= 20)])**(-0.55)
        F_fl[S_fl > 20] = 216.49 * (S_fl[S_fl > 20])**(-3)
        D_fl = 3 * (np.sin(df) * np.cos(theta) + np.cos(df) * np.sin(theta) * np.cos(phi))**2
        P_fl = (K_FL * M**a_FL * G_fl * (rho * c**3 * b_w**2 ))
        p_e_2_fl = (rho * c * P_fl * D_fl * F_fl) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

        #MLG calculations
        G_mlg = n_MLG * (d_MLG/b_w)**2
        L_mlg = d_MLG
        S_mlg = (f_n * L_mlg * (1 - M * np.cos(theta))) / (M * c)
        #F_mlg = n_MLG * 0.0577 * S_mlg**2 * (0.25 * S_mlg**2 + 1)**(-1.5) #for MLG with 4 wheels
        F_mlg = n_MLG * 13.59 * S_mlg**2*(S_mlg**2 + 12.5)**(-2.25) #for MLG with 2 wheels
        D_mlg = 3/2 * (np.sin(theta))**2
        P_mlg = (K_LDG_4 * M**a_LDG * G_mlg * (rho * c**3 * b_w**2 ))
        p_e_2_mlg = (rho * c * P_mlg * D_mlg * F_mlg) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

        #NLG calculations
        G_nlg = 2 * (d_NLG/b_w)**2
        L_nlg = d_NLG
        S_nlg = (f_n * L_nlg * (1 - M * np.cos(theta))) / (M * c)
        F_nlg = 2 * 13.59 * S_nlg**2*(S_nlg**2 + 12.5)**(-2.25)
        D_nlg = 3/2 * (np.sin(theta))**2
        P_nlg = (K_LDG_2 * M**a_LDG * G_nlg * (rho * c**3 * b_w**2 ))
        p_e_2_nlg = (rho * c * P_nlg * D_nlg * F_nlg) / (4 * np.pi * r**2 * (1 - M * np.cos(theta))**4)

        #OSPL calculations
        PBL_wing = 10 * np.log10(p_e_2_cw/pe02) #- 10 * np.log10(delta_f) #PSL - wing
        PBL_slats = 10 * np.log10(p_e_2_sl/pe02) #- 10 * np.log10(delta_f) #PSL - slats
        PBL_flap = 10 * np.log10(p_e_2_fl/pe02) #- 10 * np.log10(delta_f) #PSL - flaps
        PBL_MLG = 10 * np.log10(p_e_2_mlg/pe02) #- 10 * np.log10(delta_f) #PSL - main landing gear
        PBL_NLG = 10 * np.log10(p_e_2_nlg/pe02) #- 10 * np.log10(delta_f) #PSL - nose landing gear


        PBL_total= 10 * np.log10(10**(PBL_wing/10) + 10**(PBL_slats/10) * flaps + 10**(PBL_flap/10) * flaps + 10**(PBL_MLG/10) * gear + 10**(PBL_NLG/10) * gear) #PSL - total with landing gear

        #A-weighting function
        dL_A = -145.528 + 98.262 * np.log10(f_n) - 19.509 * (np.log10(f_n))**2 + 0.975 * (np.log10(f_n))**3
        PBL_total_A = PBL_total + dL_A - 20 * np.log10(dist) + 4 # a few decibels added to account for  engine noise

        #OASPL (or L_A) calculation + very rudimentary atmospheric attentuation
        total_pressure_sum = 0
        for i in range(len(PBL_total_A)):
            if 63 <= f_n[i] < 125:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 0.108 * 1e-3 * dist) / 10) 
            elif 125 <= f_n[i] < 250:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 0.373 * 1e-3 * dist) / 10) 
            elif 250 <= f_n[i] < 500:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 1.02 * 1e-3 * dist) / 10) 
            elif 500 <= f_n[i] < 1000:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 1.96 * 1e-3 * dist) / 10) 
            elif 1000 <= f_n[i] < 2000:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 3.57 * 1e-3 * dist) / 10) 
            elif 2000 <= f_n[i] < 4000:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 8.80 * 1e-3 * dist) / 10) 
            elif 4000 <= f_n[i] < 8000:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 29.0 * 1e-3 * dist) / 10) 
            elif 8000 <= f_n[i]:
                total_pressure_sum += 10 ** ((PBL_total_A[i] - 105 * 1e-3 * dist) / 10) 
            else:
                total_pressure_sum += 10 ** (PBL_total_A[i]/10)

        OASPL = 10 * np.log10(total_pressure_sum)

        return OASPL
    
    def flyover_is_active(self, ac_id):
        ''' Check if the given aircraft is still flying within the monitored area. '''
        if ac_id not in traf.id:
            return False  # Aircraft no longer exists

        i = traf.id.index(ac_id)
        lat, lon, alt = traf.lat[i], traf.lon[i], traf.alt[i]

        min_lat, max_lat = np.min(self.grid_points[:, :, 0]), np.max(self.grid_points[:, :, 0])
        min_lon, max_lon = np.min(self.grid_points[:, :, 1]), np.max(self.grid_points[:, :, 1])

        if not ((min_lat <= lat) and (lat <= max_lat) and (min_lon <= lon) and (lon <= max_lon)):
            return False  # Aircraft left the monitoring area

        if alt > 7000 * 0.3048:  # Assume 6500ft as max altitude for monitoring
            return False

        return True
    
    def finalize_flyover(self, ac_id, utc_time):
        ''' Compute SEL for a specific aircraft and update total L_DEN contribution. '''
        if ac_id not in self.E_sum:
            return  # No energy data for this aircraft (should not happen)
        
        hour = utc_time.hour  

        # Check if within the night period (23:00 - 07:00)
        if hour >= 23 or hour < 7:
            W = 10
        elif 7 <= hour < 19:
            W = 0
        else:
            W = 5

        SEL = 10 * np.log10(np.maximum(self.E_sum[ac_id], 1e-10))  # Prevent log(0) issues. Compute SEL from stored E_sum
        self.E_period += 10 ** ((SEL + W) / 10)  # Add to daily noise energy

        del self.E_sum[ac_id]  # Remove this aircraft’s E_sum data

    def finalize_simulation(self, flyover_time):
        ''' Compute L_DEN at the end of all flyovers. '''
        L_DEN = 10*np.log10(1/flyover_time) + 10 * np.log10(np.maximum(self.E_period, 1e-10))  # Convert accumulated SEL to L_DEN
        
        # Apply minimum noise level of 45 dB
        L_DEN = np.maximum(L_DEN, 30)
        
        return L_DEN     

    def visualize_noise_mpl(self, L_DEN):
        ''' Visualize noise contour using matplotlib. '''
        filename = f'noise_{time.strftime("%Y%m%d_%H%M%S")}'
        num_ticks = 5
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Set extent so the image spans the real-world coordinate ranges
        extent = [self.noise_area_lons[0], self.noise_area_lons[-1], self.noise_area_lats[0], self.noise_area_lats[-1]]
        m = ax.imshow(L_DEN, cmap='jet', origin='lower', extent=extent)
        
        # Create tick locations directly from lon/lat ranges
        xticks = np.linspace(self.noise_area_lons[0], self.noise_area_lons[-1], num_ticks)
        yticks = np.linspace(self.noise_area_lats[0], self.noise_area_lats[-1], num_ticks)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.2f}" for x in xticks])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.2f}" for y in yticks])
        
        ax.set_aspect('auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Noise Heat Map')
        plt.colorbar(m, ax=ax, label='L_DEN (dBA)')
        
        plt.tight_layout()  # Remove extra white space around the plot
        plt.savefig(f'output/{filename}.png', dpi=300)
        stack.stack(f"ECHO Noise contour saved as output/{filename}.png")
        plt.show()
        return
    

### constants
#Flight parameters
T_0 = 288.15
rho_0 = 1.225
R = 287.05
g = 9.80665
l = -0.0065
gamma = 1.4 
mu = 1.84E-5 #ambient dynamic viscosity [kg/(ms)]
r = 1
pe02 = (2E-5)**2 #reference value

#Aircraft parameters (currently defined for B737)
A_w  = 130 #wing area [m^2]
b_w = 34 #wing span [m]
A_f = 18 #flap area [m^2]
b_f = 17 #flap span [m]

n_MLG = 2 #number of wheels per boggie (MLG) [-]
d_MLG = 1.13 #diameter of MLG [m]
d_NLG = 0.7 #diameter of NLG [m]   

#Parameters for geometry function
K_CW = 4.646E-5 #K constant for trailing edge clean wing [-]
K_SL = 4.646E-5 #K constant for leading edge slats [-]
K_FL = 2.787E-4 #K constant for trailing edge slats [-]
K_LDG_2 = 4.349E-4 #K constant for landing gear with 2 wheels [-]
K_LDG_4 = 3.414E-4 #K constant for landing gear with 4 wheels [-]
a_CW = 5 #a constant for clean wing [-]
a_SL = 5 #a constant for LE slats [-]
a_FL = 6 #a constant for TE flaps [-]
a_LDG = 6 #a constant for landing gear (both MLG and NLG) [-]