
import numpy as np
import matplotlib.pyplot as plt
from bluesky import core, stack, traf, sim
from bluesky.tools import geo
import time

### Initialization function of the plugin
def init_plugin():
    ''' Plugin initialisation function. '''
    traf.noise_contour = NoiseContour()

    config = {
        'plugin_name': 'NOISECONTOUR',
        'plugin_type': 'sim',
    }

    # stackfunctions = {
    # 'NOISESETUP': [
    #     'NOISESETUP', '<lat1>', '<lon1>', '<lat2>', '<lon2>', '<mesh_x>', '<mesh_y>', '<num>',
    #     NoiseContour.noisesetup,
    #     'Turn on plugin and set up noise contour monitoring area'],
    # }
    # return config, stackfunctions

    return config

### Define NoiseContour Entity
class NoiseContour(core.Entity):
    def __init__(self):
        super().__init__()
        self.active = False
        self.grid_points = None
        self.E_day = None
        self.active_flyovers = {}  # Track active flyovers
        self.E_sum = {}  # Track per-aircraft energy summation
        self.finalization_time = {} # Track when aircraft left monitoring area

    def generate_grid(self):
        ''' Generate grid points in local aircraft-relative coordinates (Δx, Δy). '''
        # self.grid_points[:,:,2] to acces it
        self.grid_points = np.zeros((self.mesh_size[0], self.mesh_size[1], 2))
        
        self.noise_area_lats = np.linspace(self.lat1, self.lat2, num=self.mesh_size[0])	
        self.noise_area_lons = np.linspace(self.lon1, self.lon2, num=self.mesh_size[0])
        
        self.noise_area_lats = np.sort(self.noise_area_lats)
        self.noise_area_lons = np.sort(self.noise_area_lons)
        
        
        for i in range(self.mesh_size[0]):
            for j in range(self.mesh_size[1]):
                self.grid_points[i, j, 0] = self.noise_area_lats[i].copy()
                self.grid_points[i, j, 1] = self.noise_area_lons[j].copy()
        
        
        stack.stack(f"POLYGON NOISEAREA,{self.lat1},{self.lon1},{self.lat1},{self.lon2},{self.lat2},{self.lon2},{self.lat2},{self.lon1}")
        stack.stack(f"COLOUR NOISEAREA,255,0,0")
        
        return
    
    def latlon2xy(self, lat1, lon1, lat2, lon2):
            ''' Convert lat/lon difference to meters. '''
            dx = geo.latlondist(lat1, lon1, lat1, lon2)
            dy = geo.latlondist(lat1, lon1, lat2, lon1)
            return dx, dy
        
    def compute_relative_pos(self, ac_id, ac_lat, ac_lon, ac_alt, ac_spd, ac_hdg, dt):
        ''' Compute noise at each grid point based on aircraft position. '''
        
    
        if ac_id not in self.E_sum:
            self.E_sum[ac_id] = np.zeros(self.grid_points.shape[:2])

        # Compute relative position in meters
        dx, dy = self.latlon2xy(ac_lat, ac_lon, self.grid_points[:, :, 0], self.grid_points[:, :, 1]) # Convert lat/lon diff to meters
        dz = ac_alt  # Altitude difference (assuming ground level at 0m) converted to meters
        spd = ac_spd 
        ac_hdg_rad = np.radians(ac_hdg)
        # Rotate dx, dy into aircraft body frame (aircraft-centric coordinates)
        dx_body = dx * np.cos(ac_hdg_rad) + dy * np.sin(ac_hdg_rad) # Forward direction
        dy_body = -dx * np.sin(ac_hdg_rad) + dy * np.cos(ac_hdg_rad) # Side direction
        dist = np.sqrt(dx_body**2 + dy_body**2 + dz**2) # 3D distance to aircraft 

        # Compute polar directivity angle (wrt aircraft nose direction)
        polar = np.arctan2(dz, dx_body) # Angle from horizontal axis

        # Compute azimuth angle (angle from aircraft yaw axis)
        azimuth = np.arctan2(dy_body, dz) # Angle from vertical axis

        OASPL = np.zeros(self.grid_points.shape[:2])
        
        for i in range(self.mesh_size[0]):
            for j in range(self.mesh_size[1]):
                OASPL[i,j] = self.ANOPP_OASPL(azimuth[i, j], polar[i, j], dz, dist[i, j], spd)
        # Placeholder for ANOPP noise calculation
        # OASPL = self.ANOPP_OASPL(azimuth, polar, dz, dist, spd)

        # Update energy sum for SEL
        self.E_sum[ac_id] += 10**(OASPL / 10) * dt
    
    def ANOPP_OASPL(self, azimuth, polar, dz, dist, spd):
        
        #Flight parameters
        T_0 = 288.15
        rho_0 = 1.225
        R = 287.05
        g = 9.80665
        l = -0.0065
        gamma = 1.4 
        mu = 1.84E-5 #ambient dynamic viscosity [kg/(ms)]
        theta = polar
        phi = azimuth
        r = dist
        pe02 = (2E-5)**2 #reference value

        #Aircraft parameters (currently defined for B737)
        A_w  = 130 #wing area [m^2]
        b_w = 34 #wing span [m]
        A_f = 18 #flap area [m^2]
        b_f = 17 #flap span [m]
        df = 30 * np.pi/180 #flap deflection agle #[rad]
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

        PBL_total= 10 * np.log10(10**(PBL_wing/10) + 10**(PBL_slats/10) + 10**(PBL_flap/10) + 10**(PBL_MLG/10) + 10**(PBL_NLG/10)) #PSL - total

        #A-weighting function
        dL_A = -145.528 + 98.262 * np.log10(f_n) - 19.509 * (np.log10(f_n))**2 + 0.975 * (np.log10(f_n))**3
        PBL_total_A = PBL_total + dL_A

        #OASPL (or L_A) calculation
        sum = 0
        for i in range(len(PBL_total_A)):
            sum += 10 ** (PBL_total_A[i] / 10) 

        OASPL = 10 * np.log10(sum)

        return OASPL
    
    def visualize_noise(self, L_DEN):
        ''' Placeholder function for visualizing the noise contour. '''
        filename = f'noise_{time.strftime("%Y%m%d_%H%M%S")}'
        num_ticks = 5
        fig, ax = plt.subplots(figsize=(6, 6))
        m = plt.imshow(L_DEN, cmap='viridis', origin='lower')
        ax.set_xticks(np.linspace(0+self.mesh_size[0]/(2*num_ticks), self.mesh_size[0]-self.mesh_size[0]/(2*num_ticks), num_ticks),
                      labels=[f'{t:.2f}'for t in np.linspace(self.noise_area_lons[0], self.noise_area_lons[1], num_ticks)])
        ax.set_yticks(np.linspace(0+self.mesh_size[1]/(2*num_ticks), self.mesh_size[1]-self.mesh_size[1]/(2*num_ticks), num_ticks),
                      labels=[f'{t:.2f}'for t in np.linspace(self.noise_area_lats[0], self.noise_area_lats[1], num_ticks)])
        
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Noise Heat Map')
        plt.colorbar(m, ax=ax, label='L_DEN (dB)')
        plt.savefig(f'{filename}.png', dpi=300)
        plt.show()
        # plt.close()
        return

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

        if alt > 1000:  # Assume 3000ft as max altitude for monitoring
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
        self.E_day += 10 ** ((SEL + W) / 10)  # Add to daily noise energy

        del self.E_sum[ac_id]  # Remove this aircraft’s E_sum data

    def finalize_day(self, flyover_time):
        ''' Compute L_DEN at the end of all flyovers. '''
        L_DEN = 10*np.log10(1/flyover_time) + 10 * np.log10(np.maximum(self.E_day, 1e-10))  # Convert accumulated SEL to L_DEN
        return L_DEN
    
    def run_simulation(self, num_flyovers):
        ''' Manage multiple flyovers without freezing the simulation. '''
        self.remaining_flyovers = num_flyovers  # Store flyover count
        self.active_flyover = False  # No flyover is currently running


    @core.timed_function(name='update_noise', dt=1)
    def update(self):
        ''' Update noise contour based on aircraft position and altitude. '''
        print("Updating noise contour")
        if len(traf.id) == 0:
            return  # No aircraft in simulation
        
        min_lat, max_lat = np.min(self.grid_points[:, :, 0]), np.max(self.grid_points[:, :, 0])
        min_lon, max_lon = np.min(self.grid_points[:, :, 1]), np.max(self.grid_points[:, :, 1])
        
        for i in range(len(traf.id)):  # Loop over all aircraft
            ac_id = traf.id[i]  # Get aircraft ID
            lat, lon, alt, spd, hdg = traf.lat[i], traf.lon[i], traf.alt[i], traf.tas[i], traf.hdg[i]
       
            
            if not ((min_lat <= lat) and (lat <= max_lat) and (min_lon <= lon) and (lon <= max_lon)):
                print(f"Aircraft {ac_id} is outside the monitoring area.")
                continue  # Skip if aircraft is outside the grid
            
            if alt > 1000:  # Ignore aircraft above 3000 ft
                print(f"Aircraft {ac_id} is above 1000 ft.")
                continue  
            
            print(f"Computing noise for aircraft {ac_id}")
            # Compute noise contribution for this aircraft
            self.compute_relative_pos(ac_id, lat, lon, alt, spd, hdg, dt=1)


    @core.timed_function(name='monitor_flyovers', dt=1)  # Run every 0.05s
    def monitor_flyovers(self):
        ''' Monitors and finalizes flyovers in the background. '''
        print("Monitoring flyovers")
        if self.remaining_flyovers <= 0:  
            return  # Stop when all flyovers are completed

        active_aircraft = set(traf.id)  # Get current aircraft IDs in the simulation
        # Iterate over all tracked aircraft
        for ac_id in list(self.active_flyovers.keys()):
            if ac_id not in active_aircraft or not self.flyover_is_active(ac_id):  
                if ac_id not in self.finalization_time:
                    self.finalization_time[ac_id] = time.time()  # Record exit time

                elif (time.time() - self.finalization_time[ac_id] > 1) and (self.remaining_flyovers > 1): # Wait for 1s before finalizing
                    # If aircraft is no longer active, finalize its flyover
                    end_time = sim.simtclock
                    self.finalize_flyover(ac_id, end_time)
                    del self.active_flyovers[ac_id] # Remove from active list
                    print(f"Aircraft {ac_id} left the monitoring area.")
                    self.finalization_time.pop(ac_id, None)  # Safe delete. Remove from finalization tracking
                    self.remaining_flyovers -= 1
                else:
                    end_time = sim.simtclock
                    self.finalize_flyover(ac_id, end_time)
                    del self.active_flyovers[ac_id] # Remove from active list
                    print(f"Aircraft {ac_id} left the monitoring area.")
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
                print(f"Aircraft {ac_id} entered the monitoring area.")
                if self.remaining_flyovers == self.num_flyovers:
                    print("First flyover started.")
                    self.sim_start_time = sim.simt

        if self.remaining_flyovers == 0:
            print("All flyovers completed.")
            
            flyover_time = self.sim_end_time - self.sim_start_time
            final_L_DEN = self.finalize_day(flyover_time)
            sim.hold() # Pause simulation
            self.visualize_noise(final_L_DEN)


    @stack.command
    def noisesetup(self, 
                   lat1:float, lon1:float, 
                   lat2:float, lon2:float, 
                   mesh_x:int=25, mesh_y:int=25,
                   num: int = None):
        
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        
        self.mesh_size = (mesh_y, mesh_x)
        self.generate_grid()         # Define grid
        self.E_day = np.zeros(self.grid_points.shape[:2]) # Track daily energy summation


        self.active = not self.active

        if num is not None:
            if num <= 0:
                return False, "Number of flyovers must be positive."
            self.num_flyovers = num

        if self.active:
            self.run_simulation(num_flyovers=self.num_flyovers)  # Start non-blocking simulation

        if self.active and num is None:
            return False, "Noise contour is already active."

        return True, f"Noise visualization {'enabled' if self.active else 'disabled'}, flyovers: {self.num_flyovers}"
        
        
        
        