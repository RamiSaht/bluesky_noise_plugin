import numpy as np
from bluesky import core, stack, traf
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

    stackfunctions = {
    'TOGGLENOISE': [
        'TOGGLENOISE', '<num>',
        NoiseContour.toggle_noise_contour,
        'Toggles noise visualization on or off (optionally sets flyovers)']
    }

    return config, stackfunctions

### Define NoiseContour Entity
class NoiseContour(core.Entity):
    def __init__(self):
        super().__init__()
        self.active = False
        self.active_flyovers = {}  # Track active flyovers
        self.grid_points = self.generate_grid()         # Define grid
        self.E_sum = {}  # Track per-aircraft energy summation
        self.E_day = np.zeros(self.grid_points.shape[:2]) # Track daily energy summation
        self.finalization_time = {} # Track when aircraft left monitoring area
        self.W = 0  # Weighting factor for SEL calculation needs to be developed further

    def generate_grid(self):
        ''' Generate grid points in local aircraft-relative coordinates (Δx, Δy). '''
        pass
    
    def compute_relative_pos(self, ac_id, ac_lat, ac_lon, ac_alt, ac_spd, ac_hdg, dt):
        ''' Compute noise at each grid point based on aircraft position. '''

        if ac_id not in self.E_sum:
            self.E_sum[ac_id] = np.zeros(self.grid_points.shape[:2])

        # Compute relative position in meters
        dx, dy = geo.latlon2xy(ac_lat, ac_lon, self.grid_points[:, :, 0], self.grid_points[:, :, 1]) # Convert lat/lon diff to meters
        dz = ac_alt * 0.3048  # Altitude difference (assuming ground level at 0m) converted to meters
        spd = ac_spd * 0.514444  # Convert speed from knots to m/s

        # Rotate dx, dy into aircraft body frame (aircraft-centric coordinates)
        dx_body = dx * np.cos(ac_hdg) + dy * np.sin(ac_hdg) # Forward direction
        dy_body = -dx * np.sin(ac_hdg) + dy * np.cos(ac_hdg) # Side direction
        dist = np.sqrt(dx_body**2 + dy_body**2 + dz**2) # 3D distance to aircraft 

        # Compute polar directivity angle (wrt aircraft nose direction)
        polar = np.arctan2(dz, dx_body) # Angle from horizontal axis

        # Compute azimuth angle (angle from aircraft yaw axis)
        azimuth = np.arctan2(dy_body, dz) # Angle from vertical axis

        # Placeholder for ANOPP noise calculation
        OASPL = self.ANOPP_OASPL(azimuth, polar, dz, dist, spd)

        # Update energy sum for SEL
        self.E_sum[ac_id] += 10**(OASPL / 10) * dt
    
    def ANOPP_OASPL(self, azimuth, polar, dz):
        
        ''' Placeholder function for ANOPP-based noise calculation. '''
        pass
    
    def visualize_noise(self, L_DEN):
        ''' Placeholder function for visualizing the noise contour. '''
        pass


    def flyover_is_active(self, ac_id):
        ''' Check if the given aircraft is still flying within the monitored area. '''
        if ac_id not in traf.id:
            return False  # Aircraft no longer exists

        i = traf.id.index(ac_id)
        lat, lon, alt = traf.lat[i], traf.lon[i], traf.alt[i]

        min_lat, max_lat = np.min(self.grid_points[:, :, 0]), np.max(self.grid_points[:, :, 0])
        min_lon, max_lon = np.min(self.grid_points[:, :, 1]), np.max(self.grid_points[:, :, 1])

        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return False  # Aircraft left the monitoring area

        if alt > 3000:  # Assume 3000ft as max altitude for monitoring
            return False

        return True
    

    def finalize_flyover(self, ac_id):
        ''' Compute SEL for a specific aircraft and update total L_DEN contribution. '''
        if ac_id not in self.E_sum:
            return  # No energy data for this aircraft (should not happen)

        SEL = 10 * np.log10(np.maximum(self.E_sum[ac_id], 1e-10))  # Prevent log(0) issues. Compute SEL from stored E_sum
        self.E_day += 10 ** ((SEL + self.W) / 10)  # Add to daily noise energy

        del self.E_sum[ac_id]  # Remove this aircraft’s E_sum data


    def finalize_day(self):
        ''' Compute L_DEN at the end of all flyovers. '''
        L_DEN = -49.4 + 10 * np.log10(np.maximum(self.E_day, 1e-10))  # Convert accumulated SEL to L_DEN
        return L_DEN
    
    def run_simulation(self, num_flyovers):
        ''' Manage multiple flyovers without freezing the simulation. '''
        self.remaining_flyovers = num_flyovers  # Store flyover count
        self.active_flyover = False  # No flyover is currently running


    @core.timed_function(name='update_noise', dt=0.05)
    def update(self):
        ''' Update noise contour based on aircraft position and altitude. '''
        if len(traf.id) == 0:
            return  # No aircraft in simulation
        
        min_lat, max_lat = np.min(self.grid_points[:, :, 0]), np.max(self.grid_points[:, :, 0])
        min_lon, max_lon = np.min(self.grid_points[:, :, 1]), np.max(self.grid_points[:, :, 1])

        for i in range(len(traf.id)):  # Loop over all aircraft
            ac_id = traf.id[i]  # Get aircraft ID
            lat, lon, alt, spd, hdg = traf.lat[i], traf.lon[i], traf.alt[i], traf.spd[i], traf.trk[i]

            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue  # Skip if aircraft is outside the grid
            
            if alt > 3000:  # Ignore aircraft above 3000 ft
                continue  

            # Compute noise contribution for this aircraft
            self.compute_relative_pos(ac_id, lat, lon, alt, spd, hdg, dt=0.05)


    @core.timed_function(name='monitor_flyovers', dt=0.05)  # Run every 0.05s
    def monitor_flyovers(self):
        ''' Monitors and finalizes flyovers in the background. '''
        if self.remaining_flyovers <= 0:  
            return  # Stop when all flyovers are completed

        active_aircraft = set(traf.id)  # Get current aircraft IDs in the simulation

        # Iterate over all tracked aircraft
        for ac_id in list(self.active_flyovers.keys()):
            if ac_id not in active_aircraft or not self.flyover_is_active(ac_id):  
                if ac_id not in self.finalization_time:
                    self.finalization_time[ac_id] = time.time()  # Record exit time

                elif time.time() - self.finalization_time[ac_id] > 5:  # Wait 5 sec before finalizing
                    # If aircraft is no longer active, finalize its flyover
                    self.finalize_flyover(ac_id)
                    del self.active_flyovers[ac_id] # Remove from active list
                    self.finalization_time.pop(ac_id, None)  # Safe delete. Remove from finalization tracking
                    self.remaining_flyovers -= 1

            else:
                if ac_id in self.finalization_time:  # Reset finalization timer if aircraft returns
                    del self.finalization_time[ac_id]

        # Check for new aircraft entering the monitored area
        for i in range(len(traf.id)):
            ac_id = traf.id[i]
            if ac_id not in self.active_flyovers and self.flyover_is_active(ac_id):
                self.active_flyovers[ac_id] = True  # Mark new flyover as active

        if self.remaining_flyovers == 0:
            final_L_DEN = self.finalize_day()
            self.visualize_noise(final_L_DEN)


    @stack.command
    def toggle_noise_contour(self, num: int = None):
        """Toggle noise contour visualization on/off."""
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

    
#IMPLEMENT LAST SUGGESTIONS