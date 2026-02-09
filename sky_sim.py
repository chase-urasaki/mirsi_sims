#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#%%
class SkyBackgroundExposure:
    def __init__(self, sky_background_rate, read_noise, dark_current=0, 
                 quantum_efficiency=0.5, array_shape=(240, 320)):
        """
        Initialize sky background limited exposure simulator.
        
        Parameters:
        -----------
        sky_background_rate : float
            Sky background photon rate (e-/s/pixel)
        read_noise : float
            Read noise (e- rms)
        dark_current : float
            Dark current (e-/s/pixel)
        quantum_efficiency : float
            Detector quantum efficiency (0-1)
        array_shape : tuple
            Detector array dimensions (height, width)
        """
        self.sky_background_rate = sky_background_rate
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.quantum_efficiency = quantum_efficiency
        self.array_shape = array_shape
    
    def simulate_exposure(self, exposure_time, seed=None):
        """
        Simulate a single exposure with sky background, dark current, and read noise.
        
        Parameters:
        -----------
        exposure_time : float
            Exposure time in seconds
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        frame : ndarray
            Simulated detector frame (electrons)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sky background signal (Poisson noise)
        sky_signal = self.sky_background_rate * self.quantum_efficiency * exposure_time
        sky_frame = np.random.poisson(sky_signal, self.array_shape)
        
        # Dark current (Poisson noise)
        dark_signal = self.dark_current * exposure_time
        dark_frame = np.random.poisson(dark_signal, self.array_shape)
        
        # Read noise (Gaussian)
        read_noise_frame = np.random.normal(0, self.read_noise, self.array_shape)
        
        # Total frame
        frame = sky_frame + dark_frame + read_noise_frame
        
        return frame
    
#%%
def compute_sky_background_rate(wavelength, temperature):
# Compute the sky background rate for an atmosphere at 273k at 10 microns using the planck function
    def planck(wavelength, temperature):
        """
        Compute the Planck function for a black body at a given temperature.

        Parameters:
        -----------
        wavelength : float
            Wavelength in meters
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        intensity : float
            Spectral radiance in W/m^2/m
        """
        h = 6.626e-34  # Planck's constant (J*s)
        c = 3.0e8      # Speed of light (m/s)
        k = 1.381e-23  # Boltzmann's constant (J/K)

        # Planck's law
        intensity = (2*h*c**2) / (wavelength**5 * (np.exp((h*c)/(wavelength*k*temperature)) - 1))
        return intensity

    # Constants
    wavelength = 10e-6   # 10 microns in meters
    temperature = 273    # Temperature in Kelvin
    h = 6.626e-34       # Planck's constant (J*s)
    c = 3.0e8           # Speed of light (m/s)

    # Detector and telescope parameters
    pixel_size = 50e-6  # Pixel size in meters (typical for IR detectors)
    f_number = 37       # Typical f/# for IRTF instrument
    bandpass = 5e-6     # Filter bandpass in meters (e.g., 1 micron wide)
    telescope_area = np.pi * (1.5)**2  # Telescope area in m^2 (8m diameter example)
    transmission = 0.3  # Typical system transmission

    # Calculate solid angle per pixel
    solid_angle_per_pixel = (pixel_size / f_number)**2  # steradians

    # Spectral radiance from Planck function
    spectral_radiance = planck(wavelength, temperature)  # W/m^2/sr/m

    # Power per pixel
    power_per_pixel = spectral_radiance * solid_angle_per_pixel * bandpass  # W/m^2

    # Photon energy
    photon_energy = h * c / wavelength  # J/photon

    # Photons per second per pixel
    photon_rate = (power_per_pixel / photon_energy) * transmission  # photons/s/pixel

    sky_background_rate = photon_rate
    print(f"Planck Sky background rate: {sky_background_rate:.2e} photons/s/pixel")

    return sky_background_rate

def compute_sky_background_rate_with_tel(wavelength, temperature_sky, temperature_telescope, 
                                emissivity_telescope=0.1):
    """
    Compute the sky background rate including contributions from both sky and telescope.
    
    Parameters:
    -----------
    wavelength : float
        Wavelength in meters
    temperature_sky : float
        Sky temperature in Kelvin
    temperature_telescope : float
        Telescope mirror temperature in Kelvin
    emissivity_telescope : float
        Telescope thermal emissivity (0-1), default 0.1 for aluminum mirrors
    
    Returns:
    --------
    total_rate : float
        Total photon rate (photons/s/pixel)
    """
    def planck(wavelength, temperature):
        """
        Compute the Planck function for a black body at a given temperature.

        Parameters:
        -----------
        wavelength : float
            Wavelength in meters
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        intensity : float
            Spectral radiance in W/m^2/sr/m
        """
        h = 6.626e-34  # Planck's constant (J*s)
        c = 3.0e8      # Speed of light (m/s)
        k = 1.381e-23  # Boltzmann's constant (J/K)

        # Planck's law
        intensity = (2*h*c**2) / (wavelength**5 * (np.exp((h*c)/(wavelength*k*temperature)) - 1))
        return intensity

    # Constants
    h = 6.626e-34       # Planck's constant (J*s)
    c = 3.0e8           # Speed of light (m/s)

    # Detector and telescope parameters
    pixel_size = 50e-6  # Pixel size in meters (typical for IR detectors)
    f_number = 37       # Typical f/# for IRTF instrument
    bandpass = 5e-6     # Filter bandpass in meters (e.g., 1 micron wide)
    telescope_area = np.pi * (1.5)**2  # Telescope area in m^2 (3m diameter)
    transmission = 0.3  # Typical system transmission

    # Calculate solid angle per pixel
    solid_angle_per_pixel = (pixel_size / f_number)**2  # steradians

    # Photon energy
    photon_energy = h * c / wavelength  # J/photon

    # ===== SKY CONTRIBUTION =====
    # Spectral radiance from sky
    spectral_radiance_sky = planck(wavelength, temperature_sky)  # W/m^2/sr/m
    
    # Power per pixel from sky
    power_per_pixel_sky = spectral_radiance_sky * solid_angle_per_pixel * bandpass  # W/m^2
    
    # Photons per second per pixel from sky (transmitted through optics)
    photon_rate_sky = (power_per_pixel_sky / photon_energy) * transmission  # photons/s/pixel

    # ===== TELESCOPE CONTRIBUTION =====
    # Spectral radiance from telescope
    spectral_radiance_telescope = planck(wavelength, temperature_telescope)  # W/m^2/sr/m
    
    # Power per pixel from telescope emission (scaled by emissivity)
    power_per_pixel_telescope = (spectral_radiance_telescope * solid_angle_per_pixel * 
                                 bandpass * emissivity_telescope)  # W/m^2
    
    # Photons per second per pixel from telescope (no additional transmission loss)
    photon_rate_telescope = power_per_pixel_telescope / photon_energy  # photons/s/pixel

    # ===== TOTAL RATE =====
    total_rate = photon_rate_sky + photon_rate_telescope
    
    print(f"Sky background rate: {photon_rate_sky:.2e} photons/s/pixel")
    print(f"Telescope thermal emission rate: {photon_rate_telescope:.2e} photons/s/pixel")
    print(f"Total background rate: {total_rate:.2e} photons/s/pixel")
    print(f"Telescope contribution: {100*photon_rate_telescope/total_rate:.1f}%")

    return total_rate

#%%
# ============================================================================
# Test/Demo code - only runs when this file is executed directly
# ============================================================================
if __name__ == "__main__":
    #%%
    # Detector parameters
    read_noise = 800 #e- / read rms
    dark_current = 50 # e- / s / pixel
    #%%

    # Compute sky background rate 
    sky_background_rate = compute_sky_background_rate(10e-6, 273)

    # Initialize the SkyBackgroundExposure simulator
    simulator = SkyBackgroundExposure(sky_background_rate, read_noise, dark_current)
    # Simulate an exposure of 50 ms 
    exposure_time = 0.05  # seconds
    frame = simulator.simulate_exposure(exposure_time)

# %%
# Insert a photometric aperture function into the SkyBackgroundExposure cl

if __name__ == "__main__":
    # %%
    # Redo with aperture photometry
    # Use photutils to do aperture photometry
    from photutils import CircularAnnulus

    # Test 

   
    # %%