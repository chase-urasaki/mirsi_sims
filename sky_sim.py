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
    print(f"Sky background rate: {sky_background_rate:.2e} photons/s/pixel")

    return sky_background_rate
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
# Insert a photometric aperture function into the SkyBackgroundExposure class
class SkyBackgroundExposure_with_aperture(SkyBackgroundExposure):
    def photometric_aperture(self, frame, aperture_radius, center=None):
        """
        Perform photometric aperture on the given frame.
        
        Parameters:
        -----------
        frame : ndarray
            Simulated detector frame (electrons)
        aperture_radius : float
            Radius of the aperture in pixels
        center : tuple, optional
            (x, y) coordinates of the aperture center. If None, use center of the frame.
            
        Returns:
        --------
        total_signal : float
            Total signal within the aperture (electrons)
        """
        y_size, x_size = frame.shape
        if center is None:
            center = (x_size // 2, y_size // 2)
        
        y_indices, x_indices = np.ogrid[:y_size, :x_size]
        distance_from_center = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)
        
        aperture_mask = distance_from_center <= aperture_radius
        aperture_counts = frame[aperture_mask]
        total_signal = np.sum(frame[aperture_mask])

        # Draw the aperture on the frame for visualization
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='gray', origin='lower')
        aperture_circle = Circle(center, aperture_radius, color='red', fill=False)
        ax.add_patch(aperture_circle)
        plt.show()

        return aperture_counts, total_signal
# %%
# Redo with aperture photometry
simulator_aperture = SkyBackgroundExposure_with_aperture(sky_background_rate, read_noise, dark_current)
frame = simulator_aperture.simulate_exposure(exposure_time)
aperture_radius = 10  # pixels
aperture_counts, total_signal = simulator_aperture.photometric_aperture(frame, aperture_radius, center=None)
print(f"Total signal within aperture: {total_signal:.2f} electrons")
#%% 
# Plot with the simulated image and two histograms: one for the aperture counts and one for the full frame counts
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Image
axs[0].imshow(frame, cmap='gray', origin='lower')
axs[0].add_patch(Circle((frame.shape[1]//2, frame.shape[0]//2), aperture_radius, color='red', fill=False))
# Color bar for image
cbar = plt.colorbar(axs[0].imshow(frame, cmap='gray', origin='lower'), ax=axs[0])
cbar.set_label('Electrons')
axs[0].set_title('Simulated Detector Frame')
# Aperture counts histogram
axs[1].hist(aperture_counts.flatten(), bins=30, color='blue', alpha=0.7)
axs[1].set_title('Aperture Counts Histogram')
# Full frame counts histogram
axs[2].hist(frame.flatten(), bins=30, color='green', alpha=0.7)
axs[2].set_title('Full Frame Counts Histogram')
plt.tight_layout()
plt.show()
# %%
