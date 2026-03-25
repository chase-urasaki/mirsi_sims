#%%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

wl_ang, E = np.genfromtxt('mk_skybg_nq_10_10_ph.dat', unpack=True)
# Data has units of ph/sec/as^2/nm/m^2, and wavelength in nm. Convert to microns and check units carefully!
#%%
# convert from angstroms to nm
wl_nm = wl_ang / 10.0

#%% print the first few values to check
print("Wavelength (um):", wl_nm[:5])
print("Sky emission (ph/s/m^2/arcsec^2/nm):", E[:5])

#%% 
# Plot the sky emission spectrum
plt.figure(figsize=(10, 6))
plt.plot(wl_nm, E, label='Sky Emission Spectrum')
#plt.xlim(8, 14)
plt.ylim(0, np.max(E)*1.1)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Sky Emission (ph/s/m^2/arcsec^2/nm)')
plt.title('Sky Emission Spectrum (7-14 um)')
plt.grid()
#%%
# load the filter profile from the .dat file 
filter_wl, filter_trans = np.genfromtxt('IRTF_MIRSI.N.dat', unpack=True)
#%%
print(filter_wl)
#%% 
# Convert filter wl from angstroms to nm 
filter_wl_nm = filter_wl / 100.0  # angstroms to nm
#%% 
# Use the twin y axis to plot the filter transmission on the same plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(wl_nm, E, label='Sky Emission Spectrum', color='blue')
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Sky Emission (ph/s/m^2/arcsec^2/nm)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(filter_wl_nm, filter_trans, label='Filter Transmission', color='orange')
ax2.set_ylabel('Filter Transmission', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.title('Sky Emission Spectrum and Filter Transmission')
plt.grid()
plt.show()  


#%% 
# Interpolate the filter transmission onto the sky emission wavelength grid
from scipy.interpolate import interp1d
filter_interp = interp1d(filter_wl_nm, filter_trans, bounds_error=False, fill_value=0.0)
filter_on_sky = filter_interp(wl_nm)
#%%
# Apply the filter transmission to the sky emission spectrum
E_filtered = E * filter_on_sky  

plt.figure(figsize=(10, 6))
plt.plot(wl_nm, E, label='Sky Emission Spectrum', color='blue')
plt.plot(wl_nm, E_filtered, label='Filtered Sky Emission', color='red')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Sky Emission (ph/s/m^2/arcsec^2/nm)')
plt.title('Sky Emission Spectrum Before and After Filter')
plt.legend()
plt.grid()
plt.show()
#%%
# Integrate the filtered sky emission over the wavelength range of interest to get total sky brightness in ph/s/m^2/arcsec^2
from scipy.integrate import simpson
# Integrate over the filered spectrum
E_band = simpson(E_filtered, wl_nm)  # ph/s/m^2/arcsec^2
print(f"Integrated sky brightness over the N-band: {E_band:.3e} ph/s/m^2/arcsec^2")

#%% 
# Now compute the emission of the telescope itself, which is a blackbody at 273K with an emissivity at 0.05 use plank function to compute the blackbody emission at 273K, and then apply the emissivity to get the telescope emission in ph/s/m^2/arcsec^2/nm
from scipy.constants import h, c, k
def planck(wavelength_m, T):
    """Calculate the spectral radiance of a blackbody at temperature T for wavelength in meters."""
    a = 2.0 * h * c**2
    b = h * c / (wavelength_m * k * T)
    return a / (wavelength_m**5 * (np.exp(b) - 1.0))
# Convert wavelength from nm to m for the Planck function
wl_m = wl_nm * 1e-9
# Calculate the blackbody emission at 273K
T_tel = 273  # K
B_tel = planck(wl_m, T_tel)  # W/m^2/sr/m
# Convert from W/m^2/sr/m to ph/s/m^2/sr/nm
# E_tel = B_tel * (wavelength in m) / (h * c)  # ph/s/m^2/sr/nm
E_tel = B_tel * wl_m / (h * c) * 1e-9  # ph/s/m^2/sr/nm
# Apply the emissivity of the telescope
epsilon_tel = 0.05
E_tel_emission = E_tel * epsilon_tel  # ph/s/m^2/sr/nm
# Convert from per steradian to per arcsec^2
arcsec2_to_sr = (np.pi / (180 * 3600))**2
E_tel_emission_per_arcsec2 = E_tel_emission * arcsec2_to_sr  # ph/s/m^2/arcsec^2/nm
# Plot the telescope emission
plt.figure(figsize=(10, 6))
plt.plot(wl_nm, E_tel_emission_per_arcsec2, label='Telescope Emission', color='green')
plt.xlabel('Wavelength [nm]')
plt.yscale('log')
plt.ylabel('Telescope Emission (ph/s/m^2/arcsec^2/nm)')
plt.title('Telescope Emission Spectrum')
plt.legend()
plt.grid()
plt.show()
#%%
# Not a very helpful plot, but apply the filter transmission to the telescope emission as well to see how much of it gets through
E_tel_filtered = E_tel_emission_per_arcsec2 * filter_on_sky  # ph/s/m^2/arcsec^2/nm
plt.figure(figsize=(10, 6))
plt.plot(wl_nm, E_tel_emission_per_arcsec2, label='Telescope Emission', color='green')
plt.plot(wl_nm, E_tel_filtered, label='Filtered Telescope Emission', color='purple')
plt.xlabel('Wavelength [nm]')
plt.xlim(filter_wl_nm[0], filter_wl_nm[-1])  # zoom in on the filter band
plt.yscale('log')
plt.ylabel('Telescope Emission (ph/s/m^2/arcsec^2/nm)')
plt.title('Telescope Emission Before and After Filter')
plt.legend()
plt.grid()
plt.show()
#%%
# Integrate over the filter band to get the total telescope emission in ph/s/m^2/arcsec^2
E_tel_band = simpson(E_tel_filtered, wl_nm)  # ph/s/m^2/arcsec^2
print(f"Integrated telescope emission over the N-band: {E_tel_band:.3e} ph/s/m^2/arcsec^2")











#%%
wl_um /= 1000.0  # if your file wavelength is in nm; double-check!

# Filter edges (given)
lam1, lam2 = 8.5, 13.0

# Top-hat transmission approximation
T = 0.65

# Effective collecting area (geometric for now)
D = 3.2  # m
A_geo = np.pi * (D/2)**2

# Pixel solid angle in arcsec^2/pixel (MIRSI)
pixscale = 0.27  # arcsec/pixel
Omega_pix = pixscale**2

# Integrate sky spectrum over band: ph/s/m^2/arcsec^2
m = (wl_um >= lam1) & (wl_um <= lam2)
E_band = np.trapz(E[m], wl_um[m])

#efficiency of the system 
epsilon = 0.07

# Photons / second / pixel (at detector entrance, with only filter transmission applied)
Ndot_pix = E_band * A_geo * Omega_pix * T * epsilon

print(f"Integrated sky (8.5–13 um): {E_band:.3e} ph/s/m^2/arcsec^2")
print(f"Sky rate per pixel:         {Ndot_pix:.3e} ph/s/pixel")

#%%
t_exp = 0.01  # seconds, example
N_pix = Ndot_pix * t_exp
print(f"Sky photons per pixel per frame: {N_pix:.3e} ph/pixel")
# %%
N_pix
# %%
