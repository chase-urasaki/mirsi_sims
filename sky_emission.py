#%%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

wl_nm, E = np.genfromtxt('mk_skybg_nq_10_10_ph.dat', unpack=True)
# Data has units of ph/sec/as^2/nm/m^2, and wavelength in nm. Convert to microns and check units carefully!

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
# Convert filter wl from angstroms to nm 
filter_wl_nm = filter_wl / 10.0  # angstroms to nm
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
