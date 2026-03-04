#%%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

wl_um, E = np.genfromtxt('mk_skybg_nq_10_10_ph.dat', unpack=True)
wl_um /= 1000.0  # if your file wavelength is in nm; double-check!

# Filter edges (given)
lam1, lam2 = 8.5, 13.0

# Top-hat transmission approximation
T = 0.65

# Effective collecting area (geometric for now)
D = 3.0  # m
A_geo = np.pi * (D/2)**2

# Pixel solid angle in arcsec^2/pixel (MIRSI)
pixscale = 0.27  # arcsec/pixel
Omega_pix = pixscale**2

# Integrate sky spectrum over band: ph/s/m^2/arcsec^2
m = (wl_um >= lam1) & (wl_um <= lam2)
E_band = np.trapz(E[m], wl_um[m])

# Photons / second / pixel (at detector entrance, with only filter transmission applied)
Ndot_pix = E_band * A_geo * Omega_pix * T

print(f"Integrated sky (8.5–13 um): {E_band:.3e} ph/s/m^2/arcsec^2")
print(f"Sky rate per pixel:         {Ndot_pix:.3e} ph/s/pixel")

#%%
t_exp = 0.01  # seconds, example
N_pix = Ndot_pix * t_exp
print(f"Sky photons per pixel per frame: {N_pix:.3e} ph/pixel")
# %%
N_pix
# %%
