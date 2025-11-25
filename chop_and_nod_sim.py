#%%
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
#%%
# Inject a point source into an exposure sequence

def mag_to_flux_nband(mag, F0=36.0):
    """Convert N-band magnitude → flux density in Jy."""
    return F0 * 10**(-mag / 2.5)


def jy_to_photon_rate(Fnu_jy, throughput=0.2):
    """
    Convert flux density (Jy) → photon rate (photons/sec) at detector.
    Assumes N-band and MIRSI parameters.
    """
    # Constants
    c = 3e8
    h = 6.626e-34
    lam = 10e-6               # 10 µm effective
    nu = c / lam

    # Telescope effective area (MIRSI cold stop → D=3.0 m)
    A_tel = np.pi * (1.5)**2  # m^2

    # N-band bandwidth: Δλ ~ 5 µm → Δν = c/λ^2 * Δλ
    delta_lam = 5e-6
    delta_nu = c * delta_lam / (lam**2)

    # Convert Jy → W/m²/Hz
    Fnu = Fnu_jy * 1e-26

    # Photon rate
    photons_per_s = (Fnu * A_tel * delta_nu * throughput) / (h * nu)

    return photons_per_s

def photon_rate_from_mag(mag, throughput=0.2):
    """Convenience wrapper: N-mag → photon/sec"""
    Fnu = mag_to_flux_nband(mag)
    return jy_to_photon_rate(Fnu, throughput=throughput)

def inject_point_source(exposure_sequence, position, mag, extent,
                        exposure_time=0.05,
                        QE=0.5,
                        throughput=0.2,
                        nodding=False, throw=50):
    """
    Injects a point source into every chopped frame.
    """
    n_exp, H, W = exposure_sequence.shape
    y0, x0 = position

    # Photons per second from magnitude
    photons_per_s = photon_rate_from_mag(mag, throughput=throughput)

    # Photons per exposure
    photons = photons_per_s * exposure_time

    # Electrons generated
    electrons = photons * QE

    # Build a Gaussian PSF
    yy, xx = np.indices((H, W))
    rr2 = (yy - y0)**2 + (xx - x0)**2
    sigma = extent / 2.355
    psf = np.exp(-0.5 * rr2 / sigma**2)
    psf /= psf.sum()  # normalize flux

    for i in range(n_exp):
        exposure_sequence[i] += electrons * psf

        if nodding and (i % 2 == 1):
            # nod position offset
            exposure_sequence[i] += electrons * np.roll(psf, throw, axis=0)

    return exposure_sequence


 #%%

def subtract_frames(exposure_sequence): 
    """
    Subtracts frames pairwise: (frame_0 - frame_1), (frame_2 - frame_3), (frame_4 - frame_5), etc.
    This is used for chop subtraction where consecutive pairs represent chopped positions A and B.
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
        
    Returns:
    --------
    np.ndarray
        3D array of shape (n_exposures//2, height, width) containing the subtracted frame pairs.
    """
    # Make a new array to hold the subtracted frames
    n_exposures, height, width = exposure_sequence.shape
    subtracted_sequence = np.zeros((n_exposures//2, height, width))

    # Subtract frames pairwise: (0-1), (2-3), (4-5), etc.
    for pair_idx in range(n_exposures // 2):
        frame_a = pair_idx * 2
        frame_b = pair_idx * 2 + 1
        subtracted_sequence[pair_idx] = exposure_sequence[frame_a] - exposure_sequence[frame_b]

    return subtracted_sequence


# Test the frame subtraction
#%%
# Add if name == main block to prevent auto-execution on import
if __name__ == "__main__":
    """ 
    For testing purposes
    """
    # Make exposure sequence for testing 
    from exposure_sequences import make_exposure_sequence
    #%%
    expsoure_sequence = make_exposure_sequence(6, 0.02, self_similar=False)

    # Sequence with correlated drift
    #test_sequence = make_exposure_sequence(6, 0.02, drift={"tau": 0.5, "amp_frac": 0.03}, self_similar=True)

    # Imshow exposure sequence 
    for i in range(expsoure_sequence.shape[0]):
        plt.imshow(expsoure_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure {i}')
        plt.colorbar(label='Electrons')
        plt.show()

#%%
    # Subtract frames
    subtracted_sequence = subtract_frames(expsoure_sequence)
# %%
    # Show subtracted frames
    for i in range(subtracted_sequence.shape[0]):
        plt.imshow(subtracted_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Subtracted Frame {i+1} - Frame {i}')
        plt.colorbar(label='Electrons')
        plt.show()

# %%
    # Show the statistics of the subtracted frames
    for i in range(subtracted_sequence.shape[0]):
        mean = np.mean(subtracted_sequence[i])
        std = np.std(subtracted_sequence[i])
        print(f'Subtracted Frame {i}: Mean = {mean:.2f}, Std = {std:.2f}')
        plt.hist(subtracted_sequence[i].flatten(), bins=50, alpha=0.7)
        plt.show()

# %%
