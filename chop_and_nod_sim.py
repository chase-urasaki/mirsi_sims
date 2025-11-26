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
                        throughput=0.3,
                        nodding=False, throw=50):
    """
    Injects a point source into every chopped frame.
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
    position : tuple
        (y, x) pixel coordinates of the point source.
    mag : float
        N-band magnitude of the point source.
    extent : float
        Full extent (FWHM) of the point source in pixels.
    exposure_time : float
        Exposure time per frame in seconds.
    QE : float
        Quantum efficiency of the detector. Default is 0.5. Set by the sky background calculation.
    throughput : float
        Total system throughput. Default is 0.3. Set by the sky background calculation.
    nodding : bool
        If True, adds nod position offset for every other frame. Default is False.
    throw : int
        Pixel offset for nod position. Default is 50 pixels.
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

def inject_point_source_chop_nod(
    exposure_sequence,
    position,
    mag,
    extent,
    exposure_time=0.02,
    QE=0.5,
    throughput=0.3,
    chop_throw=20,
    nod_throw=40,
):
    """
    Inject a point source with a true chop–nod pattern into the exposure sequence.

    Pattern (repeats every 4 frames):
      i % 4 = 0 : nod - nod_throw, chop - chop_throw
      i % 4 = 1 : nod - nod_throw, chop + chop_throw
      i % 4 = 2 : nod + nod_throw, chop - chop_throw
      i % 4 = 3 : nod + nod_throw, chop + chop_throw
    """
    n_exp, H, W = exposure_sequence.shape
    y0, x0 = position

    # Total electrons per exposure from the source
    photons_per_s = photon_rate_from_mag(mag, throughput=throughput)
    photons = photons_per_s * exposure_time
    electrons = photons * QE

    # Print number of electrons for debugging
    print(f"Injecting point source of mag {mag} at position {position}:")
    print(f"  Photons per second: {photons_per_s:.2e}")
    print(f"  Electrons per exposure: {electrons:.2e}")

    # Base PSF (centered at (y0, x0))
    yy, xx = np.indices((H, W))
    rr2 = (yy - y0)**2 + (xx - x0)**2
    sigma = extent / 2.355  # FWHM ≈ extent
    psf_base = np.exp(-0.5 * rr2 / sigma**2)
    psf_base /= psf_base.sum()

    for i in range(n_exp):
        phase = i % 4

        # nod sign: -1 for first 2 frames, +1 for next 2
        nod_sign = -1 if phase < 2 else +1
        # chop sign: -1 for even frames, +1 for odd frames
        chop_sign = -1 if (phase % 2 == 0) else +1

        dy = nod_sign * nod_throw
        dx = chop_sign * chop_throw

        # Shift PSF without wrap (zero-padding instead of np.roll wrap)
        psf = np.zeros_like(psf_base)
        y_start_src = max(0, -dy)
        y_end_src   = min(H, H - dy)
        x_start_src = max(0, -dx)
        x_end_src   = min(W, W - dx)

        y_start_dst = max(0, dy)
        y_end_dst   = min(H, H + dy)
        x_start_dst = max(0, dx)
        x_end_dst   = min(W, W + dx)

        psf[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
            psf_base[y_start_src:y_end_src, x_start_src:x_end_src]

        # Inject into frame i
        exposure_sequence[i] += electrons * psf

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
    expsoure_sequence = make_exposure_sequence(4, 0.02, self_similar=False)

    # Sequence with correlated drift
    #test_sequence = make_exposure_sequence(6, 0.02, drift={"tau": 0.5, "amp_frac": 0.03}, self_similar=True)

    # Imshow exposure sequence 
    for i in range(expsoure_sequence.shape[0]):
        plt.imshow(expsoure_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure {i}')
        plt.colorbar(label='Electrons')
        plt.show()

#%%
    # Inject point source into exposure sequence
    expsoure_sequence = inject_point_source_chop_nod(
        expsoure_sequence,
        position=(120, 160),
        mag=2.0,
        extent=3.0,
        exposure_time=0.02,
        QE=0.5,
        throughput=0.3,
        chop_throw=20,
        nod_throw=40,
    )
#%%
    # Show frames with point source
    for i in range(expsoure_sequence.shape[0]):
        plt.imshow(expsoure_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure with Point Source {i}')
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
    # 