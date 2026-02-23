#%%
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry 
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
    shot_noise = np.random.poisson(photons) 
    total_photons = photons + shot_noise
    electrons = total_photons * QE

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

        # Don't forget shot noise from the source

    return exposure_sequence
 #%%

def subtract_frames(exposure_sequence): 
    """
    Subtracts frames pairwise to generate both A-B and B-A differences.
    For n input frames, generates n difference images:
    - Pairs (0-1), (2-3), (4-5), etc. → A-B differences at even indices
    - Pairs (1-0), (3-2), (5-4), etc. → B-A differences at odd indices
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
        
    Returns:
    --------
    np.ndarray
        3D array of shape (n_exposures, height, width) containing both difference directions.
        Even indices contain A-B, odd indices contain B-A.
    """
    # Make a new array to hold the subtracted frames
    n_exposures, height, width = exposure_sequence.shape
    subtracted_sequence = np.zeros((n_exposures, height, width))

    # Subtract frames pairwise: (0-1), (2-3), (4-5), etc.
    for pair_idx in range(n_exposures // 2):
        frame_a_idx = pair_idx * 2
        frame_b_idx = pair_idx * 2 + 1
        
        # A - B at even output indices (0, 2, 4, ...)
        subtracted_sequence[pair_idx * 2] = exposure_sequence[frame_a_idx] - exposure_sequence[frame_b_idx]
        
        # B - A at odd output indices (1, 3, 5, ...)
        subtracted_sequence[pair_idx * 2 + 1] = exposure_sequence[frame_b_idx] - exposure_sequence[frame_a_idx]

    return subtracted_sequence

def variance_vs_frequency_chop_proxy_true(x, coadds_list, frame_dt_s, method="mean", ddof=1):
    out = {"coadds": [], "f_eff_hz": [], "n_blocks": [], "n_pairs": [], "var_diff": []}

    for n in coadds_list:
        Ntrim = (len(x) // n) * n
        if Ntrim < 4*n:   # need at least 2 A-B pairs
            continue

        if method == "mean":
            xb = x[:Ntrim].reshape(-1, n).mean(axis=1)
        elif method == "sum":
            xb = x[:Ntrim].reshape(-1, n).sum(axis=1)
        else:
            raise ValueError("method must be 'mean' or 'sum'")

        n_pairs = len(xb) // 2
        xb2 = xb[:2*n_pairs]
        d = xb2[0::2] - xb2[1::2]   # A - B

        out["coadds"].append(n)
        out["f_eff_hz"].append(1.0 / (frame_dt_s * n))
        out["n_blocks"].append(len(xb))
        out["n_pairs"].append(n_pairs)
        out["var_diff"].append(np.var(d, ddof=ddof))

    return {k: np.array(v) for k, v in out.items()}


def extract_sky_timeseries(exposures, region, trim=None, annulus=None, statistic="mean"):
    """
    Build a 1D time series x[t] = sky level estimator from a fixed aperture (and optional annulus).

    Parameters
    ----------
    exposures : (n, H, W) ndarray
    region : (y, x, r)
        aperture center and radius in pixels
    trim : (i0, i1) or None
        slice frames: exposures[i0:i1]
    annulus : (r_in, r_out) or None
        if provided, compute annulus background and subtract it from aperture mean/sum
    statistic : {'mean','median','sum'}
        how to summarize pixels inside the aperture

    Returns
    -------
    out : dict with keys:
        'x_ap' : aperture statistic time series (background-subtracted if annulus given)
        'x_ann' : annulus statistic time series (if annulus given)
        'n_pix_ap', 'n_pix_ann'
    """
    if trim is not None:
        exposures = exposures[trim[0]:trim[1]]

    y, x, r = region
    pos = (x, y)

    ap = CircularAperture(pos, r=r)#
    ap_mask = ap.to_mask(method="center")
    # Fix: to_mask() returns a single mask, not a list - remove [0]
    ap_mask = ap_mask.to_image(exposures.shape[1:]).astype(bool)
    n_pix_ap = ap_mask.sum()

    if annulus is not None:
        r_in, r_out = annulus
        an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
        an_mask = an.to_mask(method="center")
        # Fix: same here - remove [0]
        an_mask = an_mask.to_image(exposures.shape[1:]).astype(bool)
        n_pix_ann = an_mask.sum()
    else:
        an_mask = None
        n_pix_ann = 0

    x_ap = np.empty(exposures.shape[0], dtype=float)
    x_ann = np.empty(exposures.shape[0], dtype=float) if an_mask is not None else None

    for i, frame in enumerate(exposures):
        ap_vals = frame[ap_mask]

        if statistic == "mean":
            ap_stat = np.mean(ap_vals)
        elif statistic == "median":
            ap_stat = np.median(ap_vals)
        elif statistic == "sum":
            ap_stat = np.sum(ap_vals)
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'")

        if an_mask is not None:
            ann_vals = frame[an_mask]
            # robust background: median is usually best
            ann_stat = np.median(ann_vals)
            x_ann[i] = ann_stat

            # subtract background in consistent units:
            # if 'sum', subtract (background * n_pix_ap); if mean/median, subtract background directly
            if statistic == "sum":
                ap_stat = ap_stat - ann_stat * n_pix_ap
            else:
                ap_stat = ap_stat - ann_stat

        x_ap[i] = ap_stat

    out = {"x_ap": x_ap, "n_pix_ap": n_pix_ap}
    if x_ann is not None:
        out.update({"x_ann": x_ann, "n_pix_ann": n_pix_ann})
    return out

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
    #%%
    exposure_sequence = make_exposure_sequence(2000, 0.02, self_similar=False)

    # Define the apertures for photometry
    region = (120, 160, 3)  # (y, x, r) for 
    test_sky = extract_sky_timeseries(exposure_sequence, region, statistic="sum")

    # Plot as a timeseries 
    plt.figure(figsize=(10, 4))
    plt.plot(test_sky["x_ap"], marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Aperture Sum')
    plt.title('Sky Time Series')
    plt.grid(True)
    plt.show()
    #%%
    # run the chop proxy 
    coadds_list = np.arange(5, 251)

    test_sky_chopped = variance_vs_frequency_chop_proxy_true(test_sky["x_ap"], coadds_list, frame_dt_s=0.02, method="sum", ddof=1)
    #%%
    plt.figure(figsize=(8, 5))
    plt.plot(test_sky_chopped["f_eff_hz"], test_sky_chopped["var_diff"], marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Effective Frequency (Hz)')
    plt.ylabel('Variance of A-B Differences')
    plt.title('Variance vs Effective Frequency for Chopped Sky')
    plt.grid(True, which="both", ls="--")
    plt.show()
    #%%
    #Inject a point source and test the chop sequence and return the 
    #%%

    #%%
    # Sequence with correlated drift
    #test_sequence = make_exposure_sequence(6, 0.02, drift={"tau": 0.5, "amp_frac": 0.03}, self_similar=True)

    # # Imshow exposure sequence 
    # for i in range(exposure_sequence.shape[0]):
    #     plt.imshow(exposure_sequence[i], cmap='gray', origin='lower')
    #     plt.title(f'Exposure {i}')
    #     plt.colorbar(label='Electrons')
    #     plt.show()

    #%%
    # Inject point source into exposure sequence
    exposure_sequence = inject_point_source_chop_nod(
        exposure_sequence,
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
    for i in range(exposure_sequence.shape[0]):
        plt.imshow(exposure_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure with Point Source {i}')
        plt.colorbar(label='Electrons')
        plt.show()
#%%
    # Subtract frames
    subtracted_sequence = subtract_frames(exposure_sequence)


#%% 
    # Draw three paertures, one off-source, one on-source and an annulus around the on-source
    fig, ax = plt.subplots()
    ax.imshow(subtracted_sequence[0], cmap='gray', origin='lower')
    
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
# %%
