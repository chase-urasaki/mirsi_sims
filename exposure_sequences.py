#%% 
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

#%%
from sky_sim import SkyBackgroundExposure
#from sky_sim import SkyBackgroundExposure_with_aperture
# %%
# Generate a sequence of exposures with aperture
from sky_sim import compute_sky_background_rate

#%%
# def make_exposure_sequence(n_exposures, exposure_time, drift = None, self_similar=False): 
#     """
#     Creates a sequence of sky background limited exposures using an aperture.
#     Stores results in 3D array (n_exposures, height, width).
    
#     Parameters:
#     -----------
#     n_exposures : int
#         Number of exposures to simulate.
#     exposure_time : float
#         Exposure time in seconds for each exposure.
#     self_similar : bool
#         If True, generates the 
#     """
    
#     # Detector parameters
#     read_noise = 800 #e- / read rms
#     dark_current = 50 # e- / s / pixel
#     array_shape = (240, 320)

#     # Initialize the array to hold exposures
#     exposures = np.zeros((n_exposures, *array_shape))
#     if self_similar:
#         # Compute sky background rate 
#         sky_background_rate = compute_sky_background_rate(10e-6, 273)

#         # Initialize the SkyBackgroundExposure_with_aperture simulator
#         simulator = SkyBackgroundExposure(sky_background_rate, read_noise, dark_current)

#         # Initialize array to hold exposures
#         exposures = np.zeros((n_exposures, *array_shape))

#         for i in range(n_exposures):
#             frame = simulator.simulate_exposure(exposure_time)
            
#             exposures[i] = frame

#         return exposures
    
#     else:
#         for i in range(n_exposures):
#             simulator = SkyBackgroundExposure(compute_sky_background_rate(10e-6, 273), read_noise, dark_current)
#             frame = simulator.simulate_exposure(exposure_time)
#             exposures[i] = frame

#         return exposures
    
#%% 
# test for adding in drift 
import numpy as np

def generate_slow_drift(n_exposures, exposure_time, tau, amp_frac, rng=None):
    """
    Generate a slow, correlated fractional drift term for the sky background
    using an Ornstein-Uhlenbeck (OU) process / AR(1) model.

    This simulates slow atmospheric variations that cause the sky background
    to drift coherently across multiple exposures. The drift follows an
    exponentially-correlated Gaussian process.

    Parameters
    ----------
    n_exposures : int
        Number of frames in the sequence. Must be >= 2.
    exposure_time : float
        Cadence between frames [s] (assuming back-to-back exposures).
        Must be positive.
    tau : float
        Correlation timescale of the drift [s]. Longer tau means slower,
        more persistent drift. Must be positive.
    amp_frac : float
        RMS amplitude of the fractional drift (e.g., 0.03 for ~3% variation).
        Must be non-negative.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new
        generator is created.

    Returns
    -------
    drift_frac : ndarray, shape (n_exposures,)
        Zero-mean fractional drift with RMS amplitude of ~amp_frac.
        Apply to sky background as: sky_rate * (1 + drift_frac[i])

    Notes
    -----
    The OU process is defined by:
        x[i] = rho * x[i-1] + sqrt(1 - rho^2) * eps[i]
    where rho = exp(-dt/tau) and eps ~ N(0,1).
    
    This ensures the stationary variance is 1, which is then scaled by amp_frac.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> drift = generate_slow_drift(100, 0.05, tau=10.0, amp_frac=0.03, rng=rng)
    >>> print(f"RMS drift: {np.std(drift):.4f}")  # Should be ~0.03
    """
    # Input validation
    if n_exposures < 2:
        raise ValueError(f"n_exposures must be >= 2, got {n_exposures}")
    if exposure_time <= 0:
        raise ValueError(f"exposure_time must be positive, got {exposure_time}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if amp_frac < 0:
        raise ValueError(f"amp_frac must be non-negative, got {amp_frac}")
    
    if rng is None:
        rng = np.random.default_rng()

    # Correlation coefficient between adjacent frames
    dt = exposure_time
    rho = np.exp(-dt / tau)
    
    # Handle edge case: if dt >> tau, correlation is essentially zero
    if rho < 1e-10:
        # Pure white noise case (no correlation)
        return amp_frac * rng.normal(size=n_exposures)
    
    # Generate OU/AR(1) process
    x = np.empty(n_exposures, dtype=float)
    x[0] = rng.normal()

    # Stationary variance factor ensures long-term variance = 1
    noise_scale = np.sqrt(1 - rho**2)
    
    for i in range(1, n_exposures):
        eps = rng.normal()
        x[i] = rho * x[i-1] + noise_scale * eps

    # Normalize to unit variance (handles finite sample effects)
    # For very short sequences, std could be zero
    x_std = np.std(x, ddof=1)
    if x_std > 1e-10:
        x = (x - np.mean(x)) / x_std  # Also ensure zero mean
    
    # Scale to desired amplitude
    drift_frac = amp_frac * x
    
    return drift_frac


def make_exposure_sequence(n_exposures, exposure_time, drift=None, self_similar=False): 
    """
    Creates a sequence of sky background limited exposures using an aperture.
    Stores results in 3D array (n_exposures, height, width).
    
    Parameters
    ----------
    n_exposures : int
        Number of exposures to simulate.
    exposure_time : float
        Exposure time in seconds for each exposure.
    drift : None, dict, or array-like
        If None, no slow drift is applied (pure Poisson + detector noise).
        If dict, interpreted as parameters for an OU/AR(1) drift process:
            {
                "tau": <correlation timescale in seconds>,
                "amp_frac": <RMS fractional amplitude, e.g. 0.03>,
                "rng": np.random.Generator (optional)
            }
        If array-like of shape (n_exposures,), it is taken as the
        fractional drift series directly (zero-mean, small amplitude),
        and used as a multiplicative factor: (1 + drift[i]).
    self_similar : bool
        If True, uses a single SkyBackgroundExposure instance and updates its
        sky background rate each frame (preserving any internal structure).
    """
    
    # Detector parameters
    read_noise = 800  # e- / read rms
    dark_current = 50 # e- / s / pixel
    array_shape = (240, 320)

    # Initialize the array to hold exposures
    exposures = np.zeros((n_exposures, *array_shape), dtype=float)

    # --- Base sky background rate (no drift yet) ---
    base_sky_background_rate = compute_sky_background_rate(10e-6, 273)

    # --- Build drift time series if requested ---
    if drift is None:
        drift_series = np.zeros(n_exposures, dtype=float)
    elif isinstance(drift, dict):
        tau = drift["tau"]
        amp_frac = drift["amp_frac"]
        rng = drift.get("rng", None)
        drift_series = generate_slow_drift(
            n_exposures=n_exposures,
            exposure_time=exposure_time,
            tau=tau,
            amp_frac=amp_frac,
            rng=rng
        )
    else:
        # Treat as user-supplied array-like with length n_exposures
        drift_series = np.asarray(drift, dtype=float)
        if drift_series.shape[0] != n_exposures:
            raise ValueError("drift array must have length n_exposures")

    if self_similar:
        # One simulator; we update its sky background rate each frame.
        simulator = SkyBackgroundExposure(
            base_sky_background_rate,
            read_noise,
            dark_current
        )

        for i in range(n_exposures):
            # Apply slow fractional drift to *mean* sky background rate
            sky_background_rate_i = base_sky_background_rate * (1.0 + drift_series[i])
            # Assuming simulator exposes this as an attribute:
            simulator.sky_background_rate = sky_background_rate_i

            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

    else:
        # New simulator each frame, each with its own drifted background level.
        for i in range(n_exposures):
            sky_background_rate_i = base_sky_background_rate * (1.0 + drift_series[i])

            simulator = SkyBackgroundExposure(
                sky_background_rate_i,
                read_noise,
                dark_current
            )
            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

    return exposures

#%%
# ============================================================================
# Test/Demo code - only runs when this file is executed directly
# ============================================================================
if __name__ == "__main__":
    #%%
    test_sequence = make_exposure_sequence(6, 0.02, drift={"tau": 0.5, "amp_frac": 0.03}, self_similar=True)
    # %%
    for i in range(test_sequence.shape[0]):
        plt.imshow(test_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure {i+1}')
        plt.colorbar(label='Electrons')
        plt.show()
    # %%
def coadd_exposures(exposure_sequence, coadds, mode = 'average'):
    """ 
    This function takes in an exposure sequence and coadds 
    then number of frames by averaging them together.
    
    Parameters:
    -----------
    expsoure_seuqence : np.ndarray
        3D array of exposures (n_exposures, height, width)
    coadds : int
        Number of frames to average together
        
    Returns:
    --------
    coadded_exposures : np.ndarray
        3D array with coadded frames (n_exposures//coadds, height, width)
    """  
    # Initialize array 
    coadded_exposures = np.zeros((len(exposure_sequence)//coadds, exposure_sequence.shape[1], exposure_sequence.shape[2]))
    # Take every cadded frame and average them. e.g. if coadd is two then coadd pairs of image . 
    if mode == 'average':
        for i in range(0, len(exposure_sequence), coadds):
            coadded_frame = np.mean(exposure_sequence[i:i+coadds], axis=0)
            coadded_exposures[i//coadds] = coadded_frame
        # Return a new array with the coadded frames.
    elif mode == 'sum':
        for i in range(0, len(exposure_sequence), coadds):
            coadded_frame = np.sum(exposure_sequence[i:i+coadds], axis=0)
            coadded_exposures[i//coadds] = coadded_frame

    else: 
        raise ValueError("Mode must be either 'average' or 'sum'")
    return coadded_exposures

def find_statistics(exposure_sequence):
    """"
    Function takes an exposure sequence (coadded or not), and computes the mean and variance of each frame.
    """
    means = []
    variances = []
    for exposure in exposure_sequence:
        means.append(np.mean(exposure))
        variances.append(np.var(exposure))
    return np.array(means), np.array(variances)

#%%
# ============================================================================
# More test/demo code
# ============================================================================
if __name__ == "__main__":
    #%%
    rng = np.random.default_rng(123)

    exposures = make_exposure_sequence(
        n_exposures=10,
        exposure_time=0.02,  # 50 ms
        #drift={"tau": 100, "amp_frac": 0.001, "rng": rng},
        drift=None,
        self_similar=False
    )
    #%%
    plt.imshow(exposures[0], cmap='gray', origin='lower')
    plt.colorbar(label='Electrons')
    #%%
    # show all the exposures
    for i in range(exposures.shape[0]):
        plt.imshow(exposures[i], cmap='gray', origin='lower')
        plt.title(f'Exposure {i+1}')
        plt.colorbar(label='Electrons')
        plt.show()
    #%%
    # Plot the mean and variance of each 
    for i, exposure in enumerate(exposures):
        plt.hist(exposure.flatten(), bins=50, alpha=0.5, label=f'Exposure {i+1}')
        plt.legend()

    # #%%
    coadded_sequence = coadd_expsoures(exposures, 2)
    #%% 
    # Show coadded frames 
    for i in range(coadded_sequence.shape[0]):
        plt.imshow(coadded_sequence[i], cmap='gray', origin='lower', label= f'exposure {i+1}')
        plt.title(f'Coadded Exposure {i+1}')
        plt.legend()
        plt.colorbar(label='Electrons')
        plt.show()

    #%%
    # Plot the mean and variance of each coadded frame
    for i, exposure in enumerate(coadded_sequence):
        plt.hist(exposure.flatten(), bins=50, alpha=0.5, label=f'Coadded Exposure {i+1}')
        plt.legend()
    # %%