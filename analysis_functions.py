#%% 
# Important functions to import 
import numpy as np
import matplotlib.pyplot as plt
import os 
from astropy.io import fits


def read_in_fits(file_directory):
    """
    Sorts by frame number (order matters) and reads in FITS files.
    Appends data into a 3D numpy array (num_frames, height, width).

    """
    # Load the directory 
    files = os.listdir(file_directory)
    
    # Filter for .fits files only
    fits_files = [f for f in files if f.endswith('.fits')]
    
    # Sort by extracting the numeric sequence from filename
    # Example: sky1-00001.Z.00123.fits -> extract 00123
    import re
    def extract_number(filename):
        match = re.search(r'\.Z\.(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    fits_files = sorted(fits_files, key=extract_number)
    
    # print(f"First 5 filenames: {fits_files[:5]}")
    
    # Get shape from first file
    with fits.open(os.path.join(file_directory, fits_files[0])) as hdul:
        array_shape = hdul[0].data.shape
    
    n_exposures = len(fits_files)
    print(f"Processing {n_exposures} FITS files from {file_directory}")
    
    # Initialize the array to hold exposures
    exposures = np.zeros((n_exposures, *array_shape), dtype=float)

    for i, file in enumerate(fits_files):
        with fits.open(os.path.join(file_directory, file)) as hdul:
            data = hdul[0].data
            exposures[i, :, :] = data
    
    # Print a confirmation message
    print(f"Loaded {n_exposures} FITS files from {file_directory}")
    return exposures

def coadd_to_dwells(raw_cube, n, method="sum", trim=None):
    """
    Convert fast-read raw frames into 'dwell' frames by binning n consecutive frames.

    Parameters
    ----------
    raw_cube : (N, ny, nx) ndarray
        Fast-read frames.
    n : int
        Number of fast frames per dwell (software-emulated dwell / coadd).
    method : {"mean","sum"}
        How to combine frames inside each dwell.
    trim : (i0, i1) or None
        Optional slice on the raw frames before binning.

    Returns
    -------
    dwell_cube : (N_dwell, ny, nx) ndarray
        Coadded dwell frames.
    """
    if trim is not None:
        raw_cube = raw_cube[trim[0]:trim[1]]

    raw_cube = np.asarray(raw_cube)
    if raw_cube.ndim != 3:
        raise ValueError("raw_cube must have shape (N, ny, nx)")

    if n < 1:
        raise ValueError("n must be >= 1")

    N, ny, nx = raw_cube.shape
    Ntrim = (N // n) * n
    if Ntrim < n:
        raise ValueError("Not enough frames to form one dwell.")

    blocks = raw_cube[:Ntrim].reshape(-1, n, ny, nx)

    if method == "mean":
        dwell_cube = blocks.mean(axis=1)
    elif method == "sum":
        dwell_cube = blocks.sum(axis=1)
    else:
        raise ValueError("method must be 'mean' or 'sum'")

    return dwell_cube

def chop_demodulate_pairwise(dwell_cube):
    """
    Pairwise demodulation for ABAB... sequence:
        diff[m] = A[m] - B[m]  with A=even dwells, B=odd dwells

    Parameters
    ----------
    dwell_cube : (N_dwell, ny, nx) ndarray

    Returns
    -------
    diff_cube : (N_pairs, ny, nx) ndarray
        One chopped difference image per AB pair.
    """
    dwell_cube = np.asarray(dwell_cube)
    if dwell_cube.ndim != 3:
        raise ValueError("dwell_cube must have shape (N_dwell, ny, nx)")

    N, ny, nx = dwell_cube.shape
    n_pairs = N // 2
    if n_pairs < 1:
        raise ValueError("Need at least 2 dwells (A and B) to form one pair.")

    d2 = dwell_cube[:2 * n_pairs]
    A = d2[0::2]
    B = d2[1::2]
    return A - B

def dwell_and_chop_frequencies(frame_dt_s, n):
    """
    frame_dt_s: cadence of fast-read frames
    n: frames per dwell

    Returns:
      t_dwell, f_state, f_chop
    where:
      f_state = 1/t_dwell  (A->B state switching rate)
      f_chop  = 1/(2*t_dwell) (full AB cycle rate)
    """
    t_dwell = frame_dt_s * n
    f_state = 1.0 / t_dwell
    f_chop = 1.0 / (2.0 * t_dwell)
    return t_dwell, f_state, f_chop

def extract_channels(exposures, channel_width=20, trim=None, y_range=None): 
    """
    Extract the individual channel images as separate cubes, without summarizing into a timeseries.
    """
    if trim is not None:
        exposures = exposures[trim[0]:trim[1]]

    N, H, W = exposures.shape
    y0, y1 = (0, H) if y_range is None else y_range

    strip = exposures[:, y0:y1, :]          # (N, H', W)
    n_channels = W // channel_width
    channels = np.empty((n_channels, N, y1 - y0, channel_width), dtype=exposures.dtype)
    channel_x_centers = np.arange(n_channels) * channel_width + (channel_width - 1) / 2.0

    for ch in range(n_channels):
        x_lo = ch * channel_width
        x_hi = x_lo + channel_width
        channels[ch] = strip[:, :, x_lo:x_hi]      # (N, H', channel_width)

    return {
        "channels": channels,
        "channel_x_centers": channel_x_centers,
        "n_channels": n_channels,
        "channel_width": channel_width,
    }

def timeseries_channels(channels, n, method="sum", statistic="sum"):
    """
    For each channel cube, apply coadd_to_dwells and chop_demodulate_pairwise, then summarize into a timeseries.
    """
    n_channels, N_frames, H_channel, channel_width = channels.shape
    timeseries = np.empty((N_frames // (2 * n), n_channels), dtype=float)

    for ch in range(n_channels):
        dwell_cube = coadd_to_dwells(channels[ch], n=n, method=method)
        diff_cube = chop_demodulate_pairwise(dwell_cube)

        if statistic == "mean":
            timeseries[:, ch] = diff_cube.mean(axis=(1, 2))
        elif statistic == "median":
            timeseries[:, ch] = np.median(diff_cube.reshape(diff_cube.shape[0], -1), axis=1)
        elif statistic == "sum":
            timeseries[:, ch] = diff_cube.sum(axis=(1, 2))
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'")

    return timeseries

def create_variance_curve(channels, coadds_array, frame_dt_s): 
    """
    Create a curve of variance vs coadd factor for each channel.
    """
    timeseries_by_coadd = {}

    t_dwell_by_coadd = []
    f_state_by_coadd = []
    f_chop_by_coadd = []

    for n in coadds_array:
        timeseries_by_coadd[n] = timeseries_channels(channels, n=n, method="sum", statistic="sum")
        t_dwell, f_state, f_chop = dwell_and_chop_frequencies(frame_dt_s, n)
        t_dwell_by_coadd.append(t_dwell)
        f_state_by_coadd.append(f_state)
        f_chop_by_coadd.append(f_chop)

    variances_by_coadd = {n: np.var(timeseries_by_coadd[n], axis=0) for n in coadds_array}

    return variances_by_coadd, t_dwell_by_coadd, f_state_by_coadd, f_chop_by_coadd

