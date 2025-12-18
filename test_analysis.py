#%% 
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry 

    
    # Sort by extracting the numeric sequence from filename
    # Example: sky1-00001.Z.fits -> extract 00001
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

def preliminary_analysis(exposures, region=None, trim = None ):
    """
    Perform preliminary analysis on the exposures.

    For example, compute the mean and standard deviation across frames.
    Optional region parameter to focus on a specific area of the images.
    Optional trim parameter to take specific frame numbers 
    """
    
    # Do aperture photometry if region is specified
    if region is not None:
        y, x, r = region  # unpack region parameters
        positions = [(x, y)]
        apertures = CircularAperture(positions, r=r)
        
        aperture_sum = []
        aperture_mean = []
        aperture_var = []
        
        if trim is not None:
            exposures = exposures[trim[0]:trim[1]]

        for frame in exposures:
            # Get the aperture photometry sum
            phot_table = aperture_photometry(frame, apertures)
            aperture_sum.append(phot_table['aperture_sum'][0])
            
            # Create a mask for pixels within the aperture
            mask = apertures.to_mask(method='center')[0]
            aperture_data = mask.multiply(frame)
            
            # Get only the pixels within the aperture (non-zero values)
            aperture_pixels = aperture_data[aperture_data > 0]
            
            # Calculate mean and variance
            aperture_mean.append(np.mean(aperture_pixels))
            aperture_var.append(np.var(aperture_pixels))
        
        # Convert to numpy arrays
        photometry_results = {
            'aperture_sum': np.array(aperture_sum),
            'aperture_mean': np.array(aperture_mean),
            'aperture_var': np.array(aperture_var)
        }
        
        return photometry_results
        
# %%
if __name__ == "__main__":

    # From inspection only use between frame 200 and 400
    TRIM_FRAMES = (200, 400)
    zenith1_dir = "./data/251204/zenith1_fits"  
    zenith1_exp = read_in_fits(zenith1_dir)

    region = (30,80,15)  # Example region (y, x, radius)

    zenith1_phot = preliminary_analysis(zenith1_exp, region=region, trim=TRIM_FRAMES)
    

    # Make it for the second set of sky frames 
    zenith2_dir = "./data/251204/zenith2_fits"  
    zenith2_exp = read_in_fits(zenith2_dir)
    zenith2_phot = preliminary_analysis(zenith2_exp, region=region  , trim=TRIM_FRAMES)

    zenith3_dir = "./data/251204/zenith3_fits"  
    zenith3_exp = read_in_fits(zenith3_dir)
    zenith3_phot = preliminary_analysis(zenith3_exp, region=region, trim=TRIM_FRAMES)
#%%
    # Pickle the photometry results to push to git
    import pickle
    with open('zenith_photometry.pkl', 'wb') as f:
        pickle.dump({
            'zenith1_phot': zenith1_phot,
            'zenith2_phot': zenith2_phot,
            'zenith3_phot': zenith3_phot
        }, f)

#%%
    # Plot the region and the photometry results
    apertures = CircularAperture((region[1], region[0]), r=region[2])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(zenith1_exp[0], cmap='gray', origin='lower')
    apertures.plot(color='red', lw=1)
    plt.title('Aperture on First Frame')
    plt.subplot(1, 2, 2)
    plt.plot(zenith1_phot['aperture_sum'], marker='o')
    plt.plot(zenith2_phot['aperture_sum'], marker='x')
    plt.plot(zenith3_phot['aperture_sum'], marker='s')
    plt.title('Aperture Photometry Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Sum')
    plt.ylim(5.64e6, 5.665e6)
    plt.legend(['Zenith 1', 'Zenith 2', 'Zenith 3'])
    plt.tight_layout()
    plt.show()

    #%%
    # Plot the statistics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(zenith1_phot['aperture_mean'], marker='o')
    plt.plot(zenith2_phot['aperture_mean'], marker='x')
    plt.plot(zenith3_phot['aperture_mean'], marker='s')
    plt.title('Aperture Mean Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Mean')

    plt.ylim(0.95*np.min(zenith1_phot['aperture_mean']), 1.05*np.max(zenith1_phot['aperture_mean']))
    plt.legend(['Zenith 1', 'Zenith 2', 'Zenith 3'])
    plt.subplot(1, 2, 2)
    plt.plot(zenith1_phot['aperture_var'], marker='o')
    plt.plot(zenith2_phot['aperture_var'], marker='x')
    plt.plot(zenith3_phot['aperture_var'], marker='s')
    plt.axhline(y = np.median(zenith1_phot['aperture_var']), color='r', linestyle='--', label='Median Zenith 1 Var')
    plt.axhline(y = np.median(zenith2_phot['aperture_var']), color='g', linestyle='--', label='Median Zenith 2 Var')
    plt.axhline(y = np.median(zenith3_phot['aperture_var']), color='b', linestyle='--', label='Median Zenith 3 Var')
    # plot the 1 sigma lines
    plt.axhline(y = np.median(zenith1_phot['aperture_var']) + np.std(zenith1_phot['aperture_var']), color='r', linestyle=':', label='1 Sigma Zenith 1 Var')
    plt.axhline(y = np.median(zenith1_phot['aperture_var']) - np.std(zenith1_phot['aperture_var']), color='r', linestyle=':')   
    plt.axhline(y = np.median(zenith2_phot['aperture_var']) + np.std(zenith2_phot['aperture_var']), color='g', linestyle=':', label='1 Sigma Zenith 2 Var')
    plt.axhline(y = np.median(zenith2_phot['aperture_var']) - np.std(zenith2_phot['aperture_var']), color='g', linestyle=':')   
    plt.axhline(y = np.median(zenith3_phot['aperture_var']) + np.std(zenith3_phot['aperture_var']), color='b', linestyle=':', label='1 Sigma Zenith 3 Var') 
    plt.title('Aperture Variance Over Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Variance [counts^2]')
    plt.ylim(770**2,775**2)
    plt.legend(['Zenith 1', 'Zenith 2', 'Zenith 3'])
    plt.tight_layout()
    plt.show()
    #%% 
    # plot the variance plot separately for clarity
    CHAR_FREQ = 1/0.0113
    plt.figure(figsize=(10, 5))
    plt.plot(zenith1_phot['aperture_var'], marker='o')
    plt.plot(zenith2_phot['aperture_var'], marker='x')
    plt.plot(zenith3_phot['aperture_var'], marker='s')
    plt.axhline(y = np.median(zenith1_phot['aperture_var']), color='r', linestyle='--', label='Median Zenith 1 Var')
    plt.axhline(y = np.median(zenith2_phot['aperture_var']), color='g', linestyle='--', label='Median Zenith 2 Var')
    plt.axhline(y = np.median(zenith3_phot['aperture_var']), color='b', linestyle='--', label='Median Zenith 3 Var')
    # plot the 1 sigma lines as filled regions
    plt.fill_between(range(len(zenith1_phot['aperture_var'])),
                     np.median(zenith1_phot['aperture_var']) - np.std(zenith1_phot['aperture_var']),
                     np.median(zenith1_phot['aperture_var']) + np.std(zenith1_phot['aperture_var']),
                     color='r', alpha=0.2, label='1 Sigma Zenith 1 Var')
    plt.fill_between(range(len(zenith2_phot['aperture_var'])),
                     np.median(zenith2_phot['aperture_var']) - np.std(zenith2_phot['aperture_var']),
                     np.median(zenith2_phot['aperture_var']) + np.std(zenith2_phot['aperture_var']),
                     color='g', alpha=0.2, label='1 Sigma Zenith 2 Var')
    plt.fill_between(range(len(zenith3_phot['aperture_var'])),
                     np.median(zenith3_phot['aperture_var']) - np.std(zenith3_phot['aperture_var']),
                     np.median(zenith3_phot['aperture_var']) + np.std(zenith3_phot['aperture_var']),
                     color='b', alpha=0.2, label='1 Sigma Zenith 3 Var')
    plt.title(f'Aperture Variance Over Frame at Characteristic Frequency {CHAR_FREQ:.1f} Hz')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Variance [counts^2]')
    plt.ylim(770**2,775**2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    #%%
    print(f"Median Variance Zenith 1: {np.median(zenith1_phot['aperture_var'])}")
    #%%
    # Now coadd exposures 
    from exposure_sequences import coadd_exposures
    zenith1_coadd = coadd_exposures(zenith1_exp, coadds =2, mode='average')

    #%%
    zenith1_coadd_phot = preliminary_analysis(zenith1_coadd, region=region, trim=(50, -1))

    COADD_2_FREQ = CHAR_FREQ / 2
    #%%
    # Make the plots 
    plt.figure(figsize=(10, 4))
    plt.plot(zenith1_coadd_phot['aperture_var'], marker='o')
    plt.axhline(y = np.median(zenith1_coadd_phot['aperture_var']), color='r', linestyle='--', label='Median Zenith 1 Coadd Var')
    # plot the 1 sigma lines
    plt.axhline(y = np.median(zenith1_coadd_phot['aperture_var']) + np.std(zenith1_coadd_phot['aperture_var']), color='r', linestyle=':', label='1 Sigma Zenith 1 Coadd Var')  
    plt.axhline(y = np.median(zenith1_coadd_phot['aperture_var']) - np.std(zenith1_coadd_phot['aperture_var']), color='r', linestyle=':')   

    plt.title(f'Aperture Variance Over Frame at characteristic frequency {COADD_2_FREQ:.1f} Hz')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Variance [counts^2]')

    print(f"Median Variance after Coadding: {np.median(zenith1_coadd_phot['aperture_var'])}")  
# %%
    # Now try coadding by 4
    zenith1_coadd4 = coadd_exposures(zenith1_exp, coadds =4)
    zenith1_coadd4_phot = preliminary_analysis(zenith1_coadd4, region=region, trim=(25, -1))
    COADD_4_FREQ = CHAR_FREQ / 4
    #%%
    # Make the plots 
    plt.figure(figsize=(10, 4))
    plt.plot(zenith1_coadd4_phot['aperture_var'], marker='o')
    plt.axhline(y = np.median(zenith1_coadd4_phot['aperture_var']), color='r', linestyle='--', label='Median Zenith 1 Coadd4 Var')
    # plot the 1 sigma lines
    plt.axhline(y = np.median(zenith1_coadd4_phot['aperture_var']) + np.std(zenith1_coadd4_phot['aperture_var']), color='r', linestyle=':', label='1 Sigma Zenith 1 Coadd4 Var')  
    plt.axhline(y = np.median(zenith1_coadd4_phot['aperture_var']) - np.std(zenith1_coadd4_phot['aperture_var']), color='r', linestyle=':')
    plt.title(f'Aperture Variance Over Frame at characteristic frequency {COADD_4_FREQ:.2f} Hz')
    plt.xlabel('Frame Number')
    plt.ylabel('Aperture Variance [counts^2]')
    print(f"Median Variance after Coadding by 4: {np.median(zenith1_coadd4_phot['aperture_var'])}")
    plt.show()

#%%
    x = zenith1_phot['aperture_sum']   # time series at native frame rate

    n = 2  # coadd factor
    x_trim = x[: (len(x)//n)*n]        # drop remainder so it reshapes cleanly

    # dwell measurements (longer effective integration)
    x_coadd = x_trim.reshape(-1, n).sum(axis=1)   # <-- SUM for physical integration

    # synthetic chop differences between consecutive dwells
    d = x_coadd[1:] - x_coadd[:-1]

    # variance vs frequency point
    var_d = np.var(d, ddof=1)
# %%
def plot_frame_vs_temporal_variance(aperture_sum, n_coadd=1, label=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Trim so reshape works
    N = len(aperture_sum)
    Ntrim = (N // n_coadd) * n_coadd
    x = aperture_sum[:Ntrim]

    # Coadd in time (SUM = physical integration)
    x_coadd = x.reshape(-1, n_coadd).sum(axis=1)

    # Chop-like differences
    d = x_coadd[1:] - x_coadd[:-1]

    frames = np.arange(len(d))

    plt.figure(figsize=(8, 4))
    plt.plot(frames, d**2, marker='o', lw=1, label=label)
    plt.axhline(np.median(d**2), ls='--', color='k',
                label='Median $D^2$')
    plt.xlabel("Difference index")
    plt.ylabel("$(\\Delta$ Aperture Sum$)^2$ [counts$^2$]")
    plt.title(f"Frame vs Temporal Variance (coadd={n_coadd})")
    if label is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Variance of differences: {np.var(d, ddof=1):.3e}")

# %%
    plot_frame_vs_temporal_variance(
        zenith1_phot['aperture_sum'],
        n_coadd=5,
        label='Zenith 1'
)

# %%
