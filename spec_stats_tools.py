import numpy as np
from astropy.io import fits
from astropy.io import ascii
from pathlib import Path
import matplotlib.pyplot as plt

def compute_spectral_stats(target_dir):

    """
    Loads sorted x1d files, applies an Ubermask, and computes first order
    Mean and RMS spectra.
    """

    target_path = Path(target_dir)
    files = sorted(list(target_path.glob("*.fits")))

    if not files:
        print(f"No FITS files found in {target_dir}")
        return

    # 1. First pass: Identify all "bad" pixels across all epochs
    bad_indices = set()
    all_flux = []
    all_err = []
    wavelength = None

    print(f"Analyzing {len(files)} epochs for Ubermask...")

    for f in files:
        with fits.open(f) as hdul:
            data = hdul[1].data
            # Assuming COS/STIS x1d format where data is in the first row [0]
            flux = data['FLUX'][0, :]
            err = data['ERROR'][0, :]
            if wavelength is None:
                wavelength = data['WAVELENGTH'][0, :]

            # Identify bad pixels (NaNs, zero/negative error, or negative SNR)
            snr = flux / err
            bad = np.where(np.isnan(snr) | (err <= 0) | (snr < 0))[0]
            bad_indices.update(bad)

            all_flux.append(flux)
            all_err.append(err)

    # Convert to arrays and create the Ubermask
    flux_stack = np.array(all_flux)
    err_stack = np.array(all_err)
    ubermask = np.array(list(bad_indices))

    # 2. Apply Ubermask: Set bad pixels to NaN (better for averaging)
    # Using NaN allows np.nanmean to ignore them without biasing the mean to 0
    flux_stack[:, ubermask] = np.nan
    err_stack[:, ubermask] = np.nan

    # 3. Compute Mean and RMS
    # Mean Spectrum: $\bar{f}(\lambda) = \frac{1}{N} \sum f_i(\lambda)$
    mean_spectrum = np.nanmean(flux_stack, axis=0)

    # RMS Spectrum: $S(\lambda) = \sqrt{\frac{1}{N-1} \sum (f_i(\lambda) - \bar{f}(\lambda))^2}$
    rms_spectrum = np.nanstd(flux_stack, axis=0)

    return wavelength, mean_spectrum, rms_spectrum

# --- Execution ---
RMID = '303'
data_dir = f"./hst_data_analysis/rm{RMID}_x1d_files"

wave, mean_spec, rms_spec = compute_spectral_stats(data_dir)

# --- Simple Diagnostic Plot ---
plt.figure(figsize=(12, 6))
plt.step(wave, mean_spec, label='Mean Spectrum', color='black', lw=1)
plt.step(wave, rms_spec, label='RMS Spectrum', color='red', lw=1)
plt.xlabel(r'Wavelength (\AA)')
plt.ylabel('Flux')
plt.title(f'Target RM{RMID} - Mean & RMS')
plt.legend()
plt.savefig('rms_results.pdf')
