import numpy as np
import sys
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt


def arx2(n, dat, sig, verbose = False):
    """
    Maximum likelihood estimator of the mean and intrinsic rms (ARX2).

    Given independent samples with error bars, this function calculates the
    weighted mean and excess scatter (RMS). This implements the initial setup,
    initial estimates, and the core calculation loop for the iterative solver.

    Args:
        n (int): Number of data points (length of dat and sig).
        dat (np.ndarray): Data values (float32/4).
        sig (np.ndarray): 1-sigma error bars (float32/4). (<0 to skip).
        verbose (bool, optional): If True, prints diagnostic messages. Defaults to False.

    Returns:
        tuple: (good, avg, rms, sigavg, sigrmsm, sigrmsp)
               - good (float): Number of "good" data points.
               - avg (float): Mean value.
               - rms (float): Excess rms.
               - sigavg (float): 1-sigma error bar on the mean.
               - sigrmsm (float): 1-sigma lower limit on excess rms.
               - sigrmsp (float): 1-sigma upper limit on excess rms.
    """

    # --- Initialization of Output Variables ---
    good = 0.0
    avg = 0.0
    rms = 0.0
    sigavg = -1.0
    sigrmsm = -1.0
    sigrmsp = -1.0
    e1 = 0.0 # Mean error bar (for scaling)

    # Placeholder for sigrms (calculated uncertainty on rms)
    sigrms = -1.0

    # Placeholder for the convergence metrics, needed for final report
    test = 1000.0
    chia = 0.0
    chir = 0.0
    chix = 0.0
    chib = 0.0
    chip = 0.0
    chim = 0.0

    # Fortran: trap no input data
    if n <= 0:
        if verbose:
            print(f"** ERROR in ARX2: NO INPUT DATA n={n}", file=sys.stderr)
        return good, avg, rms, sigavg, sigrmsm, sigrmsp

    # --- Filtering Data and Calculating ng ---
    positive_sig_mask = (sig > 0.0)
    ng = np.sum(positive_sig_mask)
    dat_good = dat[positive_sig_mask]
    sig_good = sig[positive_sig_mask]

    # --- Error Check 1: No valid data ---
    if ng <= 0:
        if verbose:
            print(f"** ERROR in ARX2: NO >0 ERROR BARS n {n} ng {ng}", file=sys.stderr)
        return good, avg, rms, sigavg, sigrmsm, sigrmsp

    # --- Calculate Mean Error Bar (e1) for Scaling ---
    sum_e = np.sum(sig_good.astype(np.float64))
    e1 = sum_e / ng

    # --- Error Check 2: Zero mean error bar (e1) ---
    if e1 <= 0.0:
        print(f"** ERROR in ARX2. MEAN ERROR BAR e1 {e1}", file=sys.stderr)
        print(f"** sum {sum_e} n {n} ng {ng}", file=sys.stderr)
        return good, avg, rms, sigavg, sigrmsm, sigrmsp

    good = float(ng)

    # --- Initial Estimate Sums ---
    # Ensure inputs are double precision for accurate calculation of sums throughout
    dat_good_d = dat_good.astype(np.float64)
    sig_good_d = sig_good.astype(np.float64)
    e1_initial = float(e1)

    # Initial weights w = (e1 / e)**2
    w_initial = (e1_initial / sig_good_d) ** 2
    sum1_initial = np.sum(w_initial * dat_good_d) # sum1 = sum(w * d)
    sum2_initial = np.sum(w_initial)              # sum2 = sum(w)

    # --- Error Check 3: Check for zero sum of weights ---
    if sum2_initial <= 0.0:
        print(f"** ERROR in ARX2. n {n} ng {ng} sum2={sum2_initial:.6f}", file=sys.stderr)
        return good, avg, rms, sigavg, sigrmsm, sigrmsp

    # --- Optimal Average and its Uncertainty (Initial values) ---
    # Fortran: avg = sum1 / sum2
    avg = sum1_initial / sum2_initial

    # Fortran: sigavg = e1 / dsqrt( sum2 )
    sigavg = e1_initial / np.sqrt(sum2_initial)

    # Fortran: e1 is scaled here (normalization)
    # w = ng / sum2; e1 = e1 * sqrt(w)
    w_scale = float(ng) / sum2_initial
    e1 = e1_initial * np.sqrt(w_scale)
    rms = e1 # Initial estimate for intrinsic rms

    # Convergence and minimum constants
    tiny = 1.0e-4
    rmin = 1.0e-3

    if verbose:
        print(f" sum1 {sum1_initial:.6f} sum2 {sum2_initial:.6f} ng {ng}")
        print(f" avg {avg:.6f} sigavg {sigavg:.6f}")
        print(f" rms {rms:.6f} tiny {tiny:.6f} rmin {rmin:.6f}")

    # --- Initial Upper/Lower Limits on Extra Variance ---
    xp = 1.0  # Initial fractional upper limit factor
    xm = -1.0 # Initial fractional lower limit factor
    sigrms_calc = -1.0 # Local variable for the calculated error bar on rms

    # --- MLE ITERATION START ---
    nloop = 5000
    for loop in range(1, nloop + 1):
        its = loop

        # Stow for convergence test
        oldavg = avg
        oldrms = rms

        # Zero sums for the current iteration (using 0.0 for double precision)
        sum_total = 0.0  # Renamed from 'sum' to avoid conflict with Python's sum()
        sumg = 0.0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0

        # Best BoF (Best of Fit) at v0 (current intrinsic variance)
        v0 = rms * rms
        bof = 0.0

        # Upper/lower limits at vp/vm aim for dBoF=1
        vp = v0 * (1.0 + xp)
        vm = v0 * (1.0 + xm)

        # Ensure vm is non-negative variance
        vm = max(0.0, vm)

        dbp = 0.0 # dBoF for upper limit
        dbm = 0.0 # dBoF for lower limit

        # --- Core Summation: Calculation of weights and terms (Scaled data loop) ---

        # Fortran: scaled avg and rms
        a = avg / e1
        r = rms / e1
        rr = r * r

        # 1. Scale data and error bar (d and e are now dimensionless)
        d_scaled = dat_good_d / e1
        e_scaled = sig_good_d / e1

        # 2. "Goodness" of this data point (g)
        # g = rr / ( rr + e * e )
        g = rr / (rr + e_scaled**2)

        # 3. Intermediate terms
        # x = g * ( d - a )
        x = g * (d_scaled - a)
        # xx = x * x
        xx = x * x

        # 4. Sums for avg and rms update (Vectorized sum)
        # sum  = sum  + g * g
        sum_total = np.sum(g**2)
        # sumg = sumg + g
        sumg = np.sum(g)
        # sum1 = sum1 + g * d
        sum1 = np.sum(g * d_scaled)
        # sum2 = sum2 + xx
        sum2 = np.sum(xx)
        # sum3 = sum3 + g * xx
        sum3 = np.sum(g * xx)

        # --- Update 'good' count and Error Check for sumg ---
        # Fortran: good = sumg
        good = sumg # Update number of effective "good" data points (sum of weights g)

        # Fortran: if( sumg .le. 0.d0 ) then ... stop
        if sumg <= 0.0:
            print(f"** ERROR in ARX2. NON-POSITIVE SUM(G)={sumg:.6f}", file=sys.stderr)
            print(f"** loop {loop} ndat {n} ngood {ng}", file=sys.stderr)
            print(f"** sumg {sumg:.6f} sum1 {sum1:.6f} sum {sum_total:.6f}", file=sys.stderr)
            print(f"** r {r:.6f} rr {rr:.6f}", file=sys.stderr)
            print(f"** dat:\n{dat}", file=sys.stderr)
            print(f"** sig:\n{sig}", file=sys.stderr)
            print("Should we abort here ? (Aborting via sys.exit(1))", file=sys.stderr)
            sys.exit(1) # Mimics Fortran STOP

        # --- Likelihood (BoF) Calculation for Confidence Intervals ---

        # Pre-calculate common terms outside the BoF sum loop
        d_diff = dat_good_d - avg
        dd_diff = d_diff**2
        ss_sq = sig_good_d**2

        # 1. Best fit at v0 (current variance)
        # v = ss + v0
        v_bof = ss_sq + v0
        # add = dd / v + alog( v )
        add_bof = dd_diff / v_bof + np.log(v_bof)
        # bof = bof + add
        bof = np.sum(add_bof)

        # 2. dBoF at upper limit vp
        v_p = ss_sq + vp
        add_p = dd_diff / v_p + np.log(v_p)
        dbp = np.sum(add_p)

        # 3. dBof at lower limit vm
        v_m = ss_sq + vm
        add_m = dd_diff / v_m + np.log(v_m)
        dbm = np.sum(add_m)

        # --- Subtract Best BoF ---
        # Note: The Fortran verbose block reporting intermediate BoF values is now skipped for brevity,
        # but the core calculation remains.

        # dbp = dbp - bof
        dbp = dbp - bof
        # dbm = dbm - bof
        dbm = dbm - bof

        # --- Revise Estimates and Error Bars (Unsmoothed) ---

        # * revise avg and rms (Scaled)
        a = sum1 / sumg
        r = np.sqrt(sum2) / np.sqrt(sumg)
        r = max(rmin, r)

        # * error bars on scaled avg and rms
        siga = r / np.sqrt(sumg)

        # Fortran: g = ( sum3 + sum3 ) / sum2 * sumg - sum
        g_denom = (sum3 * 2.0) / sum2 * sumg - sum_total
        # Fortran: g = g + g (Note: This is doubling the result, likely 2*G - G_squared)
        g_denom = g_denom + g_denom

        sigr = r
        if g_denom > 0.0:
            sigr = r / np.sqrt(g_denom)

        # * restore scaling (Unsmoothed avg and rms are stored temporarily)
        temp_avg = a * e1
        temp_rms = r * e1
        sigavg = siga * e1
        sigrms_calc = sigr * e1 # Local variable for the calculated error bar (sigrms)

        # * KDH: CATCH SUM2=0 (NO VARIANCE)
        if sum2 <= 0.0:
            rms = 0.0
            sigavg = -1.0
            sigrmsm = -1.0
            sigrmsp = -1.0
            if verbose:
                print("** ERROR in ARX2. NON-POSITIVE SUM(DD*G)=0. ABORTING.", file=sys.stderr)
            return good, avg, rms, sigavg, sigrmsm, sigrmsp # Exit subroutine

        # --- Revise Lower/Upper Limits (Confidence Interval Logic) ---
        safe = 0.9

        # * upper limit ( xp > 0 )
        old_xp = xp
        if dbp > 0.0:
            # xp = xp / dsqrt( dabs( dbp ) )
            xp = xp / np.sqrt(abs(dbp))
            # xp = old * ( 1. - safe ) + safe * xp (A weighted average to dampen oscillations)
            xp = old_xp * (1.0 - safe) + safe * xp

        # * lower limit ( xm < 0 )
        old_xm = xm
        if dbm > 0.0:
            # xm = xm / dsqrt( dabs( dbm ) )
            xm = xm / np.sqrt(abs(dbm))
            # xm = max( xm, -1. )
            xm = max(xm, -1.0) # Ensure xm is not less than -1 (variance v = v0*(1+xm) >= 0)
            # xm = oldxm * ( 1. - safe ) + safe * xm
            xm = old_xm * (1.0 - safe) + safe * xm

        # --- Confidence Interval Steps (Updates sigrmsm/p from Fortran logic) ---
        if loop > 1:
            # Stow old values
            oldp = sigrmsp
            oldm = sigrmsm

            # * step upper limit
            sigrmsp = temp_rms * (np.sqrt(xp + 1.0) - 1.0)

            # * step lower limit
            sigrmsm = temp_rms
            if dbm > 0.0:
                sigrmsm = temp_rms * (1.0 - np.sqrt(max(0.0, 1.0 + xm)))

            # Safety check: ensure sigrmsm >= 0
            sigrmsm = max(0.0, sigrmsm)

        # --- Smoothing and Convergence Metrics ---

        # * step avg and rms (Dampening)
        # Note: We now use temp_avg/temp_rms for the calculated (unsmoothed) values
        avg = oldavg * (1.0 - safe) + safe * temp_avg
        rms = oldrms * (1.0 - safe) + safe * temp_rms

        # * floor on rms halts ln(rms) iterating to -infty
        if loop > 1 and sigrmsp > 0:
            rms = max(rms, tiny * sigrmsp)

        # * changes in avg and rms in units of their uncertainty
        sigrms = sigrms_calc
        chia = (avg - oldavg) / sigavg if sigavg > 1e-12 else 1000.0
        chir = (rms - oldrms) / sigrms if sigrms > 1e-12 else 1000.0

        # * changes in upper limit at dBoF = 1
        if loop > 1:
            # old_xp is available from before it was updated
            chix = (xp - old_xp)
            # chib = ( dbp - 1.d0 ) * tiny / 0.01 (Normalized difference from target dBoF=1)
            chib = (dbp - 1.0) * tiny / 0.01

            # chip = ( sigrmsp - oldp ) / sigrmsp
            chip = (sigrmsp - oldp) / sigrmsp if sigrmsp > 1e-12 else 0.0

            # chim = ( sigrmsm - oldm ) / sigrmsm
            chim = (sigrmsm - oldm) / sigrmsm if sigrmsm > 1e-12 else 0.0

        else:
            # Set convergence change metrics to large values for the first loop
            chix, chib, chip, chim = 100.0, 100.0, 100.0, 100.0

        # * changes < tiny
        # test = max( abs( chia ), abs( chir ) )
        test = max(abs(chia), abs(chir))

        # If loop > 1, update test with confidence interval metrics
        if loop > 1:
            test = max(test, abs(chib))
            test = max(test, abs(chip))

            # if( sigrmsm .lt. 0.99 * rms ) test = max( test, abs( chim ) )
            if sigrmsm < 0.99 * rms:
                test = max(test, abs(chim))

        # test = test / tiny
        test = test / tiny

        # --- Report last 5 of max iterations (ONLY if verbose is True) ---
        if verbose:
            if loop > nloop - 3 or verbose:
                # Use the calculated sigrmsm/p for reporting
                sigrmsp_report = sigrmsp if loop > 1 else rms * (np.sqrt(1.0 + xp) - 1.0)
                sigrmsm_report = sigrmsm if loop > 1 else rms * (1.0 - np.sqrt(1.0 + xm))
                sigrmsm_report = max(0.0, sigrmsm_report)

                # Use the calculated chib/chix for reporting if loop > 1
                chib_report = chib if loop > 1 else 0.0
                chix_report = chix if loop > 1 else 0.0

                print()
                print(f"Loop {loop} of {nloop} in ARX2")
                print(f"Ndat {n} Ngood {good:.4f} Neff {sumg:.4f}")
                print(f" avg {avg:.6f} rms {rms:.6f}")
                print(f" +/- {sigavg:.6f} +/- {sigrms_calc:.6f} ({sigrmsm_report:.6f} {sigrmsp_report:.6f} )")
                # Note: The test metrics are printed normalized by tiny for the report
                print(f" testa {chia/tiny:.6f} testr {chir/tiny:.6f} test {test:.6f}")
                print(f" testx {chix/tiny:.6f} test+ {chip/tiny:.6f} test {test:.6f}")
                print(f" testb {chib/tiny:.6f} test- {chim/tiny:.6f} test {test:.6f}")

                if test < 1.0:
                    print('** ARX2 CONVERGED ** :))')

        # * converged
        # if( test .lt. 1. ) goto 10
        if (test < 1.0) and (loop > 1):
            its_converged = loop
            break # Exit the for loop (equivalent to GOTO 10)

        # * quit if rms vanishes
        if rms <= 0.0:
            # We already set the final outputs (sigrmsm=0.0, sigrmsp=-1.0 if sum2=0)
            # upon exiting the sum2 check, but this is a final fail-safe.
            if verbose:
                print("** ARX2 RMS VANISHED. RETURNING.")
            return good, avg, rms, sigavg, sigrmsm, sigrmsp

        # * next loop is handled by the for loop structure

    # --- END MLE ITERATION ---

    # Check if loop completed without converging
    if loop == nloop and test >= 1.0:
        its_converged = nloop
        print('** ARX2 MAX ITERATIONS', nloop)
        print('** FAILED TO CONVERGE :(((((', file=sys.stderr)
    else:
        its_converged = its

    # --- Finalization (Safety checks for edge cases) ---
    # Fallback/Safety Check (If rms is zero or very small, the limits might be off)
    if rms <= rmin * e1:
        # If rms is zero (consistent with no intrinsic scatter), the lower limit is zero.
        sigrmsm = 0.0

        # If the upper limit is still positive, set it based on the current xp
        if np.sqrt(1.0 + xp) > 1.0:
            sigrmsp = rms * (np.sqrt(1.0 + xp) - 1.0)
        else:
             # If xp is small or negative (shouldn't happen for upper bound), use calculated sigr
            sigrmsp = sigrms_calc

    # Final check: ensure sigrmsm is non-negative on exit.
    sigrmsm = max(0.0, sigrmsm)

    # Final Verbose Report (Equivalent to Fortran GOTO 10 block)
    if verbose:
        print()
        print('ARX2 iterations', its_converged)
        print(f" N {n} good {good:.4f} E1 {e1:.6f}")
        print(f" avg = {avg:.6f} +/- {sigavg:.6f}")
        print(f" rms = {rms:.6f} +/- {sigrms:.6f}")
        print(f"  => ({rms-sigrmsm:.6f}, {rms:.6f}, {rms + sigrmsp:.6f} )")
        print(f"     ({-sigrmsm:.6f}, 0.000000, {sigrmsp:.6f} )")

    return good, avg, rms, sigavg, sigrmsm, sigrmsp





def compute_mle_spectra(target_dir, use_ubermask=True):
    """
    Aggregates x1d files, applies optional Ubermask, and runs pixel-by-pixel ARX2.
    """
    target_path = Path(target_dir)
    files = sorted(list(target_path.glob("*.fits")))

    if not files:
        print(f"Error: No FITS files found in {target_dir}")
        return None

    all_flux, all_err, wavelength = [], [], None

    for f in files:
        with fits.open(f) as hdul:
            all_flux.append(hdul[1].data['FLUX'][0, :])
            all_err.append(hdul[1].data['ERROR'][0, :])
            if wavelength is None:
                wavelength = hdul[1].data['WAVELENGTH'][0, :]

    flux_stack = np.array(all_flux)
    err_stack = np.array(all_err)
    num_epochs, num_pixels = flux_stack.shape

    # 1. Ubermask Step
    if use_ubermask:
        print(f"Applying Ubermask to {num_epochs} epochs...")
        is_bad = (np.isnan(flux_stack) | (err_stack <= 0) | (flux_stack/err_stack < 0))
        uber_bad_indices = np.any(is_bad, axis=0)
        err_stack[:, uber_bad_indices] = -1.0 # Force ARX2 to skip these pixels

    # 2. Results Containers
    m_mean, m_rms, m_rms_lo, m_rms_hi = np.zeros(num_pixels), np.zeros(num_pixels), np.zeros(num_pixels), np.zeros(num_pixels)

    # 3. MLE Loop
    print(f"Running ARX2 MLE on {num_pixels} pixels...")
    for i in range(num_pixels):
        res = arx2(num_epochs, flux_stack[:, i], err_stack[:, i])
        m_mean[i], m_rms[i], m_rms_lo[i], m_rms_hi[i] = res[1], res[2], res[4], res[5]

    return wavelength, m_mean, m_rms, m_rms_lo, m_rms_hi





if __name__ == "__main__":

    # Example Usage for Target 303
    TARGET_ID = '303'
    DATA_PATH = Path.cwd() / "hst_data_analysis" / f"rm{TARGET_ID}_x1d_files"

    results = compute_mle_spectra(DATA_PATH, use_ubermask = True)

    if results:
        wave, mean_spec, rms_spec, rms_lo, rms_hi = results

        plt.figure(figsize=(10, 5))
        plt.step(wave, mean_spec * 1.0e16, color='black', label='Mean')
        plt.step(wave, rms_spec * 1.0e16, color='red', label='Intrinsic RMS')
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux')
        plt.legend()
        plt.title(f'Target RM{TARGET_ID} Variability Analysis')
        plt.savefig('rms_results_intrinsic.pdf')
