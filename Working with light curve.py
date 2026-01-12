import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# filename = "lightcurve_3.0_80.0keV.lc"
filename = "test4.lc"
# filename = ""
' put the file name here in the above code'

# 
hdul = fits.open("test4.lc")
data = hdul[1].data
print(data.columns.names)
# 
with fits.open(filename) as hdul:
    data = hdul[1].data
    time = data['TIME']
    flux = data['RATE']
    if 'ERROR' in data.columns.names:
        flux_err = data['ERROR']
    else:
        flux_err = np.sqrt(np.abs(flux))  # Poisson approximation if no error column

plt.figure(figsize=(8, 5))
plt.errorbar(time, flux, yerr=flux_err, fmt='.', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Flux (counts/s)")
plt.tight_layout()
plt.show()       

mask = np.isfinite(time) & np.isfinite(flux)
time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

dt = np.median(np.diff(time))
N = len(flux)

seg_len = 1024        # number of points per segment
M = N // seg_len      # number of complete segments
print(f"Using {M} segments of {seg_len} points each.")

# Trim data to full segments
flux = flux[:M * seg_len]
time = time[:M * seg_len]
flux_err = flux_err[:M * seg_len]

def compute_pds(segment_flux, dt):
    Nseg = len(segment_flux)
    flux_mean = np.mean(segment_flux)
    flux_zero_mean = segment_flux - flux_mean
    fft_vals = np.fft.rfft(flux_zero_mean)
    abs_fft2 = np.abs(fft_vals)**2
    df = 1.0 / (Nseg * dt)
    pds = (2.0 * abs_fft2) / (Nseg**2 * flux_mean**2 * df)
    return pds

freqs = np.fft.rfftfreq(seg_len, dt)
df = freqs[1] - freqs[0]

pds_all = []
for i in range(M):
    seg_flux = flux[i * seg_len:(i + 1) * seg_len]
    pds_seg = compute_pds(seg_flux, dt)
    pds_all.append(pds_seg)

pds_all = np.array(pds_all)
pds_avg = np.mean(pds_all, axis=0)
pds_err = np.std(pds_all, axis=0, ddof=1) / np.sqrt(M)  # 1σ error of mean

print(f"Averaged PDS from {M} segments, reduced scatter by √{M:.0f}")

def log_rebin_with_error(freq, power, power_err, bins_per_decade=100):
    """Logarithmically rebin PDS with propagated errors."""
    pos = (freq > 0)
    freq, power, power_err = freq[pos], power[pos], power_err[pos]
    log_min, log_max = np.log10(freq[0]), np.log10(freq[-1])
    n_bins = int((log_max - log_min) * bins_per_decade)
    bin_edges = np.logspace(log_min, log_max, n_bins)

    freq_rb, power_rb, err_rb = [], [], []
    for i in range(len(bin_edges) - 1):
        m = (freq >= bin_edges[i]) & (freq < bin_edges[i + 1])
        if np.any(m):
            freq_rb.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
            pw = power[m]
            pe = power_err[m]
            power_rb.append(np.mean(pw))
            err_rb.append(np.sqrt(np.sum(pe**2)) / np.sum(m))
    return np.array(freq_rb), np.array(power_rb), np.array(err_rb)

freq_rb, pds_rb, pds_err_rb = log_rebin_with_error(freqs[1:], pds_avg[1:], pds_err[1:], bins_per_decade=100)

plt.figure(figsize=(8, 5))
plt.loglog(freqs[1:], pds_avg[1:], color='k', alpha=0.2, label='Raw averaged PDS')
plt.errorbar(freq_rb, pds_rb, yerr=pds_err_rb, fmt='o', ms=4,
             color='green', ecolor='b', elinewidth=1, capsize=2,
             label='Log-rebinned averaged PDS (1σ errors)')

plt.xlabel('Frequency [Hz or 1/time unit]')
plt.ylabel(r'Power Density $[(\mathrm{rms/mean})^2/\mathrm{Hz}]$')
plt.title(f'Averaged Power Density Spectrum (M={M})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"dt = {dt:.6f}")
print(f"df = {df:.5e}")
print(f"Segment length = {seg_len} bins")
print(f"Frequency range: {freqs[1]:.3e} – {freqs[-1]:.3e}")
print(f"Mean fractional rms² (integrated PDS): {np.trapz(pds_avg[1:], freqs[1:]):.3e}")

def powerlaw(f, A, alpha, C):
    """Power-law + constant model."""
    return A * f**(-alpha) + C

def lorentzian(f, norm, f0, width):
    """Lorentzian profile for QPO."""
    return norm * (width / (2*np.pi)) / ((f - f0)**2 + (width/2)**2)

def powerlaw_plus_lorentzian(f, A, alpha, C, norm, f0, width):
    """Combined broadband + Lorentzian QPO."""
    return powerlaw(f, A, alpha, C) + lorentzian(f, norm, f0, width)

#  new lorentz

def powerlaw_2_lorentzian(f, A, alpha, C, norm, f0, width, norm1, f01, width1):
    return powerlaw(f, A, alpha, C) + lorentzian(f, norm, f0, width) + lorentzian(f, norm1, f01, width1)



fmin = float(input("Enter minimum frequency for fitting (e.g., 0.001): "))
fmax = float(input("Enter maximum frequency for fitting (e.g., 100.0): "))

sel = (freq_rb > fmin) & (freq_rb < fmax)
f_fit, p_fit, err_fit = freq_rb[sel], pds_rb[sel], pds_err_rb[sel]

from scipy.optimize import curve_fit

print("\nEnter initial guesses for Power-law fit:")
A_guess = float(input("  A (amplitude) ≈ "))
alpha_guess = float(input("  alpha (index) ≈ "))
C_guess = float(input("  C (constant noise floor) ≈ "))

p0_pl = [A_guess, alpha_guess, C_guess]
bounds_pl = ([1e-15, 0, 0], [1e5, 10, np.max(p_fit)*2])

popt_pl, pcov_pl = curve_fit(
    powerlaw, f_fit, p_fit, sigma=err_fit,
    p0=p0_pl, bounds=bounds_pl, maxfev=10000
)
perr_pl = np.sqrt(np.diag(pcov_pl))

# χ² and reduced χ² for power-law
model_pl = powerlaw(f_fit, *popt_pl)
chi2_pl = np.sum(((p_fit - model_pl) / err_fit) ** 2)
red_chi2_pl = chi2_pl / (len(f_fit) - len(popt_pl))

print("\nPower-law fit results:")
print(f"  A        = {popt_pl[0]:.3e} ± {perr_pl[0]:.3e}")
print(f"  alpha    = {popt_pl[1]:.3f} ± {perr_pl[1]:.3f}")
print(f"  constant = {popt_pl[2]:.3e} ± {perr_pl[2]:.3e}")
print(f"  χ² = {chi2_pl:.2f},  reduced χ² = {red_chi2_pl:.2f}")

# there should be the plot for only power law graph 

'below is yhe code power + lorentzian '

print("\nEnter initial guesses for Lorentzian (QPO) component:")
norm_guess  = float(input("  norm ≈ "))
f0_guess    = float(input("  f0 (centroid freq, Hz) ≈ "))
width_guess = float(input("  width (Hz) ≈ "))

p0_ql = [
    popt_pl[0], popt_pl[1], popt_pl[2],
    norm_guess, f0_guess, width_guess
]
bounds_ql = (
    [1e-15, 0, 0, 1e-10, fmin, 1e-5],
    [1e5, 10, np.max(p_fit)*2, 1e3, fmax, np.max(f_fit)]
)

popt_ql, pcov_ql = curve_fit(
    powerlaw_plus_lorentzian, f_fit, p_fit, sigma=err_fit,
    p0=p0_ql, bounds=bounds_ql, maxfev=20000
)
perr_ql = np.sqrt(np.diag(pcov_ql))

A, alpha, C, norm, f0, width = popt_ql
model_ql = powerlaw_plus_lorentzian(f_fit, *popt_ql)

# χ² and reduced χ² for combined model
chi2_ql = np.sum(((p_fit - model_ql) / err_fit) ** 2)
red_chi2_ql = chi2_ql / (len(f_fit) - len(popt_ql))

Q_factor = f0 / width

print("\nPower-law + Lorentzian fit results:")
print(f"  A        = {A:.3e} ± {perr_ql[0]:.3e}")
print(f"  alpha    = {alpha:.3f} ± {perr_ql[1]:.3f}")
print(f"  constant = {C:.3e} ± {perr_ql[2]:.3e}")
print(f"  norm(QPO)= {norm:.3e} ± {perr_ql[3]:.3e}")
print(f"  f0(QPO)  = {f0:.3f} ± {perr_ql[4]:.3f} Hz")
print(f"  width(QPO)= {width:.3f} ± {perr_ql[5]:.3f} Hz")
print(f"  Q-factor = {Q_factor:.2f}")
print(f"  χ² = {chi2_ql:.2f},  reduced χ² = {red_chi2_ql:.2f}")

'changes for 2 lorentzian '

# ================== Power-law + 2 Lorentzians fit ==================

print("\nEnter initial guesses for SECOND Lorentzian (QPO2) component:")
norm1_guess  = float(input("  norm2 ≈ "))
f01_guess    = float(input("  f02 (centroid freq, Hz) ≈ "))
width1_guess = float(input("  width2 (Hz) ≈ "))

# Initial guess for powerlaw + 2 Lorentzians:
p0_2lor = [
    popt_pl[0], popt_pl[1], popt_pl[2],   # from power-law fit
    norm_guess, f0_guess, width_guess,    # first Lorentzian
    norm1_guess, f01_guess, width1_guess  # second Lorentzian
]

bounds_2lor = (
    [1e-15, 0, 0,        1e-10,  fmin, 1e-5,    1e-10,  fmin, 1e-5],
    [1e5,   10, np.max(p_fit)*2, 1e3,   fmax, np.max(f_fit),
     1e3,   fmax, np.max(f_fit)]
)

popt_2lor, pcov_2lor = curve_fit(
    powerlaw_2_lorentzian, f_fit, p_fit, sigma=err_fit,
    p0=p0_2lor, bounds=bounds_2lor, maxfev=40000
)
perr_2lor = np.sqrt(np.diag(pcov_2lor))

A2, alpha2, C2, norm2a, f0a, width2a, norm2b, f0b, width2b = popt_2lor
model_2lor = powerlaw_2_lorentzian(f_fit, *popt_2lor)

# χ² and reduced χ² for powerlaw + 2 Lorentzians
chi2_2lor = np.sum(((p_fit - model_2lor) / err_fit) ** 2)
red_chi2_2lor = chi2_2lor / (len(f_fit) - len(popt_2lor))

Q1 = f0a / width2a
Q2 = f0b / width2b

print("\nPower-law + 2 Lorentzian fit results:")
print(f"  A          = {A2:.3e} ± {perr_2lor[0]:.3e}")
print(f"  alpha      = {alpha2:.3f} ± {perr_2lor[1]:.3f}")
print(f"  constant   = {C2:.3e} ± {perr_2lor[2]:.3e}")
print(f"  norm1(QPO) = {norm2a:.3e} ± {perr_2lor[3]:.3e}")
print(f"  f01(QPO)   = {f0a:.3f} ± {perr_2lor[4]:.3f} Hz")
print(f"  width1     = {width2a:.3f} ± {perr_2lor[5]:.3f} Hz")
print(f"  Q1-factor  = {Q1:.2f}")
print(f"  norm2(QPO) = {norm2b:.3e} ± {perr_2lor[6]:.3e}")
print(f"  f02(QPO)   = {f0b:.3f} ± {perr_2lor[7]:.3f} Hz")
print(f"  width2     = {width2b:.3f} ± {perr_2lor[8]:.3f} Hz")
print(f"  Q2-factor  = {Q2:.2f}")
print(f"  χ² = {chi2_2lor:.2f},  reduced χ² = {red_chi2_2lor:.2f}")

# ================== end Power-law + 2 Lorentzians fit ==================


'this is the plot for power law + lorentzian '
ratio = p_fit / model_ql
residuals = (p_fit - model_ql) / err_fit

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                               gridspec_kw={'height_ratios':[3, 1]})

# Top panel: data + model
ax1.errorbar(f_fit, p_fit, yerr=err_fit, fmt='o', ms=3, color='black',
             ecolor='gray', alpha=0.6, label='Averaged PDS')
ax1.plot(f_fit, powerlaw(f_fit, *popt_pl), 'b--', lw=1.3, label='Power-law fit')
ax1.plot(f_fit, model_ql, 'r-', lw=2, label='Power-law + Lorentzian')
ax1.axvline(f0, color='r', ls=':', alpha=0.5, label=f'QPO ~ {f0:.3f} Hz')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'Power Density $(\mathrm{rms/mean})^2/\mathrm{Hz}$')
ax1.set_title('PDS Fit: Power-law + Lorentzian')
ax1.legend()

# Bottom panel: residual
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.errorbar(f_fit, residuals, yerr=np.ones_like(residuals), fmt='o', color='darkgreen', ms=3)
ax2.set_xscale('log')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Residual')

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------
' this below plot is for powerlaw + 2 Lorentzians'
# Use powerlaw + 2 Lorentzians as reference for residuals
# ratio = p_fit / model_2lor
# residuals = (p_fit - model_2lor) / err_fit

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
#                                gridspec_kw={'height_ratios':[3, 1]})

# # Top panel: data + all models
# ax1.errorbar(f_fit, p_fit, yerr=err_fit, fmt='o', ms=3, color='black',
#              ecolor='gray', alpha=0.6, label='Averaged PDS')

# ax1.plot(f_fit, model_pl, 'b--', lw=1.3, label='Power-law only')
# ax1.plot(f_fit, model_ql, 'r-',  lw=1.5, label='Power-law + 1 Lorentzian')
# ax1.plot(f_fit, model_2lor, 'm-', lw=2.0, label='Power-law + 2 Lorentzians')

# ax1.axvline(f0,  color='r', ls=':', alpha=0.5, label=f'QPO1 ~ {f0:.3f} Hz')
# ax1.axvline(f0a, color='m', ls='--', alpha=0.5, label=f'QPO2 ~ {f0a:.3f} Hz')

# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_ylabel(r'Power Density $(\mathrm{rms/mean})^2/\mathrm{Hz}$')
# ax1.set_title('PDS Fits: PL, PL+1L, PL+2L')
# ax1.legend()

# # Bottom panel: residuals w.r.t. PL+2L
# ax2.axhline(0, color='k', ls='--', lw=1)
# ax2.errorbar(f_fit, residuals, yerr=np.ones_like(residuals),
#              fmt='o', color='darkgreen', ms=3)
# ax2.set_xscale('log')
# ax2.set_xlabel('Frequency [Hz]')
# ax2.set_ylabel('Residual')

# plt.tight_layout()
# plt.show()





' below is the plot for power law only '

# # === Standalone plot: data vs power-law only ===
# plt.figure(figsize=(8, 6))
# plt.errorbar(f_fit, p_fit, yerr=err_fit, fmt='o', ms=3,
#              color='black', ecolor='gray', alpha=0.6,
#              label='Averaged PDS (rebinned)')
# plt.plot(f_fit, model_pl, 'b-', lw=2, label='Power-law fit')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel(r'Power Density $(\mathrm{rms/mean})^2/\mathrm{Hz}$')
# plt.title('PDS Fit: Power-law only')
# plt.legend()
# plt.tight_layout()
# plt.show()
# # === end standalone power-law-only plot ===















