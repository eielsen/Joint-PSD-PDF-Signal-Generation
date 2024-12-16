#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Examples showing how to generate signals with jointly specified PSD and PDF.

Mainly based on the paper:
[1] M. M. Sondhi, "Random Processes With Specified Spectral Density and First-Order Probability Density", Bell System Technical Journal, vol. 62, no. 3, pp. 679-701, Mar. 1983, doi: 10.1002/j.1538-7305.1983.tb04411.x.

Added some alternatives for synthesising correction filter.

@author: Arnfinn A. Eielsen (AAE), Ahmad Faza (AF)
@date: 12.12.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import signal
from scipy import special
import matplotlib.pyplot as plt
import math
#import control as ct
import fir_filter_ls as ff
import pdf_shaping as ps


def h2_norm(S, w):
    # compute the PSD norm/area using simple numerical integration of freq. resp. 
    #   S - real valued PSD frequency response, where S = H*H'
    #   w - frequency sampling points, [-pi, pi]

    S = np.abs(S)  # ensure real-valued TODO: error handling
    dw = np.mean(np.diff(w))  # assume uniform freq. sampling
    H_2norm = np.sqrt((np.sum(S)*dw)/(2*np.pi))

    return H_2norm


plt.rcParams.update({
    "text.usetex": True
})


PDF_SEL = 1  # uniform
#PDF_SEL = 2  # binary

#PSD_SEL = 1  # PSD derived from FIR filter with random coeffs.
#PSD_SEL = 2  # PSD derived from random state-space realisation
PSD_SEL = 3  # PSD derived from moving average filter

N = 1e6  # signal length
Fs = 1e4  # assumed sampling freq.
Ts = 1/Fs

rng = np.random.default_rng()  # set up random number generator
v = rng.normal(size=int(N))  # normally distr. random sequence

# PSD for testing
match PSD_SEL:
    case 1:  # random filter alt. 1 (FIR)
        Nf = 5  # filter order
        g = rng.normal(size=int(Nf))
        S_num = np.convolve(g, np.flip(g))
        S_den = np.zeros(Nf)
        S_den[-1] = 1
    case 2:  # random filter alt. 2 (IIR)
        Nf = 5  # filter order
        #G = ct.drss(Nf, outputs=1, inputs=1)  # random, stable LTI system
        G_tf = ct.ss2tf(G)
        G_num = np.squeeze(G_tf.num)
        G_den = np.squeeze(G_tf.den)
        S_num = np.convolve(G_num, np.flip(G_num))
        S_den = np.convolve(G_den, np.flip(G_den))
    case 3:  # moving avg. filter
        Nf = 10  # filter order
        g = np.ones(Nf)/Nf
        S_num = np.convolve(g, np.flip(g))
        S_den = np.zeros(Nf)
        S_den[-1] = 1


# %% frequency response of S(omega)
M = 1024  # no. of frequency samples
# frequency response; sample whole circle
w, S_fr_ = signal.freqz(b=S_num, a=S_den, worN=M, whole=True, include_nyquist=True)

if False:  # check if response is real (usually there is some imaginary numerical residue)
    plt.plot(np.real(S_fr_))
    plt.plot(np.imag(S_fr_))
    plt.show()


# %% Compute FIR approximation to S (implicitly yields the impulse response)
S_fr = np.abs(S_fr_)  # S is symmetric so should be real, but make sure
N_fir = 256
R, R_win, R_beta = ff.fir_filter_ls(S_fr, N_fir)  # least squares FIR filter approx.
w_fir, S_fir_fr = signal.freqz(b=R_win, a=R_beta, worN=M, whole=True, include_nyquist=True)


# %% Plot S and FIR approximation
fg1, ax1 = plt.subplots()
ax1.set_title('PSD frequency response')
ax1.plot(w, 10*np.log10(np.abs(S_fr)), label='Specified PSD $S$')
ax1.plot(w_fir, 10*np.log10(abs(S_fir_fr)), linestyle='dashed', label='FIR filter approx.')
ax1.set_ylabel('Amplitude [dB]')
ax1.set_xlabel('Frequency [rad/sample]')


# %% Determining the norm/variance
match PDF_SEL:
    case 1:  # uniform; the values are as in (Sondhi, 1983)
        a = -np.sqrt(3); b = np.sqrt(3); c = 0  # lower, upper, mode 
        out_var = (b - a)**2/12
    case 2:  # binary
        out_var = (1)^2
    #case 3:  # triangular
    #    a = -1; b = 1; c = 0  # lower, upper, mode
    #    out_var = (a**2 + b**2 + c**2 - a*b - a*c - b*c)/18


# %% Compute phi
R_win_inf = np.max(np.abs(R_win))  # infinity norm
R_win_ = R_win/(R_win_inf + np.sqrt(np.finfo(np.float64).eps))  # avoid zero division

match PDF_SEL:
    case 1:  # uniform
        phi = 2*np.sin((np.pi/6)*R_win_)
    case 2:  # binary
        phi = np.sin((np.pi/2)*R_win_)
    #case 3:  # triangular
    #    phi = 0  # TODO: need to derive the expression for this case

w, Phi_fr = signal.freqz(b=phi, a=1, worN=M, whole=True, include_nyquist=True)
Phi_fr_2norm = h2_norm(Phi_fr, w)  # validation: should be 1

match 1:
    case 1:  # use FFT/IFFT to synth. H; same as (Sondhi, 1983)
        phi_ = np.roll(phi, 128)  # circular shift
        fg2, ax2 = plt.subplots()
        ax2.set_title('Impulse response')
        ax2.stem(phi_, label='$\\phi$ shifted')
        ax2.legend()

        Phi = np.fft.fft(phi_)
        
        mus = np.emath.sqrt(np.real(Phi))  # shaping filter

        fg3, ax3 = plt.subplots(2, 1, sharey=True, tight_layout=True)
        ax3[0].plot(np.real(Phi), label='$\\Phi$ real')
        ax3[0].plot(np.real(mus), label='$\\mu$ real')
        ax3[1].plot(np.imag(Phi), label='$\\Phi$ imaginary')
        ax3[1].plot(np.imag(mus), label='$\\mu$ imaginary')
        ax3[0].legend()
        ax3[1].legend()
        fg3.suptitle(t='Filter coefficients')

        h = np.fft.ifft(np.real(mus))  # real(mus) excludes real(Phi) < 0
        h = np.real(np.roll(h, 128))      
    case 2:  # use LS on frequency response to synth. H (added by AAE)
        mus = np.sqrt(abs(Phi_fr))
        h_alpha, h_alpha_win, h_beta = ff.fir_filter_ls(mus, 128)

        h = h_alpha_win

# %% Filter h, shaping/correction filter
fg4, ax4 = plt.subplots()
ax4.set_title('Impulse response')
ax4.stem(h, label='$h$')
ax4.legend()

# %% Specified PSD vs PDF corrected PSD
fg5, ax5 = plt.subplots()
ax5.set_title('Power Spectral Densities')
ax5.plot(w/(2*np.pi), 10*np.log10(np.abs(S_fr)), label='$S$ -- specified response')
ax5.plot(w/(2*np.pi), 10*np.log10(np.abs(Phi_fr)), label='$\\Phi$ -- corrected response (before PDF transform)')
ax5.legend()

# %% Specified PSD FIR repr.
fg6, ax6 = plt.subplots()
ax6.set_title('Specified PSD FIR repr. -- Impulse response')
ax6.stem(R_win_, label='$R_{win}$')
ax6.legend()

# %% PDF corrected PSD FIR repr.
fg7, ax7 = plt.subplots()
ax7.set_title('PDF corrected PSD FIR repr. -- impulse response')
ax7.stem(phi, label='$\\phi$')
ax7.legend()

# %% Generate coloured input to PDF transform
match 1:
    case 1:  # apply PDF corrected filter
        x_ = signal.lfilter(h, 1, v)
    case 2:  # apply "original" filter (for testing)
        x_ = signal.lfilter(g, 1, v)

x = x_/np.std(x_)  # normalise (should not be necessary, but it is)

# %% Transform Gaussian sequence x to obtain specified PDF for y

match PDF_SEL:
    case 1:  # uniform
        y = 2*np.sqrt(3)*(ps.norm_cdf(x) - 1/2)
    case 2:  # binary
        y = np.sign(x)
    #case 3:  # triangular
    #    u = ps.norm_cdf(x)
    #    y = ps.triang_icdf(u)

# %% Histogram for output (should match specification)
fg8, ax8 = plt.subplots()
ax8.set_title('Histogram for output')
Nbins = 100
ax8.hist(y, bins=Nbins)

# %% PSD for output (approximating the specified PSD)
fg9, ax9 = plt.subplots()

# compute and plot the PSD estimate for the output
win = signal.windows.kaiser(int(N/250), beta=10)  # implicitly defines no. of avgs. in Welch
Fy, Pyy = signal.welch(y, window=win, return_onesided=False, detrend=False, scaling='density')
I = np.argwhere(0.0 <= Fy)  # response is symmertric (enough to look at half)
ax9.plot(Fy[I], 10*np.log10(Pyy[I]), label='$P_{yy}$')

# plot the specified PSD for the output
w_S_fr_, S_fr_ = signal.freqz(b=S_num, a=S_den, whole=True)
I = np.argwhere(w_S_fr_ < np.pi) # response is symmertric (enough to look at half)
S_fr_variance = h2_norm(S_fr_, w_S_fr_)**2  # variance of S output when S driven by unity variance white noise
S_fr_var_corrected = (S_fr_/S_fr_variance)*out_var  # scale response to to match PDF variance 
ax9.plot(w_S_fr_[I]/(2*np.pi), 10*np.log10(S_fr_var_corrected[I]), label='$S$')

ax9.set_xlabel('Frequency (Hz)')
ax9.set_ylabel('Power (V$^2$/Hz)')
ax9.legend()


# %%
