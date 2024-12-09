#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Examples of a method for generating arbitrary continuous probability density functions.

This script demonstrates the basics the general method for transforming stochastic
variables between continuous distributions.

[1] D. E. Knuth, The Art of Computer Programming, 3rd ed., vol. 2.
Reading, Massachusetts: Addison-Wesley, 1997.

Summarily, we can compute a random variable X with distribution F(x) by setting

X = F^-1(U)

where with F(x) the "desired" cumulative distribution function (CDF) and U uniformly
distributed variable, and F^-1(U) is the inverse CDF.

@author: Arnfinn A. Eielsen
@date: 05.12.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import math

def norm_cdf(x, mu=0, sigma=1):
    # Normal cumulative distribution function
    # https://en.wikipedia.org/wiki/Normal_distribution

    y = (1/2)*(1 + special.erf((x - mu)/(sigma*math.sqrt(2))))

    return y


def norm_icdf(y, mu=0, sigma=1):
    # Normal inverse cumulative distribution function

    if np.max(y) > 1.0:
        print('Warning: Invalid input range.')
    if np.min(y) < 0.0:
        print('Warning: Invalid input range.')
    
    x = special.erfinv(2*y - 1)*sigma*math.sqrt(2) + mu

    return x


def triang_cdf(x, a=-1, b=1, c=0):
    # Triangular cumulative distribution function
    # https://en.wikipedia.org/wiki/Triangular_distribution

    y = np.zeros(x.size)

    k1 = (b - a)*(c - a)
    k2 = (b - a)*(b - c)

    I = np.argwhere(x <= a)
    y[I] = 0
    I = np.argwhere((x > a) & (x <= c))
    y[I] = (x[I] - a)**2/k1
    I = np.argwhere((x > c) & (x <= b))
    y[I] = 1 - (b - x[I])**2/k2
    I = np.argwhere(x > b)
    y[I] = 1

    return y

def triang_icdf(y, a=-1, b=1, c=0):
    # Triangular inverse cumulative distribution function

    if np.max(y) > 1.0:
        print('Warning: Invalid input range.')
    if np.min(y) < 0.0:
        print('Warning: Invalid input range.')

    x = np.ones(y.size)*np.nan  # y values ouside defined range assigned NaN

    m = (c - a)/(b - a)
    k1 = (b - a)*(c - a)
    k2 = (b - a)*(b - c)

    I = np.argwhere((y >= 0.0) & (y <= m))
    x[I] = np.sqrt(y[I])*np.sqrt(k1) + a
    I = np.argwhere((y > m) & (y <= 1.0))
    x[I] = -np.sqrt(1 - y[I])*np.sqrt(k2) + b

    return x


# %% Plot CDFs

if False:
    v = np.linspace(-5, 5, 1000)  # 
    w_norm = norm_cdf(v, 0.5, 2.0)

    fg1, ax1 = plt.subplots()
    ax1.plot(v, w_norm)
    ax1.set(title='Normal CDF')
    ax1.grid()
    plt.show()

    v = np.linspace(0, 1, 1000)  # 
    w_inorm = norm_icdf(v, 0.5, 2.0)

    fg2, ax2 = plt.subplots()
    ax2.plot(v, w_inorm)
    ax2.set(title='Normal ICDF')
    ax2.grid()
    plt.show()

    v = np.linspace(-2, 2, 1000)  # 
    w_triang = triang_cdf(v, -2, 3, 1.5)

    fg3, ax3 = plt.subplots()
    ax3.plot(v, w_triang)
    ax3.set(title='Triangular CDF')
    ax3.grid()
    plt.show()

    v = np.linspace(0, 1, 1000)  # 
    w_itriang = triang_icdf(v, -2, 3, 1.5)

    fg4, ax4 = plt.subplots()
    ax4.plot(v, w_itriang)
    ax4.set(title='Triangular ICDF')
    ax4.grid()
    plt.show()

# %% Example: Transform normal variable to uniform

N = 1e5  # number of samples

rng = np.random.default_rng()  # set up random number generator
x = rng.normal(size=int(N))

u = norm_cdf(x)

fg5, ax5 = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
Nbins = 100
ax5[0].hist(x, bins=Nbins)
ax5[1].hist(u, bins=Nbins)
fg5.suptitle(t='Transforming normal to uniform.')

plt.show()


# %% Example: Transform uniform variable to normal

N = 1e5  # number of samples

rng = np.random.default_rng()  # set up random number generator
x = rng.uniform(size=int(N))

u = norm_icdf(x)

fg6, ax6 = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
Nbins = 100
ax6[0].hist(x, bins=Nbins)
ax6[1].hist(u, bins=Nbins)
fg6.suptitle(t='Transforming uniform to normal.')

plt.show()

# %% Example: Transform normal variable to triangular

N = 1e5  # number of samples

rng = np.random.default_rng()  # set up random number generator
x = rng.normal(size=int(N))

v = norm_cdf(x)
u = triang_icdf(v)

fg7, ax7 = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
Nbins = 100
ax7[0].hist(x, bins=Nbins)
ax7[1].hist(u, bins=Nbins)
fg7.suptitle(t='Transforming normal to triangular.')

plt.show()

# %%
