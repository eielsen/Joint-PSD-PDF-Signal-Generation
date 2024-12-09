# Methods for generating signals with jointly specified PSD and PDF

The goal of this repository is to publish methods for generating (pseudorandom) signals for where both the power spectral density (PSD) and the probabilty density function (PDF) have been specified.

It is well-know how to generate normally distributed noise-like signals with arbitrary PSD, and similarly how to transform a white (flat PSD) noise-like signal to a large class of PDFs.

Generating signals where both the PSD and PDF are specified simultaneously is a bit more challenging. Two main approaches:
1. Analytical non-linear transforms
2. Numerical manipulation using stochastic optimisation

