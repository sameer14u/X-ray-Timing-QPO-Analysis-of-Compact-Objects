X-ray Timing QPO Analysis of Compact Objects.
Currently most of the files are empty as they are under construction for showcasing, project has been completed i just slowly constructing the representation here.
I have uploaded all 6he function and code commands in one python code file.
I have uploaded 4 sample data, that were used in actual project.
We will have to use algorithm according to data, as how many peak will be visibble in graph. Accordig to that we will decide where to use power law, lorentz algo , power law + lorentz and similarly power law ++..








Short Description : X-ray-Timing-QPO-Analysis-of-Compact-Objects is a Python-based tool for analyzing quasi-periodic oscillations (QPOs) in X-ray light curves from compact objects like black holes and neutron stars. The script loads FITS-format light curve files using Astropy, computes averaged power density spectra (PDS) via FFT on segmented data with Leahy normalization, applies logarithmic rebinning for noise reduction, and performs non-linear least-squares fitting with SciPy to models including power-law continuum plus one or two Lorentzians to characterize QPO centroids, widths, and quality factors (Q = f0/width). Interactive input for fit ranges and initial parameters enables parameter estimation with uncertainties, χ² statistics, and visualization of light curves, PDS, fits, and residuals, aiding research in accretion disk dynamics and timing properties.
