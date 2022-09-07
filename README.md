# PAHDecomp

## Overview
PAHDecomp ([Donnan et al. 2022]()) is a tool for modelling mid-infrared spectra of galaxies based on the popular PAHFIT code [Smith et al. 2007](https://ui.adsabs.harvard.edu/abs/2007ApJ...656..770S/abstract). Unlike PAHFIT this model decomposes the continuum into a star-forming component and an obscured nuclear component based on Bayesian priors on the shape of the star-forming component (using templates + prior on extinction). This makes this tool ideally suited for modelling the spectra of heavily obscured galaxies.

The fit is achieved using MCMC sampling from [NUMPYRO](https://github.com/pyro-ppl/numpyro). Specifically we use the NUTS sampler which is a Hamiltonian Monte Carlo allowing parallelisation to speed up computation time. This can even by ran on GPUs or TPUs. 

An example fit is shown below 
![alt text](./MockDataFig.png?raw=true)


## Installation



## Running the Code

This code can be ran in two ways. Either by simply modelling a single spectrum of a galaxy or modelling multiple spectra simultaneously, extracted from different sized apertures centered on a given nucleus where the nuclear component is shared by each spectrum. 

### Changing the extinction or nuclear silicate template
To run the model with a different extinction law for the star-forming component add the file to the 'Extinction Curves' folder and at the top of 'SetupFit.py' make sure the file is read in. Similarly to change the silicate profile for the nuclear component, add the file to the 'Nuclear Templates' folder and make sure it is read in in the 'SetupFit.py' file. The ice/CH templates can also be changed in the 'Ice Templates' folder.


## Output

The output from the fit will be written to the 'Results' directory which will contain
 - A plot of the fit
 - .txt files containing the constituent parts of the model
 - NuclearParameters.csv contains the parameters defining the shape of the nuclear component such as the nuclear optical depth
 - Output.csv contains the measured strengths and equivalent widths of the emission features. The strengths displayed for each of the line and PAH features are corrected by the extinction on the star-forming component. The PAH flux ratios given towards the end of the .csv file are the observed flux ratios and are not corrected by extinction.


## Citation
If you use this code please cite ([Donnan et al. 2022]()):

