# Fit_Model_to_Datacube

This script is used to fit the emission and absorption line profiles found in Gemini GMOS IFU datacubes of three candidates with possible double supermassive blackholes. The script loops through each spatial pixel (spaxel) and uses the MPFIT package to fit Gaussian profiles to the NII + Halpha, SII, OIII, and Hbeta emission lines and the Na ID absorption lines. 

Examples of line profile fits are shows below: 

<img src="https://github.com/hsshih/Fit_Model_to_Datacube/blob/main/Line_Fit_Examples.png" class="center" width="800" height="300" />

The script also uses the fitted line profiles to produce line ratios and BPT diagrams to help determine the ionization state of the gas. An example is shown below:

<img src="https://github.com/hsshih/Fit_Model_to_Datacube/blob/main/SDSSJ151505_bpt.png" class="center" width="800" height="600" />
