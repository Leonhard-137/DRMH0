# Sigma5 U compared with the matched U without independent S

The black curve in panel (a) is original U minus its saved independent slow
component. It equals the existing abl_disk_blr input to text-file precision.
The red curve is the actual sigma5 input already used for both MICA models.
Both share the same underlying disk, BLR and measurement-noise realization.
No curves are independently rescaled or recentered for this figure.

Panel (b) shows red minus black and the residual independent slow component
for scale. The difference contains removed disk/BLR variability, the change
in the same measurement-noise realization, and residual S. It is not a new
independent-noise residual, so no reduced chi-squared is calculated.

RMS means the unweighted standard deviation at the saved observation epochs,
not mean flux, transfer-function area or an error-corrected variance estimate.
Component columns permit a direct calculation of retained disk, BLR, S and
noise RMS. No fitting or input modifications are performed by this script.

Reproduce with ../../../../plot_sigma5_vs_clean_u.py using the astro Python.
