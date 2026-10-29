# Randomized QMC component-stratified importance prototype

Implement and validate a standalone candidate using the existing frozen
regularized proposal and scientific target unchanged. This task permits
analytic toy callbacks and proposal-only calculations, with no GP likelihood,
scientific pilot or official stability gate run.

For each conditional center stratum, each Student-mixture component and the
defensive prior component gets Nk=2**max(3,ceil(log2(target_N*alpha_k))) points
per independent scramble, target_N=2048 and R=8. Use scrambled Sobol bits=52,
normal and chi-square inverse transforms for Student components and logit for
the defensive component. Preserve alpha_k/Nk, full mixture density q, 1/R,
and the center prior mass rho. Never clip uniform endpoints or silently redraw.
Natural expit saturation follows the existing explicit continuous-extension
policy. Preserve every row and its component/scramble/stratum identity.

Estimate uncertainty from independent scramble integrals and their covariance,
using the ratio delta method for posterior means. Do not use iid row MCSE.
Validate inverse maps, unequal-Nk constants, complete prior strata, a known
bimodal target and multiple independent randomizations. Save fixed settings,
seed hierarchy, exact budgets and the finite numerical evidence boundary.
