# Fixed covariance regularization of a frozen proposal

Create a new proposal candidate from the existing 12+12 component refined
importance proposal. No EM retraining, GP likelihood calls, scientific fit,
model/target/prior change, or alteration of prior artifacts is authorized.

For each base/tail block separately, recover training Gaussian covariance by
dividing Student shape by inflation^2*(df-2)/df. Compute the block total
covariance using normalized within-block mixture weights, including the
between-component mean term. With dimension d and recorded component
responsibility ESS_k, fix alpha_k=(d+2)/(ESS_k+d+2) and shrink each training
covariance toward its block total covariance. Convert to Student shape using
the original df=7 and SD inflation=1.2. Preserve means, full component weights,
and epsilon=.05. The d+2 heuristic modifies the proposal only.

Save three frozen NPZ proposals, covariance audit arrays, a numerical validation
report and a short explanation. Check positive definiteness, normalization,
preserved values and density agreement. Compare log q on existing high-weight
training rows only as density placement, without calculating new scientific
importance estimates or claiming improved efficiency. An independent agent
will audit the finite calculations. All numerical work uses float64.
