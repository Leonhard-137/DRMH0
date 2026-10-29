# Completed shared-S recovery transfer functions

Six-case order: raw, both sigma5, both sigma10, both sigma20, both sigma40,
U-only sigma20. Both models use all existing posterior draws, reordered by
centroid within each joint draw: C0 has smaller centroid; C1 has larger.
Black: pointwise total median, after summing both components within each draw.
Gray: total pointwise 15.85--84.15 percentiles. Blue/red: component pointwise
medians; their sum need not equal the total pointwise median.

Log response axes show weak long tails. Each panel uses the same full lag
limits as its existing case-local transfer_function plot (central 99.9% mass
of the posterior-averaged unit-area total response); axes differ by panel.
Native normalized-flux response amplitudes are retained without rescaling.
All amplitudes are constrained positive. Response at negative lag is not
negative response amplitude. No physical input kernel is overlaid because
the central-source response and observed W2-to-U predictor are distinct.

Existing original linear/log figures remain in CASE/results/MODEL/.
u_s20/Gamma has only 45 posterior draws (ESS about 45.9); this plot does not
establish convergence. No fitting, input regeneration or prior changes occur.
