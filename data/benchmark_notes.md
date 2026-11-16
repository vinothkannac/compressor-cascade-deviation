# Benchmark row: how it was obtained (read this before quoting it)

Source: NACA RM L51G31 (Systematic two-dimensional cascade tests of NACA 65-series compressor blades at low speeds,
1951; consolidated as NACA Report 1368, 1958), NTRS 19930092353. Fetched from ntrs.nasa.gov and read from the scan.
The Report 1368 PDF itself was not available here; RM L51G31 holds the same test data.

Case: blade 65-(12)10, solidity 1.00, beta1 = 45 deg, Figure 30(g) "Section characteristics".
- Stagger 30 deg => angle of attack alpha1 (angle between flow and chord, report symbol list) = 45 - 30 = 15.0 deg.
- incidence_deg = alpha1 - 0.5 * 25.3 * 1.2 = -0.18 deg (camber 25.3 deg x cl0 is my assumption;
  it only affects the incidence LABEL, which compare_benchmark.py matches against the case).
- The report's design angle of attack for this cascade is about 12 deg (arrow in the figure), so this operating
  point is 3 deg above design, on the rising part of the turning curve.

Values read off the plot (gridlines calibrated, interpolated between measured points at alpha1 = 14.1 and 16.1):
- turning theta (report: flow turning angle beta2 - beta1) = 22.2 deg, reading uncertainty about +-0.3 deg
- drag coefficient Cd1 = 0.0136 (wake coefficient Cw1 = 0.0159), uncertainty about +-0.0006
These are READ OFF A PLOT BY THE ASSISTANT, not tabulated numbers.

loss_coefficient = 0.0167 is a CONVERSION, not a reported number:
  omega = Cd1 * sigma / cos(beta_m),  tan(beta_m) = (tan(beta1) + tan(beta2)) / 2,  beta2 = 45 - 22.2 = 22.8 deg
  => cos(beta_m) = 0.8153, omega = 0.0136 / 0.8153 = 0.0167
(Cd1 is the force coefficient parallel to the vector-mean velocity, based on q1 and chord; the relation to the
total-pressure loss (p01 - p02)/q1 is my derivation from momentum balance, please check it. Using Cw1 instead
would give 0.0195.)

Differences between experiment and this CFD (error sources to list in the README):
- Standard test Reynolds number 245 000 (confirmed in the report); laminar separation on the convex surface
  persisted up to this Re in the tunnel, the CFD is fully turbulent.
- Wake total-pressure rake about 1 chord downstream of the TE; CFD plane is 0.5 axial chord downstream.
- Report states drag values near design varied up to 30 percent with Reynolds number and leading-edge roughness.
- Tested blades used the "65(216)-010 combined with y = 0.0015x" thickness form (data/naca65_216_010_combined_thickness.csv);
  the default CFD blade uses the basic 65-010 form plus a 0.0015 x trailing-edge closure.
- Porous-wall tunnel, finite aspect ratio, measurement uncertainty.
