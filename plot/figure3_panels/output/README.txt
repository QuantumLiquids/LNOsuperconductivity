Figure 3 panels — spin-spin correlation bubble plots
=====================================================

All panels share a single global bubble scale:
  global_max |S·S| = 0.407
  BASE_MARKER_SIZE = 220 pt²  (marker area at global_max)
  INCHES_PER_BOND  = 0.22     (physical size per lattice bond)

Colors (same as Figures 3/4 in the paper):
  Positive (FM):  RGB = (142, 139, 254)/256  purple
  Negative (AFM): RGB = (232, 132, 130)/256  red

Star = reference site for the correlation measurement.

Illustrator: place all files at 100% scale (no zoom/resize).
Bubbles are directly comparable across all panels.

----------------------------------------------------------------------
Single-layer panels (Ly=4, no interlayer coupling)
----------------------------------------------------------------------
Parameters: Lx=20, Ly=4, t'=0.3t, J_H=4t, U=14t, D=18000, OBC

  sl_Ly4_itinerant    — itinerant d_{x²-y²} spin correlation
  sl_Ly4_localized    — localized d_{z²} spin correlation

----------------------------------------------------------------------
Double-layer panels (Ly=2, with interlayer coupling)
----------------------------------------------------------------------
Parameters: Lx=20, Ly=2, t'=0.3t, J_H=4t, U=14t, J_perp=0.1t, D=20000, OBC

  dl_Ly2_loc_L0_intra     — localized d_{z²}, layer 0 intralayer
  dl_Ly2_loc_L1_intra     — localized d_{z²}, layer 1 intralayer
  dl_Ly2_loc_interlayer   — localized d_{z²}, interlayer (ref on L0, target on L1)
  dl_Ly2_elec_L0_intra    — itinerant d_{x²-y²}, layer 0 intralayer
  dl_Ly2_elec_L1_intra    — itinerant d_{x²-y²}, layer 1 intralayer (= L0 by symmetry)
  dl_Ly2_elec_interlayer  — itinerant d_{x²-y²}, interlayer (ref on L0, target on L1)

Note: L0 and L1 intralayer correlations are identical to ~1e-4 by interlayer
symmetry (verified numerically for localized spins). The electron L1 panel
reuses the L0 data since only L0 was measured explicitly.

----------------------------------------------------------------------
Legend
----------------------------------------------------------------------
  legend  — standalone bubble legend bar for the shared scale
