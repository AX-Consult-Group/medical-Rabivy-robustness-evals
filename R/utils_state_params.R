# utils_state_params.R
#
# One config object per state (Nebraska, Wisconsin, Mississippi). This is the
# only place covariate-shift assumptions live — 01_simulate_state_data.R reads
# these and otherwise reuses the original repo's DGP code unchanged.
#
# Shifted features:
#   specialty mix, obesity prevalence, payer mix, PA burden, rep engagement
#   recency, within-specialty script volume (rural/access-constrained
#   discount), and (Mississippi only) academic engagement. Years in practice
#   and the DGP coefficients themselves are never shifted. 
#
# rx_volume_scale and rep_engagement$urban_share share one HCP-level "urban"
# draw (made once, right after specialty, in 01_simulate_state_data.R): 
# sparse rural coverage plausibly depresses both
# a rural HCP's raw GLP-1 patient volume AND how recently a rep last reached
# them, so both are driven off the same underlying indicator. 
#
# baseline_params reproduces the original repo's assumptions
# (hcp_propensity_model.qmd, V1/V4/V6/V8/V11/V13 chunks) so 01_simulate and
# 03_drift_metrics can diff a state against "no shift" as a sanity check.

baseline_params <- list(
  state = "baseline (original repo)",

  specialty_probs = c(
    "Primary Care"     = 0.82,
    "Endocrinology"     = 0.10,
    "Obesity Medicine" = 0.08
  ),

  # rbeta(shape1 = mean*9, shape2 = (1-mean)*9), truncated 0.08-0.92
  obesity_prev_mean = c(
    "Primary Care"     = 0.22,
    "Endocrinology"     = 0.42,
    "Obesity Medicine" = 0.68
  ),

  # rmultinom probs per specialty; each vector sums to 1
  payer_probs = list(
    "Primary Care"     = c(Commercial = 0.30, Medicare = 0.40, Medicaid = 0.20, OOP = 0.10),
    "Endocrinology"     = c(Commercial = 0.50, Medicare = 0.30, Medicaid = 0.12, OOP = 0.08),
    "Obesity Medicine" = c(Commercial = 0.55, Medicare = 0.18, Medicaid = 0.07, OOP = 0.20)
  ),

  # rbeta(shape1, shape2) per dominant payer; OOP fixed at 0
  pa_params = list(
    Commercial = c(shape1 = 4,   shape2 = 6),
    Medicare   = c(shape1 = 5.5, shape2 = 4.5),
    Medicaid   = c(shape1 = 6,   shape2 = 4),
    OOP        = c(shape1 = 0,   shape2 = 0)
  ),

  # rexp(rate = 1/mean_days); single population, no urban/rural mixture
  rep_engagement = list(
    urban_share              = 1.0,   # no rural/urban split at baseline
    targeted_days_mean       = 30,
    nontargeted_days_mean    = 120,
    urban_targeted_days_mean = 30,
    urban_nontargeted_days_mean = 120
  ),

  # Multiplier on mu_by_specialty (V2), by the same urban/rural draw used
  # above. No reduction at baseline — original repo assumed one undifferentiated
  # population.
  rx_volume_scale = list(urban = 1.0, rural = 1.0),

  # rpois(lambda) + rbinom(size, prob), capped at 12
  academic_engagement = list(
    lambda      = 1.8,
    binom_size  = 4,
    binom_prob  = 0.25
  )
)

state_params <- list(

  # Nebraska: full Medicaid expansion but work-requirement
  # enforcement begins May 2026, adding access friction on top of nominal
  # coverage. Rural/plains, low specialist density, obesity ~35%+, sparse
  # rural rep coverage (stale engagement).
  Nebraska = list(
    state = "Nebraska",

    specialty_probs = c(
      "Primary Care"     = 0.88,
      "Endocrinology"     = 0.07,
      "Obesity Medicine" = 0.05
    ),

    obesity_prev_mean = c(
      "Primary Care"     = 0.36,
      "Endocrinology"     = 0.48,
      "Obesity Medicine" = 0.72
    ),

    payer_probs = list(
      "Primary Care"     = c(Commercial = 0.28, Medicare = 0.40, Medicaid = 0.22, OOP = 0.10),
      "Endocrinology"     = c(Commercial = 0.47, Medicare = 0.31, Medicaid = 0.14, OOP = 0.08),
      "Obesity Medicine" = c(Commercial = 0.52, Medicare = 0.19, Medicaid = 0.09, OOP = 0.20)
    ),

    # Moderate PA burden overall; Medicaid nudged up to reflect new
    # work-requirement administrative friction layered on top of expansion.
    pa_params = list(
      Commercial = c(shape1 = 4,   shape2 = 6),
      Medicare   = c(shape1 = 5.5, shape2 = 4.5),
      Medicaid   = c(shape1 = 6.5, shape2 = 3.5),
      OOP        = c(shape1 = 0,   shape2 = 0)
    ),

    # Sparse rural coverage means most HCPs are effectively
    # "rural" (low urban_share) and even targeted contact cadence slips.
    rep_engagement = list(
      urban_share              = 0.15,
      targeted_days_mean       = 45,
      nontargeted_days_mean    = 150,
      urban_targeted_days_mean = 30,
      urban_nontargeted_days_mean = 120
    ),

    # Rural HCPs write ~75% of the baseline specialty-conditional volume:
    # full Medicaid expansion still gives reasonable underlying access, so
    # the discount is moderate rather than severe.
    rx_volume_scale = list(urban = 1.0, rural = 0.75),

    academic_engagement = list(
      lambda      = 1.8,
      binom_size  = 4,
      binom_prob  = 0.25
    )
  ),

  # Wisconsin: partial Medicaid expansion only (covers to 100% FPL, no
  # coverage gap but a real 100-138% FPL band that leans on commercial/
  # marketplace coverage instead). Mixed urban (Milwaukee, Madison) / rural.
  # Obesity ~35%+. More balanced specialty mix in urban cores.
  Wisconsin = list(
    state = "Wisconsin",

    specialty_probs = c(
      "Primary Care"     = 0.74,
      "Endocrinology"     = 0.15,
      "Obesity Medicine" = 0.11
    ),

    obesity_prev_mean = c(
      "Primary Care"     = 0.35,
      "Endocrinology"     = 0.46,
      "Obesity Medicine" = 0.70
    ),

    # Medicaid share down, Commercial up: the 100-138% FPL band that would
    # be Medicaid-eligible under full expansion instead relies on
    # commercial/marketplace coverage.
    payer_probs = list(
      "Primary Care"     = c(Commercial = 0.38, Medicare = 0.38, Medicaid = 0.14, OOP = 0.10),
      "Endocrinology"     = c(Commercial = 0.57, Medicare = 0.28, Medicaid = 0.07, OOP = 0.08),
      "Obesity Medicine" = c(Commercial = 0.62, Medicare = 0.16, Medicaid = 0.03, OOP = 0.19)
    ),

    # Moderate-high PA burden across the board.
    pa_params = list(
      Commercial = c(shape1 = 4.5, shape2 = 5.5),
      Medicare   = c(shape1 = 5.5, shape2 = 4.5),
      Medicaid   = c(shape1 = 6.5, shape2 = 3.5),
      OOP        = c(shape1 = 0,   shape2 = 0)
    ),

    # Mixed: a real urban cluster (Milwaukee/Madison) gets baseline-level
    # engagement recency, the rest of the state behaves closer to Nebraska.
    rep_engagement = list(
      urban_share              = 0.45,
      targeted_days_mean       = 40,
      nontargeted_days_mean    = 140,
      urban_targeted_days_mean = 25,
      urban_nontargeted_days_mean = 90
    ),

    # Rural WI HCPs write ~85% of baseline volume — the mildest discount of
    # the three states, consistent with "mixed urban/rural" rather than
    # "predominantly rural."
    rx_volume_scale = list(urban = 1.0, rural = 0.85),

    academic_engagement = list(
      lambda      = 1.8,
      binom_size  = 4,
      binom_prob  = 0.25
    )
  ),

  # Mississippi: no Medicaid expansion — genuine coverage gap for low-income
  # adults. Highest obesity prevalence in the country (~40%+). Lowest
  # specialist density, heaviest commercial/self-pay reliance, highest PA
  # burden, stalest rep engagement (largest rural share), and slightly lower
  # academic engagement proxy. 
  Mississippi = list(
    state = "Mississippi",

    specialty_probs = c(
      "Primary Care"     = 0.90,
      "Endocrinology"     = 0.06,
      "Obesity Medicine" = 0.04
    ),

    obesity_prev_mean = c(
      "Primary Care"     = 0.40,
      "Endocrinology"     = 0.52,
      "Obesity Medicine" = 0.75
    ),

    # No expansion: Medicaid share drops, OOP/self-pay absorbs the
    # coverage gap population; Commercial holds roughly steady.
    payer_probs = list(
      "Primary Care"     = c(Commercial = 0.35, Medicare = 0.35, Medicaid = 0.10, OOP = 0.20),
      "Endocrinology"     = c(Commercial = 0.53, Medicare = 0.27, Medicaid = 0.06, OOP = 0.14),
      "Obesity Medicine" = c(Commercial = 0.55, Medicare = 0.16, Medicaid = 0.04, OOP = 0.25)
    ),

    # Highest PA burden of the three states across every payer channel.
    pa_params = list(
      Commercial = c(shape1 = 5,  shape2 = 5),
      Medicare   = c(shape1 = 6,  shape2 = 4),
      Medicaid   = c(shape1 = 7,  shape2 = 3),
      OOP        = c(shape1 = 0,  shape2 = 0)
    ),

    # Stalest of the three: largest rural share, lowest urban_share, longest
    # days-since-contact for both targeted and non-targeted HCPs.
    rep_engagement = list(
      urban_share              = 0.08,
      targeted_days_mean       = 55,
      nontargeted_days_mean    = 170,
      urban_targeted_days_mean = 30,
      urban_nontargeted_days_mean = 120
    ),

    # Steepest discount of the three states: rural MS HCPs write ~55% of
    # baseline volume, reflecting no Medicaid expansion, the largest rural
    # share, and the lowest specialist density compounding on raw patient
    # access.
    rx_volume_scale = list(urban = 1.0, rural = 0.55),

    # Slightly lower academic engagement proxy 
    # (lower publication/conference-activity rate).
    academic_engagement = list(
      lambda      = 1.3,
      binom_size  = 4,
      binom_prob  = 0.15
    )
  )
)

get_state_params <- function(state) {
  valid_states <- names(state_params)
  if (!state %in% valid_states) {
    stop("Unknown state '", state, "'. Must be one of: ",
         paste(valid_states, collapse = ", "))
  }
  state_params[[state]]
}
