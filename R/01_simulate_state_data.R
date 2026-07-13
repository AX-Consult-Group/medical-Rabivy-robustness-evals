# 01_simulate_state_data.R

# Standardized (_z) predictors are centered/scaled using the ORIGINAL
# population's mean/sd (models/baseline_standardization.rds, written by
# 00_load_frozen_models.R) — never the state data's own mean/sd. This is
# required both to feed the frozen models features on the same scale they
# were trained on, and to keep the ground-truth ranking of drivers meaningful
# for the true_logit formula below (its coefficients were calibrated against
# the original population's scale).

library(pacman)
p_load(tidyverse)

source("R/utils_state_params.R")

baseline_std <- readRDS("models/baseline_standardization.rds")

zscore <- function(x, feature) {
  (x - baseline_std[[feature]]$mean) / baseline_std[[feature]]$sd
}

simulate_state_data <- function(state, n = 5000, seed) {

  params <- get_state_params(state)

  set.seed(seed)

  # ---- V1: HCP Specialty --- SHIFTED: specialty mix ----
  specialty <- sample(
    names(params$specialty_probs), size = n, replace = TRUE,
    prob = params$specialty_probs
  )

  hcp_df <- data.frame(
    hcp_id    = sprintf("%s%04d", toupper(substr(state, 1, 3)), 1:n),
    specialty = specialty,
    stringsAsFactors = FALSE
  )

  # ---- Urban/rural draw --- SHIFTED: drives both V2 (volume) and V8 (rep
  # engagement) below, since sparse rural coverage plausibly depresses both
  # a rural HCP's raw patient volume and how recently a rep last reached
  # them. Drawn once here so both effects are consistent for a given HCP,
  # not independently coincidental.
  hcp_df$urban <- rbinom(n, size = 1, prob = params$rep_engagement$urban_share)

  # ---- V2: Monthly GLP-1 Rx Volume --- SHIFTED: rural/access-constrained
  # discount on top of the original specialty-conditional means ----
  hcp_df$zero_writer <- as.logical(
    rbinom(n, size = 1, prob = ifelse(hcp_df$specialty == "Primary Care", 0.6, 0))
  )
  mu_by_specialty <- ifelse(
    hcp_df$specialty == "Obesity Medicine", 60,
    ifelse(hcp_df$specialty == "Endocrinology", 35, 12)
  )
  volume_scale <- with(params$rx_volume_scale, ifelse(hcp_df$urban == 1, urban, rural))
  rx_volume_raw <- rnbinom(n, mu = mu_by_specialty * volume_scale, size = 1.5)
  hcp_df$rx_volume_monthly <- ifelse(hcp_df$zero_writer, 0, rx_volume_raw)

  # ---- V3: NRx Share --- unchanged ----
  nrx_share_raw <- rbeta(n, shape1 = 2, shape2 = 6)
  hcp_df$nrx_share   <- ifelse(hcp_df$zero_writer, 0, nrx_share_raw)
  hcp_df$nrx_monthly <- round(hcp_df$rx_volume_monthly * hcp_df$nrx_share)

  # ---- V4: Payer Mix --- SHIFTED: payer mix / Medicaid GLP-1 access ----
  payer_mix <- t(sapply(hcp_df$specialty, function(sp) {
    rmultinom(1, size = 100, prob = params$payer_probs[[sp]]) / 100
  }))
  colnames(payer_mix) <- c("pct_commercial", "pct_medicare", "pct_medicaid", "pct_oop")
  hcp_df <- cbind(hcp_df, payer_mix)

  payer_cols  <- c("pct_commercial", "pct_medicare", "pct_medicaid", "pct_oop")
  payer_names <- c("Commercial", "Medicare", "Medicaid", "OOP")
  hcp_df$dominant_payer <- payer_names[apply(hcp_df[, payer_cols], 1, which.max)]

  # ---- V5: Formulary Coverage --- unchanged (conditional on dominant payer) ----
  formulary_probs <- list(
    Commercial = c(Preferred = 0.35, NonPreferred = 0.35, PARequired = 0.25, NotCovered = 0.05),
    Medicare   = c(Preferred = 0.05, NonPreferred = 0.15, PARequired = 0.30, NotCovered = 0.50),
    Medicaid   = c(Preferred = 0.03, NonPreferred = 0.10, PARequired = 0.22, NotCovered = 0.65),
    OOP        = c(Preferred = 1.00, NonPreferred = 0,    PARequired = 0,    NotCovered = 0)
  )
  hcp_df$formulary_tier <- sapply(hcp_df$dominant_payer, function(p) {
    sample(names(formulary_probs[[p]]), size = 1, prob = formulary_probs[[p]])
  })

  # ---- V6: Prior Authorization Burden --- SHIFTED: PA burden ----
  hcp_df$pa_burden <- mapply(function(payer) {
    if (payer == "OOP") return(0)
    p <- params$pa_params[[payer]]
    rbeta(1, shape1 = p["shape1"], shape2 = p["shape2"])
  }, hcp_df$dominant_payer)

  # ---- V7: Existing AXPharmaceuticals Relationship --- unchanged ----
  AXPharmaceuticals_probs <- list(
    "Primary Care"     = c(None = 0.40, One = 0.38, TwoPlus = 0.22),
    "Endocrinology"     = c(None = 0.35, One = 0.40, TwoPlus = 0.25),
    "Obesity Medicine" = c(None = 0.65, One = 0.25, TwoPlus = 0.10)
  )
  hcp_df$AXPharmaceuticals_relationship <- sapply(hcp_df$specialty, function(sp) {
    sample(names(AXPharmaceuticals_probs[[sp]]), size = 1, prob = AXPharmaceuticals_probs[[sp]])
  })

  # ---- V8: Rep Engagement Recency --- SHIFTED: engagement recency ----
  # Targeting logic (volume + relationship) is unchanged; only the
  # days-since-contact distribution shifts, via an urban/rural mixture that
  # captures "mixed, urban clusters more recent" (Wisconsin) vs. uniformly
  # stale rural coverage (Nebraska, Mississippi).
  volume_pctile <- rank(hcp_df$rx_volume_monthly) / n
  relationship_weight <- ifelse(
    hcp_df$AXPharmaceuticals_relationship == "TwoPlus", 1.0,
    ifelse(hcp_df$AXPharmaceuticals_relationship == "One", 0.5, 0)
  )
  targeting_score <- 0.6 * volume_pctile + 0.4 * relationship_weight
  targeting_prob  <- targeting_score / max(targeting_score) * 0.7
  hcp_df$targeted <- rbinom(n, size = 1, prob = targeting_prob)

  # hcp_df$urban was already drawn above, right after V1, so it's shared
  # with the V2 rural volume discount rather than redrawn independently here.
  days_mean <- with(params$rep_engagement, ifelse(
    hcp_df$urban == 1,
    ifelse(hcp_df$targeted == 1, urban_targeted_days_mean, urban_nontargeted_days_mean),
    ifelse(hcp_df$targeted == 1, targeted_days_mean, nontargeted_days_mean)
  ))
  hcp_df$days_since_contact   <- rexp(n, rate = 1 / days_mean)
  hcp_df$rep_engagement_score <- 0.97 ^ hcp_df$days_since_contact

  # ---- V9: Competitor Brand Mix --- unchanged ----
  brand_probs <- c(NovoNordisk = 0.54, EliLilly = 0.35, Other = 0.11)
  brand_mix <- t(sapply(1:n, function(i) rmultinom(1, size = 100, prob = brand_probs) / 100))
  colnames(brand_mix) <- c("pct_novo", "pct_lilly", "pct_other_brand")
  hcp_df <- cbind(hcp_df, brand_mix)
  brand_cols  <- c("pct_novo", "pct_lilly", "pct_other_brand")
  brand_names <- c("Novo Nordisk", "Eli Lilly", "Other")
  hcp_df$dominant_competitor <- brand_names[apply(hcp_df[, brand_cols], 1, which.max)]

  # ---- V10: Years in Practice --- unchanged (no assumed shift, any state) ----
  hcp_df$years_practice <- pmax(5, pmin(40, round(rnorm(n, mean = 18, sd = 8))))

  # ---- V11: Obesity Prevalence --- SHIFTED: obesity prevalence ----
  ob_prev_mean <- unname(params$obesity_prev_mean[hcp_df$specialty])
  hcp_df$obesity_prev <- rbeta(n, shape1 = ob_prev_mean * 9, shape2 = (1 - ob_prev_mean) * 9)
  hcp_df$obesity_prev <- pmax(0.08, pmin(0.92, hcp_df$obesity_prev))

  # ---- V12: Recent Sample Request --- unchanged (driven by targeting + volume) ----
  sample_prob <- pmax(0.08, pmin(0.82,
    0.22 + 0.35 * hcp_df$targeted + 0.28 * scale(hcp_df$rx_volume_monthly)[, 1]))
  hcp_df$sample_request_recent <- rbinom(n, 1, prob = sample_prob)

  # ---- V13: Academic Engagement --- SHIFTED only for Mississippi ----
  academic_raw <- rpois(n, lambda = params$academic_engagement$lambda) +
    rbinom(n, size = params$academic_engagement$binom_size,
           prob = params$academic_engagement$binom_prob)
  hcp_df$academic_engagement <- pmin(12, academic_raw)

  # ---- Standardize against the frozen ORIGINAL population constants ----
  formulary_score <- case_when(
    hcp_df$formulary_tier == "Preferred"    ~  2.0,
    hcp_df$formulary_tier == "NonPreferred" ~  0.8,
    hcp_df$formulary_tier == "PARequired"   ~ -0.6,
    TRUE ~ -1.8
  )
  AXPharmaceuticals_relationship_num <- ifelse(
    hcp_df$AXPharmaceuticals_relationship == "TwoPlus", 2,
    ifelse(hcp_df$AXPharmaceuticals_relationship == "One", 1, 0)
  )

  hcp_df$rx_volume_z                      <- zscore(hcp_df$rx_volume_monthly, "rx_volume_monthly")
  hcp_df$nrx_share_z                      <- zscore(hcp_df$nrx_share, "nrx_share")
  hcp_df$obesity_prev_z                   <- zscore(hcp_df$obesity_prev, "obesity_prev")
  hcp_df$pa_burden_z                      <- zscore(hcp_df$pa_burden, "pa_burden")
  hcp_df$rep_engagement_z                 <- zscore(hcp_df$rep_engagement_score, "rep_engagement_score")
  hcp_df$years_practice_z                 <- zscore(hcp_df$years_practice, "years_practice")
  hcp_df$academic_engagement_z            <- zscore(hcp_df$academic_engagement, "academic_engagement")
  hcp_df$formulary_z                      <- zscore(formulary_score, "formulary_score")
  hcp_df$AXPharmaceuticals_relationship_z <- zscore(AXPharmaceuticals_relationship_num,
                                                     "AXPharmaceuticals_relationship_num")

  # ---- Ground-truth DGP --- UNCHANGED, byte-for-byte identical coefficients ----
  specialty_effect <- ifelse(
    hcp_df$specialty == "Obesity Medicine", 1.35,
    ifelse(hcp_df$specialty == "Endocrinology", 0.72, 0)
  )
  payer_effect <- ifelse(hcp_df$dominant_payer %in% c("Commercial", "OOP"), 0.32, -0.18)

  true_logit <- -2.05 +
    0.92 * hcp_df$rx_volume_z +
    0.68 * hcp_df$nrx_share_z +
    1.18 * hcp_df$obesity_prev_z +
    0.88 * hcp_df$formulary_z -
    0.82 * hcp_df$pa_burden_z +
    0.58 * specialty_effect +
    0.62 * hcp_df$AXPharmaceuticals_relationship_z +
    0.78 * hcp_df$rep_engagement_z +
    0.42 * hcp_df$academic_engagement_z +
    0.35 * hcp_df$sample_request_recent +
    0.28 * payer_effect -
    0.28 * hcp_df$years_practice_z

  hcp_df$true_logit <- true_logit

  # Noise draw uses its own seed, decoupled from however many random draws
  # happened above, mirroring the original repo's separate set.seed(123)
  # immediately before its noise draw.
  set.seed(seed + 500)
  noise <- rnorm(n, mean = 0, sd = 1.18)
  hcp_df$true_prob <- plogis(true_logit + noise)
  hcp_df$prescribe_likely <- rbinom(n, size = 1, prob = hcp_df$true_prob)

  hcp_df$state <- state
  hcp_df
}

state_seeds <- c(Nebraska = 4301, Wisconsin = 4302, Mississippi = 4303)

dir.create("data/state_simulations", recursive = TRUE, showWarnings = FALSE)

for (state in names(state_seeds)) {
  df <- simulate_state_data(state, n = 5000, seed = state_seeds[[state]])
  out_path <- file.path("data/state_simulations", paste0(tolower(state), ".rds"))
  saveRDS(df, out_path)
  cat(state, ": ", nrow(df), " HCPs, ", round(mean(df$prescribe_likely) * 100, 1),
      "% prescribe_likely, saved to ", out_path, "\n", sep = "")
}
