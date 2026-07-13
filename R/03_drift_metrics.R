# 03_drift_metrics.R
#
# Builds the full evaluation harness for all 3 models x 3 states (9 cells):
#   - Discrimination: AUC, Gini vs. original golden-test-set baseline
#   - Calibration: Brier score + predicted-vs-observed calibration bins
#   - Feature importance stability: SHAP (XGBoost) / standardized
#     coefficients (LR), Spearman rank correlation vs. original ranking
#   - PSI per shifted feature, per state, vs. the ORIGINAL 5000-HCP
#     population (not just the golden test set)
#   - Confusion matrix / sensitivity-specificity at the original 0.5
#     threshold
#
# Reads only frozen artifacts (models/, data/state_simulations/scored/) — writes to results/drift_metrics.rds 

library(pacman)
p_load(tidyverse, broom, shapviz, xgboost, caret, pROC)

dir.create("results", showWarnings = FALSE)

original_df    <- readRDS("data/hcp_simulation_data.rds")
golden         <- readRDS("models/golden_baseline.rds")
scored         <- readRDS("data/state_simulations/scored/all_states_scored.rds")
feature_config <- readRDS("models/feature_config.rds")
lr_model       <- readRDS("models/lr_model.rds")
xgb_model      <- readRDS("models/xgb_model.rds")

predictors            <- feature_config$predictors
specialty_levels      <- feature_config$specialty_levels
dominant_payer_levels <- feature_config$dominant_payer_levels

states <- names(scored)
model_names <- c("lr", "xgb", "mlp")

# golden_baseline.rds (written by 00_load_frozen_models.R) only stored AUC and
# Brier for the golden test set. The fuller metric set is recomputed here from the predictions already stored in golden$test_df, not from refitting anything.
compute_metrics <- function(pred_prob, actual) {
  roc_obj <- roc(actual, pred_prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  gini    <- (2 * auc_val - 1) * 100
  brier   <- mean((pred_prob - actual)^2)
  pred_class <- factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0, 1))
  cm <- confusionMatrix(pred_class, factor(actual, levels = c(0, 1)), positive = "1")
  list(
    auc = auc_val, gini = gini, brier = brier,
    balanced_accuracy = unname(cm$byClass["Balanced Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    precision = unname(cm$byClass["Precision"])
  )
}

golden_full_metrics <- list(
  lr  = compute_metrics(golden$test_df$logit_pred_prob, golden$test_df$prescribe_likely),
  xgb = compute_metrics(golden$test_df$xgb_pred_prob, golden$test_df$prescribe_likely),
  mlp = compute_metrics(golden$test_df$dl_pred_prob, golden$test_df$prescribe_likely)
)
model_labels <- c(lr = "Logistic Regression", xgb = "XGBoost", mlp = "MLP")

# ---------------------------------------------------------------------------
# 1. Population Stability Index — vs. the ORIGINAL 5000-HCP population.
# ---------------------------------------------------------------------------
psi_continuous <- function(reference, comparison, bins = 10) {
  breaks <- unique(quantile(reference, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE))
  breaks[1] <- -Inf
  breaks[length(breaks)] <- Inf
  ref_pct  <- as.numeric(table(cut(reference, breaks = breaks, include.lowest = TRUE))) / length(reference)
  comp_pct <- as.numeric(table(cut(comparison, breaks = breaks, include.lowest = TRUE))) / length(comparison)
  ref_pct[ref_pct == 0]   <- 1e-4
  comp_pct[comp_pct == 0] <- 1e-4
  sum((comp_pct - ref_pct) * log(comp_pct / ref_pct))
}

psi_categorical <- function(reference, comparison) {
  levels_all <- union(unique(as.character(reference)), unique(as.character(comparison)))
  ref_pct  <- sapply(levels_all, function(l) mean(as.character(reference) == l))
  comp_pct <- sapply(levels_all, function(l) mean(as.character(comparison) == l))
  ref_pct[ref_pct == 0]   <- 1e-4
  comp_pct[comp_pct == 0] <- 1e-4
  sum((comp_pct - ref_pct) * log(comp_pct / ref_pct))
}

# Shifted features plus years_practice as an unshifted control (should show ~0 PSI everywhere — a sanity check that
# the simulation didn't accidentally perturb something it shouldn't have).
psi_features <- list(
  specialty            = "categorical",
  dominant_payer        = "categorical",
  obesity_prev         = "continuous",
  pa_burden            = "continuous",
  rep_engagement_score = "continuous",
  rx_volume_monthly    = "continuous",
  academic_engagement  = "continuous",
  years_practice       = "continuous"
)

psi_table <- expand.grid(feature = names(psi_features), state = states, stringsAsFactors = FALSE) %>%
  rowwise() %>%
  mutate(
    psi = {
      ftype <- psi_features[[feature]]
      ref <- original_df[[feature]]
      cmp <- scored[[state]]$scored_df[[feature]]
      if (ftype == "categorical") psi_categorical(ref, cmp) else psi_continuous(ref, cmp)
    }
  ) %>%
  ungroup() %>%
  mutate(
    interpretation = case_when(
      psi < 0.1  ~ "no significant shift",
      psi < 0.25 ~ "moderate shift",
      TRUE       ~ "major shift"
    )
  )

cat("=== PSI (feature x state), vs. original 5000-HCP population ===\n")
print(psi_table %>% arrange(state, desc(psi)), n = 100)

# ---------------------------------------------------------------------------
# 2. Discrimination / calibration deltas vs. the golden-test-set baseline.
# ---------------------------------------------------------------------------
performance_table <- map_dfr(states, function(state) {
  map_dfr(model_names, function(m) {
    base_met  <- golden_full_metrics[[m]]
    state_met <- scored[[state]]$metrics[[m]]
    tibble(
      state = state,
      model = model_labels[[m]],
      auc_baseline = base_met$auc,
      auc_state    = state_met$auc,
      auc_delta    = state_met$auc - base_met$auc,
      gini_baseline = base_met$gini,
      gini_state    = state_met$gini,
      balanced_accuracy_baseline = base_met$balanced_accuracy,
      balanced_accuracy_state    = state_met$balanced_accuracy,
      sensitivity_baseline = base_met$sensitivity,
      sensitivity_state    = state_met$sensitivity,
      specificity_baseline = base_met$specificity,
      specificity_state    = state_met$specificity,
      precision_baseline = base_met$precision,
      precision_state    = state_met$precision,
      brier_baseline = base_met$brier,
      brier_state    = state_met$brier,
      brier_delta    = state_met$brier - base_met$brier,
      n_unscorable = if (!is.null(state_met$n_unscorable)) state_met$n_unscorable else 0
    )
  })
})

cat("\n=== Performance deltas vs. golden-test-set baseline ===\n")
print(performance_table, n = 100)

# ---------------------------------------------------------------------------
# 3. Calibration bins (predicted vs. observed) — model x state.
# ---------------------------------------------------------------------------
calibration_bins <- function(pred_prob, actual, bins = 10) {
  valid <- !is.na(pred_prob)
  tibble(pred_prob = pred_prob[valid], actual = actual[valid]) %>%
    mutate(bin = cut(pred_prob, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarise(obs_rate = mean(actual), pred_rate = mean(pred_prob), n = n(), .groups = "drop")
}

calibration_table <- map_dfr(states, function(state) {
  map_dfr(model_names, function(m) {
    df <- scored[[state]]$scored_df
    pred_col <- paste0(m, "_pred_prob")
    calibration_bins(df[[pred_col]], df$prescribe_likely) %>%
      mutate(state = state, model = model_labels[[m]])
  })
})

# ---------------------------------------------------------------------------
# 4. Feature importance stability: SHAP (XGBoost) and standardized
#    coefficients (LR), Spearman rank correlation vs. the original ranking.
# ---------------------------------------------------------------------------
build_matrix <- function(df) {
  df <- df %>% mutate(
    specialty      = factor(specialty, levels = specialty_levels),
    dominant_payer = factor(dominant_payer, levels = dominant_payer_levels)
  )
  model.matrix(~ . - 1, data = df[, predictors])
}

aggregate_shap_by_feature <- function(shap_matrix, predictors) {
  cn <- colnames(shap_matrix)
  base <- vapply(cn, function(x) {
    matches <- predictors[vapply(predictors, function(p) startsWith(x, p), logical(1))]
    matches[which.max(nchar(matches))]
  }, character(1))
  vapply(split(seq_along(base), base), function(idx) {
    mean(rowSums(abs(shap_matrix[, idx, drop = FALSE])))
  }, numeric(1))
}

# Baseline SHAP importance, computed on the golden test set.
baseline_matrix <- build_matrix(golden$test_df)
baseline_shap <- shapviz(xgb_model, X = baseline_matrix, X_pred = baseline_matrix, predict_function = predict)
baseline_importance <- aggregate_shap_by_feature(baseline_shap$S, predictors)

state_shap_importance <- map(states, function(state) {
  state_matrix <- build_matrix(scored[[state]]$scored_df)
  state_shap <- shapviz(xgb_model, X = state_matrix, X_pred = state_matrix, predict_function = predict)
  aggregate_shap_by_feature(state_shap$S, predictors)
}) %>% set_names(states)

shap_rank_stability <- map_dfr(states, function(state) {
  state_importance <- state_shap_importance[[state]]
  common <- intersect(names(baseline_importance), names(state_importance))
  tibble(
    state = state,
    spearman_rho = cor(baseline_importance[common], state_importance[common], method = "spearman")
  )
})

shap_importance_table <- map_dfr(states, function(state) {
  state_importance <- state_shap_importance[[state]]
  tibble(feature = names(state_importance), mean_abs_shap = state_importance, state = state)
}) %>%
  bind_rows(tibble(feature = names(baseline_importance), mean_abs_shap = baseline_importance, state = "Baseline"))

cat("\n=== XGBoost SHAP rank stability (Spearman rho vs. golden-test baseline) ===\n")
print(shap_rank_stability)

# LR standardized coefficients — frozen, so this ranking is identical across
# every state by construction (the model is never refit). Included as a
# validation check (confirms we didn't accidentally refit), not a
# drift signal.
lr_coefs <- tidy(lr_model) %>% filter(term != "(Intercept)")
lr_coefs$base_feature <- vapply(lr_coefs$term, function(x) {
  matches <- predictors[vapply(predictors, function(p) startsWith(x, p), logical(1))]
  matches[which.max(nchar(matches))]
}, character(1))
lr_importance <- lr_coefs %>%
  group_by(base_feature) %>%
  summarise(mean_abs_std_coef = mean(abs(estimate)), .groups = "drop") %>%
  arrange(desc(mean_abs_std_coef))

cat("\n=== LR standardized coefficient ranking (frozen — identical every state) ===\n")
print(lr_importance, n = 100)

# ---------------------------------------------------------------------------
# 4b. Feature importance stability: MLP (permutation importance), mirroring
#     the original repo's approach computed fresh on baseline + each state's data through the
#     frozen MLP, then compared to the baseline ranking via Spearman
#     correlation, the same way as the XGBoost SHAP comparison above.
# ---------------------------------------------------------------------------
p_load(keras3, tensorflow, reticulate)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
use_condaenv(condaenv = "tf_env", conda = "/opt/anaconda3/bin/conda", required = TRUE)
mlp_model      <- load_model("models/mlp_model.keras")
mlp_preprocess <- readRDS("models/mlp_preprocess.rds")

mlp_scaled_matrix <- function(df) predict(mlp_preprocess, build_matrix(df))

mlp_perm_importance <- function(X_scaled, actual, n_perm = 5) {
  baseline_pred <- as.numeric(predict(mlp_model, X_scaled, verbose = 0))
  baseline_auc  <- as.numeric(auc(roc(actual, baseline_pred, quiet = TRUE)))

  importance <- numeric(ncol(X_scaled))
  names(importance) <- colnames(X_scaled)

  for (i in seq_len(ncol(X_scaled))) {
    auc_drop <- numeric(n_perm)
    for (p in seq_len(n_perm)) {
      X_perm <- X_scaled
      X_perm[, i] <- sample(X_perm[, i])
      pred_perm <- as.numeric(predict(mlp_model, X_perm, verbose = 0))
      auc_drop[p] <- baseline_auc - as.numeric(auc(roc(actual, pred_perm, quiet = TRUE)))
    }
    importance[i] <- mean(auc_drop)
  }
  importance
}

# Permutation importance yields one AUC-drop value per one-hot column already
# (unlike SHAP, there's no per-row dimension to aggregate over first) —
# base-feature groups are simply summed.
aggregate_perm_by_feature <- function(perm_importance, predictors) {
  cn <- names(perm_importance)
  base <- vapply(cn, function(x) {
    matches <- predictors[vapply(predictors, function(p) startsWith(x, p), logical(1))]
    matches[which.max(nchar(matches))]
  }, character(1))
  vapply(split(seq_along(base), base), function(idx) sum(perm_importance[idx]), numeric(1))
}

baseline_mlp_matrix     <- mlp_scaled_matrix(golden$test_df)
baseline_mlp_perm       <- mlp_perm_importance(baseline_mlp_matrix, golden$test_df$prescribe_likely)
baseline_mlp_importance <- aggregate_perm_by_feature(baseline_mlp_perm, predictors)

state_mlp_importance <- map(states, function(state) {
  state_matrix <- mlp_scaled_matrix(scored[[state]]$scored_df)
  state_perm   <- mlp_perm_importance(state_matrix, scored[[state]]$scored_df$prescribe_likely)
  aggregate_perm_by_feature(state_perm, predictors)
}) %>% set_names(states)

mlp_rank_stability <- map_dfr(states, function(state) {
  state_importance <- state_mlp_importance[[state]]
  common <- intersect(names(baseline_mlp_importance), names(state_importance))
  tibble(
    state = state,
    spearman_rho = cor(baseline_mlp_importance[common], state_importance[common], method = "spearman")
  )
})

mlp_importance_table <- map_dfr(states, function(state) {
  state_importance <- state_mlp_importance[[state]]
  tibble(feature = names(state_importance), mean_auc_drop = state_importance, state = state)
}) %>%
  bind_rows(tibble(feature = names(baseline_mlp_importance), mean_auc_drop = baseline_mlp_importance, state = "Baseline"))

cat("\n=== MLP permutation-importance rank stability (Spearman rho vs. golden-test baseline) ===\n")
print(mlp_rank_stability)

# ---------------------------------------------------------------------------
# 5. Confusion-matrix summary at the original 0.5 threshold.
# ---------------------------------------------------------------------------
confusion_summary <- map_dfr(states, function(state) {
  map_dfr(model_names, function(m) {
    cm <- scored[[state]]$metrics[[m]]$confusion_matrix
    tibble(
      state = state, model = model_labels[[m]],
      true_neg = cm[1, 1], false_neg = cm[1, 2],
      false_pos = cm[2, 1], true_pos = cm[2, 2]
    )
  })
})

cat("\n=== Confusion matrices at threshold 0.5 ===\n")
print(confusion_summary, n = 100)

# ---------------------------------------------------------------------------
# 6. Save everything 
# ---------------------------------------------------------------------------
drift_metrics <- list(
  psi_table            = psi_table,
  performance_table    = performance_table,
  calibration_table    = calibration_table,
  shap_rank_stability  = shap_rank_stability,
  shap_importance_table = shap_importance_table,
  mlp_rank_stability   = mlp_rank_stability,
  mlp_importance_table = mlp_importance_table,
  lr_importance        = lr_importance,
  confusion_summary    = confusion_summary
)

saveRDS(drift_metrics, "results/drift_metrics.rds")
cat("\nSaved results/drift_metrics.rds\n")
