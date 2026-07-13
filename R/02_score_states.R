# 02_score_states.R
#
# Scores the three FROZEN models (LR, XGBoost, MLP — never refit here) against
# all three state-simulated datasets. Writes, per state, the scored data frame
# (predictions from all 3 models attached) plus a metrics table (AUC, Gini,
# Brier, confusion matrix / sensitivity / specificity at the original 0.5
# threshold) to data/state_simulations/scored/. 03_drift_metrics.R consumes
# these for PSI, calibration, and SHAP/coefficient rank-stability comparisons.

library(pacman)
p_load(tidyverse, caret, broom, xgboost, pROC, ROCR)

feature_config <- readRDS("models/feature_config.rds")
predictors             <- feature_config$predictors
specialty_levels       <- feature_config$specialty_levels
dominant_payer_levels  <- feature_config$dominant_payer_levels

lr_model  <- readRDS("models/lr_model.rds")
xgb_model <- readRDS("models/xgb_model.rds")

mlp_preprocess <- readRDS("models/mlp_preprocess.rds")
p_load(keras3, tensorflow, reticulate)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
use_condaenv(condaenv = "tf_env", conda = "/opt/anaconda3/bin/conda", required = TRUE)
mlp_model <- load_model("models/mlp_model.keras")

# ---------------------------------------------------------------------------
# Metrics helper — same definitions as the original report's evaluation
# harness (AUC/Gini, Brier, confusion matrix at threshold 0.5).
# ---------------------------------------------------------------------------
compute_metrics <- function(pred_prob, actual) {
  roc_obj <- roc(actual, pred_prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  gini    <- (2 * auc_val - 1) * 100
  brier   <- mean((pred_prob - actual)^2)

  pred_class <- factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0, 1))
  cm <- confusionMatrix(pred_class, factor(actual, levels = c(0, 1)), positive = "1")

  list(
    auc = auc_val,
    gini = gini,
    brier = brier,
    balanced_accuracy = unname(cm$byClass["Balanced Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    precision = unname(cm$byClass["Precision"]),
    confusion_matrix = cm$table
  )
}

score_state <- function(state) {

  state_path <- file.path("data/state_simulations", paste0(tolower(state), ".rds"))
  df <- readRDS(state_path)

  # Enforce the frozen factor levels so model.matrix always produces the
  # same columns the models were trained on, even if a level (e.g.
  # "Medicaid") is rare-to-absent in this state's dominant_payer draws.
  df$specialty      <- factor(df$specialty, levels = specialty_levels)
  df$dominant_payer <- factor(df$dominant_payer, levels = dominant_payer_levels)

  actual <- df$prescribe_likely

  # ---- Logistic Regression ----
  # "OOP" never occurs as dominant_payer anywhere in the original 5000-row
  # training population — Commercial/Medicare always outweigh it there). 
  # glm() silently drops unobserved factor levels, so
  # the frozen LR model has no coefficient for dominant_payer=OOP.
  # Rather than refit or silently coerce these rows to a level they aren't, we score only what LR can score.
  lr_scorable <- df$dominant_payer %in% lr_model$xlevels$dominant_payer
  df$lr_pred_prob <- NA_real_
  df$lr_pred_prob[lr_scorable] <- predict(lr_model, newdata = df[lr_scorable, ], type = "response")
  n_lr_unscorable <- sum(!lr_scorable)
  if (n_lr_unscorable > 0) {
    cat(sprintf("  [LR] NOTE: %d/%d HCPs have dominant_payer level(s) unseen in original training (%s) — no LR coefficient exists; excluded from LR metrics.\n",
                n_lr_unscorable, nrow(df),
                paste(setdiff(unique(as.character(df$dominant_payer)), lr_model$xlevels$dominant_payer), collapse = ", ")))
  }

  # ---- XGBoost ----
  x_matrix <- model.matrix(~ . - 1, data = df[, predictors])
  xgb_dmatrix <- xgb.DMatrix(data = x_matrix)
  df$xgb_pred_prob <- predict(xgb_model, xgb_dmatrix)

  # ---- MLP ----
  x_scaled <- predict(mlp_preprocess, x_matrix)
  df$mlp_pred_prob <- as.numeric(predict(mlp_model, x_scaled, verbose = 0))

  metrics <- list(
    lr  = c(compute_metrics(df$lr_pred_prob[lr_scorable], actual[lr_scorable]),
            list(n_unscorable = n_lr_unscorable)),
    xgb = compute_metrics(df$xgb_pred_prob, actual),
    mlp = compute_metrics(df$mlp_pred_prob, actual)
  )

  list(state = state, scored_df = df, metrics = metrics)
}

dir.create("data/state_simulations/scored", recursive = TRUE, showWarnings = FALSE)

states <- c("Nebraska", "Wisconsin", "Mississippi")
all_results <- list()

for (state in states) {
  cat("Scoring", state, "...\n")
  result <- score_state(state)
  all_results[[state]] <- result

  saveRDS(result, file.path("data/state_simulations/scored", paste0(tolower(state), "_scored.rds")))

  for (m in names(result$metrics)) {
    met <- result$metrics[[m]]
    cat(sprintf("  [%s] AUC: %.4f  Gini: %.1f%%  Brier: %.4f  BalAcc: %.4f  Sens: %.4f  Spec: %.4f\n",
                toupper(m), met$auc, met$gini, met$brier, met$balanced_accuracy,
                met$sensitivity, met$specificity))
  }
}

saveRDS(all_results, "data/state_simulations/scored/all_states_scored.rds")
cat("\nSaved per-state scored data + metrics to data/state_simulations/scored/\n")
