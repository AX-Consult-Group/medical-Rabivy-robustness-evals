# 00_load_frozen_models.R
#
# Reproduces the three models from the original repo's
# hcp_propensity_model.qmd EXACTLY: same frozen dataset, same train/test
# split, same features, same hyperparameters, same seeds. Writes models/*.rds
# (and models/mlp_model.keras for the neural net) ONCE.
#
# This script is the only place training/fitting happens in this project.
# 01-03 downstream must only ever load these frozen artifacts and score new
# (state-shifted) data against them — never refit. 

library(pacman)
p_load(tidyverse, caret, broom, xgboost, pROC, ROCR)

set.seed(123)

# ---------------------------------------------------------------------------
# 1. Load the frozen synthetic dataset (generated once by the original repo;
#    never regenerated here — see the original repo's hcp_propensity_model.qmd
#    for the DGP). 
# ---------------------------------------------------------------------------
hcp_df <- readRDS("data/hcp_simulation_data.rds")

hcp_df <- hcp_df %>%
  rename(AXPharmaceuticals_relationship_z = amgen_relationship_z) %>%
  rename(AXPharmaceuticals_relationship = amgen_relationship)

stopifnot(nrow(hcp_df) == 5000)
cat("Loaded frozen dataset:", nrow(hcp_df), "HCPs,",
    round(mean(hcp_df$prescribe_likely) * 100, 1), "% positive rate\n")

# ---------------------------------------------------------------------------
# 2a. Persist the ORIGINAL population's centering/scaling constants.
#
# The frozen models were trained on z-scores computed via scale() on the
# full original 5000-HCP population. 01_simulate_state_data.R must standardize
# state-simulated covariates against these SAME fixed constants, never the
# state data's own mean/sd — re-standardizing per state would silently
# renormalize away the covariate shift. 
# ---------------------------------------------------------------------------
formulary_score_orig <- case_when(
  hcp_df$formulary_tier == "Preferred"    ~  2.0,
  hcp_df$formulary_tier == "NonPreferred" ~  0.8,
  hcp_df$formulary_tier == "PARequired"   ~ -0.6,
  TRUE ~ -1.8
)
axpharma_relationship_num_orig <- ifelse(
  hcp_df$AXPharmaceuticals_relationship == "TwoPlus", 2,
  ifelse(hcp_df$AXPharmaceuticals_relationship == "One", 1, 0)
)

baseline_standardization <- list(
  rx_volume_monthly    = list(mean = mean(hcp_df$rx_volume_monthly),    sd = sd(hcp_df$rx_volume_monthly)),
  nrx_share            = list(mean = mean(hcp_df$nrx_share),            sd = sd(hcp_df$nrx_share)),
  obesity_prev         = list(mean = mean(hcp_df$obesity_prev),         sd = sd(hcp_df$obesity_prev)),
  pa_burden            = list(mean = mean(hcp_df$pa_burden),            sd = sd(hcp_df$pa_burden)),
  rep_engagement_score = list(mean = mean(hcp_df$rep_engagement_score), sd = sd(hcp_df$rep_engagement_score)),
  years_practice       = list(mean = mean(hcp_df$years_practice),       sd = sd(hcp_df$years_practice)),
  academic_engagement  = list(mean = mean(hcp_df$academic_engagement),  sd = sd(hcp_df$academic_engagement)),
  formulary_score      = list(mean = mean(formulary_score_orig),        sd = sd(formulary_score_orig)),
  AXPharmaceuticals_relationship_num = list(mean = mean(axpharma_relationship_num_orig),
                                             sd = sd(axpharma_relationship_num_orig))
)

# Sanity check: reconstructing the _z columns from raw values + these
# constants must reproduce the frozen dataset's own _z columns exactly,
# since they were computed the same way at original generation time.
stopifnot(
  isTRUE(all.equal(scale(hcp_df$rx_volume_monthly)[, 1], hcp_df$rx_volume_z,
                    tolerance = 1e-6, check.attributes = FALSE)),
  isTRUE(all.equal((formulary_score_orig - baseline_standardization$formulary_score$mean) /
                      baseline_standardization$formulary_score$sd,
                    hcp_df$formulary_z, tolerance = 1e-6, check.attributes = FALSE)),
  isTRUE(all.equal((axpharma_relationship_num_orig - baseline_standardization$AXPharmaceuticals_relationship_num$mean) /
                      baseline_standardization$AXPharmaceuticals_relationship_num$sd,
                    hcp_df$AXPharmaceuticals_relationship_z, tolerance = 1e-6, check.attributes = FALSE))
)

dir.create("models", showWarnings = FALSE)
saveRDS(baseline_standardization, "models/baseline_standardization.rds")
cat("Saved models/baseline_standardization.rds (validated against frozen _z columns)\n")

# ---------------------------------------------------------------------------
# 2. Train/test split — identical seed, identical partition call.
# ---------------------------------------------------------------------------
set.seed(123)
train_idx <- createDataPartition(hcp_df$prescribe_likely, p = 0.7, list = FALSE)
train_df <- hcp_df[train_idx, ]
test_df  <- hcp_df[-train_idx, ]

cat("Training set:", nrow(train_df), " (", round(mean(train_df$prescribe_likely), 3), " positive)\n")
cat("Test set:    ", nrow(test_df),  " (", round(mean(test_df$prescribe_likely), 3),  " positive)\n")

predictors <- c("specialty", "rx_volume_z", "nrx_share_z", "dominant_payer",
                "formulary_z", "pa_burden_z", "AXPharmaceuticals_relationship_z",
                "rep_engagement_z", "years_practice_z", "obesity_prev_z",
                "sample_request_recent", "academic_engagement_z")

# Fix factor levels to the CANONICAL category universe (not unique(hcp_df$...))
# so state-simulated data always produces model.matrix columns identical to
# what these frozen models were trained on. Deriving levels from unique()
# is a trap here: "OOP" never happens to win the dominant_payer argmax in
# this particular 5000-row draw (a stochastic accident, not a guarantee —
# confirmed it does win the argmax in some state-simulated draws), so
# unique() silently omits it, factor() NAs any row with that value, and
# model.matrix silently drops those rows. This is a scoring-robustness
# safeguard, not a DGP change: the level sets themselves are unchanged.
specialty_levels      <- c("Endocrinology", "Obesity Medicine", "Primary Care")
dominant_payer_levels <- c("Commercial", "Medicaid", "Medicare", "OOP")
stopifnot(
  setequal(specialty_levels, unique(hcp_df$specialty)),
  setequal(dominant_payer_levels, union(unique(hcp_df$dominant_payer), "OOP"))
)

train_df$specialty      <- factor(train_df$specialty, levels = specialty_levels)
train_df$dominant_payer <- factor(train_df$dominant_payer, levels = dominant_payer_levels)
test_df$specialty       <- factor(test_df$specialty, levels = specialty_levels)
test_df$dominant_payer  <- factor(test_df$dominant_payer, levels = dominant_payer_levels)

# ---------------------------------------------------------------------------
# 3. Model 1: Logistic Regression
# ---------------------------------------------------------------------------
logit_formula <- as.formula(paste("prescribe_likely ~", paste(predictors, collapse = " + ")))
logit_model <- glm(logit_formula, data = train_df, family = binomial(link = "logit"))

test_df$logit_pred_prob <- predict(logit_model, newdata = test_df, type = "response")

lr_roc   <- roc(test_df$prescribe_likely, test_df$logit_pred_prob, quiet = TRUE)
lr_auc   <- as.numeric(auc(lr_roc))
lr_brier <- mean((test_df$logit_pred_prob - test_df$prescribe_likely)^2)
cat("\n[LR]  AUC:", round(lr_auc, 4), " Brier:", round(lr_brier, 4), "\n")

# ---------------------------------------------------------------------------
# 4. Model 2: XGBoost
# ---------------------------------------------------------------------------
train_df$prescribe_likely_num <- as.numeric(train_df$prescribe_likely)
test_df$prescribe_likely_num  <- as.numeric(test_df$prescribe_likely)

train_matrix <- xgb.DMatrix(
  data  = model.matrix(~ . - 1, data = train_df[, predictors]),
  label = train_df$prescribe_likely_num
)
test_matrix <- xgb.DMatrix(
  data  = model.matrix(~ . - 1, data = test_df[, predictors]),
  label = test_df$prescribe_likely_num
)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.03,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  gamma            = 0.1,
  nthread          = max(1, parallel::detectCores() - 1),
  seed             = 123
)

set.seed(123)
xgb_model <- xgb.train(
  params  = xgb_params,
  data    = train_matrix,
  nrounds = 1000,
  evals   = list(train = train_matrix, test = test_matrix),
  early_stopping_rounds = 50,
  verbose = 0
)

test_df$xgb_pred_prob <- predict(xgb_model, test_matrix)

xgb_pred_obj <- prediction(test_df$xgb_pred_prob, test_df$prescribe_likely_num)
xgb_auc      <- performance(xgb_pred_obj, "auc")@y.values[[1]]
xgb_brier    <- mean((test_df$xgb_pred_prob - test_df$prescribe_likely)^2)
cat("[XGB] AUC:", round(xgb_auc, 4), " Brier:", round(xgb_brier, 4), "\n")

# ---------------------------------------------------------------------------
# 5. Model 3: MLP (keras3 / TensorFlow)
# ---------------------------------------------------------------------------
p_load(keras3, tensorflow, reticulate)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
use_condaenv(condaenv = "tf_env", conda = "/opt/anaconda3/bin/conda", required = TRUE)

set.seed(123)
tf$random$set_seed(123)
Sys.setenv(TF_DETERMINISTIC_OPS = "1")

X_train_raw <- train_df[, predictors]
X_test_raw  <- test_df[, predictors]

X_train <- model.matrix(~ . - 1, data = X_train_raw) %>% as.matrix()
X_test  <- model.matrix(~ . - 1, data = X_test_raw) %>% as.matrix()

mlp_preprocess <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(mlp_preprocess, X_train)
X_test  <- predict(mlp_preprocess, X_test)

y_train <- as.numeric(train_df$prescribe_likely)
y_test  <- as.numeric(test_df$prescribe_likely)

keras_model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

keras_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy", "auc")
)

invisible(keras_model %>% fit(
  X_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
))

test_df$dl_pred_prob <- as.numeric(predict(keras_model, X_test, verbose = 0))
dl_roc   <- roc(test_df$prescribe_likely, test_df$dl_pred_prob, quiet = TRUE)
dl_auc   <- as.numeric(auc(dl_roc))
dl_brier <- mean((test_df$dl_pred_prob - test_df$prescribe_likely)^2)
cat("[MLP] AUC:", round(dl_auc, 4), " Brier:", round(dl_brier, 4), "\n")
cat("(Note: TensorFlow/Keras training is not bit-exact reproducible even with\n",
    " fixed seeds — this MLP is 'a' frozen model matching the original repo's\n",
    " architecture/hyperparameters/seeds, same caveat the original report flags.)\n")

# ---------------------------------------------------------------------------
# 6. Freeze everything to disk
# ---------------------------------------------------------------------------
dir.create("models", showWarnings = FALSE)

saveRDS(logit_model, "models/lr_model.rds")
saveRDS(xgb_model,   "models/xgb_model.rds")
save_model(keras_model, "models/mlp_model.keras", overwrite = TRUE)
saveRDS(mlp_preprocess, "models/mlp_preprocess.rds")

saveRDS(
  list(
    predictors             = predictors,
    specialty_levels       = specialty_levels,
    dominant_payer_levels  = dominant_payer_levels
  ),
  "models/feature_config.rds"
)

saveRDS(
  list(
    test_df = test_df,
    metrics = list(
      lr  = list(auc = lr_auc,  brier = lr_brier),
      xgb = list(auc = xgb_auc, brier = xgb_brier),
      mlp = list(auc = dl_auc,  brier = dl_brier)
    )
  ),
  "models/golden_baseline.rds"
)

cat("\nSaved: models/lr_model.rds, models/xgb_model.rds, models/mlp_model.keras,\n",
    "models/mlp_preprocess.rds, models/feature_config.rds, models/golden_baseline.rds\n")
