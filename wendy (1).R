# Install once if needed
install.packages(c("tidyverse", "lme4", "brms", "cmdstanr",
                   "lavaan", "blavaan", "dbarts", "Metrics"))
install.packages("remotes")

remotes::install_github("stan-dev/cmdstanr")
# Load libraries
library(cmdstanr)
install_cmdstan()
library(tidyverse)
library(lme4)
library(brms)
library(cmdstanr)
library(lavaan)
library(blavaan)
library(dbarts)
library(Metrics)
library(haven)
library(dplyr)
library(brms)
library(lavaan)
library(dbarts)
set.seed(123)

J <- 100
n <- 20
N <- J * n

school <- rep(1:J, each = n)

# Intervention
TREAT <- sample(0:3, J, replace = TRUE)
TREAT_full <- rep(TREAT, each = n)

# -------------------------
# LEVEL 1 (Child)
# -------------------------
gender <- rbinom(N, 1, 0.5)
age <- rnorm(N, 5, 1)

# -------------------------
# LEVEL 2 (Teacher)
# -------------------------
teacher_exp <- rnorm(J, 10, 3)
prep <- rnorm(J, 3, 1)
training <- rpois(J, 3)

materials <- rnorm(J, 1, 0.5)
facility  <- rnorm(J, 1, 0.5)

# Expand to child level
expand <- function(x) rep(x, each = n)

teacher_exp_f <- expand(teacher_exp)
prep_f        <- expand(prep)
training_f    <- expand(training)
materials_f   <- expand(materials)
facility_f    <- expand(facility)

# -------------------------
# LEVEL 3 (School)
# -------------------------
school_support <- rnorm(J, 5, 2)
sanitation     <- rnorm(J, 1, 0.5)

school_support_f <- expand(school_support)
sanitation_f     <- expand(sanitation)

# Random effects
u <- rnorm(J, 0, 5)
u_full <- expand(u)

epsilon <- rnorm(N, 0, 10)

# -------------------------
# OUTCOME
# -------------------------
TSRI <- 50 +
  2 * (TREAT_full == 1) +
  3.5 * (TREAT_full == 2) +
  3 * (TREAT_full == 3) +
  1.5 * gender +
  2 * age +
  1.2 * teacher_exp_f +
  1.5 * prep_f +
  1.8 * training_f +
  1.3 * materials_f +
  1.2 * facility_f +
  2 * school_support_f +
  1.5 * sanitation_f +
  u_full + epsilon

# Final dataset
data <- data.frame(
  TSRI,
  school = factor(school),
  TREAT = factor(TREAT_full),
  gender,
  age,
  teacher_exp = teacher_exp_f,
  prep = prep_f,
  training = training_f,
  materials = materials_f,
  facility = facility_f,
  school_support = school_support_f,
  sanitation = sanitation_f
)

### OBJ 1 

model_bayes_standard <- brm(
  TSRI ~ TREAT + gender + age +
    teacher_exp + prep + training +
    materials + facility +
    school_support + sanitation +
    (1 | school),
  data = data,
  family = gaussian(),
  chains = 4,
  iter = 10000,
  seed = 123
)


model_bayes_horseshoe <- brm(
  TSRI ~ TREAT + gender + age +
    teacher_exp + prep + training +
    materials + facility +
    school_support + sanitation +
    (1 | school),
  data = data,
  family = gaussian(),
  prior = prior(horseshoe(), class = "b"),
  chains = 4,
  iter = 10000,
  seed = 123
)

# WAIC
waic_standard <- waic(model_bayes_standard)
waic_horseshoe <- waic(model_bayes_horseshoe)

waic_standard
waic_horseshoe

# LOO
loo_standard <- loo(model_bayes_standard)
loo_horseshoe <- loo(model_bayes_horseshoe)

loo_standard
loo_horseshoe

# Credible intervals
posterior_interval(model_bayes_standard)
posterior_interval(model_bayes_horseshoe)

# Posterior SD
posterior_summary(model_bayes_standard)[, "Est.Error"]
posterior_summary(model_bayes_horseshoe)[, "Est.Error"]

# Predictive uncertainty
yrep_std <- posterior_predict(model_bayes_standard)
yrep_hs  <- posterior_predict(model_bayes_horseshoe)

pred_std <- apply(yrep_std, 2, quantile, probs = c(0.025, 0.975))
pred_hs  <- apply(yrep_hs, 2, quantile, probs = c(0.025, 0.975))

coverage_std <- mean(data$TSRI >= pred_std[1, ] & data$TSRI <= pred_std[2, ])
coverage_hs  <- mean(data$TSRI >= pred_hs[1, ] & data$TSRI <= pred_hs[2, ])

coverage_std
coverage_hs


## OBJ 2
library(lavaan)

set.seed(123)

school_effect <- rnorm(J, 0, 25)
school_effect_full <- school_effect[as.numeric(data$school)]

data$Y1 <- 0.6 * data$TSRI + school_effect_full + rnorm(nrow(data), 0, 8)
data$Y2 <- 0.5 * data$TSRI + school_effect_full + rnorm(nrow(data), 0, 8)
data$Y3 <- 0.7 * data$TSRI + school_effect_full + rnorm(nrow(data), 0, 8)
data$Y4 <- 0.6 * data$TSRI + school_effect_full + rnorm(nrow(data), 0, 8)

dummies <- model.matrix(~ TREAT, data)
dummies <- dummies[, -1]
data <- cbind(data, dummies)

sem_model <- '
eta =~ Y1 + Y2 + Y3 + Y4

eta ~ TREAT1 + TREAT2 + TREAT3 +
      gender + age +
      teacher_exp + prep + training +
      materials + facility +
      school_support + sanitation
'

fit_sem <- sem(sem_model, data = data)

fit_msem <- sem(
  sem_model,
  data = data,
  cluster = "school"
)


summary(fit_sem, fit.measures = TRUE, standardized = TRUE)
summary(fit_msem, fit.measures = TRUE, standardized = TRUE)


# RMSEA
rmsea_sem  <- fitMeasures(fit_sem, "rmsea")
rmsea_msem <- fitMeasures(fit_msem, "rmsea")

# CFI
cfi_sem  <- fitMeasures(fit_sem, "cfi")
cfi_msem <- fitMeasures(fit_msem, "cfi")

# SRMR
srmr_sem  <- fitMeasures(fit_sem, "srmr")
srmr_msem <- fitMeasures(fit_msem, "srmr")

rmsea_sem
rmsea_msem
cfi_sem
cfi_msem
srmr_sem
srmr_msem

inspect(fit_sem, "std")$lambda
inspect(fit_msem, "std")$lambda

inspect(fit_sem, "r2")
inspect(fit_msem, "r2")



### OBJ 3
# Convert treatment to numeric dummies
# Treatment dummies
bart_dummies <- model.matrix(~ TREAT, data)[, -1]

# BART predictors (standard)
X_bart <- cbind(
  bart_dummies,
  data$gender,
  data$age,
  data$teacher_exp,
  data$prep,
  data$training,
  data$materials,
  data$facility,
  data$school_support,
  data$sanitation
)

# BART = add school structure explicitly
school_num <- as.numeric(data$school)

X_mbart <- cbind(
  X_bart,
  school_num   # THIS WAS MISSING → caused your error
)

# Outcome
y_bart <- data$TSRI

# Standard BART
bart_model <- bart(
  x.train = X_bart,
  y.train = y_bart,
  keeptrees = TRUE
)

# Multilevel BART (proxy using school feature)
mbart_model <- bart(
  x.train = X_mbart,
  y.train = y_bart,
  keeptrees = TRUE
)

pred_bart  <- colMeans(predict(bart_model, X_bart))
pred_mbart <- colMeans(predict(mbart_model, X_mbart))

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

rmse_bart  <- rmse(y_bart, pred_bart)
rmse_mbart <- rmse(y_bart, pred_mbart)

rmse_bart
rmse_mbart

pred_draws_bart  <- predict(bart_model, X_bart)
pred_draws_mbart <- predict(mbart_model, X_mbart)

pred_sd_bart  <- apply(pred_draws_bart, 2, sd)
pred_sd_mbart <- apply(pred_draws_mbart, 2, sd)

mean(pred_sd_bart)
mean(pred_sd_mbart)

# BART intervals
pred_int_bart <- apply(pred_draws_bart, 2, quantile, c(0.025, 0.975))

coverage_bart <- mean(
  y_bart >= pred_int_bart[1, ] &
    y_bart <= pred_int_bart[2, ]
)

# MBART intervals
pred_int_mbart <- apply(pred_draws_mbart, 2, quantile, c(0.025, 0.975))

coverage_mbart <- mean(
  y_bart >= pred_int_mbart[1, ] &
    y_bart <= pred_int_mbart[2, ]
)

coverage_bart
coverage_mbart



varimp_bart  <- colMeans(bart_model$varcount)
varimp_mbart <- colMeans(mbart_model$varcount)

# Normalize
varimp_bart_norm  <- varimp_bart / sum(varimp_bart)
varimp_mbart_norm <- varimp_mbart / sum(varimp_mbart)

varimp_bart_norm
varimp_mbart_norm








# ================================
# LOAD PACKAGES
# ================================
library(haven)
library(dplyr)

# ================================
# LOAD DATA (FROM DOWNLOADS)
# ================================
child <- read_dta("C:/Users/HP/Downloads/Tayari_Assessment_AllRounds.dta")
teacher <- read_dta("C:/Users/HP/Downloads/Tayari_Teachers_AllRounds.dta")
school <- read_dta("C:/Users/HP/Downloads/Tayari_ECDEProfile_AllRounds (4).dta")

# ================================
# STEP 1: AGGREGATE TEACHER DATA (ONE ROW PER SCHOOL)
# ================================
teacher_agg <- teacher %>%
  group_by(schid) %>%
  summarise(
    teacher_quality = mean(
      rowMeans(across(c(q38, q39, q320, q323)), na.rm = TRUE),
      na.rm = TRUE
    ),
    training_quality = mean(
      rowMeans(across(c(q324, q325, q326, q327, q328, q329)), na.rm = TRUE),
      na.rm = TRUE
    ),
    resources = mean(
      rowMeans(across(c(q411a, q412a, q413a, q414a)), na.rm = TRUE),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# ================================
# STEP 2: AGGREGATE SCHOOL DATA (ONE ROW PER SCHOOL)
# ================================
school_agg <- school %>%
  group_by(schid) %>%
  summarise(
    school_env = mean(
      rowMeans(across(c(q54, q55, q56, q57)), na.rm = TRUE),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# ================================
# STEP 3: MERGE (NO DUPLICATION)
# ================================
data <- child %>%
  left_join(teacher_agg, by = "schid") %>%
  left_join(school_agg, by = "schid")

# ================================
# STEP 4: CREATE VARIABLES
# ================================

# Cluster
data$school <- as.factor(data$schid)

# Treatment
data$TREAT <- as.factor(data$treat)

# Gender (already exists)
data$gender <- data$gender

# ================================
# STEP 5: CREATE OUTCOME (TSRI)
# ================================
tsri_items <- grep("^c[2-4]_", names(data), value = TRUE)

data$TSRI <- rowMeans(data[, tsri_items], na.rm = TRUE)

# ================================
# STEP 6: FINAL CLEAN DATASET
# ================================
data <- data %>%
  select(
    TSRI,
    school,
    TREAT,
    gender,
    teacher_quality,
    training_quality,
    resources,
    school_env
  ) %>%
  na.omit()


### OBJ 1

model_bayes_standard <- brm(
  TSRI ~ TREAT + gender +
    teacher_quality + training_quality +
    resources + school_env +
    (1 | school),
  data = data,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  seed = 123
)

model_bayes_horseshoe <- brm(
  TSRI ~ TREAT + gender +
    teacher_quality + training_quality +
    resources + school_env +
    (1 | school),
  data = data,
  family = gaussian(),
  prior = prior(horseshoe(), class = "b"),
  chains = 4,
  iter = 2000,
  seed = 123
)


waic_standard   <- waic(model_bayes_standard)
waic_horseshoe  <- waic(model_bayes_horseshoe)

waic_standard
waic_horseshoe

loo_standard  <- loo(model_bayes_standard)
loo_horseshoe <- loo(model_bayes_horseshoe)

loo_standard
loo_horseshoe

loo_compare(waic_standard, waic_horseshoe)

# 95% credible intervals
posterior_interval(model_bayes_standard, prob = 0.95)
posterior_interval(model_bayes_horseshoe, prob = 0.95)

posterior_summary(model_bayes_standard)[, "Est.Error"]
posterior_summary(model_bayes_horseshoe)[, "Est.Error"]


# Draw predictions
yrep <- posterior_predict(model_bayes_standard)

# 95% predictive interval
pred_int <- apply(yrep, 2, quantile, probs = c(0.025, 0.975))

# Coverage probability
coverage <- mean(
  data$TSRI >= pred_int[1, ] &
    data$TSRI <= pred_int[2, ]
)

coverage
yrep1 <- posterior_predict(model_bayes_horseshoe)

# 95% predictive interval
pred_int <- apply(yrep1, 2, quantile, probs = c(0.025, 0.975))

# Coverage probability
coverage <- mean(
  data$TSRI >= pred_int[1, ] &
    data$TSRI <= pred_int[2, ]
)



summary(model_bayes_horseshoe)

fixef(model_bayes_standard)
fixef(model_bayes_horseshoe)



### OBJ 2
set.seed(123)

data$Y1 <- scale(data$TSRI) + rnorm(nrow(data), 0, 0.5)
data$Y2 <- scale(data$TSRI) + rnorm(nrow(data), 0, 0.5)
data$Y3 <- scale(data$TSRI) + rnorm(nrow(data), 0, 0.5)
data$Y4 <- scale(data$TSRI) + rnorm(nrow(data), 0, 0.5)


dummies <- model.matrix(~ TREAT, data)
dummies <- dummies[, -1]

data <- cbind(data, dummies)

sem_model <- '
eta =~ Y1 + Y2 + Y3 + Y4

eta ~ TREAT1 + TREAT2 + TREAT3 +
      gender +
      teacher_quality + training_quality +
      resources + school_env
'

fit_sem <- sem(
  sem_model,
  data = data,
  estimator = "MLR"
)


msem_model <- '
# LEVEL 1 (child)
level: 1
eta =~ Y1 + Y2 + Y3 + Y4

eta ~ TREAT1 + TREAT2 + TREAT3 +
      gender

# LEVEL 2 (school)
level: 2
eta =~ Y1 + Y2 + Y3 + Y4

eta ~ teacher_quality + training_quality +
      resources + school_env
'

fit_msem <- sem(
  msem_model,
  data = data,
  cluster = "school",
  estimator = "MLR"
)


# SEM
fitMeasures(fit_sem, c("rmsea","cfi","srmr"))

# MSEM
fitMeasures(fit_msem, c("rmsea","cfi","srmr","srmr_within","srmr_between"))
fitMeasures(fit_sem, c("chisq", "df", "pvalue", "aic", "bic"))
fitmeasures(fit_msem, c("chisq", "df", "pvalue", "aic", "bic"))



# Standardized factor loadings
lambda_sem <- inspect(fit_sem, "std")$lambda
# SEM R²
r2_sem <- inspect(fit_sem, "r2")

# MSEM R²
r2_msem <- inspect(fit_msem, "r2")
r2_within <- r2_msem$within
r2_school <- r2_msem$school
eta <- "eta"

df <- data.frame(
  Indicator = c(indicators, eta),
  StdLoading_SEM = c(lambda_sem[indicators, "eta"], NA),  # loadings for indicators, NA for eta
  R2_SEM = r2_sem[c(indicators, eta)],
  R2_MSEM_Within = r2_within[c(indicators, eta)],
  R2_MSEM_School = r2_school[c(indicators, eta)]
)

df



parameterEstimates(fit_sem, standardized = TRUE)
parameterEstimates(fit_msem, standardized = TRUE)


parameterEstimates(fit_sem, ci = TRUE)
parameterEstimates(fit_msem, ci = TRUE)


lavInspect(fit_msem, "icc")

inspect(fit_sem, "std")$lambda
inspect(fit_msem, "std")$lambda



inspect(fit_sem, "r2")
inspect(fit_msem, "r2")


summary(fit_sem, fit.measures = TRUE, standardized = TRUE)
summary(fit_msem, fit.measures = TRUE, standardized = TRUE)




### OBJ 3

# Treatment dummies
bart_dummies <- model.matrix(~ TREAT, data)
bart_dummies <- bart_dummies[, -1]

# Predictor matrix (STANDARD BART)
X_bart <- cbind(
  bart_dummies,
  data$gender,
  data$teacher_quality,
  data$training_quality,
  data$resources,
  data$school_env
)

# Outcome
y_bart <- data$TSRI

bart_model <- bart(
  x.train = X_bart,
  y.train = y_bart,
  keeptrees = TRUE   # VERY IMPORTANT
)


X_mbart <- cbind(
  X_bart,
  as.numeric(data$school)
)

mbart_model <- bart(
  x.train = X_mbart,
  y.train = y_bart,
  keeptrees = TRUE
)

pred_bart  <- colMeans(predict(bart_model, X_bart))
pred_mbart <- colMeans(predict(mbart_model, X_mbart))

rmse <- function(a, p) sqrt(mean((a - p)^2))

rmse_bart  <- rmse(y_bart, pred_bart)
rmse_mbart <- rmse(y_bart, pred_mbart)

rmse_bart
rmse_mbart



pred_draws_bart  <- predict(bart_model, X_bart)
pred_draws_mbart <- predict(mbart_model, X_mbart)

pred_sd_bart  <- apply(pred_draws_bart, 2, sd)
pred_sd_mbart <- apply(pred_draws_mbart, 2, sd)

mean(pred_sd_bart)
mean(pred_sd_mbart)



# BART intervals
pred_int_bart <- apply(pred_draws_bart, 2, quantile, c(0.025, 0.975))

coverage_bart <- mean(
  y_bart >= pred_int_bart[1, ] &
    y_bart <= pred_int_bart[2, ]
)

# MBART intervals
pred_int_mbart <- apply(pred_draws_mbart, 2, quantile, c(0.025, 0.975))

coverage_mbart <- mean(
  y_bart >= pred_int_mbart[1, ] &
    y_bart <= pred_int_mbart[2, ]
)

coverage_bart
coverage_mbart

varimp_bart  <- colMeans(bart_model$varcount)
varimp_mbart <- colMeans(mbart_model$varcount)

# Normalize
varimp_bart_norm  <- varimp_bart / sum(varimp_bart)
varimp_mbart_norm <- varimp_mbart / sum(varimp_mbart)

varimp_bart_norm
varimp_mbart_norm

