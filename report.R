# *****************************************************************************
# Step 1: Prepare dataset
# *****************************************************************************


# -----------------------------------------------------------------------------
# Download MovieLens dataset
# !!! code provided by the course
# -----------------------------------------------------------------------------


## ----step1-download, results='hide'------------------------------------------
# Download all required libraries
# Note: this process could take a couple of minutes
# These 3 libraries are used to split edx and validation sets
if (!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org"
)
if (!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org"
)
if (!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org"
)
# recosystem is used in recommendation algorithm
if (!require(recosystem)) install.packages(
  "recosystem", repos = "http://cran.us.r-project.org"
)

# Load all required libraries
library(tidyverse)
library(caret)
library(data.table)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(
  text = gsub(
    "::", "\t",
    readLines(unzip(dl, "ml-10M100K/ratings.dat"))
  ),
  col.names = c("userId", "movieId", "rating", "timestamp")
)

movies <- str_split_fixed(
  readLines(unzip(dl, "ml-10M100K/movies.dat")),
  "\\::", 3
)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
if (as.integer(R.version$major) <= 3 & as.double(R.version$minor) <= 6) {
  movies <- as.data.frame(movies) %>%
    mutate(
      movieId = as.numeric(levels(movieId))[movieId],
      title = as.character(title),
      genres = as.character(genres)
    )
}

# if using R 4.0 or later:
if (as.integer(R.version$major) >= 4) {
  movies <- as.data.frame(movies) %>%
    mutate(
      movieId = as.numeric(movieId),
      title = as.character(title),
      genres = as.character(genres)
    )
}

movielens <- left_join(ratings, movies, by = "movieId")

# -----------------------------------------------------------------------------
# Create edx set, validation set (final hold-out test set)
# !!! code provided by the course
# -----------------------------------------------------------------------------
# Split MovieLens into edx (90%) and validation (10%) sets
#   edx set is used in model development
#   validation set is NOT used anywhere except when **testing the final model**,
#     so it's hidden data


# if using R 3.5 or earlier, use `set.seed(1)`
if (as.integer(R.version$major) <= 3 & as.double(R.version$minor) <= 5) {
  set.seed(1)
} else {
  set.seed(1, sample.kind = "Rounding")
}

test_index <- createDataPartition(
  y = movielens$rating, times = 1, p = 0.1, list = FALSE
)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean up
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# -----------------------------------------------------------------------------
# Data cleaning
# -----------------------------------------------------------------------------
# Environment now contains edx and validation sets
# Free memory is low, so need to free some space


## ----step1-prepare, results='hide'-------------------------------------------
# Because my algorithm only uses `userId`, `movieId`, `rating` columns
# only keep these 3 columns and remove the rest
edx <- edx %>% select(userId, movieId, rating)
validation <- validation %>% select(userId, movieId, rating)

# Store edx and validation as rda files, which can be loaded later
if (!dir.exists("rdas")) {
  dir.create("rdas")
}
save(edx, file = "rdas/edx.rda")
save(validation, file = "rdas/validation.rda")

# Delete validation variable
rm(validation)


# *****************************************************************************
# STEP 2: Describe dataset
# *****************************************************************************


options(digits = 3)
## ----step2-rating-summary----------------------------------------------------
# 5-number summary
# min = 0.5, Q1 = 3, median = 4, Q3 = 4, max = 5
# We can see that the minimum rating is 0.5, and half of the ratings are
# between 3 and 4 stars, so overall users are kind in their ratings!
summary(edx$rating)


## ----step2-rating-histogram, echo=FALSE--------------------------------------
# Histogram shows that whole star ratings more common than half star ratings
edx %>%
  ggplot(aes(rating)) +
  geom_bar() +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  labs(title = "Rating Distribution of edx set")


## ----step2-sparsity----------------------------------------------------------
# 69878 users
n_distinct(edx$userId)
# 10677 movies
n_distinct(edx$movieId)
# 9000055 ratings
nrow(edx)
# sparsity = 1.21%
nrow(edx) / (n_distinct(edx$userId) * n_distinct(edx$movieId))


# *****************************************************************************
# STEP 3: Develop the first model
# *****************************************************************************


# -----------------------------------------------------------------------------
# Create first model
# -----------------------------------------------------------------------------


## ----step3-split, results='hide'---------------------------------------------
# Split edx set into train (90%) and test (10%) sets
set.seed(1)
test_index <- createDataPartition(
  y = edx$rating, times = 1, p = 0.1, list = FALSE
)
train <- edx[-test_index, ]
temp <- edx[test_index, ]

# Make sure userId and movieId in test set are also in training set
test <- temp %>%
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into training set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Clean up
rm(edx, test_index, temp, removed)


## ----step3-create-model------------------------------------------------------
# Create a new model object, this becomes the first model
# FUNCTION: Reco
#   OUTPUT RecoSys obj that has method $tune, $train, $predict
first_model <- Reco()


## ----step3-index1------------------------------------------------------------
# userId and movieId start with 1 and not 0
# so later in data_memory function, index1 should set to TRUE
min(train$userId)
min(train$movieId)


## ----step3-convert-data------------------------------------------------------
# Convert train and test dataframes into recosystem DataSource objects
# Add the _data suffix to new DataSource variable names to avoid mix up
# FUNCTION: data_memory
#   INPUT user_index = userId, item_index = movieId, rating
#   OUTPUT DataSource obj
# The DataSource objects will contain same data as the original dataframes
# Note: The syntax to access user id, movie id, and rating fields:
#       train$userId is train_data@source[1]
#       train$movieId is train_data@source[2]
#       train$rating is train_data@source[3]
train_data <- data_memory(
  user_index = train$userId,
  item_index = train$movieId,
  rating = train$rating,
  index1 = TRUE
)

test_data <- data_memory(
  user_index = test$userId,
  item_index = test$movieId,
  rating = test$rating,
  index1 = TRUE
)

# Since train_data and test_data contain same data as the original dataframes
# Therefore can safely remove train and test dataframes
rm(train, test)


# -----------------------------------------------------------------------------
# Tune
# -----------------------------------------------------------------------------
# The `$tune()` function uses k-fold cross validation to tune the model's params


## ----step3-default-----------------------------------------------------------
# DEFAULT tuning params and options
# Default is used when calling `model$tune()` with no `opts` argument.

# The tune function will look through each combination and pick the best one
# with the min rmse, out of $2^6 = 64$ combinations (6 params, 2 choices each).

# params can be tuned, but options are specified (not tuned)

# Note: P in `costp` means user, Q in `costq` means movies in this case
#       l2 is related to RMSE - the squared error
#       l1 is related to MAE - mean absolute error
best_tune_opts <- list(
  # params
  dim = c(10, 20), # number of factors for matrix factorization
  costp_l1 = c(0, 0.1), # L1 user factors regularization cost
  costp_l2 = c(0.01, 0.1), # L2 user factors regularization cost
  costq_l1 = c(0, 0.1), # L1 movie factors regularization cost
  costq_l2 = c(0.01, 0.1), # L2 movie factors regularization cost
  lrate = c(0.01, 0.1), # learning rate: step size in gradient descent
  # options
  loss = "l2", # error function: `loss = l2` for rmse
  nfold = 5, # number of folds in cross-validation
  niter = 20, # number of iterations
  nthread = 1, # number of threads for parallel computing
  nbin = 20, # number of bins: must be > nthread
  nmf = FALSE, # whether to perform non-negative matrix factorization
  verbose = FALSE, # whether to print progress info in console
  progress = TRUE # whether to print progress bar in console
)

# The default provides a nice starting point, but we want to customize
# the params and options before we call tune function for the first time.

## ----step3-change-options----------------------------------------------------
best_tune_opts$nthread <- 16 # parallel speeds up the process
best_tune_opts$nbin <- 128 # nbin must be > nthread, increase both vars
best_tune_opts$nmf <- TRUE # non-neg matrix factorization since rating>0
best_tune_opts$verbose <- TRUE # see the rmse results in console as we go


# The 2 tuning performance concerns are efficiency and quality.
# Efficiency: Tuning needs to be efficient because its intense computation
#   can take a long time, so we want to **run as few combinations as possible**.
# Quality: With only 2 values to test for each param,
#   we **shouldn't waste on unlikely/bad guesses**

# The goal of tuning is try to find/approach the global minimum of rmse
#   in a few rounds of tuning, but it's easy to get stuck at a local minimum.
# Avoid this by starting out big (test values that are farther apart) in the
#   first call of tuning, then we narrow down, test the best value's neighbors.


## ----step3-change-params-----------------------------------------------------
# Since movies have 18 genres, my guess is that we need at least 20 dimensions
best_tune_opts$dim <- c(20, 40)
# A learning rate step size of 0.01 is not a good guess because it's almost 0,
# try 0.2 instead
best_tune_opts$lrate <- c(0.1, 0.2)
# l1 is related to mean absolute error, not rmse, won't tune these
best_tune_opts$costp_l1 <- 0
best_tune_opts$costq_l1 <- 0
# l2 is related to rmse, and we test a bigger range
best_tune_opts$costp_l2 <- c(0, 0.2)
best_tune_opts$costq_l2 <- c(0, 0.2)
best_tune_opts


# .............................................................................
# Tuning round 1
# Now let's tune, and store the tune result in a variable


## ----step3-tune1, results='hide'---------------------------------------------
# FUNCTION tune
#   INPUT train_data = train_set, opts = train_set_tune_param
#   OUTPUT min = a value for each param that result in lowest rmse
#          res = each combination and its rmse
set.seed(1)
tune_output <- first_model$tune(train_data, opts = best_tune_opts)


## ----step3-tune1-output------------------------------------------------------
# $min shows the combination that yields the minimum rmse
#   The combination with lowest rmse of 0.801:
#   dim = 40, costp_l2 = 0, costq_l2 = 0.2, lrate = 0.1
tune_output$min
# $res shows rmses of all combinations
tune_output$res


## ----step3-tune1-dim-lrate-raster, echo=FALSE -------------------------------
# We can visualize the full result better with 2 raster plots.
# 1. dim-lrate plot: we can `group by` dim and lrate (4 combinations),
#    but because there are 4 rmses for each combination
#    (since l2 factors are not fixed), we take the minimum rmse in `summarize`.
tune_output$res %>%
  group_by(dim = factor(dim), lrate = factor(lrate)) %>%
  summarize(rmse = min(loss_fun)) %>%
  ggplot(aes(dim, lrate, fill = rmse)) +
  geom_raster() +
  labs(
    fill = "Min. rmse",
    title = "Raster Plot of rmse with dim and lrate"
  ) +
  scale_fill_distiller(palette = "Reds")

## ----step3-tune1-l2-raster, echo=FALSE --------------------------------------
# 2. l2 factors plot: similar to first plot, `group by` costp_l2 and costq_l2,
# and `summarize` min rmse
tune_output$res %>%
  group_by(costp_l2 = factor(costp_l2), costq_l2 = factor(costq_l2)) %>%
  summarize(rmse = min(loss_fun)) %>%
  ggplot(aes(costp_l2, costq_l2, fill = rmse)) +
  geom_raster() +
  labs(
    fill = "Min. rmse",
    title = "Raster Plot of rmse with l2 Cost Factors"
  ) +
  scale_fill_distiller(palette = "Blues")


## ----step3-tune1-update------------------------------------------------------
# Synced up with this round's min result
best_tune_opts$dim <- tune_output$min$dim
best_tune_opts$lrate <- tune_output$min$lrate
best_tune_opts$costp_l2 <- tune_output$min$costp_l2
best_tune_opts$costq_l2 <- tune_output$min$costq_l2


# .............................................................................
# Tuning round 2
# Don't need to tune dim and lrate again because rmses don't vary much.
# Tune costq_l2 again, because it can significantly change rmse (0.84 to 0.81).


## ----step3-tune2-param-------------------------------------------------------
# We can verify whether `costq_l2 = 0.2` result in the true minimum of rmse,
#   or if there is a better neighbor of 0.2 that results in an even lower rmse.
# We choose the neighborhood of [0.05, 0.3] with a step size of 0.05.
best_tune_opts$costq_l2 <- seq(0.05, 0.3, 0.05)


## ----step3-tune2, results='hide'---------------------------------------------
# Now let's tune, and see the result!
set.seed(1)
tune_output <- first_model$tune(train_data, opts = best_tune_opts)


## ----step3-tune2-output------------------------------------------------------
# costq_l2 = 0.1 result in the min rmse of 0.797
# We won't tune costq_l2 again, because there isn't much room for improvement.
tune_output$min


## ----step3-tune2-plot, echo=FALSE--------------------------------------------
# plot rmse vs costq_l2
tune_output$res %>%
  ggplot(aes(costq_l2, loss_fun)) +
  geom_point() +
  scale_x_continuous(breaks = best_tune_opts$costq_l2) +
  labs(
    y = "rmse",
    title = "Plot of RMSE vs. Movie l2 Cost Factors"
  )


## ----step3-tune2-update------------------------------------------------------
# Update and print the tuning opts variable,
#   and the tuning stage is now complete!
best_tune_opts$costq_l2 <- tune_output$min$costq_l2
best_tune_opts

rm(tune_output)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
# The `$train()` function will read from training data, and create a model that
#   contains matrices necessary for prediction later.
# We'll be using `train_data` as our training data.


## ----step3-train-------------------------------------------------------------
# FUNCTION train
#   INPUT train_data = train_set,
#         out_model = NULL (model is stored in memory),
#         opts = tune params and options
#   OUTPUT No return value, but $model is populated
set.seed(1)
first_model$train(train_data, opts = best_tune_opts)


# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
# The `$predict()` function predicts unknown ratings in the testing data.
# We'll be using `test_data` as our testing data.

## ----step3-predict-----------------------------------------------------------
# Even though test_set@source[[3]] contains ratings,
# predict function will simply ignore it, and only use user_index, item_index
# FUNCTION predict
#   INPUT test_data = test_set, out_pred = out_memory()
#   OUTPUT a list of predicted ratings
set.seed(1)
pred_ratings <- first_model$predict(test_data, out_memory())

# 5-number summary of test_set predicted ratings
# min = 0.1, Q1 = 3.06, median = 3.56, Q3 = 3.99, max = 6
summary(pred_ratings)


## ----step3-predict-histogram, echo=FALSE-------------------------------------
# Histogram of the predicted ratings: Unlike edx set distribution,
# predicted ratings don't favor whole stars over half stars
# It's a left skewed bell curve.
tibble(rating = pred_ratings) %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5) +
  scale_x_continuous(breaks = seq(0, 6.5, 0.5)) +
  labs(title = "Distribution of test set's Predicted Ratings")


## ----step3-rmse--------------------------------------------------------------
# Make sure ratings are not out of bounds of [0.5, 5]
# setting any rating < 0.5 to 0.5, and any rating > 5 to 5
bound_rating <- function(ratings) {
  sapply(ratings, function(r) {
    if (r > 5) return(5)
    if (r < 0.5) return(0.5)
    return(r)
  })
}

# Function that calculates the rmse: root mean squared error
evaluate_rmse <- function(true, pred) {
  sqrt(mean((true - pred)^2))
}

pred_ratings <- bound_rating(pred_ratings)

# rmse_test is 0.789
rmse_test <- evaluate_rmse(test_data@source[[3]], pred_ratings)
rmse_test


## ----step3-predict-boxplot, echo=FALSE, out.width="50%"----------------------
# Boxplot that compares predicted vs true ratings dist
# The model is a little cautious on predicting higher ratings
rbind(
  tibble(rating = pred_ratings, source = "predicted"),
  tibble(rating = test_data@source[[3]], source = "true")
) %>%
  ggplot(aes(source, rating, color = source)) +
  geom_boxplot() +
  guides(color = "none") +
  labs(title = "Boxplot of test set's \n Predicted vs True Ratings")


## ----step3-predict-jitterplot, echo=FALSE------------------------------------
# Jitter point plot that shows for each true star rating,
#   how far off our predicted ratings are.
# Taking a sample of 5000, otherwise plot has many points and code won't finish
set.seed(1)
tibble(true = test_data@source[[3]], pred = pred_ratings) %>%
  slice_sample(n = 5000) %>%
  ggplot(aes(true, pred, color = factor(true))) +
  geom_point(alpha = 0.005) +
  geom_jitter() +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  scale_y_continuous(breaks = seq(0.5, 5, 0.5)) +
  guides(color = "none") +
  labs(x = "true", y = "predicted",
       title = "test set's Predicted vs True Ratings")


# Rounding is beneficial when many points are concentrated close around true
#   rating, and only few points are far away.
# We are NOT seeing this shape in our jitter plot
#   since many points are far away, so we won't round.


## ----step3-cleanup-----------------------------------------------------------
# Clean up
rm(train_data, test_data, pred_ratings, first_model, rmse_test)


# *****************************************************************************
# STEP 4: Develop the final model
# *****************************************************************************


# -----------------------------------------------------------------------------
# Create final model
# -----------------------------------------------------------------------------


## ----step4-create-model------------------------------------------------------
# convert `edx` and `validation` dataframes to recosystem DataSource objects
# `edx_data` and `validation_data
load("rdas/edx.rda")
load("rdas/validation.rda")

edx_data <- data_memory(
  user_index = edx$userId,
  item_index = edx$movieId,
  rating = edx$rating,
  index1 = TRUE
)

validation_data <- data_memory(
  user_index = validation$userId,
  item_index = validation$movieId,
  rating = validation$rating,
  index1 = TRUE
)

rm(edx, validation)

# create a new model object named `final_model`
final_model <- Reco()


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
# We skip the tuning stage for the final model, because we are satisfied with
# our first model tuning opts, which forms the algorithm.
# We are using the same algorithm, but on the full edx set (not train set).


## ----step4-train-------------------------------------------------------------
# The training stage generates the matrices necessary for the predicting stage.
set.seed(1)
final_model$train(edx_data, opts = best_tune_opts)


# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
# The predicting stage only takes the user and movie ids (ignoring ratings
#   argument) of validation set and outputs predicted ratings.


## ----step4-predict-----------------------------------------------------------
set.seed(1)
pred_ratings <- final_model$predict(validation_data, out_memory())

# 5-number summary
summary(pred_ratings)


## ----step4-predict-histogram, echo=FALSE-------------------------------------
# Histogram of predicted ratings
tibble(rating = pred_ratings) %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5) +
  scale_x_continuous(breaks = seq(0, 6.5, 0.5)) +
  labs(title = "Distribution of validation set's Predicted Ratings")


## ----step4-rmse--------------------------------------------------------------
# Bound predicted ratings inside [0.5, 5] range
pred_ratings <- bound_rating(pred_ratings)

# rmse_validation is 0.786
rmse_validation <- evaluate_rmse(
  validation_data@source[[3]], pred_ratings
)
rmse_validation


## ----step4-predict-boxplot, echo=FALSE---------------------------------------
# Boxplot that compares predicted vs true ratings
rbind(
  tibble(rating = pred_ratings, source = "predicted"),
  tibble(rating = validation_data@source[[3]], source = "true")
) %>%
  ggplot(aes(source, rating, color = source)) +
  geom_boxplot() +
  guides(color = "none") +
  labs(title = "Boxplot of validation set's \n Predicted vs True Ratings")


## ----step4-predict-violinplot, echo=FALSE------------------------------------
# Violin plot is a boxplot with density curves on the side,
# hence the "violin" shape
# We can evaluate how well we did for each star category:
#   Prediction was too high, true rating in bottom 25%: 0.5, 1, 1.5, 2, 2.5
#   Good, true rating in middle half: 3, 3.5, 4
#   Prediction was too low, true rating in top 25%: 4.5, 5
tibble(
  true = validation_data@source[[3]], pred = pred_ratings
) %>%
  ggplot(aes(true, pred, fill = factor(true))) +
  geom_violin(alpha = 0.5, draw_quantiles = c(0.25, 0.5, 0.75)) +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  scale_y_continuous(breaks = seq(0.5, 5, 0.5)) +
  guides(fill = "none") +
  labs(x = "true", y = "predicted",
       title = "validation set's Predicted vs True Ratings")


## ----step4-cleanup-----------------------------------------------------------
# Output rmse for one last time
rmse_validation

# Clean up all the variables
rm(best_tune_opts, edx_data, validation_data, final_model, pred_ratings,
   bound_rating, evaluate_rmse, rmse_validation)
# Delete rdas, ml-10M100K (MovieLens dataset) folders
unlink(c("rdas", "ml-10M100K"), recursive = TRUE)
