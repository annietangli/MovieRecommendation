# Movie Recommendation

The context of this machine learning project is to create an algorithm for a
movie recommendation system, using the MovieLens dataset. Specifically, this
report presents an algorithm that can effectively predict movie ratings by
users. The goal is to achieve a rmse as low as possible, and our final resulting
rmse is 0.786.

Using recosystem library which supports parallel matrix factorization, the main
focus of this report is model development and its 3 stages: tune, train, and
predict. The recosystem library is flexible because we don't need to get stuck
with the generic recosystem algorithm - it allows us to tune the opts (params
and options) and customize a best recosystem algorithm for our data.

In Step 1: Prepare dataset, we split MovieLens dataset into edx (known) and
validation (unknown) sets.

In Step 2: Describe dataset, we discovered that despite the huge number of
ratings, edx set is in fact very sparse.

In Step 3: Develop the first model, we further split edx set into training
(known) and test (unknown) sets. Tuning with training set, we found the best
recosystem algorithm to use in training stage. Then we trained the model and
predicted test set's ratings, and evaluated a rmse of 0.789. We were satisfied
with this best algorithm, and expected it to perform equally well with
validation set.

In Step 4: Develop the final model, we no longer needed the tuning stage, and
used the same algorithm to train the final model. Finally we predicted
validation set's ratings, and evaluated a rmse of 0.786.
