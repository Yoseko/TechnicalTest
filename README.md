# data-scientist-technical-test

My code for this technical test, provided in `technical_test.ipynb`, consists of the following parts:
* Import data from Kaggle
* Data preprocessing
* Hyperparameter tuning and label probability prediction with Logistic Regression
* Hyperparameter tuning and label probability prediction with Random Forest
* Hyperparameter tuning and label probability prediction with Support Vector Machine

### Import data from Kaggle
Train and test data are downloaded with `kaggle competitions download -c auto-insurance-fall-2017`.

As indicated by the instruction, the files `SHELL_AUTO` and `MEAN_AUTO` are ignored and only `train_auto` and `test_auto` are used.

### Data preprocessing
Data from `train_auto` and `test_auto` are first combined into a single dataframe so that the feature variables (columns other than `TARGET_FLAG` and `TARGET_AMT`) can be preprocessed at the same time. 
Once the preprocessing of feature variables is done, the combined dataframe will be divided into 2 dataframes for training/prediction based on the numbers of rows of these 2 original files.

As we don't have information on `TARGET_AMT` in `test_auto`, this column has been dropped.

For the preprocessing of feature variables, after some exploration like checking NA values and numbers of unique values in each column, the following steps have been taken:
* Use (0,1) label to encode categorical features that only have 2 categories
* Use Ordinal Encoding for categorical features that have ordinal nature
* Use One Hot Encoding for categorical features that have no ordinal nature (NA values are filled with `Unknown`)
* Convert columns with `$` sign from string to float and impute missing data in these columns
* Rescale the features with `MinMaxScaler()` (for Logistic Regression)

Train-test 80-20 split is also applied for the original train data from `train_auto` for the following modelling parts.

### Hyperparameter tuning and label probability prediction 
As this project is a binary classification task where we have two labels to assign to `TARGET_FLAG`, I consider the following algorithms suitable for modelling:
Logistic Regression,
k-Nearest Neighbors,
Decision Trees / Random Forest,
Support Vector Machine.

Here I implemented Logistic Regression, Random Forest and Support Vector Machine. 

* For Logistic Regression, `GridSearchCV` (5-fold) and `RFE` (Recursive Feature Elimination) are used to find the optimal number of features `n_features_`.

* For Random Forest, as it is more resource-consuming than Logistic Regression, `RandomizedSearchCV` (5-fold),
instead of Grid Search Cross-Validation, and `RFE` are used to find the optimal `n_features_`, `max_depth`, `min_samples_split`, `min_samples_leaf`.

* For Support Vector Machine, `GridSearchCV` (5-fold) is used to find the optimal `C` and `gamma`. `RFE` isn't appied for SVM only for time-saving purposes.

### Performance metrics of different models
For this binary classification task, a few metrics are used to compare the performance of the models mentioned above: 
`f1_score`,
`roc_auc_score`, 
`accuracy_score`.
`accuracy_score` may not be very useful as the labels in our test data are imbalanced (there are much more Label 0 than Label 1).

After comparing the performance metrics of each model, we can conclude that Logistic Regression performs slightly better than Random Forest and Support Vector Machine in this project.
* Logistic Regression:
`f1_score` for the model is 0.517,
`roc_auc_score` for the model is 0.674,
`accuracy_score` for the model is 0.785.
* Random Forest:
`f1_score` for the model is 0.456,
`roc_auc_score` for the model is 0.643,
`accuracy_score` for the model is 0.783.
* Support Vector Machine:
`f1_score` for the model is 0.462,
`roc_auc_score` for the model is 0.645,
`accuracy_score` for the model is 0.786.

However, for real-life applications we would need more precise parameter tuning and careful evaluations to draw a more accurate conclusion.
