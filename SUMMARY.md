## Getting your data ready for machine learning

Steps to get the data ready:

1. Load the data using `pandas`
2. Use `pandas` methods to clean the dataset
3. Encode the categorical features to prepare the data to `scikit`

### Useful `pandas` methods for data analyzing and cleaning

```python
import panda as pd
from sklearn import preprocessing

data = pd.read_csv('some_csv.csv') #just an example

# Analyzing
## Looking at the first lines
data.head()
## Additional info about the dataset
data.info()
## See which columns are on the dataset
data.columns
## Check the datatype of the columns
data.dtypes
## Check which values occur on a certain column (e.g., column named color)
data.color.unique()


# Cleaning
## Droping columns
data = data.drop()
## Drop rows without data
data = data.dropna()
## Encoding non-numeric categorical data with LabelEncoder.fit_transform()
encoded_data = data.copy()
le = preprocessing.LabelEncoder()
encoded_data.color = le.fit_transform(encoded_data.color)
## Separating the dataset into two datasets > features and targets
features = encoded_data.drop(['desired labels'], axis=1).values
targets = encoded_data['desired labels'].values
```

## Evaluating a machine learning (ML) model

```python
# Modules being used
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm

# create model (p.s.: here <labels> stands for targets)
model = LogisticRegression
model.fit(features, labels)

# predict with the same data used to train
predictions = model.predict(features)

# metrics
metrics.accuracy_score(labels, predictions)

# ---- testing using different testing sets
# separate the dataset into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.4, random_state=0
)

# scoring using classifiers
clf = svm.SVC(kernel='linear', c=1).fit(features_train, labels_train)
clf.score(features_test, labels_test)
```

Important functions:
- *model.predict()*: used to make predictions based on our model
- *metrics.accuracy_score()*: used to score our predictions
- *train_test_split()*: used to split the dataset into two datasets, one for training and another for testing
- *svm.SVC.score()*: function from the classifier class used to score its results

## Selecting best features for training a model

```python
# additional modules used
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE

lin_reg = LinearRegression()
# calculating mean squared errors from linear regression
cross_val_score(lin_reg, features, labels, cv=10, scoring='neg_mean_squared_error')

# ---- using RFE class
features = data.drop(['targets'], axis=1).values
targets = data.['targets'].values
names = data.columns.values
model = LinearRegression()

rfe = RFE(model, n_features_to_select=1)
rfe.fit(features, labels)
# getting a list of features ranked by the importance of their values to the model
sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
```

Important functions/classes
- *RFE*: Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
- *RFE.fit()*: fits the data to a model
- *RFE.ranking_*: **ranks the most important features**

## Tuning feature performance

```python
from sklearn.grid_search import GridSearchCV

# parameter_grid needs to be defined by the user. This is a dict with all the parameters that can be tuned for our estimator. The grid search method will take them and run different tests with different value and define which set of parameters returned the best score
gridsearchcv = GridSearchCV(estimator=smv.SVC(), param_grid=parameter_grid, n_jobs=1)
gridsearchcv.fit(features_train, targets_train)

# finding the best score from the tunned paramenters
gridsearchcv.best_score_

# --- another example using Lasso
lasso = linear_model.LassoCV()
# using the standard parameters
lasso.fit(features, target)
# checking the value used for alpha using the standard parameters
lasso.alpha_
parameter_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(C=1), parameter_grid, cv=5, scoring='%s_macro' % 'precision')
clf.fit(feataures_train, target_train)
```

Important classes:
- *GridSearchCV*: Exhaustive search over specified parameter values for an estimator.

## Text analysis - sentiment

```python
from sklearn.feature_extraction.text import CountVectorizer
```

### Steps

1. Save text files into a list.
2. Convert list of text files into an array of words that appear on these files.
3. Each line on this array correspond to an entry on the text files list and their respective columns are tagged as '1' if they contain the word.
4. This new array (which represent the features) can be merged with a sentiment column (target) to run machine learning algorithms on it.

### Example:
Let's imagine that our list of text is as follows:

- This is not cool (sentiment negative)
- I am nice (sentiment positive)
- Hello world (sentiment neutral)
- I am the world (sentiment positive)

This is then converted into an array. The columns are made of the words that appeared on the text above and the lines represent each one of the texts above.

||This|is|not|cool|I|am|nice|Hello|world|the|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1|1|1|1|1|0|0|0|0|0|0|
|2|0|0|0|0|1|1|1|0|0|0|
|3|0|0|0|0|0|0|0|1|1|0|
|4|0|0|0|0|1|1|0|0|1|1|

If we have our target already defined based on each one of the texts, we can merge that with the array above to create the dataset to feed our machine learning algorithms. Let's say that if the sentiment is negative we assign 0 to it, 1 if positive and 2 if neutral. Our dataset will then look like this:

||This|is|not|cool|I|am|nice|Hello|world|the|SENTIMENT|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1|1|1|1|1|0|0|0|0|0|0|0|
|2|0|0|0|0|1|1|1|0|0|0|1|
|3|0|0|0|0|0|0|0|1|1|0|2|
|4|0|0|0|0|1|1|0|0|1|1|1|

