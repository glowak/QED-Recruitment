# QED-Recruitment
Recruitment challenge for internship at QED Software

## Introduction
The aim was to create a simple gradient boosting tree-based
model using only the first training dataset and minimal 
data clearning. I chose this particular method because of its
widespread use in contemporary Data Science, simplicity and 
effectiveness when dealing with binary classification problems.
To implement such model I used a Python library **XGBoost**.
Also, I attempted tuning the hyperparameters of the model by using
*GridSearchCV* from the *scikit-learn* library.

## Data clearing
Data clearing was minimal. The training set was split into training data and
label data. All categorical columns were encoded using *OneHotEncoder*
from the *scikit-learn* package on both training and test sets.
Columns *"alert_ids", "client_code", "ip"* and *"categoryname"* were
removed from both datasets. All the other categorical and numerical columns were
used in training the model.

## Hyperparameter tuning using *GridSearchCV*
I attempted tuning the hyperparameters of the model by using
*GridSearchCV* but due to hardware limitations the tuning
was minimal. I decided to pair important parameters into groups
that were checked sequentially, which is not an appropriate method
for parameter tuning. The groups used are listed below.


```python
    parameters1 = {
        "max_depth": range(2, 8, 2),
        "n_estimators": range(1500, 3000, 500),
        "learning_rate": [0.0025, 0.1]
    }
    parameters2 = {
        "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reg_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
    }
```

## Model
I chose to build a model with as few parameters as its 
possible while maintaining an acceptable accuracy.
This choice was mainly made because of hardware limitations.
The final model with its parameters:
```python
estimator = XGBClassifier(objective='binary:logistic',
                          max_depth=6,
                          learning_rate=0.1,
                          n_estimators=1500,
                          reg_aplha=0.7,    # L1 regularization
                          reg_lambda=0.7)   # L2 regularization
```

## Validation
To ensure that the model is working correctly, I decided to 
train and test it on the training *.csv* file using
*train_test_split* with division of 20% for testing and
80% for training. The accuracy of such model was 94%. 
The final results, of course, come from the model that was trained
on the whole *cybersecurity_training.csv* dataset and
*cybersecurity_test.csv* was used for predicting, 
without any splitting.
