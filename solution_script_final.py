# %%
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
training = pd.read_csv("cybersecurity_training.csv", delimiter="|") 
testing = pd.read_csv("cybersecurity_test.csv", delimiter="|")

def prepare_data(df):
    clean = df.loc[:, ~df.columns.isin(["alert_ids", "client_code", "categoryname",
                                            "notified", "ip"])]
    categorical = df[["categoryname", "ipcategory_name", "ipcategory_scope", 
                    "grandparent_category", "weekday", "dstipcategory_dominate",
                    "srcipcategory_dominate"]]
    numerical = clean.loc[:, ~clean.columns.isin(categorical.columns.values)]
    
    return numerical, categorical


x_train_num, x_train_cat = prepare_data(training)
x_test_num, x_test_cat = prepare_data(testing)


# %%
encoder = OneHotEncoder(sparse_output=False, 
                        categories='auto', 
                        handle_unknown='ignore')

transformed_train = encoder.fit_transform(x_train_cat)
transformed_test = encoder.transform(x_test_cat)
encoded_test = pd.DataFrame(transformed_test)
encoded_train = pd.DataFrame(transformed_train)


# %%
x_train = pd.concat([x_train_num, encoded_train], axis=1)
x_test = pd.concat([x_test_num, encoded_test], axis=1)
y_train = training["notified"]

# X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, stratify=y_train)

# %%
'''
estimator = XGBClassifier(objective='binary:logistic')

# %%
parameters = {
    "max_depth": range(2, 8, 2),
    "n_estimators": range(1500, 3000, 500),
    "learning_rate": [0.0025, 0.1],
    "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
    "reg_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

# %%
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring='roc_auc',
    n_jobs=-1,
    cv=3,
    verbose=True
)

# %%
grid_search.fit(x_train, y_train)

# %%
grid_search.best_params_
'''
# %%
estimator = XGBClassifier(objective='binary:logistic',
                          max_depth=6,
                          learning_rate=0.1,
                          n_estimators=1500,
                          reg_alpha=0.7,
                          reg_lambda=0.7)

# %%
estimator.fit(x_train, y_train)

# %%
# estimator.save_model("saved-models/model.json")

# %%
# preds = estimator.predict(x_test)


# %%
preds_probs = estimator.predict_proba(x_test)

# %%
with open("txt-files/solution4.txt", "w") as file:
    for i in preds_probs:
        file.write(str(i[1]) + "\n")


