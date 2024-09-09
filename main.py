import pandas as pd
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier

# csv file name
filename = "train.csv"

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

train_length = 25000
td = df.head(train_length)

print(td["is_claim"].value_counts())

X = td.drop(columns="is_claim")
y = td["is_claim"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=8)

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8) 
opt.fit(X_train, y_train)
print(opt.best_estimator_)
print(opt.best_score_)
print(opt.score(X_test, y_test))


# X_test = td2.drop(columns="is_claim")
# y_test = td2["is_claim"]
# test --> df2.iloc[0]
