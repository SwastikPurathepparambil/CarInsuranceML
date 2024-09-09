import pandas as pd
from xgboost import XGBClassifier

# csv file name
filename = "train.csv"

df = pd.read_csv("train.csv")

train_length = 500
td = df.head(train_length)

print(td["is_claim"].value_counts())

X = td.drop(columns="is_claim")
y = td["is_claim"]

model = XGBClassifier()
model.fit(X, y)
model.predict(df.iloc[1000])