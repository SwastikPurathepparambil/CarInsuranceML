import csv
from xgboost import XGBClassifier
# csv file name
filename = "train.csv"

features = []
rows = []

with open(filename, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    # get all feature names
    fields = next(csv_reader)
    for row in csv_reader:
        rows.append(row)
    
    # get total number of rows
    print("Total no. of rows: %d" % (csv_reader.line_num))

print('Features: ' + ', '.join(field for field in fields))

dataset_size = 250
X_train = []
y_train = []
for row in rows[:dataset_size]:
    X_train.append(row[:-1])
    y_train.append(row[-1])

# for i in range(25):
#     print(X_train[i])
#     print(y_train[i])
#     print("\n")

# Try 1
model = XGBClassifier()
model.fit(X_train, y_train)
model.predict(X_train[3000])
