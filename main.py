import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time

# csv file name
filename = "train.csv"

df = pd.read_csv("train.csv")

df.drop(
    columns=['policy_id', 'is_power_door_locks', 'is_central_locking', 
                 'length', 'is_speed_alert', 'airbags'], axis=1, inplace=True
    )




# Check for Data Types
# for column_name in df.columns:
#     print(f"{column_name}: {df[column_name].dtype} \n{df[column_name].unique()}")

train_length = 50000
td = df.head(train_length)
X = td.drop(columns=["is_claim"], axis=1).copy()
y = td["is_claim"].copy()

# ONE HOT ENCODING LETS GOOOO
X_encoded = pd.get_dummies(X, columns=['area_cluster', 'segment', 'model', 'fuel_type', 
                                       'max_torque', 'max_power', 'engine_type', 'is_esc', 
                                       'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 
                                       'is_parking_camera', 'rear_brakes_type', 'transmission_type', 
                                       'steering_type', 'is_front_fog_lights', 'is_rear_window_wiper', 
                                       'is_rear_window_washer', 'is_rear_window_defogger', 
                                       'is_brake_assist', 'is_power_steering', 'is_driver_seat_height_adjustable', 
                                       'is_day_night_rear_view_mirror', 'is_ecw'])


# Check for Data Types after one-hot encoding
# for column_name in X_encoded.columns:
#     print(f"{column_name}: {X_encoded[column_name].dtype} \n{X_encoded[column_name].unique()}")

# The stratify=y is included because there are much fewer of one classification than the other
# Stratified Random Sample allows percentage to stay the same

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

clf = xgb.XGBClassifier(
    objective="binary:logistic", 
    eval_metric="auc",
    colsample_bytree=1.0, 
    gamma=0.0,
    learning_rate=0.10292490457670343, 
    max_depth=8, 
    min_child_weight=1, 
    n_estimators=250, 
    reg_lambda=1.0, 
    subsample=0.6,
    scale_pos_weight=(len(y_resampled) - sum(y_resampled)) / sum(y_resampled)
)

clf.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

start_time = time.time() 
 
print(clf.predict(X_test))
 
end_time = time.time() 
execution_time = end_time - start_time 
print(f"Execution time: {execution_time} seconds") 

# ConfusionMatrixDisplay.from_estimator(
#     clf, X_test, y_test
# )

# plt.show()







# The part that takes an hour to do
# # Load the data
# df = pd.read_csv("train.csv")

# # Drop columns
# df.drop(
#     columns=['policy_id', 'is_power_door_locks', 'is_central_locking',
#                  'length', 'is_speed_alert', 'airbags'], axis=1, inplace=True
#     )

# # Select a subset of the data (adjust train_length as needed)
# train_length = 50000
# td = df.head(train_length)

# # Separate features and target variable
# X = td.drop(columns=["is_claim"], axis=1).copy()
# y = td["is_claim"].copy()

# # One-hot encode categorical features
# X_encoded = pd.get_dummies(X, columns=['area_cluster', 'segment', 'model', 'fuel_type',
#                                        'max_torque', 'max_power', 'engine_type', 'is_esc',
#                                        'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
#                                        'is_parking_camera', 'rear_brakes_type', 'transmission_type',
#                                        'steering_type', 'is_front_fog_lights', 'is_rear_window_wiper',
#                                        'is_rear_window_washer', 'is_rear_window_defogger',
#                                        'is_brake_assist', 'is_power_steering', 'is_driver_seat_height_adjustable',
#                                        'is_day_night_rear_view_mirror', 'is_ecw'])

# # Apply SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
# )

# # Pipeline with scaling and classifier
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="auc",
#         use_label_encoder=False,  # Updated for newer versions of xgboost
#         random_state=42,
#         scale_pos_weight=(len(y_resampled) - sum(y_resampled)) / sum(y_resampled)  # Handle class imbalance
#     ))
# ])

# # Hyperparameter search space
# param_space = {
#     'classifier__learning_rate': Real(0.01, 0.3, 'log-uniform', name='learning_rate'),
#     'classifier__max_depth': Integer(3, 8, name='max_depth'),
#     'classifier__gamma': Real(0.0, 1.0, 'uniform', name='gamma'),
#     'classifier__reg_lambda': Real(1.0, 100.0, 'log-uniform', name='reg_lambda'),
#     'classifier__subsample': Real(0.6, 1.0, 'uniform', name='subsample'),
#     'classifier__colsample_bytree': Real(0.6, 1.0, 'uniform', name='colsample_bytree'),
#     'classifier__min_child_weight': Integer(1, 10, name='min_child_weight'),  # Added
#     'classifier__n_estimators': Integer(100, 500, name='n_estimators')  # Added
# }

# # Bayesian Optimization with StratifiedKFold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# bayes_search = BayesSearchCV(
#     pipeline,
#     param_space,
#     n_iter=100,  # Increase iterations for better hyperparameter tuning
#     cv=cv,
#     n_jobs=-1,
#     scoring='roc_auc'
# )

# # Fit the model
# bayes_search.fit(X_train, y_train)

# # Evaluate and print results
# best_clf = bayes_search.best_estimator_
# y_pred = best_clf.predict(X_test)
# y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

# print("Best parameters:", bayes_search.best_params_)
# print("Best ROC AUC score on training data:", bayes_search.best_score_)

# # Calculate AUC score on test data
# test_auc = roc_auc_score(y_test, y_pred_proba)
# print("ROC AUC score on test data:", test_auc)

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm).plot()
# plt.show()




