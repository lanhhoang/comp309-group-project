#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:52:43 2023

@author: Cong Lanh Hoang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

data = pd.read_csv("./bicycle-thefts-open-data.csv")

# Drop columns
drop_columns = [
    "X", "Y", "OBJECTID", "EVENT_UNIQUE_ID",
    "OCC_DATE", "OCC_DOY", "REPORT_DATE", "REPORT_DOY",
    "LOCATION_TYPE", "NEIGHBOURHOOD_158", "HOOD_140", "NEIGHBOURHOOD_140",
    "LONG_WGS84", "LAT_WGS84"
]

data.drop(drop_columns, axis=1, inplace=True)

# Identify missing value columns
print(len(data) - data.count())

# Handle missing values
bike_make_list = data["BIKE_MAKE"].value_counts()
bike_model_list = data["BIKE_MODEL"].value_counts()
bike_colour_list = data["BIKE_COLOUR"].value_counts()
categorial_imputer = SimpleImputer(
    missing_values=np.nan, strategy="most_frequent")
median_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
encoder = preprocessing.LabelEncoder()
scaler = preprocessing.StandardScaler()

# BIKE_MAKE
unknown_make_list = [
    "UNKOWN", "UNKONWN", "UNKNOWNN",
    "UNKNOWN MAKE", "UNKNOWN", "UNKNOW",
    "UNKNONW", "UNKNONN", "UNKN",
    "UNK", "UKNOWN", "UK",
    "U/K", "(UNK)", "NO"
    "NO NAME", "OTHER", "OTHE",
    "OTH", "OT", "?", "0", "-"
]

specialized_make_list = [
    "SPELIAZED", "SPECILIZED", "SPECILAIZED",
    "SPECILAISED", "SPECIALIZED", "SPECIALIZE",
    "SPECIALIST", "SPECIALISED", "SPEALIZED", "SPEACIALIZED"
]

giant_make_list = data["BIKE_MAKE"][data["BIKE_MAKE"].str.contains(
    "giant", case=False, na=False)].unique().tolist()
giant_make_list.append("GI")

data["BIKE_MAKE"].replace(unknown_make_list, "UNKNOWN", inplace=True)
data["BIKE_MAKE"].replace(specialized_make_list, "SPECIALIZED", inplace=True)
data["BIKE_MAKE"].replace(giant_make_list, "GIANT", inplace=True)
data["BIKE_MAKE"].fillna("UNKNOWN", inplace=True)

# BIKE_MODEL
unknown_model_list = [
    "UNKOWN", "UNKONWN", "UNKNOWN - 1991",
    "UNKNOWN", "UNKNOW", "UNKNONW",
    "UNKNO", "UNKN PARTICULAR", "UNKN",
    "UNKJNOWN", "UNK", "UN",
    "UKN", "UK", "U/K",
    "0", "----", "--", "-", "+"
]

data["BIKE_MODEL"].replace(unknown_model_list, "UNKNOWN", inplace=True)
data["BIKE_MODEL"].fillna("UNKNOWN", inplace=True)

# BIKE_COLOUR
data["BIKE_COLOUR"].replace(
    ["18", "OTH", "OTHBLK", "OTHRED"], np.nan, inplace=True)
# Reshape 1D array to 2D array
bike_colour_data = data["BIKE_COLOUR"].values.reshape(-1, 1)
imputed_bike_colour_data = categorial_imputer.fit_transform(bike_colour_data)
data["BIKE_COLOUR"] = imputed_bike_colour_data

# BIKE_SPEED
data["BIKE_SPEED"].replace(0, np.nan, inplace=True)
# Reshape 1D array to 2D array
bike_speed_data = data["BIKE_SPEED"].values.reshape(-1, 1)
imputed_bike_speed_data = median_imputer.fit_transform(bike_speed_data)
data["BIKE_SPEED"] = imputed_bike_speed_data

# BIKE_COST
# replace zero with NaN
data["BIKE_COST"].replace(0, np.nan, inplace=True)
# fill NaN value with median value
bike_cost_data = data["BIKE_COST"].values.reshape(-1, 1)
imputed_bike_cost_data = median_imputer.fit_transform(bike_cost_data)
data["BIKE_COST"] = imputed_bike_cost_data

## OCC_MONTH & REPORT_MONTH
# replace month name with number
data["OCC_MONTH"] = data["OCC_MONTH"].apply(
    lambda month_name: datetime.strptime(month_name, "%B").month)
data["REPORT_MONTH"] = data["REPORT_MONTH"].apply(
    lambda month_name: datetime.strptime(month_name, "%B").month)

# add RECOVERED column
data["RECOVERED"] = [1 if status == "RECOVERED" else 0 for status in data["STATUS"]]
data.drop("STATUS", axis=1, inplace=True)

# balance the data
print(data["RECOVERED"].value_counts())
data_majority = data[data["RECOVERED"] == 0]
data_minority = data[data["RECOVERED"] == 1]
# upsample minority class
upsampled_data_minority = resample(
    data_minority,
    replace=True,
    n_samples=33923,
    random_state=123
)

data = pd.concat([data_majority, upsampled_data_minority])
print(data["RECOVERED"].value_counts())

# transform categorical data to numeric
categoricals = [col for col, col_type in data.dtypes.items() if col_type == "O"]
print(categoricals)
for col in categoricals:
    data[col] = encoder.fit_transform(data[col])

# split the data
features, target = data.drop("RECOVERED", axis=1), data["RECOVERED"]
column_names = features.columns
scaled_features = scaler.fit_transform(features)
features = pd.DataFrame(scaled_features, columns=column_names)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Logistic Regression
# lr_model = LogisticRegression(solver="liblinear", random_state=0)
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(x_train, y_train)

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

score = np.mean(cross_val_score(lr_model, x_train, y_train,
                scoring="accuracy", cv=crossvalidation, n_jobs=1))
print('The score of the 10 fold run is: ', score)

y_pred = lr_model.predict(x_test)

print('Classification Report(N): \n', classification_report(y_test, y_pred))
print('Confusion Matrix(N): \n', confusion_matrix(y_test, y_pred))
print('Accuracy(N): \n', accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

plt.style.use("ggplot")
cm = confusion_matrix(y_test, y_pred, labels=lr_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

joblib.dump(lr_model, "./lr_model.pkl")
print("Model dumped!")

joblib.dump(column_names, "./lr_model_columns.pkl")
print("Model columns dumped!")