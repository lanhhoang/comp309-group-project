#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:06:30 2023

@author: Cong Lanh Hoang
"""

import pandas as pd
import matplotlib.pyplot as plt

import os

path = "./"
filename = "bicycle-thefts-open-data.csv"
filepath = os.path.join(path, filename)

data = pd.read_csv(filepath)

print(data.head(100))
print(data.info())
print(data.shape)
print(data.dtypes)

occ_year_list = data["OCC_YEAR"].value_counts()
occ_month_list = data["OCC_MONTH"].value_counts()
occ_dow_list = data["OCC_DOW"].value_counts()
occ_day_list = data["OCC_DAY"].value_counts()
occ_hour_list = data["OCC_HOUR"].value_counts()
report_year_list = data["REPORT_YEAR"].value_counts()
report_month_list = data["REPORT_MONTH"].value_counts()
report_dow_list = data["REPORT_DOW"].value_counts()
report_day_list = data["REPORT_DAY"].value_counts()
report_hour_list = data["REPORT_HOUR"].value_counts()
division_list = data["DIVISION"].value_counts()
premises_type_list = data["PREMISES_TYPE"].value_counts()
bike_make_list = data["BIKE_MAKE"].value_counts()
bike_model_list = data["BIKE_MODEL"].value_counts()
bike_type_list = data["BIKE_TYPE"].value_counts()
bike_speed_list = data["BIKE_SPEED"].value_counts()
bike_colour_list = data["BIKE_COLOUR"].value_counts()
bike_cost_list = data["BIKE_COST"].value_counts()
hood_list = data["HOOD_158"].value_counts()
neighbourhood_list = data["NEIGHBOURHOOD_158"].value_counts()
status_list = data["STATUS"].value_counts()

# Bike Cost Distribution
print(data['BIKE_COST'].describe())

bike_costs = data["BIKE_COST"]

plt.figure(figsize=(10, 10))

plt.hist(bike_costs, bins=32)

plt.xlabel("Cost [CAD]")
plt.ylabel("Frequency")
plt.title("Bike Cost Histogram")

plt.show()

# Average Bike Cost by Premises
avg_bike_cost = data.groupby("PREMISES_TYPE")["BIKE_COST"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 10))

plt.barh(avg_bike_cost.index, avg_bike_cost.values)

plt.xlabel("Average Bike Cost [CAD]")
plt.ylabel("Premise Type")
plt.title("Average Bike Cost by Premise Type")

plt.show()

# Bike Thefts Frequency by Year
subset = data[data["OCC_YEAR"] >= 2014]

plt.figure(figsize=(10, 10))

plt.hist(subset["OCC_YEAR"], bins=20)

plt.xlabel("Year")
plt.ylabel("Total number of Bike Thefts")
plt.title("Bike Thefts Frequency by Year")

plt.show()

# Bike Thefts Frequency by Month
plt.figure(figsize=(10, 10))

plt.hist(subset["OCC_MONTH"], bins=24)

plt.xticks(rotation=45, ha="right")  # Rotate labels by 45 degrees and align them to the right
plt.xlabel("Month")
plt.ylabel("Total number of Bike Thefts")
plt.title("Bike Thefts Frequency by Month")

plt.show()

# Bike Thefts Frequency by Day of Month
plt.figure(figsize=(10, 10))

plt.hist(subset["OCC_DAY"], bins=62)

plt.xlabel("Day of Month")
plt.ylabel("Total number of Bike Thefts")
plt.title("Bike Thefts Frequency by Day of Month")

plt.show()

# Bike Thefts Frequency by Day of Week
plt.figure(figsize=(10, 10))

plt.hist(subset["OCC_DOW"], bins=14)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.xticks(range(7), day_order, rotation=45, ha="right")  # Rotate labels by 45 degrees and align them to the right
plt.xlabel("Day of Week")
plt.ylabel("Total number of Bike Thefts")
plt.title("Bike Thefts Frequency by Day of Week")

plt.tight_layout()  # Adjust the spacing between the subplots to prevent label overlap

plt.show()

# Bike Thefts Frequency by Hour
plt.figure(figsize=(10, 10))

plt.hist(subset["OCC_HOUR"], bins=48)

plt.xlabel("Hour")
plt.ylabel("Total number of Bike Thefts")
plt.title("Bike Thefts Frequency by Hour")

plt.show()

# Status Percentage
plt.figure(figsize=(10, 10))

plt.pie(status_list.values, labels=status_list.index, autopct="%1.1f%%")

plt.title("Status Percentage")
plt.axis("equal")

plt.show()

# Bike Status by Month
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

crosstab_data = pd.crosstab(data["OCC_MONTH"], data["STATUS"])
crosstab_data = crosstab_data.reindex(month_order)

plt.figure(figsize=(10, 10))

crosstab_data.plot(kind="bar", stacked=True)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Month")
plt.ylabel("Status Frequency")
plt.title("Bike Status by Month")

plt.show()


# Bike Status by Division
crosstab_data = pd.crosstab(data["DIVISION"], data["STATUS"])

plt.figure(figsize=(10, 10))

crosstab_data.plot(kind="bar", stacked=True)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Division")
plt.ylabel("Status Frequency")
plt.title("Bike Status by Division")

plt.show()

# Bike Type
plt.figure(figsize=(10, 10))

plt.hist(data["BIKE_TYPE"], bins=26)

plt.xlabel("Bike Type")
plt.ylabel("Total number of Bike Type")
plt.title("Bike Type Distribution")

plt.show()

# Premises Type
plt.figure(figsize=(10, 10))

plt.pie(premises_type_list.values, labels=premises_type_list.index, autopct="%1.1f%%")

plt.title("Premises Type Percentage")
plt.axis("equal")

plt.show()