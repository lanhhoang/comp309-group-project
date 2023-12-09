#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:14:13 2023

@author: Cong Lanh Hoang
"""

import traceback
import joblib
import sys

import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score

app = Flask(__name__)
cors = CORS(app)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    json_ = request.json
    print(json_)
    query = pd.DataFrame([json_])
    features, model = query.drop("MODEL", axis=1), query["MODEL"]
    print(features)
    print(model)

    if model[0] == "lr":
        try:
            # Load models
            lr_model = joblib.load("./lr_model.pkl")
            print("Logistic Regression Model Loaded")
            lr_y_test = joblib.load("./lr_y_test.pkl")
            lr_y_pred = joblib.load("./lr_y_pred.pkl")
            prediction = list(lr_model.predict(features))
            result = "Sorry! The bike is likely stolen!" if prediction[0] == 0 else "Congrats! The bike is likely recovered!"
            accuracy = accuracy_score(lr_y_test, lr_y_pred)
            print({ "result": result, "accuracy": str(accuracy) })
            return jsonify({ "result": result, "accuracy": str(accuracy) })
        except Exception:
            return jsonify({ "trace": traceback.format_exc() })
    elif model[0] == "dt":
        try:
            # Load models
            dt_model = joblib.load("./dt_model.pkl")
            print("Decisions Tree Model Loaded")
            dt_y_test = joblib.load("./dt_y_test.pkl")
            dt_y_pred = joblib.load("./dt_y_pred.pkl")
            prediction = list(dt_model.predict(features))
            result = "Sorry! The bike is likely stolen!" if prediction[0] == 0 else "Congrats! The bike is likely recovered!"
            accuracy = accuracy_score(dt_y_test, dt_y_pred)
            print({ "result": result, "accuracy": str(accuracy) })
            return jsonify({ "result": result, "accuracy": str(accuracy) })
        except Exception:
            return jsonify({ "trace": traceback.format_exc() })
    else:
        print("Train the model first")
        return jsonify({ "message": "No model here to use" })

if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except Exception:
        port = 12345

    app.run(port=port, debug=True)