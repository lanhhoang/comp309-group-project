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

app = Flask(__name__)


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
            prediction = list(lr_model.predict(features))
            result = "Sorry! The bike is likely stolen!" if prediction[0] == 0 else "Congrats! The bike is likely recovered!"
            accuracy = 0
            print({ "result": result, "accuracy": str(accuracy) })
            return jsonify({ "result": result, "accuracy": str(accuracy) })
        except Exception:
            return jsonify({ "trace": traceback.format_exc() })
    elif model[0] == "dt":
        try:
            prediction = list(dt_model.predict(features))
            result = "Sorry! The bike is likely stolen!" if prediction[0] == 0 else "Congrats! The bike is likely recovered!"
            accuracy = 0
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

    # Load models
    lr_model = joblib.load("./lr_model.pkl")
    print("Logistic Regression Model Loaded")
    # Load models
    dt_model = joblib.load("./dt_model.pkl")
    print("Decisions Tree Model Loaded")

    app.run(port=port, debug=True)