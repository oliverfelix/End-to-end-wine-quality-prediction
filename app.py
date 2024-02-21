from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
            Id = float(request.form['ID'])
       
            data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, Id]
            data = np.array(data).reshape(1, 12)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            logging.info(f"Prediction successful. Result: {predict}")

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            logging.error(f'The Exception message is: {e}')
            return 'Something went wrong. Check the logs for details.'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
