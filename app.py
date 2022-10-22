from flask import Flask
from flask_cors import CORS
from numpy import array
import tensorflow as tf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template



app = Flask(__name__)

# enable cors
CORS(app)

# load the models from disk
filename2 = r'model2.h5'

# Loading a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
model = load_model(filename2)


@app.route('/')
def home():
    return render_template('index.html')


# Radiation model api endpoint http://127.0.0.1:8000/api/model/radiation
@app.route('/predict',methods=['POST'])
def predict():
    first_interval = request.form.get('first_interval')
    second_interval = request.form.get('second_interval')
    third_interval = request.form.get('third_interval')
    capacity_of_one_panel = request.form.get('capacity_of_one_panel')
    number_of_solar_panels = request.form.get('number_of_solar_panels')
    area_of_panel = request.form.get('area_of_panel')
    



    actual_solar_radiation = array(
        [float(first_interval), float(second_interval), float(third_interval)])
    actual_solar_radiation = actual_solar_radiation.reshape((1, 3, 1))

    capacity_of_one_panel = float(capacity_of_one_panel)
    number_of_solar_panels = float(number_of_solar_panels)
    area_of_panel = float(area_of_panel)
    yield_of_one_panel = (capacity_of_one_panel)/(area_of_panel * 10)


    predictions = model.predict(actual_solar_radiation)*0.75*number_of_solar_panels*area_of_panel*yield_of_one_panel/(1000)

    output = predictions.tolist()

    return render_template('index.html', prediction_text='Predicted Power: kWatts-hour {}'.format(output))


    
    

if __name__ == "__main__":
    app.run(debug=True)  
