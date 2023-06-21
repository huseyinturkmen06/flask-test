import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential, model_from_json
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras
# import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from flask import Flask, jsonify, request, json
from sklearn.preprocessing import OrdinalEncoder
from keras.utils import to_categorical    
import json as use_json
import joblib
import pickle


app = Flask(__name__)
model = joblib.load('backend\svc_model.pkl')
class_labels = {0: "Long Term", 1: "Short Term", 2: "Chill", 3: "Lux Care", 4: "High Mark"}
def create_feature_vector(getParameters):
    feature_vector = np.zeros(39)
    # sleep_time smoking pet luxury gpa rent price
    sleep_time = getParameters["Sleep"]
    hour = int(sleep_time[:2])
    minute = int(sleep_time[3:])
    if "0" in str(hour) or len(str(hour)) == 1:
        hour += 24
    index = int((hour - 21) * 4 + minute // 15)
    feature_vector[index] = 1
    
    # GPA
    average = getParameters["GPA"]
    feature_vector[33] = average
    
    # Smoking
    smoking = getParameters["Smoking"]
    feature_vector[34] = 1 if smoking == "evet" else 0
    
    # Pet
    pet = getParameters["PET"]
    feature_vector[35] = 1 if pet == "evet" else 0
    
    # Luxury
    luxury = getParameters["Luxury"]
    feature_vector[36] = luxury
    
    # Rent Duration
    lease_duration = getParameters["Renting"]
    feature_vector[37] = lease_duration
    
    # Price
    price_range = getParameters["Price"]
    feature_vector[38] = price_range
    
    return feature_vector

@app.route('/predictSVCGet',methods=['GET'])
def predictSVCGet():
    args = request.args
    args = args.to_dict()
    feature_vector = create_feature_vector(args)
    feature_vector = feature_vector.astype(int).reshape(1,-1)
    #print(feature_vector)
    y_pred = model.predict(feature_vector)
    #print(type(y_pred[0].item())) #[3] #.item() --> converts to numpy int32 to native python int
    return np.array_str(y_pred)



@app.route('/husDeneme',methods=['GET'])
def husDeneme():
    
    return "Bu bir deneme yazısıdır"

@app.route("/")
def route():
    return "hello"
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)# debug=True