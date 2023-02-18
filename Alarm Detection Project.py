# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:51:36 2023

@author: Shraddha
"""
#import libraries

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, jsonify, request
import joblib


#creating object
app = Flask(__name__)

# creating Home Page 
@app.route('/')
def home():
    return "Welcome to Alarm Detection Website..."


@app.route('/train_model')
def train():
    #read the dataset
    df = pd.read_excel('Historical Alarm Cases.xlsx')
    # split columns in independent and dependant columns
    x = df.iloc[:,1:7]  # independent variables
    y = df['Spuriosity Index(0/1)']
    logm = LogisticRegression()
    # trainning model
    logm.fit(x,y)
    # save trainning results in "train.pkl" pickle file
    joblib.dump(logm, 'train.pkl')
    return "Model Trained Successfully..."

#creating new page for testing
@app.route('/testing_model', methods = ['POST'])
def test():
    pkl_file = joblib.load('train.pkl')  #load pickle file
    # store requested data
    
    test_data = request.get_json()
    
    # store each value in json file to a seperate variable
    f1 = test_data['Ambient Temperature']
    f2 = test_data['Calibration']
    f3 = test_data['Unwanted substance deposition']
    f4 = test_data['Humidity']
    f5 = test_data['H2S Content']
    f6 = test_data['detected by']
    
    # create list of values 
    my_test_data = [f1,f2,f3,f4,f5,f6]
    my_data_array = np.array(my_test_data)
    
    # reshape array
    test_array = my_data_array.reshape(1,6)
    
    # create dataframe
    df_test = pd.DataFrame(test_array,
                           columns=['Ambient Temperature', 'Calibration', 
                                    'Unwanted substance deposition', 'Humidity',
                                    'H2S Content', 'detected by'])
    
    #testing model
    y_pred = pkl_file.predict(df_test)
    
    if y_pred == 1:
        return "False Alarm, Alarm is not harmful."
    else:
        return "True Alarm, Alarm is harmful."
    
      
# run the app on port
app.run(port=5010)