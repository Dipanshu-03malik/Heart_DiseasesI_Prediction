# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 19:27:19 2024

@author: deepa
"""

# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("D:\Heart_DiseasesI_Prediction/Heart2.csv")

# Preprocessing
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], data['fbs'],
                data['restecg'], data['thalach'], data['exang'], data['oldpeak'], data['slope'],
                data['ca'], data['thal']]
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    
    
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)




