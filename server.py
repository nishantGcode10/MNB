# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl','rb'))
cv=pickle.load(open("vectorizer.pickle", 'rb')) 
@app.route('/api',methods = ['POST'])
def predict():
    # Get the data from the POST request.
    #data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    #cv=CountVectorizer()
    data=cv.transform([request.args.get('headline')]).toarray()
    prediction = model.predict(data)
    # Take the first value of prediction
    #output = prediction[0]
    print(prediction)
    return prediction[0]
if __name__ == '__main__':
    app.run()