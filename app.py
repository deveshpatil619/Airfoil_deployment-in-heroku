import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app,jsonify,url_for,render_template

app =Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods =['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    new_data = [list(data.values())]  ## converting the data into 2-d list
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods =['POST'])

def predict():

    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]  ## converting the data into 2-d array
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html',prediction_text="Airfoil pressure is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)



