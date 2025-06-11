from flask import Flask,request,jsonify,render_template # type: ignore          # request: to handle HTTP requests (like form data).
import pandas as pd                                     # type: ignore          # render_template: used to render HTML templates.
import numpy as np                                      # type: ignore
import pickle                                                                   # pickle: used to load your pre-trained machine learning models.
from sklearn.preprocessing import StandardScaler        # type: ignore          # jsonify: to return JSON responses if needed.
                                                                
# Here you create a Flask application object.
# application is assigned Flask(__name__), then app = application just gives you another name to refer to the Flask app.
# Sometimes people use application when deploying to certain platforms (like AWS Elastic Beanstalk), but usually, just app is fine.
application=Flask(__name__)                                     
app=application


# import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('basic reg project/ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('basic reg project/scaler.pkl','rb'))


# This defines the home page of your web app.
# When the user goes to http://localhost:5000/, Flask will return the index.html page.
# You need to have an index.html file in the templates folder (Flask automatically looks for templates inside templates/).
@app.route("/")
def index():
    return render_template('index.html')



# This function will handle both GET and POST requests.
# GET request: happens when user simply opens the page (home.html) without submitting any form.
# POST request: happens when user submits form data (i.e. input values for prediction).
@app.route('/predictdata',methods=['GET','post'])
def predict_datapoint():
    if request.method=='POST':                                       # Check if the form is submitted.
        Temperature=float(request.form.get('Temperature'))          # You are retrieving form inputs one by one using request.form.get('<field_name>').
        RH = float(request.form.get('RH'))                          # You convert all values to float because ML models expect numerical input.
        Ws = float(request.form.get('Ws'))                          # These names 'Temperature', 'RH', 'Ws', etc. must exactly match the name attributes of your input fields in your home.html form.
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=Standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])              #The output result is an array (since model outputs are usually arrays), so result[0] extracts the scalar value.

    else:
        return render_template("home.html")                                 #When user directly visits /predictdata, without submitting anything, the home.html template will simply load.

# This checks if the file is being run directly (not imported as a module).
# app.run() starts the Flask development server on localhost:5000 by default.
if __name__=="__main__":
    app.run()