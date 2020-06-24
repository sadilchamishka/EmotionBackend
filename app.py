from flask import Flask,send_file,request,jsonify
from flask_cors import CORS
import pandas as pd
from scipy.io import arff

# Create Flask app and enable CORS
app = Flask(__name__)
cors = CORS(app)

@app.route("/emotion",methods = ['POST'])
def predictEmotion():
	f = request.files['file']
	data_train = arff.loadarff(f)
    return "Success"

if __name__ == "__main__":
    app.run()