from flask import Flask,send_file,request,jsonify
from flask_cors import CORS
import pandas as pd
import json
import base64

# Create Flask app and enable CORS
app = Flask(__name__)
cors = CORS(app)

@app.route("/emotion")
def predictEmotion():
    return "Success"

if __name__ == "__main__":
    app.run()