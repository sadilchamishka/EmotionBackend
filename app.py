from flask import Flask,send_file,request,jsonify
from flask_cors import CORS
import pandas as pd
from scipy.io import arff

# Create Flask app and enable CORS
app = Flask(__name__)
cors = CORS(app)

@app.route("/emotion",methods = ['POST'])
def predictEmotion():
	print("*******************")
	f = request.files['file']
	f.save('data.arff')
	data_train = arff.loadarff('data.arff')
	df_train = pd.DataFrame(data_train[0])
	return str(df_train.values[0][10])

if __name__ == "__main__":
    app.run()