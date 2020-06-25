from flask import Flask,send_file,request,jsonify
from flask_cors import CORS
import pandas as pd
from scipy.io import arff
from models import SimpleAttention,MatchingAttention,DialogueRNNCell,DialogueRNN,BiModel
import torch

D_m = 2000
D_g = 150
D_p = 150
D_e = 100
D_h = 100
D_a = 100
n_classes = 6

model = BiModel(D_m, D_g, D_p, D_e, D_h,
                  n_classes=n_classes,
                  listener_state=False,
                  context_attention='general',
                  dropout_rec=0.1,
                  dropout=0.1) 

model.load_state_dict(torch.load('rnn_model.pt'))
model.eval()
		
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
	acouf = torch.FloatTensor([df_train.values[0]])
	qmask = torch.FloatTensor([[[1,0]]])
	umask = torch.FloatTensor([[1]*1])
	
	log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask,umask)
	lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2])
	pred_ = torch.argmax(lp_,1)
	print("-----------------")
	print(pred_)
	return str(pred_)

if __name__ == "__main__":
	BiModel.__module__ = "BiModel"
	app.run()