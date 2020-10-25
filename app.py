from flask import Flask, jsonify, request
import os
import json
from datetime import datetime
import sys
import joblib
from pythainlp.tokenize import word_tokenize
import re
import numpy as np
import pandas as pd



app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def load_model():
	model = joblib.load("./model/clf_log_tfidf_20201022.joblib")
	vectorizer = joblib.load("./model/vectorizer_20201022.joblib")
	return model, vectorizer

def text_treatment(_text):   
    result = re.sub(r"http\S+", "", _text)
    result = result.replace("\n", "")    
    return result 

def inference(_user_input):

	treat_text = text_treatment(_user_input)

	test_ls = word_tokenize(treat_text)
	test_ls = " ".join(test_ls)
	X_test = vectorizer.fit_transform([test_ls])

	if X_test.toarray().sum() == 0:
		return 0
	else:
		y_test_prob = loaded_model.predict_proba(X_test)
		y_val = y_test_prob[0][0]
		return y_val

@app.route("/api")
def getAPI():
	user_input = request.args.get("text", type=str, default="")
	y_val = inference(user_input)
	return {"io_prob": y_val}

@app.route("/hello")
def hello():
	return "hello world"

if __name__ == "__main__":
	loaded_model, vectorizer = load_model()
	app.run()