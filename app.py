import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
from movie_recommendation import predict_movies

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	movie = [x for x in request.form.values()]
	pre_movies = predict_movies(movie[0])
	text = ""
	i = 1
	for m in pre_movies:
		text = text + "{}. {}<br>".format(i,m)
		i = i+1
	return render_template('index.html',prediction_text="Recommended Movies : " + text)

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
# 	'''
# 	For direct API calls through request
# 	'''
# 	

if __name__ == "__main__":
	app.run(debug=True)