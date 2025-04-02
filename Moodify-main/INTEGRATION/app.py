from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your models
mental_health_model = pickle.load(open('mentalhealth.pkl', 'rb'))
song_recommendation_model = pickle.load(open('songrecc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mental Health Analysis
    data = [
        request.form['a'], request.form['b'], request.form['c'],
        request.form['d'], request.form['e'], request.form['f'],
        request.form['g'], request.form['h'], request.form['i'],
        request.form['j'], request.form['k'], request.form['l'],
        request.form['m'], request.form['n'], request.form['o'],
        request.form['p']
    ]
    arr = np.array([data])
    pred = mental_health_model.predict(arr)
    return render_template('home.html', data=pred)

@app.route('/song', methods=['POST'])
def song():
    # Song Recommendation
    data = [
        request.form['a'], request.form['b'], request.form['c'],
        request.form['d'], request.form['e'], request.form['f'],
        request.form['g'], request.form['h'], request.form['i'],
        request.form['j'], request.form['k'], request.form['l'],
        request.form['m'], request.form['n'], request.form['o']
    ]
    arr = np.array([data])
    pred = song_recommendation_model.predict(arr)
    return render_template('song.html', data=pred)

@app.route('/song-page')
def song_page():
    return render_template('song.html')

if __name__ == '__main__':
    app.run(debug=True)
