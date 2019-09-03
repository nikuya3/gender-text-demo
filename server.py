import numpy as np
import tensorflow as tf
from flask import Flask
from flask import render_template, request
from keras.models import load_model

from common import tokenize

app = Flask(__name__)
app.config['SECRET_KEY'] = 'haBc2aunwKBxZpkn577Q5LoqY'
model = load_model('2019-07-19-17_22_09_vdcnn_blogs_pan13_tr_en.h5')
graph = tf.get_default_graph()


def predict(text):
    x = tokenize(text)
    with graph.as_default():
        probability = model.predict(np.array([x],))[0][0]
    return probability


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict/text', methods=['POST'])
def predictText():
    text = request.form.get('text')
    probability = predict(text)
    gender = 'Female' if probability > 0.5 else 'Male'
    return render_template('index.html', probability=probability, gender=gender)
