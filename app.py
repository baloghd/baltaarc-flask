import os
import pickle
from typing import List

from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

import face_recognition
import pandas as pd

ROOT_FOLDER = '/home/domi/PycharmProjects/baltaarc-flask/'
UPLOAD_FOLDER = 'static/files/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_similars(filename: str, howmany: int = 4):
    with open("model", "rb") as model:
        model_pickle = pickle.load(model)
        model_ready = True
        print("model loaded")
    face_encodings = model_pickle
    image_to_test = face_recognition.load_image_file(ROOT_FOLDER + UPLOAD_FOLDER + filename)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
    known_encodings = [x[1] for x in face_encodings]
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
    s = sorted(list(zip([f[0] for f in face_encodings], face_distances)), key = lambda x: x[1])
    print(s[:10])
    top_n_similar_files = lambda n: ["dataset/" + "_".join(image[0]) for image in s[:n]]
    return s, top_n_similar_files(howmany)

def histogram(similarities: List, howmany: int = 20):
    df = pd.DataFrame(similarities)
    df[0] = df[0].apply(lambda x: " ".join(x[:-1]))
    hist = df[0][:howmany].value_counts()
    hist = list(zip(list(hist.index), map(lambda y: y * (100 / howmany), list(hist.values))))
    return hist

@app.route("/", methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            similarities, similar_files = get_similars(filename)
            jogalapok = [" ".join(s.split("/")[-1].split("_")[:-1]) for s in similar_files]


            return render_template("base.html",
                                   file = filename,
                                   hasonlok = list(zip(similar_files, jogalapok)),
                                   histogram = histogram(similarities, 20))

    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug = True)