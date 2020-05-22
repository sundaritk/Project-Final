from flask import Flask, jsonify
import numpy as np
import json
import requests
from flask import render_template
<<<<<<< HEAD
from model_train import record, playback, main
=======
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import librosa
import glob
import librosa.display
import keras
from flask import render_template
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from numpy import load
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
>>>>>>> 5d0943182fa47d9efac9fd9d0166f6b756c2076c

app = Flask(__name__)

@app.route('/record')
def record1():

#function for recording an .wav audio file, records 5 seconds
    record()
    return render_template("index.html")

@app.route('/playback')
def playback1():
    playback()
#function for playing back/listening to created audio file
    return render_template("index.html")

@app.route('/predict')
def model():

<<<<<<< HEAD
    gender,emotion=main()
    print(emotion)
    print(gender)
# Voice regognition and emotion prediction
    return render_template("index.html",emotion=emotion,gender=gender)
=======
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.savefig("Resources/images/spectrogram.png")

    def draw_pitch(pitch):
        # Extract selected pitch contour, and  replace unvoiced samples by NaN to not plot
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        plt.ylabel("fundamental frequency [Hz]")

    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])
    plt.savefig("Resources/images/spectrogram_0.03.png")

    #livedf= pd.DataFrame(columns=['feature'])
    X, sample_rate = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    
    # plt.savefig("Resources/images/test1.png") # for testing

    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    livedf2
    
    # plt.savefig("Resources/images/test2.png") # for testing
    
    json_file = open('Resources/data/analysis_output/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # plt.savefig("Resources/images/readdone.png") # for testing
    # load weights into new model
    loaded_model.load_weights("Resources/data/saved_models/Emotion_Voice_Detection_Model.h5")
    
    # plt.savefig("Resources/images/test3.png") # for testing
    
    twodim= np.expand_dims(livedf2, axis=2)

    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

    livepreds1=livepreds.argmax(axis=1)
    
    # plt.savefig("Resources/images/test4.png")
    
    liveabc = livepreds1.astype(int).flatten()
    print(liveabc)
    lb = LabelEncoder()
    y_train=load('static/py/y_train.npy',allow_pickle=True)
    y_test=load('static/py/y_test.npy',allow_pickle=True)
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    livepredictions = str(lb.inverse_transform((liveabc))[0])
    gender_emotion = livepredictions.split('_')
    gender=gender_emotion[0].capitalize()
    emotion=gender_emotion[1].capitalize()
    
    output = {
      'gender': gender,
      'emotion': emotion
    }
    return render_template('index.html',result=output)
>>>>>>> 5d0943182fa47d9efac9fd9d0166f6b756c2076c

# Define what to do when a user a specific route
@app.route("/")
def index1():
    return render_template("index.html")

@app.route("/data.html")
def data():
    return render_template("data.html")

@app.route("/about.html")
def about():
    # print("Server received request for 'Home' page...")
     return render_template("about.html")

@app.route("/license.html")
def license():
    return render_template("license.html")

@app.route("/index.html")
def index2():  
    return render_template("index.html")

# run app
if __name__ == "__main__":
<<<<<<< HEAD
    app.run(host="127.0.0.1",port=5000,threaded=False)
=======
    app.run(threaded=False)
>>>>>>> 5d0943182fa47d9efac9fd9d0166f6b756c2076c
