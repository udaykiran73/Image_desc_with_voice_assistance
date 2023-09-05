from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import os
from nltk.translate.bleu_score import corpus_bleu
from gtts import gTTS

BASE_DIR = 'static/'

app = Flask(__name__)
app.secret_key = 'caption'
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = BASE_DIR

def load_features_dump():
    with open(os.path.join('features.pkl'), 'rb') as f:
        features = pickle.load(f)
        return features
    
def load_captions_text():
    with open(os.path.join('captions.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()
        return captions_doc

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            #caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            caption = " ".join([word for word in caption.split() if len(word)>1])
            captions[i] = caption

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def generate_caption(image_name, mapping, features, model, tokenizer, max_length):
    image_id = image_name.split('.')[0]
    #img_path = os.path.join('uploads', image_name)
    img_path = BASE_DIR+'uploads/'+image_name
    print(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    captions = mapping[image_id]
    max_length = 35
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    y_pred = captions[-1]
    return y_pred

@app.route('/')
@app.route('/index', methods =['GET', 'POST'])
def index():
    if request.method=='POST':
        f = request.files['image_url']
        file_path = BASE_DIR+'uploads/'+f.filename
        f.save(file_path) 
        
        # load features dump
        features = load_features_dump()

        # load captions text
        captions_doc = load_captions_text()
        #print(captions_doc)

        # load mapping function
        mapping = {}
        for line in tqdm(captions_doc.split('\n')):
        
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            caption = " ".join(caption)
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)

        #load clean mapping function
        clean(mapping)

        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1

        max_length = max(len(caption.split()) for caption in all_captions)

        model = keras.models.load_model('best_model.h5')

        generated_caption = generate_caption(f.filename, mapping, features, model, tokenizer, max_length)

        #generating audio

        language = 'en'
        myobj = gTTS(text=generated_caption, lang=language, slow=False)
        audio_path = BASE_DIR+"uploads/caption.mp3"
        myobj.save(audio_path)
        #print(generated_caption)
        return render_template('index.html', generated_caption=generated_caption, file_path=file_path, audio_path=audio_path)
    return render_template('index.html')