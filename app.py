
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

model = load_model("emotion_lstm_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
class_names = np.load("classes.npy")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        sequences = tokenizer.texts_to_sequences([text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post')
        prediction = model.predict(padded_seq)
        predicted_class = class_names[np.argmax(prediction)]
        return jsonify({"emotion": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
