from flask import Flask, render_template, request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and tokenizer
loaded_data = joblib.load('Hate_Speech_Detection.pkl')

model = loaded_data['model']
tokenizer = loaded_data['tokenizer']

max_length = 55  # Assuming the maximum length used during training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        print("sentence: ", sentence)
        sentence = str(sentence)

        # Tokenize and pad the input sentence
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])
        print("tokenized_sentence: ", tokenized_sentence)
        padded_sentence = pad_sequences(tokenized_sentence, padding='post', maxlen=max_length)

        # Make prediction using the loaded model
        prediction = model.predict(padded_sentence).argmax(axis=1)
        print(prediction)
        # Determine the prediction result (customize as needed)
        if prediction == 0:
            result = 'Not Hate Speech'
        else:
            result = 'Hate Speech'

        return render_template('result.html', sentence=sentence, result=result)

if __name__ == '__main__':
    app.run(debug=True)
