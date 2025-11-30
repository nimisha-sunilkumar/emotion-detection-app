from flask import Flask, render_template, request
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = emotion_classifier(text)
    
    # Get the emotion with highest score
    emotion = max(result[0], key=lambda x: x['score'])['label']
    
    return render_template('index.html', prediction_text=f"Emotion Detected: {emotion}")

if __name__ == '__main__':
    app.run(debug=True)
