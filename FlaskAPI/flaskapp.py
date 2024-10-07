from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model and vectorizer
model_rf = joblib.load(os.path.join('models', 'random_forest_model.pkl'))
vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))

# Define route for inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_text = data['text']

        # Vectorize input text
        vectorized_text = vectorizer.transform([input_text])

        # Make prediction
        prediction = model_rf.predict(vectorized_text)

        # Convert prediction to standard Python int
        prediction_value = int(prediction[0])  # Assuming it's a single output

        # Send back prediction result as a JSON response
        response = {
            'input_text': input_text,
            'prediction': prediction_value
        }
        if response == 0:
            sentiment = "negative"
        else:
            sentiment = "positive"
        return jsonify(sentiment)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
