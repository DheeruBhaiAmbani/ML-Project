from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = 'model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: {MODEL_PATH} not found. Train the model using train.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not trained or missing.'}), 500
        
    try:
        data = request.json
        input_data = pd.DataFrame([{
            'Fever': float(data.get('Fever', 98.6)),
            'Headache': float(data.get('Headache', 0)),
            'Cough': float(data.get('Cough', 0)),
            'Fatigue': float(data.get('Fatigue', 0)),
            'Body_Pain': float(data.get('Body_Pain', 0))
        }])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities) * 100)
        
        return jsonify({
            'disease': prediction,
            'confidence': f"{confidence:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)