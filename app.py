from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model
model = joblib.load('weather_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    data = request.get_json()
    features = [
        data.get('tempmax'), data.get('tempmin'), data.get('temp'), 
        data.get('dew'), data.get('humidity'), data.get('windgust'), data.get('windspeed'), data.get('winddir'), data.get('sealevelpressure'),
        data.get('cloudcover'), data.get('visibility'), data.get('solarradiation'), data.get('solarenergy'), data.get('uvindex'),
        data.get('day'), data.get('month')
    ]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': 'rain' if prediction[0] == 1 else 'no rain'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
