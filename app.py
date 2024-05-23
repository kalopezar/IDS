from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = load('random_forest_5_22.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        print("Received data:", data)
        
        time = data['Time']
        speed = data['Speed']
        
        # Create a DataFrame for the input
        input_df = pd.DataFrame({'Time': [time], 'Speed': [speed]})
        print("Input DataFrame:", input_df)

        # Predict using the loaded model
        prediction = model.predict(input_df)
        print("Prediction:", prediction)
        
        # Send back the result as JSON
        return jsonify(prediction=prediction[0])
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
