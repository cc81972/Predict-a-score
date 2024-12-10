from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load  # Replace with your machine learning model import

app = Flask(__name__)

# Load your pre-trained machine learning model here
model = load_model('../neuralnetwork')  # Replace with your model loading logic
scaler = load('../scaler.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input1 = float(request.form.get("input1"))
        input2 = float(request.form.get("input2"))
        input3 = float(request.form.get("input3"))
        input4 = float(request.form.get("input4"))
        input_data = np.array([[input1, input2, input3, input4]])
        # Preprocess the input data for your model (if needed)
        # ... your data preprocessing code here ...
        preprocessed_data = scaler.transform(input_data)
        # Make prediction using your model
        prediction = model.predict(preprocessed_data)  # Assuming a list input

        # Format the prediction for display
        predicted_class = round(prediction[0][0])  # Assuming single class output

        return render_template("results.html", prediction=predicted_class)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)