from flask import Flask, render_template, request, session
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load 
from flask_sqlalchemy import SQLAlchemy
import secrets
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from datetime import datetime, timedelta

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'  # Replace with MySQL URL for MySQL
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
app.secret_key = secrets.token_hex(16)  
admin = Admin(app, name='Database Admin', template_mode='bootstrap3')

# class Prediction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)  # Auto-incrementing ID
#     hours_studied = db.Column(db.Float, nullable=False)
#     hours_attendance = db.Column(db.Float, nullable=False)
#     previous_score = db.Column(db.Float, nullable=False)
#     tutoring_sessions = db.Column(db.Float, nullable=False)
#     predicted_score = db.Column(db.Float, nullable=False)
#     timestamp = datetime.now()

# # Create the database tables (run this once)
# with app.app_context():
#     db.create_all()

# admin.add_view(ModelView(Prediction, db.session))
model = load_model('./neuralnetwork')  # neural network
scaler = load('./scaler.pkl')

@app.route("/")
def index():
    predictions = session.get('predictions', [])  # Retrieve predictions from the session
    return render_template("index.html", predictions=predictions)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input
        input1 = float(request.form.get("input1"))
        input2 = float(request.form.get("input2"))
        input3 = float(request.form.get("input3"))
        input4 = float(request.form.get("input4"))
        input_data = np.array([[input1, input2, input3, input4]])
        
        # Preprocess and predict
        preprocessed_data = scaler.transform(input_data)
        prediction = model.predict(preprocessed_data)
        predicted_class = round(prediction[0][0])
        
        # Save to session
        if 'predictions' not in session:
            session['predictions'] = []
        session['predictions'].append({
            'hours_studied': input1,
            'attendance': input2,
            'prev_score': input3,
            'tutoring_sessions': input4,
            'predicted_scores' : predicted_class
        })
        session.modified = True
        # Storing to sql database
        # new_prediction = Prediction(
        #     hours_studied=input1,
        #     hours_attendance=input2,
        #     previous_score=input3,
        #     tutoring_sessions=input4,
        #     predicted_score=predicted_class
        # )
        # db.session.add(new_prediction)
        # db.session.commit()
        
        # cutoff_date = datetime.now() - timedelta(days=30)
        # Prediction.query.filter(Prediction.timestamp < cutoff_date).delete()
        # db.session.commit()

        return render_template(
            "results.html", 
            prediction=predicted_class, 
            past_predictions=session['predictions']
        )




if __name__ == "__main__":
    app.run(debug=True)