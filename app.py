from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Loading the saved model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up 2 levels
model_path = os.path.join(BASE_DIR, "model", "final_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


# ------------------------
# Routes
# ------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Extracting form data
        gender = 1 if request.form['Gender'] == 'Male' else 0
        married = 1 if request.form['Married'] == 'Yes' else 0
        dependents = request.form['Dependents'].strip()
        dependents = 3 if dependents == '3+' else int(dependents)
        education = 1 if request.form['Education'] == 'Graduate' else 0
        self_employed = 1 if request.form['Self_Employed'] == 'Yes' else 0
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])

        prop_area = request.form['Property_Area']
        # One-hot encoding for Property_Area
        prop_urban = 1 if prop_area == 'Urban' else 0
        prop_semiurban = 1 if prop_area == 'Semiurban' else 0
        # If neither, Rural = [0, 0]

        # Final feature vector (must match training order!)
        features = [[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, credit_history,
            prop_semiurban, prop_urban
        ]]

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        print("Model Prediction:", prediction)
        print("Prediction Probabilities:", proba)

        # Choose response content
        if prediction == 1:
            return render_template('submit.html', status="Approved")
        else:
            return render_template('submit.html', status="Rejected")

    except Exception as e:
        print("Error during prediction:", e)
        return "Something went wrong during prediction. Please try again."



# Run the app (for local dev only)
if __name__ == '__main__':
    app.run(debug=True)