
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import urllib.parse

app = Flask(__name__)

# Load the data
heart_data = pd.read_csv('heart.csv')
label_encoders = {}
for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    label_encoders[column] = LabelEncoder()
    heart_data[column] = label_encoders[column].fit_transform(heart_data[column])

# Split data into features and target
X = heart_data.drop(columns='HeartDisease', axis=1)
Y = heart_data['HeartDisease']

# Create a RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=1,
                                       random_state=42)

# Train the model
rf_classifier.fit(X, Y)

# Save the trained model
joblib.dump(rf_classifier, 'random_forest_model.pkl')


def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def calculate_lifestyle_risk(lifestyle_data):
    risk_score = 0

    # Lifestyle factors
    if lifestyle_data['physical_activity'] == 'rarely':
        risk_score += 2
    elif lifestyle_data['physical_activity'] == '1-2':
        risk_score += 1

    if lifestyle_data['fish_oil'] == 'rarely':
        risk_score += 1

    if lifestyle_data['vegetables_fruits'] == 'rarely':
        risk_score += 1

    if lifestyle_data['potassium'] == 'no':
        risk_score += 1

    if lifestyle_data['unsalted_nuts'] == 'no':
        risk_score += 1

    if lifestyle_data['plant_sterols'] == 'no':
        risk_score += 1

    if lifestyle_data['alcohol_consumption_freq'] in ['3-5', 'daily']:
        risk_score += 1

    if lifestyle_data['smoking_frequency'] in ['6-10', '11-20', 'more_than_20']:
        risk_score += 2

    if lifestyle_data['vitamin_E_supplements'] == 'no':
        risk_score += 1

    if lifestyle_data['sodium_intake'] == 'high':
        risk_score += 1

    if lifestyle_data['trans_fatty_acids'] in ['3-5', 'daily']:
        risk_score += 1

    if lifestyle_data['unfiltered_boiled_coffee'] == 'yes':
        risk_score += 1

    if lifestyle_data['existing_medical_conditions'] == 'yes':
        risk_score += 2

    if lifestyle_data['family_history'] == 'yes':
        risk_score += 2

    # Calculate BMI
    bmi = calculate_bmi(lifestyle_data['height_cm'], lifestyle_data['weight_kg'])
    if bmi >= 25:  # BMI threshold for overweight (adjust as needed)
        risk_score += 1
    return risk_score


def interpret_risk(risk_score):
    if risk_score >= 10:  # Adjust threshold as needed
        return "High risk of heart disease"
    elif risk_score >= 5:
        return "Moderate risk of heart disease"
    else:
        return "Great job, you have a low risk of heart failure!"


def get_medical_tips(age, sex, trestbps, chol, fbs, thalach, oldpeak):
    tips = []

    # Personalized tips based on user input
    if age > 60:
        tips.append("Consider regular check-ups due to increased risk at older ages.")
    if sex == 'male':
        if trestbps > 120:
            tips.append("Keep an eye on your blood pressure, it's higher than normal.")
    else:
        if trestbps > 110:
            tips.append("Monitor your blood pressure, it's higher than normal.")
    if chol > 200:
        tips.append("High cholesterol levels can increase the risk of heart disease. Consider dietary changes.")
    if fbs > 120:
        tips.append("High fasting blood sugar levels may indicate diabetes. Consult a healthcare provider.")
    if thalach < 100:
        tips.append("Your maximum heart rate is lower than average. Consider increasing physical activity.")
    if oldpeak > 2:
        tips.append("ST depression induced by exercise over 2 indicates potential risk. Consult a physician.")

    # Additional personalized tips for all parameters
    if age < 30:
        tips.append("Maintain a healthy lifestyle to prevent future heart problems.")
    if sex == 'female':
        tips.append("Women should be particularly cautious about heart health as symptoms may differ from men.")
    if trestbps < 90:
        tips.append("Your blood pressure is lower than average. Monitor for any signs of hypotension.")
    if chol < 150:
        tips.append("Low cholesterol levels may also pose health risks. Consult a healthcare professional.")
    if fbs < 70:
        tips.append("Low fasting blood sugar levels may indicate hypoglycemia. Monitor your blood sugar regularly.")
    if thalach > 180:
        tips.append("Your maximum heart rate is higher than average. Regular exercise is still important.")
    if 0.5 <= oldpeak <= 1:
        tips.append("ST depression between 0.5 and 1 may indicate a moderate risk. Keep monitoring your heart health.")

    return tips



def get_lifestyle_tips(lifestyle_data):
    tips = []
    if lifestyle_data['physical_activity'] == 'rarely':
        tips.append("Engage in regular physical activity to improve heart health.")
    if lifestyle_data['fish_oil'] == 'rarely':
        tips.append("Incorporate foods rich in EHA and DHA, such as fish or flaxseeds and walnuts (for vegetarians), into your diet for heart benefits.")
    if lifestyle_data['vegetables_fruits'] == 'rarely':
        tips.append("Include more vegetables and fruits, especially berries, in your diet for heart-healthy nutrients.")
    if lifestyle_data['potassium'] == 'no':
        tips.append("Consume foods rich in potassium, such as bananas, sweet potatoes, and spinach, to support heart health.")
    if lifestyle_data['unsalted_nuts'] == 'no':
        tips.append("Incorporate unsalted nuts and wholegrain cereals into your diet for heart-healthy fats and fiber.")
    if lifestyle_data['plant_sterols'] == 'no':
        tips.append("Include foods containing plant sterols/stanols, such as margarine fortified with sterols or plant-based milk alternatives, in your diet to help lower cholesterol.")
    if lifestyle_data['alcohol_consumption_freq'] in ['3-5', 'daily']:
        tips.append("Limit alcohol consumption to improve heart health and reduce risk.")
    if lifestyle_data['smoking_frequency'] in ['6-10', '11-20', 'more_than_20']:
        tips.append("Quit smoking or reduce cigarette consumption to lower heart disease risk.")
    if lifestyle_data['vitamin_E_supplements'] == 'no':
        tips.append("Consider taking vitamin E supplements or other dietary supplements for heart health support.")
    if lifestyle_data['sodium_intake'] == 'high':
        tips.append("Reduce sodium intake to lower blood pressure and improve heart health.")
    if lifestyle_data['trans_fatty_acids'] in ['3-5', 'daily']:
        tips.append("Limit consumption of foods high in trans fatty acids, such as fried foods, commercially baked goods, and hydrogenated vegetable oils, to reduce heart disease risk.")
    if lifestyle_data['unfiltered_boiled_coffee'] == 'yes':
        tips.append("Avoid consuming unfiltered boiled coffee, which may increase cholesterol levels.")
    if lifestyle_data['existing_medical_conditions'] == 'yes':
        tips.append("Manage existing medical conditions such as diabetes, hypertension, or hypercholesterolemia to reduce heart disease risk.")
    if lifestyle_data['family_history'] == 'yes':
        tips.append("Be aware of family history of cardiovascular diseases, particularly heart disease, and take preventive measures.")

    return tips



@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prediction_type = request.form.get('prediction_type')

        # Redirect to the appropriate prediction page based on the selected type
        if prediction_type == 'medical':
            return redirect(url_for('medical_predict'))
        elif prediction_type == 'lifestyle':
            return redirect(url_for('lifestyle_predict'))
        else:
            return "Invalid prediction type."



@app.route('/medical_predict', methods=['GET', 'POST'])
def medical_predict():
    if request.method == 'POST':
        # Extract form data from the result page
        age = int(request.form['Age'])
        sex = int(request.form['Sex'])  # Convert sex to int
        cp = int(request.form['ChestPainType'])  # Convert Chest Pain Type to int
        trestbps = int(request.form['RestingBP'])  # Resting Blood Pressure
        chol = int(request.form['Cholesterol'])  # Serum Cholesterol
        fbs = int(request.form['FastingBS'])  # Fasting Blood Sugar
        restecg = int(request.form['RestingECG'])  # Resting Electrocardiographic Results
        thalach = int(request.form['MaxHR'])  # Maximum Heart Rate Achieved
        exang = int(request.form['ExerciseAngina'])  # Exercise Induced Angina
        oldpeak = float(request.form['Oldpeak'])  # ST Depression Induced by Exercise
        slope = int(request.form['ST_Slope'])  # Slope of the Peak Exercise ST Segment

        # Make prediction
        prediction = rf_classifier.predict(np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                                      exang, oldpeak, slope]]))

        # Get medical tips
        tips = get_medical_tips(age, sex, trestbps, chol, fbs, thalach, oldpeak)
        patient_info = f"Age: {age}\nSex: {'Female' if sex == 0 else 'Male'}\nChest Pain Type: {cp}\nResting Blood Pressure: {trestbps}\n" \
                       f"Serum Cholesterol: {chol}\nFasting Blood Sugar: {fbs}\nResting Electrocardiographic " \
                       f"Results: {restecg}\nMaximum Heart Rate Achieved: {thalach}\nExercise Induced Angina: {exang}\n" \
                       f"ST Depression Induced by Exercise: {oldpeak}\nSlope of the Peak Exercise ST Segment: {slope}"

        # If heart failure, prompt user to send report to doctor
        if prediction[0] == 1:
            return render_template('result.html', prediction="Prone to heart disease", tips=tips,
                                   email=request.args.get('email'), doctor_email=request.args.get('doctor_email'),
                                   patient_info=patient_info)
        else:
            return render_template('result.html', prediction="Not prone to heart disease", tips=tips)

    return render_template('medical_predict.html')


@app.route('/lifestyle_predict', methods=['GET', 'POST'])
def lifestyle_predict():
    if request.method == 'POST':
        lifestyle_data = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'height_cm': float(request.form['height_cm']),
            'weight_kg': float(request.form['weight_kg']),
            'physical_activity': request.form['physical_activity'],
            'fish_oil': request.form['fish_oil'],
            'vegetables_fruits': request.form['vegetables_fruits'],
            'potassium': request.form['potassium'],
            'unsalted_nuts': request.form['unsalted_nuts'],
            'plant_sterols': request.form['plant_sterols'],
            'alcohol_consumption_freq': request.form['alcohol_consumption_freq'],
            'smoking_frequency': request.form['smoking_frequency'],
            'vitamin_E_supplements': request.form['vitamin_E_supplements'],
            'sodium_intake': request.form['sodium_intake'],
            'trans_fatty_acids': request.form['trans_fatty_acids'],
            'unfiltered_boiled_coffee': request.form['unfiltered_boiled_coffee'],
            'existing_medical_conditions': request.form['existing_medical_conditions'],
            'family_history': request.form['family_history']
        }

        risk_score = calculate_lifestyle_risk(lifestyle_data)
        result = interpret_risk(risk_score)


        # Get lifestyle tips
        tips = get_lifestyle_tips(lifestyle_data)

        return render_template('lifestyle_result.html', result=result, tips=tips)
    return render_template('lifestyle_predict.html')


@app.route('/send_report', methods=['POST'])
def send_report():
    if request.method == 'POST':
        doctor_email = request.form.get('doctor_email')
        patient_info = request.form.get('patient_info')  # Retrieve patient information

        # Decode the patient information
        email_body = patient_info

        # Encode the email body for inclusion in the URL
        encoded_email_body = urllib.parse.quote(email_body)

        # Construct the URL for Gmail compose page with prefilled subject and body
        compose_url = f'https://mail.google.com/mail/u/0/?view=cm&fs=1&to={doctor_email}&su=Potential%20Heart%20Failure%20Alert&body={encoded_email_body}'

        # Redirect to the compose page
        return redirect(compose_url)

if __name__ == '__main__':
    app.run(debug=True)