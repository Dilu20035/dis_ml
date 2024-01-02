import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# Load data
df = pd.read_csv("Disease-prediction-using-Machine-Learning-master/Training.csv")
test_df = pd.read_csv("Disease-prediction-using-Machine-Learning-master/Testing.csv")

# Replace disease names with numerical labels
disease_map = {
    'Fungal infection': 0,
    'Allergy': 1,
    # Add more disease labels here...
    'Impetigo': 40
}
df.replace({'prognosis': disease_map}, inplace=True)
test_df.replace({'prognosis': disease_map}, inplace=True)

# Define symptoms list
symptoms = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
    # Add more symptoms here...
    'red_sore_around_nose', 'yellow_crust_ooze'
]

# Separate features and target
X = df[symptoms]
y = df["prognosis"]

X_test = test_df[symptoms]
y_test = test_df["prognosis"]

# Train models
clf_decision_tree = tree.DecisionTreeClassifier().fit(X, y)
clf_random_forest = RandomForestClassifier().fit(X, y)
clf_naive_bayes = GaussianNB().fit(X, y)

# Streamlit app
st.title("Disease Predictor using Machine Learning")

st.sidebar.title("Enter Symptoms")

# Sidebar inputs for symptoms
symptom_inputs = []
for i in range(5):
    symptom = st.sidebar.selectbox(f"Symptom {i+1}", options=symptoms)
    symptom_inputs.append(symptom)

# Function to predict disease
def predict_disease(model, symptoms_input):
    input_array = [0] * len(symptoms)
    for symptom in symptoms_input:
        if symptom in symptoms:
            input_array[symptoms.index(symptom)] = 1
    prediction = model.predict([input_array])[0]
    return prediction

# Prediction using Decision Tree
if st.sidebar.button("Predict - Decision Tree"):
    prediction_dt = predict_disease(clf_decision_tree, symptom_inputs)
    st.write(f"Predicted Disease (Decision Tree): {prediction_dt}")

# Prediction using Random Forest
if st.sidebar.button("Predict - Random Forest"):
    prediction_rf = predict_disease(clf_random_forest, symptom_inputs)
    st.write(f"Predicted Disease (Random Forest): {prediction_rf}")

# Prediction using Naive Bayes
if st.sidebar.button("Predict - Naive Bayes"):
    prediction_nb = predict_disease(clf_naive_bayes, symptom_inputs)
    st.write(f"Predicted Disease (Naive Bayes): {prediction_nb}")
