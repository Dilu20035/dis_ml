import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def load_data(train_file, test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Replace disease names with numerical labels
    disease_map = {
        'Fungal infection': 0,
        'Allergy': 1,
        # Add more disease labels here...
        'Impetigo': 40
    }
    df_train.replace({'prognosis': disease_map}, inplace=True)
    df_test.replace({'prognosis': disease_map}, inplace=True)
    
    return df_train, df_test

def train_models(df_train, symptoms):
    X_train = df_train[symptoms]
    y_train = df_train["prognosis"]

    clf_decision_tree = DecisionTreeClassifier().fit(X_train, y_train)
    clf_random_forest = RandomForestClassifier().fit(X_train, y_train)
    clf_naive_bayes = GaussianNB().fit(X_train, y_train)
    
    return clf_decision_tree, clf_random_forest, clf_naive_bayes

def predict_disease(model, symptoms, symptom_inputs):
    input_array = [0] * len(symptoms)
    for symptom in symptom_inputs:
        if symptom in symptoms:
            input_array[symptoms.index(symptom)] = 1
    prediction = model.predict([input_array])[0]
    return prediction

def main():
    st.title("Disease Predictor using Machine Learning")
    st.sidebar.title("Enter Symptoms")

    df_train, df_test = load_data("Disease-prediction-using-Machine-Learning-master/Training.csv", "Disease-prediction-using-Machine-Learning-master/Testing.csv")
    
    symptoms = [
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
        # Add more symptoms here...
        'red_sore_around_nose', 'yellow_crust_ooze'
    ]

    clf_decision_tree, clf_random_forest, clf_naive_bayes = train_models(df_train, symptoms)

    symptom_inputs = []
    for i in range(5):
        symptom = st.sidebar.selectbox(f"Symptom {i+1}", options=symptoms)
        symptom_inputs.append(symptom)

    if st.sidebar.button("Predict - Decision Tree"):
        prediction_dt = predict_disease(clf_decision_tree, symptoms, symptom_inputs)
        st.write(f"Predicted Disease (Decision Tree): {prediction_dt}")

    if st.sidebar.button("Predict - Random Forest"):
        prediction_rf = predict_disease(clf_random_forest, symptoms, symptom_inputs)
        st.write(f"Predicted Disease (Random Forest): {prediction_rf}")

    if st.sidebar.button("Predict - Naive Bayes"):
        prediction_nb = predict_disease(clf_naive_bayes, symptoms, symptom_inputs)
        st.write(f"Predicted Disease (Naive Bayes): {prediction_nb}")

if __name__ == "__main__":
    main()
