import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def load_data(train_file, test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Replace disease names with numerical labels
    # Disease map dictionary
    disease_map = {
        'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40
    }
    df_train.replace({'prognosis': disease_map}, inplace=True)
    df_test.replace({'prognosis': disease_map}, inplace=True)
    
    return df_train, df_test

def train_models(df_train, symptoms):
    X_train = df_train[symptoms]
    y_train = df_train["prognosis"]

    # Train classifiers
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

    df_train, df_test = load_data("Training.csv", "Testing.csv")
    
    symptoms = [
        'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze'
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

    # Display the predicted disease names
    diseases = {
        # Define disease names according to the mapping
    }
    if prediction_dt in diseases:
        st.write(f"Predicted Disease Name (Decision Tree): {diseases[prediction_dt]}")
    if prediction_rf in diseases:
        st.write(f"Predicted Disease Name (Random Forest): {diseases[prediction_rf]}")
    if prediction_nb in diseases:
        st.write(f"Predicted Disease Name (Naive Bayes): {diseases[prediction_nb]}")

if __name__ == "__main__":
    main()
