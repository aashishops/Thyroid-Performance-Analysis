import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


#modelpath=r'E:\College\2nd year\Winter project\Thyroid-Performance-Analysis\SAVED_MODEL'
modelpath=r'C:\Users\dwija\OneDrive\Thyroid-Performance-Analysis\SAVED_MODEL'
with open(modelpath + r'\models.pkl', 'rb') as file:
    models = pkl.load(file)
#annpath=r'E:\College\2nd year\Winter project\Thyroid-Performance-Analysis\SAVED_MODEL\ann_model.h5'
annpath=r'C:\Users\dwija\OneDrive\Thyroid-Performance-Analysis\SAVED_MODEL\ANN_model.h5'
ann_model = load_model(annpath)
#lstmpath=r'E:\College\2nd year\Winter project\Thyroid-Performance-Analysis\SAVED_MODEL\lstm_model.h5'
lstmpath=r'C:\Users\dwija\OneDrive\Thyroid-Performance-Analysis\SAVED_MODEL\LSTM_model.h5'
lstm_model = load_model(lstmpath)


def map_to_thyroid_condition(prediction):
    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Hypothyroidism"
    elif prediction == 2:
        return "Hyperthyroidism"
    else:
        return "Invalid Prediction"
def main():
    st.title("Thyroid Prediction App")

    
    st.header("Enter Patient Information:")

  
    age = st.number_input("Age", value=34.0)
    on_thyroxine = st.selectbox("On Thyroxine", ['No', 'Yes'], index=0)
    query_on_thyroxine = st.selectbox("Query on Thyroxine", ['No', 'Yes'], index=0)
    on_antithyroid_medication = st.selectbox("On Antithyroid Medication", ['No', 'Yes'], index=0)
    sick = st.selectbox("Sick", ['No', 'Yes'], index=0)
    pregnant = st.selectbox("Pregnant", ['No', 'Yes'], index=0)
    thyroid_surgery = st.selectbox("Thyroid Surgery", ['No', 'Yes'], index=0)
    I131_treatment = st.selectbox("I131 Treatment", ['No', 'Yes'], index=0)
    query_hypothyroid = st.selectbox("Query Hypothyroid", ['No', 'Yes'], index=0)
    query_hyperthyroid = st.selectbox("Query Hyperthyroid", ['No', 'Yes'], index=0)
    lithium = st.selectbox("Lithium", ['No', 'Yes'], index=0)
    goitre = st.selectbox("Goitre", ['No', 'Yes'], index=0)
    tumor = st.selectbox("Tumor", ['No', 'Yes'], index=0)
    hypopituitary = st.selectbox("Hypopituitary", ['No', 'Yes'], index=0)
    psych = st.selectbox("Psych", ['No', 'Yes'], index=0)
    TSH_measured = st.selectbox("TSH Measured", ['No', 'Yes'], index=1)
    TSH = st.number_input("TSH", value=6.2)
    T3_measured = st.selectbox("T3 Measured", ['No', 'Yes'], index=0)
    T3 = st.number_input("T3", value=2.052089)
    TT4_measured = st.selectbox("TT4 Measured", ['No', 'Yes'], index=1)
    TT4 = st.number_input("TT4", value=116.0)
    T4U_measured = st.selectbox("T4U Measured", ['No', 'Yes'], index=1)
    T4U = st.number_input("T4U", value=1.13)
    FTI_measured = st.selectbox("FTI Measured", ['No', 'Yes'], index=1)
    FTI = st.number_input("FTI", value=103.0)


    true_value = st.selectbox("Select true value", ['Negative', 'Hypothyroidism', 'Hyperthyroidism'])

    if st.button("Predict"):
        user_input = np.array([age, 1 if on_thyroxine == 'Yes' else 0, 1 if query_on_thyroxine == 'Yes' else 0,
                            1 if on_antithyroid_medication == 'Yes' else 0, 1 if sick == 'Yes' else 0,
                            1 if pregnant == 'Yes' else 0, 1 if thyroid_surgery == 'Yes' else 0,
                            1 if I131_treatment == 'Yes' else 0, 1 if query_hypothyroid == 'Yes' else 0,
                            1 if query_hyperthyroid == 'Yes' else 0, 1 if lithium == 'Yes' else 0,
                            1 if goitre == 'Yes' else 0, 1 if tumor == 'Yes' else 0, 1 if hypopituitary == 'Yes' else 0,
                            1 if psych == 'Yes' else 0, 1 if TSH_measured == 'Yes' else 0, TSH,
                            1 if T3_measured == 'Yes' else 0, T3, 1 if TT4_measured == 'Yes' else 0, TT4,
                            1 if T4U_measured == 'Yes' else 0, T4U, 1 if FTI_measured == 'Yes' else 0, FTI])

    
        ann_prediction = int(np.round(ann_model.predict(user_input.reshape(1, -1))[0][0]))

        
        user_input_lstm = user_input.reshape((1, 1, user_input.shape[0]))
        lstm_prediction = int(np.round(lstm_model.predict(user_input_lstm)[0][0]))


        other_model_predictions = {}
        for model_name, model in models.items():
            prediction = int(np.round(model.predict(user_input.reshape(1, -1))[0]))
            other_model_predictions[model_name] = prediction
        def map_to_thyroid_condition(prediction):
            if prediction == 0:
                return "Negative"
            elif prediction == 1:
                return "Hypothyroidism"
            elif prediction == 2:
                return "Hyperthyroidism"
            else:
                return "Invalid Prediction"

    
        ann_condition = map_to_thyroid_condition(ann_prediction)
        if ann_condition == true_value:
            st.success(f"ANN Prediction: {ann_condition}")
        else:
            st.error(f"ANN Prediction: {ann_condition}",)

        lstm_condition = map_to_thyroid_condition(lstm_prediction)
        if lstm_condition == true_value:
            st.success(f"LSTM Prediction: {lstm_condition}")
        else:
            st.error(f"LSTM Prediction: {lstm_condition}")

        
        for model_name, prediction in other_model_predictions.items():
            condition = map_to_thyroid_condition(prediction)
            if condition == true_value  :
                st.success(f"{model_name} Prediction: {condition}")
            else:
                st.error(f"{model_name} Prediction: {condition}")


if __name__ == "__main__":
    main()
