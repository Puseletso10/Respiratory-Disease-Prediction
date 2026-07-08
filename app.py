import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

model_p = joblib.load("models/best_model.pkl")
shap_explainer = joblib.load("models/shap_explainer.pkl")

st.set_page_config(
    page_title="Respiratory Disease Predictor",
    layout="wide"
)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 250px !important;
}

section[data-testid="stSidebar"] > div {
    width: 250px !important;
}
div[role="radiogroup"] > label {
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


#-----
def patient_page():
    st.title("Respiratory Disease Risk Prediction")
    st.caption("Enter patient information below to assess the risk of respiratory disease." )
    
    left, right = st.columns([1, 1])

    with left:
        
        
        with st.container(border=True):
            st.subheader("👤 Patient Information")
            col1, col2 = st.columns(2)

            with col1:
                country = st.selectbox("Country",["Botswana","Eswatini","Lesotho","Mozambique","Zimbabwe"])

                age = st.number_input("Age",18,100,45)
                
                No_Positive_TB_Diagnosis_in_Family = st.number_input("No_Positive_TB_Diagnosis_in_Family",0,10,0)

                aqi = st.slider("Air Quality Index",0,500,220)

                pollution_exposure = st.selectbox(
                    "Exposure to Pollution",
                    ["Low","Medium","High"]
                )

                history = st.selectbox(
                    "History of Respiratory Disease",
                    ["No","Yes"]
                )
                
                hiv_status = st.selectbox(
                    "HIV Diagnosis",
                    ["No","Yes"]
                )

            with col2:
                gender = st.selectbox("Gender",["Male","Female"])

                smoking = st.selectbox(
                    "Smoking Status",
                    ["No","Yes"]
                )
            
                cough = st.selectbox(
                    "Cough",
                    ["No","Yes"]
                )
            
                wheezing = st.selectbox(
                    "Wheezing",
                    ["No","Yes"]
                )
                shortness = st.selectbox(
                    "Shortness of Breath",
                    ["No","Yes"]
                )
                chest = st.selectbox(
                    "Chest Pain",
                    ["No","Yes"]
                )

        predict = st.button(
        "Predict Risk",
        use_container_width=True
        )



    country= {
        "Botswana":0,
        "Eswatini":1,
        "Lesotho":2,
        "South Africs":3,
        "Mozambique":4,
        "Zimbabwe":5
        
    }[country]

    gender = 1 if gender=="Male" else 0

    history = 1 if history=="Yes" else 0

    hiv_status = 1 if hiv_status=="Yes" else 0

    cough = 1 if cough=="Yes" else 0

    wheezing = 1 if wheezing=="Yes" else 0

    shortness = 1 if shortness=="Yes" else 0

    chest = 1 if chest=="Yes" else 0

    smoking = 1 if smoking=="Yes" else 0

    pollution_exposure = {
        "Low":0,
        "Medium":1,
        "High":2
    }[pollution_exposure]


    features = pd.DataFrame([{
        "country": country,
        "cough": cough,
        "wheezing": wheezing,
        "shortness": shortness,
        "chest": chest,
        "smoking": smoking,
        "age": age,
        "gender": gender,
        "pollution_exposure": pollution_exposure,
        "No_Positive_TB_Diagnosis_in_Family": No_Positive_TB_Diagnosis_in_Family,
        "history": history,
        "hiv_status": hiv_status,
        "aqi": aqi
    }])
    prediction = model_p.predict(features)

    probability = model_p.predict_proba(features)[0][1]

    if predict:
        prediction = model_p.predict(features)
        probability = model_p.predict_proba(features)[0][1]
        shap_values = shap_explainer(features)

        
        with right:
            with st.container(border=True):
                st.markdown("###### 📈 Risk Prediction Results")
                
                fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability*100,
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{"color": "rgba(0,0,0,0)"},
                            'steps':[
                                {'range':[0,40],'color':'blue'},
                                {'range':[40,70],'color':'yellow'},
                                {'range':[70,100],'color':'red'}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 6},
                                "value": probability * 100
                            }
                        }
                    ))
                fig.update_layout(
                        width=250,     # Width in pixels
                        height=150,    # Height in pixels
                        margin=dict(l=10, r=10, t=10, b=10)
                        )

                st.plotly_chart(fig)

                if prediction==1:

                    st.error("High Risk")

                else:

                    st.success("Low Risk")
            with st.container(border=True):
                st.markdown("###### 📈 Risk Factors Contribution")
                            
                shap_values = shap_explainer(features)
                    
                shap_fig = plt.figure()

                shap.plots.bar(
                    shap_values[0],
                    show=False
                )

                st.pyplot(shap_fig)

                    # Get SHAP values for the first prediction
                values = shap_values.values[0]
                feature_names = shap_values.feature_names

                #Sort by absolute contribution
                top_idx = np.argsort(np.abs(values))[::-1][:5]
                    
                patient_values = features.iloc[0]

                for i in top_idx:
                    direction = "increased" if values[i] > 0 else "decreased"

                    st.write(
                        f"• **{feature_names[i]} = {patient_values[feature_names[i]]}** "
                        f"{direction} the predicted risk."
                    )
                                
def about_page():

    st.title("ℹ️ About")

    st.write(
        "This application predicts the risk of Respiratory Disease."
    )

    with st.expander("🎯 Purpose"):
        st.write(
            "To assist healthcare professionals by identifying patients "
            "at higher risk of Respiratory Disease."
        )

    with st.expander("📋 Features Used"):
        st.write("""
        - Age
        - Smoking
        - HIV Diagnosis
        - Exposure to pollution
        - Air Quality
        - Previous Respiratory Illness
        - country
        - Cough
        - Gender
        - Wheezing
        - No of Positive TB Diagnosis in Family
        - Shortness of breath
        - Chest pain
        """)

    with st.expander("⚠ Disclaimer"):
        st.warning(
            "This application is for educational and research purposes "
            "and should not replace professional medical advice."
        )
    
if "page" not in st.session_state:
    st.session_state.page = "patient"

#-------
with st.sidebar:
    
    logo_col, text_col = st.columns([1, 3])

    with logo_col:
        st.image("logo.jfif", width=70)

    with text_col:
        st.markdown("""
        <div style="line-height:1.1;">
            <span style="font-size:18px; font-weight:700;">Respiratory</span><br>
            <span style="font-size:18px; font-weight:700;">Risk Predictor</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button(" Patient Prediction", use_container_width=True):
        st.session_state.page = "patient"

    if st.button(" About", use_container_width=True):
        st.session_state.page = "about"
        
# Display the selected page
if st.session_state.page == "patient":
    patient_page()

elif st.session_state.page == "about":
    about_page()

