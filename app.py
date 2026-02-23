import streamlit as st
import pandas as pd
import pickle

# ===============================
# LOAD MODEL + ENCODER
# ===============================
model = pickle.load(open("best_rf.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))

st.title("❤️ Heart Disease Prediction App")

st.write(f"✅ Feature Count: {model.n_features_in_}")

# ===============================
# USER INPUTS
# ===============================
age = st.number_input("Age",20,100,45)
sex = st.selectbox("Sex",["Male","Female"])
cp = st.selectbox("Chest Pain Type",["typical angina","atypical angina","non-anginal","asymptomatic"])
trestbps = st.number_input("Resting BP",80,200,120)
chol = st.number_input("Cholesterol",100,600,200)
fbs = st.selectbox("Fasting Blood Sugar >120",[True,False])
restecg = st.selectbox("Rest ECG",["normal","st-t abnormality","lv hypertrophy"])
thalch = st.number_input("Max Heart Rate",60,220,150)
exang = st.selectbox("Exercise Angina",[True,False])
oldpeak = st.number_input("Oldpeak",0.0,6.0,1.0)
slope = st.selectbox("Slope",["upsloping","flat","downsloping"])
thal = st.selectbox("Thal",["normal","fixed defect","reversable defect"])

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_dict = {
    "age":[age],
    "sex":[sex],
    "cp":[cp],
    "trestbps":[trestbps],
    "chol":[chol],
    "fbs":[fbs],
    "restecg":[restecg],
    "thalch":[thalch],
    "exang":[exang],
    "oldpeak":[oldpeak],
    "slope":[slope],
    "thal":[thal]
}

input_df = pd.DataFrame(input_dict)

st.subheader("🧾 Input Preview")
st.write(input_df)

# ===============================
# PREPROCESSING (MATCH TRAINING)
# ===============================
input_df["sex"] = input_df["sex"].map({"Male":1,"Female":0})

cat_cols = ['cp','restecg','slope','thal']

encoded = encoder.transform(input_df[cat_cols].values)

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out()
)

num_df = input_df.drop(columns=cat_cols).reset_index(drop=True)

final_input = pd.concat([num_df, encoded_df], axis=1)

final_input = final_input.reindex(columns=model.feature_names_in_, fill_value=0)

# ===============================
# SIMPLE STREAMLIT RISK GAUGE
# ===============================
def show_gauge(prob):
    st.write("### ❤️ Risk Gauge")
    st.progress(float(prob))
    st.write(f"Risk Score: {prob*100:.2f}%")

# ===============================
# PREDICTION + SEVERITY TIERS
# ===============================
if st.button("Predict Risk"):

    prob = model.predict_proba(final_input)[0][1]

    st.subheader("📊 Prediction Result")

    if prob < 0.30:
        st.success("🟢 LOW RISK")
        st.info("Minimal cardiac risk detected.")

    elif prob < 0.60:
        st.warning("🟡 MODERATE RISK")
        st.write("Lifestyle monitoring recommended.")

    elif prob < 0.80:
        st.warning("🟠 HIGH RISK")
        st.write("Clinical evaluation advised.")

    else:
        st.error("🔴 CRITICAL RISK")
        st.write("Immediate medical consultation recommended.")

    show_gauge(prob)