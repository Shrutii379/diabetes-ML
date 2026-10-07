import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GlucoSense · Diabetes Risk", page_icon="🩺", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
:root{--bg:#0D0F2B;--panel:#171A45;--line:#2B2F6B;--violet:#7C5CFF;--teal:#19E3C1;--pink:#FF4D8D;--amber:#FFB84D;--text:#EEF0FF;--muted:#9EA3D6;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;color:var(--text);}
.stApp{background:radial-gradient(900px 500px at 8% -5%,#2a1f7a55,transparent),radial-gradient(800px 500px at 100% 10%,#0fb9a533,transparent),var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2.2rem;max-width:1150px;}
h1,h2,h3{font-family:'Sora',sans-serif!important;}
.hero h1{font-size:2.6rem;font-weight:800;margin:0;background:linear-gradient(90deg,#fff 0%,var(--teal) 55%,var(--violet) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{color:var(--muted);margin:.4rem 0 1.6rem;font-size:1.02rem;max-width:620px;}
.panel{background:linear-gradient(160deg,#1b1f55,var(--panel));border:1px solid var(--line);border-radius:20px;padding:1.4rem 1.5rem;}
.panel h3{font-size:1.05rem;margin:0 0 .6rem;color:#fff;}
label,.stSlider label,.stSelectbox label,.stNumberInput label{color:var(--muted)!important;font-weight:500!important;}
div[data-baseweb="select"]>div,.stNumberInput input{background:#10133a!important;border:1px solid var(--line)!important;border-radius:12px!important;color:var(--text)!important;}
.stSlider [data-baseweb="slider"]>div>div{background:linear-gradient(90deg,var(--violet),var(--teal))!important;}
.stSlider [role="slider"]{background:var(--teal)!important;border:3px solid #fff!important;}
.stButton>button{width:100%;border:0;border-radius:14px;padding:.85rem 1rem;font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;color:#0D0F2B;background:linear-gradient(90deg,var(--teal),#7cf0ff);box-shadow:0 8px 28px #19e3c133;transition:transform .15s,box-shadow .15s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 34px #19e3c155;color:#0D0F2B;}
.stButton>button:focus-visible{outline:3px solid var(--violet);}
.gauge-wrap{text-align:center;padding:.5rem 0 0;}
.gauge-wrap svg{width:100%;max-width:360px;}
.verdict{font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700;margin:.2rem 0;}
.sub{color:var(--muted);font-size:.95rem;max-width:420px;margin:0 auto;}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.1rem;justify-content:center;}
.chip{padding:.35rem .8rem;border-radius:999px;font-size:.85rem;font-weight:500;border:1px solid var(--line);background:#10133a;}
.chip.warn{border-color:var(--pink);color:#ffc2d9;}
.chip.ok{border-color:var(--teal);color:#b5fbee;}
.note{color:var(--muted);font-size:.8rem;margin-top:1.4rem;text-align:center;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
FEATURES = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

# LabelEncoder sorts alphabetically (uppercase first). Change here if your encoding differs.
GENDER = {"Female": 0, "Male": 1, "Other": 2}
SMOKING = {"No Info": 0, "Current": 1, "Ever": 2, "Former": 3, "Never": 4, "Not current": 5}


@st.cache_resource
def load_model():
    model = joblib.load("finalModel.pkl")
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, scaler


def gauge(p):
    color = "#19E3C1" if p < 0.3 else "#FFB84D" if p < 0.6 else "#FF4D8D"
    arc = 251.3
    return f"""
    <div class="gauge-wrap"><svg viewBox="0 0 200 120">
      <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#7C5CFF"/><stop offset="1" stop-color="{color}"/></linearGradient></defs>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#252a63" stroke-width="16" stroke-linecap="round"/>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#g)" stroke-width="16" stroke-linecap="round"
            stroke-dasharray="{arc}" stroke-dashoffset="{arc*(1-p):.1f}"/>
      <text x="100" y="92" text-anchor="middle" font-family="Sora" font-weight="800" font-size="34" fill="#fff">{p*100:.0f}%</text>
      <text x="100" y="112" text-anchor="middle" font-family="Inter" font-size="9" fill="#9EA3D6">estimated probability</text>
    </svg></div>"""


# ---------- Header ----------
st.markdown("""<div class="hero"><h1>GlucoSense</h1>
<p>Enter a few health details and the logistic regression model estimates the chance of diabetes.</p></div>""",
            unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="panel"><h3>Patient details</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", list(GENDER))
    smoking = c2.selectbox("Smoking history", list(SMOKING), index=5)
    age = st.slider("Age", 1, 100, 45)
    bmi = st.slider("BMI", 10.0, 60.0, 27.0, 0.1)
    c3, c4 = st.columns(2)
    hba1c = c3.number_input("HbA1c level (%)", 3.5, 9.0, 5.8, 0.1)
    glucose = c4.number_input("Blood glucose (mg/dL)", 70, 300, 140, 5)
    c5, c6 = st.columns(2)
    hyper = c5.toggle("Hypertension")
    heart = c6.toggle("Heart disease")
    run = st.button("Check risk")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><h3>Result</h3>', unsafe_allow_html=True)
    if not run:
        st.markdown(gauge(0.0) + '<p class="sub" style="text-align:center">Fill in the details and press <b>Check risk</b>.</p>',
                    unsafe_allow_html=True)
    else:
        try:
            model, scaler = load_model()
            row = pd.DataFrame([{
                "gender": GENDER[gender], "age": age, "hypertension": int(hyper),
                "heart_disease": int(heart), "smoking_history": SMOKING[smoking],
                "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose}])[FEATURES]
            X = scaler.transform(row) if scaler else row
            p = float(model.predict_proba(X)[0][1])

            if p < 0.3:
                title, text = "Low risk", "The inputs look healthy. Keep up regular check-ups."
            elif p < 0.6:
                title, text = "Moderate risk", "Some values are elevated. A routine blood test is a good idea."
            else:
                title, text = "High risk", "Several values point to diabetes. Please consult a doctor."

            chips = []
            if hba1c >= 6.5: chips.append(("warn", f"HbA1c {hba1c}% is in the diabetic range"))
            elif hba1c >= 5.7: chips.append(("warn", f"HbA1c {hba1c}% is in the pre-diabetic range"))
            else: chips.append(("ok", f"HbA1c {hba1c}% is normal"))
            if glucose >= 200: chips.append(("warn", f"Glucose {glucose} mg/dL is high"))
            elif glucose >= 140: chips.append(("warn", f"Glucose {glucose} mg/dL is raised"))
            else: chips.append(("ok", f"Glucose {glucose} mg/dL is fine"))
            if bmi >= 30: chips.append(("warn", f"BMI {bmi:.1f} is in the obese range"))
            elif bmi >= 25: chips.append(("warn", f"BMI {bmi:.1f} is overweight"))
            else: chips.append(("ok", f"BMI {bmi:.1f} is healthy"))
            if hyper: chips.append(("warn", "Hypertension"))
            if heart: chips.append(("warn", "Heart disease"))
            chip_html = "".join(f'<span class="chip {k}">{t}</span>' for k, t in chips)

            st.markdown(gauge(p) + f'<div style="text-align:center"><div class="verdict">{title}</div>'
                        f'<p class="sub">{text}</p></div><div class="chips">{chip_html}</div>',
                        unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("finalModel.pkl not found. Put it in the same folder as app.py and run again.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.markdown('<p class="note">Educational project, not a medical diagnosis.</p></div>', unsafe_allow_html=True)
