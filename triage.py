import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.markdown("""
    <style>
    /* Dark background with soft gradient */
    .stApp {
        background: linear-gradient(to bottom right, #1e1e2f, #2c2c3e);
        font-family: 'Segoe UI', 'Inter', sans-serif;
        color: #f0f0f5;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #2b2b3c;
        color: #f0f0f5;
        border-right: 1px solid #444;
    }

    /* Inputs & selectboxes */
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div {
        background-color: #3a3a4f;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 8px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #2563eb;
    }

    /* DataFrame/Table background */
    .stDataFrame, .stTable {
        background-color: #2e2e3e;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        padding: 1rem;
        color: #f0f0f5;
    }

    /* Fix table font */
    .stDataFrame table {
        font-size: 0.95rem;
        color: #f0f0f5;
    }

    /* Misc elements */
    .stMarkdown, .stText {
        color: #f0f0f5;
    }
    </style>
""", unsafe_allow_html=True)



# Initialize session state
if "lang" not in st.session_state:
    st.session_state["lang"] = "English"
if "export_format" not in st.session_state:
    st.session_state["export_format"] = "CSV"
if "input_data" not in st.session_state:
    st.session_state["input_data"] = None
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "explanation" not in st.session_state:
    st.session_state["explanation"] = []

# Model selection
model_files = {
    "Extra Trees": "extra_trees.pkl",
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Linear SVM": "linear_svm.pkl",
    "RBF SVM": "rbf_svm.pkl",
    "AdaBoost": "adaboost.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "HistGradientBoosting": "hist_gradient_boosting.pkl"
}

selected_model_name = st.sidebar.selectbox("Choose Model", list(model_files.keys()))
model = joblib.load(model_files[selected_model_name])


# Button and dropdown hover/animation styles
st.markdown("""
    <style>
    .stButton>button {
        transition: all 0.3s ease-in-out;
        border: 2px solid transparent;
        border-radius: 8px;
    }
    .stButton>button:hover {
        border: 2px solid #2c6bed;
        background-color: #eef3ff;
        color: #2c6bed;
    }
    .stSelectbox>div>div {
        transition: all 0.3s ease-in-out;
        border: 2px solid transparent;
        border-radius: 5px;
    }
    .stSelectbox>div>div:hover {
        border: 2px solid #2c6bed;
        box-shadow: 0 0 6px rgba(44, 107, 237, 0.4);
    }
    label[data-baseweb="form-control"]:before {
        content: attr(data-icon) " ";
        font-size: 1rem;
        margin-right: 0.5rem;
        color: #2c6bed;
    }
    </style>
""", unsafe_allow_html=True)

# Language selector
st.session_state["lang"] = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"], index=["English", "Arabic"].index(st.session_state["lang"]))
lang = st.session_state["lang"]
is_ar = lang == "Arabic"

# RTL layout for Arabic
if is_ar:
    st.markdown("""
        <style>
        .stApp {
            direction: rtl;
            text-align: right;
            font-family: 'Amiri', serif;
            font-size: 1.05rem;
        }
        .css-1kyxreq, .css-1v0mbdj, .css-1q8dd3e, .css-1rs6os {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Translations
T = {
    "title": {"English": "Patient Triage Prediction", "Arabic": "نظام تصنيف حالة المريض"},
    "gender": {"English": "👨‍⚕️ Gender", "Arabic": "👨‍⚕️ النوع"},
    "chest pain type": {"English": "❤️ Chest Pain Type", "Arabic": "❤️ نوع ألم الصدر"},
    "exercise angina": {"English": "🏃‍♂️ Exercise Angina", "Arabic": "🏃‍♂️ الذبحة الصدرية من المجهود"},
    "hypertension": {"English": "🔳 Hypertension", "Arabic": "🔳 ارتفاع ضغط الدم"},
    "heart_disease": {"English": "❤️ Heart Disease", "Arabic": "❤️ أمراض القلب"},
    "smoking_status": {"English": "🚬 Smoking Status", "Arabic": "🚬 حالة التدخين"},
    "blood pressure": {"English": "⏰ Blood Pressure", "Arabic": "⏰ ضغط الدم"},
    "cholesterol": {"English": "🧃 Cholesterol", "Arabic": "🧃 الكوليسترول"},
    "max heart rate": {"English": "❤️ Max Heart Rate", "Arabic": "❤️ أقصى معدل لضربات القلب"},
    "plasma glucose": {"English": "☕ Plasma Glucose", "Arabic": "☕ جلوكوز البلازما"},
    "skin_thickness": {"English": "🦼 Skin Thickness", "Arabic": "🦼 سمك الجلد"},
    "insulin": {"English": "📊 Insulin", "Arabic": "📊 الأنسولين"},
    "bmi": {"English": "🏋️ BMI", "Arabic": "🏋️ مؤشر كتلة الجسم"},
    "diabetes_pedigree": {"English": "👥 Diabetes Pedigree", "Arabic": "👥 تاريخ العائلة مع السكري"},
    "age": {"English": "⏳ Age", "Arabic": "⏳ العمر"},
    "predict": {"English": "🔮 Predict Triage Level", "Arabic": "🔮 تنبؤ بمستوى الطوارئ"},
    "confidence": {"English": "🔢 Confidence Levels", "Arabic": "🔢 مستويات الثقة"},
    "explanation": {"English": "📄 Explanation Report", "Arabic": "📄 تقرير التفسير"},
    "patient_data": {"English": "📈 Patient Input Data", "Arabic": "📈 بيانات المريض"},
}

def _(key):
    return T[key]["Arabic"] if is_ar else T[key]["English"]

# Value translations for user inputs
VALUE_TRANSLATIONS = {
    "Male": "ذكر", "Female": "أنثى",
    "typical": "نموذجي", "atypical": "غير نمطي", "non-anginal": "غير صدري", "asymptomatic": "بدون أعراض",
    "Yes": "نعم", "No": "لا",
    "never smoked": "لم يدخن أبدًا", "formerly smoked": "كان يدخن سابقًا", "smokes": "يدخن", "Unknown": "غير معروف"
}

def translate_value(value):
    if isinstance(value, str) and is_ar:
        return VALUE_TRANSLATIONS.get(value, value)
    return value

# Translate input options to Arabic if needed
REVERSE_TRANSLATIONS = {v: k for k, v in VALUE_TRANSLATIONS.items()}

def get_input_options(options):
    return [VALUE_TRANSLATIONS[o] if is_ar else o for o in options]

def reverse_translate(value):
    return REVERSE_TRANSLATIONS.get(value, value)

# Adjust sidebar style for RTL
if is_ar:
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Patient Input Form ---
def user_input_features():
    gender = st.sidebar.selectbox(_("gender"), get_input_options(["Male", "Female"]))
    chest_pain_type = st.sidebar.selectbox(_("chest pain type"), get_input_options(["typical", "atypical", "non-anginal", "asymptomatic"]))
    exercise_angina = st.sidebar.selectbox(_("exercise angina"), get_input_options(["Yes", "No"]))
    hypertension = st.sidebar.selectbox(_("hypertension"), get_input_options(["Yes", "No"]))
    heart_disease = st.sidebar.selectbox(_("heart_disease"), get_input_options(["Yes", "No"]))
    smoking_status = st.sidebar.selectbox(_("smoking_status"), get_input_options(["never smoked", "formerly smoked", "smokes", "Unknown"]))

    blood_pressure = st.sidebar.number_input(_("blood pressure"), 80, 200, 120)
    cholesterol = st.sidebar.number_input(_("cholesterol"), 100, 400, 180)
    max_heart_rate = st.sidebar.number_input(_("max heart rate"), 60, 220, 160)
    plasma_glucose = st.sidebar.number_input(_("plasma glucose"), 50, 300, 90)
    skin_thickness = st.sidebar.number_input(_("skin_thickness"), 7, 100, 25)
    insulin = st.sidebar.number_input(_("insulin"), 0, 900, 80)
    bmi = st.sidebar.number_input(_("bmi"), 10.0, 60.0, 24.5)
    diabetes_pedigree = st.sidebar.number_input(_("diabetes_pedigree"), 0.0, 2.5, 0.5)
    age = st.sidebar.number_input(_("age"), 1, 120, 50)

    data = {
        "gender": reverse_translate(gender),
        "chest pain type": reverse_translate(chest_pain_type),
        "blood pressure": blood_pressure,
        "cholesterol": cholesterol,
        "max heart rate": max_heart_rate,
        "exercise angina": reverse_translate(exercise_angina),
        "plasma glucose": plasma_glucose,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "hypertension": reverse_translate(hypertension),
        "heart_disease": reverse_translate(heart_disease),
        "smoking_status": reverse_translate(smoking_status),
        "age": age
    }

    return pd.DataFrame([data])

# Translate prediction result
PREDICTION_TRANSLATIONS = {
    "Low": "منخفض",
    "Medium": "متوسط",
    "High": "مرتفع",
    "Critical": "حرج"
}

def translated_prediction(pred):
    return PREDICTION_TRANSLATIONS.get(pred, pred) if is_ar else pred

# Translate column values
original_input_df = st.session_state["input_data"] if st.session_state["input_data"] is not None else pd.DataFrame()
translated_df = original_input_df.applymap(translate_value)
translated_export_df = translated_df.copy()
translated_export_df["Triage Prediction"] = translated_prediction(st.session_state["prediction"]) if st.session_state["prediction"] else ""


st.title(_( "title"))
st.sidebar.header(_( "patient_data"))

def user_input_features():
    gender = st.sidebar.selectbox(_( "gender"), ["Male", "Female"])
    chest_pain_type = st.sidebar.selectbox(_( "chest pain type"), ["typical", "atypical", "non-anginal", "asymptomatic"])
    exercise_angina = st.sidebar.selectbox(_( "exercise angina"), ["Yes", "No"])
    hypertension = st.sidebar.selectbox(_( "hypertension"), ["Yes", "No"])
    heart_disease = st.sidebar.selectbox(_( "heart_disease"), ["Yes", "No"])
    smoking_status = st.sidebar.selectbox(_( "smoking_status"), ["never smoked", "formerly smoked", "smokes", "Unknown"])

    blood_pressure = st.sidebar.number_input(_( "blood pressure"), 80, 200, 120)
    cholesterol = st.sidebar.number_input(_( "cholesterol"), 100, 400, 180)
    max_heart_rate = st.sidebar.number_input(_( "max heart rate"), 60, 220, 160)
    plasma_glucose = st.sidebar.number_input(_( "plasma glucose"), 50, 300, 90)
    skin_thickness = st.sidebar.number_input(_( "skin_thickness"), 7, 100, 25)
    insulin = st.sidebar.number_input(_( "insulin"), 0, 900, 80)
    bmi = st.sidebar.number_input(_( "bmi"), 10.0, 60.0, 24.5)
    diabetes_pedigree = st.sidebar.number_input(_( "diabetes_pedigree"), 0.0, 2.5, 0.5)
    age = st.sidebar.number_input(_( "age"), 1, 120, 50)

    data = {
        "gender": gender,
        "chest pain type": chest_pain_type,
        "blood pressure": blood_pressure,
        "cholesterol": cholesterol,
        "max heart rate": max_heart_rate,
        "exercise angina": exercise_angina,
        "plasma glucose": plasma_glucose,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_status": smoking_status,
        "age": age
    }

    return pd.DataFrame([data])

# Collect input
original_input_df = user_input_features()
input_df = original_input_df.copy()

st.subheader(_( "patient_data"))
st.write(original_input_df)

# Label encoders
label_encoders = {
    "gender": LabelEncoder(),
    "chest pain type": LabelEncoder(),
    "exercise angina": LabelEncoder(),
    "hypertension": LabelEncoder(),
    "heart_disease": LabelEncoder(),
    "smoking_status": LabelEncoder()
}

# Export format
export_format = st.selectbox("Download format", ["CSV", "PDF"])

# Predict button
if st.button(_( "predict")):
    try:
        with st.spinner("🔄 ..."):
            for col, le in label_encoders.items():
                input_df[col] = le.fit_transform(input_df[col])
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

        st.success(f"{_('predict')}: **{prediction}**")



    
        if prediction.lower() == "yellow":
          prediction = "Medium"
        elif prediction.lower() == "red":
          prediction = "Critical"
        elif prediction.lower() == "green":
          prediction = "Low"
        elif prediction.lower() == "orange":
          prediction = "High"


        

        # ---- TRIAGE ANIMATED BAR ----
        triage_colors = {
            "Critical": "#FF4B4B",
            "High": "#FFA500",
            "Medium": "#FFD700",
            "Low": "#4CAF50"
        }
        triage_order = ["Critical", "High", "Medium", "Low"]

        st.markdown("### 🖪 Triage Level Indicator")
        st.markdown(f"""
        <style>
        .triage-container {{
            width: 100%;
            background-color: #eee;
            border-radius: 20px;
            padding: 3px;
            box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
        }}

        .triage-fill {{
            height: 30px;
            width: 0%;
            background-color: {triage_colors.get(prediction, "#ccc")};
            border-radius: 15px;
            animation: fillBar 2s forwards;
        }}

        @keyframes fillBar {{
            from {{ width: 0%; }}
            to {{ width: 100%; }}
        }}
        </style>

        <div class='triage-container'>
          <div class='triage-fill'></div>
        </div>
        """, unsafe_allow_html=True)

        # ---- END TRIAGE BAR ----

    


        st.markdown(f"### 🔍 {_('confidence')}")
        prob_df = pd.DataFrame({
            "Triage Level": model.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        # Color map for triage levels
        triage_color_map = {
        "Critical": "#FF4B4B",  # red
        "High": "#FFA500",      # orange
    "Medium": "#FFD700",    # yellow
    "Low": "#4CAF50",       # green
    "red": "#FF4B4B",
    "orange": "#FFA500",
    "yellow": "#FFD700",
    "green": "#4CAF50"
}
        fig = px.bar(
            prob_df,
            x="Triage Level",
            y="Probability",
            text_auto='.2f',
            title=_("confidence"),
            color="Triage Level",
            color_discrete_map=triage_color_map
        )

        fig.update_layout(
            plot_bgcolor="#2c2c3e",
            paper_bgcolor="#2c2c3e",
            font_color="#f0f0f5",
            title_font_color="#ffffff",
            xaxis=dict(
                title_font_color="#f0f0f5",
                tickfont_color="#f0f0f5"
            ),
            yaxis=dict(
                title_font_color="#f0f0f5",
                tickfont_color="#f0f0f5"
            ),
    showlegend=False
        )
        # Style box using Streamlit markdown and target the iframe or div
        st.markdown("""
    <style>
    .plotly-container iframe, .plotly-container div {
        background-color: #2e2e3e;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        padding: 1rem;
        color: #f0f0f5;
    }
    </style>
""", unsafe_allow_html=True)

         # Display with container class
        st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if max(probabilities) < 0.6:
            st.warning("⚠️ " + ("النموذج غير واثق بدرجة كافية. يُرجى استشارة الطبيب." if is_ar else "Model not confident. Clinical review advised."))

        st.markdown(f"### 📝 {_('explanation')}")
        raw = original_input_df.iloc[0]
        explanation = []

        thresholds = {
            "blood pressure": 140,
            "cholesterol": 240,
            "plasma glucose": 150,
            "bmi": 30,
            "max heart rate": 100
        }

        bp = raw["blood pressure"]
        if bp > thresholds["blood pressure"]:
            explanation.append(f"🫀 {_( 'blood pressure')} = {bp} (> {thresholds['blood pressure']}) — " +
                               ("ضغط الدم مرتفع." if is_ar else "High blood pressure."))
        elif bp < 90:
            explanation.append(f"🫀 {_( 'blood pressure')} = {bp} (< 90) — " +
                               ("ضغط الدم منخفض." if is_ar else "Low blood pressure."))

        chol = raw["cholesterol"]
        if chol > thresholds["cholesterol"]:
            explanation.append(f"🫀 {_( 'cholesterol')} = {chol} (> {thresholds['cholesterol']}) — " +
                               ("كوليسترول مرتفع." if is_ar else "High cholesterol."))

        hr = raw["max heart rate"]
        if hr < thresholds["max heart rate"]:
            explanation.append(f"🫀 {_( 'max heart rate')} = {hr} (< {thresholds['max heart rate']}) — " +
                               ("معدل ضربات القلب منخفض." if is_ar else "Low heart rate."))

        glucose = raw["plasma glucose"]
        if glucose > thresholds["plasma glucose"]:
            explanation.append(f"🍩 {_( 'plasma glucose')} = {glucose} (> {thresholds['plasma glucose']}) — " +
                               ("جلوكوز مرتفع." if is_ar else "High glucose."))

        bmi = raw["bmi"]
        if bmi > thresholds["bmi"]:
            explanation.append(f"💓 {_( 'bmi')} = {bmi} (> {thresholds['bmi']}) — " +
                               ("مؤشر كتلة الجسم مرتفع." if is_ar else "High BMI."))
        elif bmi < 18.5:
            explanation.append(f"💓 {_( 'bmi')} = {bmi} (< 18.5) — " +
                               ("نقص في الوزن." if is_ar else "Underweight."))

        if raw["age"] > 60:
            explanation.append(f"📅 {_( 'age')} = {raw['age']} — " +
                               ("العمر المتقدم يزيد من الخطورة." if is_ar else "Older age increases risk."))
        if raw["heart_disease"] == "Yes":
            explanation.append("❤️ " + ("تاريخ مرضي بالقلب." if is_ar else "Heart disease history."))
        if raw["hypertension"] == "Yes":
            explanation.append("💢 " + ("ارتفاع ضغط الدم مزمن." if is_ar else "Chronic hypertension."))
        if raw["smoking_status"] in ["smokes", "formerly smoked"]:
            explanation.append("🚬 " + ("تاريخ مع التدخين." if is_ar else "Smoking history."))
        if raw["chest pain type"] in ["atypical", "asymptomatic"]:
            explanation.append("⚠️ " + ("ألم صدر غير نمطي." if is_ar else "Atypical chest pain."))

        if explanation:
            for line in explanation:
                st.markdown(f"- {line}")
        else:
            st.markdown("- لا توجد عوامل خطورة." if is_ar else "- No major risk factors detected.")

        # Export
        if export_format == "CSV":
            csv_data = original_input_df.copy()
            csv_data["Triage Prediction"] = prediction
            st.download_button("⬇️ Download CSV", csv_data.to_csv(index=False), file_name="triage_result.csv")
        else:
            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
            pdf.setFont("Helvetica", 12)
            y = 750
            pdf.drawString(30, y, "Patient Triage Prediction Report")
            y -= 30
            for key, val in raw.items():
                pdf.drawString(30, y, f"{key}: {val}")
                y -= 20
            y -= 10
            pdf.drawString(30, y, f"Prediction: {prediction}")
            y -= 30
            for line in explanation:
                if y < 100:
                    pdf.showPage()
                    y = 750
                pdf.drawString(30, y, "- " + line)
                y -= 20
            pdf.save()
            st.download_button("⬇️ Download PDF", data=pdf_buffer.getvalue(), file_name="triage_report.pdf")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<div style='position: fixed; bottom: 10px; right: 10px; font-size: 10px; color: gray;'>FOR ACADEMIC PURPOSES ONLY</div>", unsafe_allow_html=True)
