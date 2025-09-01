from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import os
import json
from pathlib import Path
import io

# --------------------------------------------------
# Guard: If user runs `python app.py` instead of `streamlit run ...`,
# suppress the flood of ScriptRunContext warnings and print clear instructions.
# --------------------------------------------------
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    if get_script_run_ctx() is None and __name__ == "__main__":
        print("\n[!] This is a Streamlit application.\n"
              "Run it with:\n  streamlit run \"{}\"\n".format(__file__))
        # Early graceful exit to avoid warning spam when executed directly.
        raise SystemExit(0)
except Exception:
    # If the internal API changes or import fails we just continue; Streamlit will handle context.
    pass

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go  # lightweight core
# Heavy / optional libs will be imported lazily to speed first paint
px = None
alt = None
sns = None
plt = None

# --------------------------------------------------
# Page Configuration & Global Paths
# --------------------------------------------------
st.set_page_config(
    page_title='Heart Disease Risk Prediction',
    page_icon='❤️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --------------------------------------------------
# Session State Initialization (Landing page flag & last prediction store)
# --------------------------------------------------
if 'show_landing' not in st.session_state:
    st.session_state['show_landing'] = True  # Landing shows first time
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'nav_memory' not in st.session_state:
    st.session_state['nav_memory'] = 'Home'
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'model_report' not in st.session_state:
    st.session_state['model_report'] = {}
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = pd.DataFrame()

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model.pkl'
REPORT_PATH = PROJECT_ROOT / 'results' / 'best_model_report.json'
DATA_PATH = PROJECT_ROOT / 'data' / 'selected_features.csv'

# --------------------------------------------------
# Compatibility wrappers (minimal) for unpickling saved threshold model
# --------------------------------------------------


class ThresholdSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, **svc_params):
        from sklearn.svm import SVC
        self.threshold = threshold
        self.svc_params = svc_params
        self.model_ = SVC(**svc_params)

    def fit(self, X, y):
        self.model_.fit(X, y)
        return self

    def _raw_proba(self, X):
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)[:, 1]
        from scipy.special import expit
        scores = self.model_.decision_function(X)
        # Min-max to [0,1] (fallback)
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - mn) / (mx - mn)

    def predict(self, X):
        return (self._raw_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        raw = self._raw_proba(X)
        return np.vstack([1 - raw, raw]).T

    def decision_function(self, X):
        if hasattr(self.model_, 'decision_function'):
            return self.model_.decision_function(X)
        return self.predict_proba(X)[:, 1]

    def get_params(self, deep=True):
        return {'threshold': self.threshold, **self.svc_params}

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')
        self.svc_params.update(params)
        self.model_.set_params(**self.svc_params)
        return self


class ThresholdedEstimator(BaseEstimator, ClassifierMixin):
    """Fallback wrapper name used in earlier notebook variants."""

    def __init__(self, pipeline=None, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        # Assume already fitted
        return self

    def predict_proba(self, X):
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        if hasattr(self.pipeline, 'decision_function'):
            from scipy.special import expit
            scores = self.pipeline.decision_function(X)
            probs = expit(scores)
            return np.vstack([1 - probs, probs]).T
        raise AttributeError(
            'Underlying estimator lacks probability interface')

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def decision_function(self, X):
        if hasattr(self.pipeline, 'decision_function'):
            return self.pipeline.decision_function(X)
        return self.predict_proba(X)[:, 1]


@st.cache_resource(show_spinner=True)
def load_pipeline():
    """Load the serialized best model and its report from disk."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}. Ensure training/export was completed.")
        st.stop()
    model_obj = joblib.load(MODEL_PATH)
    report = {}
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception:
            pass
    return model_obj, report


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load engineered feature dataset for visualization (if available)."""
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception as e:
            st.warning(f"Failed to load dataset for visualizations: {e}")
    return pd.DataFrame()


def ensure_model_loaded():
    """Lazy-load model into session_state if absent."""
    if st.session_state.get('model') is None:
        with st.spinner('Loading model...'):
            m, r = load_pipeline()
            st.session_state['model'] = m
            st.session_state['model_report'] = r
    return st.session_state.get('model') is not None


def ensure_data_loaded():
    """Lazy-load dataset into session_state if absent."""
    if st.session_state.get('raw_df') is None or st.session_state.get('raw_df').empty:
        with st.spinner('Loading dataset...'):
            st.session_state['raw_df'] = load_dataset()
    return not st.session_state.get('raw_df', pd.DataFrame()).empty


def _lazy_import_plotting(kind: str = 'px'):
    """Import heavier visualization libs only when actually needed.
    kind can be 'px','alt','sns','plt','all'. Returns requested module(s).
    """
    global px, alt, sns, plt
    if kind in ('px', 'all') and px is None:
        import plotly.express as _px  # type: ignore
        px = _px
    if kind in ('alt', 'all') and alt is None:
        import altair as _alt  # type: ignore
        alt = _alt
    if kind in ('sns', 'all') and sns is None:
        import seaborn as _sns  # type: ignore
        sns = _sns
    if kind in ('plt', 'all') and plt is None:
        import matplotlib.pyplot as _plt  # type: ignore
        plt = _plt
    if kind == 'all':
        return px, alt, sns, plt
    return {'px': px, 'alt': alt, 'sns': sns, 'plt': plt}[kind]


def get_current_threshold():
    return (st.session_state.get('model_report') or {}).get('best_threshold', 0.5)


def get_best_params():
    return (st.session_state.get('model_report') or {}).get('best_params', {})


model = st.session_state.get('model')
model_report = st.session_state.get('model_report', {})
raw_df = st.session_state.get('raw_df', pd.DataFrame())
NAV_PAGES = ["Home", "Prediction", "Models & Results", "About"]
default_page = 'Home' if st.session_state.get(
    'show_landing', True) else 'Prediction'
nav_page = st.session_state.get('nav_memory', default_page)
st.sidebar.markdown(
    "<h3 style='margin-top:0;margin-bottom:0.6rem;color:#fff;font-weight:700;'>Navigation</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='nav-btn-group'>", unsafe_allow_html=True)
active_index = 0
for idx, p in enumerate(NAV_PAGES, start=1):
    if st.sidebar.button(p, key=f"nav_{p}"):
        nav_page = p
        st.session_state['nav_memory'] = p
    if nav_page == p:
        active_index = idx
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='margin-top:1.5rem;font-size:0.65rem;font-style:italic;color:#DDDDDD;line-height:1.2;'>For educational purposes only — not a medical device.</div>", unsafe_allow_html=True)

# nav_choice removed; direct nav_page usage

# --------------------------------------------------
# Landing Page (Home) - shown only once unless user re-selects Home
# --------------------------------------------------
ACCENT_COLORS = {
    'Home': '#D62828',
    'Prediction': '#D62828',
    'Models & Results': '#6A4C93',
    'About': '#AAAAAA'
}

# Dynamic CSS (updated gradient + active nav highlight with white default buttons)
gradient_bg = 'linear-gradient(135deg, #f76ca2, #fbc7a4)'
active_accent = ACCENT_COLORS.get(nav_page, '#D62828')
# Active button now keeps white background with accent border & dark text
active_css = f".nav-btn-group .stButton:nth-of-type({active_index}) button {{ background:#ffffff; color:{active_accent}; border:2px solid {active_accent}; box-shadow:0 0 0 3px rgba(0,0,0,0.05); font-weight:700; }}" if active_index else ""
BUTTON_TEXT_COLOR = "#FFFFFF"
CUSTOM_CSS = r"""
<style>
body, .stApp { background:GRADIENT_BG; background-attachment:fixed; color:#F0F0F0; }
.block-container { padding-top:1rem; }
section[data-testid='stSidebar'] { background:GRADIENT_BG !important; border-right:1px solid rgba(255,255,255,0.25); }
.nav-btn-group .stButton button { width:100%; text-align:left; margin-bottom:0.45rem; border:2px solid #eeeeee; font-weight:600; border-radius:18px; padding:0.65rem 0.95rem; font-size:0.95rem; color:#333333; transition:all .18s; box-shadow:0 2px 4px rgba(0,0,0,0.08); background:#ffffff; }
.nav-btn-group .stButton button { width:100%; text-align:left; margin-bottom:0.45rem; border:2px solid #D62828; font-weight:600; border-radius:18px; padding:0.65rem 0.95rem; font-size:0.95rem; color:#FFFFFF !important; transition:all .18s; box-shadow:0 2px 6px rgba(0,0,0,0.25); background:#D62828; }
.nav-btn-group .stButton button:hover { transform:translateY(-2px); box-shadow:0 4px 16px rgba(0,0,0,0.30); filter:brightness(1.05); }
ACTIVE_BUTTON_CSS /* active button outline */
h1,h2,h3,h4,h5,p,span,label { color:#F0F0F0; }
.card { background:#ffffff; color:#222; border-radius:12px; padding:1.15rem 1.25rem 1.25rem; box-shadow:0 6px 22px -8px rgba(0,0,0,0.35); margin-bottom:1.1rem; border:1px solid rgba(255,255,255,0.35); }
.card h3, .card h4 { color:#222; margin-top:0; font-weight:700; }
.accent-bar { height:6px; border-radius:4px; margin:-1.15rem -1.25rem 1rem -1.25rem; }
.accent-red { background:#D62828; }
.accent-blue { background:#1D70A2; }
.accent-purple { background:#6A4C93; }
.accent-gray { background:#AAAAAA; }
div.stButton > button[kind='primary'] { background:#D62828; color:BTN_COLOR; border-radius:30px; font-weight:600; letter-spacing:.5px; box-shadow:0 4px 14px -4px rgba(0,0,0,0.45); }
div.stButton > button[kind='primary'] { background:#D62828 !important; color:BTN_COLOR !important; border-radius:30px; font-weight:700; letter-spacing:.5px; box-shadow:0 4px 14px -4px rgba(0,0,0,0.45); border:2px solid #D62828; }
div.stButton > button[kind='primary']:hover { filter:brightness(1.1); }
div.stButton > button { border-radius:30px; color:BTN_COLOR; }
div.stButton > button { border-radius:30px; color:BTN_COLOR !important; background:#D62828; border:2px solid #D62828; }
.stButton > button { color:BTN_COLOR !important; }
.nav-btn-group .stButton button { color:BTN_COLOR !important; }
div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] p, label, .stSlider label { color:BTN_COLOR !important; font-weight:600; }
input, select, textarea { border-radius:8px !important; }
.result-highlight { font-size:1.25rem; font-weight:700; padding:0.9rem 1rem; border-radius:12px; text-align:center; margin-top:0.6rem; }
.risk-low { background:#e9f9ee; color:#1b5e20; border:2px solid #2e7d32; }
.risk-high { background:#ffe9e9; color:#b71c1c; border:2px solid #D62828; }
.subtle-label { font-size:0.7rem; color:#444; text-transform:uppercase; letter-spacing:1px; font-weight:600; }
.home-title { font-size:3.2rem; font-weight:800; margin:0 0 .75rem; background:linear-gradient(90deg,#D62828,#6A4C93); -webkit-background-clip:text; color:transparent; text-align:center; letter-spacing:0.5px; }
.home-sub { font-size:1.20rem; color:#333; margin-bottom:1.15rem; line-height:1.5; text-align:center; font-weight:600; }
.home-team { font-weight:700; margin-top:0.2rem; color:#D62828; text-align:center; font-size:0.95rem; letter-spacing:0.5px; }
.home-desc { font-size:0.95rem; color:#444; margin-top:0.9rem; text-align:center; line-height:1.55; }
.home-wrapper { display:flex; justify-content:center; align-items:center; gap:2.5rem; flex-wrap:wrap; padding-top:1.2rem; }
.home-card-equal { width:100%; display:flex; flex-direction:column; justify-content:center; padding:0 0 1rem 0; }
.home-hero { display:flex; gap:2rem; align-items:stretch; flex-wrap:wrap; min-height:72vh; }
.home-card-equal img { object-fit:cover; width:100%; height:100%; }
.predict-btn-big button { width:100%; background:#D62828 !important; padding:1rem 0 !important; font-size:1.05rem; color:BTN_COLOR !important; }
.predict-btn-big button { width:100%; background:#D62828 !important; padding:1rem 0 !important; font-size:1.05rem; color:BTN_COLOR !important; border:2px solid #D62828; }
.footer-note { font-size:0.65rem; color:#222; text-align:center; margin-top:3rem; opacity:.75; }
.small-note { font-size:0.65rem; color:#666; }
</style>
"""
CUSTOM_CSS = CUSTOM_CSS.replace('GRADIENT_BG', gradient_bg).replace(
    'ACTIVE_BUTTON_CSS', active_css).replace('BTN_COLOR', BUTTON_TEXT_COLOR)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Home Page --------------------
if nav_page == "Home":
    left_spacer, main_col, right_spacer = st.columns([1, 2, 1], gap="large")
    with main_col:
        home_html = """
        <div class='card home-card-equal' 
             style='min-height:420px;display:flex;flex-direction:column;
                    justify-content:center;padding:25px;
                    background:linear-gradient(135deg,#ffffff,#fdfdfd);
                    border-radius:15px;box-shadow:0 6px 14px rgba(0,0,0,0.15);'>

          <h2 style='text-align:center; color:#c62828; font-size:32px; font-weight:800; margin-bottom:15px;'>
            ❤️ Heart Disease Risk Prediction App
          </h2>
          
          <p style='font-size:17px; text-align:center; line-height:1.8; color:#2c3e50;'>
            <span style='font-weight:600; color:#d84315;'>⚡ An AI-powered application</span> 
            designed to assist in <b style="color:#1976d2;">predicting the risk of heart disease</b> 
            based on patient health indicators.
          </p>
          
          <p style='font-size:17px; text-align:center; line-height:1.8; color:#37474f;'>
            By leveraging <span style='color:#2e7d32;font-weight:600;'>🤖 advanced machine learning techniques</span>, 
            the app provides users with <span style='color:#6a1b9a; font-weight:600;'>📊 accurate, data-driven insights</span> 
            to support awareness and early risk detection.
          </p>
          
          <p style='font-size:17px; text-align:center; line-height:1.8; color:#455a64;'>
            This tool is built for <b style='color:#00838f;'>🎓 educational and research purposes</b> — 
            helping users explore how <span style="color:#5d4037;">💡 AI can be applied in healthcare decision support</span>.
          </p>
          
          <p style='text-align:center; font-weight:bold; margin-top:25px; font-size:18px; color:#263238;'>
             Developed by: <span style='color:#c62828;'>Yousef Salah</span> & 
             <span style='color:#1565c0;'>Abdelrahman Mohsen</span>
          </p>
        </div>
        """
        st.markdown(home_html, unsafe_allow_html=True)
    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    spacer_l, btn_col, spacer_r = st.columns([2, 3, 2])
    with btn_col:
        if st.button('🚀 Start Prediction', type='primary', key='start_pred_home', use_container_width=True):
            ensure_model_loaded()
            st.session_state['show_landing'] = False
            st.session_state['nav_memory'] = 'Prediction'
            # Updated for Streamlit >=1.32: experimental_rerun deprecated
            st.rerun()
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------
# Styling / Theming
# --------------------------------------------------
# (Previous theme block replaced by dynamic gradient styling above)

# --------------------------------------------------
# Legacy radio removed per new requirements.

# Page banner removed per requirements (no nav_page heading)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
CHEST_PAIN_OPTIONS = [
    'Typical angina', 'Atypical angina', 'Non-anginal', 'Asymptomatic'
]
SLOPE_OPTIONS = ['upsloping', 'flat', 'downsloping']
THAL_OPTIONS = ['normal', 'fixed defect', 'reversable defect']
REST_ECG_OPTIONS = ['normal', 'st-t abnormality',
                    'left ventricular hypertrophy']  # Provided for completeness

DUMMY_FEATURES = [
    'cp_non-anginal', 'cp_atypical angina', 'fbs_True', 'exang_True',
    'slope_flat', 'slope_upsloping', 'sex_Male', 'thal_normal', 'thal_reversable defect'
]
NUMERIC_BASE = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
# Engineered numeric features expected: chol_per_age, heart_rate_reserve plus maybe columns seen in report

EXPECTED_NUMERIC_FINAL = [
    'chol', 'chol_per_age', 'ca', 'trestbps', 'heart_rate_reserve', 'oldpeak', 'thalch', 'age'
]

# Map form values to model frame


def build_feature_frame(form_vals: dict) -> pd.DataFrame:
    # Start with base numeric
    age = form_vals['age']
    trestbps = form_vals['trestbps']
    chol = form_vals['chol']
    thalach = form_vals['thalach']
    oldpeak = form_vals['oldpeak']
    ca = form_vals['ca']
    sex = form_vals['sex']
    fbs = form_vals['fbs']
    exang = form_vals['exang']
    slope = form_vals['slope']
    cp = form_vals['cp']
    thal = form_vals['thal']

    # Derived engineered
    chol_per_age = chol / (age + 1)
    heart_rate_reserve = thalach - trestbps

    # Initialize feature dict with numeric using names expected by pipeline
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalach,  # pipeline expects 'thalch'
        'oldpeak': oldpeak,
        'ca': ca,
        'chol_per_age': chol_per_age,
        'heart_rate_reserve': heart_rate_reserve,
    }

    # Initialize all dummy features to 0
    for col in DUMMY_FEATURES:
        data[col] = 0

    # Sex
    if sex == 'Male':
        data['sex_Male'] = 1

    # Fasting blood sugar
    if fbs == 'Yes':
        data['fbs_True'] = 1

    # Exercise induced angina
    if exang == 'Yes':
        data['exang_True'] = 1

    # Slope one-hot (only upsloping & flat present in dummy feature list)
    if slope == 'flat':
        data['slope_flat'] = 1
    elif slope == 'upsloping':
        data['slope_upsloping'] = 1

    # Chest pain types (encode atypical & non-anginal; typical/asymptomatic baseline)
    if cp.lower().startswith('atypical'):
        data['cp_atypical angina'] = 1
    elif cp.lower().startswith('non-anginal'):
        data['cp_non-anginal'] = 1

    # Thal
    if thal == 'normal':
        data['thal_normal'] = 1
    elif thal == 'reversable defect':
        data['thal_reversable defect'] = 1

    df = pd.DataFrame([data])
    model_obj = st.session_state.get('model')
    if model_obj is not None and hasattr(model_obj, 'feature_names_in_'):
        for col in model_obj.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model_obj.feature_names_in_]
    else:
        ordered = [c for c in EXPECTED_NUMERIC_FINAL if c in df.columns]
        rest = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]
    return df


def predict_risk(feat_df: pd.DataFrame):
    ensure_model_loaded()
    model_obj = st.session_state['model']
    threshold = get_current_threshold()
    try:
        proba = model_obj.predict_proba(feat_df)[:, 1]
    except Exception:
        from scipy.special import expit
        scores = model_obj.decision_function(feat_df)
        proba = expit(scores)
    prob = float(proba[0])
    return int(prob >= threshold), prob


def render_gauge(prob: float):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=prob*100,
        number={'suffix': '%', 'font': {'size': 26}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#D62828' if prob >= get_current_threshold() else '#1B5E20'},
            'steps': [
                {'range': [0, get_current_threshold()*100],
                 'color': '#e2f3f1'},
                {'range': [get_current_threshold()*100, 100],
                 'color': '#fdecec'}
            ],
            'threshold': {
                'line': {'color': '#6A4C93', 'width': 3},
                'value': get_current_threshold()*100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)


if nav_page == 'Prediction':
    model = st.session_state.get('model')
    model_report = st.session_state.get('model_report', {})
    raw_df = st.session_state.get('raw_df', pd.DataFrame())
    if model is None:
        st.info(
            'Please click "🚀 Start Prediction" from the Home page first to load the model.')
        st.stop()
    # Single wider column: inputs then (below) prediction result
    form_col = st.container()
    with form_col:
        st.markdown(f"<div class='card'><div class='accent-bar accent-red'></div><h3 style='color:#D62828;margin-top:0;'>Input Parameters</h3><p class='small-note'>Provide patient metrics then click Predict.</p>", unsafe_allow_html=True)
        example_clicked = st.button('✨ Use Example Data')
        use_example = example_clicked
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider('Age', 20, 100, 55 if use_example else 50)
            trestbps = st.number_input(
                'Resting BP (trestbps)', 80, 220, 130 if use_example else 120)
            chol = st.number_input('Cholesterol (chol)',
                                   100, 700, 250 if use_example else 240)
            oldpeak = st.number_input(
                'Oldpeak (ST depression)', 0.0, 10.0, 1.2 if use_example else 1.0, step=0.1)
        with c2:
            thalach = st.number_input(
                'Max HR (thalach)', 60, 250, 160 if use_example else 150)
            ca = st.selectbox('Major Vessels (ca)', [
                              0, 1, 2, 3], index=0 if not use_example else 1)
            sex = st.radio('Sex', ['Male', 'Female'],
                           index=0 if use_example else 0, horizontal=True)
            fbs = st.selectbox('Fasting Blood Sugar > 120 (fbs)', [
                               'No', 'Yes'], index=1 if use_example else 0)
        with c3:
            exang = st.radio('Exercise Angina (exang)', [
                             'No', 'Yes'], index=1 if use_example else 0, horizontal=True)
            slope = st.selectbox('Slope', SLOPE_OPTIONS,
                                 index=1 if use_example else 0)
            cp = st.selectbox('Chest Pain Type (cp)',
                              CHEST_PAIN_OPTIONS, index=2 if use_example else 0)
            thal = st.selectbox('Thalassemia (thal)',
                                THAL_OPTIONS, index=2 if use_example else 0)
        with st.expander('Optional: Resting ECG (not modeled directly)'):
            restecg = st.selectbox('Resting ECG (restecg)', REST_ECG_OPTIONS)
        submit = st.button('🔮 Predict', type='primary')
        if submit:
            form_dict = dict(age=age, trestbps=trestbps, chol=chol, thalach=thalach, oldpeak=oldpeak, ca=ca,
                             sex=sex, fbs=fbs, exang=exang, slope=slope, cp=cp, thal=thal, restecg=restecg)
            feat_df = build_feature_frame(form_dict)
            pred_class, prob = predict_risk(feat_df)
            st.session_state['prediction'] = {'class': int(
                pred_class), 'prob': prob, 'input': form_dict}
            st.session_state['last_prediction'] = st.session_state['prediction']
        st.markdown("</div>", unsafe_allow_html=True)  # close input card
    # Inline prediction result directly beneath form
    st.markdown("<div class='card'><div class='accent-bar accent-red'></div><h3 style='color:#D62828;margin-top:0;'>Prediction Result</h3>", unsafe_allow_html=True)
    if 'prediction' not in st.session_state:
        st.info('No prediction yet.')
    else:
        pred = st.session_state['prediction']
        risk_label = 'High Risk' if pred['class'] == 1 else 'Low Risk'
        prob_pct = pred['prob'] * 100
        css_class = 'risk-high' if pred['class'] == 1 else 'risk-low'
        st.markdown(
            f"<div class='result-highlight {css_class}'>Prediction: {risk_label}<br><span style='font-size:0.9rem;font-weight:600;color:inherit;'>Risk Score: {prob_pct:.1f}%</span></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:0.8rem;font-size:0.75rem;color:#555;'>Threshold: <b>{get_current_threshold():.3f}</b> (≥ threshold => High Risk)</div>")
        render_gauge(pred['prob'])
    st.markdown("</div>", unsafe_allow_html=True)
    if 'prediction' in st.session_state:
        st.markdown("<div class='card'><div class='accent-bar accent-red'></div><h4 style='color:#D62828;margin-top:0;'>Feature Vector Preview</h4>", unsafe_allow_html=True)
        feat_df = build_feature_frame(st.session_state['prediction']['input'])
        st.dataframe(feat_df)
        csv_buf = io.StringIO()
        out_df = pd.DataFrame([{**st.session_state['prediction']['input'], 'risk_probability': round(
            st.session_state['prediction']['prob'], 4), 'predicted_class': st.session_state['prediction']['class']}])
        out_df.to_csv(csv_buf, index=False)
        st.download_button('⬇️ Download Result (CSV)', data=csv_buf.getvalue(
        ), file_name='prediction_result.csv', mime='text/csv')
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Data Visualization Page
# --------------------------------------------------
elif nav_page == 'Data & Insights':
    ensure_data_loaded()
    raw_df = st.session_state.get('raw_df', pd.DataFrame())
    st.subheader('📊 Data & Insights')
    if raw_df.empty:
        st.warning('Dataset not available.')
    else:
        # Identify target column
        target_col = None
        for cand in ['target', 'num', 'heart_disease']:
            if cand in raw_df.columns:
                target_col = cand
                break
        if target_col is None:
            target_col = raw_df.columns[-1]
        df = raw_df.copy()

        st.markdown("<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Dataset Overview</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('Rows', len(df))
        with c2:
            st.metric('Columns', df.shape[1])
        with c3:
            miss_pct = (df.isna().sum().sum() / (df.shape[0]*df.shape[1]))*100
            st.metric('Missing %', f"{miss_pct:.1f}%")
        with c4:
            if target_col in df.columns:
                pos_rate = df[target_col].mean(
                )*100 if df[target_col].nunique() <= 5 else None
                st.metric('Positive Rate',
                          f"{pos_rate:.1f}%" if pos_rate is not None else '—')
        st.markdown('</div>', unsafe_allow_html=True)

        # Class balance
        if target_col in df.columns and df[target_col].nunique() <= 10:
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Class Balance</h4>", unsafe_allow_html=True)
            vc = df[target_col].value_counts(dropna=False)
            vc = vc.reset_index().rename(
                columns={'index': 'class', target_col: 'count'})
            vc['percent'] = (vc['count'].astype(float) /
                             vc['count'].sum() * 100).round(1)
            _lazy_import_plotting('px')
            fig_cls = px.bar(vc, x='class', y='count', text='percent', color='class',
                             title='Target Class Distribution', color_discrete_sequence=px.colors.qualitative.Set2)
            fig_cls.update_traces(texttemplate='%{text}%')
            st.plotly_chart(fig_cls, use_container_width=True)
            st.dataframe(vc)
            st.markdown('</div>', unsafe_allow_html=True)

        # Numeric summary (selected numeric columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        focus_cols = [c for c in ['age', 'trestbps', 'chol', 'thalach',
                                  'oldpeak', 'ca', 'chol_per_age', 'heart_rate_reserve'] if c in num_cols]
        if focus_cols:
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Key Numeric Summary</h4>", unsafe_allow_html=True)
            _lazy_import_plotting('px')
            st.dataframe(df[focus_cols].describe().T.round(2))
            if target_col in df.columns:
                grp_stats = df.groupby(target_col)[
                    focus_cols].mean().T.round(2)
                st.markdown('<b>Mean by Target</b>', unsafe_allow_html=True)
                st.dataframe(grp_stats)
            st.markdown('</div>', unsafe_allow_html=True)

        # Distribution charts
        dist_cols = [c for c in ['age', 'chol',
                                 'trestbps', 'thalach'] if c in df.columns]
        if dist_cols:
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Distributions</h4>", unsafe_allow_html=True)
            for col in dist_cols:
                _lazy_import_plotting('px')
                fig_hist = px.histogram(df, x=col, nbins=30, color=target_col if target_col in df.columns else None,
                                        marginal='box', opacity=0.85,
                                        title=f'{col} Distribution' +
                                        (" by Target" if target_col in df.columns else ""),
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Categorical impact: chest pain, sex, slope, thal (if encoded or original)
        cat_blocks = []
        if 'cp' in df.columns:
            cat_blocks.append(('cp', 'Chest Pain Type'))
        elif any(c.startswith('cp_') for c in df.columns):
            cp_cols = [c for c in df.columns if c.startswith('cp_')]
            melt_cp = df[cp_cols].copy()
            melt_cp['idx'] = range(len(melt_cp))
            long = melt_cp.melt(
                id_vars='idx', var_name='cp_type', value_name='val')
            long = long[long['val'] == 1]
            if target_col in df.columns:
                merged = long.merge(
                    df[[target_col]], left_on='idx', right_index=True)
                cp_rate = merged.groupby('cp_type')[
                    target_col].mean().reset_index()
                fig_cp = px.bar(cp_rate, x='cp_type', y=target_col,
                                title='Disease Rate by Chest Pain (One-Hot)', color='cp_type')
                st.markdown(
                    "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Chest Pain & Outcome</h4>", unsafe_allow_html=True)
                st.plotly_chart(fig_cp, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        if 'sex_Male' in df.columns and target_col in df.columns:
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Sex & Outcome</h4>", unsafe_allow_html=True)
            df['sex_label'] = df['sex_Male'].apply(
                lambda x: 'Male' if x == 1 else 'Female')
            sex_out = df.groupby('sex_label')[target_col].mean(
            ).reset_index().rename(columns={target_col: 'disease_rate'})
            fig_sex = px.bar(sex_out, x='sex_label', y='disease_rate', color='sex_label',
                             title='Disease Rate by Sex', color_discrete_sequence=['#1D70A2', '#D62828'])
            st.plotly_chart(fig_sex, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        if 'slope' in df.columns and target_col in df.columns:
            slope_rate = df.groupby('slope')[target_col].mean().reset_index()
            fig_slope = px.bar(slope_rate, x='slope', y=target_col,
                               color='slope', title='Disease Rate by Slope')
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Slope & Outcome</h4>", unsafe_allow_html=True)
            st.plotly_chart(fig_slope, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        if 'thal' in df.columns and target_col in df.columns:
            thal_rate = df.groupby('thal')[target_col].mean().reset_index()
            fig_thal = px.bar(thal_rate, x='thal', y=target_col,
                              color='thal', title='Disease Rate by Thalassemia')
            st.plotly_chart(fig_thal, use_container_width=True)

        # Correlation heatmap for key numeric subset (avoid overcrowding)
        if len(num_cols) > 2:
            subset = [c for c in num_cols if c not in [target_col]][:12]
            if target_col in num_cols:
                subset.append(target_col)
            corr = df[subset].corr()
            st.markdown("<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Correlation Matrix (Key Features)</h4>", unsafe_allow_html=True)
            _lazy_import_plotting('plt')
            _lazy_import_plotting('sns')
            fig, ax = plt.subplots(figsize=(7.5, 6))
            sns.heatmap(corr, cmap='Reds', center=0,
                        ax=ax, cbar_kws={'shrink': 0.6})
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Engineered feature relationships
        eng_pairs = [('chol_per_age', 'chol'),
                     ('heart_rate_reserve', 'thalach')]
        eng_pairs = [p for p in eng_pairs if all(
            col in df.columns for col in p)]
        if eng_pairs:
            st.markdown(
                "<div class='card'><div class='accent-bar accent-blue'></div><h4 style='margin-top:0;color:#1D70A2;'>Engineered Features</h4>", unsafe_allow_html=True)
            for x, y in eng_pairs:
                color_arg = target_col if target_col in df.columns else None
                fig_sc = px.scatter(df, x=x, y=y, color=color_arg,
                                    trendline='ols' if color_arg is None else None, title=f'{y} vs {x}')
                st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption('Note: This summary derives from core EDA focusing on class balance, key distributions, categorical impacts, correlations, and engineered features.')

# --------------------------------------------------
# Models & Results Page (NEW)
# --------------------------------------------------
elif nav_page == 'Models & Results':
    ensure_model_loaded()
    ensure_data_loaded()
    model = st.session_state.get('model')
    model_report = st.session_state.get('model_report', {})
    raw_df = st.session_state.get('raw_df', pd.DataFrame())
    st.subheader('📈 Models & Results')
    st.markdown('Overview of model performance and evaluation artifacts.')

    results_dir = PROJECT_ROOT / 'results'
    tuning_path = results_dir / 'tuning_cv_results.csv'
    cv_df = None
    if tuning_path.exists():
        try:
            cv_df = pd.read_csv(tuning_path)
        except Exception as e:
            st.warning(f'Failed to read tuning results: {e}')

    # Static model comparison table (replaces CV summary)
    comp_data = [
        ['baseline', 'DecisionTree', 'default',
            0.7772, 0.7905, 0.8137, 0.8019, 0.7727],
        ['tuned', 'DecisionTree', 'grid/random',
            0.7826, 0.7500, 0.9118, 0.8230, 0.8510],
        ['baseline', 'LogisticRegression', 'default',
            0.8370, 0.8333, 0.8824, 0.8571, 0.9081],
        ['tuned', 'LogisticRegression', 'grid/random',
            0.8424, 0.8614, 0.8529, 0.8571, 0.9108],
        ['baseline', 'RandomForest', 'default',
            0.8370, 0.8462, 0.8627, 0.8544, 0.9030],
        ['tuned', 'RandomForest', 'grid/random',
            0.8315, 0.8318, 0.8725, 0.8517, 0.9039],
        ['baseline', 'SVC', 'default', 0.8587, 0.8393, 0.9216, 0.8785, 0.9044],
        ['tuned', 'SVC', 'grid/random', 0.8370, 0.8000, 0.9412, 0.8649, 0.9096],
        ['refined', 'dt', 'smote', 0.8152, 0.7931, 0.9020, 0.8440, 0.8760],
        ['threshold_tuned', 'dt', 'thr_opt', 0.8152, 0.7931, 0.9020, 0.8440, 0.8760],
        ['refined', 'logreg', 'none', 0.8587, 0.8519, 0.9020, 0.8762, 0.9058],
        ['threshold_tuned', 'logreg', 'thr_opt',
            0.8641, 0.8738, 0.8824, 0.8780, 0.9058],
        ['refined', 'rf', 'smoteenn', 0.8424, 0.8687, 0.8431, 0.8557, 0.8997],
        ['threshold_tuned', 'rf', 'thr_opt', 0.8587, 0.8958, 0.8431, 0.8687, 0.8997],
        ['ensemble', 'soft_voting', 'mixed', 0.8478, 0.8627, 0.8627, 0.8627, 0.9136],
        ['threshold_tuned', 'soft_voting', 'thr_opt',
            0.8696, 0.9062, 0.8529, 0.8788, 0.9136],
        ['ensemble', 'stacking', 'mixed', 0.8533, 0.8571, 0.8824, 0.8696, 0.9021],
        ['refined', 'svc', 'none', 0.8152, 0.7931, 0.9020, 0.8440, 0.9052],
        ['threshold_tuned', 'svc', 'thr_opt',
            0.8804, 0.8846, 0.9020, 0.8932, 0.9052],
        ['refined', 'xgb', 'smoteenn', 0.8533, 0.8713, 0.8627, 0.8670, 0.9000],
        ['threshold_tuned', 'xgb', 'thr_opt',
            0.8696, 0.9149, 0.8431, 0.8776, 0.9000],
    ]
    comp_df = pd.DataFrame(comp_data, columns=[
                           'group', 'model', 'strategy', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    st.markdown('#### Model Comparison Overview')
    st.dataframe(comp_df)
    top_acc = comp_df.loc[comp_df['accuracy'].idxmax()]
    top_f1 = comp_df.loc[comp_df['f1'].idxmax()]
    st.caption(
        f"Top Accuracy: {top_acc['model']} ({top_acc['group']}) = {top_acc['accuracy']:.4f} | Top F1: {top_f1['model']} ({top_f1['group']}) = {top_f1['f1']:.4f}")

    # Best model card
    if model_report:
        m = model_report.get('metrics', {})
        acc = m.get('accuracy')
        f1v = m.get('f1')
        roc = m.get('roc_auc')
        st.markdown(
            f"""
            <div style='border:2px solid #D62828;padding:1rem 1.2rem;border-radius:14px;background:linear-gradient(90deg,#ffe5e5,#ffffff);box-shadow:0 4px 12px -4px rgba(214,40,40,0.45);'>
            <div style='font-size:1.05rem;font-weight:800;color:#D62828;margin-bottom:0.25rem;'>🏅 Best Model: Threshold-Tuned SVC</div>
            <div style='color:#B00020;font-size:0.9rem;font-weight:600;'>Accuracy ≈ {acc} | F1 ≈ {f1v} | ROC AUC ≈ {roc}</div>
            <div style='margin-top:0.35rem;font-size:0.85rem;color:#222;'><span style='color:#D62828;font-weight:700;'>Threshold:</span> {get_current_threshold():.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Confusion matrix & classification report (using available dataset)
    if not raw_df.empty:
        try:
            # Ensure plotting libs loaded
            _lazy_import_plotting('plt')
            _lazy_import_plotting('sns')
            tcol = None
            for cand in ['target', 'num', 'heart_disease']:
                if cand in raw_df.columns:
                    tcol = cand
                    break
            if tcol is None:
                tcol = raw_df.columns[-1]
            X_all = raw_df.drop(columns=[tcol])
            y_all = raw_df[tcol].values
            try:
                y_prob_all = model.predict_proba(X_all)[:, 1]
            except Exception:
                from scipy.special import expit
                y_prob_all = expit(model.decision_function(X_all))
            y_pred_all = (y_prob_all >= get_current_threshold()).astype(int)
            cm = confusion_matrix(y_all, y_pred_all)
            rep = classification_report(
                y_all, y_pred_all, output_dict=True, zero_division=0)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('#### Confusion Matrix')
                fig_cm, ax_cm = plt.subplots(figsize=(3.8, 3.3))
                sns.heatmap(cm, annot=True, fmt='d',
                            cmap='Reds', cbar=False, ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)
            with c2:
                st.markdown('#### Classification Report')
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format(precision=3))
        except Exception as e:
            st.warning(f'Could not compute confusion matrix/report: {e}')

    # Metric bars (if we have model_report)
    if model_report.get('metrics'):
        # Ensure plotly express loaded
        _lazy_import_plotting('px')
        metric_df = pd.DataFrame([
            {'metric': 'accuracy',
                'value': model_report['metrics'].get('accuracy')},
            {'metric': 'precision',
                'value': model_report['metrics'].get('precision')},
            {'metric': 'recall',
                'value': model_report['metrics'].get('recall')},
            {'metric': 'f1', 'value': model_report['metrics'].get('f1')},
            {'metric': 'roc_auc',
                'value': model_report['metrics'].get('roc_auc')},
        ])
        fig_bar = px.bar(metric_df, x='metric', y='value', title='Best Model Metrics',
                         color='metric', color_discrete_sequence=px.colors.qualitative.Set2)
        fig_bar.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# About Page
# --------------------------------------------------
else:
    st.markdown("<div class='section-header gray'>About</div>",
                unsafe_allow_html=True)
    st.markdown('''
**Project:** Heart Disease Risk Prediction  
**Model:** Threshold-Tuned Support Vector Classifier (SVC) with integrated preprocessing pipeline.  
**Performance (approx.):** Accuracy ~88%, F1 ~0.89, ROC AUC ~0.905 (depending on split and rounding).  
**Dataset:** UCI Heart Disease (processed & engineered feature set).  
**Engineering Notes:** Includes feature engineering (chol_per_age, heart_rate_reserve) and custom decision threshold (≈0.617) applied to calibrated probabilities/decision scores.

**Disclaimer:** This tool is for educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment.

**Team:** Yousef Salah & Abdelrahman Mohsen
''')

    if model_report:
        with st.expander('Model Report JSON'):
            st.json(model_report)

# Footer
st.markdown("<div class='footer-note'>© 2025 Heart Disease Risk Prediction Demo. Educational use only.</div>",
            unsafe_allow_html=True)

# --------------------------------------------------
# Download / Export section (Bonus) - available on any page if last prediction exists
# --------------------------------------------------
if st.session_state.get('last_prediction'):
    with st.sidebar.expander('⬇️ Export Last Prediction'):
        pred = st.session_state['last_prediction']
        csv_io = io.StringIO()
        pd.DataFrame([{**pred['input'], 'probability': pred['prob'],
                     'class': pred['class']}]).to_csv(csv_io, index=False)
        st.download_button('Download CSV', data=csv_io.getvalue(
        ), file_name='last_prediction.csv', mime='text/csv')
        # Optional PDF (simple text) if fpdf installed
        try:
            from fpdf import FPDF  # type: ignore
            if st.button('Generate PDF (simple)'):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', size=12)
                pdf.cell(0, 10, 'Heart Disease Prediction Result', ln=1)
                for k, v in pred['input'].items():
                    pdf.cell(0, 8, f"{k}: {v}", ln=1)
                pdf.cell(0, 8, f"Probability: {pred['prob']:.4f}", ln=1)
                pdf.cell(0, 8, f"Class: {pred['class']}", ln=1)
                pdf_file = 'last_prediction.pdf'
                pdf.output(pdf_file)
                with open(pdf_file, 'rb') as fpdf_bytes:
                    st.download_button('Download PDF', data=fpdf_bytes.read(
                    ), file_name=pdf_file, mime='application/pdf')
        except Exception:
            st.caption('Install fpdf to enable PDF export (optional).')

# (Debug checklist removed for production)
